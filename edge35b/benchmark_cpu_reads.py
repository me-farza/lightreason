"""Search exact AVX2 expert read policies on trace-shaped Qwen fixtures."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter_ns

import numpy as np
import torch
from freetoken.kernel import _cpu_moe

from edge35b.bf16_expert_codec import PROJECTIONS, checkpoint_bank
from edge35b.cpu_read_packing import CPU_READ_POLICIES, PackedReadBanks
from edge35b.download import REVISION


SEEDS = ("raw-no-prefetch", "raw-prefetch-256", "packed-gate", "packed-down")
MUTATIONS = {
    "raw-no-prefetch": ("raw-prefetch-256", "raw-prefetch-1024"),
    "raw-prefetch-256": ("raw-no-prefetch", "raw-prefetch-1024"),
    "raw-prefetch-1024": ("raw-prefetch-256", "raw-no-prefetch"),
    "packed-gate": ("packed-both",), "packed-down": ("packed-both",),
    "packed-both": ("packed-both-16", "packed-both-prefetch"),
    "packed-both-16": ("packed-both",), "packed-both-prefetch": ("packed-both",),
}


@dataclass(frozen=True)
class Fixture:
    case: str
    layer: int
    routes: int
    experts: int


def fixtures(report, case, quantiles, excluded_layers=()):
    if report["revision"] != REVISION or not report["transport_summary"]["exact"]:
        raise ValueError("Use the exact native CUDA profile.")
    rows = [r for r in report["rows"] if r["case"] == case and r["scenario"] == "native-packed-hot-profile"]
    records = Counter()
    for row in rows:
        for step in row["diagnostics"]["passes"]:
            for layer, stamps in enumerate(step["gpu_layer_stamps"]):
                fetched, missed, routes = stamps[-3:]
                experts = missed - fetched
                if 3 <= experts <= 16 and experts <= routes <= min(8, experts) * 8:
                    records[experts, routes, layer] += 1
    selected = []
    excluded = set(excluded_layers)
    for q in quantiles:
        candidates = sorted(key for key, count in records.items() for _ in range(count) if key[2] not in excluded)
        if not candidates:
            raise ValueError(f"Missing representative CPU route shapes for {case}.")
        experts, routes, layer = candidates[int(q * (len(candidates) - 1))]
        selected.append(Fixture(case, layer, routes, experts))
        excluded.add(layer)
    return selected


class ExpertFixture:
    def __init__(self, model, index, fixture, repeats):
        self.fixture = fixture
        rng = Random(8400 + fixture.layer)
        bank_rows = rng.sample(range(256), 40)
        self.raw = [torch.from_numpy(checkpoint_bank(model, index, fixture.layer, projection)[bank_rows].copy())
                    .view(torch.bfloat16) for projection in PROJECTIONS]
        self.raw_tables = [np.array([weights.data_ptr()], dtype=np.uint64) for weights in self.raw]
        self.packed = PackedReadBanks(1)
        for bank, weights in enumerate(self.raw):
            self.packed.add(bank, 0, weights)

        gen = torch.Generator().manual_seed(8832 + fixture.layer)
        self.x = torch.randn(8, 2048, generator=gen).to(torch.bfloat16)
        self.out = torch.empty_like(self.x)
        self.ids = torch.full((8, 8), -1, dtype=torch.int32)
        self.weights = torch.rand(8, 8, generator=gen)
        raw_ptrs = dict(zip(("gate_up", "down"), (t.ctypes.data for t in self.raw_tables), strict=True))
        self.worker = _cpu_moe.CpuMoeExecutor(
            8, 1, 256, 8, 2048, 512, 8, 0, 0, [("bf16", raw_ptrs, {})], [0], 1.0, float("inf"),
            list(range(8, 16)), self.x.data_ptr(), self.ids.data_ptr(), self.weights.data_ptr(), self.out.data_ptr(),
        )
        self.worker.set_pair_threshold(1)
        self.packed.configure(self.worker)
        self.inputs = [self._routes(rng) for _ in range(repeats + 4)]

    def _routes(self, rng):
        experts = rng.sample(range(40), self.fixture.experts)
        possibilities = [token for token in range(8) for _ in range(min(8, len(experts)))]
        counts = Counter(rng.sample(possibilities, self.fixture.routes))
        ids = torch.full((8, 8), -1, dtype=torch.int32)
        offset = 0
        for token, count in sorted(counts.items()):
            for lane, slot in enumerate(rng.sample(range(8), count)):
                ids[token, slot] = experts[(offset + lane) % len(experts)]
            offset += count
        return ids

    def _run(self, policy):
        self.worker.set_cpu_read_policy(CPU_READ_POLICIES[policy])
        self.out.zero_()
        started = perf_counter_ns()
        self.worker.run(0, 8)
        return (perf_counter_ns() - started) / 1_000_000, self.out.view(torch.int16).clone()

    def measure(self, policy):
        samples = []
        for index, ids in enumerate(self.inputs):
            self.ids.copy_(ids)
            order = ("stock", policy) if index % 2 == 0 else (policy, "stock")
            observed = {arm: self._run(arm) for arm in order}
            stock_ms, stock_bits = observed["stock"]
            trial_ms, trial_bits = observed[policy]
            if not torch.equal(stock_bits, trial_bits):
                raise RuntimeError(f"CPU BF16 output changed for {self.fixture} under {policy}.")
            if index >= 4:
                samples.append((stock_ms, trial_ms))
        return {"fixture": asdict(self.fixture), "packed_tile_coverage": self.packed.audit,
                "stock_ms": median(a for a, _ in samples), "candidate_ms": median(b for _, b in samples),
                "speedup": median(a / b for a, b in samples), "pairs_ms": samples, "bitwise_match": True}


def next_policy(results):
    unused = set(CPU_READ_POLICIES) - {"stock", *(row["policy"] for row in results)}
    for seed in SEEDS:
        if seed in unused:
            return seed, None
    for parent in sorted(results, key=lambda row: row["speedup"], reverse=True):
        for child in MUTATIONS[parent["policy"]]:
            if child in unused:
                return child, parent["policy"]
    return next(policy for policy in CPU_READ_POLICIES if policy in unused), None


def measure(policy, group, parent=None):
    rows = [fixture.measure(policy) for fixture in group]
    result = {"policy": policy, "parent": parent, "speedup": median(r["speedup"] for r in rows), "rows": rows}
    print(policy, f"{result['speedup']:.4f}x", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", default=20, type=int)
    parser.add_argument("--trials", default=8, type=int)
    args = parser.parse_args()
    if args.out.exists() or args.repeats < 8 or not len(SEEDS) <= args.trials < len(CPU_READ_POLICIES):
        parser.error("Use a fresh output, at least eight pairs, and four to eight candidate trials.")
    if json.loads((args.model / "lightreason-download.json").read_text())["revision"] != REVISION:
        parser.error("Use the pinned Qwen BF16 checkpoint.")
    torch.set_num_threads(1)
    profile = json.loads(args.profile.read_text())
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    dev_shapes = fixtures(profile, "studio", (0.15, 0.4, 0.65, 0.9))
    dev = [ExpertFixture(args.model, index, shape, args.repeats) for shape in dev_shapes]
    report = {"revision": REVISION, "repeats": args.repeats, "development": []}
    for _ in range(args.trials):
        policy, parent = next_policy(report["development"])
        report["development"].append(measure(policy, dev, parent))
    report["selected"] = max(report["development"], key=lambda row: row["speedup"])["policy"]
    del dev
    excluded = [shape.layer for shape in dev_shapes]
    held = []
    for case in ("known-math", "coding", "chat"):
        shape, = fixtures(profile, case, (0.5,), excluded)
        excluded.append(shape.layer)
        held.append(ExpertFixture(args.model, index, shape, args.repeats))
    report["held_out"] = measure(report["selected"], held)
    with args.out.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
