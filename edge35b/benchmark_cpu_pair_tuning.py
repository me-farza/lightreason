"""Evolve AVX2 expert kernels against a paired CPU benchmark on real Qwen weights."""

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from random import Random
import re
from statistics import median
from time import perf_counter_ns

import numpy as np
import torch
from freetoken.kernel import _cpu_moe

from edge35b.benchmark_bf16_transport import get_rows
from edge35b.download import REVISION


@dataclass(frozen=True)
class Fixture:
    name: str
    layer: int
    width: int
    routes: int
    experts: int


@dataclass(frozen=True)
class Kernel:
    widen: int
    chunk: int
    max_routes: int = 64

    @property
    def name(self):
        name = f"widen-{self.widen}-chunk-{self.chunk}"
        return name if self.max_routes == 64 else f"{name}-routes-{self.max_routes}"


CONTROL = Kernel(0, 1)
SEEDS = (Kernel(1, 1), Kernel(2, 1), Kernel(3, 1), Kernel(0, 2), Kernel(0, 4))


def fixture_sets(report, *, route_gate=False):
    """Use measured route counts, reserving later passes and different layers."""
    if report["revision"] != REVISION or report["dtype"] != "bfloat16":
        raise ValueError("Use the pinned BF16 route profile.")
    by_split = {key: {4: [], 8: []} for key in ("development", "held_out", "reserve")}
    for run in report["rows"]:
        diagnostic = run.get("diagnostics", {})
        if diagnostic.get("mode") != "profile":
            continue
        passes = diagnostic["passes"]
        for index, step in enumerate(passes[5:], 5):
            development = index < len(passes) // 2
            split = "development" if development else "held_out"
            width = step["width"]
            for layer, row in enumerate(step["gpu_layer_stamps"]):
                fixture_split = split if (layer < 20) == development else "reserve"
                fetched, missed, routes = row[-3:]
                experts = missed - fetched
                if 0 < experts <= routes <= min(8, experts) * width:
                    by_split[fixture_split][width].append((routes, experts, layer))

    def select(widths):
        cases = []
        for width, measurements in widths.items():
            pairs = [row for row in measurements if row[0] > row[1]]
            if not pairs:
                raise ValueError("No repeat-expert fixtures in the trace split.")
            common = sorted(measurements)[len(measurements) // 2]
            typical = sorted(pairs, key=lambda row: (row[0] - row[1], *row))[len(pairs) // 2]
            busy = sorted(pairs, key=lambda row: (row[0] - row[1], *row))[int(0.9 * len(pairs))]
            for name, (routes, experts, layer) in (("common", common), ("repeated", typical), ("busy", busy)):
                cases.append(Fixture(f"width-{width}-{name}", layer, width, routes, experts))
        return cases

    early = select(by_split["development"])
    late = select(by_split["held_out"])
    if not route_gate:
        return {"development": early, "held_out": late}

    # The previous held-out timings informed the cutoff experiment. Move
    # those fixtures into tuning and exclude their exact shapes from reserve.
    used = {(f.width, f.routes, f.experts, f.layer) for f in early + late}
    reserve = {w: [row for row in rows if (w, *row) not in used]
               for w, rows in by_split["reserve"].items()}
    return {"development": early + [replace(f, name="late-" + f.name) for f in late],
            "held_out": select(reserve)}


def benchmark(model, index, fixture, kernels, repeats):
    hidden, inner = 2048, 512
    rng = Random(8400 + fixture.layer + fixture.width)
    experts = rng.sample(range(256), fixture.experts)
    bits = get_rows(model, fixture.layer, experts, index).reshape(fixture.experts, -1)
    gate_size = 2 * inner * hidden
    gate = np.ascontiguousarray(bits[:, :gate_size])
    down = np.ascontiguousarray(bits[:, gate_size:])
    gate_table = np.array([gate.ctypes.data], dtype=np.uint64)
    down_table = np.array([down.ctypes.data], dtype=np.uint64)

    generator = torch.Generator().manual_seed(8832 + fixture.layer)
    x = torch.randn(fixture.width, hidden, generator=generator).to(torch.bfloat16)
    out = torch.empty_like(x)
    ids = torch.full((fixture.width, 8), -1, dtype=torch.int32)
    weights = torch.rand(fixture.width, 8, generator=generator)
    # The trace records counts. Construct a deterministic route layout with
    # those counts and distinct experts within each token.
    token_slots = [token for token in range(fixture.width) for _ in range(min(8, fixture.experts))]
    counts = [0] * fixture.width
    for token in rng.sample(token_slots, fixture.routes):
        counts[token] += 1
    offset = 0
    for token, count in enumerate(counts):
        active_slots = rng.sample(range(8), count)
        for i, slot in enumerate(active_slots):
            ids[token, slot] = (offset + i) % fixture.experts
        offset += count

    worker = _cpu_moe.CpuMoeExecutor(
        8, 1, 256, 8, hidden, inner, 8, 0, 0,
        [("bf16", {"gate_up": gate_table.ctypes.data, "down": down_table.ctypes.data}, {})],
        [0], 1.0, float("inf"), list(range(8, 16)),
        x.data_ptr(), ids.data_ptr(), weights.data_ptr(), out.data_ptr(),
    )
    worker.set_pair_threshold(1)

    def run(kernel):
        if hasattr(worker, "set_pair_tuning"):
            if kernel.max_routes == 64:
                worker.set_pair_tuning(kernel.widen, kernel.chunk)
            else:
                worker.set_pair_tuning(kernel.widen, kernel.chunk, kernel.max_routes)
        started = perf_counter_ns()
        worker.run(0, fixture.width)
        elapsed = (perf_counter_ns() - started) / 1_000_000
        digest = hashlib.sha256(out.view(torch.uint8).numpy().tobytes()).hexdigest()
        return elapsed, digest

    hashes = set()
    if not kernels:
        for _ in range(4):
            run(CONTROL)
        samples = [run(CONTROL) for _ in range(repeats)]
        hashes.update(digest for _, digest in samples)
        result = {"control_ms": median(ms for ms, _ in samples), "arms": {}}
    else:
        timings = {kernel: [] for kernel in kernels}
        for i in range(repeats + 4):
            for j, kernel in enumerate(kernels if i % 2 == 0 else kernels[::-1]):
                pair = [CONTROL, kernel] if (i + j) % 2 == 0 else [kernel, CONTROL]
                observed = {k: run(k) for k in pair}
                hashes.update(digest for _, digest in observed.values())
                if i >= 4:
                    timings[kernel].append((observed[CONTROL][0], observed[kernel][0]))
        result = {"arms": {kernel.name: {
            "control_ms": median(a for a, _ in times), "candidate_ms": median(b for _, b in times),
            "speedup": median(a / b for a, b in times), "paired_ms": times,
        } for kernel, times in timings.items()}}
    if len(hashes) != 1:
        raise RuntimeError(f"CPU candidate changed BF16 output on {fixture.name}.")
    return {"fixture": asdict(fixture), "output_sha256": hashes.pop(), **result}


def score(kernel, results, *, width=None):
    return median(row["arms"][kernel.name]["speedup"] for row in results
                  if width is None or row["fixture"]["width"] == width)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--profile", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--repeats", type=int, default=40)
    p.add_argument("--frozen-control", action="store_true")
    p.add_argument("--route-gate-from", type=Path, help="Tune cutoffs for the previous CPU winner on fresh held-out fixtures")
    args = p.parse_args()
    if args.out.exists() or args.repeats < 8:
        p.error("Use a fresh result path and at least eight paired repeats.")
    manifest = json.loads((args.model / "lightreason-download.json").read_text())
    if manifest["revision"] != REVISION:
        p.error("Use the pinned Qwen BF16 checkpoint.")
    if not args.frozen_control and not hasattr(_cpu_moe.CpuMoeExecutor, "set_pair_tuning"):
        p.error("Load the experimental CPU extension or choose --frozen-control.")
    torch.set_num_threads(1)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    fixtures = fixture_sets(json.loads(args.profile.read_text()), route_gate=bool(args.route_gate_from))

    def run(split, kernels=()):
        return [benchmark(args.model, index, fixture, kernels, args.repeats) for fixture in fixtures[split]]

    winner = promoted = None
    if args.frozen_control:
        development, held_out = run("development"), run("held_out")
    elif args.route_gate_from:
        prior = json.loads(args.route_gate_from.read_text())
        match = re.fullmatch(r"widen-([0-3])-chunk-(1|2|4)", prior["selected"] or "")
        if prior["revision"] != REVISION or not match:
            p.error("The cutoff experiment needs a measured winner from the CPU tuning run.")
        widen, chunk = map(int, match.groups())
        candidates = tuple(Kernel(widen, chunk, limit) for limit in (5, 8, 12, 64))
        development = run("development", candidates)
        winner = max(candidates, key=lambda k: (score(k, development, width=8), score(k, development)))
        held_out = run("held_out", (winner,))
        if (score(winner, development, width=8) >= 1.005 and score(winner, held_out, width=8) >= 1.005
                and all(row["arms"][winner.name]["speedup"] >= 0.98 for row in held_out)):
            promoted = winner
    else:
        development = run("development", SEEDS)
        best_widen = max(SEEDS[:3], key=lambda k: score(k, development))
        best_chunk = max(SEEDS[3:], key=lambda k: score(k, development))
        joint = Kernel(best_widen.widen, best_chunk.chunk)
        for row, combined in zip(development, run("development", (joint,)), strict=True):
            if row["output_sha256"] != combined["output_sha256"]:
                raise RuntimeError("Combined candidate changed the BF16 output.")
            row["arms"].update(combined["arms"])
        winner = max((*SEEDS, joint), key=lambda k: score(k, development))
        held_out = run("held_out", (winner,))
        if (score(winner, development) >= 1.005 and score(winner, held_out) >= 1.005
                and all(row["arms"][winner.name]["speedup"] >= 0.98 for row in held_out)):
            promoted = winner
    args.out.write_text(json.dumps({
        "revision": REVISION, "repeats": args.repeats, "frozen_control": args.frozen_control,
        "development": development, "held_out": held_out,
        "selected": winner.name if winner else None, "promoted": promoted.name if promoted else None,
        "median_development_speedup": score(winner, development) if winner else None,
        "median_held_out_speedup": score(winner, held_out) if winner else None,
        "median_held_out_width8_speedup": score(winner, held_out, width=8) if winner else None,
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
