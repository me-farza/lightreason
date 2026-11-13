"""Check and time shared expert-weight loads with real Qwen BF16 rows."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from freetoken.kernel import _cpu_moe

from edge35b.benchmark_bf16_transport import get_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.out.exists() or args.repeats < 2:
        parser.error("Use a fresh result path and at least two repeats")

    torch.set_num_threads(1)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    experts = [1, 7, 33, 61, 98, 107, 157, 221]
    hidden, inner = 2048, 512
    bits = get_rows(args.model, 13, experts, index).reshape(len(experts), -1)
    gate_size = 2 * inner * hidden
    gate = np.ascontiguousarray(bits[:, :gate_size])
    down = np.ascontiguousarray(bits[:, gate_size:])
    gate_table = np.array([gate.ctypes.data], dtype=np.uint64)
    down_table = np.array([down.ctypes.data], dtype=np.uint64)
    random = torch.Generator().manual_seed(8832)
    x = torch.randn(8, hidden, generator=random).to(torch.bfloat16)
    ids = torch.full((8, 8), -1, dtype=torch.int32)
    weights = torch.rand(8, 8, generator=random)
    out = torch.empty_like(x)
    worker = _cpu_moe.CpuMoeExecutor(
        8, 1, 256, 8, hidden, inner, 8, 0, 0,
        [("bf16", {"gate_up": gate_table.ctypes.data, "down": down_table.ctypes.data}, {})],
        [0], 1.0, float("inf"), list(range(8, 16)),
        x.data_ptr(), ids.data_ptr(), weights.data_ptr(), out.data_ptr(),
    )
    workloads = {
        "distinct": ([0, 1, 2, 3, 4, 5, 6, 7],),
        "two-pairs": ([0, 1, 2, 3, 0, 1, 4, 5],),
        "four-pairs": ([0, 1, 2, 3, 0, 1, 2, 3],),
        "eight-pairs": ([0, 1, 2, 3, 0, 1, 2, 3], [4, 5, 6, 7, 4, 5, 6, 7]),
    }
    modes = (0, 1, 3) if hasattr(worker, "set_pair_threshold") else (0,)
    rows = []
    for case, routes in workloads.items():
        ids.fill_(-1)
        for col, route in enumerate(routes):
            ids[:, col] = torch.tensor(route, dtype=torch.int32)
        times = {mode: [] for mode in modes}
        hashes = {}
        for i in range(args.repeats + 4):
            for mode in modes if i % 2 == 0 else reversed(modes):
                if len(modes) > 1:
                    worker.set_pair_threshold(mode)
                start = perf_counter()
                worker.run(0, 8)
                elapsed = (perf_counter() - start) * 1000
                digest = hashlib.sha256(out.view(torch.uint8).numpy().tobytes()).hexdigest()
                if mode in hashes and digest != hashes[mode]:
                    raise RuntimeError(f"Non-deterministic output on {case}, threshold {mode}")
                hashes[mode] = digest
                if i >= 4:
                    times[mode].append(elapsed)
        if len(set(hashes.values())) != 1:
            raise RuntimeError(f"CPU grouped arithmetic changed the BF16 output on {case}")
        rows.append({"case": case, "output_sha256": hashes[0], "median_ms": {
            mode: median(values) for mode, values in times.items()}, "ms": times})
    args.out.write_text(json.dumps({"isa": worker.isa_name(), "experts": experts, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
