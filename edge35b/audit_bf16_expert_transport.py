"""Check all rows of the candidate transport layers without CUDA."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from edge35b.bf16_expert_codec import LAYERS, PROJECTIONS, checkpoint_bank, pack_verified_row, plan_bank, row_bytes
from edge35b.download import REVISION
from hardware_exact10 import cpu_inventory, cpu_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a fresh audit output path")
    manifest = json.loads((args.model / "lightreason-download.json").read_text())
    if manifest["revision"] != REVISION:
        parser.error("Use the pinned Qwen3.6 BF16 checkpoint")
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    inventory = cpu_inventory()
    peak, sampled = 0.0, float("-inf")

    def check_temperature():
        nonlocal peak, sampled
        if perf_counter() - sampled < 1:
            return
        temperature = cpu_sample(inventory)["cpu_temperature_max_c"]
        if temperature is None or temperature >= 75:
            raise RuntimeError(f"CPU temperature unavailable or above the 75 °C stop: {temperature}")
        sampled = perf_counter()
        peak = max(peak, temperature)

    result = {"revision": REVISION, "banks": []}
    started = perf_counter()
    for layer in (*LAYERS, 0):
        for projection in PROJECTIONS:
            check_temperature()
            source = checkpoint_bank(args.model, index, layer, projection).reshape(256, -1)
            plans, storage = plan_bank(source, check=check_temperature)
            raw = packed = fallback = 0
            for bits, plan in zip(source, plans, strict=True):
                check_temperature()
                if plan is None:
                    raw += bits.nbytes
                    fallback += 1
                    continue
                _, base, escapes = plan
                length = row_bytes(bits.size, escapes)
                pack_verified_row(bits, np.empty(length, dtype=np.uint8), base, escapes)
                packed += length
            record = {"layer": layer, "projection": projection, "bf16_bytes": source.nbytes,
                      "packed_bytes": packed, "raw_fallback_bytes": raw, "raw_rows": fallback,
                      "bitwise_checked_rows": len(source) - fallback, "packed_storage_bytes": storage}
            result["banks"].append(record)
            print(layer, projection, f"{100 * (1 - (packed + raw) / source.nbytes):.2f}% fewer bytes",
                  f"{fallback} raw rows", flush=True)
    check_temperature()
    result.update(seconds=perf_counter() - started, peak_cpu_c=peak)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
