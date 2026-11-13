"""Measure lossless expert-weight transfers on actual Qwen3.6 checkpoint rows.

This is an isolated H2D test. It includes GPU unpacking and exact escape repair,
but does not include expert GEMMs, CPU overflow or speculative verification.
"""

import argparse
import json
from pathlib import Path
from statistics import median
import numpy as np
import torch
import triton
import triton.language as tl

from edge35b.bf16_expert_codec import PROJECTIONS, checkpoint_bank, exponent_window, pack_row, row_bytes


def get_rows(root, layer, experts, index):
    columns = []
    for name in PROJECTIONS:
        bank = checkpoint_bank(root, index, layer, name)
        columns.append(bank[experts].reshape(len(experts), -1))
    return np.concatenate(columns, axis=1).reshape(-1)


def encode(bits):
    base, escapes = exponent_window(bits)
    packed = np.empty(row_bytes(bits.size, escapes), dtype=np.uint8)
    pack_row(bits, base, escapes, packed)
    return packed, base, escapes


@triton.jit
def decode(sign_fraction, exponent, output, n: tl.constexpr, base, block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    sm = tl.load(sign_fraction + i, i < n, 0).to(tl.int32)
    pair = tl.load(exponent + i // 2, i < n, 0).to(tl.int32)
    exp = ((pair >> (4 * (i % 2))) & 15) + base
    tl.store(output + i, ((sm & 128) << 8) | (exp << 7) | (sm & 127), i < n)


@triton.jit
def repair(indices, values, output, k, block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    where = tl.load(indices + i, i < k, 0)
    values = tl.load(values + i, i < k, 0)
    tl.store(output + where, values, i < k)


@triton.jit
def gather(host, output, n: tl.constexpr, block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    src = tl.cast(host, tl.pointer_type(tl.int16))
    tl.store(output + i, tl.load(src + i, i < n, 0), i < n)


@triton.jit
def gather_decode(host, output, n: tl.constexpr, base, block: tl.constexpr):
    # One PCIe byte for two neighboring exponents. Load it once per pair.
    i = tl.program_id(0) * block + tl.arange(0, block)
    src = tl.cast(host, tl.pointer_type(tl.uint8))
    exp = tl.load(src + n + i, i < n // 2, 0).to(tl.int32)
    sm = tl.load(tl.cast(src, tl.pointer_type(tl.uint16)) + i, i < n // 2, 0).to(tl.int32)
    lo = ((sm & 128) << 8) | (((exp & 15) + base) << 7) | (sm & 127)
    hi = (sm & 32768) | (((exp >> 4) + base) << 7) | ((sm >> 8) & 127)
    tl.store(tl.cast(output, tl.pointer_type(tl.uint32)) + i, lo | (hi << 16), i < n // 2)


@triton.jit
def gather_repair(host, output, offset: tl.constexpr, count: tl.constexpr, block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    indices = tl.cast(host + offset, tl.pointer_type(tl.int32))
    values = tl.cast(host + offset + 4 * count, tl.pointer_type(tl.int16))
    where = tl.load(indices + i, i < count, 0)
    tl.store(output + where, tl.load(values + i, i < count, 0), i < count)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--repeats", type=int, default=12)
    p.add_argument("--gather", action="store_true", help="Also test direct reads from mapped host memory")
    args = p.parse_args()
    if not 1 <= args.experts <= 32 or args.repeats < 2 or args.out.exists():
        p.error("Use 1–32 experts, at least two repeats and a fresh result path")
    manifest = json.loads((args.model / "lightreason-download.json").read_text())
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    random = np.random.default_rng(912)
    rows = []
    modes = ("bf16", "packed")
    if args.gather:
        from freetoken.kernel.pinned import device_ptr
        modes += ("gather-bf16", "gather-packed")
    for layer in (0, 13, 26, 39):
        expert_ids = sorted(int(x) for x in random.choice(256, args.experts, replace=False))
        bits = get_rows(args.model, layer, expert_ids, index)
        packed, base, count = encode(bits)
        n = bits.size
        split = 3 * n // 2
        raw_host = torch.from_numpy(bits.view(np.int16)).pin_memory()
        packed_host = torch.from_numpy(packed).pin_memory()
        gpu = torch.empty_like(packed_host, device="cuda")
        out = torch.empty_like(raw_host, device="cuda")
        sign_fraction, exponent = gpu[:n], gpu[n:split]
        indices = gpu[split:split + 4 * count].view(torch.int32)
        values = gpu[split + 4 * count:].view(torch.int16)

        def transfer(mode):
            if mode == "bf16":
                out.copy_(raw_host, non_blocking=True)
            elif mode == "gather-bf16":
                gather[(triton.cdiv(n, 1024),)](raw_addr, out, n, 1024)
            elif mode == "gather-packed":
                gather_decode[(triton.cdiv(n // 2, 1024),)](packed_addr, out, n, base, 1024)
                if count:
                    gather_repair[(triton.cdiv(count, 256),)](packed_addr, out, split, count, 256)
            else:
                gpu.copy_(packed_host, non_blocking=True)
                decode[(triton.cdiv(n, 1024),)](sign_fraction, exponent, out, n, base, 1024)
                if count:
                    repair[(triton.cdiv(count, 256),)](indices, values, out, count, 256)

        if args.gather:
            raw_addr, packed_addr = device_ptr(raw_host), device_ptr(packed_host)
        for mode in modes:
            transfer(mode)
            if not torch.equal(out.cpu(), raw_host):
                raise RuntimeError(f"Layer {layer}: {mode} did not preserve BF16 bits")
        for _ in range(3):
            for mode in modes:
                transfer(mode)
        torch.cuda.synchronize()
        records = {mode: [] for mode in modes}
        for repeat in range(args.repeats):
            for mode in (modes if repeat % 2 == 0 else reversed(modes)):
                first = torch.cuda.Event(enable_timing=True)
                last = torch.cuda.Event(enable_timing=True)
                first.record()
                transfer(mode)
                last.record()
                last.synchronize()
                records[mode].append(first.elapsed_time(last))
        rows.append({"layer": layer, "expert_ids": expert_ids, "bf16_bytes": bits.nbytes,
                     "packed_bytes": packed.nbytes, "base": base, "escape_count": count,
                     "bitwise_match": True, "transfer_ms": records,
                     "median_ms": {key: median(values) for key, values in records.items()}})
    args.out.write_text(json.dumps({"revision": manifest["revision"], "gpu": torch.cuda.get_device_name(),
                                    "experts_per_transfer": args.experts, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
