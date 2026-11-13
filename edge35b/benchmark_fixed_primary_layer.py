"""Measure packed primary replay with the original Qwen BF16 decode GEMV."""

import argparse
import json
from pathlib import Path
from random import Random
from statistics import median

import numpy as np
import torch
import triton
import triton.language as tl

from edge35b.bf16_expert_codec import PROJECTIONS, checkpoint_bank
from edge35b.cpu_read_packing import TILE_VALUES
from edge35b.download import REVISION
from edge35b.fixed_packed_primary import _plan_fixed_prefix
from edge35b.gpu_resident_experts import device_rows
from edge35b.native_bf16_gather import NativeBf16Gather
from edge35b.packed_primary_cache import PACKED_TILE_BYTES, ROUTES


FIXTURES = (("studio", 2), ("known-math", 17), ("coding", 37), ("chat", 0))
EXPERTS = 24
SCRATCH = ROUTES + 1
REPLAYS = 8
DEVICE = "cuda"


@triton.jit
def _decode_compact_gemv(a, descriptors, out, weights, slots, N: tl.constexpr, K: tl.constexpr,
                         BN: tl.constexpr, BK: tl.constexpr, TILE: tl.constexpr, TILE_BYTES: tl.constexpr,
                         ROUTED_INPUT: tl.constexpr,
                         ROUTED_WEIGHT: tl.constexpr):
    route = tl.program_id(0)
    expert = tl.load(slots + route)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    token = route if ROUTED_INPUT else route // 8
    accumulator = tl.zeros((BN,), tl.float32)
    if expert >= 0:
        descriptor = descriptors + expert * 3
        data = tl.load(descriptor)
        headers = tl.load(descriptor + 1)
        raw = tl.load(descriptor + 2)
        row = n[None, :].to(tl.int64) * K + k[:, None]
        for begin in range(K // BK):
            x = tl.load(a + token * K + begin * BK + k)
            offset = row + begin * BK
            if headers == 0:
                b = tl.load(raw.to(tl.pointer_type(tl.bfloat16)) + offset)
            else:
                tile = n.to(tl.int64) * (K // TILE) + begin // (TILE // BK)
                header = tl.load(headers.to(tl.pointer_type(tl.uint32)) + tile)
                start = data + tile * TILE_BYTES + (header >> 8) * 128
                lane = begin % (TILE // BK) * BK + k
                wide = (header & 255) == 255
                raw_tile = tl.load(start[None, :].to(tl.pointer_type(tl.uint16)) + lane[:, None],
                                   wide[None, :], other=0)
                code = tl.load(start[None, :].to(tl.pointer_type(tl.uint8)) + TILE + lane[:, None] // 2,
                               ~wide[None, :], other=0).to(tl.uint16)
                code = (code >> ((lane[:, None] & 1) * 4)) & 15
                low = tl.load(start[None, :].to(tl.pointer_type(tl.uint8)) + lane[:, None],
                              ~wide[None, :], other=0).to(tl.uint16)
                high = ((header[None, :] & 255) + (code & 7)) | ((code & 8) << 4)
                b = tl.where(wide[None, :], raw_tile, (high << 8) | low).to(tl.uint16).to(tl.bfloat16, bitcast=True)
            accumulator += tl.sum(x[:, None] * b, axis=0)
        if ROUTED_WEIGHT:
            accumulator *= tl.load(weights + route)
    tl.store(out + route * N + n, accumulator.to(tl.bfloat16))


def fixture(model, index, case, layer, run, native, repeats):
    from freetoken.kernel import _cpu_moe, moe_sum_reduce_triton
    from freetoken.layers import gated_act_and_mul
    from freetoken.moe.fused import fused_experts_decode_impl

    steps = [step for step in run["diagnostics"]["passes"] if len(step["expert_fetch_ids"][layer]) <= EXPERTS]
    step = steps[len(steps) // 2]
    fetched = step["expert_fetch_ids"][layer]
    cpu_routes = min(step["gpu_layer_stamps"][layer][9], ROUTES - EXPERTS)
    rng = Random(1173 + layer)
    keys = fetched + rng.sample(sorted(set(range(256)) - set(fetched)), EXPERTS - len(fetched))
    cut = EXPERTS // 2
    eligible = np.ones(EXPERTS, dtype=np.bool_)
    layouts, descriptors, prefix, raw, work, owners = [], [], [], [], [], []
    direct_tables = []
    for projection in PROJECTIONS:
        source = torch.from_numpy(checkpoint_bank(model, index, layer, projection)[keys].copy()).view(torch.bfloat16)
        tiles = source[0].numel() // TILE_VALUES
        packed = np.empty((EXPERTS, tiles * PACKED_TILE_BYTES), dtype=np.uint8)
        bases = np.empty((EXPERTS, tiles), dtype=np.uint8)
        _cpu_moe.pack_cpu_read_tiles(source.data_ptr(), packed.ctypes.data, bases.ctypes.data, bases.size)
        fallback = bases.reshape(EXPERTS, -1, 32) == 255
        flags = np.packbits(fallback, axis=-1, bitorder="little").view(np.uint32).squeeze(-1)
        counts = fallback.sum(axis=-1, dtype=np.uint64)
        header = ((counts.cumsum(axis=-1) - counts) << 32) | flags
        row_bytes = source[0].nbytes
        stride = row_bytes * 7 // 8
        eligible &= tiles * PACKED_TILE_BYTES + counts.sum(axis=-1) * 128 <= stride

        raw_bank = torch.cat((source, torch.zeros_like(source[:1]))).to(DEVICE)
        work_bank = torch.empty((SCRATCH + EXPERTS - cut, *source.shape[1:]), device=DEVICE, dtype=torch.bfloat16)
        work_bank[0].zero_()
        compact = torch.empty(cut * stride, device=DEVICE, dtype=torch.uint8)
        codes = torch.from_numpy(packed).to(DEVICE)
        exponent = torch.from_numpy(bases).to(DEVICE)
        owners.extend((compact, codes, exponent))
        table, table_owners, _ = device_rows(source, raw_bank)
        direct_tables.append(table)
        owners.extend(table_owners)
        raw.append(raw_bank)
        work.append(work_bank)
        prefix.append(torch.from_numpy(header.view(np.int64)).to(DEVICE))
        layouts.append((compact.data_ptr(), work_bank[SCRATCH].data_ptr(), work_bank.data_ptr(), stride, row_bytes))
        descriptors.append([(codes[e].data_ptr(), exponent[e].data_ptr(), raw_bank[e].data_ptr())
                            for e in range(EXPERTS)])

    source_table = torch.tensor(descriptors, device=DEVICE, dtype=torch.int64)
    layout_table = torch.tensor(layouts, device=DEVICE, dtype=torch.int64)
    prefix_ptrs = torch.tensor([h.data_ptr() for h in prefix], device=DEVICE, dtype=torch.int64)
    logical = list(range(EXPERTS)) + rng.choices(range(EXPERTS), k=ROUTES - EXPERTS - cpu_routes)
    logical.extend([-1] * cpu_routes)
    rng.shuffle(logical)
    slot_to_id = torch.arange(EXPERTS, device=DEVICE, dtype=torch.int32) + layer * EXPERTS
    slots = torch.tensor(logical, device=DEVICE, dtype=torch.int32).view(8, 8)
    cpu_ids = torch.where(slots >= 0, -1, 0)
    stock_slots = torch.where(slots >= 0, slots, EXPERTS)
    weights = torch.linspace(0.03, 0.11, ROUTES, device=DEVICE).view_as(slots)
    weights.masked_fill_(slots < 0, 0)
    hidden = torch.randn((8, 2048), device=DEVICE, dtype=torch.bfloat16,
                         generator=torch.Generator(device=DEVICE).manual_seed(layer + 4982))
    evicted = torch.arange(ROUTES, device=DEVICE, dtype=torch.int32)
    fetched_count = torch.tensor([EXPERTS], device=DEVICE, dtype=torch.int64)
    eligible_mask = torch.from_numpy(eligible).to(DEVICE)
    packed_ids = torch.full((cut,), -1, device=DEVICE, dtype=torch.int32)
    plan = torch.empty((3, ROUTES), device=DEVICE, dtype=torch.int32)
    count = torch.empty(1, device=DEVICE, dtype=torch.int32)
    gemm_slots = torch.empty_like(slots)
    refill_trace = torch.empty((40, ROUTES), device=DEVICE, dtype=torch.int32)

    def stage():
        _plan_fixed_prefix[(1,)](slots, cpu_ids, slot_to_id, evicted, fetched_count, eligible_mask, packed_ids,
                                 plan, count, gemm_slots, refill_trace, layer, cut, SCRATCH, EXPERTS,
                                 ROUTES, False, num_warps=4)
        native.stage_primary(source_table, source_table, layout_table, prefix_ptrs, plan, count,
                             experts=EXPERTS, packed_slots=cut)

    def stock():
        return fused_experts_decode_impl(hidden, *raw, weights, stock_slots)

    def candidate():
        stage()
        return fused_experts_decode_impl(hidden, *work, weights, gemm_slots)

    def direct():
        gate_up = torch.empty((8, 8, 1024), device=DEVICE, dtype=torch.bfloat16)
        activated = torch.empty((64, 512), device=DEVICE, dtype=torch.bfloat16)
        down = torch.empty((8, 8, 2048), device=DEVICE, dtype=torch.bfloat16)
        output = torch.empty_like(hidden)
        _decode_compact_gemv[(64, 32)](hidden, direct_tables[0], gate_up, weights, slots,
                                      1024, 2048, 32, 64, TILE_VALUES, PACKED_TILE_BYTES, False, False, num_warps=8)
        gated_act_and_mul("silu", gate_up.view(-1, 1024), activated, alpha=1.0, limit=float("inf"))
        _decode_compact_gemv[(64, 64)](activated, direct_tables[1], down, weights, slots,
                                      2048, 512, 32, 64, TILE_VALUES, PACKED_TILE_BYTES, True, True, num_warps=8)
        moe_sum_reduce_triton(down, output)
        return output

    expected = stock()
    for first in (True, False):
        if not first:
            fetched_count.zero_()
        output = candidate()
        for a, b in zip(raw, work, strict=True):
            if not torch.equal(a[stock_slots].view(torch.int16), b[gemm_slots].view(torch.int16)):
                raise RuntimeError(f"Fixed primary weights diverged on layer {layer}, fill={first}")
        if not torch.equal(expected.view(torch.int16), output.view(torch.int16)):
            raise RuntimeError(f"Fixed primary BF16 expert outputs diverged on layer {layer}, fill={first}")

    direct_result = direct()
    direct_exact = torch.equal(expected.view(torch.int16), direct_result.view(torch.int16))
    differing = int((expected.view(torch.int16) != direct_result.view(torch.int16)).sum().item())
    max_error = (expected.float() - direct_result.float()).abs().max().item()
    graphs = {}
    for name, fn in (("raw", stock), ("fixed-prefix", candidate), ("unpack-only", stage), ("direct-packed", direct)):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(REPLAYS):
                fn()
        graphs[name] = graph
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = {name: [] for name in graphs}
    for i in range(repeats):
        for name in graphs if i % 2 == 0 else reversed(graphs):
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end) / REPLAYS)
    return {"case": case, "layer": layer, "bf16_weight_and_expert_output_exact": True,
            "direct_expert_output_exact": direct_exact,
            "direct_different_values": differing, "direct_max_absolute_error": max_error,
            "packed_hits": int(eligible[:cut].sum()), "raw_refills": int((~eligible[:cut]).sum()),
            "raw_hits": EXPERTS - cut, "cpu_routes": cpu_routes,
            "median_ms": {name: median(samples) for name, samples in timings.items()}, "samples_ms": timings}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=24)
    args = p.parse_args()
    if args.out.exists() or args.repeats < 8:
        p.error("Use a fresh output path and at least eight repeats")
    report = json.loads(args.trace.read_text())
    manifest = json.loads((args.model / "lightreason-download.json").read_text())
    if report["revision"] != REVISION or report["dtype"] != "bfloat16" or manifest["revision"] != REVISION:
        p.error("Use the pinned Qwen BF16 checkpoint and trace")
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    native = NativeBf16Gather()
    runs = {r["case"]: r for r in report["rows"] if r["scenario"] == "gpu-packed-profile"}
    rows = []
    for case, layer in FIXTURES:
        row = fixture(args.model, index, case, layer, runs[case], native, args.repeats)
        rows.append(row)
        print(case, layer, row["packed_hits"], row["median_ms"], flush=True)
    with args.out.open("x") as stream:
        json.dump({"revision": REVISION, "repeats": args.repeats, "graph_replays_per_sample": REPLAYS,
                   "scope": "Single BF16 MoE layer, real checkpoint rows, GPU-only, synthetic routes sized by trace",
                   "rows": rows}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
