"""Test direct packed-weight loads in the existing grouped BF16 expert GEMM."""

import argparse
import json
from pathlib import Path
from random import Random
from statistics import median

import torch
import triton
import triton.language as tl

from edge35b.bf16_expert_codec import PROJECTIONS, checkpoint_bank
from edge35b.download import REVISION
from edge35b.gpu_resident_experts import GEOMETRY, device_rows
from edge35b.native_bf16_gather import NativeBf16Gather
from freetoken.kernel.triton.fused_moe import fused_moe_kernel
from freetoken.moe.fused import moe_align_block_size, try_get_optimal_moe_config


REPLAYS = 8
EXPERTS = 24
FIXTURES = (("studio", 2), ("known-math", 17), ("coding", 37), ("chat", 0))
MODES = ("raw", "direct", "shared-header", "staged")


def run_projection(model, index, layer, expert_ids, routes, coefficients, bank, repeats, gather):
    projection = PROJECTIONS[bank]
    weight = torch.from_numpy(checkpoint_bank(model, index, layer, projection)[expert_ids].copy()).view(torch.bfloat16)
    raw = weight.to("cuda")
    descriptors, owners, bytes_used = device_rows(weight, raw)
    n, k = weight.shape[1:]
    assert k % 256 == 0
    gen = torch.Generator(device="cuda").manual_seed(1173 + layer + bank)
    hidden = torch.randn((8 if bank == 0 else 64, k), device="cuda", dtype=torch.bfloat16, generator=gen)
    outputs = {mode: torch.empty((8, 8, n), device="cuda", dtype=torch.bfloat16) for mode in MODES}
    staged = torch.empty_like(raw)
    active = routes.unique().contiguous()
    sizes = torch.tensor([n * k * 2], device="cuda", dtype=torch.int64)
    count = torch.tensor([active.numel()], device="cuda", dtype=torch.int64)
    destinations = torch.tensor([staged.data_ptr()], device="cuda", dtype=torch.int64)
    config = try_get_optimal_moe_config((2560, 1024, 2048), (2560, 2048, 512), 8, 8)
    sorted_ids, expert_blocks, padded = moe_align_block_size(routes, config["BLOCK_SIZE_M"], EXPERTS)
    grid = (triton.cdiv(sorted_ids.numel(), config["BLOCK_SIZE_M"]) * triton.cdiv(n, config["BLOCK_SIZE_N"]),)

    def run(mode):
        direct = mode in ("direct", "shared-header")
        if mode == "staged":
            gather.copy(descriptors, destinations, sizes, active, active, count,
                        max_copies=active.numel(), experts=EXPERTS, geometry=GEOMETRY)
        source = descriptors if direct else staged if mode == "staged" else raw
        fused_moe_kernel[grid](
            hidden, source, outputs[mode], coefficients, sorted_ids, expert_blocks, padded,
            n, k, sorted_ids.numel(), routes.numel(), hidden.stride(0), hidden.stride(1),
            raw.stride(0), raw.stride(2), raw.stride(1), outputs[mode].stride(1), outputs[mode].stride(2),
            MUL_ROUTED_WEIGHT=bool(bank), top_k=1 if bank else 8, compute_type=tl.bfloat16,
            even_Ks=k % config["BLOCK_SIZE_K"] == 0, RESIDENT_BF16=direct,
            SHARED_HEADER=mode == "shared-header", **config,
        )

    graphs = {}
    for mode in MODES:
        run(mode)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(REPLAYS):
                run(mode)
        graphs[mode] = graph

    same = {mode: torch.equal(outputs["raw"].view(torch.int16), outputs[mode].view(torch.int16))
            for mode in MODES[1:]}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = {mode: [] for mode in MODES}
    for i in range(repeats):
        for mode in MODES if i % 2 == 0 else MODES[::-1]:
            start.record()
            graphs[mode].replay()
            end.record()
            end.synchronize()
            samples[mode].append(start.elapsed_time(end) / REPLAYS)
    # Keep the descriptor owners live until the last replay finishes.
    assert bytes_used == weight.numel() * 2 or owners
    return {"projection": projection, "shape": [8, n, k], "raw_bytes": weight.numel() * 2,
            "packed_bytes": bytes_used, "bitwise_match": same, "config": config, "active_experts": active.numel(),
            "median_ms": {mode: median(times) for mode, times in samples.items()}, "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=24)
    args = parser.parse_args()
    if args.repeats < 8 or args.out.exists():
        parser.error("Use a fresh output and at least eight alternating repeats")
    trace = json.loads(args.trace.read_text())
    if (trace["revision"] != REVISION or trace["dtype"] != "bfloat16"
            or json.loads((args.model / "lightreason-download.json").read_text())["revision"] != REVISION):
        parser.error("Use the pinned Qwen BF16 trace and checkpoint")
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    gather = NativeBf16Gather()
    rows = []
    for case, layer in FIXTURES:
        runs = [r for r in trace["rows"] if r["case"] == case and r["scenario"] == "fixed-eight-profile"]
        steps = runs[-1]["diagnostics"]["passes"]
        eligible = [step for step in steps if 0 < len(step["expert_fetch_ids"][layer]) <= EXPERTS]
        step = eligible[len(eligible) // 2]
        fetched = step["expert_fetch_ids"][layer]
        cpu_routes = step["gpu_layer_stamps"][layer][9]
        rng = Random(4637 + layer)
        keys = fetched + rng.sample(sorted(set(range(256)) - set(fetched)), EXPERTS - len(fetched))
        routes = torch.tensor([rng.sample(range(EXPERTS), 8) for _ in range(8)], device="cuda", dtype=torch.int32)
        inactive = rng.sample(range(64), cpu_routes)
        routes.view(-1)[inactive] = 0
        weights = torch.linspace(0.04, 0.12, 64, device="cuda", dtype=torch.float32).view(8, 8)
        weights.view(-1)[inactive] = 0
        for bank in range(len(PROJECTIONS)):
            row = run_projection(args.model, index, layer, keys, routes, weights, bank, args.repeats, gather)
            row.update(case=case, layer=layer, cpu_routes=cpu_routes, experts=keys)
            rows.append(row)
            print(case, layer, row["projection"], row["bitwise_match"], row["median_ms"], flush=True)
    with args.out.open("x") as stream:
        json.dump({"revision": REVISION, "repeats": args.repeats, "graph_replays_per_sample": REPLAYS,
                   "scope": "One Ada expert projection, real Qwen BF16 weights, trace-shaped synthetic routes",
                   "rows": rows}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
