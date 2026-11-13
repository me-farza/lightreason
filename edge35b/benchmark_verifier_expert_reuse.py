"""Time exact BF16 expert-weight reuse across two verifier rows on Ada."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from random import Random
from statistics import median

import torch
import triton
import triton.language as tl

from edge35b.bf16_expert_codec import PROJECTIONS, checkpoint_bank
from edge35b.download import REVISION


WIDTH = 8
TOP_K = 8
ROUTES = WIDTH * TOP_K
REPLAYS = 8
FIXTURES = (("studio", 2), ("known-math", 17), ("coding", 37), ("chat", 0))


@triton.jit
def _pair_plan(slots, weights, pairs, ROUTES: tl.constexpr):
    i = tl.arange(0, ROUTES)
    expert = tl.load(slots + i)
    active = tl.load(weights + i) != 0
    matching = (expert[:, None] == expert[None, :]) & active[None, :]
    rank = tl.sum((matching & (i[None, :] < i[:, None])).to(tl.int32), axis=1)
    partner = tl.min(tl.where(matching & (i[None, :] > i[:, None]), i[None, :], ROUTES), axis=1)
    tl.store(pairs + i, tl.where(active & (rank % 2 == 0), partner, -1))


@triton.jit
def _paired_gemv(a_ptr, b_ptr, out, weights, slots, pairs, N: tl.constexpr, K: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr, ROUTES: tl.constexpr, TOP_K: tl.constexpr,
                 ROUTED_INPUT: tl.constexpr):
    r0 = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    weight = tl.load(weights + r0)
    r1 = tl.load(pairs + r0)
    if weight == 0:
        tl.store(out + r0 * N + n, 0)
    elif r1 >= 0:
        expert = tl.load(slots + r0)
        a0 = a_ptr + (r0 if ROUTED_INPUT else r0 // TOP_K) * K + k
        a1 = a_ptr + (r1 if ROUTED_INPUT else r1 // TOP_K) * K + k
        b = b_ptr + expert.to(tl.int64) * N * K + n[None, :] * K + k[:, None]
        acc0 = tl.zeros((BN,), dtype=tl.float32)
        acc1 = tl.zeros((BN,), dtype=tl.float32)
        for _ in range(K // BK):
            x0 = tl.load(a0)
            tile = tl.load(b, mask=n[None, :] < N, other=0.0)
            acc0 += tl.sum(x0[:, None] * tile, axis=0)
            if r1 < ROUTES:
                x1 = tl.load(a1)
                acc1 += tl.sum(x1[:, None] * tile, axis=0)
            a0 += BK
            a1 += BK
            b += BK
        if ROUTED_INPUT:
            acc0 = acc0 * weight
        tl.store(out + r0 * N + n, acc0.to(tl.bfloat16))
        if r1 < ROUTES:
            if ROUTED_INPUT:
                acc1 = acc1 * tl.load(weights + r1)
            tl.store(out + r1 * N + n, acc1.to(tl.bfloat16))


@triton.jit
def _skip_gemv(a_ptr, b_ptr, out, weights, slots, N: tl.constexpr, K: tl.constexpr,
               BN: tl.constexpr, BK: tl.constexpr, TOP_K: tl.constexpr, ROUTED_INPUT: tl.constexpr):
    route = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    weight = tl.load(weights + route)
    acc = tl.zeros((BN,), tl.float32)
    if weight != 0:
        expert = tl.load(slots + route)
        a = a_ptr + (route if ROUTED_INPUT else route // TOP_K) * K + k
        b = b_ptr + expert.to(tl.int64) * N * K + n[None, :] * K + k[:, None]
        for _ in range(K // BK):
            x = tl.load(a)
            tile = tl.load(b, mask=n[None, :] < N, other=0.0)
            acc += tl.sum(x[:, None] * tile, axis=0)
            a += BK
            b += BK
        if ROUTED_INPUT:
            acc *= weight
    tl.store(out + route * N + n, acc.to(tl.bfloat16))


def gpu_experts(hidden, gate, down, weights, slots, *, pair):
    from freetoken.kernel import moe_sum_reduce_triton
    from freetoken.layers import gated_act_and_mul

    if hidden.shape != (WIDTH, 2048) or weights.shape != (WIDTH, TOP_K) or slots.shape != weights.shape:
        raise ValueError("Expected the eight-position Qwen verifier")
    if pair:
        pairs = torch.empty(ROUTES, dtype=torch.int32, device=hidden.device)
        _pair_plan[(1,)](slots, weights, pairs, ROUTES, num_warps=4)
    gate_out = torch.empty((WIDTH, TOP_K, 1024), dtype=torch.bfloat16, device=hidden.device)
    gated = torch.empty((ROUTES, 512), dtype=torch.bfloat16, device=hidden.device)
    down_out = torch.empty((WIDTH, TOP_K, 2048), dtype=torch.bfloat16, device=hidden.device)
    if pair:
        _paired_gemv[(ROUTES, 32)](hidden, gate, gate_out, weights, slots, pairs, 1024, 2048,
                                   32, 64, ROUTES, TOP_K, False, num_warps=8)
    else:
        _skip_gemv[(ROUTES, 32)](hidden, gate, gate_out, weights, slots, 1024, 2048,
                                 32, 64, TOP_K, False, num_warps=8)
    gated_act_and_mul("silu", gate_out.view(ROUTES, 1024), gated, alpha=1.0, limit=float("inf"))
    if pair:
        _paired_gemv[(ROUTES, 64)](gated, down, down_out, weights, slots, pairs, 2048, 512,
                                   32, 64, ROUTES, TOP_K, True, num_warps=8)
    else:
        _skip_gemv[(ROUTES, 64)](gated, down, down_out, weights, slots, 2048, 512,
                                 32, 64, TOP_K, True, num_warps=8)
    out = torch.empty_like(hidden)
    moe_sum_reduce_triton(down_out, out)
    return out


@dataclass
class Fixture:
    hidden: torch.Tensor
    banks: list[torch.Tensor]
    weights: torch.Tensor
    slots: torch.Tensor
    trace: dict


def prepare(model, index, profile, layer, *, rotating=False):
    candidates = [s for s in profile["diagnostics"]["passes"] if 2 <= len(s["expert_fetch_ids"][layer]) <= 28]
    step = candidates[len(candidates) // 2]
    fetched = step["expert_fetch_ids"][layer]
    cpu_routes = step["gpu_layer_stamps"][layer][9]
    active_count = ROUTES - cpu_routes
    distinct = active_count * 3 // 5 if rotating else 16
    count = min(active_count, max(distinct, len(fetched)))
    rng = Random(8101 + layer)
    keys = fetched + rng.sample(sorted(set(range(256)) - set(fetched)), count - len(fetched))
    banks = [torch.from_numpy(checkpoint_bank(model, index, layer, bank)[keys].copy()).view(torch.bfloat16).cuda()
             for bank in PROJECTIONS]
    if rotating:
        # Match the observed inactive route count and keep top-k IDs unique per row.
        inactive = set(rng.sample(range(ROUTES), cpu_routes))
        unused = rng.sample(range(count), count)
        slots, weights, active = [], [], []
        for row in range(WIDTH):
            experts = set()
            for route in range(row * TOP_K, (row + 1) * TOP_K):
                if route in inactive:
                    slots.append(0)
                    weights.append(0.)
                    continue
                expert = unused.pop() if unused else rng.choice(sorted(set(range(count)) - experts))
                experts.add(expert)
                active.append(expert)
                slots.append(expert)
                weights.append(rng.uniform(0.03, 0.14))
    else:
        active = list(range(count)) + rng.choices(range(count), k=active_count - count)
        slots = active + [0] * cpu_routes
        weights = [rng.uniform(0.03, 0.14) for _ in active] + [0.] * cpu_routes
        order = rng.sample(range(ROUTES), ROUTES)
        slots, weights = [slots[i] for i in order], [weights[i] for i in order]
    slots = torch.tensor(slots, device="cuda", dtype=torch.int32).view(WIDTH, TOP_K)
    weights = torch.tensor(weights, device="cuda", dtype=torch.float32).view(WIDTH, TOP_K)
    hidden = torch.randn((WIDTH, 2048), dtype=torch.bfloat16, device="cuda",
                         generator=torch.Generator(device="cuda").manual_seed(layer + 2801))
    groups = sum((active.count(e) + 1) // 2 for e in range(count))
    return Fixture(hidden, banks, weights, slots, {"case": profile["case"], "layer": layer,
                   "cpu_routes": cpu_routes, "active_routes": active_count, "experts": count,
                   "weight_read_groups": groups, "weight_bytes": sum(bank.nbytes for bank in banks)})


def measure(fixtures, repeats):
    from freetoken.moe.fused import fused_experts_decode_impl

    def raw(fixture):
        return fused_experts_decode_impl(fixture.hidden, *fixture.banks, fixture.weights, fixture.slots)

    def grouped(fixture, *, pair):
        return gpu_experts(fixture.hidden, *fixture.banks, fixture.weights, fixture.slots, pair=pair)

    expected = [raw(fixture) for fixture in fixtures]
    graphs, checks = {}, [{} for _ in fixtures]
    for name, run in (("control", raw), ("skip-zero-routes", lambda f: grouped(f, pair=False)),
                      ("paired", lambda f: grouped(f, pair=True))):
        for fixture in fixtures:
            run(fixture)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(REPLAYS):
                outputs = [run(fixture) for fixture in fixtures]
        graph.replay()
        for i, (ref, output) in enumerate(zip(expected, outputs, strict=True)):
            delta = (ref.float() - output.float()).abs()
            changed = (ref.view(torch.int16) != output.view(torch.int16)).sum().item()
            checks[i][name] = {"bitwise": not changed, "max_error": delta.max().item(), "different_values": changed}
        graphs[name] = graph

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = {name: [] for name in graphs}
    for repeat in range(repeats):
        for name in graphs if repeat % 2 == 0 else reversed(graphs):
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end) / (REPLAYS * len(fixtures)))
    return {"fixtures": [{**fixture.trace, "checks": check} for fixture, check in zip(fixtures, checks, strict=True)],
            "samples_ms": timings, "median_ms": {name: median(samples) for name, samples in timings.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", default=24, type=int)
    parser.add_argument("--rotating", action="store_true", help="Cycle through different layers to test sustained weight traffic")
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    manifest = json.loads((args.model / "lightreason-download.json").read_text())
    if trace["revision"] != REVISION or trace["dtype"] != "bfloat16" or manifest["revision"] != REVISION:
        parser.error("Use the pinned BF16 checkpoint and matching verifier trace")
    if args.out.exists() or args.repeats < 8:
        parser.error("Use a fresh output and at least eight alternating repeats")
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    profiles = {r["case"]: r for r in trace["rows"] if r["scenario"] == "gpu-packed-profile"}
    rows = []
    sequences = (FIXTURES + (("studio", 4), ("coding", 24), ("chat", 30), ("known-math", 36)),) if args.rotating else (
        (item,) for item in FIXTURES
    )
    for sequence in sequences:
        fixtures = [prepare(args.model, index, profiles[case], layer, rotating=args.rotating) for case, layer in sequence]
        measured = measure(fixtures, args.repeats)
        if args.rotating:
            row = measured
        else:
            row = {**measured.pop("fixtures")[0], **measured}
        rows.append(row)
        print(sequence, row["median_ms"], flush=True)
    with args.out.open("x") as output:
        json.dump({"revision": REVISION, "rows": rows, "repeats": args.repeats, "replays": REPLAYS,
                   "rotating": args.rotating,
                   "scope": "Original Qwen BF16 expert rows and synthetic eight-position routes sized by the frozen trace"},
                  output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
