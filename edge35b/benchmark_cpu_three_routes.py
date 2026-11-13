"""Test sharing one exact BF16 expert weight read across three CPU routes."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from statistics import median

import torch

from edge35b.benchmark_cpu_reads import ExpertFixture, fixtures
from edge35b.download import REVISION


def clustered_routes(shape, offset):
    """Keep the recorded route and expert counts while concentrating repetitions."""
    counts = [1] * shape.experts
    extra = shape.routes - shape.experts
    for expert in range(shape.experts):
        take = min(extra, 7)
        counts[expert] += take
        extra -= take
    if extra:
        raise ValueError("The route count exceeds the eight-position fixture")
    ids = torch.full((8, 8), -1, dtype=torch.int32)
    token_sizes = [0] * 8
    for expert, count in enumerate(counts):
        for _ in range(count):
            token = offset % 8
            ids[token, token_sizes[token]] = expert
            token_sizes[token] += 1
            offset += 1
    return ids


def check(fixture, ids, policy, reverse):
    fixture.ids.copy_(ids)
    routes = Counter(e for e in ids.flatten().tolist() if e >= 0)
    observed = {}
    for enabled in (True, False) if reverse else (False, True):
        fixture.worker.set_three_route_reuse(enabled)
        ms, bits = fixture._run(policy)
        expected = sum((n + (2 if enabled else 1)) // (3 if enabled else 2) for n in routes.values())
        groups = fixture.worker.weight_read_groups()
        if groups != expected:
            raise RuntimeError(f"Weight read group count {groups} differs from {expected} for {dict(routes)}")
        observed[enabled] = (ms, bits, groups)
    a_ms, a_bits, a_groups = observed[False]
    b_ms, b_bits, b_groups = observed[True]
    if not torch.equal(a_bits, b_bits):
        raise RuntimeError(f"Three-route grouping changed BF16 output on {fixture.fixture}, {policy}")
    return a_ms, b_ms, a_groups, b_groups


def measure(fixture, policy):
    cases = (("balanced", fixture.inputs),
             ("clustered", [clustered_routes(fixture.fixture, i) for i in range(len(fixture.inputs))]))
    rows = []
    for pattern, inputs in cases:
        pairs = [check(fixture, ids, policy, i % 2 != 0) for i, ids in enumerate(inputs)][4:]
        a_ms, b_ms, a_groups, b_groups = map(median, zip(*pairs, strict=True))
        rows.append({"shape": asdict(fixture.fixture), "pattern": pattern, "read_policy": policy,
                     "two_route_ms": a_ms, "three_route_ms": b_ms, "two_route_groups": a_groups,
                     "three_route_groups": b_groups, "paired_speedup": median(a / b for a, b, _, _ in pairs),
                     "bitwise_match": True, "pairs": pairs})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--repeats", default=24, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() or args.repeats < 8:
        parser.error("Use a fresh output and at least eight alternating pairs")
    if json.loads((args.model / "lightreason-download.json").read_text())["revision"] != REVISION:
        parser.error("Use the pinned BF16 Qwen checkpoint")
    torch.set_num_threads(1)
    profile = json.loads(args.profile.read_text())
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    shapes = fixtures(profile, "studio", (0.20, 0.60, 0.90))
    for case in ("known-math", "coding", "chat"):
        shapes.extend(fixtures(profile, case, (0.70,), [shape.layer for shape in shapes]))
    rows = []
    for shape in shapes:
        fixture = ExpertFixture(args.model, index, shape, args.repeats)
        for policy in ("stock", "packed-both-prefetch"):
            rows.extend(measure(fixture, policy))
        del fixture
        print(shape, flush=True)
    result = {"revision": REVISION, "repeats": args.repeats,
              "scope": "Real BF16 Qwen expert banks, trace-shaped synthetic routes, eight CPU workers",
              "rows": rows}
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
