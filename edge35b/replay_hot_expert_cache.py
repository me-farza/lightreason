"""Estimate a second-tier cache's headroom without changing the target routing."""

import argparse
from collections import Counter, OrderedDict, defaultdict, deque
import json
from pathlib import Path

from edge35b.tune_hot_expert_cache import EXPERT_BYTES, read_plan


def replay(sequence, capacity):
    """LRU and offline optimal replacement with optional bypass, both starting empty."""
    future = defaultdict(deque)
    for i, expert in enumerate(sequence):
        future[expert].append(i)
    lru, oracle = OrderedDict(), {}
    lru_hits = oracle_hits = 0
    for i, expert in enumerate(sequence):
        assert future[expert].popleft() == i
        next_use = future[expert][0] if future[expert] else len(sequence)
        if expert in lru:
            lru_hits += 1
            lru.move_to_end(expert)
        else:
            lru[expert] = None
            if len(lru) > capacity:
                lru.popitem(last=False)
        if expert in oracle:
            oracle_hits += 1
            oracle[expert] = next_use
        elif len(oracle) < capacity:
            oracle[expert] = next_use
        else:
            victim = max(oracle, key=oracle.get)
            if next_use < oracle[victim]:
                del oracle[victim]
                oracle[expert] = next_use
    return lru_hits, oracle_hits


def run(report, plan):
    rows = []
    seen = set()
    for row in report["rows"]:
        case = row["case"]
        if row["scenario"] != "raw-profile" or case in seen:
            continue
        seen.add(case)
        passes = row["diagnostics"]["passes"]
        sequence = [(layer, e) for p in passes for layer, ids in enumerate(p["expert_fetch_ids"]) for e in ids]
        frequencies = Counter(sequence)
        budgets = []
        for n in (32, 64, 128, 192, 256):
            lru, oracle = replay(sequence, n)
            budgets.append({
                "rows": n, "vram_mib": n * EXPERT_BYTES // 2**20,
                "dev_static_hits": sum(frequencies[e] for e in plan[:n]) if n <= len(plan) else None,
                "prompt_static_oracle_hits": sum(v for _, v in frequencies.most_common(n)),
                "second_lru_hits": lru, "second_fetch_oracle_hits": oracle,
            })
        rows.append({"case": case, "fetches": len(sequence), "passes": len(passes), "budgets": budgets})
    return {
        "source": "One exact raw-profile episode per prompt",
        "interpretation": (
            "Stock routing unchanged. Static oracle knows the entire answer before loading. "
            "Dynamic overlays start empty, add only rows already fetched by the stock primary cache, "
            "and ignore copies/metadata overhead. The fetch oracle knows future demand and may bypass insertion. "
            "These are hit counts, not timing results, and do not bound a policy that can also retain rows from prefill."
        ),
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("plan", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    _, plan = read_plan(args.plan)
    report = json.loads(args.report.read_text())
    if not report["transport_summary"]["exact"]:
        p.error("Expected an exact transfer comparison.")
    result = run(report, plan)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
