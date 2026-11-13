"""Rank read-only GPU weight replicas from a development fetch trace.

The primary cache and CPU/GPU assignment stay the same. This file predicts
payload savings only; the timing and token-ID gates still require live runs.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from edge35b.download import REVISION


EXPERT_BYTES = 6 * 2**20
MAX_HOT_EXPERTS = 128
BUDGETS = (32, 64, 128)


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan["revision"] != REVISION or plan["width"] != 8 or plan["dtype"] != "bfloat16":
        raise ValueError("Hot cache plan must match the pinned BF16 eight-position verifier.")
    base = plan["base_transport"]
    if base not in ("raw", "packed"):
        raise ValueError("Hot cache base transport must be raw or packed.")
    ids = plan["experts"]
    if (not 1 <= len(ids) <= MAX_HOT_EXPERTS
            or any(len(pair) != 2 or any(type(i) is not int for i in pair)
                   or not 0 <= pair[0] < 40 or not 0 <= pair[1] < 256 for pair in ids)):
        raise ValueError("Invalid hot cache rows or budget.")
    ids = tuple(tuple(pair) for pair in ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate expert in the hot cache plan.")
    return base, ids


def tune(report, development_cases, budget, base="packed"):
    if (report["revision"] != REVISION or report["dtype"] != "bfloat16"
            or report["transport_settings"]["width"] != 8 or not report["transport_summary"]["exact"]):
        raise ValueError("The source comparison must pass the exact eight-position transfer gate.")
    if base not in ("raw", "packed"):
        raise ValueError("Unknown transport for tuning.")
    development_cases = set(development_cases)
    profile_rows = [
        r for r in report["rows"]
        if r["case"] in development_cases and r["scenario"] == f"{base}-profile"
    ]
    if not development_cases or development_cases != {r["case"] for r in profile_rows}:
        raise ValueError("Every development case needs an expert-fetch trace for the chosen transport.")
    if budget not in BUDGETS:
        raise ValueError(f"Choose an expert budget from {BUDGETS}.")

    counts, estimated = defaultdict(Counter), defaultdict(Counter)
    totals, tokens = Counter(), Counter()
    for row in profile_rows:
        case = row["case"]
        for p in row["diagnostics"]["passes"]:
            tokens[case] += p["committed"]
            layers = zip(p["expert_fetch_ids"], p["expert_pcie_payload_bytes"], strict=True)
            for layer, (experts, payload) in enumerate(layers):
                if (not experts and payload != 0) or not 0 <= payload <= len(experts) * EXPERT_BYTES:
                    raise ValueError(f"Inconsistent payload record on layer {layer}.")
                totals[case] += payload
                for expert in experts:
                    counts[case][layer, expert] += 1
                    # Source reports layer totals, so use the mean byte size of each fetch group.
                    estimated[case][layer, expert] += payload / len(experts)

    score = Counter()
    for case, weights in estimated.items():
        for key, size in weights.items():
            score[key] += size / tokens[case]
    ranking = sorted(score, key=lambda key: (-score[key], key))
    if not ranking:
        raise ValueError("Development traces did not fetch experts.")

    def projection(n):
        chosen = ranking[:n]
        return {"experts": len(chosen), "additional_vram_bytes": len(chosen) * EXPERT_BYTES,
                "development": {
                    case: {
                        "fetches_avoided": sum(counts[case][key] for key in chosen),
                        "estimated_pcie_bytes_avoided": round(sum(estimated[case][key] for key in chosen)),
                        "traced_pcie_bytes": totals[case],
                    } for case in sorted(development_cases)
                }}

    return {
        "revision": REVISION, "width": 8, "dtype": "bfloat16", "base_transport": base,
        "development_cases": sorted(development_cases),
        "experts": [list(key) for key in ranking[:budget]],
        "budgets": [projection(n) for n in BUDGETS],
        "estimate": "Layer mean payload per fetch. No runtime speedup is implied.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--development", default="studio", help="Comma-separated development cases")
    p.add_argument("--budget", type=int, choices=BUDGETS, default=64)
    p.add_argument("--base", choices=("raw", "packed"), default="packed")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    plan = tune(json.loads(args.report.read_text()), args.development.split(","), args.budget, args.base)
    with args.out.open("x") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    print(json.dumps(plan["budgets"], indent=2))


if __name__ == "__main__":
    main()
