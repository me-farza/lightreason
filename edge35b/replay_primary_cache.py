"""Replay a compact GPU tier against frozen verifier fetches and byte counts."""

import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path

from edge35b.download import REVISION
from edge35b.plan_gpu_resident_cache import read_plan
from edge35b.replay_hot_expert_cache import replay
from edge35b.tune_hot_expert_cache import read_plan as read_hot


def count_admissions(sequence, slots, repeat_window):
    cache = OrderedDict()
    previous = {}
    hits = inserted = 0
    for i, key in enumerate(sequence):
        if key in cache:
            hits += 1
            cache.move_to_end(key)
        elif key in previous and i - previous[key] <= repeat_window:
            inserted += 1
            cache[key] = None
            if len(cache) > slots:
                cache.popitem(last=False)
        previous[key] = i
    return hits, inserted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("hot_plan", type=Path)
    parser.add_argument("--resident-plan", type=Path, help="Also evaluate the next ranked, preloaded rows")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report["revision"] != REVISION or report["dtype"] != "bfloat16":
        parser.error("Expected the pinned Qwen BF16 report")
    rows = []
    _, original = read_hot(args.hot_plan)
    resident = set(original) | {tuple(key) for key in report["gpu_resident_cache"]["manifest"]["experts"]["packed"]}
    extra = [] if args.resident_plan is None else [key for key in read_plan(args.resident_plan, original)
                                                   if key not in resident]
    for row in report["rows"]:
        if row["scenario"] != "gpu-packed-profile":
            continue
        sequence, approx_bytes = [], Counter()
        for step in row["diagnostics"]["passes"]:
            for layer, (ids, payload) in enumerate(zip(step["expert_fetch_ids"], step["expert_pcie_payload_bytes"],
                                                       strict=True)):
                cold = [(layer, e) for e in ids if (layer, e) not in resident]
                sequence.extend(cold)
                for key in cold:
                    approx_bytes[key] += payload / len(cold)
        counts = Counter(sequence)
        capacities = []
        for slots in (128, 512, 624, 1024):
            lru, offline = replay(sequence, slots)
            capacities.append({"slots": slots, "lru_hits": lru, "future_aware_hits": offline,
                               "best_static_hits": sum(n for _, n in counts.most_common(slots)),
                               "repeat_admission": {str(window): count_admissions(sequence, slots, window)
                                                    for window in (1024, 4096, 8192)}})
        estimates = []
        total_bytes = approx_bytes.total()
        for slots in (128, 192, 240, 256):
            if len(extra) >= slots:
                estimates.append({"slots": slots, "ranked_hits": sum(counts[key] for key in extra[:slots]),
                                  "ranked_saved_byte_share": sum(approx_bytes[key] for key in extra[:slots]) / total_bytes,
                                  "hindsight_static_byte_share": (sum(n for _, n in approx_bytes.most_common(slots))
                                                                  / total_bytes)})
        rows.append({"case": row["case"], "fetches": len(sequence), "capacities": capacities,
                     "preloaded_rows_estimates": estimates})
    with args.out.open("x") as stream:
        json.dump({"revision": REVISION, "scope": "Fixed PCIe fetch sequence after excluding existing GPU replicas. No prefill seed or timing.",
                   "rows": rows}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
