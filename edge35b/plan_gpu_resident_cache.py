"""Rank GPU expert replicas using the separate eight-position calibration runs."""

import argparse
from collections import Counter
import json
from pathlib import Path

from edge35b.download import REVISION
from edge35b.tune_hot_expert_cache import read_plan as read_hot_plan


def read_plan(path, hot):
    plan = json.loads(Path(path).read_text())
    keys = [tuple(key) for key in plan["experts"]]
    if (plan["revision"] != REVISION or plan["dtype"] != "bfloat16" or plan["width"] != 8
            or not 1 <= len(keys) <= 1024 or len(keys) != len(set(keys)) or set(keys) & set(hot)
            or any(len(key) != 2 or any(type(i) is not int for i in key)
                   or not 0 <= key[0] < 40 or not 0 <= key[1] < 256 for key in keys)):
        raise ValueError("Expected distinct Qwen BF16 replicas outside the original hot plan")
    return keys


def rank(report, hot, count):
    if report["revision"] != REVISION or report["dtype"] != "bfloat16":
        raise ValueError("Use the pinned BF16 profile")
    calibration = [row for row in report["rows"] if row["scenario"] == "budget-calibration-8"]
    hot = set(hot)
    score = Counter()
    for row in calibration:
        tokens = sum(p[1] for p in row["passes"])
        if tokens <= 0 or row["verify_width"] != 8 or row["cached_prompt_tokens"] != 0:
            raise ValueError("Expected fixed-width calibration with uncached input")
        for step in row["diagnostics"]["passes"]:
            layers = zip(step["expert_fetch_ids"], step["expert_pcie_payload_bytes"], strict=True)
            for layer, (ids, payload) in enumerate(layers):
                cold = [(layer, expert) for expert in ids if (layer, expert) not in hot]
                for key in cold:
                    score[key] += payload / (len(cold) * tokens)
    if len(calibration) < 2 or len(score) < count:
        raise ValueError("The calibration trace does not cover the requested resident budget")
    return {"revision": REVISION, "dtype": "bfloat16", "width": 8,
            "calibration_cases": [row["case"] for row in calibration],
            "experts": sorted(score, key=lambda key: (-score[key], key))[:count]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("hot_plan", type=Path)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.rows <= 1024:
        parser.error("Use one to 1,024 ranked rows")
    _, hot = read_hot_plan(args.hot_plan)
    result = rank(json.loads(args.profile.read_text()), hot, args.rows)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
