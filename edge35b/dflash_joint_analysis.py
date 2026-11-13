"""Estimate DFlash2 acceptance headroom from frozen trace prefixes."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import mean

from edge35b.analyze_pass_diagnostics import profile_pass, summarize_recall
from edge35b.download import REVISION
from edge35b.draft_budget import forecast_commits
from edge35b.expert_transport_eval import first_different_id


def analyze(audit, resident):
    if any(r["revision"] != REVISION or r["dtype"] != "bfloat16" for r in (audit, resident)):
        raise ValueError("Use the pinned BF16 target traces")
    controls = {row["case"]: row for row in resident["rows"] if row["scenario"] == "gpu-packed-profile"}
    results = []
    for row in audit["rows"]:
        if row["scenario"] != "audit-8":
            continue
        control = controls[row["case"]]
        commits = [p[:2] for p in row["passes"]]
        if (first_different_id(control["token_ids"], row["token_ids"]) is not None
                or control["finish_reason"] != row["finish_reason"]
                or [p[:2] for p in control["passes"]] != commits):
            raise ValueError(f"The recorded control and shortlist use different prefixes on {row['case']}")
        ms = mean(p[2] for p in control["passes"])
        recall = summarize_recall(row, ms)
        passes = control["diagnostics"]["passes"]
        pcie = mean(sum(p["expert_pcie_payload_bytes"]) for p in passes)
        cpu_reads = mean(sum(p["cpu_weight_read_groups"]) for p in passes)
        cpu_unique = mean(sum(stamp[8] - stamp[7] for stamp in p["gpu_layer_stamps"]) for p in passes)
        gather_ms = mean(sum(s[3] - s[2] for s in p["gpu_layer_stamps"]) / 1e6 for p in passes)
        live = recall["live_ids_per_pass"]
        oracle = recall["oracle_ids_per_pass"]
        results.append({"case": row["case"], "ms_per_pass": ms, "actual_ids_per_pass": live,
                        "shortlist_oracle_ids_per_pass": oracle, "actual_pcie_bytes_per_id": pcie / live,
                        "actual_cpu_read_groups_per_id": cpu_reads / live,
                        "cpu_read_groups_per_pass": cpu_reads,
                        "distinct_cpu_experts_per_pass": cpu_unique,
                        "fraction_cpu_read_groups_redundant": 1 - cpu_unique / cpu_reads,
                        "actual_gather_ms_per_pass": gather_ms,
                        "oracle_pcie_bytes_per_id_at_fixed_traffic": pcie / oracle,
                        "shortlist_oracle_tok_s_at_fixed_cost": 1000 * oracle / ms,
                        "all_eight_tok_s_at_fixed_cost": 8000 / ms,
                        "pass_ms_budget_for_100_at_actual_acceptance": 10 * live,
                        "pass_ms_budget_for_100_with_oracle_acceptance": 10 * oracle})
    return {"revision": REVISION, "scope": "Same archived prefixes. Hindsight recall holds pass boundaries and cost fixed.",
            "cases": results}


def matching_passes(row, prefix_ids):
    output_ids, pairs = 1, []
    for step, diag in zip(row["passes"], row["diagnostics"]["passes"], strict=True):
        width, committed, _ = step
        if (width, committed) != (diag["width"], diag["committed"]):
            raise ValueError("Verification passes and profiles disagree")
        if output_ids + committed > prefix_ids:
            break
        output_ids += committed
        pairs.append((step, diag))
    if not pairs:
        return {"passes": 0}
    ids = output_ids - 1
    elapsed = sum(step[2] for step, _ in pairs)
    return {"passes": len(pairs), "output_ids": output_ids, "committed_ids_per_pass": ids / len(pairs),
            "ms_per_pass": elapsed / len(pairs), "tok_s": 1000 * ids / elapsed,
            "gather_ms_per_pass": mean(profile_pass(diag, step[2])["pcie_gather_ms"] for step, diag in pairs),
            "pcie_bytes_per_id": sum(sum(diag["expert_pcie_payload_bytes"]) for _, diag in pairs) / ids,
            "cpu_read_groups_per_id": sum(sum(diag["cpu_weight_read_groups"]) for _, diag in pairs) / ids}


def analyze_teacher(live):
    if live["revision"] != REVISION or live["dtype"] != "bfloat16":
        raise ValueError("Use the pinned BF16 live trial")
    controls = {r["case"]: r for r in live["rows"] if r["scenario"] == "gpu-packed-profile"}
    results = []
    for row in live["rows"]:
        if row["scenario"] != "teacher-profile":
            continue
        control = controls[row["case"]]
        if not row["draft_teacher"] or row["verify_width"] != 8 or control["verify_width"] != 8:
            raise ValueError("Use the eight-position continuation and packed control profiles")
        mismatch = first_different_id(control["token_ids"], row["token_ids"])
        prefix = mismatch - 1 if mismatch is not None else len(control["token_ids"])
        results.append({"case": row["case"], "first_different_id": mismatch, "matching_prefix_ids": prefix,
                        "control": matching_passes(control, prefix), "teacher": matching_passes(row, prefix)})
    if not results:
        raise ValueError("Missing live continuation diagnostic")
    return {"scope": "Complete profiled verification passes before the first mismatch. Teacher proposals use saved IDs.",
            "cases": results}


def acceptance_calibration(live):
    if live["revision"] != REVISION or live["dtype"] != "bfloat16":
        raise ValueError("Use the pinned BF16 live trial")
    probabilities = live["dflash_prefix"]["policy"]["probabilities"]
    result = {}
    for mode in ("stock-adaptive-profile", "prefix-adaptive-profile"):
        totals = []
        for row in live["rows"]:
            if row["scenario"] != mode:
                continue
            emitted = 1
            for p in row["diagnostics"]["passes"]:
                width, commits = p["width"], p["committed"]
                remaining = row["cap"] - emitted
                emitted += commits
                if remaining > width and emitted < len(row["token_ids"]):
                    forecast = forecast_commits(width, p["draft_confidence"], probabilities, remaining)
                    totals.append((forecast, commits))
            if emitted != len(row["token_ids"]):
                raise ValueError("Profile commits do not account for every output ID")
        result[mode] = {"passes": len(totals), "mean_forecast_commits": mean(p for p, _ in totals),
                        "mean_actual_commits": mean(a for _, a in totals),
                        "mean_absolute_error": mean(abs(p - a) for p, a in totals)}
    return result


def load_report(path):
    read = gzip.open if path.suffix == ".gz" else open
    with read(path, "rt") as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("resident", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--live", type=Path, help="Include the live acceptance calibration and continuation diagnostic")
    args = parser.parse_args()
    report = analyze(load_report(args.audit), load_report(args.resident))
    if args.live:
        live = load_report(args.live)
        report["acceptance_calibration"] = acceptance_calibration(live)
        if any(row["scenario"] == "teacher-profile" for row in live["rows"]):
            report["live_teacher"] = analyze_teacher(live)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
