"""Check captured verifier passes and summarize timing and candidate coverage."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median


LAYER_SPANS = ("route", "publish", "pcie_gather", "gpu_experts", "cpu_wait", "combine")
EXPERT_BYTES = 6 * 1024 ** 2


def profile_pass(pass_record, scheduler_ms):
    result = {f"{stage}_ms": 0.0 for stage in LAYER_SPANS}
    result.update(fetched=0, missed=0, cpu_experts=0, cpu_routes=0, cpu_service_ms=0.0)
    for row, service_ns in zip(pass_record["gpu_layer_stamps"], pass_record["cpu_service_ns"], strict=True):
        stamps, fetched, missed, cpu_routes = row[:7], *row[7:]
        if (stamps[0] <= 0 or any(end < start for start, end in zip(stamps, stamps[1:]))
                or not 0 <= fetched <= missed or not 0 <= missed - fetched <= cpu_routes
                or (cpu_routes and service_ns <= 0)):
            raise ValueError(f"Inconsistent GPU timing or expert counters: {row}, CPU {service_ns}")
        for stage, start, end in zip(LAYER_SPANS, stamps, stamps[1:]):
            result[f"{stage}_ms"] += (end - start) / 1_000_000
        result["fetched"] += fetched
        result["missed"] += missed
        result["cpu_experts"] += missed - fetched
        result["cpu_routes"] += cpu_routes
        result["cpu_service_ms"] += service_ns / 1_000_000 if cpu_routes else 0.0
    result["h2d_GB"] = result["fetched"] * EXPERT_BYTES / 1_000_000_000
    result["target_ms"] = pass_record["target_graph_ms"]
    result["other_target_ms"] = result["target_ms"] - sum(result[f"{stage}_ms"] for stage in LAYER_SPANS)
    if result["other_target_ms"] < -0.1:
        raise ValueError("Layer marks extend past the measured target graph")
    result["scheduler_ms"] = scheduler_ms
    result["committed"] = pass_record["committed"]
    return result


def summarize_profile(run):
    passes = [profile_pass(record, timing[2])
              for record, timing in zip(run["diagnostics"]["passes"], run["passes"], strict=True)]
    if len(passes) < 37:
        raise ValueError("The timing comparison needs five starting and 32 later passes")

    def summarized(rows):
        if not rows:
            return {"passes": 0}
        return {"passes": len(rows), **{k: mean(r[k] for r in rows) for k in rows[0]}}

    def rate(rows):
        return 1000 * sum(r["committed"] for r in rows) / sum(r["scheduler_ms"] for r in rows)

    start = max(range(5, len(passes) - 31), key=lambda i: rate(passes[i:i + 32]))
    fast = passes[start:start + 32]
    others = passes[:start] + passes[start + 32:]
    remaining = sorted(passes[5:], key=lambda p: p["target_ms"])
    quartile = len(remaining) // 4
    return {
        "case": run["case"], "width": run["verify_width"], "repeat": run["repeat"],
        "rate": run["timings"]["predicted_per_second"],
        "fast_window_start": start + 1, "fast_window_rate": rate(fast),
        "full": summarized(passes), "first_five": summarized(passes[:5]),
        "fast_window": summarized(fast), "outside_window": summarized(others),
        "outside_window_after_startup": summarized(passes[5:start] + passes[start + 32:]),
        "fast_quartile_after_startup": summarized(remaining[:quartile]),
        "slow_quartile_after_startup": summarized(remaining[-quartile:]),
        "per_pass": passes,
    }


def summarize_recall(run, ordinary_ms=None):
    ids = run["token_ids"]
    passes = run["diagnostics"]["passes"]
    prefix = 1
    rejection_positions = Counter()
    recovered_ranks = Counter()
    position_tested = Counter()
    position_recalled = Counter()
    oracle_committed = 0
    independent_top1 = 0
    fixed_pool_oracle = []
    for p in passes:
        drafts, candidates = p["draft_ids"], p["candidate_ids"]
        accepted, committed = p["accepted_drafts"], p["committed"]
        width = p["width"]
        continuation = ids[prefix:prefix + width]
        if (len(drafts) != width - 1 or len(candidates) != width - 1 or not 1 <= committed <= accepted + 1
                or not 0 <= accepted < width or len(continuation) < committed
                or p["target_ids"][:committed] != continuation[:committed]
                or drafts[:min(accepted, committed)] != continuation[:min(accepted, committed)]
                or any(d not in pool for d, pool in zip(drafts, candidates, strict=True))):
            raise ValueError(f"Candidate or prefix alignment failed after output ID {prefix}")
        if accepted < len(drafts) and committed > accepted:
            if drafts[accepted] == continuation[accepted]:
                raise ValueError("First rejected draft matches the target")
            pool = candidates[accepted]
            correction = continuation[accepted]
            rejection_positions[accepted + 1] += 1
            if correction in pool:
                recovered_ranks[pool.index(correction) + 1] += 1

        # Actual committed continuations supply later labels. Verifier logits
        # after the first wrong draft would describe a different prefix.
        ranks = []
        for j, (token, pool) in enumerate(zip(continuation, candidates), 1):
            rank = pool.index(token) + 1 if token in pool else None
            ranks.append(rank)
            position_tested[j] += 1
            position_recalled[j] += rank is not None
        reaches = next((i for i, rank in enumerate(ranks) if rank is None), len(ranks))
        unary_reaches = next((i for i, rank in enumerate(ranks) if rank != 1), len(ranks))
        oracle = min(reaches + 1, len(continuation))
        if oracle < committed:
            raise ValueError("Candidate oracle contradicts the verified live prefix")
        oracle_committed += oracle
        independent_top1 += min(unary_reaches + 1, len(continuation))
        fixed_pool_oracle.append(oracle)
        prefix += committed
    if prefix != len(ids):
        raise ValueError("The audit does not account for every committed token")
    return {
        "case": run["case"], "passes": len(passes), "committed": prefix - 1,
        "live_ids_per_pass": (prefix - 1) / len(passes),
        "rejections": sum(rejection_positions.values()),
        "rejection_position_histogram": dict(rejection_positions),
        "recovered_at_first_rejection": sum(recovered_ranks.values()),
        "recoverable_rank_histogram": dict(recovered_ranks),
        "position_tested": dict(position_tested), "position_recalled": dict(position_recalled),
        "oracle_ids_per_pass": oracle_committed / len(passes),
        "top1_ids_per_pass": independent_top1 / len(passes),
        "oracle_committed_by_pass": fixed_pool_oracle,
        "ordinary_ms_per_pass": ordinary_ms,
        "oracle_rate_at_ordinary_ms": (1000 * oracle_committed / len(passes) / ordinary_ms
                                       if ordinary_ms is not None else None),
    }


def compare_ids(run, reference):
    return (run["token_ids"] == reference["token_ids"] and run["finish_reason"] == reference["finish_reason"])


def analyze(report, frozen):
    rows = report["rows"]
    controls = defaultdict(list)
    for r in rows:
        if r["scenario"] in {"plain-4", "paired-4", "paired-8"}:
            controls[r["scenario"]].append(r)
    ordinary = {
        tag: {"rates": [r["timings"]["predicted_per_second"] for r in rs],
              "median_rate": median(r["timings"]["predicted_per_second"] for r in rs),
              "median_wall_s": median(r["http_wall_s"] for r in rs),
              "median_first_text_s": median(r["http_first_text_s"] for r in rs),
              "median_cycle_ms": median(r["timings"]["predicted_ms"] / len(r["passes"]) for r in rs)}
        for tag, rs in controls.items()
    }
    ordinary["paired_4_exact"] = [
        compare_ids(a, b) for a, b in zip(controls["plain-4"], controls["paired-4"], strict=True)
    ]
    frozen_by_case = defaultdict(list)
    for r in frozen["rows"]:
        if r["scenario"] == "paired-1":
            frozen_by_case[r["case"]].append(r)
    profiles, audits = [], []
    for r in rows:
        if r["scenario"].startswith("profile-"):
            profile = summarize_profile(r)
            profile["exact_to_ordinary"] = all(
                compare_ids(r, x) for x in controls[f"paired-{r['verify_width']}"]
            )
            profiles.append(profile)
        elif r["scenario"] == "audit-8":
            reference = controls["paired-8"] if r["case"] == "studio" else frozen_by_case[r["case"]]
            exact = bool(reference) and all(compare_ids(r, ref) for ref in reference)
            ordinary_ms = (median(ref["timings"]["predicted_ms"] / len(ref["passes"]) for ref in reference)
                           if exact else None)
            audit = summarize_recall(r, ordinary_ms)
            audit["exact_to_ordinary"] = exact
            audits.append(audit)
    return {"ordinary": ordinary, "profiles": profiles, "audits": audits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--frozen", type=Path, required=True, help="Frozen eight-position fused benchmark")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.report.read_text())
    frozen = json.loads(args.frozen.read_text())
    if frozen["revision"] != data["revision"]:
        raise ValueError("The reference and audited target revisions differ")
    args.out.write_text(json.dumps(analyze(data, frozen), indent=2) + "\n")


if __name__ == "__main__":
    main()
