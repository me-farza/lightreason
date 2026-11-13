"""Correctness gates and paired summaries for the fixed-width BF16 transfer trial."""

from collections import defaultdict
from statistics import median


def frozen_controls(archive, caps, revision):
    if archive["revision"] != revision or archive["dtype"] != "bfloat16":
        raise ValueError("Frozen reference must use the pinned BF16 checkpoint.")
    controls = {}
    for row in archive["rows"]:
        case = row["case"]
        if case not in caps or row["scenario"] != "paired-1":
            continue
        if (row["cap"] != caps[case] or row["verify_width"] != 8
                or row["pair_threshold"] != 1 or row["cached_prompt_tokens"] != 0
                or not row["passes"] or any(p[0] != 8 for p in row["passes"])):
            raise ValueError(f"Archived control settings differ for {case}.")
        if not row["token_ids"] or row["finish_reason"] not in {"stop", "length"}:
            raise ValueError(f"No complete archived token record for {case}.")
        reference = controls.setdefault(case, row)
        if not compare_runs(reference, row, reference)["exact"]:
            raise ValueError(f"Archived controls disagree for {case}.")
    if set(controls) != set(caps):
        raise ValueError(f"Missing archived controls for {sorted(set(caps) - set(controls))}.")
    return controls


def first_different_id(left, right):
    for i, (a, b) in enumerate(zip(left, right), 1):
        if a != b:
            return i
    return min(len(left), len(right)) + 1 if len(left) != len(right) else None


def compare_runs(control, candidate, reference):
    def same_passes(a, b):
        return [p[:2] for p in a["passes"]] == [p[:2] for p in b["passes"]]

    fields = {
        "first_different_id": first_different_id(control["token_ids"], candidate["token_ids"]),
        "control_first_different_id_vs_frozen": first_different_id(control["token_ids"], reference["token_ids"]),
        "same_stop": control["finish_reason"] == candidate["finish_reason"],
        "control_same_stop_as_frozen": control["finish_reason"] == reference["finish_reason"],
        "same_commits_per_pass": same_passes(control, candidate),
        "control_same_commits_as_frozen": same_passes(control, reference),
    }
    fields["exact"] = (
        fields["first_different_id"] is None and fields["control_first_different_id_vs_frozen"] is None
        and fields["same_stop"] and fields["control_same_stop_as_frozen"]
        and fields["same_commits_per_pass"] and fields["control_same_commits_as_frozen"]
    )
    return fields


def summarize_transport(pairs, control="raw", candidate="packed"):
    groups = defaultdict(list)
    for row in pairs:
        if row["diagnostics"] == "off":
            groups[(row["case"], row["stream"])].append(row)
    if not groups:
        raise ValueError("Expected ordinary, unprofiled paired measurements.")
    summaries = []
    for (case, stream), rows in groups.items():
        metrics = ("decode_tok_s", "request_s", "first_text_s", "prefill_ms", "pass_ms", "committed_per_pass")
        runs = {
            arm: {
                k: median(row[arm][k] for row in rows) if rows[0][arm][k] is not None else None
                for k in metrics
            }
            for arm in (control, candidate)
        }
        summaries.append({
            "case": case, "stream": stream, "pairs": len(rows),
            "exact": all(row["exact"] for row in rows),
            "paired_decode_speedup": median(row[candidate]["decode_tok_s"] / row[control]["decode_tok_s"] for row in rows),
            "paired_request_speedup": median(row[control]["request_s"] / row[candidate]["request_s"] for row in rows),
            "candidate_min_decode_tok_s": min(row[candidate]["decode_tok_s"] for row in rows),
            **runs,
        })
    exact = all(row["exact"] for row in pairs)
    return {
        "exact": exact, "ordinary": summaries, "control": control, "candidate": candidate,
        "exact_pairs": sum(row["exact"] for row in pairs), "pairs": len(pairs),
        "at_least_50_on_every_measured_run": exact and all(r["candidate_min_decode_tok_s"] >= 50 for r in summaries),
    }
