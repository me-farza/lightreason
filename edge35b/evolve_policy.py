"""Search measured Qwen3.6 BF16 verifier policies with an exact-output gate."""

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import json
from math import isfinite
from pathlib import Path
from random import Random
from statistics import median

from edge35b.download import REVISION
from edge35b.width_autotune import STATES, WIDTHS, state


PAIR_THRESHOLDS = (0, 1, 3)
MIN_DECODE_RATIO = 1.005


@dataclass(frozen=True)
class Policy:
    widths: tuple[int, ...]
    min_pairs: int

    def __post_init__(self):
        if (len(self.widths) != len(STATES)
                or any(type(w) is not int or w not in WIDTHS for w in self.widths)
                or type(self.min_pairs) is not int or self.min_pairs not in PAIR_THRESHOLDS):
            raise ValueError("Use real verifier widths 4, 6, or 8 and CPU pair thresholds 0, 1, or 3.")

    @property
    def width(self):
        return self.widths[0] if len(set(self.widths)) == 1 else "learned"

    @property
    def table(self):
        return dict(zip(STATES, self.widths, strict=True))

    def as_dict(self):
        return {"width_by_state": self.table, "min_pairs": self.min_pairs}

    def check_trace(self, passes):
        table = self.table
        previous = None
        for entry in passes:
            if entry[0] != table[state(previous)]:
                raise RuntimeError("The actual verifier width differs from the candidate policy.")
            previous = entry


CONTROL = Policy((8,) * len(STATES), 0)


def load_proposals(path):
    rows = json.loads(path.read_text()) if path else []
    proposals = []
    for row in rows:
        table = row["width_by_state"]
        if set(table) != set(STATES):
            raise ValueError(f"Width policy must define {', '.join(STATES)}.")
        proposals.append(Policy(tuple(table[s] for s in STATES), row["min_pairs"]))
    return proposals


def mutations(policy):
    for threshold in PAIR_THRESHOLDS:
        if threshold != policy.min_pairs:
            yield replace(policy, min_pairs=threshold)
    for index, current in enumerate(policy.widths):
        for width in WIDTHS:
            if width != current:
                yield replace(policy, widths=policy.widths[:index] + (width,) + policy.widths[index + 1:])


def _index(rows, cases):
    indexed = {}
    for row in rows:
        if row["case"] not in cases:
            continue
        key = (row["case"], row["stream"], row["repeat"])
        if key in indexed:
            raise ValueError(f"Duplicate measurement for {key}.")
        indexed[key] = row
    return indexed


def _check_request(before, after):
    if before["cap"] != after["cap"] or before["usage"]["prompt_tokens"] != after["usage"]["prompt_tokens"]:
        raise ValueError("Paired measurements must have the same input length and token limit.")
    for row in (before, after):
        ids = row["token_ids"]
        if (row["cached_prompt_tokens"] != 0 or not ids
                or any(type(token) is not int or token < 0 for token in ids)
                or len(ids) != row["usage"]["completion_tokens"] or len(ids) > row["cap"]
                or row["finish_reason"] not in ("length", "stop")):
            raise ValueError("Each request needs raw IDs, a valid stop reason, and zero cached input tokens.")
        if row["finish_reason"] == "length" and len(ids) != row["cap"]:
            raise ValueError("The recorded length stop differs from its token limit.")
        if any(not isfinite(value) or value <= 0 for value in (
                row["timings"]["predicted_per_second"], row["http_wall_s"])):
            raise ValueError("The paired request is missing a valid timing.")


def compare_trials(control, candidate, cases, min_pairs):
    """Judge paired runs on the supplied cases only."""
    before, after = _index(control, cases), _index(candidate, cases)
    counts = Counter(case for case, _, _ in after.keys() & before.keys())
    missing = sorted(case for case in cases if counts[case] < min_pairs)
    if missing or before.keys() != after.keys():
        return {"passed": False, "missing_cases": missing,
                "unpaired_runs": len(before.keys() ^ after.keys())}

    per_case = {}
    mismatches = []
    unstable_control = []
    for case in cases:
        pairs = [key for key in sorted(before) if key[0] == case]
        decode, wall, speed = [], [], []
        control_outputs = set()
        for key in pairs:
            a, b = before[key], after[key]
            _check_request(a, b)
            a_ids, b_ids = a["token_ids"], b["token_ids"]
            control_outputs.add((tuple(a_ids), a["finish_reason"]))
            if a_ids != b_ids or a["finish_reason"] != b["finish_reason"]:
                first = next((i for i, (x, y) in enumerate(zip(a_ids, b_ids), 1) if x != y), None)
                if first is None and len(a_ids) != len(b_ids):
                    first = min(len(a_ids), len(b_ids)) + 1
                mismatches.append({"case": case, "stream": key[1], "repeat": key[2],
                                   "first_different_token": first,
                                   "stop_reasons": [a["finish_reason"], b["finish_reason"]]})
            a_rate, b_rate = (row["timings"]["predicted_per_second"] for row in (a, b))
            decode.append(b_rate / a_rate)
            speed.append(b_rate)
            wall.append(a["http_wall_s"] / b["http_wall_s"])
        if len(control_outputs) != 1:
            unstable_control.append(case)
        per_case[case] = {"pairs": len(pairs), "decode_tokens_per_s": median(speed),
                          "decode_ratio": median(decode), "http_speedup": median(wall)}

    decode_ratio = median(row["decode_ratio"] for row in per_case.values())
    http_speedup = median(row["http_speedup"] for row in per_case.values())
    return {"passed": (not mismatches and not unstable_control and decode_ratio >= MIN_DECODE_RATIO
                       and all(row["http_speedup"] >= 1 for row in per_case.values())),
            "pairs": len(before), "exact_pairs": len(before) - len(mismatches),
            "median_decode_ratio": decode_ratio, "median_http_speedup": http_speedup,
            "mismatches": mismatches, "unstable_control": unstable_control, "cases": per_case}


def _rank(report):
    return report["median_decode_ratio"], report["median_http_speedup"]


def replay_archive(archive, control, development, held_out, repeats):
    """Apply the same gate to prior measured requests without inventing results."""
    if (archive["revision"] != REVISION or archive["dtype"] != "bfloat16"
            or archive["prompt_tokenizer_roundtrip_match"] is not True):
        raise ValueError("The archive must use the pinned BF16 target and verified input IDs.")
    groups = {}
    for row in archive["rows"]:
        groups.setdefault(row["scenario"], []).append(row)
    control_rows = groups.pop(control)
    reports = {name: compare_trials(control_rows, rows, development, repeats)
               for name, rows in groups.items()}
    eligible = [name for name, result in reports.items() if result["passed"]]
    winner = max(eligible, key=lambda name: _rank(reports[name]), default=None)
    held = compare_trials(control_rows, groups[winner], held_out, repeats) if winner else None
    return {"source": "recorded measurements", "control": control,
            "development_cases": development, "held_out_cases": held_out, "development": reports,
            "selected": winner, "held_out": held, "promoted": winner if held and held["passed"] else None}


def evolve(measure, development, held_out, repeats, trials, proposals=(), *, progress=None):
    """Measure candidates on development cases, then test one frozen winner."""
    rng = Random(4090)
    seeds = [*proposals, replace(CONTROL, min_pairs=1), replace(CONTROL, min_pairs=3),
             Policy((4,) * len(STATES), 0), Policy((6,) * len(STATES), 0)]
    visited = {CONTROL}
    history = []
    eligible = []

    def evaluate(policy, name, cases, trial):
        control, candidate = [], []
        for index, case in enumerate(cases):
            for repeat in range(repeats):
                modes = (True, False) if repeat % 2 == 0 else (False, True)
                for stream in modes:
                    pair = [(name + "-control", CONTROL, control), (name, policy, candidate)]
                    if (index + repeat + int(not stream)) % 2:
                        pair.reverse()
                    for label, config, rows in pair:
                        rows.append(measure(case, stream, label, config, trial * repeats + repeat))
        return compare_trials(control, candidate, cases, 2 * repeats)

    for trial in range(trials):
        policy = next((p for p in seeds if p not in visited), None)
        if policy is None:
            parents = [p for _, p, _ in eligible[:3]] or [CONTROL]
            pool = dict.fromkeys(p for parent in parents for p in mutations(parent) if p not in visited)
            if not pool:
                break
            policy = rng.choice(list(pool))
        visited.add(policy)
        name = f"evolve-{trial + 1:02d}"
        result = evaluate(policy, name, development, trial)
        history.append({"name": name, "policy": policy.as_dict(), **result})
        if result["passed"]:
            eligible.append((name, policy, result))
            eligible.sort(key=lambda row: _rank(row[2]), reverse=True)
        if progress is not None:
            progress(history)

    selected = held = promoted = None
    if eligible:
        name, policy, _ = eligible[0]
        selected = {"name": name, "policy": policy.as_dict()}
        held = evaluate(policy, name + "-held-out", held_out, trials)
        if held["passed"]:
            promoted = selected
    return {"source": "live 4090 measurements", "control": CONTROL.as_dict(),
            "development_cases": development, "held_out_cases": held_out, "development": history,
            "selected": selected, "held_out": held, "promoted": promoted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--control", default="fixed-8")
    parser.add_argument("--development", default="studio")
    parser.add_argument("--held-out", default="known-math,coding,chat")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    dev = args.development.split(",")
    held = args.held_out.split(",")
    if not all(dev + held) or len(set(dev + held)) != len(dev + held) or args.repeats < 1:
        parser.error("Choose distinct development and held-out cases and at least one repeat.")
    report = replay_archive(json.loads(args.archive.read_text()), args.control, dev, held, args.repeats)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
