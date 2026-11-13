"""Compare DFlash2 prefix selection and verifier width against the frozen runner."""

import json
from statistics import median

from edge35b.benchmark_draft_budget import summarize
from edge35b.expert_transport_eval import compare_runs, summarize_transport
from edge35b.width_autotune import SHARED_CPU_TRANSPORT


CONTROL = "gpu-packed"
ARMS = (CONTROL, "prefix-eight", "stock-adaptive", "prefix-adaptive")


def benchmark(result, cases, repeats, frozen, packed_layers, cpu_policy, policy, manifest,
              set_width, measure, generate, save, *, teacher=False):
    resident = json.loads(manifest.read_text())
    if resident["arena_bytes"] != 880 * 2**20 or len(resident["experts"]["packed"]) != 187:
        raise RuntimeError("The frozen packed replica budget did not reproduce")
    trial = {"manifest": resident, "policy": policy, "warmup": [], "pairs": []}
    result["dflash_prefix"] = trial
    packed_layers = frozenset(packed_layers)
    config = {"transport": SHARED_CPU_TRANSPORT, "cpu_read_policy": cpu_policy,
              "three_route_reuse": True, "gpu_resident_layout": "packed"}

    def settings(arm):
        adaptive = arm.endswith("-adaptive")
        return "draft-budget" if adaptive else 8, {
            **config, "draft_budget": policy if adaptive else None,
            "draft_prefix_selection": arm.startswith("prefix-"),
        }

    for mode in ("off", "profile"):
        for arm in ARMS:
            width, options = settings(arm)
            set_width(width, 1, mode, **options)
            run = generate(cases[0], 8, False)
            trial["warmup"].append({"arm": arm, "mode": mode, "seconds": run["http_wall_s"]})
    save()

    def compare(case, stream, repeat, mode, order):
        runs = {}
        for arm in order:
            width, options = settings(arm)
            runs[arm] = measure(case, stream, f"{arm}-{mode}", width, 1, repeat, mode, **options)
        control = runs[CONTROL]
        for arm, run in runs.items():
            if arm == CONTROL:
                continue
            check = compare_runs(control, run, frozen[case])
            control_exact = (check["control_first_different_id_vs_frozen"] is None
                             and check["control_same_stop_as_frozen"] and check["control_same_commits_as_frozen"])
            if not control_exact:
                raise RuntimeError("The packed control diverged from the frozen token IDs, stop, or commits")
            check["exact"] = check["first_different_id"] is None and check["same_stop"]
            row = {"case": case, "stream": stream, "repeat": repeat, "diagnostics": mode, "order": order,
                   "candidate": arm, **check, CONTROL: summarize(control, packed_layers),
                   arm: summarize(run, packed_layers)}
            trial["pairs"].append(row)
            print(case, arm, mode, "exact", row["exact"], "first differing ID", row["first_different_id"],
                  "widths", row[arm]["width_counts"], flush=True)
        save()

    for repeat in range(repeats):
        for case in cases if repeat % 2 == 0 else cases[::-1]:
            for stream in (True, False) if repeat % 2 == 0 else (False, True):
                start = (cases.index(case) + int(not stream)) % len(ARMS)
                order = ARMS[start:] + ARMS[:start]
                compare(case, stream, repeat, "off", order if repeat % 2 == 0 else order[::-1])
    for i, case in enumerate(cases):
        compare(case, True, 0, "profile", ARMS if i % 2 == 0 else ARMS[::-1])
    trial["summary"] = {}
    for arm in ARMS[1:]:
        pairs = [p for p in trial["pairs"] if p["candidate"] == arm]
        summary = summarize_transport(pairs, CONTROL, arm)
        summary["median_paired_decode_ratio"] = median(p[arm]["decode_tok_s"] / p[CONTROL]["decode_tok_s"]
                                                       for p in pairs if p["diagnostics"] == "off")
        summary["at_least_100_on_every_measured_run"] = summary["exact"] and all(
            row["candidate_min_decode_tok_s"] >= 100 for row in summary["ordinary"])
        trial["summary"][arm] = summary
    save()

    if teacher:
        trial["teacher_diagnostic"] = []
        for case in cases:
            run = measure(case, True, "teacher-profile", 8, 1, diagnostics="profile",
                          draft_teacher=frozen[case]["token_ids"], **config)
            check = compare_runs(frozen[case], run, frozen[case])
            trial["teacher_diagnostic"].append({
                "case": case, "first_different_id": check["first_different_id"],
                "same_stop": check["same_stop"], **summarize(run, packed_layers),
            })
            save()
