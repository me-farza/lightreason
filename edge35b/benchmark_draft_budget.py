"""Train the prefix budget on separate prompts, then alternate live exact-output pairs."""

from collections import Counter

from edge35b.analyze_pass_diagnostics import EXPERT_BYTES, profile_pass
from edge35b.draft_budget import WIDTHS, fit, memory_metrics
from edge35b.expert_transport_eval import compare_runs, summarize_transport
from edge35b.selector_calibration import TOKEN_LIMIT, prompts
from edge35b.width_autotune import SHARED_CPU_TRANSPORT


CONTROL = "fixed-eight"
CANDIDATE = "draft-budget"
THREE_ROUTES = "three-route-reads"


def calibration_suite():
    return {f"budget-{name}": {**prompt, "cap": TOKEN_LIMIT}
            for name, prompt in list(prompts().items())[:8]}


def summarize(run, packed_layers):
    passes = run["passes"]
    committed = sum(n for _, n, _ in passes)
    decode_s = run["timings"]["predicted_ms"] / 1000
    summary = {"decode_tok_s": committed / decode_s, "decode_s": decode_s, "committed": committed,
               "verify_passes": len(passes), "committed_per_pass": committed / len(passes),
               "width_counts": dict(Counter(w for w, _, _ in passes)),
               "first_text_s": run["http_first_text_s"], "request_s": run["http_wall_s"],
               "prefill_ms": run["timings"]["prompt_ms"], "pass_ms": sum(ms for _, _, ms in passes) / len(passes)}
    if "diagnostics" in run:
        records = run["diagnostics"]["passes"]
        profiles = [profile_pass(step, timing[2]) for step, timing in zip(records, passes, strict=True)]
        pcie, cpu = map(sum, zip(*(memory_metrics(step, packed_layers) for step in records), strict=True))
        summary.update(pcie_bytes_per_commit=pcie / committed, cpu_min_bytes_per_commit=cpu / committed,
                       pcie_bytes_per_pass=pcie / len(passes), cpu_min_bytes_per_pass=cpu / len(passes))
        for key in ("pcie_gather_ms", "cpu_service_ms", "cpu_wait_ms", "gpu_experts_ms", "other_target_ms", "target_ms"):
            summary[key] = sum(p[key] for p in profiles) / len(passes)
        summary["outside_target_ms"] = summary["pass_ms"] - summary["target_ms"]
        if "cpu_weight_read_groups" in records[0]:
            groups = logical_bytes = 0
            for step in records:
                layers = zip(step["cpu_weight_read_groups"], step["gpu_layer_stamps"], strict=True)
                for layer, (count, stamps) in enumerate(layers):
                    if not stamps[9]:
                        continue
                    if not stamps[8] - stamps[7] <= count <= stamps[9]:
                        raise ValueError("CPU weight-read groups disagree with the routed expert counts")
                    groups += count
                    logical_bytes += count * (3 * EXPERT_BYTES // 4 if layer in packed_layers else EXPERT_BYTES)
            summary.update(cpu_read_groups_per_pass=groups / len(passes),
                           cpu_group_min_bytes_per_commit=logical_bytes / committed)
    return summary


def benchmark(result, calibration, cases, repeats, frozen, packed_layers, cpu_policy, set_width, measure, generate, save,
              *, cpu_three_routes=False):
    packed_layers = frozenset(packed_layers)
    config = {"transport": SHARED_CPU_TRANSPORT, "cpu_read_policy": cpu_policy}
    trial = {"widths": WIDTHS, "packed_layers": sorted(packed_layers), "calibration_cases": calibration,
             "warmup": [], "pairs": []}
    result["draft_budget"] = trial

    def warm(width, mode, **kwargs):
        set_width(width, 1, mode, **config, **kwargs)
        run = generate(cases[0], 4, False)
        trial["warmup"].append({"width": width, "mode": mode, "http_wall_s": run["http_wall_s"]})

    warm(8, "off")
    for width in WIDTHS:
        warm(width, "profile", draft_confidence=True)
        warm(width, "off")
    policy = None

    def pair(case, stream, repeat, profile, reverse, *, arm_name=CANDIDATE, pairs=trial["pairs"]):
        diagnostics = "profile" if profile else "off"
        order = (arm_name, CONTROL) if reverse else (CONTROL, arm_name)
        runs = {}
        for arm in order:
            adaptive = arm == CANDIDATE
            runs[arm] = measure(case, stream, f"{arm}-{diagnostics}", CANDIDATE if adaptive else 8, 1,
                                repeat, diagnostics, draft_budget=policy if adaptive else None,
                                three_route_reuse=arm == THREE_ROUTES, **config)
        control, candidate = runs[CONTROL], runs[arm_name]
        checks = compare_runs(control, candidate, frozen[case])
        control_exact = (checks["control_first_different_id_vs_frozen"] is None
                         and checks["control_same_stop_as_frozen"] and checks["control_same_commits_as_frozen"])
        if arm_name == CANDIDATE:
            # The adaptive graph can commit the same IDs on a different schedule.
            checks["exact"] = control_exact and checks["first_different_id"] is None and checks["same_stop"]
        elif profile:
            a, b = (run["diagnostics"]["passes"] for run in (control, candidate))
            checks["same_fetch_plan"] = [p["expert_fetch_ids"] for p in a] == [p["expert_fetch_ids"] for p in b]
            checks["exact"] &= checks["same_fetch_plan"]
        row = {"case": case, "stream": stream, "repeat": repeat, "diagnostics": diagnostics,
               "order": order, **checks, **{arm: summarize(run, packed_layers) for arm, run in runs.items()}}
        pairs.append(row)
        save()
        print(case, arm_name, diagnostics, "exact", row["exact"], "first differing ID", row["first_different_id"],
              "widths", row[arm_name]["width_counts"], flush=True)
        if not control_exact:
            raise RuntimeError("The fixed eight-position control diverged from its archived IDs or commits")

    if cpu_three_routes:
        cpu_trial = result["cpu_three_routes"] = {"pairs": []}
        warm(8, "off", three_route_reuse=True)
        selected = [cases[0]] + [case for case in cases[1:] if case == "coding"]
        for i, case in enumerate(selected):
            for repeat in range(repeats):
                pair(case, True, repeat, False, (i + repeat) % 2 != 0,
                     arm_name=THREE_ROUTES, pairs=cpu_trial["pairs"])
            pair(case, True, 0, True, i % 2 != 0, arm_name=THREE_ROUTES, pairs=cpu_trial["pairs"])
        cpu_trial["summary"] = summarize_transport(cpu_trial["pairs"], CONTROL, THREE_ROUTES)
        save()

    training = []
    for i, case in enumerate(calibration):
        offset = i % len(WIDTHS)
        for width in WIDTHS[offset:] + WIDTHS[:offset]:
            training.append(measure(case, False, f"budget-calibration-{width}", width, 1,
                                    diagnostics="profile", draft_confidence=True, **config))
    policy = fit(training, packed_layers)
    trial["policy"] = policy
    save()
    print("Draft budget fitted costs:", policy["costs"], flush=True)
    warm(CANDIDATE, "off", draft_budget=policy)
    warm(CANDIDATE, "profile", draft_budget=policy)

    for repeat in range(repeats):
        for case in cases if repeat % 2 == 0 else cases[::-1]:
            for stream in (True, False) if repeat % 2 == 0 else (False, True):
                pair(case, stream, repeat, False, (cases.index(case) + repeat + int(not stream)) % 2 != 0)
    for i, case in enumerate(cases):
        pair(case, True, 0, True, i % 2 != 0)
    trial["summary"] = summarize_transport(trial["pairs"], CONTROL, CANDIDATE)
    save()
