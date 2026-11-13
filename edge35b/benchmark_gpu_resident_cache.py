"""Compare frozen, raw-resident, and packed-resident BF16 generation."""

import json
from statistics import median

from edge35b.benchmark_draft_budget import summarize
from edge35b.expert_transport_eval import compare_runs, summarize_transport
from edge35b.width_autotune import SHARED_CPU_CONTROL_TRANSPORT, SHARED_CPU_TRANSPORT


CONTROL, RAW, PACKED = "fixed-eight", "gpu-raw", "gpu-packed"
ARMS = (CONTROL, RAW, PACKED)


def benchmark(result, cases, repeats, frozen, packed_layers, cpu_policy, cpu_three_routes, manifest_path,
              set_width, measure, generate, save):
    manifest = json.loads(manifest_path.read_text())
    replica_keys = {mode: {tuple(key) for key in rows} for mode, rows in manifest["experts"].items()}
    trial = {"manifest": manifest, "cpu_three_routes": cpu_three_routes, "warmup": [], "pairs": []}
    result["gpu_resident_cache"] = trial
    packed_layers = frozenset(packed_layers)
    active_layout = None

    def options(arm):
        layout = None if arm == CONTROL else arm.removeprefix("gpu-")
        return {"transport": SHARED_CPU_CONTROL_TRANSPORT if arm == CONTROL else SHARED_CPU_TRANSPORT,
                "cpu_read_policy": cpu_policy, "three_route_reuse": cpu_three_routes,
                "gpu_resident_layout": layout}

    def warm(arm, diagnostics, cap):
        nonlocal active_layout
        settings = options(arm)
        set_width(8, 1, diagnostics, **settings)
        run = generate(cases[0], cap, False)
        trial["warmup"].append({"arm": arm, "mode": diagnostics, "cap": cap, "http_wall_s": run["http_wall_s"]})
        if arm != CONTROL:
            active_layout = settings["gpu_resident_layout"]

    for diagnostics in ("off", "profile"):
        for arm in ARMS:
            warm(arm, diagnostics, 4)

    def pair(case, stream, repeat, diagnostics, order):
        runs = {}
        for arm in order:
            settings = options(arm)
            if arm != CONTROL and active_layout != settings["gpu_resident_layout"]:
                # The only state retained from this request is the weight
                # arena. The ordinary request resets the primary cache.
                warm(arm, "off", 1)
            runs[arm] = measure(case, stream, f"{arm}-{diagnostics}", 8, 1, repeat, diagnostics, **settings)
        control = runs[CONTROL]
        for arm in (RAW, PACKED):
            run = runs[arm]
            comparison = compare_runs(control, run, frozen[case])
            if diagnostics == "profile":
                a, b = (r["diagnostics"]["passes"] for r in (control, run))
                comparison["same_fetch_plan"] = [p["expert_fetch_ids"] for p in a] == [p["expert_fetch_ids"] for p in b]
                comparison["exact"] &= comparison["same_fetch_plan"]
                keys = replica_keys[arm.removeprefix("gpu-")]
                comparison["resident_fetches"] = sum((layer, e) in keys for step in b
                                                      for layer, ids in enumerate(step["expert_fetch_ids"]) for e in ids)
            row = {"case": case, "stream": stream, "repeat": repeat, "diagnostics": diagnostics,
                   "candidate": arm, "order": order, **comparison,
                   CONTROL: summarize(control, packed_layers), arm: summarize(run, packed_layers)}
            trial["pairs"].append(row)
            print(case, diagnostics, arm, "exact", row["exact"], "first differing ID",
                  row["first_different_id"], flush=True)
        save()

    for repeat in range(repeats):
        for case in cases if repeat % 2 == 0 else cases[::-1]:
            for stream in (True, False) if repeat % 2 == 0 else (False, True):
                offset = (cases.index(case) + int(not stream)) % len(ARMS)
                order = ARMS[offset:] + ARMS[:offset]
                pair(case, stream, repeat, "off", order if repeat % 2 == 0 else order[::-1])
    for i, case in enumerate(cases):
        pair(case, True, 0, "profile", ARMS if i % 2 == 0 else ARMS[::-1])
    trial["summary"] = {arm: summarize_transport([p for p in trial["pairs"] if p["candidate"] == arm], CONTROL, arm)
                        for arm in (RAW, PACKED)}
    ordinary = [p for p in trial["pairs"] if p["diagnostics"] == "off"]
    raw = {(p["case"], p["stream"], p["repeat"]): p[RAW]["decode_tok_s"] for p in ordinary if p["candidate"] == RAW}
    trial["median_packed_over_equal_budget_raw"] = median(
        p[PACKED]["decode_tok_s"] / raw[p["case"], p["stream"], p["repeat"]]
        for p in ordinary if p["candidate"] == PACKED
    )
    trial["layout_loads"] = [json.loads(line) for line in manifest_path.with_suffix(".loads.jsonl").read_text().splitlines()]
    save()
