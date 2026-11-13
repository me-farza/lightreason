"""Compare a lossless primary cache with the packed-replica runner."""

import json
from statistics import median

from edge35b.benchmark_draft_budget import summarize
from edge35b.expert_transport_eval import compare_runs, summarize_transport
from edge35b.width_autotune import PACKED_PRIMARY_TRANSPORT, SHARED_CPU_TRANSPORT


CONTROL = "gpu-packed"
CANDIDATE = "packed-primary"


def benchmark(result, cases, repeats, frozen, packed_layers, cpu_policy, primary_manifest, resident_manifest,
              set_width, measure, generate, save):
    trial = {"manifest": json.loads(primary_manifest.read_text()), "pairs": [], "warmup": [],
             "resident": json.loads(resident_manifest.read_text())}
    result["packed_primary"] = trial
    packed_layers = frozenset(packed_layers)
    fixed = trial["manifest"]["layout"] == "fixed-prefix"

    def options(arm):
        return {"transport": SHARED_CPU_TRANSPORT if arm == CONTROL else PACKED_PRIMARY_TRANSPORT,
                "cpu_read_policy": cpu_policy, "three_route_reuse": True, "gpu_resident_layout": "packed"}

    for diagnostics in ("off", "profile"):
        for arm in (CONTROL, CANDIDATE):
            set_width(8, 1, diagnostics, **options(arm))
            run = generate(cases[0], 8, False)
            trial["warmup"].append({"arm": arm, "mode": diagnostics, "http_wall_s": run["http_wall_s"]})
    save()

    def pair(case, stream, repeat, diagnostics, reverse):
        order = (CANDIDATE, CONTROL) if reverse else (CONTROL, CANDIDATE)
        runs = {arm: measure(case, stream, f"{arm}-{diagnostics}", 8, 1, repeat, diagnostics, **options(arm))
                for arm in order}
        control, candidate = runs[CONTROL], runs[CANDIDATE]
        checks = compare_runs(control, candidate, frozen[case])
        checks["same_output"] = checks["first_different_id"] is None and checks["same_stop"]
        row = {"case": case, "stream": stream, "repeat": repeat, "diagnostics": diagnostics, "order": order,
               **checks, **{arm: summarize(run, packed_layers) for arm, run in runs.items()}}
        if diagnostics == "profile":
            steps = candidate["diagnostics"]["passes"]
            if fixed:
                row["same_logical_fetches"] = [p["expert_fetch_ids"] for p in steps] == [
                    p["expert_fetch_ids"] for p in control["diagnostics"]["passes"]]
                row["exact"] &= row["same_logical_fetches"]
                row[CANDIDATE]["additional_refills"] = sum(sum(p["primary_additional_refills"]) for p in steps)
                row[CANDIDATE]["extra_replica_hits"] = sum(sum(p["extra_replica_hits"]) for p in steps)
                row[CANDIDATE]["pcie_upload_bytes_per_commit"] = (trial["manifest"]["replica_upload_bytes"]
                                                                  / row[CANDIDATE]["committed"])
            else:
                row[CANDIDATE]["packed_tail_hits"] = sum(sum(p.get("packed_tail_hits", ())) for p in steps)
        trial["pairs"].append(row)
        save()
        print(case, diagnostics, "exact", row["exact"], "first differing ID", row["first_different_id"], flush=True)
        if (checks["control_first_different_id_vs_frozen"] is not None or not checks["control_same_stop_as_frozen"]
                or not checks["control_same_commits_as_frozen"]):
            raise RuntimeError("The packed-replica control did not reproduce the frozen output")

    # Complete the comparisons even after a token mismatch so that the speed,
    # traffic, and acceptance effects are available for diagnosis.
    for repeat in range(repeats):
        for case in cases if repeat % 2 == 0 else cases[::-1]:
            for stream in (True, False) if repeat % 2 == 0 else (False, True):
                pair(case, stream, repeat, "off", (cases.index(case) + repeat + int(not stream)) % 2 != 0)
    for i, case in enumerate(cases):
        pair(case, True, 0, "profile", i % 2 != 0)
    trial["summary"] = summarize_transport(trial["pairs"], CONTROL, CANDIDATE)
    trial["summary"]["same_output_pairs"] = sum(p["same_output"] for p in trial["pairs"])
    ordinary = [p for p in trial["pairs"] if p["diagnostics"] == "off"]
    trial["median_decode_speedup"] = median(p[CANDIDATE]["decode_tok_s"] / p[CONTROL]["decode_tok_s"] for p in ordinary)
    save()
