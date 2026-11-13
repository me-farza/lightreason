"""Evolve expert copy geometry using complete, exact-output GPU requests."""

from statistics import median

from edge35b.evolve_policy import MIN_DECODE_RATIO
from edge35b.expert_transport_eval import compare_runs, summarize_transport
from edge35b.width_autotune import GATHER_GEOMETRIES


CONTROL = "hot"
SEEDS = ("hot-tiles-4-4096", "hot-tiles-8-2048")


def _metrics(run):
    passes = run["passes"]
    committed = sum(count for _, count, _ in passes)
    return {
        "decode_tok_s": 1000 * committed / run["timings"]["predicted_ms"],
        "request_s": run["http_wall_s"],
        "first_text_s": run["http_first_text_s"],
        "prefill_ms": run["timings"]["prompt_ms"],
        "pass_ms": sum(ms for _, _, ms in passes) / len(passes),
        "committed_per_pass": committed / len(passes),
    }


def _evaluate(measure, candidate, cases, frozen, repeats, trial):
    pairs = []
    for repeat in range(repeats):
        for index, case in enumerate(cases if repeat % 2 == 0 else reversed(cases)):
            for stream in (True, False) if repeat % 2 == 0 else (False, True):
                order = (CONTROL, candidate)
                if (repeat + index + int(not stream)) % 2:
                    order = order[::-1]
                runs = {mode: measure(case, stream, mode, trial * repeats + repeat) for mode in order}
                control, challenger = runs[CONTROL], runs[candidate]
                exact = compare_runs(control, challenger, frozen[case])
                if exact["control_first_different_id_vs_frozen"] is not None or not (
                    exact["control_same_stop_as_frozen"] and exact["control_same_commits_as_frozen"]
                ):
                    raise RuntimeError(f"The live tuned control diverged from the frozen run for {case}.")
                pairs.append({"case": case, "stream": stream, "repeat": repeat, "diagnostics": "off",
                              "order": order, **exact, CONTROL: _metrics(control), candidate: _metrics(challenger)})
    summary = summarize_transport(pairs, CONTROL, candidate)
    ratio = median(r["paired_decode_speedup"] for r in summary["ordinary"])
    request_by_case = {case: median(p[CONTROL]["request_s"] / p[candidate]["request_s"]
                                    for p in pairs if p["case"] == case) for case in cases}
    return {"name": candidate, "geometry": GATHER_GEOMETRIES[candidate], "pairs": pairs,
            "summary": summary, "median_decode_speedup": ratio, "request_speedup_by_case": request_by_case,
            "passed": summary["exact"] and ratio >= MIN_DECODE_RATIO and min(request_by_case.values()) >= 1}


def _rank(result):
    return result["median_decode_speedup"], median(result["request_speedup_by_case"].values())


def _next_candidate(tested):
    visited = {r["name"] for r in tested}
    remaining = [name for name in GATHER_GEOMETRIES if name not in visited]
    for name in SEEDS:
        if name in remaining:
            return name
    parents = sorted((r for r in tested if r["summary"]["exact"]), key=_rank, reverse=True)
    for parent in parents:
        stripes, tile = parent["geometry"]
        # Change one launch dimension of the fastest exact parent.
        for name in remaining:
            new_stripes, new_tile = GATHER_GEOMETRIES[name]
            if (stripes == new_stripes) != (tile == new_tile):
                return name
    return remaining[0]


def evolve(measure, frozen, development, held_out, trials, repeats, *, checkpoint):
    """Alternate live control and candidate runs, then lock one winner for validation."""
    report = {"control": CONTROL, "development_cases": development, "held_out_cases": held_out,
              "development": [], "selected": None, "held_out": None, "promoted": None}
    for trial in range(trials):
        name = _next_candidate(report["development"])
        result = _evaluate(measure, name, development, frozen, 1, trial)
        report["development"].append(result)
        checkpoint(report)
        print(f"Gather trial {name}: {result['median_decode_speedup']:.4f}x decode, "
              f"{result['summary']['exact_pairs']}/{result['summary']['pairs']} exact pairs", flush=True)

    eligible = [result for result in report["development"] if result["passed"]]
    winner = max(eligible, key=_rank, default=None)
    if winner is not None:
        report["selected"] = winner["name"]
        checkpoint(report)
        result = _evaluate(measure, winner["name"], held_out, frozen, repeats, trials)
        report["held_out"] = result
        if result["passed"]:
            report["promoted"] = winner["name"]
    checkpoint(report)
    return report
