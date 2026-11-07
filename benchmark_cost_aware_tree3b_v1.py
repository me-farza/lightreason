"""Alternating unprofiled native-EOS tests against two frozen controls.

Original HF, the current N64/B15 compiled runner, and a separately labeled
candidate receive identical input IDs and output settings. Preparation is
synthetic and untimed. Streaming additionally includes actual queue delivery,
UTF-8 decoding, and JSON serialization on a consumer thread; it excludes HTTP
transport and browser painting. Answer grading must run in a separate process.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import queue
import statistics
import threading
import time
from types import SimpleNamespace

from profile_cost_aware_tree3b_v1 import (
    TARGET_SHA, CHECKPOINT_SHA, ROOT, require, sha, source_snapshot,
    verify_frozen_manifest, atomic_json, load_control, check_state, audit_runner,
    prepare_public_shape,
)

SUITE_SHA = "66102e5612e1e3cab38bee5053aa107446901f603425f639163fafd70dbfbce0"
SUPPORTED_BUDGETS = (16, 32, 48, 64, 96, 128)


def draft_coordination_from_trace(path):
    """Offline only: expose coordination hidden inside the broad draft range.

    With the frozen profiler's record_shapes=False, copy events cannot establish
    which exact tensor was copied. The full-prefix interpretation comes from
    source inspection; this report never calls every copy a full-history copy.
    ATen and CUDA runtime inclusive intervals overlap and must not be summed.
    """
    events = json.loads(Path(path).read_text())["traceEvents"]
    by_thread = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("name") == "drafting":
            by_thread[event.get("tid")].append((float(event["ts"]), float(event["ts"])+float(event.get("dur", 0))))
    starts = {}
    for thread, ranges in by_thread.items():
        ranges.sort()
        starts[thread] = [row[0] for row in ranges]
    operators = {"aten::equal", "aten::_to_copy", "aten::copy_", "aten::item", "aten::_local_scalar_dense",
                 "cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize",
                 "cudaMemcpyAsync", "cudaMemcpy", "cudaGraphLaunch"}
    totals = defaultdict(lambda: {"calls": 0, "inclusive_host_us": 0.})
    for event in events:
        if event.get("ph") != "X" or event.get("name") not in operators:
            continue
        thread = event.get("tid")
        if thread not in starts:
            continue
        timestamp = float(event.get("ts", -1))
        index = bisect_right(starts[thread], timestamp)-1
        if index < 0 or timestamp > by_thread[thread][index][1]:
            continue
        row = totals[event["name"]]
        row["calls"] += 1
        row["inclusive_host_us"] += float(event.get("dur", 0))
    count = sum(len(rows) for rows in by_thread.values())
    for value in totals.values():
        value["inclusive_host_us_per_draft_call"] = value["inclusive_host_us"]/count if count else None
    return {"draft_calls": count, "operations_inside_draft_ranges": dict(totals),
        "ATen_and_runtime_inclusive_intervals_overlap_do_not_add": True,
        "exact_copied_tensor_geometry_not_recorded_in_this_trace": True,
        "source_audit_distinguishes_full_prefix_audit_and_snapshot_from_incremental_KV_updates": True,
        "auxiliary_weight_quantization_headroom_not_equal_to_entire_draft_host_wall": True}


def load_cases(path, split="development", case_ids=None):
    data = json.loads(Path(path).read_text())
    require(data.get("schema") == "lightreason-cost-aware-native-eos-suite-v1", "Unknown fixed suite schema")
    require(sha(path) == SUITE_SHA, "Frozen development/held-out input suite changed")
    cases = data.get("cases", [])
    require(len(cases) == 12 and len({row["id"] for row in cases}) == 12, "Require all twelve frozen suite cases")
    require(all(row.get("split") in ("development", "heldout") and
                isinstance(row.get("prompt"), str) and row["prompt"].strip() and
                row.get("max_new_tokens") == 1024 for row in cases), "Suite prompt/output settings changed")
    wanted = set(case_ids or [])
    selected = [row for row in cases if (split == "all" or row["split"] == split)
                and (not wanted or row["id"] in wanted)]
    require(selected and (not wanted or {row["id"] for row in selected} == wanted), "Unknown case or split")
    return data, selected


def alternating_order(engines, repeat):
    require(type(repeat) is int and repeat >= 0 and engines and len(set(engines)) == len(engines),
            "Invalid repeated engine order")
    return list(engines) if repeat % 2 == 0 else list(reversed(engines))


def stop_valid(tokens, reason, eos, budget):
    valid = isinstance(tokens, list) and tokens and len(tokens) <= budget and all(
        type(value) is int and 0 <= value < 151936 for value in tokens)
    if not valid:
        return False
    first = next((i for i, value in enumerate(tokens) if value in eos), None)
    return (reason == "eos" and first == len(tokens)-1) or (
        reason == "max_new_tokens" and first is None and len(tokens) == budget)


def output_comparison(left, right):
    a, b = left["token_ids"], right["token_ids"]
    prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    exact = a == b and bool(a)
    return {"exact_token_match": exact, "matching_prefix_tokens": prefix,
        "first_difference_index": None if exact else prefix,
        "left_tokens": len(a), "right_tokens": len(b),
        "both_native_eos": left["finished_eos"] and right["finished_eos"],
        "complete_latency_ratio_left_over_right": left["external_wall_s"]/right["external_wall_s"],
        "token_rate_ratio_right_over_left": (len(b)/right["external_wall_s"])/(len(a)/left["external_wall_s"]),
        "same_output_latency_speedup": left["external_wall_s"]/right["external_wall_s"] if exact else None,
        "quality_preservation_inferred_from_target_verification": False}


def graph_signature(runner):
    if hasattr(runner, "execution_signature"):
        return runner.execution_signature()
    from benchmark_native_eos_fibonacci3b_v2 import graph_signature as frozen_signature
    return frozen_signature(runner)


def committed_cache_fingerprint(runner):
    """Untimed bit-level parity check of persistent ORIGINAL target state only."""
    import torch
    verifier = runner.verifier
    position = verifier.current_position
    def tensor_sha(value):
        raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()
    layers = [{"keys": tensor_sha(layer.keys[:, :, :position]),
               "values": tensor_sha(layer.values[:, :, :position])}
              for layer in verifier.cache.layers]
    return {"committed_position_excluding_anchor": position,
            "anchor": int(verifier.anchor.item()), "layers": layers,
            "committed_feature_history": tensor_sha(verifier.committed_feature_history[:, :position]),
            "speculative_uncommitted_cache_rows_deliberately_excluded": True}


def validate_control_parity(current, candidate, current_cache, candidate_cache):
    exact = current["token_ids"] == candidate["token_ids"]
    cache = current_cache == candidate_cache
    return {"fixed_packed_rows": 64, "exact_output_tokens": exact,
        "exact_committed_target_KV_anchor_positions_and_features": cache,
        "passed": bool(exact and cache), "current": current_cache, "candidate": candidate_cache,
        "this_finite_control_equivalence_check_is_not_arbitrary_prompt_equivalence": True}


class StreamingSink:
    """Actual cumulative-token consumption; never sleeps or simulates typing."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.queue = queue.Queue()
        self.events = []
        self.error = None
        self.last_tokens = None
        self.serialized_bytes = 0
        self.thread = threading.Thread(target=self._consume, name="benchmark-stream-consumer", daemon=True)
        self.thread.start()

    def emit(self, values):
        self.queue.put((time.perf_counter(), list(values)))

    def _consume(self):
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    return
                received, tokens = item
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                packet = json.dumps({"type": "token", "cumulative": True, "tokens": len(tokens), "text": text},
                                    ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
                self.serialized_bytes += len(packet)
                self.events.append({"driver_emit_s": received, "serialized_s": time.perf_counter(),
                                    "tokens": len(tokens), "serialized_bytes": len(packet)})
                self.last_tokens = tokens
        except BaseException as exc:
            self.error = exc

    def close(self):
        self.queue.put(None)
        self.thread.join(timeout=30.)
        require(not self.thread.is_alive(), "Streaming consumer did not drain")
        if self.error is not None:
            raise self.error

    def metrics(self, started):
        return {"events": len(self.events), "serialized_bytes": self.serialized_bytes,
            "first_driver_emit_s": self.events[0]["driver_emit_s"]-started if self.events else None,
            "first_serialized_text_s": self.events[0]["serialized_s"]-started if self.events else None,
            "last_serialized_text_s": self.events[-1]["serialized_s"]-started if self.events else None,
            "queue_UTF8_decode_and_JSON_drain_included_in_external_timer": True,
            "HTTP_transport_and_browser_rendering_included": False,
            "policy": "Original emits per token; speculative runners emit each target-committed chunk, cumulative full text as in the live demo"}


def invoke(engine, model, tokenizer, runner, ids, budget, eos, sink):
    if engine == "baseline":
        if sink is None:
            from run_numeric_quality import raw_hf_natural
            return raw_hf_natural(model, ids, budget, eos)
        from demo_chat.inference import LiveInference
        tokens, metrics = LiveInference._run_baseline(SimpleNamespace(model=model, eos=eos), ids, budget, sink.emit)
        return tokens, "eos" if tokens[-1] in eos else "max_new_tokens", metrics
    if sink is not None and "stream_callback" in inspect.signature(runner.run).parameters:
        tokens, metrics = runner.run(ids, budget, eos_ids=eos, stream_callback=sink.emit)
    elif sink is not None:
        from demo_chat.inference import _instance_observers
        with _instance_observers(runner, sink.emit) as observed:
            tokens, metrics = runner.run(ids, budget, eos_ids=eos)
        require(observed == tokens, "Streaming observers differ from the frozen accepted output")
    else:
        tokens, metrics = runner.run(ids, budget, eos_ids=eos)
    return tokens, metrics.get("stop_reason", "eos" if tokens[-1] in eos else "max_new_tokens"), metrics


def measure(engine, model, tokenizer, runner, ids, budget, eos, streaming):
    import torch
    sink = StreamingSink(tokenizer) if streaming else None
    signature = graph_signature(runner) if runner is not None else None
    torch.cuda.synchronize()
    began = time.perf_counter()
    try:
        tokens, reason, internal = invoke(engine, model, tokenizer, runner, ids, budget, eos, sink)
        torch.cuda.synchronize()
    finally:
        if sink is not None:
            sink.close()
    ended = time.perf_counter()
    require(stop_valid(tokens, reason, eos, budget), "Generation failed the first-EOS/full-public-budget rule")
    if runner is not None:
        require(graph_signature(runner) == signature, "Online generation created/replaced an execution graph")
        require(internal.get("all_output_tokens_target_verified") is True, "Candidate lacks final original-target verification")
    if sink is not None:
        require(sink.last_tokens == tokens, "Delivered stream differs from final target-accepted output")
    seconds = ended - began
    result = {"engine": engine, "streaming": streaming, "external_wall_s": seconds,
        "generated_tokens": len(tokens), "tokens_per_second": len(tokens)/seconds,
        "token_ids": tokens, "stop_reason": reason, "finished_eos": reason == "eos",
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
        "raw_text": tokenizer.decode(tokens, skip_special_tokens=False), "internal_metrics": internal,
        "profile_enabled": False, "new_execution_graphs_during_generation": 0 if runner is not None else None,
        "full_output_and_host_token_delivery_included": True,
        "preparation_and_posthoc_output_comparison_excluded": True}
    if sink is not None:
        result["stream"] = sink.metrics(began)
    if runner is not None:
        result["cycles"] = list(getattr(runner, "last_cycles", []))
    return result


def summarize(rows, engines, *, completed, integrity_passed):
    groups = {}
    configs = sorted({row.get("packed_budget", 64) for row in rows}, key=str)
    for streaming, packed in ((streaming, packed) for streaming in sorted({row["streaming"] for row in rows}) for packed in configs):
        selected = [row for row in rows if row["streaming"] == streaming and row.get("packed_budget", 64) == packed]
        if not selected:
            continue
        key = "streaming" if streaming else "non_streaming"
        if len(configs) > 1:
            key += ":N"+str(packed)
        comparisons = {}
        for left, right in (("baseline", "current"), ("baseline", "candidate"), ("current", "candidate")):
            pairs = [output_comparison(row["runs"][left], row["runs"][right]) for row in selected
                     if left in row.get("runs", {}) and right in row.get("runs", {})]
            if not pairs:
                continue
            native = all(row["both_native_eos"] for row in pairs)
            exact = all(row["exact_token_match"] for row in pairs)
            ratios = [row["complete_latency_ratio_left_over_right"] for row in pairs]
            comparisons[left+"_over_"+right] = {"paired_trials": len(pairs), "exact_matching_pairs": sum(
                row["exact_token_match"] for row in pairs), "all_native_eos": native,
                "complete_latency_ratio_median": statistics.median(ratios),
                "complete_latency_ratio_min": min(ratios),
                "same_output_speedup_eligible": bool(completed and integrity_passed and exact and native),
                "same_output_speedup_median": statistics.median(ratios) if completed and integrity_passed and exact and native else None,
                "same_output_speedup_min": min(ratios) if completed and integrity_passed and exact and native else None,
                "ten_x_in_every_exact_native_pair": bool(completed and integrity_passed and exact and native and min(ratios)>=10.),
                "numerical_difference_quality_must_be_graded_separately": not exact,
                "ratios_with_different_outputs_are_not_same_output_speedups": not exact}
        candidate_runs = [row["runs"]["candidate"] for row in selected if "candidate" in row.get("runs", {})]
        cycle_count = sum(run["internal_metrics"].get("cycles", 0) for run in candidate_runs)
        decode_s = sum(run["internal_metrics"].get("decode_s", 0.) for run in candidate_runs)
        committed = sum(run["generated_tokens"]-1 for run in candidate_runs)
        groups[key] = {"packed_budget": packed, "comparisons": comparisons,
            "candidate_cycle_objective": {"total_cycles": cycle_count, "committed_tokens": committed,
                "complete_decode_s": decode_s,
                "mean_complete_cycle_ms": 1000*decode_s/cycle_count if cycle_count else None,
                "mean_committed_tokens_per_cycle": committed/cycle_count if cycle_count else None,
                "committed_tokens_per_complete_decode_second": committed/decode_s if decode_s else None,
                "stream_consumer_final_drain_only_in_external_request_time": streaming},
            "engine_latency_median_s": {engine: statistics.median(
            row["runs"][engine]["external_wall_s"] for row in selected if engine in row.get("runs", {}))
            for engine in engines if any(engine in row.get("runs", {}) for row in selected)}}
    return {"complete_benchmark_and_integrity_passed": bool(completed and integrity_passed), "modes": groups,
        "quality_assessed_in_generation_process": False, "general_quality_preservation_certified": False,
        "ten_x_general_performance_guaranteed": False,
        "partial_or_failed_trials_eligible_for_improvement_selection": False}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="model-3b")
    p.add_argument("--draft-path", default="/tmp/lightreason-models/pard2-qwen3-8b")
    p.add_argument("--adaptation-checkpoint-path", default="/tmp/lightreason-pard2-lora-v1-B15/inference-best.pt")
    p.add_argument("--controls-manifest", default="optimization_20261106/controls/runtime-manifest.json")
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--suite", default="optimization_20261106/suite.json")
    p.add_argument("--split", choices=("development", "heldout", "all"), default="development")
    p.add_argument("--case-ids", help="Comma-separated IDs within the selected split")
    p.add_argument("--request-json", help="Reproduce a separately labeled archived workload using only its input fields")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--stream-modes", choices=("both", "streaming", "non-streaming"), default="both")
    p.add_argument("--engines", default="baseline,current,candidate")
    p.add_argument("--candidate-factory", default="cost_aware_tree3b_v1:CostAwareTree3BRunner")
    p.add_argument("--node-budget", type=int, choices=SUPPORTED_BUDGETS, default=64)
    p.add_argument("--sweep", action="store_true", help="Prepare all selected physical shapes in ONE process, then time fixed shapes separately")
    p.add_argument("--packed-budgets", help="Comma-separated actual prepared row counts for cost-aware selection")
    p.add_argument("--cycle-costs", help="JSON mapping actual packed shape to measured full cycle milliseconds")
    p.add_argument("--trace-ranking", action="store_true", help="Capture a separately timed diagnostic ranking run; no trace bookkeeping in performance trials")
    p.add_argument("--max-seconds", type=float, default=3000.)
    p.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    destination = Path(args.out)
    require(not destination.exists(), "Preserve prior evidence; use a fresh --out")
    require(1 <= args.repeats <= 12 and 30 <= args.max_seconds <= 3300, "Use bounded repeat/time settings")
    engines = args.engines.split(",")
    require(engines and set(engines) <= {"baseline", "current", "candidate"} and len(set(engines)) == len(engines), "Choose distinct valid engines")
    controls = verify_frozen_manifest(args.controls_manifest)
    sources_before = source_snapshot()
    observer_before = sha(ROOT / "demo_chat/inference.py")
    suite, cases = load_cases(args.suite, args.split, args.case_ids.split(",") if args.case_ids else None)
    if args.request_json:
        record = json.loads(Path(args.request_json).read_text())
        request = record.get("request", record)
        case = {key: request[key] for key in ("prompt", "system_prompt", "max_new_tokens") if key in request}
        require(isinstance(case.get("prompt"), str) and type(case.get("max_new_tokens")) is int, "Archived request needs original prompt and exact resolved output cap")
        cases = [{**case, "id": "archived-interactive-workload", "split": "diagnostic"}]
    packed = tuple(int(value) for value in args.packed_budgets.split(",")) if args.packed_budgets else None
    if args.sweep and packed is None:
        packed = SUPPORTED_BUDGETS
    require(packed is None or set(packed) <= set(SUPPORTED_BUDGETS), "Choose supported actual packed shapes")
    require(not args.sweep or ("candidate" in engines and 64 in packed), "A packed sweep needs candidate and the64-row parity family")
    costs = json.loads(Path(args.cycle_costs).read_text()) if args.cycle_costs else None
    require(not (args.sweep and costs is not None), "Measure fixed-shape sweep first; apply its calibrated cost curve in a separate adaptive run")
    if costs is not None:
        costs = {int(key): float(value) for key, value in costs.items()}
    report = {"schema": "lightreason-cost-aware-alternating-native-eos-v1", "frozen_controls": controls,
        "suite_sha256": sha(args.suite), "suite_selection_policy": suite["selection_policy"], "cases": cases,
        "split": cases[0]["split"] if args.request_json else args.split, "engines": engines,
        "requested_repeats": args.repeats, "requested_stream_modes": args.stream_modes,
        "candidate_configuration": {"factory": args.candidate_factory, "node_budget": args.node_budget,
            "packed_budgets": packed, "cycle_costs_ms": costs, "trace_ranking": args.trace_ranking},
        "sources_before": sources_before, "stream_observer_sha256": observer_before,
        "timing_policy": "External CUDA-synchronized full native-EOS generation plus host output delivery; setup and grading excluded; no profiler active",
        "streaming_policy": "Queue delivery, tokenizer UTF-8 decoding and JSON serialization drained inside timer; no HTTP/browser timing",
        "output_and_decoding_settings_identical_across_engines_and_streaming_modes": True,
        "baseline_changed_to_claim_speedup": False, "reference_answers_or_lengths_passed_to_candidate": False,
        "preparations": [], "trials": [], "ranking_diagnostics": [],
        "sweep": args.sweep, "sweep_budgets": list(packed) if args.sweep else None,
        "performance_trials_have_ranking_trace_disabled": True,
        "integrity_before": {}, "integrity_after": {}}
    model = tokenizer = control = candidate = state = None
    completed = False
    failure = None
    started = time.perf_counter()

    def deadline():
        require(time.perf_counter()-started <= args.max_seconds, "Bounded benchmark segment exceeded")

    def save():
        atomic_json(destination, report)

    try:
        model, tokenizer, control, eos, state, env = load_control(args)
        report["environment"] = env
        runners = {"current": control}
        if "candidate" in engines:
            module_name, class_name = args.candidate_factory.split(":")
            cls = getattr(importlib.import_module(module_name), class_name)
            candidate = cls.from_control(control, node_budget=args.node_budget, packed_budgets=packed,
                                        cycle_costs_ms=costs, trace_ranking=False)
            runners["candidate"] = candidate
        for name, runner in runners.items():
            report["integrity_before"][name] = audit_runner(runner)
            report[name+"_metadata"] = runner.metadata
        modes = [False, True] if args.stream_modes == "both" else [args.stream_modes == "streaming"]
        import torch
        for case_index, case in enumerate(cases):
            deadline()
            messages = []
            if case.get("system_prompt"):
                messages.append({"role": "system", "content": case["system_prompt"]})
            messages.append({"role": "user", "content": case["prompt"]})
            prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            budget = case["max_new_tokens"]
            require(len(prompt_ids)+budget+64 <= 4096, "Frozen output budget does not fit the frozen context")
            preparations = {"case_id": case["id"], "prompt_tokens": len(prompt_ids), "max_new_tokens": budget, "engines": {}}
            for name, runner in runners.items():
                preparations["engines"][name] = prepare_public_shape(model, runner, len(prompt_ids), budget)
            report["preparations"].append(preparations)
            ids = torch.tensor([prompt_ids], device=model.device, dtype=torch.long)
            if args.sweep and case_index == 0:
                # All later M values depend on this guard. Reading persistent
                # target cache bytes is excluded from every performance timer.
                candidate.select_fixed_budget(64)
                parity_current = measure("current", model, tokenizer, control, ids, budget, eos, False)
                current_cache = committed_cache_fingerprint(control)
                parity_candidate = measure("candidate", model, tokenizer, candidate, ids, budget, eos, False)
                candidate_cache = committed_cache_fingerprint(candidate)
                report["initial_N64_control_parity"] = validate_control_parity(
                    parity_current, parity_candidate, current_cache, candidate_cache)
                report["initial_N64_control_parity"].update(case_id=case["id"],
                    performance_trials=False, current_run=parity_current, candidate_run=parity_candidate)
                save()
                require(report["initial_N64_control_parity"]["passed"], "N64 candidate failed frozen-control output/cache parity; no sweep accepted")
            configs = list(packed) if args.sweep else [args.node_budget]
            for config_index, packed_budget in enumerate(configs):
                if args.sweep:
                    candidate.select_fixed_budget(packed_budget)
                for repeat in range(args.repeats):
                    mode_order = modes if repeat % 2 == 0 else list(reversed(modes))
                    for streaming in mode_order:
                        config_label = "adaptive" if costs is not None else packed_budget
                        row = {"case_id": case["id"], "split": case["split"], "repeat": repeat,
                            "packed_budget": config_label, "streaming": streaming,
                            "order": alternating_order(engines, repeat+case_index+config_index), "runs": {}}
                        report["trials"].append(row)
                        for engine in row["order"]:
                            deadline()
                            check_state(model, tokenizer, state)
                            result = measure(engine, model, tokenizer, runners.get(engine), ids, budget, eos, streaming)
                            result.update(case_id=case["id"], split=case["split"], repeat=repeat,
                                          packed_budget=config_label)
                            row["runs"][engine] = result
                            check_state(model, tokenizer, state)
                            save()
                            print(json.dumps({"case_id": case["id"], "repeat": repeat, "streaming": streaming,
                                "packed_budget": packed_budget, "engine": engine, "seconds": result["external_wall_s"],
                                "tokens": result["generated_tokens"], "finished_eos": result["finished_eos"],
                                "cycles": result["internal_metrics"].get("cycles"),
                                "committed_per_cycle": result["internal_metrics"].get("mean_committed_tokens_per_cycle")}), flush=True)
                        row["comparisons"] = {left+"_over_"+right: output_comparison(row["runs"][left], row["runs"][right])
                            for left, right in (("baseline", "current"), ("baseline", "candidate"), ("current", "candidate"))
                            if left in row["runs"] and right in row["runs"]}
                        save()
                if args.trace_ranking and candidate is not None and packed_budget == max(configs):
                    deadline()
                    signature = graph_signature(candidate)
                    candidate.trace_ranking = True
                    try:
                        torch.cuda.synchronize()
                        diagnostic_start = time.perf_counter()
                        ranking_tokens, ranking_metrics = candidate.run(ids, budget, eos_ids=eos)
                        torch.cuda.synchronize()
                        ranking_s = time.perf_counter()-diagnostic_start
                    finally:
                        candidate.trace_ranking = False
                    require(graph_signature(candidate) == signature, "Ranking diagnostic unexpectedly captured graphs")
                    report["ranking_diagnostics"].append({"case_id": case["id"], "packed_budget": packed_budget,
                        "excluded_from_performance_trials": True, "external_wall_s": ranking_s,
                        "token_ids": ranking_tokens, "metrics": ranking_metrics,
                        "trace": list(candidate.last_ranking_trace)})
                    save()
            del ids
        completed = True
    except BaseException as exc:
        failure = exc
        report["error"] = type(exc).__name__+": "+str(exc)
    finally:
        errors = []
        if model is not None and state is not None:
            try:
                check_state(model, tokenizer, state, full_bytes=True)
            except BaseException as exc:
                errors.append("target/tokenizer final guard: "+type(exc).__name__+": "+str(exc))
        for name, runner in (("current", control), ("candidate", candidate)):
            if runner is not None:
                try:
                    report["integrity_after"][name] = audit_runner(runner)
                except BaseException as exc:
                    errors.append(name+" final full audit: "+type(exc).__name__+": "+str(exc))
        from benchmark_folded_pard2_lora_tree3b_v1 import identity_gates
        gates = {}
        for name in report["integrity_before"]:
            gates[name] = identity_gates(report["integrity_before"].get(name), report["integrity_after"].get(name), CHECKPOINT_SHA)
        report["sources_after"] = source_snapshot()
        source_ok = report["sources_after"] == sources_before and sha(ROOT / "demo_chat/inference.py") == observer_before
        try:
            verify_frozen_manifest(args.controls_manifest)
            controls_ok = True
        except BaseException as exc:
            controls_ok = False
            errors.append("frozen manifest: "+str(exc))
        report["integrity_gates"] = gates
        report["final_guard_errors"] = errors
        report["all_sources_unchanged_during_run"] = source_ok
        report["frozen_controls_unchanged"] = controls_ok
        integrity = bool(gates and all(all(group.values()) for group in gates.values()) and not errors and source_ok and controls_ok)
        expected = len(cases)*args.repeats*(2 if args.stream_modes == "both" else 1)*(len(packed) if args.sweep else 1)
        complete = completed and len(report["trials"]) == expected and all(
            set(row["runs"]) == set(engines) for row in report["trials"])
        report["summary"] = summarize(report["trials"], engines, completed=complete, integrity_passed=integrity)
        report["elapsed_wall_including_setup_and_full_audits_s"] = time.perf_counter()-started
        save()
    if failure is not None:
        raise failure
    require(report["summary"]["complete_benchmark_and_integrity_passed"], "Incomplete benchmark or failed integrity")
    print(json.dumps({"out": str(destination), "summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
