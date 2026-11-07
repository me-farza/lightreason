"""Diagnostic full-cycle profiling of the frozen Qwen3B speculative control.

This does not edit or replace the frozen control.  A mechanically equivalent
diagnostic controller adds explicit ranges around its original operations. Its
output is checked against a fresh call of the uninstrumented frozen runner.
Profiled times are never used as final speedup evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

TARGET_SHA = "ab4dc8672b41311882ed404da089baad41db202eda849c68c14df3352c06e9a2"
CHECKPOINT_SHA = "8e5b53abec75a33825afca5c08a151d2cefd0a8a567bfff2ebc507bdd4ecc6b6"
EFFECTIVE_SHA = "e7f784c2cc8cc201cf6f36155a529ddfc7fc35c5eb557714200ef5bcb60e3f13"
ROOT = Path(__file__).resolve().parent
STAGES = ("request_prepare_check", "target_prefill", "initial_output_sync", "cycle",
          "drafting", "lattice_transfer", "selection_tree", "verification",
          "prediction_transfer", "acceptance", "visible_update", "cache_commit",
          "anchor_sync", "streaming", "final_sync_and_guards")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_snapshot():
    return {p.name: sha(p) for p in sorted(ROOT.glob("*.py")) if p.is_file()}


def verify_frozen_manifest(path):
    manifest = json.loads(Path(path).read_text())
    files = manifest.get("files", {})
    require(files and manifest.get("target_sha256", TARGET_SHA) == TARGET_SHA, "Invalid frozen controls manifest")
    mismatches = [name for name, expected in files.items()
                  if not (ROOT / name).is_file() or sha(ROOT / name) != expected]
    require(not mismatches, "Frozen controls changed: " + ", ".join(mismatches))
    return {"path": str(Path(path)), "sha256": sha(path), "files_checked": len(files),
            "all_frozen_control_sources_unchanged": True}


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def load_control(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from exact10_validation import hash_model_parameters
    from run_model_3b import validate_target, eos_token_ids
    from graph_prefill_compiled_pard2_lora_tree3b_v1 import GraphPrefillCompiledPARD2LoRATree3BRunner
    from feature_target_setup3b import validate_lazy_forward_transition
    from pard2_proposer3b_v1 import model_state, tokenizer_snapshot
    require(torch.__version__.split("+")[0] == "2.9.1" and transformers.__version__ == "4.57.1",
            "Use pinned Torch2.9.1/Transformers4.57.1")
    require("4090" in torch.cuda.get_device_name(), "Use the measured RTX4090")
    require(args.max_len == 4096, "The frozen current optimized control has max_len4096")
    require(sha(args.adaptation_checkpoint_path) == CHECKPOINT_SHA, "Frozen auxiliary checkpoint differs")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    began = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().requires_grad_(False).cuda()
    torch.cuda.synchronize()
    count, _ = validate_target(model)
    require(hash_model_parameters(model) == TARGET_SHA, "Original BF16 target weights differ")
    initial_state = model_state(model)
    versions = [(p, p._version, id(p), p.data_ptr()) for p in model.parameters()]
    loaded = time.perf_counter()
    control = GraphPrefillCompiledPARD2LoRATree3BRunner(model, tokenizer, args.draft_path,
        adaptation_checkpoint_path=args.adaptation_checkpoint_path,
        max_len=4096, node_budget=64, draft_tokens=15, top_k=8,
        compile_static=True, compile_draft=True, cuda_graph=True)
    control.set_graph_prefill_enabled(True)
    torch.cuda.synchronize()
    validate_lazy_forward_transition(initial_state, model_state(model))
    state = {"versions": versions, "model_state": model_state(model),
             "tokenizer_state": tokenizer_snapshot(tokenizer)}
    metadata = {"model_load_excluded_s": loaded - began,
        "control_constructor_excluded_s": time.perf_counter() - loaded,
        "model": "Qwen/Qwen2.5-3B-Instruct", "target_dtype": "BF16",
        "target_sha256": TARGET_SHA, "parameters": count,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "capability": list(torch.cuda.get_device_capability()),
        "control_configuration": {"max_len": 4096, "node_budget": 64,
            "draft_tokens": 15, "top_k": 8, "compile_static": True,
            "compile_draft": True, "cuda_graph": True}}
    return model, tokenizer, control, eos_token_ids(model, tokenizer), state, metadata


def check_state(model, tokenizer, state, *, full_bytes=False):
    from pard2_proposer3b_v1 import model_state, tokenizer_snapshot
    require(all(p._version == v and id(p) == identity and p.data_ptr() == pointer
                for p, v, identity, pointer in state["versions"]), "Target storage/version changed")
    require(model_state(model) == state["model_state"], "Initialized target module/forward state changed")
    require(tokenizer_snapshot(tokenizer) == state["tokenizer_state"], "Tokenizer changed")
    if full_bytes:
        from exact10_validation import hash_model_parameters
        require(hash_model_parameters(model) == TARGET_SHA, "Original target live BF16 bytes changed")


def audit_runner(runner):
    from benchmark_folded_pard2_lora_tree3b_v1 import full_audit, identity_gates
    audit = full_audit(runner)
    require(all(identity_gates(audit, audit, CHECKPOINT_SHA).values()), "Full target/auxiliary audit failed")
    require(audit.get("effective_adapted_auxiliary_live_sha256") == EFFECTIVE_SHA,
            "Selected effective auxiliary differs")
    return audit


def prepare_public_shape(model, runner, input_length, budget):
    import torch
    from run_numeric_quality import raw_hf_natural
    with torch.inference_mode(False), torch.no_grad():
        synthetic = torch.zeros((1, input_length), device=model.device, dtype=torch.long)
        torch.cuda.synchronize()
        began = time.perf_counter()
        result = runner.prepare(synthetic, budget)
        raw_hf_natural(model, synthetic, 3, set())
        torch.cuda.synchronize()
        result = {**result, "external_prepare_and_synthetic_HF_warmup_excluded_s": time.perf_counter() - began,
                  "synthetic_only": True, "answer_warmup_or_cache_used": False,
                  "input_length_only": input_length, "max_new_tokens": budget}
        del synthetic
    return result


class StageRecorder:
    def __init__(self):
        self.records = []

    @contextmanager
    def __call__(self, name):
        import torch
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        began = time.perf_counter()
        with torch.profiler.record_function(name):
            start.record()
            try:
                yield
            finally:
                end.record()
        self.records.append({"stage": name, "host_wall_us": (time.perf_counter() - began) * 1e6,
                             "start": start, "end": end})

    def resolve(self):
        summary = defaultdict(lambda: {"calls": 0, "host_wall_us": 0., "cuda_stream_span_us": 0.})
        calls = []
        for record in self.records:
            row = {"stage": record["stage"], "host_wall_us": record["host_wall_us"],
                   "cuda_stream_span_us": record["start"].elapsed_time(record["end"]) * 1000}
            calls.append(row)
            group = summary[row["stage"]]
            group["calls"] += 1
            for key in ("host_wall_us", "cuda_stream_span_us"):
                group[key] += row[key]
        for row in summary.values():
            row["mean_host_wall_us"] = row["host_wall_us"] / row["calls"]
            row["mean_cuda_stream_span_us"] = row["cuda_stream_span_us"] / row["calls"]
        return {"summary": dict(summary), "calls": calls,
                "diagnostic_only": True,
                "cuda_stream_spans_include_submission_gaps_and_host_synchronization": True,
                "cycle_ranges_enclose_child_ranges_do_not_sum_inclusive_totals": True}


def instrumented_control_run(runner, ids, n, eos_ids, stage, stream_callback=None):
    """The frozen Numerical3BTreeRunner operations, with diagnostic stage scopes."""
    import torch
    from pard_tree import best_first_tree
    from pard_tree_policies import trunk_best_first_tree
    runner._check(ids, n)
    with stage("request_prepare_check"):
        prepared = runner.prepare(ids, n)
    eos = set(eos_ids or [])
    with torch.inference_mode(False), torch.no_grad():
        torch.cuda.synchronize()
        start = time.perf_counter()
        with stage("target_prefill"):
            runner.proposer.reset()
            runner.visible[:, :ids.shape[1]].copy_(ids)
            anchor = runner.verifier.prefill(ids)
            runner.visible[:, ids.shape[1]:ids.shape[1] + 1].copy_(anchor)
        with stage("initial_output_sync"):
            tokens = [int(anchor.item())]
            torch.cuda.synchronize()
        if stream_callback:
            with stage("streaming"):
                stream_callback(tokens.copy())
        prefill_end = time.perf_counter()
        cycles = []
        while len(tokens) < n and (not eos or tokens[-1] not in eos):
            with stage("cycle"):
                visible_length = ids.shape[1] + len(tokens)
                with stage("drafting"):
                    top_ids, logps = runner.proposer.propose_lattice(runner.visible[:, :visible_length], runner.draft_tokens)
                with stage("lattice_transfer"):
                    cpu_ids, cpu_logps = top_ids.detach().cpu(), logps.detach().cpu()
                with stage("selection_tree"):
                    builder = best_first_tree if runner.tree_policy == "best-first" else trunk_best_first_tree
                    tree = builder(tokens[-1], cpu_ids, cpu_logps, node_budget=runner.node_budget)
                with stage("verification"):
                    predicted = runner.verifier.verify_tree(tree["tokens"], tree["depths"], tree["ancestors"], runner.verifier.current_position)
                with stage("prediction_transfer"):
                    predictions = predicted.reshape(-1).cpu().tolist()
                with stage("acceptance"):
                    path, emitted = runner.accept_path(tree, predictions, n - len(tokens), eos)
                    require(bool(emitted) and len(path) == len(emitted), "Rooted verification failed to advance")
                committed = len(emitted)
                with stage("visible_update"):
                    selected = torch.tensor(path, device=runner.model.get_input_embeddings().weight.device, dtype=torch.long)
                    runner.visible[:, visible_length:visible_length + committed].copy_(predicted.index_select(0, selected).view(1, -1))
                    tokens.extend(emitted)
                with stage("cache_commit"):
                    runner.verifier.commit_path(path)
                with stage("anchor_sync"):
                    require(int(runner.verifier.anchor.item()) == tokens[-1], "Compacted cache/anchor mismatch")
                if stream_callback:
                    with stage("streaming"):
                        stream_callback(tokens.copy())
                cycles.append({"node_count": tree["tokens"].numel(), "committed_tokens": committed,
                               "matched_draft_tokens": committed - 1, "accepted_node_indices": path})
        with stage("final_sync_and_guards"):
            torch.cuda.synchronize()
            end = time.perf_counter()
            runner.verifier.assert_target_unchanged()
            runner.proposer.assert_draft_unchanged()
    runner.last_cycles = cycles
    return tokens, {"prefill_s": prefill_end - start, "decode_s": end - prefill_end,
        "total_s": end - start, "generated_tokens": len(tokens), "cycles": len(cycles),
        "mean_committed_tokens_per_cycle": (len(tokens)-1)/len(cycles) if cycles else 0.,
        "verified_nodes": sum(c["node_count"] for c in cycles), "excluded_setup_s": prepared["setup_s"],
        "all_output_tokens_target_verified": True, "stop_reason": "eos" if tokens[-1] in eos else "max_new_tokens"}


def trace_coordination(path):
    events = json.loads(Path(path).read_text())["traceEvents"]
    names = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize",
             "cuStreamSynchronize", "cudaMemcpyAsync", "cudaMemcpy", "aten::item",
             "aten::_local_scalar_dense", "aten::_to_copy")
    totals = defaultdict(lambda: {"calls": 0, "inclusive_host_us": 0.})
    for event in events:
        if event.get("ph") != "X" or event.get("name") not in names:
            continue
        row = totals[event["name"]]
        row["calls"] += 1
        row["inclusive_host_us"] += float(event.get("dur", 0))
    return {"operations": dict(totals),
            "nested_ATen_and_runtime_totals_overlap_do_not_add": True,
            "synchronization_is_measured_inside_original_operation_not_subtracted_from_cycle": True}


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="model-3b")
    p.add_argument("--draft-path", default="/tmp/lightreason-models/pard2-qwen3-8b")
    p.add_argument("--adaptation-checkpoint-path", default="/tmp/lightreason-pard2-lora-v1-B15/inference-best.pt")
    p.add_argument("--controls-manifest", default="optimization_20261106/controls/manifest.json")
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--prompt", default="Write a Python function that returns the first n Fibonacci numbers and explain it.")
    p.add_argument("--system-prompt", default="")
    p.add_argument("--request-json", help="Input-only request or archived interactive record; prompt/system_prompt/max_new_tokens")
    p.add_argument("--suite", help="Input-only frozen suite JSON")
    p.add_argument("--case-id")
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = argument_parser().parse_args(argv)
    destination = Path(args.out)
    require(not destination.exists(), "Preserve existing profile artifact; choose a fresh --out")
    controls = verify_frozen_manifest(args.controls_manifest)
    sources_before = source_snapshot()
    case = {"prompt": args.prompt, "system_prompt": args.system_prompt, "max_new_tokens": args.max_new_tokens}
    if args.request_json:
        record = json.loads(Path(args.request_json).read_text())
        request = record.get("request", record)
        case.update({key: request[key] for key in ("prompt", "system_prompt", "max_new_tokens") if key in request})
    if args.suite:
        suite = json.loads(Path(args.suite).read_text())
        cases = [row for row in suite["cases"] if args.case_id is None or row["id"] == args.case_id]
        require(bool(cases), "Unknown case ID")
        case.update(cases[0])
    require(isinstance(case.get("prompt"), str) and case["prompt"].strip(), "Require the actual original prompt")
    budget = case["max_new_tokens"]
    require(type(budget) is int and 1 <= budget <= 3967, "Use the frozen positive public output cap")
    report = {"schema": "lightreason-complete-cycle-profile-v1", "diagnostic_only": True,
              "speedup_claimed": False, "frozen_controls": controls, "case": case,
              "sources_before": sources_before, "streaming": args.streaming}
    model, tokenizer, runner, eos, state, env = load_control(args)
    report["environment"] = env
    import torch
    messages = []
    if case.get("system_prompt"):
        messages.append({"role": "system", "content": case["system_prompt"]})
    messages.append({"role": "user", "content": case["prompt"]})
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    runner._check(torch.empty((1, len(prompt_ids))), budget)
    report["prompt_tokens"] = len(prompt_ids)
    report["preparation"] = prepare_public_shape(model, runner, len(prompt_ids), budget)
    before = audit_runner(runner)
    report["integrity_before"] = before
    ids = torch.tensor([prompt_ids], device=model.device, dtype=torch.long)
    recorder, streamed = StageRecorder(), []
    callback = (lambda values: streamed.append(list(values))) if args.streaming else None
    trace = destination.with_suffix(".trace.json")
    torch.cuda.synchronize()
    began = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False, with_stack=False, profile_memory=False) as profiler:
        tokens, metrics = instrumented_control_run(runner, ids, budget, eos, recorder, callback)
        torch.cuda.synchronize()
    report["profiled_external_wall_s"] = time.perf_counter() - began
    profiler.export_chrome_trace(str(trace))
    report["stages"] = recorder.resolve()
    report["coordination"] = trace_coordination(trace)
    from profile_numerical3b import summarize_trace, key_average_rows
    report["kernel_attribution"] = summarize_trace(trace, set(STAGES))
    report["key_averages"] = key_average_rows(profiler, 100)
    report["profiled_metrics"] = metrics
    report["profiled_token_ids"] = tokens
    report["profiled_cycles"] = runner.last_cycles
    report["profiled_text"] = tokenizer.decode(tokens, skip_special_tokens=True)
    require(not streamed or streamed[-1] == tokens, "Diagnostic streamed tokens differ from accepted output")
    # This fresh invocation checks instrumentation equivalence only. It is one
    # workload and is not an alternating benchmark or general speed evidence.
    torch.cuda.synchronize()
    began = time.perf_counter()
    frozen_tokens, frozen_metrics = runner.run(ids, budget, eos_ids=eos)
    torch.cuda.synchronize()
    report["fresh_uninstrumented_control_check"] = {"external_wall_s": time.perf_counter() - began,
        "token_ids": frozen_tokens, "metrics": frozen_metrics,
        "exact_output_matches_profiled_controller": frozen_tokens == tokens,
        "final_speedup_evidence": False}
    require(frozen_tokens == tokens, "Profiling controller changed output versus frozen run")
    check_state(model, tokenizer, state, full_bytes=True)
    after = audit_runner(runner)
    from benchmark_folded_pard2_lora_tree3b_v1 import identity_gates
    report["integrity_after"] = after
    report["integrity_gates"] = identity_gates(before, after, CHECKPOINT_SHA)
    report["sources_after"] = source_snapshot()
    require(report["sources_after"] == sources_before, "Project source changed during profile")
    verify_frozen_manifest(args.controls_manifest)
    require(all(report["integrity_gates"].values()), "Whole-run audit failed")
    report["complete_profile_and_integrity_passed"] = True
    atomic_json(destination, report)
    print(json.dumps({"out": str(destination), "profiled_cycles": metrics["cycles"],
        "profiled_metrics": metrics, "stages": report["stages"]["summary"],
        "fresh_uninstrumented_control_check": report["fresh_uninstrumented_control_check"],
        "complete_profile_and_integrity_passed": True}), flush=True)


if __name__ == "__main__":
    main()
