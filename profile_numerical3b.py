"""Instrument the actual numerical3B/PARD chain; timings are diagnostic only.

Loads/prepares/warms outside profiling, then profiles runner.run(ids, tokens)
with a fixed public output budget. Instance-local wrappers mark draft, target,
commit and their original-HF prefills. CUDA-event phase spans supplement
Kineto graph-kernel correlation; they include host submission gaps and are not
pure kernel times. Optional isolated original-weight F.linear calls identify
the actual per-shape cuBLAS dispatch after the chain trace, outside generation.

No strict runner/source/model is modified. Kernel-name Tensor-Core/SIMT labels
are evidence from names, not SASS instruction verification or speedup claims.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from numerical3b_runner import Numerical3BRunner


PHASES = {"draft", "target", "commit", "target_prefill", "draft_prefill"}


def kernel_kind(name):
    lower = name.lower()
    if "gemvx" in lower or "gemv" in lower:
        return "simt_gemv_name_evidence"
    if any(word in lower for word in ("tensorop", "tensor_op", "xmma", "hmma", "s168", "s884", "h168", "h884")):
        return "tensor_core_name_evidence"
    if "gemm" in lower or "cutlass" in lower:
        return "gemm_arithmetic_not_identified"
    if any(word in lower for word in ("flash", "fmha", "attention", "attn", "softmax")):
        return "attention_or_softmax"
    if any(word in lower for word in ("reduce", "argmax", "topk")):
        return "reduction"
    if "triton" in lower:
        return "triton_other"
    return "other"


def event_ids(event):
    args = event.get("args") or {}
    result = []
    for key in ("correlation", "Correlation id", "correlation_id"):
        if key in args and args[key] not in (None, 0, -1):
            result.append(("correlation", str(args[key])))
    for key in ("External id", "external id", "external_id"):
        if key in args and args[key] not in (None, 0, -1):
            result.append(("external", str(args[key])))
    return result


def summarize_trace(path, range_names=None):
    events = json.loads(path.read_text())["traceEvents"]
    range_names = PHASES if range_names is None else range_names
    ranges = [event for event in events if event.get("name") in range_names and event.get("ph") == "X"]
    cpu_to_phase = {}
    for event in events:
        if event.get("cat") == "kernel" or not event_ids(event):
            continue
        timestamp = event.get("ts", -1)
        enclosing = [item for item in ranges if item.get("tid") == event.get("tid") and
                     item.get("ts", 0) <= timestamp <= item.get("ts", 0) + item.get("dur", 0)]
        if enclosing:
            # Prefer nested original-HF prefill over its enclosing draft range.
            phase = min(enclosing, key=lambda item: item.get("dur", float("inf")))["name"]
            for identifier in event_ids(event):
                cpu_to_phase[identifier] = phase
    groups = {}
    phase_totals = defaultdict(lambda: {"kernel_calls": 0, "summed_kernel_us": 0.0})
    for event in events:
        if event.get("cat") != "kernel":
            continue
        phase = next((cpu_to_phase[key] for key in event_ids(event) if key in cpu_to_phase), "unattributed")
        name = event.get("name", "unknown")
        key = (phase, name)
        if key not in groups:
            groups[key] = {"phase": phase, "name": name, "kind": kernel_kind(name),
                           "calls": 0, "summed_kernel_us": 0.0, "representative_args": event.get("args", {})}
        duration = float(event.get("dur", 0))
        groups[key]["calls"] += 1
        groups[key]["summed_kernel_us"] += duration
        phase_totals[phase]["kernel_calls"] += 1
        phase_totals[phase]["summed_kernel_us"] += duration
    grouped = sorted(groups.values(), key=lambda row: row["summed_kernel_us"], reverse=True)
    category_totals = defaultdict(lambda: {"calls": 0, "summed_kernel_us": 0.0})
    for row in grouped:
        category_totals[row["kind"]]["calls"] += row["calls"]
        category_totals[row["kind"]]["summed_kernel_us"] += row["summed_kernel_us"]
    return {"trace_path": str(path), "kernel_calls": sum(row["calls"] for row in grouped),
            "summed_kernel_us": sum(row["summed_kernel_us"] for row in grouped),
            "phase_kernel_totals": dict(phase_totals), "kernel_category_totals": dict(category_totals),
            "kernel_groups": grouped,
            "attribution": "CPU range to CUDA runtime/operator correlation/external IDs; unmatched kernels remain unattributed",
            "summed_kernel_duration_is_not_wall_time": True}


def key_average_rows(profile, maximum=80):
    rows = []
    for event in profile.key_averages(group_by_input_shape=True):
        device = float(getattr(event, "device_time_total", 0.0))
        self_device = float(getattr(event, "self_device_time_total", 0.0))
        rows.append({"name": event.key, "calls": event.count,
                     "cpu_total_us": float(event.cpu_time_total), "self_cpu_us": float(event.self_cpu_time_total),
                     "device_total_us_inclusive": device, "self_device_us": self_device,
                     "input_shapes": event.input_shapes})
    return sorted(rows, key=lambda row: row["self_device_us"], reverse=True)[:maximum]


@contextmanager
def phase_wrappers(runner):
    originals, records = [], []

    def install(obj, method_name, phase):
        original = getattr(obj, method_name)

        def wrapped(*args, **kwargs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            record = {"phase": phase, "call_index": sum(item["phase"] == phase for item in records),
                      "tensor_input_shapes": [list(value.shape) for value in args if isinstance(value, torch.Tensor)]}
            if phase == "draft":
                record["public_draft_tokens"] = kwargs.get("num_draft_tokens", args[1] if len(args) > 1 else None)
            elif phase == "target":
                record["base_position"] = kwargs.get("basep", args[1] if len(args) > 1 else None)
            elif phase == "commit":
                record["committed_inputs"] = args[0] if args else kwargs.get("acceptedinputcount")
            with torch.profiler.record_function(phase):
                start.record()
                output = original(*args, **kwargs)
                end.record()
            if phase == "draft" and getattr(obj, "last_report", None) is not None:
                record["draft_report"] = dict(obj.last_report)
            records.append({**record, "start_event": start, "end_event": end})
            return output

        originals.append((obj, method_name, original))
        setattr(obj, method_name, wrapped)

    install(runner.proposer, "propose", "draft")
    install(runner.verifier, "verify", "target")
    install(runner.verifier, "commit", "commit")
    install(runner.verifier, "prefill", "target_prefill")
    if hasattr(runner.proposer, "_prefill_true_history"):
        install(runner.proposer, "_prefill_true_history", "draft_prefill")
    try:
        yield records
    finally:
        for obj, method, original in reversed(originals):
            setattr(obj, method, original)


def resolve_phase_events(records):
    rows, summary = [], defaultdict(lambda: {"calls": 0, "cuda_stream_span_us": 0.0})
    for record in records:
        elapsed = record["start_event"].elapsed_time(record["end_event"]) * 1000.0
        row = {key: value for key, value in record.items() if key not in ("start_event", "end_event")}
        row["cuda_stream_span_us"] = elapsed
        rows.append(row)
        summary[row["phase"]]["calls"] += 1
        summary[row["phase"]]["cuda_stream_span_us"] += elapsed
    for value in summary.values():
        value["mean_cuda_stream_span_us"] = value["cuda_stream_span_us"] / value["calls"]
    return {"summary": dict(summary), "calls": rows,
            "scope": "stream spans include Python submission gaps; nested prefills overlap their parent draft spans"}


def projection_modules(model):
    layer = model.model.layers[0]
    return {"q": layer.self_attn.q_proj, "k": layer.self_attn.k_proj,
            "gate": layer.mlp.gate_proj, "down": layer.mlp.down_proj, "head": model.lm_head}


def projection_inventory(target, draft, target_m, draft_m, path):
    cases, inventory = [], []
    for role, model, size in (("target", target, target_m), ("draft", draft, draft_m)):
        for name, module in projection_modules(model).items():
            x = torch.randn((1, size, module.weight.shape[1]), device=module.weight.device, dtype=torch.bfloat16)
            cases.append((role, name, module, x))
            inventory.append({"role": role, "projection": name, "M": size,
                              "range": f"projection_{role}_{name}_M{size}",
                              "input_shape": list(x.shape), "weight_shape": list(module.weight.shape),
                              "bias_shape": list(module.bias.shape) if module.bias is not None else None})
            for _ in range(3):
                F.linear(x, module.weight, module.bias)
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
        for role, name, module, x in cases:
            label = f"projection_{role}_{name}_M{x.shape[1]}"
            with torch.profiler.record_function(label):
                F.linear(x, module.weight, module.bias)
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    raw_kernels = [{"name": event["name"], "duration_us": event.get("dur"),
                    "kind": kernel_kind(event["name"]), "args": event.get("args", {})}
                   for event in events if event.get("cat") == "kernel"]
    return {"shapes": inventory, "trace_path": str(path), "kernels": raw_kernels,
            "dispatch_by_projection_range": summarize_trace(path, {row["range"] for row in inventory}),
            "key_averages": key_average_rows(profile, 45),
            "scope": "isolated original-weight F.linear dispatch; compare with actual chain kernel names"}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="model-3b")
    parser.add_argument("--draft-path", default="model-pard")
    parser.add_argument("--block-width", type=int, choices=[8, 16, 32], default=8)
    parser.add_argument("--compile-static", action="store_true")
    parser.add_argument("--draft-engine", choices=["hf", "graph"], default="graph")
    parser.add_argument("--draft-attention", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--cache-mode", choices=["dynamic", "static"], default="static")
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--share-pool", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--warm-runs", type=int, default=1)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--prompt")
    parser.add_argument("--projection-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", default="results3b/profile-numerical.json")
    args = parser.parse_args()
    if args.tokens < 2 or args.warmups < 1 or args.warm_runs < 1:
        parser.error("Use tokens>=2 and positive warmup counts")
    if args.cache_mode != "static" and (args.cuda_graph or args.compile_static):
        parser.error("CUDA graphs/compilation require static target cache")
    if args.draft_engine == "graph" and args.draft_attention != "sdpa":
        parser.error("Graph PARD's boolean causal mask requires SDPA")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    from benchmark import PROMPTS
    if args.prompt is None and not 0 <= args.prompt_index < len(PROMPTS):
        parser.error("Prompt index is outside the original public prompt list")
    prompt = args.prompt if args.prompt is not None else PROMPTS[args.prompt_index]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().requires_grad_(False).cuda()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt").to(model.device)
    constructor_started = time.perf_counter()
    runner = Numerical3BRunner(model, tokenizer, args.draft_path, max_len=args.max_len,
        block_width=args.block_width, cache_mode=args.cache_mode, cuda_graph=args.cuda_graph,
        compile_static=args.compile_static, draft_engine=args.draft_engine,
        draft_attention=args.draft_attention, share_pool=args.share_pool, warmups=args.warmups)
    torch.cuda.synchronize()
    constructor_s = time.perf_counter() - constructor_started
    prepared = runner.prepare(ids, args.tokens)
    warm_tokens, warm_metrics = None, None
    for _ in range(args.warm_runs):
        warm_tokens, warm_metrics = runner.run(ids, args.tokens, eos_ids=None)
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with phase_wrappers(runner) as phase_records:
        with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
            with torch.profiler.record_function("actual_numerical3b_fixed_budget_run"):
                tokens, metrics = runner.run(ids, args.tokens, eos_ids=None)
            torch.cuda.synchronize()
    trace_path = path.with_name(path.stem + ".trace.json")
    profile.export_chrome_trace(str(trace_path))
    chain = summarize_trace(trace_path)
    environment = {"arguments": vars(args), "torch": torch.__version__,
        "transformers": __import__("transformers").__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "scope": "instrumented diagnostic actual-chain trace; no end-to-end speedup or strict-token identity claim",
        "load_s": load_s, "runner_constructor_s": constructor_s, "untimed_prepare": prepared,
        "public_prompt_tokens": ids.shape[1], "fixed_public_output_budget": args.tokens,
        "runner_metadata": runner.metadata,
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("profile_numerical3b.py", "numerical3b_runner.py", "relaxed3b_verifier.py",
                                       "pard_proposer.py", "graph_pard_proposer.py")}}
    draft = runner.proposer.model
    result = {"environment": environment, "chain_trace": chain,
        "phase_stream_spans": resolve_phase_events(phase_records),
        "key_averages": key_average_rows(profile), "profiled_run_metrics_diagnostic_only": metrics,
        "warm_run_metrics_diagnostic_only": warm_metrics, "profiled_token_ids": tokens,
        "profiled_ids_equal_uninstrumented_warm_run": tokens == warm_tokens,
        "cycles": runner.last_cycles,
        "linear_shapes": {role: {name: list(module.weight.shape) for name, module in projection_modules(source).items()}
                          for role, source in (("target", model), ("draft", draft))}}
    path.write_text(json.dumps(result, indent=2))
    if args.projection_profile:
        # Representative steady draft input length for one new visible token.
        # Actual lengths/reuse are in phase_stream_spans.calls.draft_report.
        projection_path = path.with_name(path.stem + ".projections.trace.json")
        result["projection_inventory"] = projection_inventory(model, draft, args.block_width,
            args.block_width - 1, projection_path)
    runner.verifier.assert_target_unchanged()
    runner.proposer.assert_draft_unchanged()
    result["target_and_draft_parameter_versions_unchanged"] = True
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"summary_path": str(path), "phase_stream_spans": result["phase_stream_spans"]["summary"],
        "kernel_calls": chain["kernel_calls"], "summed_kernel_us": chain["summed_kernel_us"],
        "kernel_categories": chain["kernel_category_totals"], "top_kernel_groups": chain["kernel_groups"][:18],
        "warm_profiled_token_agreement": tokens == warm_tokens}), flush=True)


if __name__ == "__main__":
    main()
