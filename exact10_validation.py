"""Locked-output validation and untimed tensor diagnostics for exact runners.

Required output gate: original six benchmark prompts at fixed128 tokens, plus
all32 raw-greedy quality answers through their first EOS. Default quality cache
is quality-exact.json because the older128-token quality-reference.json truncates
one answer. Cached expected tokens/text are never passed to a candidate.

  python exact10_validation.py --inspect-only
  python exact10_validation.py --candidate-factory exact_graph:make_quality_runner

Factory protocol: factory(model=model, tokenizer=tokenizer, args=args) returns
a runner with run(ids,n,eos_ids=None), or a quality callable whose .runner has
run(ids,n) for fixed-budget cases. Natural-EOS calls must actually stop at EOS.

This script is not a latency benchmark. Tensor hooks, CPU EOS checks, cached
answers and instrumentation must never enter a speed measurement. Use the same
candidate configuration in a separate paired timing run with identical prompts,
fixed budgets, warmups and CUDA synchronization. Require exact output on every
timed trial; report fixed-budget and natural-EOS results separately. A >=10x
latency claim requires measured request latency, not batched throughput.

capture_hf_stages() records original HF norm/projection/rotary/attention/MLP/head
stages with hooks. compare_stage_traces() reports the first differing stage and
byte-level parity. Compiled graph replay bypasses Python hooks; eager diagnostic
traces do not prove parity of a captured or fused kernel. Custom kernels should
provide their actual stage tensors via the same StageTrace.add(name,tensor) API.
"""
from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
CONFIG_ENVIRONMENT = ("EXACT10_NORM", "EXACT10_POINTWISE", "EXACT10_GATE",
                      "EXACT10_COMBINE", "EXACT10_SHARE_POOL", "EXACT10_FUSE_CACHE",
                      "EXACT10_PRELOAD_GATE_UP", "EXACT10_SCREEN_HEAD")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def project_python_hashes(root=ROOT):
    """Hash active project Python source, excluding environments/data/results."""
    root = Path(root)
    excluded = {"__pycache__", "model", "models", "node_modules", "results", "results10x", "results20x"}
    files = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts) or any(part in excluded for part in relative.parts[:-1]):
            continue
        files.append(path)
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(files)}


def resolved_candidate_configuration(candidate):
    """Read explicit configuration fields; never serialize tensors/arbitrary env."""
    driver = getattr(candidate, "runner", candidate)
    decoder = getattr(driver, "decoder", None)
    def fields(obj, names):
        values = {}
        if obj is not None:
            for name in names:
                value = getattr(obj, name, None)
                if value is None or isinstance(value, (str, bool, int, float)):
                    values[name] = value
                else:
                    values[name] = f"<non-scalar {type(value).__name__}>"
        return values
    def qualified(obj):
        return f"{type(obj).__module__}.{type(obj).__qualname__}" if obj is not None else None
    implementation_sources = {}
    for obj in (candidate, driver, decoder):
        if obj is None:
            continue
        try:
            path = Path(inspect.getfile(type(obj))).resolve()
        except (TypeError, OSError):
            continue
        if path.is_file() and path.suffix == ".py":
            implementation_sources[str(path)] = sha256_file(path)
    return {
        "adapter_class": qualified(candidate), "driver_class": qualified(driver),
        "decoder_class": qualified(decoder),
        "driver": fields(driver, ("max_len", "max_output", "warmups", "share_pool", "track_logits", "precompute_rope", "screen_head", "verify_head_steps", "head_verified_steps")),
        "decoder": fields(decoder, ("norm_mode", "pointwise", "gate_mode", "emulate_casts", "cache_rope", "combine_projections", "fuse_cache", "preload_gate_up")),
        "implementation_source_sha256": implementation_sources,
    }


def original_prompts(path=ROOT / "benchmark.py"):
    """Read the original literal without importing torch or running the file."""
    tree = ast.parse(Path(path).read_text())
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "PROMPTS" for target in statement.targets):
            value = ast.literal_eval(statement.value)
            if len(value) != 6 or not all(isinstance(prompt, str) for prompt in value):
                break
            return value
    raise ValueError("The benchmark must retain the original six literal PROMPTS")


def locked_cases(fixed_path, quality_path, task_path):
    fixed = json.loads(Path(fixed_path).read_text())
    quality = json.loads(Path(quality_path).read_text())
    fixed_environment = fixed["environment"]
    reference_runtime = quality["environment"]["identity"]["reference_runtime"]
    for name in ["torch", "transformers", "gpu"]:
        if fixed_environment.get(name) != reference_runtime.get(name):
            raise ValueError(f"Fixed and quality reference runtimes differ for {name}")
    if fixed_environment.get("dtype") != "torch.bfloat16":
        raise ValueError("Original fixed baseline must use BF16")
    task_data = json.loads(Path(task_path).read_text())
    task_by_id = {task["id"]: task for task in task_data["tasks"]}
    if len(task_by_id) != 32:
        raise ValueError("The quality gate requires all32 original diagnostic tasks")
    cases = []
    for index, prompt in enumerate(original_prompts()):
        matches = [row for row in fixed["trials"] if row["prompt_index"] == index]
        if not matches or any(row["prompt"] != prompt for row in matches):
            raise ValueError(f"Missing or changed original prompt {index}")
        tokens = matches[0]["baseline_tokens"]
        if len(tokens) != 128 or any(row["baseline_tokens"] != tokens for row in matches):
            raise ValueError(f"Original prompt {index} lacks a deterministic fixed128 baseline")
        cases.append({"id": f"original-{index}", "suite": "original-fixed128", "prompt": prompt,
                      "max_new_tokens": 128, "expected_tokens": tokens,
                      "expected_stop_reason": "fixed_budget", "skip_special_tokens": False})
    if len(quality["tasks"]) != 32 or {row["id"] for row in quality["tasks"]} != set(task_by_id):
        raise ValueError("Quality reference must contain exactly the original32 task IDs")
    eos = set(quality["environment"]["identity"]["eos_token_ids"])
    for row in quality["tasks"]:
        task = task_by_id[row["id"]]
        prompt = task["prompt"] + "\n\n" + task_data["response_instruction"]
        if prompt != row["prompt"] or row["expected"] != task["expected"]:
            raise ValueError(f"Quality task text/expected answer changed: {row['id']}")
        baseline = row["generations"]["baseline"]
        tokens = baseline.get("token_ids")
        if "error" in baseline or not tokens or baseline.get("stop_reason") != "eos":
            raise ValueError(f"Reference {row['id']} is incomplete; use a cache with all32 natural EOS answers")
        first_eos = next((i for i, token in enumerate(tokens) if token in eos), None)
        if first_eos != len(tokens) - 1 or len(tokens) > row["max_new_tokens"]:
            raise ValueError(f"Reference {row['id']} is not stopped on its first EOS")
        cases.append({"id": row["id"], "suite": "quality-natural-eos", "prompt": prompt,
                      "max_new_tokens": row["max_new_tokens"], "expected_tokens": tokens,
                      "expected_text_cached": baseline["text"], "expected_stop_reason": "eos",
                      "skip_special_tokens": True, "original_strict_score": row["scores"]["baseline"]})
    return cases, quality["environment"]["identity"]


def compare_output(case, candidate_tokens, candidate_text, expected_text, stop_reason):
    valid_tokens = isinstance(candidate_tokens, list) and all(type(token) is int for token in candidate_tokens)
    token_exact = valid_tokens and candidate_tokens == case["expected_tokens"]
    first = None
    if valid_tokens:
        first = next((i for i, (a, b) in enumerate(zip(case["expected_tokens"], candidate_tokens)) if a != b), None)
        if first is None and len(case["expected_tokens"]) != len(candidate_tokens):
            first = min(len(case["expected_tokens"]), len(candidate_tokens))
    text_exact = isinstance(candidate_text, str) and candidate_text.encode("utf-8") == expected_text.encode("utf-8")
    stop_exact = stop_reason == case["expected_stop_reason"]
    return {"tokens_exact_including_eos": token_exact, "text_utf8_exact": text_exact,
            "stop_reason_exact": stop_exact, "first_token_difference_zero_based": first,
            "expected_output_tokens": len(case["expected_tokens"]),
            "candidate_output_tokens": len(candidate_tokens) if valid_tokens else None,
            "exact_output_gate_passed": token_exact and text_exact and stop_exact}


class StageTrace:
    """Ordered untimed snapshots; custom implementations may emit these too."""
    def __init__(self, to_cpu=True):
        self.tensors = OrderedDict()
        self.counts = {}
        self.to_cpu = to_cpu

    def add(self, name, tensor):
        import torch
        if not isinstance(tensor, torch.Tensor):
            return
        occurrence = self.counts.get(name, 0)
        self.counts[name] = occurrence + 1
        key = f"{name}#{occurrence}"
        value = tensor.detach().contiguous()
        if self.to_cpu:
            value = value.cpu()
        self.tensors[key] = value.clone()


def capture_hf_stages(model, forward_call, to_cpu=True):
    """Run one instrumented eager forward and return (result, StageTrace).

    forward_call takes no arguments; caller owns separate reference/candidate
    cache states and identical input/prefix preparation. This temporarily wraps
    rotary and SDPA functions, restores them in finally, and is not thread-safe.
    No tensor returned to model execution is changed.
    """
    import torch
    from torch.nn import functional as functional
    trace, handles = StageTrace(to_cpu), []
    active = {"layer": None}

    def output_hook(name):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            trace.add(name, value)
        return hook

    def input_hook(name):
        def hook(module, inputs):
            if inputs:
                trace.add(name, inputs[0])
        return hook

    def add_output(module, name):
        handles.append(module.register_forward_hook(output_hook(name)))

    add_output(model.model.embed_tokens, "embedding")
    def rotary_output(module, inputs, output):
        trace.add("rotary.cos", output[0])
        trace.add("rotary.sin", output[1])
    handles.append(model.model.rotary_emb.register_forward_hook(rotary_output))
    for index, layer in enumerate(model.model.layers):
        prefix = f"layer.{index}"
        def enter(module, inputs, layer_index=index):
            active["layer"] = layer_index
        def leave(module, inputs, output):
            active["layer"] = None
        handles.append(layer.self_attn.register_forward_pre_hook(enter))
        handles.append(layer.self_attn.register_forward_hook(leave))
        add_output(layer.input_layernorm, prefix + ".norm_input")
        for name in ["q_proj", "k_proj", "v_proj"]:
            add_output(getattr(layer.self_attn, name), prefix + "." + name)
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(input_hook(prefix + ".attention_context")))
        add_output(layer.self_attn.o_proj, prefix + ".attention_projection")
        add_output(layer.post_attention_layernorm, prefix + ".norm_post")
        for name in ["gate_proj", "up_proj"]:
            add_output(getattr(layer.mlp, name), prefix + "." + name)
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(input_hook(prefix + ".gated_mlp")))
        add_output(layer.mlp.down_proj, prefix + ".mlp_output")
        add_output(layer, prefix + ".output")
    add_output(model.model.norm, "final_norm")
    add_output(model.lm_head, "lm_head")
    rotary_module = importlib.import_module(model.model.layers[0].self_attn.__module__)
    original_rope = getattr(rotary_module, "apply_rotary_pos_emb", None)
    original_sdpa = functional.scaled_dot_product_attention

    def rope(*args, **kwargs):
        output = original_rope(*args, **kwargs)
        if active["layer"] is not None:
            trace.add(f"layer.{active['layer']}.q_rope", output[0])
            trace.add(f"layer.{active['layer']}.k_rope", output[1])
        return output

    def sdpa(*args, **kwargs):
        output = original_sdpa(*args, **kwargs)
        if active["layer"] is not None:
            prefix = f"layer.{active['layer']}"
            trace.add(prefix + ".attention_q", args[0] if args else kwargs["query"])
            trace.add(prefix + ".attention_k", args[1] if len(args) > 1 else kwargs["key"])
            trace.add(prefix + ".attention_v", args[2] if len(args) > 2 else kwargs["value"])
            trace.add(prefix + ".attention", output)
        return output

    try:
        if original_rope is not None:
            rotary_module.apply_rotary_pos_emb = rope
        functional.scaled_dot_product_attention = sdpa
        with torch.no_grad():
            result = forward_call()
        return result, trace
    finally:
        functional.scaled_dot_product_attention = original_sdpa
        if original_rope is not None:
            rotary_module.apply_rotary_pos_emb = original_rope
        for handle in handles:
            handle.remove()


def tensor_parity(reference, candidate):
    import torch
    ref, test = reference.detach().contiguous().cpu(), candidate.detach().contiguous().cpu()
    result = {"reference_shape": list(ref.shape), "candidate_shape": list(test.shape),
              "reference_dtype": str(ref.dtype), "candidate_dtype": str(test.dtype)}
    if ref.shape != test.shape or ref.dtype != test.dtype:
        return {**result, "bitwise_equal": False, "reason": "shape_or_dtype_mismatch"}
    a, b = ref.reshape(-1).view(torch.uint8), test.reshape(-1).view(torch.uint8)
    differences = a != b
    count = int(differences.sum().item())
    result.update({"bitwise_equal": count == 0, "different_bytes": count,
                   "first_different_byte": int(differences.nonzero()[0].item()) if count else None})
    if ref.is_floating_point():
        finite = torch.isfinite(ref) & torch.isfinite(test)
        error = (ref.float() - test.float()).abs()[finite]
        result["max_abs_finite_error"] = error.max().item() if error.numel() else None
        result["mean_abs_finite_error"] = error.mean().item() if error.numel() else None
        result["nonfinite_elements"] = int((~finite).sum().item())
    return result


def compare_stage_traces(reference, candidate):
    reference = reference.tensors if isinstance(reference, StageTrace) else reference
    candidate = candidate.tensors if isinstance(candidate, StageTrace) else candidate
    missing = [key for key in reference if key not in candidate]
    extra = [key for key in candidate if key not in reference]
    stages = [{"stage": key, **tensor_parity(value, candidate[key])}
              for key, value in reference.items() if key in candidate]
    first = next((row["stage"] for row in stages if not row["bitwise_equal"]), None)
    return {"all_stages_bitwise_equal": not missing and not extra and bool(stages) and first is None,
            "first_differing_stage_in_reference_execution_order": first,
            "missing_candidate_stages": missing, "extra_candidate_stages": extra, "stages": stages,
            "scope": "Instrumented eager-stage diagnostic; captured/fused path parity is unproven unless these tensors came from that actual path"}


def candidate_fixed(candidate, ids, budget, tokenizer):
    driver = getattr(candidate, "runner", candidate)
    if not hasattr(driver, "run"):
        raise TypeError("Fixed-budget validation requires candidate.run or candidate.runner.run")
    arguments = {"eos_ids": None} if "eos_ids" in inspect.signature(driver.run).parameters else {}
    tokens, metrics = driver.run(ids, budget, **arguments)
    return tokens, tokenizer.decode(tokens), "fixed_budget", metrics


def candidate_natural(candidate, case, ids, tokenizer, eos_ids):
    if hasattr(candidate, "run"):
        if "eos_ids" not in inspect.signature(candidate.run).parameters:
            raise TypeError("Natural-EOS runner.run must accept eos_ids; fixed-budget trimming is not a native EOS check")
        tokens, metrics = candidate.run(ids, case["max_new_tokens"], eos_ids=eos_ids)
        reason = metrics.get("stop_reason") or ("eos" if tokens and tokens[-1] in eos_ids else "max_new_tokens")
        return tokens, tokenizer.decode(tokens, skip_special_tokens=True), reason, metrics
    if not callable(candidate):
        raise TypeError("Candidate must expose native run or a quality-generation callable")
    from quality_eval import as_generation
    result = as_generation(candidate(case["prompt"], case["max_new_tokens"]))
    return result.token_ids, result.text, result.stop_reason, result.metrics


def validate_identity(identity, args, tokenizer, model):
    import torch
    import transformers
    for name, expected in identity["model_and_tokenizer_files_sha256"].items():
        path = Path(args.model_path) / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Reference identity mismatch for {name}")
    if tokenizer.chat_template != identity["chat_template"]:
        raise ValueError("Chat template differs from the locked raw-greedy reference")
    expected_runtime = identity["reference_runtime"]
    current = {"torch": torch.__version__, "transformers": transformers.__version__,
               "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
               "capability": list(torch.cuda.get_device_capability())}
    for name, value in current.items():
        if expected_runtime.get(name) != value:
            raise ValueError(f"Reference runtime mismatch for {name}: expected {expected_runtime.get(name)}, got {value}")
    if any(parameter.dtype != torch.bfloat16 or parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Target must use frozen unchanged BF16 weights")
    return current


def hash_model_parameters(model, chunk_elements=1024 * 1024):
    """Hash actual parameter bytes, not only the unchanged checkpoint file.

    This CPU transfer is intentionally untimed and occurs only before candidate
    construction and after validation. It also detects writes through .data
    that can evade a parameter's ordinary version-counter check.
    """
    import torch
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(json.dumps([name, list(parameter.shape), str(parameter.dtype)]).encode())
        flat = parameter.detach().reshape(-1)
        for start in range(0, flat.numel(), chunk_elements):
            raw = flat[start:start + chunk_elements].contiguous().view(torch.uint8).cpu()
            digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="model")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--candidate-factory", default="exact_graph:make_quality_runner")
    parser.add_argument("--factory-args", default="{}", help="JSON object of additional factory-only args")
    parser.add_argument("--fixed-reference", default=str(ROOT / "results/sdpa-final.json"))
    parser.add_argument("--quality-reference", default=str(ROOT / "results20x/quality-exact.json"))
    parser.add_argument("--tasks", default=str(ROOT / "quality_tasks.json"))
    parser.add_argument("--out", default="results20x/exact10-validation.json")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--trace-factory", help="Optional module:function returning (reference_trace,candidate_trace) for first failing case")
    args = parser.parse_args()
    cases, identity = locked_cases(args.fixed_reference, args.quality_reference, args.tasks)
    locked = {"case_count": len(cases), "fixed128_cases": sum(c["suite"] == "original-fixed128" for c in cases),
              "natural_eos_cases": sum(c["suite"] == "quality-natural-eos" for c in cases),
              "source_sha256": {str(path): sha256_file(path) for path in
                                  [args.fixed_reference, args.quality_reference, args.tasks, __file__]},
              "candidate_gets_cached_output_tokens_or_eos_length": False}
    sources_before = project_python_hashes()
    sources_after = None
    explicit_configuration_environment = {name: os.environ.get(name) for name in CONFIG_ENVIRONMENT}
    locked["project_python_source_sha256_before"] = sources_before
    locked["project_python_source_scope"] = "All project *.py recursively, excluding hidden paths, model/models, node_modules, __pycache__ and results directories"
    locked["explicit_configuration_environment"] = explicit_configuration_environment
    if args.inspect_only:
        print(json.dumps(locked, indent=2))
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().requires_grad_(False).cuda()
    runtime = validate_identity(identity, args, tokenizer, model)
    eos = sorted(identity["eos_token_ids"])
    current_eos = model.generation_config.eos_token_id
    current_eos = [current_eos] if isinstance(current_eos, int) else list(current_eos or [])
    if tokenizer.eos_token_id is not None:
        current_eos.append(tokenizer.eos_token_id)
    if sorted(set(current_eos)) != eos:
        raise ValueError("EOS settings differ from the locked reference")
    versions = [(parameter, parameter._version) for parameter in model.parameters()]
    parameter_hash_before = hash_model_parameters(model)
    parameter_hash_after = None
    extra = json.loads(args.factory_args)
    if not isinstance(extra, dict) or set(extra) & set(vars(args)):
        raise ValueError("Factory-only args must be an object and cannot replace validation protocol args")
    # Reference file paths and expected tokens/lengths are deliberately absent.
    factory_args = SimpleNamespace(model_path=args.model_path, max_len=args.max_len, **extra)
    module, name = args.candidate_factory.rsplit(":", 1)
    candidate = getattr(importlib.import_module(module), name)(model=model, tokenizer=tokenizer, args=factory_args)
    resolved_configuration = resolved_candidate_configuration(candidate)
    rows = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    def payload():
        versions_untouched = all(parameter._version == version for parameter, version in versions)
        untouched = versions_untouched and parameter_hash_after == parameter_hash_before
        parameter_status = None if versions_untouched and parameter_hash_after is None else untouched
        sources_unchanged = None if sources_after is None else sources_before == sources_after
        driver = getattr(candidate, "runner", candidate)
        step_audits = getattr(driver, "head_verified_steps", 0)
        expected_step_audits = sum(max(0, len(row.get("candidate_tokens", [])) - 1) for row in rows)
        require_head_audits = bool(getattr(driver, "verify_head_steps", False))
        head_audits_ok = not require_head_audits or (step_audits == expected_step_audits and step_audits > 0)
        return {"environment": {**locked, "runtime": runtime, "candidate_factory": args.candidate_factory,
                    "factory_args": extra, "max_len": args.max_len, "target_parameters_unchanged": parameter_status,
                    "resolved_candidate_configuration": resolved_configuration,
                    "head_step_audits": {"required": require_head_audits, "completed": step_audits,
                                         "expected": expected_step_audits, "passed": head_audits_ok},
                    "project_python_source_sha256_after": sources_after,
                    "project_python_sources_unchanged": sources_unchanged,
                    "target_parameter_versions_unchanged": versions_untouched,
                    "target_parameter_bytes_sha256_before": parameter_hash_before,
                    "target_parameter_bytes_sha256_after": parameter_hash_after,
                    "timing_scope": "Validation only; reported runner timings are not accepted as speed evidence"},
                "summary": {"completed": len(rows), "required": 38,
                    "exact_cases": sum(row.get("exact_output_gate_passed", False) for row in rows),
                    "exact_output_gate_passed": untouched and sources_unchanged is True and head_audits_ok and len(rows) == 38 and all(row.get("exact_output_gate_passed", False) for row in rows),
                    "mismatch_ids": [row["id"] for row in rows if not row.get("exact_output_gate_passed", False)],
                    "speedup_demonstrated_by_this_script": False,
                    "quality_scope": "Complete cached output identity on38 cases; no broad accuracy claim or tolerance-based exception"},
                "cases": rows}
    for case in cases:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": case["prompt"]}],
                         add_generation_prompt=True, return_tensors="pt").cuda()
        expected_text = tokenizer.decode(case["expected_tokens"], skip_special_tokens=case["skip_special_tokens"])
        if "expected_text_cached" in case and expected_text != case["expected_text_cached"]:
            raise ValueError(f"Cached text/token decoding mismatch: {case['id']}")
        row = {"id": case["id"], "suite": case["suite"], "max_new_tokens": case["max_new_tokens"],
               "input_tokens": ids.shape[1], "input_token_ids_sha256": hashlib.sha256(json.dumps(ids[0].cpu().tolist()).encode()).hexdigest()}
        try:
            if case["suite"] == "original-fixed128":
                tokens, text, reason, metrics = candidate_fixed(candidate, ids, case["max_new_tokens"], tokenizer)
            else:
                tokens, text, reason, metrics = candidate_natural(candidate, case, ids, tokenizer, eos)
            row.update(compare_output(case, tokens, text, expected_text, reason))
            row.update({"candidate_tokens": tokens, "candidate_text": text, "candidate_stop_reason": reason,
                        "expected_tokens": case["expected_tokens"], "expected_text": expected_text,
                        "diagnostic_runner_metrics_not_speed_evidence": metrics})
        except Exception as exc:
            row.update({"exact_output_gate_passed": False, "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
        out.write_text(json.dumps(payload(), indent=2))
        print(json.dumps({"completed": len(rows), "id": case["id"],
                          "exact": row["exact_output_gate_passed"], "error": row.get("error")}), flush=True)
    diagnostic = None
    if args.trace_factory:
        failure = next((case for case, row in zip(cases, rows) if not row.get("exact_output_gate_passed", False)), None)
        if failure is not None:
            module, name = args.trace_factory.rsplit(":", 1)
            reference_trace, candidate_trace = getattr(importlib.import_module(module), name)(
                model=model, candidate=candidate, tokenizer=tokenizer, args=factory_args, case=failure)
            diagnostic = compare_stage_traces(reference_trace, candidate_trace)
    parameter_hash_after = hash_model_parameters(model)
    sources_after = project_python_hashes()
    result = payload()
    if diagnostic is not None:
        result["first_failure_tensor_diagnostic"] = diagnostic
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"summary": result["summary"]}), flush=True)
    if not result["summary"]["exact_output_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
