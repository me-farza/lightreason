"""Fresh, checkpoint-locked Qwen2.5-3B raw-greedy exact-output validation.

First collect a NEW original HF baseline, never reuse the 0.5B outputs:
  python exact3b_validate.py --mode baseline --model-path model3b \
      --out results3b/hf-reference.json
Then validate a factory returning an IntegratedRunner-like driver or adapter:
  python exact3b_validate.py --mode candidate --model-path model3b \
      --reference results3b/hf-reference.json \
      --candidate-factory exact3b_graphs:make_quality_runner \
      --out results3b/validation.json

The locked protocol is six original prompts at fixed128 tokens plus all32
diagnostic tasks at natural EOS, budget512 each. Every reasoning/output token,
EOS, decoded UTF8 byte and stop reason must match; there is no tolerance or
answer-only exception. A missing reference EOS fails the complete-answer gate.
The target is original frozen BF16 with raw argmax and no logits processors.

Candidate factories receive the model, tokenizer, max_len and explicit factory
configuration only. Individual calls receive the prompt/ids and public budget,
never reference tokens, output lengths, answers, or a reference-file argument.
This is an invocation protocol, not a sandbox against malicious Python code.

All timings are diagnostic: token-by-token EOS checks synchronize the GPU.
Use a separate warmed paired benchmark to establish single-request speedup.
Passing these38 finite cases establishes tested output identity, not a global
claim about reasoning, model accuracy or every possible future prompt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

from exact10_validation import (CONFIG_ENVIRONMENT, hash_model_parameters,
    original_prompts, project_python_hashes, resolved_candidate_configuration,
    sha256_file)
from quality_eval import as_generation, score


ROOT = Path(__file__).resolve().parent
SCHEMA = "lightreason-qwen2.5-3b-raw-greedy-reference-v1"
FIXED_BUDGET = 128
QUALITY_BUDGET = 512
ORIGINAL_TASKS_SHA256 = "f32fa00d4bfb30f49fd68d155845ead4b92ee9867e0a0c37717673a35cd6b176"
ORIGINAL_PROMPTS_SHA256 = "2dc83a91bb2bf772be78c68721f2c28b12a14cd8a608666bbb37592cd32db9d8"
ARCHITECTURE = {"model_type": "qwen2", "hidden_size": 2048,
                "num_hidden_layers": 36, "num_attention_heads": 16,
                "num_key_value_heads": 2, "head_dim": 128}


def protocol_cases(task_path):
    if sha256_file(task_path) != ORIGINAL_TASKS_SHA256:
        raise ValueError("Use the unchanged original32 diagnostic tasks and response instruction")
    data = json.loads(Path(task_path).read_text())
    tasks = data["tasks"]
    if len(tasks) != 32 or len({task["id"] for task in tasks}) != 32:
        raise ValueError("Require all32 unique original diagnostic tasks")
    prompts = original_prompts()
    if hashlib.sha256(json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest() != ORIGINAL_PROMPTS_SHA256:
        raise ValueError("Use the unchanged original six benchmark prompts")
    cases = [{"id": f"original-{index}", "suite": "original-fixed128", "prompt": prompt,
              "max_new_tokens": FIXED_BUDGET, "skip_special_tokens": False}
             for index, prompt in enumerate(prompts)]
    for task in tasks:
        cases.append({"id": task["id"], "suite": "quality-natural-eos",
                      "prompt": task["prompt"] + "\n\n" + data["response_instruction"],
                      "max_new_tokens": QUALITY_BUDGET, "skip_special_tokens": True,
                      "expected_answer_for_diagnostic_scoring_only": task["expected"]})
    return cases


def generation_eos(model, tokenizer):
    value = model.generation_config.eos_token_id
    values = [value] if isinstance(value, int) else list(value or [])
    if tokenizer.eos_token_id is not None:
        values.append(tokenizer.eos_token_id)
    if not values or any(type(value) is not int for value in values):
        raise ValueError("A nonempty explicit set of integer EOS token IDs is required")
    return sorted(set(values))


def model_identity(model, tokenizer, args):
    import torch
    import transformers
    config = model.config
    architecture = {name: getattr(config, name, None) for name in ARCHITECTURE}
    architecture["head_dim"] = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    if architecture != ARCHITECTURE:
        raise ValueError(f"This protocol requires the original Qwen2.5-3B architecture, got {architecture}")
    count = sum(parameter.numel() for parameter in model.parameters())
    if not 2_500_000_000 <= count <= 3_500_000_000:
        raise ValueError(f"Loaded parameter count {count} is not a 3B target")
    if model.training or any(parameter.dtype != torch.bfloat16 or parameter.requires_grad
                             for parameter in model.parameters()):
        raise ValueError("Target parameters must all be frozen original BF16")
    if getattr(config, "_attn_implementation", None) != "sdpa":
        raise ValueError("Reference and candidate must load original HF SDPA")
    base = Path(args.model_path)
    weights = sorted(base.glob("*.safetensors"))
    if not weights:
        raise ValueError("Require locally present safetensors; this script never downloads weights")
    files = weights + [base / name for name in (
        "model.safetensors.index.json", "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json")
        if (base / name).is_file()]
    hf_sources = {}
    for module_name in (type(model).__module__, "transformers.cache_utils",
                        "transformers.integrations.sdpa_attention"):
        module_path = inspect.getsourcefile(importlib.import_module(module_name))
        if module_path:
            hf_sources[module_name] = sha256_file(module_path)
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "architecture": architecture, "parameter_count": count, "dtype": "torch.bfloat16",
        "model_and_tokenizer_files_sha256": {path.name: sha256_file(path) for path in files},
        "chat_template": tokenizer.chat_template, "eos_token_ids": generation_eos(model, tokenizer),
        "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "hf_implementation_source_sha256": hf_sources,
        "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                    "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
                    "capability": list(torch.cuda.get_device_capability()),
                    "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
                    "sdpa_memory_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                    "sdpa_math_enabled": torch.backends.cuda.math_sdp_enabled(),
                    "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
        "decoding": "Original HF forward, raw greedy argmax, no logits processors, no sampling",
        "tasks_sha256": sha256_file(args.tasks),
        "original_prompts_source_sha256": sha256_file(ROOT / "benchmark.py"),
        "reference_generator_source_sha256": sha256_file(__file__),
    }


def validate_tokens(tokens, budget, eos, natural):
    valid = isinstance(tokens, list) and bool(tokens) and all(type(token) is int and token >= 0 for token in tokens)
    if not valid or len(tokens) > budget or (not natural and len(tokens) != budget):
        return False
    if natural:
        first = next((index for index, token in enumerate(tokens) if token in eos), None)
        return first == len(tokens) - 1
    return True


def load_reference(path, cases, tasks_path):
    reference = json.loads(Path(path).read_text())
    if reference.get("schema") != SCHEMA or reference.get("mode") != "baseline":
        raise ValueError("Require a NEW exact3b_validate.py HF baseline; old0.5B results are invalid")
    if reference.get("protocol") != protocol_description():
        raise ValueError("Reference generation protocol differs from locked3B protocol")
    identity = reference["identity"]
    if identity["architecture"] != ARCHITECTURE or identity["tasks_sha256"] != sha256_file(tasks_path):
        raise ValueError("Reference architecture or diagnostic tasks differ")
    if identity["reference_generator_source_sha256"] != sha256_file(__file__):
        raise ValueError("Reference generator source changed; collect a new baseline with the frozen script")
    if identity["original_prompts_source_sha256"] != sha256_file(ROOT / "benchmark.py"):
        raise ValueError("Original prompt/reference implementation source changed")
    if not reference.get("summary", {}).get("complete_reference_gate_passed"):
        raise ValueError("Reference is incomplete, changed weights/sources, or lacks natural EOS on all32 tasks")
    environment = reference["environment"]
    before = environment.get("target_parameter_bytes_sha256_before")
    if not isinstance(before, str) or len(before) != 64 or before != environment.get("target_parameter_bytes_sha256_after"):
        raise ValueError("Reference actual parameter-byte hashes are absent or changed")
    if environment.get("target_parameters_unchanged") is not True or environment.get("project_python_sources_unchanged") is not True:
        raise ValueError("Reference failed its live target/source identity checks")
    rows = reference["cases"]
    if len(rows) != 38 or [row["id"] for row in rows] != [case["id"] for case in cases]:
        raise ValueError("Require exactly the original ordered38 reference cases")
    eos = set(identity["eos_token_ids"])
    for case, row in zip(cases, rows):
        if any(row.get(key) != value for key, value in case.items()):
            raise ValueError(f"Reference case protocol changed: {case['id']}")
        natural = case["suite"] == "quality-natural-eos"
        if not validate_tokens(row.get("token_ids"), case["max_new_tokens"], eos, natural):
            raise ValueError(f"Reference token budget/EOS is invalid: {case['id']}")
        if row.get("stop_reason") != ("eos" if natural else "fixed_budget"):
            raise ValueError(f"Reference stop reason is invalid: {case['id']}")
        if not isinstance(row.get("text"), str) or not isinstance(row.get("raw_text"), str):
            raise ValueError(f"Reference text is missing: {case['id']}")
    return reference


def protocol_description():
    return {"fixed128_cases": 6, "natural_eos_cases": 32, "natural_eos_budget": QUALITY_BUDGET,
            "batch_size": 1, "sampling": False, "logits_processors": [],
            "target_weights": "Original frozen BF16, unchanged before/after",
            "output_gate": "All tokens including EOS, raw/full decoded UTF8 text, and stop reason exact"}


def input_ids(tokenizer, prompt, device):
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt").to(device)


def raw_hf_natural(model, ids, budget, eos):
    """Actual first-EOS raw greedy HF, independent of generation_config penalties."""
    import torch
    with torch.inference_mode():
        output = model(ids, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        tokens = []
        for index in range(budget):
            value = int(token.item())
            tokens.append(value)
            if value in eos or index + 1 == budget:
                break
            output = model(token, past_key_values=cache, use_cache=True, logits_to_keep=1)
            token = output.logits[:, -1].argmax(-1, keepdim=True)
    return tokens, "eos" if tokens[-1] in eos else "max_new_tokens", {"not_a_speed_benchmark": True}


def candidate_generation(candidate, case, ids, tokenizer, eos):
    """Never emulate natural EOS by running a fixed-budget job and trimming."""
    import torch
    driver = getattr(candidate, "runner", candidate)
    budget = case["max_new_tokens"]
    if case["suite"] == "original-fixed128":
        if not hasattr(driver, "run"):
            raise TypeError("Fixed output validation requires run(ids,budget)")
        kwargs = {"eos_ids": None} if "eos_ids" in inspect.signature(driver.run).parameters else {}
        tokens, metrics = driver.run(ids, budget, **kwargs)
        return tokens, tokenizer.decode(tokens), "fixed_budget", metrics
    if hasattr(candidate, "run") and "eos_ids" in inspect.signature(candidate.run).parameters:
        tokens, metrics = candidate.run(ids, budget, eos_ids=eos)
        reason = metrics.get("stop_reason") or ("eos" if tokens and tokens[-1] in eos else "max_new_tokens")
        return tokens, tokenizer.decode(tokens, skip_special_tokens=True), reason, metrics
    if callable(candidate):
        result = as_generation(candidate(case["prompt"], budget))
        return result.token_ids, result.text, result.stop_reason, result.metrics
    if all(hasattr(driver, name) for name in ("prepare", "prefill", "replay_one", "token")):
        prepared = driver.prepare(ids, budget)
        with torch.inference_mode(False), torch.no_grad():
            driver.prefill(ids)
            tokens = []
            for index in range(budget):
                value = int(driver.token.item())
                tokens.append(value)
                if value in eos or index + 1 == budget:
                    break
                driver.replay_one()
        reason = "eos" if tokens[-1] in eos else "max_new_tokens"
        return tokens, tokenizer.decode(tokens, skip_special_tokens=True), reason, {
            "not_a_speed_benchmark": True, "native_eos_adapter": True, "preparation": prepared}
    raise TypeError("Natural EOS requires native run(...,eos_ids), generation callable, or graph-step driver")


def compare(reference, tokens, text, reason, tokenizer, eos):
    valid = validate_tokens(tokens, reference["max_new_tokens"], eos,
                            reference["suite"] == "quality-natural-eos")
    exact_tokens = valid and tokens == reference["token_ids"]
    first = None
    if isinstance(tokens, list):
        first = next((index for index, pair in enumerate(zip(reference["token_ids"], tokens))
                      if pair[0] != pair[1]), None)
        if first is None and len(tokens) != len(reference["token_ids"]):
            first = min(len(tokens), len(reference["token_ids"]))
    decoded_raw = tokenizer.decode(tokens) if valid else None
    decoded = tokenizer.decode(tokens, skip_special_tokens=reference["skip_special_tokens"]) if valid else None
    text_exact = isinstance(text, str) and text.encode("utf-8") == reference["text"].encode("utf-8")
    decode_exact = decoded == reference["text"] and decoded_raw == reference["raw_text"]
    reason_exact = reason == reference["stop_reason"]
    return {"tokens_exact_including_eos": exact_tokens, "text_utf8_exact": text_exact,
            "raw_and_display_decoding_exact": decode_exact, "stop_reason_exact": reason_exact,
            "candidate_tokens_valid": valid, "first_token_difference_zero_based": first,
            "exact_output_gate_passed": exact_tokens and text_exact and decode_exact and reason_exact}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "candidate"], required=True)
    parser.add_argument("--model-path", default="model3b")
    parser.add_argument("--tasks", default=str(ROOT / "quality_tasks.json"))
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--reference", help="Fresh3B HF baseline JSON, mandatory for candidate mode")
    parser.add_argument("--candidate-factory", default="exact3b_graphs:make_quality_runner")
    parser.add_argument("--factory-args", default="{}", help="Explicit non-protocol factory configuration JSON")
    parser.add_argument("--out", required=True)
    parser.add_argument("--inspect-only", action="store_true", help="Validate reference/protocol on CPU; no model loading")
    args = parser.parse_args()
    if args.max_len <= QUALITY_BUDGET:
        parser.error("max-len must leave room for a prompt plus the locked512-token quality budget")
    if args.mode == "candidate" and not args.reference:
        parser.error("Candidate mode requires --reference pointing to a NEW3B HF baseline")
    if args.reference and Path(args.reference).resolve() == Path(args.out).resolve():
        parser.error("Output must not replace the locked reference")
    if args.mode == "baseline" and Path(args.out).exists() and not args.inspect_only:
        parser.error("Refusing to overwrite a baseline; choose a new output file")
    extra = json.loads(args.factory_args)
    forbidden = {"reference", "expected_tokens", "expected_text", "eos_lengths", "eos_length",
                 "output_lengths", "max_new_tokens", "tasks", "fixed_reference", "quality_reference"}
    if not isinstance(extra, dict) or set(extra) & (set(vars(args)) | forbidden):
        raise ValueError("Factory args must be a JSON object and may not replace protocol/reference fields")
    cases = protocol_cases(args.tasks)
    locked = load_reference(args.reference, cases, args.tasks) if args.mode == "candidate" else None
    if args.inspect_only:
        print(json.dumps({"schema": SCHEMA, "protocol": protocol_description(), "cases": len(cases),
                          "reference_valid": locked is not None, "gpu_work_performed": False}, indent=2))
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from benchmark import Reference
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().requires_grad_(False).cuda()
    identity = model_identity(model, tokenizer, args)
    if locked is not None and identity != locked["identity"]:
        differences = [name for name in set(identity) | set(locked["identity"])
                       if identity.get(name) != locked["identity"].get(name)]
        raise ValueError(f"Fresh3B baseline identity differs: {differences}")
    eos = set(identity["eos_token_ids"])
    device = model.model.embed_tokens.weight.device
    sources_before = project_python_hashes()
    parameter_hash_before = hash_model_parameters(model)
    if locked is not None and parameter_hash_before != locked["environment"]["target_parameter_bytes_sha256_before"]:
        raise ValueError("Actual loaded parameter bytes differ from the fresh3B HF baseline")
    versions = [(parameter, parameter._version) for parameter in model.parameters()]
    parameter_hash_after, sources_after, fatal_error = None, None, None
    candidate, configuration = None, None
    rows = []
    baseline = Reference(model)
    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "max_len": args.max_len,
        "candidate_factory": args.candidate_factory if args.mode == "candidate" else None,
        "factory_args": extra if args.mode == "candidate" else None,
        "reference_file_sha256": sha256_file(args.reference) if args.reference else None,
        "explicit_configuration_environment": {name: os.environ.get(name) for name in CONFIG_ENVIRONMENT},
        "candidate_gets_cached_outputs_or_eos_lengths": False,
        "project_python_source_sha256_before": sources_before,
        "validation_timings_are_speed_evidence": False,
    }

    def payload():
        versions_ok = all(parameter._version == version for parameter, version in versions)
        weights_ok = parameter_hash_after == parameter_hash_before and versions_ok
        sources_ok = sources_after == sources_before
        if args.mode == "baseline":
            exact = len(rows) == 38 and all(row.get("reference_case_complete", False) for row in rows)
        else:
            exact = len(rows) == 38 and all(row.get("exact_output_gate_passed", False) for row in rows)
        driver = getattr(candidate, "runner", candidate)
        required_audits = bool(getattr(driver, "verify_head_steps", False))
        completed_audits = getattr(driver, "head_verified_steps", 0)
        expected_audits = sum(max(0, len(row.get("token_ids", [])) - 1) for row in rows)
        audits_ok = not required_audits or (completed_audits == expected_audits and completed_audits > 0)
        passed = exact and weights_ok and sources_ok and audits_ok and fatal_error is None
        key = "complete_reference_gate_passed" if args.mode == "baseline" else "exact_output_gate_passed"
        return {"schema": SCHEMA, "mode": args.mode, "protocol": protocol_description(), "identity": identity,
            "environment": {**environment, "resolved_candidate_configuration": configuration,
                "target_parameter_versions_unchanged": versions_ok,
                "target_parameter_bytes_sha256_before": parameter_hash_before,
                "target_parameter_bytes_sha256_after": parameter_hash_after,
                "target_parameters_unchanged": weights_ok if parameter_hash_after is not None else None,
                "head_step_audits": {"required": required_audits, "completed": completed_audits,
                                     "expected": expected_audits, "passed": audits_ok},
                "project_python_source_sha256_after": sources_after,
                "project_python_sources_unchanged": sources_ok if sources_after is not None else None},
            "summary": {"completed": len(rows), "required": 38, key: passed,
                "exact_or_complete_cases": sum(row.get("reference_case_complete" if args.mode == "baseline"
                                               else "exact_output_gate_passed", False) for row in rows),
                "natural_eos_cases": sum(row.get("stop_reason") == "eos" for row in rows),
                "strict_final_json_correct": sum(bool(row.get("strict_final_json_score", {}).get("correct")) for row in rows),
                "strict_final_json_baseline_correct_regressions": sum(
                    bool(row.get("baseline_strict_final_json_score", {}).get("correct"))
                    and not bool(row.get("strict_final_json_score", {}).get("correct")) for row in rows),
                "speedup_demonstrated": False,
                "scope": "Full output identity on38 diagnostics only; no global reasoning/accuracy guarantee",
                "fatal_error": fatal_error}, "cases": rows}

    try:
        if args.mode == "candidate":
            module, name = args.candidate_factory.rsplit(":", 1)
            factory_args = SimpleNamespace(model_path=args.model_path, max_len=args.max_len, **extra)
            with torch.inference_mode(False), torch.no_grad():
                candidate = getattr(importlib.import_module(module), name)(model=model, tokenizer=tokenizer, args=factory_args)
            configuration = resolved_candidate_configuration(candidate)
            driver = getattr(candidate, "runner", candidate)
            decoder = getattr(driver, "decoder", None)
            configuration["driver"]["verifier_lanes"] = getattr(driver, "verifier_lanes", None)
            configuration["decoder"].update({name: getattr(decoder, name, None) for name in
                ("mean_threads", "norm_warps", "mean_plan")})
            cache = getattr(driver, "cache", None)
            configuration["cache_class"] = f"{type(cache).__module__}.{type(cache).__qualname__}" if cache is not None else None
            if cache is not None and hasattr(cache, "allocated_bytes"):
                configuration["cache_allocated_bytes"] = cache.allocated_bytes()
        for index, case in enumerate(cases):
            ids = input_ids(tokenizer, case["prompt"], device)
            row = {**case, "input_tokens": ids.shape[1],
                   "input_token_ids_sha256": hashlib.sha256(json.dumps(ids[0].cpu().tolist()).encode()).hexdigest()}
            try:
                if ids.shape[1] + case["max_new_tokens"] > args.max_len:
                    raise ValueError("Prompt plus locked generation budget exceeds max_len")
                if args.mode == "baseline":
                    if case["suite"] == "original-fixed128":
                        tokens, metrics = baseline.run(ids, FIXED_BUDGET)
                        reason = "fixed_budget"
                    else:
                        tokens, reason, metrics = raw_hf_natural(model, ids, QUALITY_BUDGET, eos)
                    text = tokenizer.decode(tokens, skip_special_tokens=case["skip_special_tokens"])
                    row["reference_case_complete"] = validate_tokens(tokens, case["max_new_tokens"], eos,
                        case["suite"] == "quality-natural-eos")
                else:
                    expected = locked["cases"][index]
                    if row["input_token_ids_sha256"] != expected["input_token_ids_sha256"]:
                        raise ValueError("Tokenized input differs from locked reference")
                    if tokenizer.decode(expected["token_ids"], skip_special_tokens=case["skip_special_tokens"]) != expected["text"]:
                        raise ValueError("Locked reference text disagrees with its token IDs")
                    if tokenizer.decode(expected["token_ids"]) != expected["raw_text"]:
                        raise ValueError("Locked reference raw text disagrees with its token IDs")
                    tokens, text, reason, metrics = candidate_generation(candidate, case, ids, tokenizer, eos)
                    row.update(compare(expected, tokens, text, reason, tokenizer, eos))
                    row.update({"expected_token_ids": expected["token_ids"], "expected_text": expected["text"],
                                "expected_stop_reason": expected["stop_reason"]})
                row.update({"token_ids": tokens, "text": text,
                            "raw_text": tokenizer.decode(tokens), "stop_reason": reason,
                            "diagnostic_metrics_not_speed_evidence": metrics})
                if "expected_answer_for_diagnostic_scoring_only" in case:
                    row["strict_final_json_score"] = score(text, case["expected_answer_for_diagnostic_scoring_only"])
                    if args.mode == "candidate":
                        row["baseline_strict_final_json_score"] = expected["strict_final_json_score"]
            except Exception as error:
                row.update({"reference_case_complete" if args.mode == "baseline" else "exact_output_gate_passed": False,
                            "error": f"{type(error).__name__}: {error}"})
            rows.append(row)
            atomic_json(args.out, payload())
            print(json.dumps({"completed": len(rows), "id": case["id"],
                "passed": row.get("reference_case_complete" if args.mode == "baseline" else "exact_output_gate_passed", False),
                "output_tokens": len(row.get("token_ids", [])), "stop_reason": row.get("stop_reason"),
                "error": row.get("error")}), flush=True)
    except Exception as error:
        fatal_error = f"{type(error).__name__}: {error}"
    finally:
        parameter_hash_after = hash_model_parameters(model)
        sources_after = project_python_hashes()
        result = payload()
        atomic_json(args.out, result)
        print(json.dumps({"summary": result["summary"]}), flush=True)
    gate = "complete_reference_gate_passed" if args.mode == "baseline" else "exact_output_gate_passed"
    if not result["summary"][gate]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
