"""Compare unchanged BF16 Qwen3.6 expert execution on one CUDA device."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from safetensors import safe_open
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, Qwen3_5MoeForCausalLM

from edge35b.download import GIB, REVISION, available_memory
from edge35b.expert_stream import MODES, ExpertPool, patch_experts


PROMPTS = {
    "math": "A train leaves A at 60 km/h. Another leaves B at 90 km/h toward A half an hour later. If A and B are 255 km apart, how long after the first train leaves do they meet? Explain and give the exact answer.",
    "code": "Write a Python function merge_intervals(intervals) that merges overlapping closed intervals without mutating its input. Think through a counterexample before giving the function and its time complexity.",
    "logic": "A, B, C, D and E each arrive at different times. C arrives before B. E is after A. D is immediately after B. A is before C. Which people can arrive last? Explain.",
}


def validate_bytes(model, path, *, release_shard_cache=False, gpu_weight_views=None):
    """Compare every used tensor with its official shard, outside timing."""
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    by_file = {}
    seen = set()
    for name, parameter in model.named_parameters():
        key = name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name
        seen.add(key)
        by_file.setdefault(index[key], []).append((name, key, parameter))
    text_keys = {key for key in index if key.startswith("model.language_model.") or key == "lm_head.weight"}
    if seen != text_keys:
        raise RuntimeError(f"Text checkpoint coverage differs: {sorted(seen ^ text_keys)}")
    count = 0
    for file, parameters in by_file.items():
        shard = path / file
        # Separate mappings let the expert audit release its pages even when
        # Scheduled retains a few parameter views from this same shard.
        retained = safe_open(shard, framework="pt", device="cpu") if gpu_weight_views is not None else nullcontext()
        with safe_open(shard, framework="pt", device="cpu") as checkpoint, retained as resident_checkpoint:
            for name, key, parameter in parameters:
                retain_view = resident_checkpoint is not None and parameter.is_cuda
                source = resident_checkpoint if retain_view else checkpoint
                expected = source.get_tensor(key)
                actual = parameter.detach().cpu()
                if actual.shape != expected.shape or actual.dtype != torch.bfloat16 or expected.dtype != actual.dtype:
                    raise RuntimeError(f"Checkpoint geometry or dtype mismatch: {key}")
                if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
                    raise RuntimeError(f"Checkpoint byte mismatch: {key}")
                count += parameter.numel()
                if retain_view:
                    gpu_weight_views[name] = expected
                if release_shard_cache:
                    del actual, expected
        if release_shard_cache:
            # Studio already holds the expert bank in RAM. Release the audit's
            # temporary checkpoint pages before touching the next shard.
            with shard.open("rb") as source:
                os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return count


def load_target(path, *, shared_experts=None, gpu_weight_views=None):
    if transformers.__version__ != "5.7.0":
        raise RuntimeError("Use the audited Transformers 5.7.0 expert implementation")
    if json.loads((path / "lightreason-download.json").read_text())["revision"] != REVISION:
        raise RuntimeError("Use the verified pinned download")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This runner needs a BF16 CUDA GPU")
    if available_memory() < 14 * GIB:
        raise MemoryError("Need at least 14 GiB of available host RAM after staging")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    config = AutoConfig.from_pretrained(path, local_files_only=True).get_text_config()
    if (config.num_hidden_layers, config.num_experts, config.num_experts_per_tok, config.hidden_size) != (40, 256, 8, 2048):
        raise RuntimeError("Unexpected Qwen3.6-35B text architecture")
    model, info = Qwen3_5MoeForCausalLM.from_pretrained(
        path, config=config, dtype=torch.bfloat16, output_loading_info=True,
        local_files_only=True, attn_implementation="sdpa", experts_implementation="eager",
        key_mapping={r"^model\.language_model\.": "model."},
    )
    errors = {key: sorted(values) for key, values in info.items() if values}
    if errors:
        raise RuntimeError(f"Checkpoint loading did not cleanly cover the text model: {errors}")
    if shared_experts is not None:
        from demo_chat.resident_runners import attach_checkpoint_experts

        attach_checkpoint_experts(model, shared_experts)
    model.eval().requires_grad_(False)
    experts = []
    try:
        for layer in model.model.layers:
            experts.append((layer.mlp, layer.mlp._modules.pop("experts")))
        model.to("cuda:0")
    finally:
        for mlp, module in experts:
            mlp.experts = module
    if any(parameter.device.type != ("cpu" if ".mlp.experts." in name else "cuda")
           for name, parameter in model.named_parameters()):
        raise RuntimeError("The live model does not match the intended CPU/GPU placement")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    audit_started = time.perf_counter()
    count = validate_bytes(model, path, release_shard_cache=shared_experts is not None,
                           gpu_weight_views=gpu_weight_views)
    audit_s = time.perf_counter() - audit_started
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    if not eos or any(type(token) is not int for token in eos):
        raise RuntimeError("The official generation config must define the EOS IDs")
    resident_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.is_cuda)
    print(f"Loaded {count:,} original BF16 text parameters; resident weights {resident_bytes/GIB:.2f} GiB", flush=True)
    return model, tokenizer, eos, {"load_s": load_s, "checkpoint_audit_s": audit_s,
                                  "target_parameters": count, "resident_weight_bytes": resident_bytes,
                                  "host_memory_after_load": host_memory()}


def host_memory():
    fields = {"VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap"}
    snapshot = {"available_bytes": available_memory()}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in fields:
            snapshot[f"{key}_bytes"] = int(value.split()[0]) * 1024
    return snapshot


@torch.inference_mode()
def generate(model, ids, cap, eos, pool, audit_logits=False, *, on_token=None):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    token_ids, logit_rows = [], []
    cache = None
    pool.reset_stats()
    started = time.perf_counter()
    for step in range(cap):
        output = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        ids = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        token = int(ids.item())
        if audit_logits:
            logit_rows.append(output.logits.detach().cpu().view(torch.uint8).numpy().tobytes())
        token_ids.append(token)
        if step == 0:
            first = time.perf_counter()
            prefill_transfers = pool.snapshot()
        if on_token is not None:
            on_token(token_ids, time.perf_counter() - started)
        if token in eos:
            break
    torch.cuda.synchronize()
    end = time.perf_counter()
    return {
        "token_ids": token_ids, "stop_reason": "eos" if token_ids[-1] in eos else "limit",
        "time_s": end - started, "first_token_s": first - started,
        "decode_s": end - first, "decode_tokens": len(token_ids) - 1,
        "decode_tokens_per_s": (len(token_ids) - 1) / (end - first) if end > first else 0.0,
        "prefill_experts": prefill_transfers, "total_experts": pool.snapshot(),
        "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved": torch.cuda.max_memory_reserved(),
        "logits_sha256": hashlib.sha256(b"".join(logit_rows)).hexdigest() if audit_logits else None,
    }


def summarize(rows, modes):
    results = {}
    for mode in modes:
        records = [run for row in rows for run in row["runs"] if run["mode"] == mode]
        if records:
            results[mode] = {key: statistics.median(record[key] for record in records)
                             for key in ("time_s", "first_token_s", "decode_tokens_per_s")}
    comparisons = {}
    by_mode = [{run["mode"]: run for run in row["runs"]} for row in rows]
    for index, reference in enumerate(modes):
        for mode in modes[index + 1:]:
            pairs = [(runs[reference], runs[mode]) for runs in by_mode if reference in runs and mode in runs]
            if not pairs:
                continue
            comparisons[f"{reference}_over_{mode}"] = {
                "pairs": len(pairs),
                "exact_output": sum((a["token_ids"], a["stop_reason"]) == (b["token_ids"], b["stop_reason"])
                                    for a, b in pairs),
                "median_paired_speedup": statistics.median(a["time_s"] / b["time_s"] for a, b in pairs),
            }
            if pairs[0][0]["logits_sha256"] is not None:
                comparisons[f"{reference}_over_{mode}"]["bitwise_logits_matches"] = sum(
                    a["logits_sha256"] == b["logits_sha256"] for a, b in pairs)
    return {"modes": results, "comparisons": comparisons}


def tokenize(tokenizer, prompt, thinking):
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=True, return_tensors="pt", enable_thinking=thinking)["input_ids"].cuda()


def write_report(path, report):
    staged = path.with_suffix(path.suffix + ".part")
    staged.write_text(json.dumps(report, indent=2) + "\n")
    staged.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", help="Comma-separated built-in cases, defaults to math,code,logic")
    parser.add_argument("--prompt", help="Use one custom prompt and print each answer")
    parser.add_argument("--modes", default="dense,selected,cached")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cache-gib", type=int, default=12)
    parser.add_argument("--reference-cache-gib", type=int, help="Cache size for the first mode in a comparison")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--audit-logits", action="store_true")
    args = parser.parse_args()
    modes = args.modes.split(",")
    cases = (args.cases or "math,code,logic").split(",")
    if (not set(modes) <= set(MODES) or len(set(modes)) != len(modes)
            or not set(cases) <= PROMPTS.keys() or not 2 <= args.tokens <= 1024
            or not 1 <= args.repeats <= 10 or not 1 <= args.cache_gib <= 16):
        parser.error("Use known modes and cases, 2–1,024 tokens, 1–10 repeats and 1–16 GiB cache")
    if args.prompt is not None and (args.cases is not None or not args.prompt.strip()):
        parser.error("Use a nonempty --prompt by itself in place of --cases")
    cache_gib = dict.fromkeys(modes, args.cache_gib)
    if args.reference_cache_gib is not None:
        if len(modes) < 2 or not 1 <= args.reference_cache_gib <= 16:
            parser.error("The reference cache override needs multiple modes and 1–16 GiB")
        cache_gib[modes[0]] = args.reference_cache_gib
    prompts = {"custom": args.prompt} if args.prompt is not None else {case: PROMPTS[case] for case in cases}
    if args.out.exists():
        parser.error("Choose a new output file")
    model, tokenizer, eos, setup = load_target(args.model_dir)
    free, total = torch.cuda.mem_get_info()
    largest_cache = max((cache_gib[mode] for mode in modes if mode not in ("dense", "selected", "packed")), default=0)
    if largest_cache and free < (largest_cache + 3) * GIB:
        raise RuntimeError("Reduce the expert cache to preserve at least 3 GiB of CUDA memory")
    pool = ExpertPool(device="cuda:0", capacity_bytes=args.cache_gib * GIB, non_blocking=True)
    report = {
        "revision": REVISION, "target_dtype": "bfloat16", "torch": torch.__version__,
        "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(),
        "total_cuda_bytes": total, "free_cuda_after_load": free, "setup": setup,
        "eos": sorted(eos), "thinking": not args.no_thinking, "max_new_tokens": args.tokens,
        "cache_capacity_bytes": args.cache_gib * GIB, "logit_audit": args.audit_logits,
        "cache_gib_by_mode": cache_gib,
        "cache_policy": "expert cache cleared before every request; one untimed CUDA warmup", "rows": [],
    }
    warm_ids = tokenize(tokenizer, "Count to three.", not args.no_thinking)
    with patch_experts(model, mode="selected", pool=pool):
        warm = generate(model, warm_ids, 2, eos, pool)
    report["untimed_warmup_s"] = warm["time_s"]
    for mode in ("fused", "batched"):
        if mode in modes:
            with patch_experts(model, mode=mode, pool=pool):
                warm = generate(model, warm_ids, 2, eos, pool)
            report[f"{mode}_warmup_s"] = warm["time_s"]
    for index, (case, prompt) in enumerate(prompts.items()):
        ids = tokenize(tokenizer, prompt, not args.no_thinking)
        for repeat in range(args.repeats):
            rotation = (index + repeat) % len(modes)
            order = modes[rotation:] + modes[:rotation]
            row = {"case": case, "prompt": prompt, "input_ids": ids[0].tolist(), "repeat": repeat, "runs": []}
            report["rows"].append(row)
            for mode in order:
                pool.clear()
                pool._capacity_bytes = cache_gib[mode] * GIB
                torch.cuda.empty_cache()
                with patch_experts(model, mode=mode, pool=pool):
                    run = generate(model, ids, args.tokens, eos, pool, args.audit_logits)
                run.update(mode=mode, text=tokenizer.decode(run["token_ids"], skip_special_tokens=True))
                row["runs"].append(run)
                report["summary"] = summarize(report["rows"], modes)
                write_report(args.out, report)
                print(f"{case} {repeat} {mode}: {run['time_s']:.3f}s, {run['decode_tokens_per_s']:.2f} decode tok/s", flush=True)
                if args.prompt is not None:
                    print(run["text"], flush=True)
            for pair in report["summary"]["comparisons"].values():
                if pair["exact_output"] != pair["pairs"] or pair.get("bitwise_logits_matches", pair["pairs"]) != pair["pairs"]:
                    raise RuntimeError("Expert transfer changed the output. See the recorded runs for the mismatch.")
    pool.clear()
    if validate_bytes(model, args.model_dir) != setup["target_parameters"]:
        raise RuntimeError("Live parameter count changed")
    report["post_run_checkpoint_bytes_match"] = True
    report["host_memory_after_run"] = host_memory()
    write_report(args.out, report)


if __name__ == "__main__":
    main()
