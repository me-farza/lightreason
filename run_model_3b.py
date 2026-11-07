"""Run the original BF16 Qwen2.5-3B with native-EOS raw-greedy decoding.

Examples:
  python run_model_3b.py --model-path model-3b --prompt "Explain why the sky is blue."
  python run_model_3b.py --engine original --prompt "What is 17 times 23?" --json

The default bucketed engine passed the full38-case exact-output validation;
the exact option retains fixed-position graphs. Preparation is reported
separately and can be expensive on first use. The original engine directly
calls the unchanged HF model with its ordinary dynamic cache. Neither path
adds response instructions, logits processors, sampling, or a forced-short
reasoning mode. This CLI does not read reference answers or validation output.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def eos_token_ids(model, tokenizer):
    configured = model.generation_config.eos_token_id
    values = [configured] if isinstance(configured, int) else list(configured or [])
    if tokenizer.eos_token_id is not None:
        values.append(tokenizer.eos_token_id)
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("The original model/tokenizer must define integer EOS token IDs")
    return set(values)


def generation_metrics(tokens, eos, started, prefill_end, ended):
    decode_tokens = len(tokens) - 1
    decode_s = ended - prefill_end
    return {"prefill_s": prefill_end - started, "decode_s": decode_s,
            "generation_s": ended - started,
            "decode_tokens": decode_tokens,
            "decode_tokens_s": decode_tokens / decode_s if decode_tokens and decode_s > 0 else 0.0,
            "generated_tokens": len(tokens),
            "native_eos_stopping": True,
            "stop_reason": "eos" if tokens[-1] in eos else "max_new_tokens",
            "per_token_host_eos_checks": True,
            "timing_scope": "interactive generation latency; setup excluded; not a comparative speed benchmark"}


@torch.no_grad()
def run_original(model, ids, budget, eos):
    """Original HF forwards and unmodified argmax, stopping at the first EOS."""
    with torch.inference_mode():
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = model(ids, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        tokens = [int(token.item())]
        torch.cuda.synchronize()
        prefill_end = time.perf_counter()
        while len(tokens) < budget and tokens[-1] not in eos:
            output = model(token, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            token = output.logits[:, -1].argmax(-1, keepdim=True)
            tokens.append(int(token.item()))
        torch.cuda.synchronize()
        ended = time.perf_counter()
    return tokens, generation_metrics(tokens, eos, started, prefill_end, ended), {
        "setup_s": 0.0, "new_graphs": 0, "prepare_wall_s": 0.0}


@torch.no_grad()
def run_exact(runner, ids, budget, eos):
    preparation_started = time.perf_counter()
    prepared = dict(runner.prepare(ids, budget))
    torch.cuda.synchronize()
    prepared["prepare_wall_s"] = time.perf_counter() - preparation_started
    # Runner banks must be ordinary tensors, not inference tensors: its graph
    # capture and subsequent prefix-cache copies mutate those permanent banks.
    with torch.inference_mode(False), torch.no_grad():
        started = time.perf_counter()
        runner.prefill(ids)
        tokens = [int(runner.token.item())]
        torch.cuda.synchronize()
        prefill_end = time.perf_counter()
        while len(tokens) < budget and tokens[-1] not in eos:
            runner.replay_one()
            tokens.append(int(runner.token.item()))
        torch.cuda.synchronize()
        ended = time.perf_counter()
    if hasattr(runner, "assert_target_unchanged"):
        runner.assert_target_unchanged()
    metrics = generation_metrics(tokens, eos, started, prefill_end, ended)
    metrics["prepared_graphs"] = len(runner.graphs)
    return tokens, metrics, prepared


def validate_target(model):
    config = model.config
    if (config.model_type != "qwen2" or config.hidden_size != 2048 or
            config.num_hidden_layers != 36 or config.num_attention_heads != 16 or
            config.num_key_value_heads != 2):
        raise ValueError("Use the original Qwen2.5-3B architecture")
    parameters = list(model.parameters())
    count = sum(parameter.numel() for parameter in parameters)
    if not 2_500_000_000 <= count <= 3_500_000_000:
        raise ValueError(f"Expected a 3B target, loaded {count} parameters")
    if model.training or any(parameter.dtype != torch.bfloat16 or parameter.requires_grad for parameter in parameters):
        raise ValueError("Every original parameter must remain frozen BF16")
    return count, [(parameter, parameter._version) for parameter in parameters]


def source_hashes():
    root = Path(__file__).resolve().parent
    names = ("run_model_3b.py", "exact3b_runner.py", "exact3b_core.py", "exact3b_graphs.py", "exact3b_head.py", "exact3b_dynamic.py",
             "exact10_graphs.py", "exact_graph.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names if (root / name).is_file()}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="model-3b")
    parser.add_argument("--engine", choices=["exact", "bucketed", "original"], default="bucketed")
    parser.add_argument("--full-head", action="store_true",
                        help="Use the complete original output head with exact/bucketed engines")
    parser.add_argument("--tokens", type=int, default=512, help="Maximum new tokens; native EOS can stop earlier")
    parser.add_argument("--max-len", type=int, default=1024, help="Prompt plus generation capacity")
    parser.add_argument("--prompt", help="Run one prompt; omit for an interactive session")
    parser.add_argument("--json", action="store_true", help="Emit one full JSON result per prompt on stdout")
    parser.add_argument("--report", type=Path, help="Write environment and all session results to this JSON file")
    args = parser.parse_args()
    if args.tokens < 1 or args.max_len < 2 or args.tokens >= args.max_len:
        parser.error("Use positive --tokens smaller than --max-len")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must contain text")
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_dir():
        parser.error(f"Local model directory does not exist: {model_path}")
    if not torch.cuda.is_available():
        parser.error("This 3B runner requires a CUDA GPU")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer_load_s = time.perf_counter() - tokenizer_started
    model_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().requires_grad_(False).cuda()
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - model_started
    parameter_count, versions = validate_target(model)
    eos = eos_token_ids(model, tokenizer)
    runner = None
    runner_settings = None
    runner_started = time.perf_counter()
    if args.engine in ("exact", "bucketed"):
        if args.engine == "bucketed":
            from exact3b_dynamic import Bucketed3BRunner
            runner_type = Bucketed3BRunner
        else:
            from exact3b_runner import Exact3BRunner
            runner_type = Exact3BRunner
        runner_settings = {name: parameter.default for name, parameter in inspect.signature(runner_type).parameters.items()
                           if name != "model" and parameter.default is not inspect.Parameter.empty}
        runner_settings.update(max_len=args.max_len, max_output=max(2, args.tokens),
            norm_mode="triton", pointwise="compiled", gate_mode="compiled", fuse_cache=True,
            combine_projections=True, combine_mode="kv", share_pool=True,
            screen_head=not args.full_head, verifier_lanes=16)
        runner = runner_type(model, **runner_settings)
        torch.cuda.synchronize()
    runner_init_s = time.perf_counter() - runner_started
    environment = {"model_path": str(model_path), "engine": args.engine,
        "runner_class": f"{type(runner).__module__}.{type(runner).__qualname__}" if runner is not None else None,
        "runner_configuration": runner_settings,
        "parameter_count": parameter_count, "target_dtype": "torch.bfloat16",
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "tokenizer_load_s": tokenizer_load_s, "model_load_s": model_load_s, "runner_init_s": runner_init_s,
        "eos_token_ids": sorted(eos), "max_len": args.max_len,
        "generation_policy": "original chat template; raw greedy argmax; native EOS; no logits processors",
        "source_sha256": source_hashes()}
    session = {"environment": environment, "runs": []}

    def write_report():
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(session, ensure_ascii=False, indent=2))

    write_report()
    print(f"{args.engine} 3B engine ready; model load {model_load_s:.2f}s, runner initialization {runner_init_s:.2f}s. "
          "Graph preparation is reported separately.", file=sys.stderr, flush=True)
    while True:
        if args.prompt is not None:
            prompt = args.prompt
        else:
            try:
                print("prompt> ", file=sys.stderr, end="", flush=True)
                prompt = input()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt.strip():
                continue
        tokenization_started = time.perf_counter()
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        tokenization_s = time.perf_counter() - tokenization_started
        if ids.shape[1] + args.tokens > args.max_len:
            message = (f"Prompt has {ids.shape[1]} tokens; prompt plus {args.tokens} new tokens exceeds "
                       f"--max-len {args.max_len}. Increase capacity or explicitly change the output budget.")
            if args.prompt is not None:
                parser.error(message)
            print(message, file=sys.stderr, flush=True)
            continue
        if runner is None:
            tokens, metrics, prepared = run_original(model, ids, args.tokens, eos)
        else:
            tokens, metrics, prepared = run_exact(runner, ids, args.tokens, eos)
        if any(parameter._version != version for parameter, version in versions):
            raise RuntimeError("An original target parameter was modified")
        metrics.update({"tokenization_s": tokenization_s,
                        "request_including_preparation_s": tokenization_s + prepared["prepare_wall_s"] + metrics["generation_s"]})
        record = {"prompt": prompt, "prompt_token_ids": ids[0].cpu().tolist(),
            "prompt_tokens": ids.shape[1], "token_ids": tokens,
            "text": tokenizer.decode(tokens, skip_special_tokens=True),
            "raw_text": tokenizer.decode(tokens, skip_special_tokens=False),
            "generated_tokens": len(tokens), "max_new_tokens": args.tokens,
            "stop_reason": metrics["stop_reason"], "reached_eos": tokens[-1] in eos,
            "eos_token_ids": sorted(eos), "target_parameter_versions_unchanged": True,
            "setup": prepared, "metrics": metrics}
        if runner is not None and hasattr(runner, "memory_report"):
            record["memory"] = runner.memory_report()
        session["runs"].append(record)
        write_report()
        if args.json:
            print(json.dumps({"environment": environment, **record}, ensure_ascii=False), flush=True)
        else:
            print(record["text"], flush=True)
            print(json.dumps({key: value for key, value in record.items()
                              if key not in ("text", "raw_text", "prompt", "prompt_token_ids")}), flush=True)
        if args.prompt is not None:
            break


if __name__ == "__main__":
    main()
