"""Pinned3B same-checkpoint paired latency and original-kernel inventory.

No speedup is accepted unless every complete timed output matches the live
unchanged HF baseline. Setup and cold loading are reported separately.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import Reference, PROMPTS
from exact10_validation import hash_model_parameters, project_python_hashes


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def load(args):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().requires_grad_(False).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    torch.cuda.synchronize()
    env = {"model_id": "Qwen/Qwen2.5-3B-Instruct", "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
           "parameters": sum(p.numel() for p in model.parameters()),
           "dtype": str(next(model.parameters()).dtype), "torch": torch.__version__,
           "transformers": __import__("transformers").__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(), "model_load_s": time.perf_counter() - start,
           "checkpoint_files": {p.name: digest(p) for p in sorted(Path(args.model_path).glob("*.safetensors"))},
           "arguments": vars(args),
           "source_sha256": project_python_hashes(Path(__file__).parent)}
    return model, tokenizer, env


def kernels(profiler, path):
    profiler.export_chrome_trace(str(path))
    return [{"name": e["name"], "dur": e.get("dur"), "args": e.get("args")}
            for e in json.loads(path.read_text())["traceEvents"] if e.get("cat") == "kernel"]


@torch.no_grad()
def profile(args, model, tokenizer, env):
    from exact_graph import ExactGraphRunner
    root = Path(args.out).parent
    ids = tokenizer.apply_chat_template([{"role": "user", "content": PROMPTS[0]}],
        add_generation_prompt=True, return_tensors="pt").cuda()
    baseline_tokens, latency = Reference(model).run(ids, 128)
    runner = ExactGraphRunner(model, 1024, 512)
    runner.prepare(ids, 2)
    runner.prefill(ids)
    runner.cache.set_position(ids.shape[1])
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as p:
        runner._step()
        torch.cuda.synchronize()
    events = kernels(p, root / "original-step.trace.json")
    grouped = defaultdict(lambda: {"calls": 0, "total_us": 0.0})
    for e in events:
        grouped[e["name"]]["calls"] += 1
        grouped[e["name"]]["total_us"] += e["dur"] or 0
    layer = model.model.layers[0]
    modules = {"q": layer.self_attn.q_proj, "k": layer.self_attn.k_proj,
               "v": layer.self_attn.v_proj, "o": layer.self_attn.o_proj,
               "gate": layer.mlp.gate_proj, "up": layer.mlp.up_proj,
               "down": layer.mlp.down_proj, "head": model.lm_head}
    inventory = {}
    for name, linear in modules.items():
        x = torch.randn((1, 1, linear.weight.shape[1]), device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            F.linear(x, linear.weight, linear.bias)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=activities, record_shapes=True) as p:
            F.linear(x, linear.weight, linear.bias)
            torch.cuda.synchronize()
        inventory[name] = {"shape": list(linear.weight.shape),
            "kernels": kernels(p, root / ("projection-" + name + ".trace.json"))}
    mean = torch.randn((1, 1, model.config.hidden_size), device="cuda", dtype=torch.float32).square()
    with torch.profiler.profile(activities=activities, record_shapes=True) as p:
        mean.mean(-1, keepdim=True)
        torch.cuda.synchronize()
    inventory["norm_mean"] = {"input_shape": list(mean.shape),
        "kernels": kernels(p, root / "norm-mean.trace.json")}
    result = {"environment": env, "baseline_diagnostic": latency, "baseline_tokens": baseline_tokens,
              "kernel_count": len(events), "kernel_time_sum_us": sum(e["dur"] or 0 for e in events),
              "kernel_groups": sorted(({"name": k, **v} for k, v in grouped.items()),
                                      key=lambda e: e["total_us"], reverse=True),
              "projection_inventory": inventory,
              "profile_scope": "Instrumented original step and isolated kernels, not speedup evidence"}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({"baseline_diagnostic": latency, "kernel_count": len(events),
                      "inventory": inventory}), flush=True)


@torch.no_grad()
def timing(args, model, tokenizer, env):
    # These untimed checks also bind the runner's construction to the original
    # live weights; checkpoint-file hashes alone cannot establish that.
    parameter_hash_before = hash_model_parameters(model)
    versions = [(parameter, parameter._version) for parameter in model.parameters()]
    env["target_parameter_bytes_sha256_before"] = parameter_hash_before
    if args.engine == "original-graphs":
        from exact3b_graphs import LinearExactGraphRunner
        runner = LinearExactGraphRunner(model, args.max_len, args.tokens + 16,
                                       share_pool=args.share_pool)
    else:
        if args.engine == "bucketed":
            from exact3b_dynamic import Bucketed3BRunner
            runner_type = Bucketed3BRunner
        else:
            from exact3b_runner import Exact3BRunner
            runner_type = Exact3BRunner
        runner = runner_type(model, args.max_len, args.tokens + 16,
            norm_mode=args.norm_mode, pointwise=args.pointwise, fuse_cache=args.fuse_cache,
            screen_head=args.screen_head, verifier_lanes=args.verifier_lanes,
            share_pool=args.share_pool, combine_projections=args.combine_projections,
            combine_mode=args.combine_mode)
    reference = Reference(model)
    env["timing"] = "Paired warmed CUDA-synchronized wall latency, alternating order; original prefill+cache transfer included; loading, compile/graph preparation and CPU output transfer excluded. Not throughput batching."
    env["head_index_setup_s"] = getattr(runner, "head_index_setup_s", 0.0)
    env["projection_copy_setup_s"] = getattr(getattr(runner, "decoder", None), "projection_copy_setup_s", 0.0)
    env["extra_projection_bytes"] = getattr(getattr(runner, "decoder", None), "extra_projection_bytes", 0)
    rows = []
    for pi, prompt in enumerate(PROMPTS[:args.prompts]):
        if args.cooldown_s and pi:
            # Outside all measured regions; retain stock hardware settings and
            # the same thermal watchdog for both sides of each pair.
            time.sleep(args.cooldown_s)
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt").cuda()
        reference.run(ids, args.tokens)
        runner.run(ids, args.tokens)
        for repeat in range(args.repeats):
            order = [("baseline", reference), ("candidate", runner)]
            if repeat % 2:
                order.reverse()
            values = {name: obj.run(ids, args.tokens) for name, obj in order}
            baseline, b = values["baseline"]
            candidate, c = values["candidate"]
            if not all(math.isfinite(metrics[key]) and metrics[key] > 0
                       for metrics in (b, c) for key in ("total_s", "decode_s")):
                raise RuntimeError("Invalid synchronized inference timing")
            difference = next((i for i, pair in enumerate(zip(baseline, candidate))
                               if pair[0] != pair[1]), None)
            if difference is None and len(baseline) != len(candidate):
                difference = min(len(baseline), len(candidate))
            row = {"prompt_index": pi, "repeat": repeat, "baseline": b, "candidate": c,
                "baseline_tokens": baseline, "candidate_tokens": candidate,
                "tokens_exact": baseline == candidate,
                "first_difference": difference,
                "total_speedup": b["total_s"] / c["total_s"],
                "decode_speedup": b["decode_s"] / c["decode_s"]}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if not k.endswith("tokens")}), flush=True)
            Path(args.out).write_text(json.dumps({"environment": env, "trials": rows}, indent=2))
    summary = {"trials": len(rows), "all_tokens_exact": all(r["tokens_exact"] for r in rows),
        "median_total_speedup": statistics.median(r["total_speedup"] for r in rows),
        "minimum_total_speedup": min(r["total_speedup"] for r in rows),
        "median_decode_speedup": statistics.median(r["decode_speedup"] for r in rows),
        "baseline_median_decode_tokens_s": statistics.median(r["baseline"]["decode_tokens_s"] for r in rows),
        "candidate_median_decode_tokens_s": statistics.median(r["candidate"]["decode_tokens_s"] for r in rows),
        "graph_setup_s": runner.setup_s, "kv_allocated_bytes": runner.cache.allocated_bytes()}
    env["target_parameter_bytes_sha256_after"] = hash_model_parameters(model)
    env["target_parameter_versions_unchanged"] = all(p._version == version for p, version in versions)
    env["target_parameters_unchanged"] = (
        env["target_parameter_bytes_sha256_after"] == parameter_hash_before
        and env["target_parameter_versions_unchanged"])
    env["source_sha256_after"] = project_python_hashes(Path(__file__).parent)
    env["project_python_sources_unchanged"] = env["source_sha256_after"] == env["source_sha256"]
    summary["full_timing_protocol_passed"] = (
        args.prompts == 6 and args.tokens == 128 and args.repeats >= 5
        and len(rows) == 6 * args.repeats
        and all(len(r["baseline_tokens"]) == len(r["candidate_tokens"]) == 128 for r in rows)
        and env["target_parameters_unchanged"] and env["project_python_sources_unchanged"])
    summary["ten_x_and_timed_outputs_exact"] = (
        summary["full_timing_protocol_passed"] and summary["all_tokens_exact"]
        and summary["minimum_total_speedup"] >= 10)
    Path(args.out).write_text(json.dumps({"environment": env, "summary": summary, "trials": rows}, indent=2))
    print(json.dumps({"summary": summary}), flush=True)
    if not (summary["all_tokens_exact"] and env["target_parameters_unchanged"]
            and env["project_python_sources_unchanged"]):
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["profile", "timing"], default="timing")
    p.add_argument("--model-path", default="model-3b")
    p.add_argument("--engine", choices=["original-graphs", "fused", "bucketed"], default="fused")
    p.add_argument("--norm-mode", choices=["eager", "split", "triton"], default="split")
    p.add_argument("--pointwise", choices=["eager", "compiled"], default="compiled")
    p.add_argument("--fuse-cache", action="store_true")
    p.add_argument("--screen-head", action="store_true")
    p.add_argument("--verifier-lanes", type=int, default=16)
    p.add_argument("--combine-projections", action="store_true")
    p.add_argument("--combine-mode", choices=["kv", "qkv"], default="kv")
    p.add_argument("--share-pool", action="store_true")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--prompts", type=int, default=2)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--cooldown-s", type=float, default=0.0,
                   help="Untimed pause between prompt groups")
    p.add_argument("--out", default="results3b/pilot.json")
    args = p.parse_args()
    if not 1 <= args.prompts <= 6 or args.tokens < 2 or args.repeats < 1 or not 0 <= args.cooldown_s <= 30:
        p.error("Require prompts1..6,tokens>=2,repeats>=1")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer, env = load(args)
    (profile if args.mode == "profile" else timing)(args, model, tokenizer, env)


if __name__ == "__main__":
    main()
