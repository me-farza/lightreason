"""Paired same-target latency for the separately labeled FP-order variant.

Output differences are reported, not automatically rejected. This fixed128
timing workload has no answer grade; a separate quality report is required.
Original target weight/source identity remains a mandatory gate.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch

from benchmark import Reference, PROMPTS
from benchmark_3b import load
from exact10_validation import hash_model_parameters, project_python_hashes
from numerical3b_runner import Numerical3BRunner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="model-3b")
    p.add_argument("--draft-path", default="/tmp/lightreason-models/pard-qwen0.5b")
    p.add_argument("--block-width", type=int, default=8)
    p.add_argument("--proposal-mode", choices=["chain", "tree"], default="chain")
    p.add_argument("--node-budget", type=int, default=64)
    p.add_argument("--draft-tokens", type=int, default=15)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--tree-policy", choices=["best-first", "trunk"], default="best-first")
    p.add_argument("--compile-draft", action="store_true")
    p.add_argument("--cache-mode", choices=["dynamic", "static"], default="static")
    p.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile-static", action="store_true")
    p.add_argument("--draft-engine", choices=["hf", "graph"], default="graph")
    p.add_argument("--draft-attention", choices=["eager", "sdpa"], default="sdpa")
    p.add_argument("--share-pool", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--prompts", type=int, default=2)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--cooldown-s", type=float, default=8)
    p.add_argument("--out", default="results3b/numerical-pilot.json")
    args = p.parse_args()
    if not 1 <= args.prompts <= 6 or args.repeats < 1 or args.tokens < 2:
        p.error("Use prompts1..6, repeats>=1, tokens>=2")
    if args.cache_mode == "dynamic" and args.cuda_graph:
        p.error("Use --no-cuda-graph with DynamicCache")
    model, tokenizer, environment = load(args)
    before = hash_model_parameters(model)
    versions = [(value, value._version) for value in model.parameters()]
    environment["target_parameter_bytes_sha256_before"] = before
    started = time.perf_counter()
    if args.proposal_mode == "tree":
        from numerical3b_tree_runner import Numerical3BTreeRunner
        candidate = Numerical3BTreeRunner(model, tokenizer, args.draft_path,
            max_len=args.max_len, node_budget=args.node_budget,
            draft_tokens=args.draft_tokens, top_k=args.top_k,
            tree_policy=args.tree_policy,
            compile_static=args.compile_static, compile_draft=args.compile_draft,
            cuda_graph=args.cuda_graph, share_pool=args.share_pool)
    else:
        candidate = Numerical3BRunner(model, tokenizer, args.draft_path,
            max_len=args.max_len, block_width=args.block_width,
            cache_mode=args.cache_mode, cuda_graph=args.cuda_graph,
            compile_static=args.compile_static, draft_engine=args.draft_engine,
            draft_attention=args.draft_attention, share_pool=args.share_pool)
    torch.cuda.synchronize()
    environment["runner_constructor_s"] = time.perf_counter() - started
    environment["candidate"] = candidate.metadata
    environment["timing_scope"] = "Warmed paired CUDA-synchronized wall latency; original target prefill and all auxiliary drafting/cache/verification/CPU acceptance costs included. Loading/setup and final CPU token transfer excluded. Fixed budget, batch size one; not serving throughput."
    environment["quality_claim"] = "Not graded here; answer-quality evaluation is separate"
    reference = Reference(model)
    rows = []
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(PROMPTS[:args.prompts]):
        if index and args.cooldown_s:
            time.sleep(args.cooldown_s)
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt").to(model.device)
        reference.run(ids, args.tokens)
        candidate.run(ids, args.tokens)
        for repeat in range(args.repeats):
            order = [("baseline", reference), ("candidate", candidate)]
            if repeat % 2:
                order.reverse()
            values = {name: obj.run(ids, args.tokens) for name, obj in order}
            b_tokens, baseline = values["baseline"]
            c_tokens, optimized = values["candidate"]
            if not all(math.isfinite(metrics[key]) and metrics[key] > 0
                       for metrics in (baseline, optimized) for key in ("total_s", "decode_s")):
                raise RuntimeError("Invalid synchronized inference timing")
            different = next((i for i, pair in enumerate(zip(b_tokens, c_tokens)) if pair[0] != pair[1]), None)
            if different is None and len(b_tokens) != len(c_tokens):
                different = min(len(b_tokens), len(c_tokens))
            row = {"prompt_index": index, "repeat": repeat,
                "baseline": baseline, "candidate": optimized,
                "baseline_tokens": b_tokens, "candidate_tokens": c_tokens,
                "complete_tokens_exact": b_tokens == c_tokens,
                "first_token_difference": different,
                "aligned_token_matches": sum(a == b for a, b in zip(b_tokens, c_tokens)),
                "total_speedup": baseline["total_s"] / optimized["total_s"],
                "decode_speedup": baseline["decode_s"] / optimized["decode_s"]}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if not k.endswith("tokens")}), flush=True)
            destination.write_text(json.dumps({"environment": environment, "trials": rows}, indent=2))
    after = hash_model_parameters(model)
    environment.update(target_parameter_bytes_sha256_after=after,
        target_parameter_versions_unchanged=all(value._version == version for value, version in versions),
        source_sha256_after=project_python_hashes(Path(__file__).parent))
    environment["target_parameters_unchanged"] = before == after and environment["target_parameter_versions_unchanged"]
    environment["project_python_sources_unchanged"] = environment["source_sha256_after"] == environment["source_sha256"]
    summary = {"trials": len(rows), "complete_exact_sequences": sum(r["complete_tokens_exact"] for r in rows),
        "aligned_token_agreement": sum(r["aligned_token_matches"] for r in rows) / (len(rows) * args.tokens),
        "median_total_speedup": statistics.median(r["total_speedup"] for r in rows),
        "minimum_total_speedup": min(r["total_speedup"] for r in rows),
        "median_decode_speedup": statistics.median(r["decode_speedup"] for r in rows),
        "baseline_median_decode_tokens_s": statistics.median(r["baseline"]["decode_tokens_s"] for r in rows),
        "candidate_median_decode_tokens_s": statistics.median(r["candidate"]["decode_tokens_s"] for r in rows),
        "mean_committed_tokens_per_cycle": statistics.mean(r["candidate"]["mean_committed_tokens_per_cycle"] for r in rows),
        "micro_committed_tokens_per_cycle": sum(len(r["candidate_tokens"]) - 1 for r in rows) / sum(r["candidate"]["cycles"] for r in rows),
        "excluded_graph_setup_s": candidate.setup_s,
        "quality_degradation_assessed": False,
        "exact_output_differences_are_not_automatic_rejection": True}
    summary["full_30_trial_protocol"] = args.prompts == 6 and args.tokens == 128 and args.repeats >= 5 and len(rows) == 6 * args.repeats
    summary["ten_x_timing_demonstrated"] = summary["full_30_trial_protocol"] and summary["minimum_total_speedup"] >= 10
    destination.write_text(json.dumps({"environment": environment, "summary": summary, "trials": rows}, indent=2))
    print(json.dumps({"summary": summary}), flush=True)
    if not environment["target_parameters_unchanged"] or not environment["project_python_sources_unchanged"]:
        raise RuntimeError("Original target weights or source changed during timing")


if __name__ == "__main__":
    main()
