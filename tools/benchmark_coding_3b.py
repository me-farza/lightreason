"""Time complete greedy coding responses with the public 3B checkpoint."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from benchmark_3b import load
from exact3b_dynamic import Bucketed3BRunner
from exact10_validation import hash_model_parameters
from run_model_3b import eos_token_ids, run_exact, run_original, validate_target


ROOT = Path(__file__).resolve().parents[1]
CAP = 768
CAPACITY = 1024
PROMPTS = {
    "python": (
        "Write a Python function stable_dedupe(words) that returns each distinct input string once, "
        "in its first-appearance order. Matching is case-sensitive. Return an empty list for empty input "
        "and do not modify the input. Give the complete function in one python code block and a brief "
        "explanation. Use only Python built-ins, with no imports."
    ),
    "javascript": (
        "Write a JavaScript function debounce(fn, delayMs) that returns a trailing-edge debounced "
        "wrapper. The wrapper must preserve this and the latest arguments, expose cancel(), and use "
        "only standard timers. Provide a complete function and a short usage example."
    ),
    "sql": (
        "Write one SQLite query for orders(customer_id, order_date, amount). For 2025, return the "
        "top 5 customers by sum of amount, highest total first, with ties broken by customer_id "
        "ascending. Exclude customers with fewer than three 2025 orders. Use a half-open date range "
        "and explain why it is useful."
    ),
}


def benchmark_case(model, tokenizer, runner, eos, case, prompt, repeats, reverse_first):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    if ids.shape[1] + CAP > CAPACITY:
        raise ValueError(f"{case} exceeds the fixed context capacity")
    engines = {"baseline": (run_original, model), "candidate": (run_exact, runner)}
    warmups = {}
    for name, (generate, runtime) in engines.items():
        _, metrics, preparation = generate(runtime, ids, CAP, eos)
        warmups[name] = {"generation_s": metrics["generation_s"], "preparation": preparation}
    rows = []
    for repeat in range(repeats):
        order = list(engines)
        if (repeat + reverse_first) % 2:
            order.reverse()
        runs = {}
        for name in order:
            generate, runtime = engines[name]
            tokens, metrics, preparation = generate(runtime, ids, CAP, eos)
            runs[name] = {
                **metrics, "preparation": preparation, "token_ids": tokens,
                "text": tokenizer.decode(tokens, skip_special_tokens=True),
            }
        left, right = runs["baseline"], runs["candidate"]
        rows.append({"repeat": repeat, "order": order, "runs": runs, "exact": (
            left["token_ids"], left["stop_reason"]
        ) == (right["token_ids"], right["stop_reason"])})
    medians = {
        name: {field: median(row["runs"][name][field] for row in rows)
               for field in ("prefill_s", "generation_s", "decode_tokens_s")}
        for name in engines
    }
    return {"case": case, "prompt": prompt, "input_ids": ids[0].tolist(),
            "warmups": warmups, "trials": rows, "medians": medians}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="model-3b")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2 or args.out.exists():
        parser.error("Use at least two repeats and a fresh output path")
    model, tokenizer, environment = load(args)
    plan = json.loads((ROOT / "results3b/download-plan.json").read_text())
    shard_hashes = {entry["rfilename"]: entry["lfs"]["sha256"]
                    for entry in plan["files"] if entry["rfilename"].endswith(".safetensors")}
    if environment["checkpoint_files"] != shard_hashes:
        raise RuntimeError("Checkpoint shard hashes differ from the pinned Qwen download")
    _, versions = validate_target(model)
    target_hash = hash_model_parameters(model)
    eos = eos_token_ids(model, tokenizer)
    started = perf_counter()
    runner = Bucketed3BRunner(model, CAPACITY, CAP, norm_mode="triton", share_pool=True,
                              fuse_cache=True, screen_head=True, combine_projections=True)
    torch.cuda.synchronize()
    setup_s = perf_counter() - started
    rows = [benchmark_case(model, tokenizer, runner, eos, case, prompt, args.repeats, index % 2)
            for index, (case, prompt) in enumerate(PROMPTS.items())]
    target_unchanged = (hash_model_parameters(model) == target_hash
                        and all(parameter._version == version for parameter, version in versions))
    exact = all(row["exact"] for case in rows for row in case["trials"])
    reached_eos = all(run["stop_reason"] == "eos"
                      for case in rows for trial in case["trials"] for run in trial["runs"].values())
    report = {
        "environment": {key: value for key, value in environment.items() if key != "arguments"},
        "protocol": {"repeats": args.repeats, "batch_size": 1, "max_new_tokens": CAP,
                     "max_len": CAPACITY, "eos_ids": sorted(eos), "policy": "raw greedy argmax",
                     "timing": "warm non-streaming generation, including prefill and host EOS checks"},
        "runner_setup_s": setup_s, "target_sha256": target_hash, "target_unchanged": target_unchanged,
        "all_pairs_exact": exact, "all_runs_reached_eos": reached_eos, "cases": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    for case in rows:
        print(case["case"], case["medians"])
    if not (target_unchanged and exact and reached_eos):
        raise SystemExit("A checkpoint, exact-output, or native-EOS check failed")


if __name__ == "__main__":
    main()
