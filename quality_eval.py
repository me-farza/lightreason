"""Natural-EOS correctness and strict target-token checks; not a speed benchmark.

Create reusable BF16 greedy target outputs:
  python quality_eval.py --reference-only --out results/quality-reference.json
Compare the compiled engine, reusing those exact target outputs:
  python quality_eval.py --reference-results results/quality-reference.json
Future runners can use --candidate-factory module:function. The factory receives
model, tokenizer, args and returns callable(prompt, max_new_tokens), producing a
Generation or a dict with the same fields. It must use the supplied target model
or verify its own equivalent target weights; that identity is not proved by this
interface. The generic evaluate() and scoring functions need only stdlib Python.

The strict gate requires identical token IDs, including EOS, plus the same stop
reason. Answer correctness is a supplemental diagnostic, not a substitute for
token identity. These public, hand-written tasks are not a held-out evaluation.
"""
import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import time


@dataclass
class Generation:
    text: str
    token_ids: list | None = None
    stop_reason: str = "unknown"
    metrics: dict = field(default_factory=dict)


def as_generation(value):
    if isinstance(value, dict):
        value = Generation(**value)
    if not isinstance(value, Generation):
        raise TypeError("Generator must return Generation or a dict with its fields")
    if not isinstance(value.text, str):
        raise TypeError("Generation.text must be a string")
    if value.token_ids is not None and (not isinstance(value.token_ids, list)
            or not value.token_ids or any(type(token) is not int for token in value.token_ids)):
        raise TypeError("token_ids must be a nonempty list of integer IDs, or None")
    return value


def load_tasks(path):
    data = json.loads(Path(path).read_text())
    tasks = data["tasks"]
    if not tasks or len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("Tasks must have unique IDs and be nonempty")
    for task in tasks:
        if not all(k in task for k in ("id", "category", "prompt", "expected")):
            raise ValueError("Each task needs id, category, prompt, and expected")
        task["full_prompt"] = task["prompt"] + "\n\n" + data["response_instruction"]
        task["max_new_tokens"] = task.get("max_new_tokens", data["max_new_tokens"])
        if task["max_new_tokens"] < 1:
            raise ValueError("max_new_tokens must be positive")
    return data


def parse_answer(text):
    """Strict final-line format; never extract an earlier plausible number."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines or not re.match(r"^FINAL\s*:", lines[-1], re.I):
        return {"format_valid": False, "parsed": None, "error": "Missing final FINAL: JSON line"}
    raw = re.sub(r"^FINAL\s*:\s*", "", lines[-1], count=1, flags=re.I)
    try:
        value = json.loads(raw, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    except (ValueError, json.JSONDecodeError) as exc:
        return {"format_valid": False, "parsed": None, "error": f"Invalid final JSON: {exc}"}
    return {"format_valid": True, "parsed": value, "error": None}


def answer_equal(actual, expected):
    """Normalize string case/whitespace; distinguish booleans from numbers."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(expected, str) and isinstance(actual, str):
        return " ".join(actual.split()).casefold() == " ".join(expected.split()).casefold()
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(answer_equal(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, dict) and isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(answer_equal(actual[k], expected[k]) for k in expected)
    return type(actual) is type(expected) and actual == expected


def score(text, expected):
    parsed = parse_answer(text)
    return {**parsed, "correct": parsed["format_valid"] and answer_equal(parsed["parsed"], expected)}


def summarize(rows, names):
    summary = {}
    for name in names:
        available = [r for r in rows if name in r["generations"]]
        successful = [r for r in available if "error" not in r["generations"][name]]
        correct = sum(r["scores"][name]["correct"] for r in available)
        summary[name] = {
            "tasks": len(available), "generation_errors": len(available) - len(successful),
            "correct": correct, "accuracy": correct / len(available) if available else None,
            "format_valid": sum(r["scores"][name]["format_valid"] for r in available),
            "natural_eos": sum(r["generations"][name].get("stop_reason") == "eos" for r in available),
            "by_category": {},
        }
        for category in sorted({r["category"] for r in available}):
            category_rows = [r for r in available if r["category"] == category]
            ncorrect = sum(r["scores"][name]["correct"] for r in category_rows)
            summary[name]["by_category"][category] = {"correct": ncorrect, "tasks": len(category_rows)}
        if name == "baseline":
            continue
        paired = [r for r in available if "baseline" in r["generations"]]
        regression = [r["id"] for r in paired if r["scores"]["baseline"]["correct"] and not r["scores"][name]["correct"]]
        gains = [r["id"] for r in paired if not r["scores"]["baseline"]["correct"] and r["scores"][name]["correct"]]
        exact = [r for r in paired if r["comparisons"][name]["token_sequence_exact"] is True]
        missing_ids = [r["id"] for r in paired if r["comparisons"][name]["token_sequence_exact"] is None]
        # A model that exactly reproduces a truncated baseline has preserved that
        # output, but has not demonstrated a complete naturally terminated answer.
        output_preserved = bool(paired) and len(exact) == len(paired) and all(
            r["comparisons"][name]["text_exact"] for r in paired)
        natural = bool(paired) and all(
            r["generations"]["baseline"].get("stop_reason") == "eos"
            and r["generations"][name].get("stop_reason") == "eos" for r in paired)
        summary[name].update({
            "paired_tasks": len(paired), "baseline_correct_regressions": regression,
            "baseline_incorrect_gains": gains, "exact_token_sequences": len(exact),
            "missing_token_ids": missing_ids,
            "strict_target_output_preserved": output_preserved,
            "no_baseline_correct_regressions": not regression,
            "all_paired_answers_naturally_terminated": natural,
            "strict_quality_gate_passed": output_preserved and not regression and natural,
        })
    return summary


def evaluate(tasks, generators, on_progress=None):
    """Evaluate named callables, including baseline, without assuming a backend.

    Each callable accepts (full_prompt, max_new_tokens). Per-task errors remain
    visible and fail strict preservation. This function does not time throughput.
    """
    if "baseline" not in generators:
        raise ValueError("A baseline generator is required")
    rows = []
    for index, task in enumerate(tasks):
        row = {"id": task["id"], "category": task["category"], "prompt": task["full_prompt"],
               "max_new_tokens": task["max_new_tokens"], "expected": task["expected"],
               "generations": {}, "scores": {}, "comparisons": {}}
        names = list(generators)
        if index % 2:
            names.reverse()
        for name in names:
            try:
                generated = as_generation(generators[name](task["full_prompt"], task["max_new_tokens"]))
                row["generations"][name] = asdict(generated)
                row["scores"][name] = score(generated.text, task["expected"])
            except Exception as exc:
                row["generations"][name] = {"error": f"{type(exc).__name__}: {exc}"}
                row["scores"][name] = {"correct": False, "format_valid": False, "parsed": None, "error": str(exc)}
        baseline = row["generations"]["baseline"]
        for name in generators:
            if name == "baseline":
                continue
            candidate = row["generations"][name]
            b_ids, c_ids = baseline.get("token_ids"), candidate.get("token_ids")
            exact = None if b_ids is None or c_ids is None else (
                b_ids == c_ids and baseline.get("stop_reason") == candidate.get("stop_reason"))
            first_diff = None
            if b_ids is not None and c_ids is not None:
                first_diff = next((i for i, (a, b) in enumerate(zip(b_ids, c_ids)) if a != b), None)
                if first_diff is None and len(b_ids) != len(c_ids):
                    first_diff = min(len(b_ids), len(c_ids))
            row["comparisons"][name] = {
                "token_sequence_exact": exact, "first_token_difference_zero_based": first_diff,
                "text_exact": baseline.get("text") == candidate.get("text") and "text" in baseline,
                "answer_correctness_equal": row["scores"][name]["correct"] == row["scores"]["baseline"]["correct"],
            }
        rows.append(row)
        if on_progress:
            on_progress(rows, summarize(rows, list(generators)))
    return {"summary": summarize(rows, list(generators)), "tasks": rows}


class CachedReference:
    def __init__(self, result, identity, tasks=None):
        if result["environment"]["identity"] != identity:
            raise ValueError("Cached reference identity does not match tasks/checkpoint/tokenizer/generation settings")
        self.outputs = {}
        for row in result["tasks"]:
            generation = row["generations"]["baseline"]
            if "error" in generation or generation.get("token_ids") is None:
                raise ValueError("Cached baseline contains errors or lacks target token IDs")
            self.outputs[(row["prompt"], row["max_new_tokens"])] = generation
        if tasks is not None and set(self.outputs) != {
                (task["full_prompt"], task["max_new_tokens"]) for task in tasks}:
            raise ValueError("Cached baseline is incomplete or contains a different set of tasks")

    def __call__(self, prompt, max_new_tokens):
        return self.outputs[(prompt, max_new_tokens)]


class NaturalEOSRunner:
    """Adapter around Reference.model or GraphRunner; actual EOS stopping.

    Copying each selected token to the CPU checks EOS immediately. Its latency
    includes these synchronizations and must not substantiate a speedup claim.
    """
    def __init__(self, runner, tokenizer, eos_ids, max_len, graph=False):
        self.runner, self.tokenizer = runner, tokenizer
        self.eos_ids, self.max_len, self.graph = set(eos_ids), max_len, graph

    def __call__(self, prompt, max_new_tokens):
        import torch
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            return_tensors="pt").cuda()
        if ids.shape[1] + max_new_tokens > self.max_len:
            raise ValueError(f"Prompt ({ids.shape[1]}) + budget ({max_new_tokens}) exceeds max_len={self.max_len}")
        with torch.inference_mode():
            torch.cuda.synchronize()
            start = time.perf_counter()
            if self.graph:
                self.runner.prefill(ids)
                token = self.runner.token
            else:
                output = self.runner.model(ids, use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                token = output.logits[:, -1].argmax(-1, keepdim=True)
            tokens = []
            for index in range(max_new_tokens):
                token_id = int(token.item())
                tokens.append(token_id)
                if token_id in self.eos_ids:
                    break
                if index + 1 == max_new_tokens:
                    break
                if self.graph:
                    self.runner.graph.replay()
                    token = self.runner.token
                else:
                    output = self.runner.model(token, past_key_values=cache, use_cache=True, logits_to_keep=1)
                    token = output.logits[:, -1].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        return Generation(
            text=self.tokenizer.decode(tokens, skip_special_tokens=True), token_ids=tokens,
            stop_reason="eos" if tokens[-1] in self.eos_ids else "max_new_tokens",
            metrics={"quality_harness_latency_s": elapsed, "input_tokens": ids.shape[1],
                     "generated_tokens_including_eos": len(tokens), "not_a_speed_benchmark": True})


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def identity_for(model_path, task_path, tokenizer, eos_ids):
    model_path = Path(model_path)
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise ValueError("Expected local safetensors weights; no download is performed")
    configuration = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                     "vocab.json", "merges.txt", "special_tokens_map.json"]
    files += [model_path / name for name in configuration if (model_path / name).is_file()]
    return {"task_file_sha256": sha256_file(task_path), "model_and_tokenizer_files_sha256": {
                path.name: sha256_file(path) for path in files},
            "chat_template": tokenizer.chat_template, "dtype": "torch.bfloat16",
            "decoding": "greedy argmax, no logits processors, stop on first EOS",
            "eos_token_ids": sorted(eos_ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(Path(__file__).with_name("quality_tasks.json")))
    parser.add_argument("--model-path", default="model")
    parser.add_argument("--mode", choices=["hfgraph", "custom", "compiled"], default="compiled")
    parser.add_argument("--attention", choices=["sdpa", "matmul"], default="sdpa")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--reference-results")
    parser.add_argument("--candidate-factory", help="module:function factory returning a generation callable")
    parser.add_argument("--out", default="results/quality.json")
    args = parser.parse_args()
    if args.reference_only and (args.reference_results or args.candidate_factory):
        parser.error("--reference-only cannot be combined with cached reference or candidate factory")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from benchmark import Reference, GraphRunner
    from optimized_qwen import OptimizedQwen
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = load_tasks(args.tasks)
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().requires_grad_(False).cuda()
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
    if tok.eos_token_id is not None:
        eos_ids.append(tok.eos_token_id)
    eos_ids = sorted(set(eos_ids))
    if not eos_ids:
        raise ValueError("No EOS token configured; natural stopping cannot be evaluated")
    identity = identity_for(args.model_path, args.tasks, tok, eos_ids)
    identity["reference_runtime"] = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "evaluator_sha256": sha256_file(__file__),
        "attention": "sdpa", "cache": "DynamicCache", "allow_tf32": False,
    }
    environment = {
        "identity": identity, "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(), "mode": args.mode, "attention": args.attention,
        "max_len": args.max_len, "candidate_factory": args.candidate_factory,
        "reference_results": args.reference_results,
        "scope": data["description"],
        "timing_warning": "EOS checks synchronize every token; quality-harness latencies are not speed benchmarks",
        "strict_gate": "All target token IDs (including EOS) and stop reasons must match; no baseline-correct answer may regress",
    }
    generators = {"baseline": NaturalEOSRunner(Reference(model), tok, eos_ids, args.max_len)}
    if args.reference_results:
        cached = json.loads(Path(args.reference_results).read_text())
        generators["baseline"] = CachedReference(cached, identity, data["tasks"])
    if not args.reference_only:
        if args.candidate_factory:
            module_name, function_name = args.candidate_factory.rsplit(":", 1)
            factory = getattr(importlib.import_module(module_name), function_name)
            generators["candidate"] = factory(model=model, tokenizer=tok, args=args)
        else:
            custom = args.mode != "hfgraph"
            engine = OptimizedQwen(model, args.max_len, attention=args.attention) if custom else model
            output_capacity = max(task["max_new_tokens"] for task in data["tasks"]) + 16
            with torch.inference_mode():
                runner = GraphRunner(engine, args.max_len, output_capacity,
                    custom=custom, compile_decode=args.mode == "compiled")
            generators["candidate"] = NaturalEOSRunner(runner, tok, eos_ids, args.max_len, graph=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    def progress(rows, summary):
        out.write_text(json.dumps({"environment": environment, "summary": summary, "tasks": rows}, indent=2))
        print(json.dumps({"completed": len(rows), "task": rows[-1]["id"], "summary": summary}), flush=True)
    result = evaluate(data["tasks"], generators, on_progress=progress)
    out.write_text(json.dumps({"environment": environment, **result}, indent=2))
    print(json.dumps({"output": str(out), "summary": result["summary"]}), flush=True)


if __name__ == "__main__":
    main()
