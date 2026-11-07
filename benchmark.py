"""Same-checkpoint BF16 single-request inference experiments on an RTX 4090."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    "Explain how a rainbow forms, step by step, in clear language.",
    "Write a Python function that returns the first n Fibonacci numbers and explain it.",
    "Give a practical plan for growing tomatoes on a sunny apartment balcony.",
    "Explain the difference between RAM and an SSD using an everyday analogy.",
    "A train travels 120 kilometers in 90 minutes. Explain how to calculate its average speed.",
    "Write a short story about a curious robot that discovers a garden.",
]


def sync():
    torch.cuda.synchronize()


class Reference:
    def __init__(self, model):
        self.model = model

    @torch.inference_mode()
    def run(self, ids, n):
        sync()
        start = time.perf_counter()
        out = self.model(ids, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        token = out.logits[:, -1].argmax(-1, keepdim=True)
        tokens = [token]
        sync()
        prefill_end = time.perf_counter()
        for _ in range(n - 1):
            out = self.model(token, past_key_values=cache, use_cache=True, logits_to_keep=1)
            token = out.logits[:, -1].argmax(-1, keepdim=True)
            tokens.append(token)
        sync()
        end = time.perf_counter()
        return torch.cat(tokens, dim=1).cpu().tolist()[0], {
            "prefill_s": prefill_end - start,
            "decode_s": end - prefill_end,
            "total_s": end - start,
            "decode_tokens_s": (n - 1) / (end - prefill_end),
        }


class GraphRunner:
    def __init__(self, model, max_len, max_output, custom=False, compile_decode=False):
        self.custom = custom
        self.model = model
        self.max_len = max_len
        self.token = torch.zeros((1, 1), device="cuda", dtype=torch.long)
        self.pos = torch.ones((1,), device="cuda", dtype=torch.long)
        self.idx = torch.ones((1,), device="cuda", dtype=torch.long)
        self.output = torch.zeros((max_output,), device="cuda", dtype=torch.long)
        if not custom:
            self.cache = StaticCache(config=model.config, max_cache_len=max_len)
            self.keys = torch.arange(max_len, device="cuda").view(1, 1, 1, -1)
        self.decode = model.decode if custom else self.hf_decode
        self.compile_s = 0.0
        if compile_decode:
            t = time.perf_counter()
            self.decode = torch.compile(self.decode, fullgraph=True, mode="default")
            self.compile_s = time.perf_counter() - t
        dummy = torch.tensor([[151644, 8948, 198, 9707]], device="cuda")
        self.prefill(dummy)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        t = time.perf_counter()
        with torch.cuda.stream(side):
            for _ in range(3):
                self.step()
        torch.cuda.current_stream().wait_stream(side)
        sync()
        self.compile_s += time.perf_counter() - t
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.step()
        sync()

    def hf_decode(self, token, pos):
        return self.model(
            token, past_key_values=self.cache, use_cache=True,
            cache_position=pos, position_ids=pos.view(1, 1),
            attention_mask=self.keys <= pos.view(1, 1, 1, 1),
            logits_to_keep=1,
        ).logits[:, -1]

    def step(self):
        logits = self.decode(self.token, self.pos)
        self.last_logits = logits
        new_token = logits.argmax(-1)
        self.output.index_copy_(0, self.idx, new_token)
        self.token.copy_(new_token.view(1, 1))
        self.pos.add_(1)
        self.idx.add_(1)

    def prefill(self, ids):
        if self.custom:
            logits = self.model.prefill(ids)
        else:
            self.cache.reset()
            logits = self.model(ids, past_key_values=self.cache, use_cache=True, logits_to_keep=1).logits[:, -1]
        first = logits.argmax(-1)
        self.token.copy_(first.view(1, 1))
        self.output[0:1].copy_(first)
        self.pos.fill_(ids.shape[1])
        self.idx.fill_(1)
        return logits

    def run(self, ids, n):
        sync()
        start = time.perf_counter()
        self.prefill(ids)
        sync()
        prefill_end = time.perf_counter()
        for _ in range(n - 1):
            self.graph.replay()
        sync()
        end = time.perf_counter()
        return self.output[:n].cpu().tolist(), {
            "prefill_s": prefill_end - start,
            "decode_s": end - prefill_end,
            "total_s": end - start,
            "decode_tokens_s": (n - 1) / (end - prefill_end),
        }


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["hfgraph", "custom", "compiled"], default="hfgraph")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--prompts", type=int, default=3)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--out", default="results.json")
    p.add_argument("--model-path", default="model")
    p.add_argument("--attention", choices=["sdpa", "matmul"], default="sdpa")
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    t = time.perf_counter()
    source = args.model_path if Path(args.model_path).is_dir() else MODEL
    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().requires_grad_(False).cuda()
    sync()
    meta = {
        "model": MODEL, "model_source": source, "revision": model.config._commit_hash,
        "dtype": str(next(model.parameters()).dtype),
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(), "tokens": args.tokens,
        "max_len": args.max_len, "mode": args.mode,
        "attention": args.attention,
        "load_s": time.perf_counter() - t,
        "parameter_count": sum(v.numel() for v in model.parameters()),
        "timing": "CUDA synchronized wall time; prefill and n-1 decode steps; fixed output length; no EOS stop",
    }
    print(json.dumps({"environment": meta}), flush=True)
    reference = Reference(model)
    if args.mode == "hfgraph":
        optimized_model = model
    else:
        from optimized_qwen import OptimizedQwen
        optimized_model = OptimizedQwen(model, args.max_len, attention=args.attention)
    build = time.perf_counter()
    optimized = GraphRunner(optimized_model, args.max_len, args.tokens + 16,
                            custom=args.mode != "hfgraph", compile_decode=args.mode == "compiled")
    meta["optimization_setup_s"] = time.perf_counter() - build
    rows = []
    for pi, prompt in enumerate(PROMPTS[:args.prompts]):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids.cuda()
        assert ids.shape[1] + args.tokens <= args.max_len
        for runner in [reference, optimized]:
            runner.run(ids, args.tokens)
        for rep in range(args.repeats):
            outputs = {}
            metrics = {}
            order = [("baseline", reference), ("optimized", optimized)]
            if rep % 2:
                order.reverse()
            for name, runner in order:
                outputs[name], metrics[name] = runner.run(ids, args.tokens)
            matches = sum(a == b for a, b in zip(outputs["baseline"], outputs["optimized"]))
            first_diff = next((i for i, (a, b) in enumerate(zip(outputs["baseline"], outputs["optimized"])) if a != b), None)
            row = {
                "prompt_index": pi, "prompt": prompt, "input_tokens": ids.shape[1], "repeat": rep,
                **metrics, "matching_tokens": matches, "first_difference": first_diff,
                "decode_speedup": metrics["baseline"]["decode_s"] / metrics["optimized"]["decode_s"],
                "total_speedup": metrics["baseline"]["total_s"] / metrics["optimized"]["total_s"],
                "baseline_tokens": outputs["baseline"], "optimized_tokens": outputs["optimized"],
                "baseline_text": tok.decode(outputs["baseline"]), "optimized_text": tok.decode(outputs["optimized"]),
            }
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if not k.endswith("tokens") and not k.endswith("text")}), flush=True)
            Path(args.out).write_text(json.dumps({"environment": meta, "trials": rows}, indent=2))
    summary = {
        "median_decode_speedup": statistics.median(r["decode_speedup"] for r in rows),
        "median_total_speedup": statistics.median(r["total_speedup"] for r in rows),
        "all_tokens_match": all(r["first_difference"] is None for r in rows),
        "baseline_median_tokens_s": statistics.median(r["baseline"]["decode_tokens_s"] for r in rows),
        "optimized_median_tokens_s": statistics.median(r["optimized"]["decode_tokens_s"] for r in rows),
        "minimum_decode_speedup": min(r["decode_speedup"] for r in rows),
        "maximum_decode_speedup": max(r["decode_speedup"] for r in rows),
        "minimum_total_speedup": min(r["total_speedup"] for r in rows),
        "maximum_total_speedup": max(r["total_speedup"] for r in rows),
    }
    meta["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    meta["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    Path(args.out).write_text(json.dumps({"environment": meta, "summary": summary, "trials": rows}, indent=2))
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
