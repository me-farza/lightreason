"""Per-position CUDA graphs using the original HF Qwen arithmetic.

Unlike a padded StaticCache, each captured decode presents contiguous K/V of
exactly the same length as DynamicCache, with no explicit attention mask. The
original model's projections, normalization, rotary and SDPA calls are retained.
This is an exactness candidate, not a claim of identity before GPU validation.

Pinned API reference (Transformers 4.57.1):
https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/cache_utils.py

Pilot (loading, graph preparation and warmups excluded from inference timings):
  python exact_graph.py --prompts 2 --repeats 2 --out results/exact-graph-pilot.json
Natural-EOS quality integration:
  python quality_eval.py --candidate-factory exact_graph:make_quality_runner \
      --reference-results results/quality-reference.json --max-len 512

Graphs are specialized to each absolute decode position. They share only
permanent input/output/cache buffers, and this runner must execute sequentially.
Independent graph allocation pools are the default. --share-pool is experimental:
all graph-private activations are consumed inside the same graph and graph outputs
are copied to permanent buffers, permitting sequential reuse without preserving
private activation values between calls. Validate this mode separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class PrefixLayer(CacheLayerMixin):
    """Fixed storage with DynamicCache-shaped, contiguous returned prefixes.

    position is a Python integer set before each capture. It represents the
    prefix length before the one new token, not the storage capacity. GPU cache
    writes use cache_position; no tensor-to-host conversion occurs in update().
    """

    is_compileable = False
    is_sliding = False

    def __init__(self, kv_heads, head_dim, max_len, dtype, device):
        super().__init__()
        self.position = 0
        self.max_cache_len = max_len
        self.max_batch_size = 1
        self.dtype, self.device = dtype, device
        self.keys = torch.zeros((1, kv_heads, max_len, head_dim), dtype=dtype, device=device)
        self.values = torch.zeros_like(self.keys)
        self.is_initialized = True

    def lazy_initialization(self, key_states):
        # Buffers must exist before graph capture and retain their addresses.
        if key_states.dtype != self.dtype or key_states.device != self.device:
            raise ValueError("Cache device/dtype must match the original model")

    def update(self, key_states, value_states, cache_kwargs=None):
        if key_states.shape[-2] != 1:
            raise ValueError("PrefixLayer is for single-token decode; prefill uses HF DynamicCache")
        if not 0 <= self.position < self.max_cache_len:
            raise ValueError("Decode position is outside cache storage")
        if cache_kwargs is None or cache_kwargs.get("cache_position") is None:
            raise ValueError("Explicit GPU cache_position is required")
        cache_position = cache_kwargs["cache_position"]
        self.keys.index_copy_(2, cache_position, key_states)
        self.values.index_copy_(2, cache_position, value_states)
        length = self.position + 1
        # Slicing the preallocated buffers alone leaves a max_len head stride.
        # Compact copies retain DynamicCache's contiguous head/sequence layout.
        return self.keys[:, :, :length, :].contiguous(), self.values[:, :, :length, :].contiguous()

    def get_seq_length(self):
        return self.position

    def get_mask_sizes(self, cache_position):
        return self.position + cache_position.shape[0], 0

    def get_max_cache_shape(self):
        return -1


class PrefixCache(Cache):
    def __init__(self, model, max_len):
        config = model.config
        weight = model.model.embed_tokens.weight
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        super().__init__(layers=[PrefixLayer(config.num_key_value_heads, head_dim, max_len,
                                            weight.dtype, weight.device)
                                 for _ in range(config.num_hidden_layers)])

    def set_position(self, position):
        for layer in self.layers:
            layer.position = position

    def copy_dynamic(self, source, length):
        if len(source.layers) != len(self.layers):
            raise ValueError("Prefill cache layer count does not match the model")
        for destination, layer in zip(self.layers, source.layers):
            if layer.keys.shape[-2] != length or layer.values.shape[-2] != length:
                raise ValueError("Unexpected DynamicCache prefill length")
            destination.keys[:, :, :length, :].copy_(layer.keys)
            destination.values[:, :, :length, :].copy_(layer.values)
        self.set_position(length)


class ExactGraphRunner:
    """Original-model decode captured separately for every key length.

    prepare(ids, n) lazily captures missing positions. run(ids, n) calls prepare
    outside its measured region and records setup separately. prefill+replay_one
    are also available for natural-EOS and common-prefix validation.
    """

    def __init__(self, model, max_len=256, max_output=256, warmups=3, share_pool=False):
        config = model.config
        weight = model.model.embed_tokens.weight
        if weight.device.type != "cuda":
            raise ValueError("ExactGraphRunner requires a CUDA model")
        if config.model_type != "qwen2" or getattr(config, "use_sliding_window", False):
            raise ValueError("Only ordinary full-attention Qwen2/Qwen2.5 is supported")
        if getattr(config, "_attn_implementation", None) != "sdpa":
            raise ValueError("Load the reference model with attn_implementation='sdpa'")
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Use model.eval().requires_grad_(False) before constructing this runner")
        if max_len < 2 or max_output < 2 or warmups < 1:
            raise ValueError("max_len/max_output must be >=2 and warmups >=1")
        self.model, self.max_len, self.max_output = model, max_len, max_output
        self.warmups, self.share_pool = warmups, share_pool
        self.cache = PrefixCache(model, max_len)
        self.token = torch.zeros((1, 1), device=weight.device, dtype=torch.long)
        self.pos = torch.zeros((1,), device=weight.device, dtype=torch.long)
        self.idx = torch.ones((1,), device=weight.device, dtype=torch.long)
        self.output = torch.zeros((max_output,), device=weight.device, dtype=torch.long)
        self.last_logits = torch.zeros((1, config.vocab_size), device=weight.device, dtype=weight.dtype)
        self.graphs = {}
        self.capture_records = []
        self.setup_s = 0.0
        self.next_position = None
        self.capture_stream = torch.cuda.Stream(device=weight.device)
        self.pool = torch.cuda.graph_pool_handle() if share_pool else None

    def _check(self, ids, n):
        if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1]:
            raise ValueError("Only nonempty, unpadded batch-one prompts are supported")
        if not 1 <= n <= self.max_output:
            raise ValueError("Output budget is outside preallocated output storage")
        if ids.shape[1] + n > self.max_len:
            raise ValueError("Prompt plus output budget exceeds max_len")

    def _step(self):
        # Cache.get_seq_length/mask sizes see a Python-constant prefix length.
        # For a single query, the noncompileable cache permits the same unmasked
        # SDPA dispatch as the eager DynamicCache reference.
        logits = self.model(
            self.token, past_key_values=self.cache, use_cache=True,
            cache_position=self.pos, position_ids=self.pos.view(1, 1),
            attention_mask=None, logits_to_keep=1,
        ).logits[:, -1]
        selected = logits.argmax(-1)
        self.last_logits.copy_(logits)
        self.output.index_copy_(0, self.idx, selected)
        self.token.copy_(selected.view(1, 1))
        self.pos.add_(1)
        self.idx.add_(1)

    def _capture(self, position):
        self.cache.set_position(position)
        self.capture_stream.wait_stream(torch.cuda.current_stream())
        start = time.perf_counter()
        with torch.cuda.stream(self.capture_stream):
            for _ in range(self.warmups):
                self.pos.fill_(position)
                self.idx.fill_(1)
                self._step()
        torch.cuda.current_stream().wait_stream(self.capture_stream)
        torch.cuda.synchronize()
        warm_end = time.perf_counter()
        self.pos.fill_(position)
        self.idx.fill_(1)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        options = {"stream": self.capture_stream}
        if self.pool is not None:
            options["pool"] = self.pool
        with torch.cuda.graph(graph, **options):
            self._step()
        torch.cuda.synchronize()
        self.graphs[position] = graph
        self.capture_records.append({"position": position, "warmup_s": warm_end - start,
                                     "capture_s": time.perf_counter() - warm_end})

    def prepare(self, ids, n):
        self._check(ids, n)
        positions = [p for p in range(ids.shape[1], ids.shape[1] + n - 1) if p not in self.graphs]
        if not positions:
            return {"new_graphs": 0, "setup_s": 0.0}
        with torch.inference_mode():
            torch.cuda.synchronize()
            start = time.perf_counter()
            self.prefill(ids)
            for position in positions:
                self._capture(position)
            # Warmup/capture alters token/cache buffers. Restore the exact
            # DynamicCache prefill state before exposing the prepared runner.
            self.prefill(ids)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        self.setup_s += elapsed
        return {"new_graphs": len(positions), "setup_s": elapsed}

    def prefill(self, ids):
        with torch.inference_mode():
            out = self.model(ids, use_cache=True, logits_to_keep=1)
            self.cache.copy_dynamic(out.past_key_values, ids.shape[1])
            logits = out.logits[:, -1]
            selected = logits.argmax(-1)
            self.last_logits.copy_(logits)
            self.token.copy_(selected.view(1, 1))
            self.output[:1].copy_(selected)
            self.pos.fill_(ids.shape[1])
            self.idx.fill_(1)
            self.next_position = ids.shape[1]
        return self.last_logits

    def replay_one(self):
        if self.next_position not in self.graphs:
            raise ValueError(f"Position {self.next_position} has not been prepared")
        self.graphs[self.next_position].replay()
        self.next_position += 1
        return self.last_logits

    def run(self, ids, n):
        prepared = self.prepare(ids, n)
        with torch.inference_mode():
            torch.cuda.synchronize()
            start = time.perf_counter()
            self.prefill(ids)
            torch.cuda.synchronize()
            prefill_end = time.perf_counter()
            for _ in range(n - 1):
                self.replay_one()
            torch.cuda.synchronize()
            end = time.perf_counter()
        return self.output[:n].cpu().tolist(), {
            "prefill_s": prefill_end - start, "decode_s": end - prefill_end,
            "total_s": end - start,
            "decode_tokens_s": (n - 1) / (end - prefill_end) if n > 1 else 0.0,
            "new_graphs": prepared["new_graphs"], "excluded_setup_s": prepared["setup_s"],
            "cumulative_setup_s": self.setup_s, "prepared_graphs": len(self.graphs),
        }


class ExactNaturalEOS:
    def __init__(self, runner, tokenizer, eos_ids):
        self.runner, self.tokenizer, self.eos_ids = runner, tokenizer, set(eos_ids)

    def __call__(self, prompt, max_new_tokens):
        from quality_eval import Generation
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            return_tensors="pt").cuda()
        prepared = self.runner.prepare(ids, max_new_tokens)
        with torch.inference_mode():
            torch.cuda.synchronize()
            start = time.perf_counter()
            self.runner.prefill(ids)
            tokens = []
            for index in range(max_new_tokens):
                token = int(self.runner.token.item())
                tokens.append(token)
                if token in self.eos_ids or index + 1 == max_new_tokens:
                    break
                self.runner.replay_one()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        return {
            "text": self.tokenizer.decode(tokens, skip_special_tokens=True), "token_ids": tokens,
            "stop_reason": "eos" if tokens[-1] in self.eos_ids else "max_new_tokens",
            "metrics": {"quality_harness_latency_s": elapsed, "not_a_speed_benchmark": True,
                        "excluded_setup_s": prepared["setup_s"], "new_graphs": prepared["new_graphs"]}}


def make_quality_runner(model, tokenizer, args):
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    runner = ExactGraphRunner(model, max_len=args.max_len, max_output=args.max_len)
    return ExactNaturalEOS(runner, tokenizer, eos)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="model")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--prompts", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--share-pool", action="store_true")
    parser.add_argument("--out", default="results/exact-graph-pilot.json")
    args = parser.parse_args()
    if args.tokens < 2 or not 1 <= args.prompts <= 6 or args.repeats < 1:
        parser.error("Use tokens>=2, prompts between1 and6, repeats>=1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from benchmark import Reference, PROMPTS
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().requires_grad_(False).cuda()
    reference = Reference(model)
    exact = ExactGraphRunner(model, args.max_len, args.tokens + 16, args.warmups, args.share_pool)
    environment = {
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(), "dtype": "torch.bfloat16", "max_len": args.max_len,
        "tokens": args.tokens, "share_pool": args.share_pool,
        "checkpoint_sha256": _hash(Path(args.model_path) / "model.safetensors"),
        "source_sha256": _hash(__file__),
        "timing": "Warm synchronized wall time; fixed token budget; preparation excluded and recorded",
        "identity_status": "Must be established by measured exact token comparisons; not assumed",
    }
    rows = []
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for pi, prompt in enumerate(PROMPTS[:args.prompts]):
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                    add_generation_prompt=True, return_tensors="pt").cuda()
        reference.run(ids, args.tokens)
        exact.run(ids, args.tokens)
        for repeat in range(args.repeats):
            order = [("baseline", reference), ("exact", exact)]
            if repeat % 2:
                order.reverse()
            generated, metrics = {}, {}
            for name, runner in order:
                generated[name], metrics[name] = runner.run(ids, args.tokens)
            first = next((i for i, pair in enumerate(zip(generated["baseline"], generated["exact"]))
                          if pair[0] != pair[1]), None)
            row = {"prompt_index": pi, "prompt": prompt, "input_tokens": ids.shape[1], "repeat": repeat,
                   **metrics, "baseline_tokens": generated["baseline"], "exact_tokens": generated["exact"],
                   "tokens_exact": generated["baseline"] == generated["exact"],
                   "first_difference_zero_based": first,
                   "decode_speedup": metrics["baseline"]["decode_s"] / metrics["exact"]["decode_s"],
                   "total_speedup": metrics["baseline"]["total_s"] / metrics["exact"]["total_s"]}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if not k.endswith("tokens")}), flush=True)
            destination.write_text(json.dumps({"environment": environment, "trials": rows,
                "graph_setup_s": exact.setup_s, "capture_records": exact.capture_records}, indent=2))
    summary = {
        "all_tokens_match": all(row["tokens_exact"] for row in rows),
        "median_decode_speedup": statistics.median(row["decode_speedup"] for row in rows),
        "median_total_speedup": statistics.median(row["total_speedup"] for row in rows),
        "baseline_median_tokens_s": statistics.median(row["baseline"]["decode_tokens_s"] for row in rows),
        "exact_median_tokens_s": statistics.median(row["exact"]["decode_tokens_s"] for row in rows),
        "prepared_graphs": len(exact.graphs), "graph_setup_s": exact.setup_s,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    destination.write_text(json.dumps({"environment": environment, "summary": summary, "trials": rows,
                                      "capture_records": exact.capture_records}, indent=2))
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
