"""Original HF arithmetic with compact-prefix caches and cheaper graph feedback.

This subclasses ExactGraphRunner without modifying it. Each absolute decode
position uses a constant CUDA position tensor, contiguous K/V slots of exactly
the DynamicCache prefix length, and torch.cat(...,out=next_slot). Original model
projections, normalization, attention and BF16 arithmetic remain in place.

RoPE constants are computed with the original rotary forward, one position at a
time using the original single-query shape. Only a captured decode step briefly
substitutes those constants; normal reference and prefill calls remain original.

Token-only graphs omit last_logits copies by default. Enable track_logits for
diagnostics; that is a different timed configuration. Position-specialized graph
and cache preparation is excluded and recorded, while original HF prefill plus
cache transfer remains inside generation timing. Bit identity and >=10x latency
are experimental goals, not assumptions; validate complete output sequences.

  python exact10_graphs.py --prompts 2 --repeats 2 --out results20x/exact10-pilot.json
  python exact10_validation.py --candidate-factory exact10_graphs:make_quality_runner
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

from exact_graph import ExactGraphRunner, ExactNaturalEOS


class PackedPrefixLayer(CacheLayerMixin):
    is_compileable = False
    is_sliding = False

    def __init__(self, kv_heads, head_dim, max_len, dtype, device):
        super().__init__()
        self.position = 0
        self.kv_heads, self.head_dim = kv_heads, head_dim
        self.max_cache_len, self.max_batch_size = max_len, 1
        self.dtype, self.device = dtype, device
        self.key_slots, self.value_slots = {}, {}
        self.ensure_slot(0)
        self.keys, self.values = self.key_slots[0], self.value_slots[0]
        self.is_initialized = True

    def ensure_slot(self, length):
        if not 0 <= length <= self.max_cache_len:
            raise ValueError("Prefix slot is outside cache capacity")
        if length not in self.key_slots:
            shape = (1, self.kv_heads, length, self.head_dim)
            self.key_slots[length] = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self.value_slots[length] = torch.zeros_like(self.key_slots[length])

    def set_position(self, position):
        self.ensure_slot(position)
        self.ensure_slot(position + 1)
        self.position = position
        self.keys, self.values = self.key_slots[position], self.value_slots[position]

    def lazy_initialization(self, key_states):
        if key_states.dtype != self.dtype or key_states.device != self.device:
            raise ValueError("Cache dtype/device must match original model tensors")

    def update(self, key_states, value_states, cache_kwargs=None):
        if key_states.shape[-2] != 1:
            raise ValueError("PackedPrefixLayer supports single-query decode only")
        position = self.position
        if position + 1 not in self.key_slots:
            raise ValueError("Allocate both compact prefix slots before capture")
        # Independent compact slots reproduce DynamicLayer's concatenation
        # layout. No padded head stride or attention mask is introduced.
        key = torch.cat((self.key_slots[position], key_states), dim=2, out=self.key_slots[position + 1])
        value = torch.cat((self.value_slots[position], value_states), dim=2, out=self.value_slots[position + 1])
        return key, value

    def get_seq_length(self):
        return self.position

    def get_mask_sizes(self, cache_position):
        return self.position + cache_position.shape[0], 0

    def get_max_cache_shape(self):
        return -1


class PackedPrefixCache(Cache):
    def __init__(self, model, max_len):
        config = model.config
        weight = model.model.embed_tokens.weight
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        super().__init__(layers=[PackedPrefixLayer(config.num_key_value_heads, head_dim, max_len,
                            weight.dtype, weight.device) for _ in range(config.num_hidden_layers)])

    def set_position(self, position):
        for layer in self.layers:
            layer.set_position(position)

    def copy_dynamic(self, source, length):
        if len(source.layers) != len(self.layers):
            raise ValueError("Dynamic prefill cache has a different number of layers")
        for destination, original in zip(self.layers, source.layers):
            destination.set_position(length)
            if original.keys.shape[-2] != length or original.values.shape[-2] != length:
                raise ValueError("Dynamic prefill cache has a different prefix length")
            destination.key_slots[length].copy_(original.keys)
            destination.value_slots[length].copy_(original.values)

    def allocated_bytes(self):
        return sum(tensor.numel() * tensor.element_size() for layer in self.layers
                   for slots in (layer.key_slots, layer.value_slots) for tensor in slots.values())


class PackedExactGraphRunner(ExactGraphRunner):
    """ExactGraphRunner API with compact cat.out slots and absolute outputs.

    replay_one() returns None in token-only mode; enable track_logits if a
    diagnostic caller needs captured logits. Production output is always read
    from token/absolute_output, never reconstructed from cached answers.
    """

    def __init__(self, model, max_len=512, max_output=512, warmups=3,
                 share_pool=False, track_logits=False, precompute_rope=True):
        rotary = model.model.rotary_emb
        if precompute_rope and getattr(rotary, "rope_type", "default") != "default":
            raise ValueError("Precomputed rotary constants currently support default non-stateful RoPE only")
        with torch.inference_mode(False), torch.no_grad():
            super().__init__(model, max_len, max_output, warmups, share_pool)
            self.cache = PackedPrefixCache(model, max_len)
            self.track_logits = track_logits
            if not track_logits:
                self.last_logits = None
            self.absolute_output = torch.zeros((max_len + 1,), device=self.token.device, dtype=torch.long)
            self.fixed_positions = {}
            self.rope_values = {}
            self.precompute_rope = precompute_rope
            self.original_rotary_forward = rotary.forward
            self.capture_position = None
            self.prompt_length = None
            self.target_versions = [(parameter, parameter._version) for parameter in model.parameters()]

    def _ensure_position(self, position):
        self.cache.set_position(position)
        if position not in self.fixed_positions:
            self.fixed_positions[position] = torch.tensor([position], device=self.token.device, dtype=torch.long)
        if self.precompute_rope and position not in self.rope_values:
            # RoPE forward uses x for dtype/device and shape, not its values.
            # Preserve [1,1,hidden] and [1,1] rather than evaluating many
            # positions together with a different floating-point matmul shape.
            x = torch.zeros((1, 1, self.model.config.hidden_size),
                            device=self.token.device, dtype=next(self.model.parameters()).dtype)
            cos, sin = self.original_rotary_forward(x, self.fixed_positions[position].view(1, 1))
            self.rope_values[position] = (cos.detach().clone(), sin.detach().clone())

    def _step(self):
        position = self.capture_position
        fixed = self.fixed_positions[position]
        rotary = self.model.model.rotary_emb
        previous_forward = rotary.forward
        try:
            if self.precompute_rope:
                values = self.rope_values[position]
                def fixed_rotary(x, position_ids):
                    return values
                rotary.forward = fixed_rotary
            logits = self.model(
                self.token, past_key_values=self.cache, use_cache=True,
                cache_position=fixed, position_ids=fixed.view(1, 1),
                attention_mask=None, logits_to_keep=1,
            ).logits[:, -1]
        finally:
            rotary.forward = previous_forward
        selected = logits.argmax(-1)
        if self.track_logits:
            self.last_logits.copy_(logits)
        self.absolute_output[position + 1:position + 2].copy_(selected)
        self.token.copy_(selected.view(1, 1))

    def _capture(self, position):
        self._ensure_position(position)
        self.capture_position = position
        self.capture_stream.wait_stream(torch.cuda.current_stream())
        start = time.perf_counter()
        with torch.cuda.stream(self.capture_stream):
            for _ in range(self.warmups):
                self._step()
        torch.cuda.current_stream().wait_stream(self.capture_stream)
        torch.cuda.synchronize()
        warm_end = time.perf_counter()
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
        with torch.inference_mode(False), torch.no_grad():
            torch.cuda.synchronize()
            start = time.perf_counter()
            self.prefill(ids)
            for position in positions:
                self._capture(position)
            self.prefill(ids)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        self.setup_s += elapsed
        return {"new_graphs": len(positions), "setup_s": elapsed}

    def prefill(self, ids):
        with torch.inference_mode(False), torch.no_grad():
            # No rotary substitution is active here; this is the original
            # reference forward, followed by exact copies into compact slots.
            out = self.model(ids, use_cache=True, logits_to_keep=1)
            length = ids.shape[1]
            self.cache.copy_dynamic(out.past_key_values, length)
            logits = out.logits[:, -1]
            selected = logits.argmax(-1)
            if self.track_logits:
                self.last_logits.copy_(logits)
            self.token.copy_(selected.view(1, 1))
            self.absolute_output[length:length + 1].copy_(selected)
            self.next_position = length
            self.capture_position = length
            self.prompt_length = length
            return logits

    def assert_target_unchanged(self):
        if any(parameter._version != version for parameter, version in self.target_versions):
            raise RuntimeError("An original target parameter was modified in place")
        if self.model.model.rotary_emb.forward != self.original_rotary_forward:
            raise RuntimeError("Rotary substitution leaked outside the captured step")

    def run(self, ids, n):
        prepared = self.prepare(ids, n)
        with torch.inference_mode(False), torch.no_grad():
            torch.cuda.synchronize()
            start = time.perf_counter()
            self.prefill(ids)
            torch.cuda.synchronize()
            prefill_end = time.perf_counter()
            for _ in range(n - 1):
                self.replay_one()
            torch.cuda.synchronize()
            end = time.perf_counter()
        self.assert_target_unchanged()
        length = ids.shape[1]
        return self.absolute_output[length:length + n].cpu().tolist(), {
            "prefill_s": prefill_end - start, "decode_s": end - prefill_end, "total_s": end - start,
            "decode_tokens_s": (n - 1) / (end - prefill_end) if n > 1 else 0.0,
            "new_graphs": prepared["new_graphs"], "excluded_setup_s": prepared["setup_s"],
            "cumulative_setup_s": self.setup_s, "prepared_graphs": len(self.graphs),
        }


def make_quality_runner(model, tokenizer, args):
    runner = PackedExactGraphRunner(
        model, max_len=args.max_len, max_output=args.max_len,
        warmups=getattr(args, "warmups", 3), share_pool=getattr(args, "share_pool", False),
        track_logits=getattr(args, "track_logits", False),
        precompute_rope=not getattr(args, "no_precompute_rope", False))
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    return ExactNaturalEOS(runner, tokenizer, eos)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="model")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--prompts", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--share-pool", action="store_true")
    parser.add_argument("--track-logits", action="store_true")
    parser.add_argument("--no-precompute-rope", action="store_true")
    parser.add_argument("--out", default="results20x/exact10-graphs.json")
    args = parser.parse_args()
    if args.tokens < 2 or not 1 <= args.prompts <= 6 or args.repeats < 1:
        parser.error("Require tokens>=2, prompts in1..6, repeats>=1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from benchmark import Reference, PROMPTS
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().requires_grad_(False).cuda()
    reference = Reference(model)
    candidate = PackedExactGraphRunner(model, args.max_len, args.tokens + 16,
        args.warmups, args.share_pool, args.track_logits, not args.no_precompute_rope)
    environment = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct", "dtype": "torch.bfloat16", "batch_size": 1,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(), "max_len": args.max_len, "tokens": args.tokens,
        "share_pool": args.share_pool, "track_logits": args.track_logits,
        "precompute_rope": not args.no_precompute_rope,
        "checkpoint_sha256": sha256(Path(args.model_path) / "model.safetensors"),
        "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                           ["exact10_graphs.py", "exact_graph.py", "benchmark.py"]},
        "timing": "Paired warmed CUDA-synchronized fixed-budget generation; loading, graph/cache/RoPE preparation and CPU result transfer excluded; original HF prefill plus K/V transfer included",
        "quality_scope": "All timed token sequences must equal the unchanged original HF reference; full38-case exact10_validation remains separate",
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(json.dumps({"environment": environment}), flush=True)
    for index, prompt in enumerate(PROMPTS[:args.prompts]):
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt").cuda()
        reference.run(ids, args.tokens)
        candidate.run(ids, args.tokens)
        for repeat in range(args.repeats):
            order = [("baseline", reference), ("candidate", candidate)]
            if repeat % 2:
                order.reverse()
            tokens, metrics = {}, {}
            for name, runner in order:
                tokens[name], metrics[name] = runner.run(ids, args.tokens)
            exact = tokens["baseline"] == tokens["candidate"]
            first = next((i for i, (a, b) in enumerate(zip(tokens["baseline"], tokens["candidate"])) if a != b), None)
            row = {"prompt_index": index, "prompt": prompt, "input_tokens": ids.shape[1], "repeat": repeat,
                   **metrics, "tokens_exact": exact, "first_difference_zero_based": first,
                   "baseline_tokens": tokens["baseline"], "candidate_tokens": tokens["candidate"],
                   "decode_speedup": metrics["baseline"]["decode_s"] / metrics["candidate"]["decode_s"],
                   "total_speedup": metrics["baseline"]["total_s"] / metrics["candidate"]["total_s"]}
            rows.append(row)
            destination.write_text(json.dumps({"environment": environment, "trials": rows,
                "graph_setup_s": candidate.setup_s, "capture_records": candidate.capture_records}, indent=2))
            print(json.dumps({key: value for key, value in row.items()
                if key not in {"prompt", "baseline_tokens", "candidate_tokens"}}), flush=True)
    candidate.assert_target_unchanged()
    protocol = args.prompts == 6 and args.tokens == 128 and args.repeats >= 3
    summary = {
        "trials": len(rows), "original_six_fixed128_protocol": protocol,
        "all_tokens_match": all(row["tokens_exact"] for row in rows),
        "median_decode_speedup": statistics.median(row["decode_speedup"] for row in rows),
        "median_total_speedup": statistics.median(row["total_speedup"] for row in rows),
        "minimum_total_speedup": min(row["total_speedup"] for row in rows),
        "maximum_total_speedup": max(row["total_speedup"] for row in rows),
        "baseline_median_tokens_s": statistics.median(row["baseline"]["decode_tokens_s"] for row in rows),
        "candidate_median_tokens_s": statistics.median(row["candidate"]["decode_tokens_s"] for row in rows),
        "ten_x_and_timed_outputs_exact": protocol and all(row["tokens_exact"] and row["total_speedup"] >= 10 for row in rows),
        "graph_setup_s": candidate.setup_s, "prepared_graphs": len(candidate.graphs),
        "compact_kv_slot_bytes": candidate.cache.allocated_bytes(),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "full38_case_validation_required": True,
    }
    destination.write_text(json.dumps({"environment": environment, "summary": summary,
        "trials": rows, "capture_records": candidate.capture_records}, indent=2))
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
