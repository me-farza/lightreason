"""Experimental batch-one Qwen3B runner with128-token attention graph buckets.

Original BF16 weights/projections/RMS/SiLU remain authoritative. Attention uses
the same pinned ATen Flash implementation, with GPU actualK and a host128-bucket
maxK that preserves the original dense split selection/partition. The standalone
varlen probe passed its finite tests; full-model output parity is still required
for THIS runner, including dynamic RoPE, append and feedback/capture lifecycle.

Original HF prefill restores token-major K/V by bit-copying a transpose. Each
decode layer fuses original-rounded RoPE with one token-major row append, never
copying the existing prefix. Positions remain GPU data; there is one graph per
K bucket, rather than per absolute position. Graph setup is separate from warmed
generation latency. No reference outputs or teacher forcing enter this runner.
"""
from __future__ import annotations

import time

import torch
from torch.nn import functional as F
import triton
import triton.language as tl

from exact3b_core import Exact3BDecoder, _rounded_product_sum, _rotate_sign
from exact3b_runner import Exact3BRunner
from exact_graph import ExactNaturalEOS


def bucket_for_position(position):
    if type(position) is not int or position < 0:
        raise ValueError("Position must be a nonnegative CPU integer")
    return (position // 128 + 1) * 128


class TokenMajorLayer:
    def __init__(self, capacity, heads, head_dim, dtype, device):
        self.keys = torch.zeros((capacity, heads, head_dim), dtype=dtype, device=device)
        self.values = torch.zeros_like(self.keys)


class TokenMajorCache:
    """Fixed linear storage plus GPU INT32 sequence metadata; decode only."""
    def __init__(self, model, max_len):
        config, weight = model.config, model.model.embed_tokens.weight
        self.max_len = max_len
        self.capacity = triton.cdiv(max_len, 128) * 128
        self.kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.layers = [TokenMajorLayer(self.capacity, self.kv_heads, self.head_dim,
                                      weight.dtype, weight.device)
                       for _ in range(config.num_hidden_layers)]
        self.cu_q = torch.tensor([0, 1], device=weight.device, dtype=torch.int32)
        self.cu_k = torch.tensor([0, 0], device=weight.device, dtype=torch.int32)
        self.length = self.cu_k[1:2]

    def copy_dynamic(self, source, length):
        if not 0 < length < self.max_len or len(source.layers) != len(self.layers):
            raise ValueError("Prefill cache dimensions exceed token-major storage")
        for destination, original in zip(self.layers, source.layers):
            expected = (1, self.kv_heads, length, self.head_dim)
            if tuple(original.keys.shape) != expected or tuple(original.values.shape) != expected:
                raise ValueError("Original prefill K/V dimensions differ")
            destination.keys[:length].copy_(original.keys[0].transpose(0, 1))
            destination.values[:length].copy_(original.values[0].transpose(0, 1))
        self.length.fill_(length)

    def head_major_prefix(self, layer_index, length):
        """Untimed diagnostics only; production Flash reads token-major storage."""
        layer = self.layers[layer_index]
        return (layer.keys[:length].permute(1, 0, 2).unsqueeze(0).contiguous(),
                layer.values[:length].permute(1, 0, 2).unsqueeze(0).contiguous())

    def allocated_bytes(self):
        return sum(value.numel() * value.element_size() for layer in self.layers
                   for value in (layer.keys, layer.values)) + self.cu_q.numel() * 4 + self.cu_k.numel() * 4


@triton.jit
def _rope_append(Q, K, V, Cos, Sin, Position, OutQ, StoreK, StoreV, CuK, TraceK,
                  NQ: tl.constexpr, NK: tl.constexpr, HEAD: tl.constexpr,
                  CAPACITY: tl.constexpr, BLOCK: tl.constexpr,
                  Q_BLOCKS: tl.constexpr, WRITE_TRACE: tl.constexpr):
    block = tl.program_id(0)
    position = tl.load(Position)
    if block == 0:
        tl.store(CuK + 1, (position + 1).to(tl.int32))
    if block < Q_BLOCKS:
        index = block * BLOCK + tl.arange(0, BLOCK)
        valid = index < NQ
        dim = index % HEAD
        peer = index // HEAD * HEAD + (dim + HEAD // 2) % HEAD
        q = tl.load(Q + index, valid, other=0).to(tl.float32)
        rotated = tl.load(Q + peer, valid, other=0).to(tl.float32)
        rotated = _rotate_sign(rotated, dim, HEAD)
        cosine = tl.load(Cos + dim, valid, other=0).to(tl.float32)
        sine = tl.load(Sin + dim, valid, other=0).to(tl.float32)
        output = _rounded_product_sum(q, cosine, rotated, sine)
        tl.store(OutQ + index, output, valid)
    else:
        index = (block - Q_BLOCKS) * BLOCK + tl.arange(0, BLOCK)
        valid = (index < NK) & (position >= 0) & (position < CAPACITY)
        dim = index % HEAD
        peer = index // HEAD * HEAD + (dim + HEAD // 2) % HEAD
        k = tl.load(K + index, valid, other=0).to(tl.float32)
        rotated = tl.load(K + peer, valid, other=0).to(tl.float32)
        rotated = _rotate_sign(rotated, dim, HEAD)
        cosine = tl.load(Cos + dim, valid, other=0).to(tl.float32)
        sine = tl.load(Sin + dim, valid, other=0).to(tl.float32)
        output = _rounded_product_sum(k, cosine, rotated, sine)
        value = tl.load(V + index, valid, other=0)
        tl.store(StoreK + position * NK + index, output, valid)
        tl.store(StoreV + position * NK + index, value, valid)
        if WRITE_TRACE:
            tl.store(TraceK + index, output, valid)


def rope_append(q, k, v, cos, sin, position, cache, layer_index, trace_key=False):
    if any(value.dtype != torch.bfloat16 or not value.is_contiguous() for value in (q, k, v)):
        raise ValueError("Dynamic row append requires contiguous BF16 batch-one Q/K/V")
    if q.shape[-2] != 1 or k.shape[-2] != 1 or k.shape != v.shape:
        raise ValueError("Dynamic row append is single-query only")
    out_q = torch.empty_like(q)
    out_k = torch.empty_like(k) if trace_key else k
    block = 256
    q_blocks = triton.cdiv(q.numel(), block)
    layer = cache.layers[layer_index]
    _rope_append[(q_blocks + triton.cdiv(k.numel(), block),)](
        q, k, v, cos, sin, position, out_q, layer.keys, layer.values, cache.cu_k, out_k,
        q.numel(), k.numel(), q.shape[-1], cache.capacity, block, q_blocks, trace_key,
        num_warps=4, enable_fp_fusion=False)
    return out_q, out_k if trace_key else None


class DynamicDecoder(Exact3BDecoder):
    """Core arithmetic plus dynamic-position/token-major original Flash calls."""
    def __init__(self, model, cache, norm_mode="split", pointwise="compiled",
                 gate_mode="compiled", mean_threads=None, norm_warps=4,
                 combine_projections=False, combine_mode="kv"):
        super().__init__(model, cache, norm_mode=norm_mode, pointwise=pointwise,
                         gate_mode=gate_mode, mean_threads=mean_threads, norm_warps=norm_warps,
                         combine_projections=combine_projections, combine_mode=combine_mode,
                         cache_rope=False, fuse_cache=False)
        self.fuse_cache = True
        self.attention_cache_kind = "token_major_varlen128bucket"
        self.flash = torch.ops.aten._flash_attention_forward.default

    @torch.no_grad()
    def forward_one(self, input_ids, position, prefix_length=None, trace=None, *, max_k=None):
        if max_k is None:
            if prefix_length is None:
                raise ValueError("Provide a host attention bucket or an untimed CPU prefix length")
            max_k = bucket_for_position(prefix_length)
        if max_k % 128 or not 128 <= max_k <= self.cache.capacity:
            raise ValueError("Attention maxK must be a valid128-token storage bucket")
        source = self.source.model
        x = source.embed_tokens(input_ids)
        def save(key, value):
            if trace is not None:
                trace[key] = value.detach().clone()
        save("embedding", x)
        # Original single-position rotary arithmetic reads the GPU position.
        cos, sin = source.rotary_emb(x, position.view(1, 1))
        save("rotary.cos", cos)
        save("rotary.sin", sin)
        pending = None
        for index, layer in enumerate(source.layers):
            key = f"layer.{index}"
            if pending is None:
                x, normalized = self._normalize(x, layer.input_layernorm)
            else:
                x, normalized = self._normalize(pending[0], layer.input_layernorm, pending[1])
            residual = x
            save(key + ".input_norm", normalized)
            attention = layer.self_attn
            input_shape = normalized.shape[:-1]
            hidden_shape = (*input_shape, -1, attention.head_dim)
            if self.combine_projections:
                if self.combine_mode == "kv":
                    q_raw = attention.q_proj(normalized)
                joined = F.linear(normalized, getattr(self, f"{self.combine_mode}_weight_{index}"),
                                  getattr(self, f"{self.combine_mode}_bias_{index}"))
                if self.combine_mode == "kv":
                    k_raw, v_raw = joined.split((attention.k_proj.out_features, attention.v_proj.out_features), dim=-1)
                else:
                    q_raw, k_raw, v_raw = joined.split((attention.q_proj.out_features,
                        attention.k_proj.out_features, attention.v_proj.out_features), dim=-1)
            else:
                q_raw, k_raw, v_raw = attention.q_proj(normalized), attention.k_proj(normalized), attention.v_proj(normalized)
            save(key + ".q_proj", q_raw)
            save(key + ".k_proj", k_raw)
            save(key + ".v_proj", v_raw)
            q = q_raw.view(hidden_shape).transpose(1, 2)
            k = k_raw.view(hidden_shape).transpose(1, 2)
            v = v_raw.view(hidden_shape).transpose(1, 2)
            q, traced_k = rope_append(q, k, v, cos, sin, position, self.cache, index, trace is not None)
            save(key + ".q_rope", q)
            if trace is not None:
                save(key + ".k_rope", traced_k)
            stored = self.cache.layers[index]
            # Exactly the auto-GQA varlen path validated by the standalone
            # probe: raw output[1,16,128], scaled as original dense SDPA.
            attended = self.flash(q.reshape(1, self.source.config.num_attention_heads, attention.head_dim),
                stored.keys, stored.values, self.cache.cu_q, self.cache.cu_k,
                1, max_k, 0.0, False, False, scale=attention.scaling)[0]
            attended = attended.unsqueeze(1)
            save(key + ".attention", attended)
            attended = attended.reshape(*input_shape, -1).contiguous()
            projected = attention.o_proj(attended)
            save(key + ".o_proj", projected)
            x, normalized = self._normalize(residual, layer.post_attention_layernorm, projected)
            save(key + ".post_residual", x)
            save(key + ".post_norm", normalized)
            if self.combine_projections:
                joined = F.linear(normalized, getattr(self, f"gate_up_weight_{index}"))
                gate, up = joined.split((layer.mlp.gate_proj.out_features, layer.mlp.up_proj.out_features), dim=-1)
            else:
                gate, up = layer.mlp.gate_proj(normalized), layer.mlp.up_proj(normalized)
            save(key + ".gate_proj", gate)
            save(key + ".up_proj", up)
            gated = self.gate(gate, up)
            save(key + ".silu_mul", gated)
            down = layer.mlp.down_proj(gated)
            save(key + ".down_proj", down)
            pending = (x, down)
            if trace is not None:
                save(key + ".output", x + down)
        _, normalized = self._normalize(pending[0], source.norm, pending[1])
        save("final_norm", normalized)
        head = getattr(self, "head_override", None)
        logits = head(normalized) if head is not None else self.source.lm_head(normalized)
        save("lm_head", logits)
        return logits[:, -1]


@triton.jit
def _feedback(Selected, Token, Position, Output, CAPACITY: tl.constexpr):
    position = tl.load(Position)
    token = tl.load(Selected)
    next_position = position + 1
    tl.store(Output + next_position, token, (next_position >= 0) & (next_position <= CAPACITY))
    tl.store(Token, token)
    tl.store(Position, next_position)


class Bucketed3BRunner(Exact3BRunner):
    """Exact3BRunner-compatible API; numerical parity remains an acceptance gate.

    The legacy fuse_cache argument is accepted for factory compatibility. This
    runner always uses fused token-major row append; requested_fuse_cache keeps
    the caller setting while decoder.fuse_cache records actual behavior.
    """
    def __init__(self, model, max_len=1024, max_output=512, warmups=3,
                 norm_mode="split", pointwise="compiled", gate_mode="compiled",
                 fuse_cache=False, mean_threads=None, norm_warps=4,
                 share_pool=False, track_logits=False, screen_head=False,
                 verifier_lanes=16, verify_head_steps=False,
                 combine_projections=False, combine_mode="kv"):
        started = time.perf_counter()
        config = model.config
        geometry = (config.hidden_size, config.num_attention_heads,
                    config.num_key_value_heads, config.hidden_size // config.num_attention_heads)
        if geometry != (2048, 16, 2, 128):
            raise ValueError("This bucketed prototype is validated only for Qwen3B2048/Q16/KV2/head128 geometry")
        if verify_head_steps and not screen_head:
            raise ValueError("Head verification requires screened-head execution")
        with torch.inference_mode(False), torch.no_grad():
            # Initialize lifecycle/output buffers cheaply. Creating the final
            # decoder only once avoids duplicate3GiB projection copies and
            # duplicate screening indices when those opt-ins are requested.
            super().__init__(model, max_len, max_output, warmups,
                norm_mode="eager", pointwise="eager", gate_mode="eager",
                fuse_cache=False, mean_threads=mean_threads, norm_warps=norm_warps,
                share_pool=share_pool, track_logits=track_logits, screen_head=False,
                verifier_lanes=verifier_lanes, verify_head_steps=False,
                combine_projections=False, combine_mode=combine_mode)
            self.cache = TokenMajorCache(model, max_len)
            self.decoder = DynamicDecoder(model, self.cache, norm_mode=norm_mode,
                pointwise=pointwise, gate_mode=gate_mode, mean_threads=mean_threads,
                norm_warps=norm_warps, combine_projections=combine_projections,
                combine_mode=combine_mode)
            self.requested_fuse_cache = fuse_cache
            self.graph_key_kind = "max_k_128_bucket"
            self.attention_bucket_size = 128
            self.capture_bucket = None
            self.screen_head = screen_head
            self.verifier_lanes = verifier_lanes
            self.verify_head_steps = verify_head_steps
            self.head_verified_steps = 0
            self.head_index_setup_s = 0.0
            if screen_head:
                from exact3b_head import BoundedHead3B
                torch.cuda.synchronize()
                index_start = time.perf_counter()
                self.decoder.head_override = BoundedHead3B(model.lm_head,
                    verifier_lanes=verifier_lanes, compact=True)
                torch.cuda.synchronize()
                self.head_index_setup_s = time.perf_counter() - index_start
                if verify_head_steps:
                    self.last_head_query = torch.empty((1, 1, config.hidden_size),
                        device=self.token.device, dtype=model.lm_head.weight.dtype)
                    def save_query(module, arguments):
                        self.last_head_query.copy_(arguments[0])
                    self.head_query_hook = self.decoder.head_override.register_forward_pre_hook(save_query)
            torch.cuda.synchronize()
            self.dynamic_constructor_s = time.perf_counter() - started

    def _step(self):
        if self.capture_bucket is None:
            raise RuntimeError("Set a host bucket before capturing the dynamic step")
        logits = self.decoder.forward_one(self.token, self.pos, max_k=self.capture_bucket)
        selected = logits.argmax(-1)
        if self.track_logits:
            self.last_logits.copy_(logits)
        _feedback[(1,)](selected, self.token, self.pos, self.absolute_output,
                        self.max_len, num_warps=1)

    def _capture_bucket(self, bucket):
        if bucket % 128 or not 128 <= bucket <= self.cache.capacity:
            raise ValueError("Invalid attention bucket")
        representative = min(bucket - 1, self.max_len - 1)
        self.capture_bucket = bucket
        self.capture_stream.wait_stream(torch.cuda.current_stream())
        started = time.perf_counter()
        with torch.cuda.stream(self.capture_stream):
            for _ in range(self.warmups):
                self.pos.fill_(representative)
                self._step()
        torch.cuda.current_stream().wait_stream(self.capture_stream)
        torch.cuda.synchronize()
        warm_end = time.perf_counter()
        self.pos.fill_(representative)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        options = {"stream": self.capture_stream}
        if self.pool is not None:
            options["pool"] = self.pool
        with torch.cuda.graph(graph, **options):
            self._step()
        torch.cuda.synchronize()
        self.graphs[bucket] = graph
        self.capture_records.append({"bucket_max_k": bucket,
                                     "representative_position": representative,
                                     "warmup_s": warm_end - started,
                                     "capture_s": time.perf_counter() - warm_end})

    def _capture(self, position):
        return self._capture_bucket(bucket_for_position(position))

    def prepare(self, ids, n):
        self._check(ids, n)
        wanted = sorted({bucket_for_position(position)
                         for position in range(ids.shape[1], ids.shape[1] + n - 1)})
        missing = [bucket for bucket in wanted if bucket not in self.graphs]
        if not missing:
            return {"new_graphs": 0, "new_buckets": [], "setup_s": 0.0,
                    "graph_granularity_tokens": 128}
        with torch.inference_mode(False), torch.no_grad():
            torch.cuda.synchronize()
            started = time.perf_counter()
            self.prefill(ids)
            for bucket in missing:
                self._capture_bucket(bucket)
            # Captures/warmups modify token, current row and feedback buffers.
            # Original HF prefill restores the true request prefix afterwards.
            self.prefill(ids)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
        self.setup_s += elapsed
        return {"new_graphs": len(missing), "new_buckets": missing, "setup_s": elapsed,
                "graph_granularity_tokens": 128}

    def prefill(self, ids):
        with torch.inference_mode(False), torch.no_grad():
            output = self.model(ids, use_cache=True, logits_to_keep=1)
            length = ids.shape[1]
            self.cache.copy_dynamic(output.past_key_values, length)
            logits = output.logits[:, -1]
            selected = logits.argmax(-1)
            if self.track_logits:
                self.last_logits.copy_(logits)
            self.token.copy_(selected.view(1, 1))
            self.absolute_output[length:length + 1].copy_(selected)
            self.pos.fill_(length)
            self.next_position = length
            self.capture_position = length
            self.capture_bucket = bucket_for_position(length)
            self.prompt_length = length
            return logits

    def replay_one(self):
        if self.next_position is None or self.next_position >= self.max_len:
            raise ValueError("No valid decode position is active")
        bucket = bucket_for_position(self.next_position)
        if bucket not in self.graphs:
            raise ValueError(f"Attention bucket{bucket} was not prepared")
        self.graphs[bucket].replay()
        self.next_position += 1
        if self.verify_head_steps:
            record = self.decoder.head_override.audit_buffers(self.last_head_query)
            required = ("selected_bits_exact", "all_finite_reference_logits_enclosed",
                        "reference_winner_selected", "argmax_exact")
            if not all(record[key] for key in required):
                raise RuntimeError("Original full-head audit failed: " + str(record))
            self.head_verified_steps += 1
        return self.last_logits

    def memory_report(self):
        return {"cache_kind": "token_major_single_bank", "public_max_len": self.max_len,
                "storage_capacity": self.cache.capacity,
                "allocated_kv_bytes": self.cache.allocated_bytes(),
                "graph_count": len(self.graphs), "bucket_count": len(self.graphs),
                "bucket_max_k_values": sorted(self.graphs), "bucket_width": 128,
                "excluded_cumulative_graph_setup_s": self.setup_s,
                "constructor_s": self.dynamic_constructor_s,
                "extra_projection_bytes": self.decoder.extra_projection_bytes,
                "projection_copy_setup_s": self.decoder.projection_copy_setup_s,
                "head_index_setup_s": self.head_index_setup_s,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.token.device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.token.device),
                "scope": "Cache storage is linear; original weights and graph-private activations are additional"}


def make_quality_runner(model, tokenizer, args):
    settings = {name: getattr(args, name) for name in
                ("norm_mode", "pointwise", "gate_mode", "fuse_cache", "mean_threads",
                 "norm_warps", "share_pool", "track_logits", "screen_head",
                 "verifier_lanes", "verify_head_steps", "combine_projections",
                 "combine_mode", "warmups") if hasattr(args, name)}
    runner = Bucketed3BRunner(model, args.max_len, args.max_len, **settings)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    return ExactNaturalEOS(runner, tokenizer, eos)
