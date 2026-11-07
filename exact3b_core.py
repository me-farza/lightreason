"""Exactness-first Qwen2.5-3B decode core; no measured parity claim yet.

No target weights are quantized, trained, or changed. By default the runner
uses separate original source projections; optional concatenation makes exact
copies whose cuBLAS parity must be audited independently. SDPA has actual-length
contiguous K/V prefixes. Standalone PTX pointwise kernels retain BF16 rounding
and signed-zero boundaries. Split RMS retains original ATen mean; optional
fused RMS reproduces the profiled vector4 reduction tree.

Predicted2048-wide source mean: block512, vector4, shared offsets256,128,64,32,
then ascending warp shuffles1,2,4,8,16. Confirm this against the installed GPU
profile and audit raw bytes at every layer before promoting fused RMS.

Primary sources:
https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/cuda/Reduce.cuh
https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/qwen2/modeling_qwen2.py
https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/torch/_inductor/config.py
"""
from __future__ import annotations

import time

import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from transformers.integrations.sdpa_attention import sdpa_attention_forward

def _begin(x):
    xf = x.to(torch.float32)
    return xf, xf.pow(2)

def _silu_mul(gate, up):
    return F.silu(gate) * up

@triton.jit
def _round_bf16_fp32(value):
    return tl.inline_asm_elementwise(
        """{
            .reg .b16 rounded;
            cvt.rn.bf16.f32 rounded, $1;
            cvt.f32.bf16 $0, rounded;
        }""", "=f,f", [value], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def _add_begin_kernel(X, Y, Value, XF, Squared, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < N
    x = tl.load(X + index, mask, other=0).to(tl.float32)
    y = tl.load(Y + index, mask, other=0).to(tl.float32)
    value = _round_bf16_fp32(x + y)
    tl.store(Value + index, value, mask)
    tl.store(XF + index, value, mask)
    tl.store(Squared + index, value * value, mask)

def _triton_add_begin(x, y):
    value = torch.empty_like(x)
    xf = torch.empty_like(x, dtype=torch.float32)
    squared = torch.empty_like(x, dtype=torch.float32)
    _add_begin_kernel[(triton.cdiv(x.numel(), 256),)](
        x, y, value, xf, squared, x.numel(), 256,
        num_warps=4, enable_fp_fusion=False)
    return value, xf, squared

@triton.jit
def _rounded_product_sum(a, b, c, d):
    # Keep both eager BF16 product-rounding boundaries opaque to the compiler.
    # v3's torch casts were absent from generated Inductor code even with
    # emulate_precision_casts=True. PTX inline assembly explicitly prevents
    # contraction/elision: mul -> BF16 RN -> FP32 -> add -> BF16 store.
    # https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
    return tl.inline_asm_elementwise(
        """{
            .reg .f32 p0, p1, t0, t1;
            .reg .b16 r0, r1;
            mul.rn.f32 p0, $1, $2;
            mul.rn.f32 p1, $3, $4;
            cvt.rn.bf16.f32 r0, p0;
            cvt.rn.bf16.f32 r1, p1;
            cvt.f32.bf16 t0, r0;
            cvt.f32.bf16 t1, r1;
            add.rn.f32 $0, t0, t1;
        }""", "=f,f,f,f,f", [a, b, c, d], dtype=tl.float32,
        is_pure=True, pack=1)

@triton.jit
def _rotate_sign(value, dim, head: tl.constexpr):
    # Triton unary minus can lower to +0-value and lose negative0. Original
    # rotate_half flips the sign bit, so preserve signed0 explicitly as well.
    bits = value.to(tl.uint32, bitcast=True)
    negated = (bits ^ tl.full(value.shape, 0x80000000, tl.uint32)).to(tl.float32, bitcast=True)
    return tl.where(dim < head // 2, negated, value)

@triton.jit
def _rotary_kernel(Q, K, Cos, Sin, OQ, OK, NQ: tl.constexpr,
                   NK: tl.constexpr, HEAD: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    is_q = index < NQ
    local = tl.where(is_q, index, index - NQ)
    valid = index < NQ + NK
    dim = local % HEAD
    rotated_index = local // HEAD * HEAD + (dim + HEAD // 2) % HEAD
    value = tl.where(is_q,
                     tl.load(Q + local, valid & is_q, other=0).to(tl.float32),
                     tl.load(K + local, valid & ~is_q, other=0).to(tl.float32))
    rotated = tl.where(is_q,
                       tl.load(Q + rotated_index, valid & is_q, other=0).to(tl.float32),
                       tl.load(K + rotated_index, valid & ~is_q, other=0).to(tl.float32))
    rotated = _rotate_sign(rotated, dim, HEAD)
    cosine = tl.load(Cos + dim, valid, other=0).to(tl.float32)
    sine = tl.load(Sin + dim, valid, other=0).to(tl.float32)
    value = _rounded_product_sum(value, cosine, rotated, sine)
    tl.store(OQ + local, value, valid & is_q)
    tl.store(OK + local, value, valid & ~is_q)

def _triton_rotary(q, k, cos, sin):
    if q.dtype != torch.bfloat16 or not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("Exact Triton RoPE requires contiguous BF16 decode Q/K")
    oq, ok = torch.empty_like(q), torch.empty_like(k)
    total = q.numel() + k.numel()
    _rotary_kernel[(triton.cdiv(total, 256),)](
        q, k, cos, sin, oq, ok, q.numel(), k.numel(), q.shape[-1], 256,
        num_warps=4, enable_fp_fusion=False)
    return oq, ok

@triton.jit
def _rotary_packed_cache_kernel(Q, K, V, Cos, Sin, OldK, OldV, OutQ, OutK, OutV,
                                NQ: tl.constexpr, KV_HEADS: tl.constexpr,
                                PREFIX: tl.constexpr, HEAD: tl.constexpr,
                                BLOCK: tl.constexpr, Q_BLOCKS: tl.constexpr,
                                APPEND_BLOCKS: tl.constexpr):
    block = tl.program_id(0)
    if block < Q_BLOCKS:
        index = block * BLOCK + tl.arange(0, BLOCK)
        valid = index < NQ
        dim = index % HEAD
        rotated_index = index // HEAD * HEAD + (dim + HEAD // 2) % HEAD
        value = tl.load(Q + index, valid, other=0).to(tl.float32)
        rotated = tl.load(Q + rotated_index, valid, other=0).to(tl.float32)
        rotated = _rotate_sign(rotated, dim, HEAD)
        cosine = tl.load(Cos + dim, valid, other=0).to(tl.float32)
        sine = tl.load(Sin + dim, valid, other=0).to(tl.float32)
        output = _rounded_product_sum(value, cosine, rotated, sine)
        tl.store(OutQ + index, output, valid)
    elif block < Q_BLOCKS + APPEND_BLOCKS:
        index = (block - Q_BLOCKS) * BLOCK + tl.arange(0, BLOCK)
        valid = index < KV_HEADS * HEAD
        head = index // HEAD
        dim = index % HEAD
        current = head * HEAD + dim
        rotated_index = head * HEAD + (dim + HEAD // 2) % HEAD
        key = tl.load(K + current, valid, other=0).to(tl.float32)
        rotated = tl.load(K + rotated_index, valid, other=0).to(tl.float32)
        rotated = _rotate_sign(rotated, dim, HEAD)
        cosine = tl.load(Cos + dim, valid, other=0).to(tl.float32)
        sine = tl.load(Sin + dim, valid, other=0).to(tl.float32)
        key = _rounded_product_sum(key, cosine, rotated, sine)
        value = tl.load(V + current, valid, other=0)
        destination = head * (PREFIX + 1) * HEAD + PREFIX * HEAD + dim
        tl.store(OutK + destination, key, valid)
        tl.store(OutV + destination, value, valid)
    else:
        if PREFIX > 0:
            # Copy-only blocks contain no RoPE work. Both output prefixes keep
            # DynamicCache's exact contiguous layout; slots(p,p+1) are disjoint.
            source = (block - Q_BLOCKS - APPEND_BLOCKS) * BLOCK + tl.arange(0, BLOCK)
            valid = source < KV_HEADS * PREFIX * HEAD
            head = source // (PREFIX * HEAD)
            destination = head * (PREFIX + 1) * HEAD + source % (PREFIX * HEAD)
            old_key = tl.load(OldK + source, valid, other=0)
            old_value = tl.load(OldV + source, valid, other=0)
            # Copy BF16 storage directly, including all old bit patterns.
            tl.store(OutK + destination, old_key, valid)
            tl.store(OutV + destination, old_value, valid)

def _triton_rotary_packed_cache(q, k, v, cos, sin, cache_layer, prefix_length):
    """One launch emits rotated Q and exact contiguous K/V prefixes of p+1."""
    if any(value.dtype != torch.bfloat16 or not value.is_contiguous() for value in (q, k, v)):
        raise ValueError("Fused RoPE/cache requires contiguous BF16 single-query Q/K/V")
    if q.shape[-2] != 1 or k.shape[-2] != 1 or v.shape != k.shape:
        raise ValueError("Fused RoPE/cache supports single-query decode only")
    old_k, old_v = cache_layer.key_slots[prefix_length], cache_layer.value_slots[prefix_length]
    out_k, out_v = cache_layer.key_slots[prefix_length + 1], cache_layer.value_slots[prefix_length + 1]
    expected = (1, k.shape[1], prefix_length + 1, k.shape[-1])
    if tuple(out_k.shape) != expected or not out_k.is_contiguous() or not out_v.is_contiguous():
        raise ValueError("Fused RoPE/cache requires exact-length contiguous destination slots")
    out_q = torch.empty_like(q)
    block = 256
    q_blocks = triton.cdiv(q.numel(), block)
    append_blocks = triton.cdiv(k.numel(), block)
    prefix_blocks = triton.cdiv(old_k.numel(), block)
    _rotary_packed_cache_kernel[(q_blocks + append_blocks + prefix_blocks,)](
        q, k, v, cos, sin, old_k, old_v, out_q, out_k, out_v,
        q.numel(), k.shape[1], prefix_length, k.shape[-1], block, q_blocks, append_blocks,
        num_warps=4, enable_fp_fusion=False)
    return out_q, out_k, out_v

def _add(x, y):
    return x + y

def compile_options(emulate_casts=True):
    # Both are present in pinned torch2.9.1. There is no public Inductor config
    # named enable_fp_fusion in this version; do not silently invent that knob.
    return {"emulate_precision_casts": emulate_casts, "use_fast_math": False}


def mean_plan(hidden, threads=None):
    """Predicted contiguous one-output ATen vector4 mean plan; profile required."""
    if hidden < 256 or hidden % 4:
        raise ValueError("This source-order port requires a width divisible by4 and>=256")
    predicted = min(512, 1 << ((hidden // 4).bit_length() - 1))
    threads = predicted if threads is None else threads
    if threads < 32 or threads > 512 or threads & (threads - 1):
        raise ValueError("Mean threads must be a power of two from32 through512")
    return {"hidden": hidden, "threads": threads, "vector": 4,
            "chunks": triton.cdiv(hidden, 4 * threads),
            "shared_offsets": [threads >> shift for shift in range(1, threads.bit_length())
                               if threads >> shift >= 32],
            "warp_offsets": [1, 2, 4, 8, 16], "profile_confirmed": False}


@triton.jit
def _rms_tail_kernel(XF, Variance, Weight, Output,
                      N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    x = tl.load(XF + index, valid, other=0)
    variance = tl.load(Variance)
    factor = libdevice.rsqrt(variance + EPS)
    rounded = _round_bf16_fp32(x * factor)
    weight = tl.load(Weight + index, valid, other=0).to(tl.float32)
    tl.store(Output + index, weight * rounded, valid)


def split_rms_tail(xf, variance, weight, eps):
    output = torch.empty(xf.shape, dtype=torch.bfloat16, device=xf.device)
    _rms_tail_kernel[(triton.cdiv(xf.numel(), 256),)](
        xf, variance, weight, output, xf.numel(), eps, 256,
        num_warps=4, enable_fp_fusion=False)
    return output


@triton.jit
def _rms_source_order_kernel(X, Residual, Weight, Value, Output,
                              N: tl.constexpr, THREADS: tl.constexpr,
                              LOG_THREADS: tl.constexpr, BLOCK: tl.constexpr,
                              EPS: tl.constexpr, HAS_RESIDUAL: tl.constexpr):
    index = tl.arange(0, BLOCK)
    valid = index < N
    xf = tl.load(X + index, valid, other=0).to(tl.float32)
    if HAS_RESIDUAL:
        other = tl.load(Residual + index, valid, other=0).to(tl.float32)
        xf = _round_bf16_fp32(xf + other)
        tl.store(Value + index, xf, valid)
    square = xf * xf
    thread = tl.arange(0, THREADS)
    a0 = tl.full((THREADS,), 0.0, tl.float32)
    a1 = tl.full((THREADS,), 0.0, tl.float32)
    a2 = tl.full((THREADS,), 0.0, tl.float32)
    a3 = tl.full((THREADS,), 0.0, tl.float32)
    for chunk in tl.static_range((N + 4 * THREADS - 1) // (4 * THREADS)):
        base = chunk * 4 * THREADS + 4 * thread
        # The four independent per-thread accumulation chains are source
        # order, followed by source's left-associative accumulator merge.
        # Gathers need in-bounds indices even for the masked final chunk.
        a0 = a0 + tl.where(base < N, tl.gather(square, tl.minimum(base, BLOCK - 1), 0), 0.0)
        a1 = a1 + tl.where(base + 1 < N, tl.gather(square, tl.minimum(base + 1, BLOCK - 1), 0), 0.0)
        a2 = a2 + tl.where(base + 2 < N, tl.gather(square, tl.minimum(base + 2, BLOCK - 1), 0), 0.0)
        a3 = a3 + tl.where(base + 3 < N, tl.gather(square, tl.minimum(base + 3, BLOCK - 1), 0), 0.0)
    summed = ((a0 + a1) + a2) + a3
    # ATen's shared-memory offsets descend to32, then warp offsets ascend.
    for stage in tl.static_range(LOG_THREADS - 5):
        distance = THREADS >> (stage + 1)
        summed = summed + tl.gather(summed, (thread + distance) % THREADS, 0)
    for stage in tl.static_range(5):
        distance = 1 << stage
        lane = thread % 32
        peer = thread // 32 * 32 + tl.where(lane + distance < 32, lane + distance, lane)
        summed = summed + tl.gather(summed, peer, 0)
    total = tl.sum(tl.where(thread == 0, summed, 0.0), 0)
    variance = total * (1.0 / N)
    factor = libdevice.rsqrt(variance + EPS)
    rounded = _round_bf16_fp32(xf * factor)
    weight = tl.load(Weight + index, valid, other=0).to(tl.float32)
    tl.store(Output + index, weight * rounded, valid)


def source_order_rms(x, weight, eps, residual=None, threads=None, warps=4):
    """Unverified-on3B fused source-order candidate, returning residual and norm."""
    if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.shape[:2] != (1, 1):
        raise ValueError("Use contiguous BF16 [1,1,hidden] input")
    n = x.numel()
    plan = mean_plan(n, threads)
    value = torch.empty_like(x) if residual is not None else x
    output = torch.empty_like(x)
    _rms_source_order_kernel[(1,)](
        x, residual if residual is not None else x, weight, value, output,
        n, plan["threads"], plan["threads"].bit_length() - 1,
        triton.next_power_of_2(n), eps, residual is not None,
        num_warps=warps, enable_fp_fusion=False)
    return value, output


class Exact3BDecoder(nn.Module):
    """Original BF16 projections; optional audited pointwise/cache fusions.

    norm_mode='split' keeps original ATen mean; 'triton' uses the explicit
    source-order candidate; 'eager' uses original source RMSNorm modules.
    Separate source projections are default. combine_projections=True makes
    exact-value KV/gate-up copies, keeping Q's original reduction geometry.
    combine_mode='qkv' is an experimental alternative. Every changed cuBLAS
    shape needs raw-byte audits; original K/V use a different lane tree to Q.
    """
    def __init__(self, model, cache, norm_mode="split", pointwise="compiled",
                 emulate_casts=True, cache_rope=True, combine_projections=False,
                 gate_mode="compiled", fuse_cache=False, mean_threads=None,
                 norm_warps=4, combine_mode="kv"):
        super().__init__()
        if norm_mode not in {"eager", "split", "triton"}:
            raise ValueError("Use eager,split,ortriton RMS arithmetic")
        if pointwise not in {"eager", "compiled"} or gate_mode not in {"eager", "compiled"}:
            raise ValueError("Use eager or compiled pointwise/SiLU arithmetic")
        if combine_mode not in {"kv", "qkv"}:
            raise ValueError("Combined projection mode must be kv or experimental qkv")
        if next(model.parameters()).dtype != torch.bfloat16:
            raise ValueError("This core requires the original BF16 model")
        if fuse_cache and not all(hasattr(layer, "key_slots") and hasattr(layer, "value_slots")
                                  for layer in cache.layers):
            raise ValueError("Fused RoPE/cache requires per-position packed K/V slots")
        self.source, self.cache = model, cache
        self.norm_mode, self.pointwise, self.gate_mode = norm_mode, pointwise, gate_mode
        self.cache_rope, self.fuse_cache = cache_rope, fuse_cache
        self.mean_threads, self.norm_warps = mean_threads, norm_warps
        self.mean_plan = mean_plan(model.config.hidden_size, mean_threads)
        self.rope_values = {}
        self.combine_projections = combine_projections
        self.combine_mode = combine_mode if combine_projections else "none"
        self.projection_grouping = {
            "q": "original" if not combine_projections or combine_mode == "kv" else "combined_qkv",
            "k_v": "original_separate" if not combine_projections else "combined_" + combine_mode,
            "gate_up": "combined" if combine_projections else "original_separate",
            "o_down": "original_separate",
        }
        self.extra_projection_bytes = 0
        self.projection_copy_setup_s = 0.0
        if combine_projections:
            device = model.model.embed_tokens.weight.device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            def store_copy(name, value):
                self.register_buffer(name, value.detach() if value is not None else None, persistent=False)
                if value is not None:
                    self.extra_projection_bytes += value.numel() * value.element_size()
            with torch.no_grad():
                for index, layer in enumerate(model.model.layers):
                    attention = layer.self_attn
                    projections = ((attention.k_proj, attention.v_proj) if combine_mode == "kv" else
                                   (attention.q_proj, attention.k_proj, attention.v_proj))
                    biases = [projection.bias for projection in projections]
                    if any(bias is None for bias in biases) and not all(bias is None for bias in biases):
                        raise ValueError("Combined attention projections require all biases present or all absent")
                    if layer.mlp.gate_proj.bias is not None or layer.mlp.up_proj.bias is not None:
                        raise ValueError("Combined gate/up expects the pinned bias-free Qwen MLP")
                    store_copy(f"{combine_mode}_weight_{index}", torch.cat([projection.weight for projection in projections], dim=0))
                    store_copy(f"{combine_mode}_bias_{index}", torch.cat(biases, dim=0) if biases[0] is not None else None)
                    store_copy(f"gate_up_weight_{index}", torch.cat((layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight), dim=0))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.projection_copy_setup_s = time.perf_counter() - started
        options = compile_options(emulate_casts)
        self.begin = torch.compile(_begin, fullgraph=True, options=options) if pointwise == "compiled" else _begin
        self.rotary = _triton_rotary if pointwise == "compiled" else apply_rotary_pos_emb
        self.gate = (torch.compile(_silu_mul, fullgraph=True, options=options)
                     if pointwise == "compiled" and gate_mode == "compiled" else _silu_mul)

    def _normalize(self, x, module, residual=None):
        if self.norm_mode == "triton":
            return source_order_rms(x, module.weight, module.variance_epsilon,
                                    residual, self.mean_threads, self.norm_warps)
        if self.norm_mode == "eager":
            x = x + residual if residual is not None else x
            return x, module(x)
        if residual is None:
            xf, square = self.begin(x)
        elif self.pointwise == "compiled":
            x, xf, square = _triton_add_begin(x, residual)
        else:
            x = x + residual
            xf, square = _begin(x)
        variance = square.mean(-1, keepdim=True)
        if self.pointwise == "compiled":
            normalized = split_rms_tail(xf, variance, module.weight, module.variance_epsilon)
        else:
            normalized = module.weight * (xf * torch.rsqrt(variance + module.variance_epsilon)).to(x.dtype)
        return x, normalized

    @torch.no_grad()
    def forward_one(self, input_ids, position, prefix_length, trace=None):
        self.cache.set_position(prefix_length)
        source = self.source.model
        x = source.embed_tokens(input_ids)
        def save(key, value):
            if trace is not None:
                trace[key] = value.detach().clone()
        save("embedding", x)
        if not self.cache_rope or prefix_length not in self.rope_values:
            cos, sin = source.rotary_emb(x, position.view(1, 1))
            if self.cache_rope:
                self.rope_values[prefix_length] = (cos.detach().clone(), sin.detach().clone())
        else:
            cos, sin = self.rope_values[prefix_length]
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
                    k_raw, v_raw = joined.split((attention.k_proj.out_features,
                                                attention.v_proj.out_features), dim=-1)
                else:
                    q_raw, k_raw, v_raw = joined.split((attention.q_proj.out_features,
                                                       attention.k_proj.out_features,
                                                       attention.v_proj.out_features), dim=-1)
            else:
                q_raw = attention.q_proj(normalized)
                k_raw = attention.k_proj(normalized)
                v_raw = attention.v_proj(normalized)
            save(key + ".q_proj", q_raw)
            save(key + ".k_proj", k_raw)
            save(key + ".v_proj", v_raw)
            q = q_raw.view(hidden_shape).transpose(1, 2)
            k = k_raw.view(hidden_shape).transpose(1, 2)
            v = v_raw.view(hidden_shape).transpose(1, 2)
            if self.fuse_cache:
                q, k, v = _triton_rotary_packed_cache(q, k, v, cos, sin, self.cache.layers[index], prefix_length)
                save(key + ".q_rope", q)
                save(key + ".k_rope", k[:, :, prefix_length:prefix_length + 1])
            else:
                q, k = self.rotary(q, k, cos, sin)
                save(key + ".q_rope", q)
                save(key + ".k_rope", k)
                k, v = self.cache.update(k, v, index, {"cache_position": position, "cos": cos, "sin": sin})
            attended, _ = sdpa_attention_forward(attention, q, k, v, None, dropout=0.0,
                scaling=attention.scaling, sliding_window=attention.sliding_window)
            save(key + ".attention", attended)
            attended = attended.reshape(*input_shape, -1).contiguous()
            projected = attention.o_proj(attended)
            save(key + ".o_proj", projected)
            x, normalized = self._normalize(residual, layer.post_attention_layernorm, projected)
            save(key + ".post_residual", x)
            save(key + ".post_norm", normalized)
            if self.combine_projections:
                joined = F.linear(normalized, getattr(self, f"gate_up_weight_{index}"))
                gate, up = joined.split((layer.mlp.gate_proj.out_features,
                                         layer.mlp.up_proj.out_features), dim=-1)
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


# Compatible import name for root's existing integration; no runner/capture or
# checkpoint-loading side effects are introduced by this standalone module.
OriginalShapeDecoder = Exact3BDecoder
