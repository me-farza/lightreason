"""Small, inference-only Qwen2 decoder with fixed-address KV storage.

This changes execution, not checkpoint values: q/k/v and gate/up projections
are concatenated copies of the source tensors. No quantization or training is
performed. It supports ordinary full-attention Qwen2/Qwen2.5 checkpoints.

``engine = OptimizedQwen(model, max_len=1024)``
``logits = engine.prefill(ids)`` initializes the caches and returns [B, vocab].
``engine.decode(token, position)`` consumes [B, 1] tokens at a GPU int64
position tensor of shape [1]. The caller owns token selection and CUDA graphs.

Reference equations: Hugging Face's transformers/models/qwen2/modeling_qwen2.py.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _buffer(module, name, value):
    module.register_buffer(name, value.detach(), persistent=False)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _norm(x, weight, eps, exact_norm):
    if exact_norm:
        # Preserve the reference implementation's BF16 rounding boundary.
        xf = x.float()
        return weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype)
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


class _Layer(nn.Module):
    def __init__(self, source, config, batch_size, max_len, exact_norm, attention):
        super().__init__()
        a, m = source.self_attn, source.mlp
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.heads
        self.groups = self.heads // self.kv_heads
        self.hidden = config.hidden_size
        self.q_size = self.heads * self.head_dim
        self.kv_size = self.kv_heads * self.head_dim
        self.intermediate = config.intermediate_size
        self.eps = config.rms_norm_eps
        self.exact_norm = exact_norm
        self.attention = attention
        _buffer(self, "qkv_weight", torch.cat((a.q_proj.weight, a.k_proj.weight, a.v_proj.weight)))
        biases = [p.bias for p in (a.q_proj, a.k_proj, a.v_proj)]
        _buffer(self, "qkv_bias", torch.cat(biases) if all(b is not None for b in biases) else None)
        _buffer(self, "o_weight", a.o_proj.weight)
        _buffer(self, "gate_up_weight", torch.cat((m.gate_proj.weight, m.up_proj.weight)))
        _buffer(self, "down_weight", m.down_proj.weight)
        _buffer(self, "input_norm", source.input_layernorm.weight)
        _buffer(self, "post_norm", source.post_attention_layernorm.weight)
        shape = (batch_size, self.kv_heads, max_len, self.head_dim)
        _buffer(self, "k_cache", torch.zeros(shape, device=a.q_proj.weight.device, dtype=a.q_proj.weight.dtype))
        _buffer(self, "v_cache", torch.zeros_like(self.k_cache))

    def _qkv(self, x, cos, sin):
        b, s, _ = x.shape
        z = F.linear(_norm(x, self.input_norm, self.eps, self.exact_norm), self.qkv_weight, self.qkv_bias)
        q, k, v = z.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = q.view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        return q, k, v

    def _finish(self, residual, attention):
        b, _, s, _ = attention.shape
        x = residual + F.linear(attention.transpose(1, 2).reshape(b, s, self.q_size), self.o_weight)
        gate, up = F.linear(_norm(x, self.post_norm, self.eps, self.exact_norm), self.gate_up_weight).split(self.intermediate, dim=-1)
        return x + F.linear(F.silu(gate) * up, self.down_weight)

    def prefill(self, x, cos, sin):
        q, k, v = self._qkv(x, cos, sin)
        s = x.shape[1]
        self.k_cache[:, :, :s].copy_(k)
        self.v_cache[:, :, :s].copy_(v)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self._finish(x, attention)

    def decode(self, x, cos, sin, position, mask):
        q, k, v = self._qkv(x, cos, sin)
        self.k_cache.index_copy_(2, position, k)
        self.v_cache.index_copy_(2, position, v)
        if self.attention == "sdpa":
            # Match HF's masked-SDPA path: explicit KV repetition permits the
            # fused CUDA attention backend, retaining its accumulation rules.
            k_full = self.k_cache.repeat_interleave(self.groups, dim=1)
            v_full = self.v_cache.repeat_interleave(self.groups, dim=1)
            attention = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask)
        else:
            # Grouped matmuls avoid materializing 7 copies of the KV cache.
            b = x.shape[0]
            grouped_q = q.reshape(b, self.kv_heads, self.groups, self.head_dim)
            scores = (grouped_q @ self.k_cache.transpose(-2, -1)) * (self.head_dim ** -0.5)
            scores = scores.masked_fill(~mask, float("-inf"))
            probability = scores.float().softmax(-1).to(q.dtype)
            attention = (probability @ self.v_cache).reshape(b, self.heads, 1, self.head_dim)
        return self._finish(x, attention)


class OptimizedQwen(nn.Module):
    """Inference engine; fixed batch size, no padding, no sliding attention.

    Set ``exact_norm=False`` to use the native F.rms_norm kernel, which may
    introduce slightly different BF16 rounding. The default follows HF's norm
    expression so greedy-token checks can isolate other arithmetic changes.
    ``attention='matmul'`` is a useful fallback on SDPA versions whose grouped
    attention with a mask selects a slow backend.
    """

    def __init__(self, model, max_len=1024, batch_size=1, exact_norm=True, attention="sdpa"):
        super().__init__()
        c = model.config
        if getattr(c, "model_type", None) != "qwen2":
            raise ValueError("This engine supports Qwen2/Qwen2.5 only")
        if getattr(c, "use_sliding_window", False):
            raise ValueError("Sliding-window attention is unsupported")
        if getattr(c, "hidden_act", "silu") != "silu":
            raise ValueError("Only SwiGLU/SiLU checkpoints are supported")
        if attention not in ("sdpa", "matmul"):
            raise ValueError("attention must be 'sdpa' or 'matmul'")
        self.max_len = max_len
        self.batch_size = batch_size
        self.eps = c.rms_norm_eps
        self.exact_norm = exact_norm
        _buffer(self, "embedding", model.model.embed_tokens.weight)
        _buffer(self, "lm_head", model.lm_head.weight)
        _buffer(self, "norm_weight", model.model.norm.weight)
        self.layers = nn.ModuleList([
            _Layer(layer, c, batch_size, max_len, exact_norm, attention)
            for layer in model.model.layers
        ])
        # Use the source model's inverse frequencies and scaling, rather than
        # reconstructing them from a potentially version-dependent config.
        rotary = model.model.rotary_emb
        inv_freq = rotary.inv_freq.detach().float().to(self.embedding.device)
        positions = torch.arange(max_len, device=self.embedding.device, dtype=torch.float32)
        freqs = positions[:, None] * inv_freq[None, :]
        angles = torch.cat((freqs, freqs), dim=-1)
        scale = getattr(rotary, "attention_scaling", 1.0)
        _buffer(self, "cos", (angles.cos() * scale).to(self.embedding.dtype))
        _buffer(self, "sin", (angles.sin() * scale).to(self.embedding.dtype))
        _buffer(self, "cache_positions", torch.arange(max_len, device=self.embedding.device))
        self.eval()

    @torch.no_grad()
    def prefill(self, input_ids):
        b, s = input_ids.shape
        if b != self.batch_size or not 0 < s <= self.max_len:
            raise ValueError("Prompt batch/length does not fit this engine")
        x = F.embedding(input_ids, self.embedding)
        cos = self.cos[:s][None, None]
        sin = self.sin[:s][None, None]
        for layer in self.layers:
            x = layer.prefill(x, cos, sin)
        return F.linear(_norm(x[:, -1], self.norm_weight, self.eps, self.exact_norm), self.lm_head)

    @torch.no_grad()
    def decode(self, input_ids, position):
        # position is deliberately a GPU tensor so graph replay can advance it
        # in-place without rebuilding the graph or synchronizing with the CPU.
        x = F.embedding(input_ids, self.embedding)
        cos = self.cos.index_select(0, position)[None, None]
        sin = self.sin.index_select(0, position)[None, None]
        mask = self.cache_positions.view(1, 1, 1, -1) <= position.view(1, 1, 1, 1)
        for layer in self.layers:
            x = layer.decode(x, cos, sin, position, mask)
        return F.linear(_norm(x[:, -1], self.norm_weight, self.eps, self.exact_norm), self.lm_head)

    def clear_cache(self):
        """Optional hygiene; prefill+causal masking already excludes old data."""
        for layer in self.layers:
            layer.k_cache.zero_()
            layer.v_cache.zero_()
