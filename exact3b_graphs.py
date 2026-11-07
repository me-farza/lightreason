"""Linear-memory, exact-length K/V storage for sequential Qwen CUDA graphs.

All even prefix lengths view one flat bank; odd lengths view the other. A
captured p->p+1 step therefore has disjoint input/output storage, with the exact
contiguous [1,kv_heads,p,head_dim] strides of HF DynamicCache at each length.
Views retain their addresses across captures and replay. No attention padding,
mask, arithmetic change, precision change, or output cache is introduced.

Only sequential increasing-position replay is supported. Previous prefixes are
not snapshots: replay/capture overwrites same-parity views. The ordinary HF
prefill must restore the prompt state after preparation and between requests.

At 36 layers, 2 KV heads, head_dim128 and BF16, capacity1024 uses72MiB for
these banks, versus about18GiB for independent slots at every prefix length.
This excludes model weights, prefill temporaries and CUDA graph activations;
graph pool sharing is a separately validated opt-in.
"""
from __future__ import annotations

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from exact10_graphs import PackedPrefixLayer, PackedExactGraphRunner
from exact_graph import ExactNaturalEOS


def cache_storage_bytes(layers, kv_heads, head_dim, max_len, element_size=2):
    """Exact storage estimates, excluding views/graphs/allocator overhead."""
    if any(type(value) is not int or value <= 0 for value in
           (layers, kv_heads, head_dim, max_len, element_size)):
        raise ValueError("Cache dimensions and element size must be positive integers")
    one_token_all_layers_kv = 2 * layers * kv_heads * head_dim * element_size
    return {
        "pingpong_bank_bytes": one_token_all_layers_kv * max_len * 2,
        "all_independent_prefix_slots_bytes": one_token_all_layers_kv * max_len * (max_len + 1) // 2,
    }


class PingPongPrefixLayer(PackedPrefixLayer):
    """PackedPrefixLayer API, backed by exactly two banks per K or V tensor."""

    def __init__(self, kv_heads, head_dim, max_len, dtype, device):
        # Parent initialization calls ensure_slot before our banks would exist.
        CacheLayerMixin.__init__(self)
        if any(type(value) is not int or value <= 0 for value in (kv_heads, head_dim, max_len)):
            raise ValueError("Cache dimensions must be positive integers")
        self.position = 0
        self.kv_heads, self.head_dim = kv_heads, head_dim
        self.max_cache_len, self.max_batch_size = max_len, 1
        self.dtype, self.device = dtype, device
        capacity = kv_heads * max_len * head_dim
        self.key_banks = tuple(torch.zeros(capacity, dtype=dtype, device=device) for _ in range(2))
        self.value_banks = tuple(torch.zeros_like(bank) for bank in self.key_banks)
        self.key_slots, self.value_slots = {}, {}
        self.ensure_slot(0)
        self.keys, self.values = self.key_slots[0], self.value_slots[0]
        self.is_initialized = True

    def ensure_slot(self, length):
        if type(length) is not int or not 0 <= length <= self.max_cache_len:
            raise ValueError("Prefix length is outside cache capacity")
        if length not in self.key_slots:
            count = self.kv_heads * length * self.head_dim
            shape = (1, self.kv_heads, length, self.head_dim)
            parity = length % 2
            self.key_slots[length] = self.key_banks[parity].narrow(0, 0, count).view(shape)
            self.value_slots[length] = self.value_banks[parity].narrow(0, 0, count).view(shape)


class PingPongPrefixCache(Cache):
    def __init__(self, model, max_len):
        config = model.config
        weight = model.model.embed_tokens.weight
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        super().__init__(layers=[PingPongPrefixLayer(config.num_key_value_heads, head_dim, max_len,
                             weight.dtype, weight.device) for _ in range(config.num_hidden_layers)])

    def set_position(self, position):
        for layer in self.layers:
            layer.set_position(position)

    def copy_dynamic(self, source, length):
        if len(source.layers) != len(self.layers):
            raise ValueError("Dynamic prefill cache has a different number of layers")
        for destination, original in zip(self.layers, source.layers):
            destination.set_position(length)
            expected = (1, destination.kv_heads, length, destination.head_dim)
            if tuple(original.keys.shape) != expected or tuple(original.values.shape) != expected:
                raise ValueError("Dynamic prefill cache has different K/V dimensions")
            destination.key_slots[length].copy_(original.keys)
            destination.value_slots[length].copy_(original.values)

    def allocated_bytes(self):
        # Count unique backing banks once, never the overlapping views.
        return sum(bank.numel() * bank.element_size() for layer in self.layers
                   for bank in (*layer.key_banks, *layer.value_banks))


class LinearExactGraphRunner(PackedExactGraphRunner):
    """Original HF arithmetic; compact ping-pong cache, fixed-position graphs."""

    def __init__(self, model, max_len=1024, max_output=512, warmups=3,
                 share_pool=False, track_logits=False, precompute_rope=True):
        with torch.inference_mode(False), torch.no_grad():
            super().__init__(model, max_len, max_output, warmups, share_pool,
                             track_logits, precompute_rope)
            # No graphs or decoder have been constructed against the old cache.
            self.cache = PingPongPrefixCache(model, max_len)

    def memory_report(self):
        weight = self.model.model.embed_tokens.weight
        config = self.model.config
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        return {**cache_storage_bytes(config.num_hidden_layers, config.num_key_value_heads,
                  head_dim, self.max_len, weight.element_size()),
                "allocated_kv_bank_bytes": self.cache.allocated_bytes(),
                "graph_count": len(self.graphs),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(weight.device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(weight.device),
                "scope": "K/V banks scale linearly; graph activations and model/prefill are additional"}


def make_quality_runner(model, tokenizer, args):
    runner = LinearExactGraphRunner(
        model, max_len=args.max_len, max_output=args.max_len,
        warmups=getattr(args, "warmups", 3), share_pool=getattr(args, "share_pool", False),
        track_logits=getattr(args, "track_logits", False),
        precompute_rope=not getattr(args, "no_precompute_rope", False))
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    return ExactNaturalEOS(runner, tokenizer, eos)
