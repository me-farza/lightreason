"""Immutable attention-prefix views over one existing full HF StaticCache.

Only returned K/V shapes change. The owner keeps every original layer object,
full-capacity tensor, pointer and head stride. No view is assigned back into
layer.keys/values, and no storage is allocated by this adapter. Cache updates
must use explicit physical positions plus a prepared4D attention mask.

Pinned contracts: Transformers4.57.1 Cache/StaticLayer.update and Qwen2Attention
use the tensors returned by update; prepared4D masks bypass size inference.
https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/cache_utils.py
https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/masking_utils.py
https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/integrations/sdpa_attention.py
"""
from __future__ import annotations

from transformers.cache_utils import Cache


DEFAULT_ATTENTION_BUCKETS = (256, 512, 1024, 2048)


def attention_buckets(capacity, buckets=None):
    if type(capacity) is not int or capacity < 1:
        raise ValueError("Use a positive full physical capacity")
    values = tuple(bucket for bucket in DEFAULT_ATTENTION_BUCKETS if bucket <= capacity) if buckets is None else tuple(buckets)
    if not values or any(type(bucket) is not int or not 1 <= bucket <= capacity for bucket in values):
        raise ValueError("Attention buckets must be positive integers within full capacity")
    return tuple(sorted(set(values)))


def select_attention_bucket(required_end, buckets):
    """CPU metadata only; required_end is the exclusive physical write end."""
    if type(required_end) is not int or required_end < 1:
        raise ValueError("Required exclusive end must be a positive CPU integer")
    for bucket in buckets:
        if required_end <= bucket:
            return bucket
    raise ValueError(f"No prepared attention bucket covers physical end {required_end}")


def public_tree_families(prompt_length, max_new_tokens, node_counts, buckets):
    """All (M,bucket) families possible from a PUBLIC prompt/output budget.

    One original-HF prefill anchor is already emitted. A decode call can start
    at p=prompt_length through p=prompt_length+max_new_tokens-2 inclusive.
    No reference output or EOS position is an input.
    """
    if type(prompt_length) is not int or prompt_length < 1 or type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("Use a positive public prompt length and output budget")
    sizes = tuple(node_counts)
    if not sizes or any(type(size) is not int or size < 1 for size in sizes):
        raise ValueError("Use positive actual physical query-row counts")
    if max_new_tokens == 1:
        return ()
    result = set()
    for size in sizes:
        first_end = prompt_length + size
        last_end = prompt_length + max_new_tokens + size - 2
        select_attention_bucket(last_end, buckets)  # Reject infeasible public bounds.
        previous = 0
        for bucket in buckets:
            if first_end <= bucket and last_end >= previous + 1:
                result.add((size, bucket))
            previous = bucket
    return tuple(sorted(result))


class StaticCachePrefixView(Cache):
    """A fixed-bucket Cache proxy sharing owner_cache.layers by identity.

    The owner may be lazily initialized (PARD prepares it before capture).
    assert_storage_consistent() binds each tensor identity the first time that
    tensor exists, then rejects replacement. Never call pointer introspection
    inside a captured/compiled model forward; update() does not do so.
    """
    _immutable_names = frozenset(("_owner_cache", "_bucket", "_capacity", "layers"))

    def __setattr__(self, name, value):
        if name in self._immutable_names and name in self.__dict__:
            raise AttributeError(f"{name} is immutable for a cache prefix view")
        super().__setattr__(name, value)

    def __init__(self, owner_cache, bucket):
        if not isinstance(owner_cache, Cache) or not owner_cache.layers:
            raise ValueError("Provide an existing full HF static cache owner")
        if getattr(owner_cache, "offloading", False):
            raise ValueError("This fixed GPU-storage view does not support cache offloading")
        capacity = owner_cache.get_max_cache_shape()
        if type(capacity) is not int or capacity < 1 or type(bucket) is not int or not 1 <= bucket <= capacity:
            raise ValueError("Prefix bucket must lie within the owner's full static capacity")
        if any(getattr(layer, "is_sliding", False) or not getattr(layer, "is_compileable", False)
               for layer in owner_cache.layers):
            raise ValueError("Use full-attention static cache layers")
        self._owner_cache, self._bucket, self._capacity = owner_cache, bucket, capacity
        super().__init__(layers=owner_cache.layers, offloading=False)
        self._layer_identity = tuple(id(layer) for layer in owner_cache.layers)
        self._storage_signatures = {}
        self.assert_storage_consistent(require_initialized=False)

    @property
    def owner_cache(self):
        return self._owner_cache

    @property
    def bucket(self):
        return self._bucket

    @property
    def capacity(self):
        return self._capacity

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        keys, values = self._owner_cache.update(key_states, value_states, layer_idx, cache_kwargs)
        return keys[:, :, :self._bucket, :], values[:, :, :self._bucket, :]

    def prefix_tensors(self, layer_idx):
        """Read-only view introspection; no update or logical-length inference."""
        layer = self._owner_cache.layers[layer_idx]
        if not getattr(layer, "is_initialized", False):
            raise RuntimeError("Initialize the owner's layer before requesting prefix tensors")
        return layer.keys[:, :, :self._bucket, :], layer.values[:, :, :self._bucket, :]

    def get_mask_sizes(self, cache_position, layer_idx=0):
        return self._bucket, 0

    def get_max_cache_shape(self, layer_idx=0):
        return self._capacity

    def get_seq_length(self, layer_idx=0):
        # This is HF's physical occupancy diagnostic, NOT an authority for the
        # committed prefix after speculative rollback. Explicit positions rule.
        return self._owner_cache.get_seq_length(layer_idx)

    def reset(self):
        self._owner_cache.reset()

    def assert_storage_consistent(self, require_initialized=True):
        if self.layers is not self._owner_cache.layers or tuple(id(layer) for layer in self.layers) != self._layer_identity:
            raise RuntimeError("Full cache layer objects were replaced")
        if self._owner_cache.get_max_cache_shape() != self._capacity:
            raise RuntimeError("Full cache physical capacity changed")
        initialized = 0
        for index, layer in enumerate(self.layers):
            for kind in ("keys", "values"):
                value = getattr(layer, kind, None)
                if not getattr(layer, "is_initialized", False) or value is None:
                    if require_initialized:
                        raise RuntimeError("Full cache has an uninitialized layer")
                    continue
                if value.ndim != 4 or value.shape[-2] != self._capacity or not value.is_contiguous():
                    raise RuntimeError("Full cache shape/contiguous backing layout changed")
                signature = (id(value), value.data_ptr(), value.untyped_storage().data_ptr(),
                    tuple(value.shape), tuple(value.stride()), value.storage_offset(), str(value.dtype), str(value.device))
                key = (index, kind)
                if key in self._storage_signatures and self._storage_signatures[key] != signature:
                    raise RuntimeError("Full cache tensor identity, pointer or stride changed")
                self._storage_signatures.setdefault(key, signature)
                initialized += 1
        return {"consistent": True, "layers": len(self.layers), "initialized_tensors": initialized,
                "bucket": self._bucket, "full_capacity": self._capacity,
                "same_layer_list": self.layers is self._owner_cache.layers}

    def metadata(self):
        return {"kind": "immutable_prefix_view_over_full_static_cache", "bucket": self._bucket,
            "full_capacity": self._capacity, "separate_cache_allocation": False,
            "updates_delegate_to_full_owner": True, "full_head_strides_preserved": True,
            "storage": self.assert_storage_consistent(require_initialized=False)}
