"""Separate FP-order-relaxed tree verifier with immutable attention buckets.

The original full HF StaticCache allocation and Triton compaction pointers are
inherited unchanged. Only the returned attention prefix and mask length vary
between explicitly prepared (physical query rows M, bucket) graph families.
Each bucket owns an immutable cache view and its own configured callable.

Prepare all public families BEFORE real prefill. Runtime verification rejects
an unprepared family; it never captures synthetic work over an active prefix.
Original-HF prefill restores the full owner cache before each request.
This module provides no measured speedup, exact-output or quality guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from bucketed_cache_view import (StaticCachePrefixView, attention_buckets,
                                 public_tree_families, select_attention_bucket)
from numerical3b_tree_verifier import NumericalTreeVerifier, _TreeEntry, validate_tree_cpu


@dataclass
class BucketedTreeEntry(_TreeEntry):
    bucket: int = 0
    cache_view: object = None
    key_positions: torch.Tensor | None = None
    forward_fn: object = None


class Bucketed3BTreeVerifier(NumericalTreeVerifier):
    """NumericalTreeVerifier-compatible verify/commit API, explicit graph setup.

    cache/full_cache always denote the SAME full-capacity HF owner. cache_views
    and entries[(M,bucket)] expose immutable view geometry for diagnostics.
    forward_for_bucket(bucket, compiled=True) returns the exact callable used
    inside that bucket's graphs, including when compile_static=False.
    """
    def __init__(self, model, max_len=2048, node_budget=64, *, cuda_graph=True,
                 compile_static=True, share_pool=False, warmups=3,
                 commit_engine="triton", validate_topology=True,
                 validate_acceptance=False, buckets=None):
        started = time.perf_counter()
        choices = attention_buckets(max_len, buckets)
        super().__init__(model, max_len=max_len, node_budget=node_budget,
            cuda_graph=cuda_graph, compile_static=False, share_pool=share_pool,
            warmups=warmups, commit_engine=commit_engine,
            validate_topology=validate_topology, validate_acceptance=validate_acceptance)
        self._full_cache = self.cache
        self.attention_buckets = choices
        self.compile_static = bool(compile_static)
        self.cache_views = {bucket: StaticCachePrefixView(self._full_cache, bucket) for bucket in choices}
        self._eager_bucket_forwards = {bucket: self._make_bucket_forward(view)
                                      for bucket, view in self.cache_views.items()}
        self._bucket_forwards = {}
        for bucket, eager in self._eager_bucket_forwards.items():
            self._bucket_forwards[bucket] = (torch.compile(eager, fullgraph=True, dynamic=False,
                options={"triton.cudagraphs": False}) if compile_static else eager)
        # Legacy diagnostic signature remains safe: select a view from mask
        # metadata instead of accidentally calling the full-capacity method.
        self._tree_forward_fn = self._dispatch_bucket_forward
        self._full_layer_list = self.cache.layers
        self._pointer_table_identity = (id(self.cache_pointer_table), self.cache_pointer_table.data_ptr(),
                                        tuple(self.cache_pointer_table.shape), str(self.cache_pointer_table.dtype))
        self.last_bucket, self.last_entry_key = None, None
        self.logical_reset_calls = 0
        self.assert_storage_consistent()
        torch.cuda.synchronize(self.device)
        self.constructor_s = time.perf_counter() - started
        self.metadata = {**self.metadata,
            "variant": "original_weight_relaxed_fp_order_bucketed_tree",
            "compile_static": self.compile_static, "attention_buckets": list(choices),
            "graph_key": "(actual physical query rows M, attention bucket)",
            "full_physical_capacity": max_len, "compaction_capacity": max_len,
            "full_cache_owner_and_pointer_table_preserved": True,
            "runtime_lazy_capture": False,
            "preparation": "public families before real prefill; logical-only reset between requests",
            "compiled_callables": "one immutable-cache-view callable per bucket; dynamic=False; no nested Inductor CUDA graphs",
            "new_speedup_or_quality_claim": False}

    @property
    def full_cache(self):
        return self._full_cache

    def _make_bucket_forward(self, view):
        model = self.model
        def forward(input_ids, physical_positions, rotary_positions, mask):
            output = model(input_ids=input_ids, attention_mask=mask,
                position_ids=rotary_positions.view(1, -1), cache_position=physical_positions,
                past_key_values=view, use_cache=True, logits_to_keep=input_ids.shape[1],
                return_dict=True, output_attentions=False, output_hidden_states=False)
            logits = output.logits.squeeze(0)
            return logits, logits.argmax(-1)
        return forward

    def forward_for_bucket(self, bucket, *, compiled=True):
        if bucket not in self.cache_views:
            raise ValueError("Unknown immutable attention bucket")
        return self._bucket_forwards[bucket] if compiled else self._eager_bucket_forwards[bucket]

    def _dispatch_bucket_forward(self, input_ids, physical_positions, rotary_positions, mask):
        return self.forward_for_bucket(mask.shape[-1])(
            input_ids, physical_positions, rotary_positions, mask)

    def select_bucket(self, required_end):
        if type(required_end) is not int or required_end > self.max_len:
            raise ValueError("Required physical end exceeds full cache capacity")
        return select_attention_bucket(required_end, self.attention_buckets)

    def bucket_for_end(self, required_end):
        return self.select_bucket(required_end)

    def assert_storage_consistent(self):
        if self.cache is not self._full_cache or self.cache.layers is not self._full_layer_list:
            raise RuntimeError("Bucketed verifier replaced the full cache owner or layer list")
        table = (id(self.cache_pointer_table), self.cache_pointer_table.data_ptr(),
                 tuple(self.cache_pointer_table.shape), str(self.cache_pointer_table.dtype))
        if table != self._pointer_table_identity:
            raise RuntimeError("Original full-capacity compaction pointer table was replaced")
        rows = {bucket: view.assert_storage_consistent(require_initialized=True)
                for bucket, view in self.cache_views.items()}
        return {"consistent": True, "full_capacity": self.max_len,
                "full_cache_owner_preserved": True, "compaction_pointer_table_preserved": True,
                "views": rows}

    def reset_for_new_request(self):
        """Invalidate logical state ONLY; next prefill restores genuine K/V.

        Call only between requests. No cache reset/zero, allocation, graph
        replay, prefix copy or pointer-table change is performed here.
        """
        self.current_position, self.anchor, self.pending = 0, None, None
        self._pending_entry, self._pending_topology = None, None
        self.last_logits, self.last_next_ids = None, None
        self.last_depths, self.last_ancestors, self.last_input_ids = None, None, None
        self.last_bucket, self.last_entry_key = None, None
        self.logical_reset_calls += 1
        self.last_report = {"operation": "logical_reset_for_new_request", "cache_bytes_untouched": True,
                            "original_hf_prefill_required": True}
        return dict(self.last_report)

    def _sizes(self, node_counts):
        sizes = ((self.node_budget,) if node_counts is None else (node_counts,) if type(node_counts) is int
                 else tuple(dict.fromkeys(node_counts)))
        if not sizes or any(type(size) is not int or not 1 <= size <= self.node_budget for size in sizes):
            raise ValueError("Prepare actual physical query counts1..node_budget")
        return sizes

    def _new_bucketed_entry(self, size, bucket):
        with torch.inference_mode(False), torch.no_grad():
            offsets = torch.arange(size, device=self.device, dtype=torch.long)
            entry = BucketedTreeEntry(
                input_ids=torch.zeros((1, size), device=self.device, dtype=torch.long),
                base=torch.zeros((1,), device=self.device, dtype=torch.long),
                offsets=offsets, physical_positions=torch.empty_like(offsets), rotary_positions=torch.empty_like(offsets),
                depths=offsets.clone(), ancestors=torch.ones((size, size), device=self.device, dtype=torch.bool).tril_(),
                mask=torch.empty((1, 1, size, bucket), device=self.device, dtype=torch.bool),
                bucket=bucket, cache_view=self.cache_views[bucket], key_positions=self.key_positions[..., :bucket],
                forward_fn=self.forward_for_bucket(bucket))
        self.entries[(size, bucket)] = entry
        return entry

    def _tree_step(self, entry):
        torch.add(entry.base, entry.offsets, out=entry.physical_positions)
        torch.add(entry.base, entry.depths, out=entry.rotary_positions)
        torch.lt(entry.key_positions.expand_as(entry.mask), entry.base.view(1, 1, 1, 1), out=entry.mask)
        entry.mask.index_copy_(3, entry.physical_positions, entry.ancestors.view(1, 1, *entry.ancestors.shape))
        entry.logits, entry.next_ids = entry.forward_fn(
            entry.input_ids, entry.physical_positions, entry.rotary_positions, entry.mask)
        return entry.next_ids

    @torch.no_grad()
    def _prepare_families(self, families):
        families = tuple(sorted(set(families)))
        if any(type(size) is not int or not 1 <= size <= self.node_budget or
               bucket not in self.cache_views or size > bucket for size, bucket in families):
            raise ValueError("Invalid physical query-count/attention-bucket family")
        missing = [(size, bucket) for size, bucket in families if (size, bucket) not in self.entries or
                   (self.cuda_graph and (size, bucket) not in self.graphs)]
        if not missing:
            return {"new_graphs": 0, "new_entries": 0, "setup_s": 0.0,
                    "requested_families": [list(key) for key in families]}
        if self.current_position != 0 or self.anchor is not None or self.pending is not None:
            raise RuntimeError("Reset logical state between requests before capturing missing bucket families")
        self.assert_storage_consistent()
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        for size, bucket in missing:
            entry = self.entries.get((size, bucket)) or self._new_bucketed_entry(size, bucket)
            entry.base.zero_()
            entry.input_ids.zero_()
            entry.depths.copy_(entry.offsets)
            entry.ancestors.fill_(True).tril_()
            warmup_s, capture_s = 0.0, 0.0
            if self.cuda_graph or self.compile_static:
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                warm_started = time.perf_counter()
                with torch.cuda.stream(stream), torch.inference_mode(False), torch.no_grad():
                    for _ in range(self.warmups):
                        self._tree_step(entry)
                torch.cuda.current_stream(self.device).wait_stream(stream)
                torch.cuda.synchronize(self.device)
                warmup_s = time.perf_counter() - warm_started
                if self.cuda_graph:
                    graph = torch.cuda.CUDAGraph()
                    capture_started = time.perf_counter()
                    with torch.cuda.graph(graph, pool=self.pool, stream=stream), torch.inference_mode(False), torch.no_grad():
                        self._tree_step(entry)
                    torch.cuda.synchronize(self.device)
                    capture_s = time.perf_counter() - capture_started
                    entry.graph, self.graphs[(size, bucket)] = graph, graph
            commit_warmup_s = self._warm_compaction(entry, size)
            self.capture_records.append({"nodes": size, "bucket": bucket,
                "warmup_s": warmup_s, "capture_s": capture_s, "commit_warmup_s": commit_warmup_s,
                "representative_base": 0, "logical_state_invalid_until_prefill": True})
        torch.cuda.synchronize(self.device)
        self.assert_storage_consistent()
        elapsed = time.perf_counter() - started
        self.setup_s += elapsed
        return {"new_graphs": len(missing) if self.cuda_graph else 0, "new_entries": len(missing),
                "setup_s": elapsed, "requested_families": [list(key) for key in families],
                "new_families": [list(key) for key in missing], "node_budget": self.node_budget}

    def prepare(self, node_counts=None, buckets=None):
        sizes = self._sizes(node_counts)
        selected = self.attention_buckets if buckets is None else (buckets,) if type(buckets) is int else tuple(buckets)
        if not selected or any(bucket not in self.cache_views for bucket in selected):
            raise ValueError("Prepare only configured immutable attention buckets")
        return self._prepare_families((size, bucket) for size in sizes for bucket in selected)

    def prepare_public(self, prompt_length, max_new_tokens, node_counts=None):
        sizes = self._sizes(node_counts)
        families = public_tree_families(prompt_length, max_new_tokens, sizes, self.attention_buckets)
        report = self._prepare_families(families)
        return {**report, "public_prompt_length": prompt_length, "public_max_new_tokens": max_new_tokens,
                "preparation_uses_future_output_or_eos": False}

    @torch.no_grad()
    def prefill(self, ids):
        self.assert_storage_consistent()
        anchor = super().prefill(ids)
        self.last_bucket, self.last_entry_key = None, None
        self.assert_storage_consistent()
        return anchor

    @torch.no_grad()
    def verify_tree(self, input_ids, depths, ancestors, basep=None):
        if not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Use original target LongTensor[1,M] tree inputs")
        size = input_ids.shape[1]
        if self.current_position < 1 or self.anchor is None:
            raise RuntimeError("Original-HF prefill is required before tree verification")
        if self.pending is not None:
            raise RuntimeError("Commit/discard the preceding tree before verifying another")
        basep = self.current_position if basep is None else basep
        if type(basep) is not int or basep != self.current_position:
            raise ValueError("basep must equal the committed contiguous prefix length")
        if not 1 <= size <= self.node_budget or basep + size > self.max_len:
            raise ValueError("Tree exceeds actual-node budget or full physical cache capacity")
        bucket = self.select_bucket(basep + size)
        key = (size, bucket)
        if key not in self.entries or (self.cuda_graph and key not in self.graphs):
            raise RuntimeError(f"Unprepared tree family {key}; prepare public families before real prefill")
        entry = self.entries[key]
        if entry.bucket != bucket or entry.cache_view is not self.cache_views[bucket] or entry.forward_fn is not self.forward_for_bucket(bucket):
            raise RuntimeError("Prepared graph/cache view identity is inconsistent")
        if basep + size > entry.bucket or entry.mask.shape != (1, 1, size, bucket):
            raise RuntimeError("Prepared graph cannot cover every physical tree slot")
        if self.cache is not self._full_cache or self.cache.layers is not self._full_layer_list:
            raise RuntimeError("The original full-capacity owner was replaced")
        if not isinstance(depths, torch.Tensor) or depths.shape != (size,) or depths.dtype not in (torch.int32, torch.int64):
            raise ValueError("Use integer depths[M]")
        if not isinstance(ancestors, torch.Tensor) or ancestors.shape != (size, size) or ancestors.dtype != torch.bool:
            raise ValueError("Use bool self-inclusive ancestors[M,M]")
        if any(value.device.type != "cpu" and value.device != self.device for value in (input_ids, depths, ancestors)):
            raise ValueError("Tree tensors must be CPU or on the original target device")
        topology = validate_tree_cpu(depths, ancestors) if self.validate_topology else None
        if self.validate_acceptance and int(input_ids[0, 0].item()) != int(self.anchor.item()):
            raise ValueError("Tree root must equal the preceding unprocessed target anchor")
        with torch.inference_mode(False), torch.no_grad():
            entry.input_ids.copy_(input_ids)
            entry.depths.copy_(depths)
            entry.ancestors.copy_(ancestors)
            entry.base.fill_(basep)
            if self.cuda_graph:
                entry.graph.replay()
            else:
                self._tree_step(entry)
        if entry.logits.shape != (size, self.model.config.vocab_size):
            raise RuntimeError("Bucketed target returned an unexpected classifier shape")
        self.pending = {"basep": basep, "node_count": size}
        self._pending_entry, self._pending_topology = entry, topology
        self.last_logits, self.last_next_ids = entry.logits, entry.next_ids
        self.last_depths, self.last_ancestors, self.last_input_ids = entry.depths, entry.ancestors, entry.input_ids
        self.last_bucket, self.last_entry_key = bucket, key
        self.verify_calls += 1
        self.last_report = {"operation": "verify_bucketed_tree", "basep": basep, "node_count": size,
            "bucket": bucket, "physical_end_guard_passed": basep + size <= bucket,
            "topology_validated": topology is not None, "strict_serial_agreement_not_assumed": True}
        return self.last_next_ids

    def memory_report(self):
        report = super().memory_report()
        return {**report, "tree_graph_node_counts": sorted({size for size, bucket in self.graphs}),
            "graph_keys": [list(key) for key in sorted(self.graphs)], "attention_buckets": list(self.attention_buckets),
            "full_physical_capacity": self.max_len, "compaction_capacity": self.max_len,
            "separate_cache_allocations_for_buckets": False, "last_bucket": self.last_bucket,
            "logical_reset_calls": self.logical_reset_calls}


# Root integration and standalone probes can use either descriptive spelling.
BucketedNumericalTreeVerifier = Bucketed3BTreeVerifier
