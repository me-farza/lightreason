"""Original-weight Qwen3B tree verification, separately relaxed FP evaluation.

This is a rooted TREE, not a merge DAG. Node0 is the unprocessed anchor and
every other node has exactly one earlier parent. A node attends the committed
prefix plus exactly its own self-inclusive ancestor chain. Distinct branches
use distinct physical K/V slots; rotary positions use semantic tree depth.
Thus each node evaluates the original model's path-conditioned equations,
with potentially different floating-point GEMM/attention evaluation order.
Exact serial agreement and answer quality require separate measurement.

Physical cache positions: base + arange(M).
Rotary positions:         base + depths[M].
Mask[row,k]:              k<base OR ancestor[row,k-base] in the M tree slots.

commit_path([0,...]) compacts a complete root-to-node path. No rerotation is
needed: path node j already has rotary position base+j. The two Triton data
kernels first gather ALL selected original BF16 K/V to independent temporary
storage, then write contiguous rows and copy the last node's target prediction
into a dedicated anchor buffer. No source/destination alias can corrupt a
later gather. The caller alone owns greedy acceptance, EOS and output budget.

The unchanged HF StaticCache, original target modules and original-HF prefill
are reused from relaxed3b_verifier. No strict runner/source file is modified.

Pinned mask/cache contracts:
https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/masking_utils.py
https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2/modeling_qwen2.py
https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/cache_utils.py
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
import triton
import triton.language as tl

from relaxed3b_verifier import Relaxed3BVerifier


def validate_tree_lists(depths, ancestors):
    """CPU-only exact topology check; rejects branch merges and missing closure."""
    count = len(depths)
    if count < 1 or len(ancestors) != count or any(len(row) != count for row in ancestors):
        raise ValueError("Use one nonempty square self-inclusive ancestor matrix")
    if any(type(depth) is not int or depth < 0 for depth in depths):
        raise ValueError("Tree depths must be nonnegative integers")
    if depths[0] != 0 or ancestors[0] != [True] + [False] * (count - 1):
        raise ValueError("Root0 must have depth0 and attend only itself within the tree")
    parents = [-1]
    for node in range(count):
        row = ancestors[node]
        if any(type(value) is not bool for value in row) or not row[node] or any(row[node + 1:]):
            raise ValueError("Use a topologically ordered bool mask including each node itself")
        if not node:
            continue
        previous = [index for index in range(node) if row[index]]
        if not previous or previous[0] != 0:
            raise ValueError("Every nonroot node must descend from root0")
        parent = previous[-1]
        expected = list(ancestors[parent])
        expected[node] = True
        if row != expected or depths[node] != depths[parent] + 1 or len(previous) != depths[node]:
            raise ValueError("Each node must have exactly its one parent's ancestors plus itself")
        parents.append(parent)
    return {"parents": parents, "depths": list(depths), "node_count": count,
            "rooted_tree": True, "self_inclusive_exact_ancestor_closure": True}


def validate_tree_cpu(depths, ancestors):
    if isinstance(depths, torch.Tensor):
        if depths.ndim != 1 or depths.dtype not in (torch.int32, torch.int64):
            raise ValueError("Use integer depths[M]")
        depths = depths.detach().cpu().tolist()
    else:
        depths = list(depths)
    if isinstance(ancestors, torch.Tensor):
        if ancestors.ndim != 2 or ancestors.dtype != torch.bool:
            raise ValueError("Use bool ancestors[M,M]")
        ancestors = ancestors.detach().cpu().tolist()
    else:
        ancestors = [list(row) for row in ancestors]
    return validate_tree_lists(depths, ancestors)


@dataclass
class _TreeEntry:
    input_ids: torch.Tensor
    base: torch.Tensor
    offsets: torch.Tensor
    physical_positions: torch.Tensor
    rotary_positions: torch.Tensor
    depths: torch.Tensor
    ancestors: torch.Tensor
    mask: torch.Tensor
    graph: object = None
    logits: torch.Tensor | None = None
    next_ids: torch.Tensor | None = None


@triton.jit
def _gather_path(Pointers, Path, Count, Base, Temporary,
                 LAYERS: tl.constexpr, CAPACITY: tl.constexpr,
                 BUDGET: tl.constexpr, NODES: tl.constexpr,
                 KV_HEADS: tl.constexpr, HEAD: tl.constexpr, BLOCK: tl.constexpr):
    layer, kind = tl.program_id(1), tl.program_id(2)
    flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    NK: tl.constexpr = KV_HEADS * HEAD
    count, base = tl.load(Count), tl.load(Base)
    slot, element = flat // NK, flat % NK
    node = tl.load(Path + slot, slot < count, other=0)
    valid = (slot < count) & (node >= 0) & (node < NODES)
    position = base + node
    valid = valid & (position >= 0) & (position < CAPACITY)
    address = tl.load(Pointers + kind * LAYERS + layer)
    pointer = address.to(tl.pointer_type(tl.bfloat16))
    source_offset = ((element // HEAD) * CAPACITY + position) * HEAD + element % HEAD
    value = tl.load(pointer + source_offset, valid, other=0)
    destination = ((kind * LAYERS + layer) * BUDGET * NK) + flat
    tl.store(Temporary + destination, value, slot < count)


@triton.jit
def _store_path(Pointers, Path, Count, Base, Temporary, NextIds, Anchor,
                LAYERS: tl.constexpr, CAPACITY: tl.constexpr,
                BUDGET: tl.constexpr, NODES: tl.constexpr,
                KV_HEADS: tl.constexpr, HEAD: tl.constexpr, BLOCK: tl.constexpr):
    layer, kind, block = tl.program_id(1), tl.program_id(2), tl.program_id(0)
    flat = block * BLOCK + tl.arange(0, BLOCK)
    NK: tl.constexpr = KV_HEADS * HEAD
    count, base = tl.load(Count), tl.load(Base)
    slot, element = flat // NK, flat % NK
    position = base + slot
    valid = (slot < count) & (position >= 0) & (position < CAPACITY)
    source = ((kind * LAYERS + layer) * BUDGET * NK) + flat
    value = tl.load(Temporary + source, slot < count, other=0)
    address = tl.load(Pointers + kind * LAYERS + layer)
    pointer = address.to(tl.pointer_type(tl.bfloat16))
    destination = ((element // HEAD) * CAPACITY + position) * HEAD + element % HEAD
    tl.store(pointer + destination, value, valid)
    if (block == 0) & (layer == 0) & (kind == 0):
        last_node = tl.load(Path + count - 1)
        token = tl.load(NextIds + last_node, (last_node >= 0) & (last_node < NODES), other=0)
        tl.store(Anchor, token)


class NumericalTreeVerifier(Relaxed3BVerifier):
    """Original-target fixed-node tree graphs; no proposer or output oracle.

    input_ids may be CPU or target-device LongTensor[1,M]. Topology tensors
    may likewise be CPU or target-device; all copies/synchronizations belong
    to online latency. CPU topology validation is enabled by default. Disable
    it only when a tested caller already guarantees the exact rooted-tree
    contract. Greedy token/path acceptance remains the caller's responsibility.
    """
    validate_tree_cpu = staticmethod(validate_tree_cpu)

    def __init__(self, model, max_len=1024, node_budget=32, *, cuda_graph=True,
                 compile_static=True, share_pool=False, warmups=3,
                 commit_engine="triton", validate_topology=True,
                 validate_acceptance=False):
        started = time.perf_counter()
        if type(node_budget) is not int or not 1 <= node_budget <= 128 or max_len <= node_budget:
            raise ValueError("Use node_budget1..128 and cache capacity larger than the node budget")
        if commit_engine not in ("triton", "torch"):
            raise ValueError("Use commit_engine='triton' or 'torch'")
        # Parent's linear block guard is intentionally not expanded: this is a
        # separately named graph family with its own M<=node_budget<=128 guard.
        super().__init__(model, max_len=max_len, cache_mode="static", cuda_graph=cuda_graph,
            compile_static=False, warmups=warmups, share_pool=share_pool, block_sizes=(1,))
        self.node_budget, self.block_sizes = node_budget, tuple(range(1, node_budget + 1))
        self.compile_static = bool(compile_static)
        self.commit_engine = commit_engine
        self.validate_topology, self.validate_acceptance = bool(validate_topology), bool(validate_acceptance)
        self._tree_forward_fn = self._forward_tree
        if compile_static:
            self._tree_forward_fn = torch.compile(self._forward_tree, fullgraph=True, mode="default")
        self.layers = len(self.cache.layers)
        self.kv_heads = model.config.num_key_value_heads
        self.head_dim = getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads
        expected = (1, self.kv_heads, max_len, self.head_dim)
        if any(tuple(value.shape) != expected or not value.is_contiguous()
               for layer in self.cache.layers for value in (layer.keys, layer.values)):
            raise ValueError("Compaction requires the original contiguous HF StaticCache layout")
        with torch.inference_mode(False), torch.no_grad():
            self.cache_pointer_table = torch.tensor(
                [[layer.keys.data_ptr() for layer in self.cache.layers],
                 [layer.values.data_ptr() for layer in self.cache.layers]],
                device=self.device, dtype=torch.uint64)
            self.key_pointer_table, self.value_pointer_table = self.cache_pointer_table[0], self.cache_pointer_table[1]
            self.path_indices = torch.empty(node_budget, device=self.device, dtype=torch.long)
            self.path_count = torch.empty((1,), device=self.device, dtype=torch.long)
            self.path_host = torch.empty(node_budget, device="cpu", dtype=torch.long, pin_memory=self.device.type == "cuda")
            self.count_host = torch.empty((1,), device="cpu", dtype=torch.long, pin_memory=self.device.type == "cuda")
            self.commit_temporary = torch.empty((2, self.layers, node_budget, self.kv_heads * self.head_dim),
                device=self.device, dtype=torch.bfloat16)
            self._commit_anchor_scratch = torch.empty((1, 1), device=self.device, dtype=torch.long)
            self._dummy_next_ids = torch.zeros(node_budget, device=self.device, dtype=torch.long)
        self._pending_entry, self._pending_topology = None, None
        self.last_depths, self.last_ancestors, self.last_input_ids = None, None, None
        self.commit_calls, self.compacted_input_tokens = 0, 0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.constructor_s = time.perf_counter() - started
        self.metadata = {**self.metadata, "variant": "original_weight_relaxed_fp_order_rooted_tree",
            "node_budget": node_budget, "compile_static": self.compile_static,
            "commit_engine": commit_engine, "validate_topology": self.validate_topology,
            "validate_greedy_acceptance": self.validate_acceptance,
            "physical_cache_position": "base + node index",
            "rotary_position": "base + tree depth",
            "causal_semantics": "committed prefix OR exact self-inclusive ancestor tree slots",
            "tree_attention": "committed prefix OR exact self-inclusive ancestor slots",
            "multi_parent_merge_dags_supported": False,
            "commit": "independent BF16 gather then contiguous writeback; no rerotation",
            "block_sizes": list(self.block_sizes)}

    def _new_tree_entry(self, size):
        with torch.inference_mode(False), torch.no_grad():
            offsets = torch.arange(size, device=self.device, dtype=torch.long)
            entry = _TreeEntry(
                input_ids=torch.zeros((1, size), device=self.device, dtype=torch.long),
                base=torch.tensor([self.current_position], device=self.device, dtype=torch.long),
                offsets=offsets, physical_positions=torch.empty_like(offsets), rotary_positions=torch.empty_like(offsets),
                depths=offsets.clone(), ancestors=torch.ones((size, size), device=self.device, dtype=torch.bool).tril_(),
                mask=torch.empty((1, 1, size, self.max_len), device=self.device, dtype=torch.bool))
        self.entries[size] = entry
        return entry

    def _forward_tree(self, input_ids, physical_positions, rotary_positions, mask):
        output = self.model(input_ids=input_ids, attention_mask=mask,
            position_ids=rotary_positions.view(1, -1), cache_position=physical_positions,
            past_key_values=self.cache, use_cache=True, logits_to_keep=input_ids.shape[1],
            return_dict=True, output_attentions=False, output_hidden_states=False)
        logits = output.logits.squeeze(0)
        return logits, logits.argmax(-1)

    def _tree_step(self, entry):
        torch.add(entry.base, entry.offsets, out=entry.physical_positions)
        torch.add(entry.base, entry.depths, out=entry.rotary_positions)
        # Expand an OPERAND: out=mask alone does not make torch.lt broadcast
        # the singleton query dimension and can otherwise resize mask to Q1.
        torch.lt(self.key_positions.expand_as(entry.mask), entry.base.view(1, 1, 1, 1), out=entry.mask)
        entry.mask.index_copy_(3, entry.physical_positions, entry.ancestors.view(1, 1, *entry.ancestors.shape))
        entry.logits, entry.next_ids = self._tree_forward_fn(
            entry.input_ids, entry.physical_positions, entry.rotary_positions, entry.mask)
        return entry.next_ids

    @torch.no_grad()
    def prepare(self, node_counts=None):
        sizes = ((self.node_budget,) if node_counts is None else (node_counts,) if type(node_counts) is int
                 else tuple(dict.fromkeys(node_counts)))
        if any(type(size) is not int or not 1 <= size <= self.node_budget for size in sizes):
            raise ValueError("Prepare node counts1..node_budget")
        if self.pending is not None:
            raise RuntimeError("Commit/discard the pending tree before preparation")
        missing = [size for size in sizes if size not in self.entries or
                   (self.cuda_graph and size not in self.graphs)]
        if any(self.current_position + size > self.max_len for size in missing):
            raise ValueError("Prepared tree exceeds physical cache capacity")
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        for size in missing:
            entry = self.entries.get(size) or self._new_tree_entry(size)
            entry.base.fill_(self.current_position)
            entry.input_ids.fill_(0)
            entry.depths.copy_(entry.offsets)
            entry.ancestors.fill_(True).tril_()
            if not self.cuda_graph and not self.compile_static:
                commit_warmup_s = self._warm_compaction(entry, size)
                self.capture_records.append({"nodes": size, "warmup_s": 0.0, "capture_s": 0.0,
                    "commit_warmup_s": commit_warmup_s, "representative_base": self.current_position,
                    "before_first_prefill": self.current_position == 0})
                continue
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            warm_started = time.perf_counter()
            with torch.cuda.stream(stream), torch.inference_mode(False), torch.no_grad():
                for _ in range(self.warmups):
                    self._tree_step(entry)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            warmup_s = time.perf_counter() - warm_started
            capture_s = 0.0
            if self.cuda_graph:
                graph = torch.cuda.CUDAGraph()
                capture_started = time.perf_counter()
                with torch.cuda.graph(graph, pool=self.pool, stream=stream), torch.inference_mode(False), torch.no_grad():
                    self._tree_step(entry)
                torch.cuda.synchronize(self.device)
                capture_s = time.perf_counter() - capture_started
                entry.graph, self.graphs[size] = graph, graph
            commit_warmup_s = self._warm_compaction(entry, size)
            self.capture_records.append({"nodes": size, "warmup_s": warmup_s, "capture_s": capture_s,
                "commit_warmup_s": commit_warmup_s,
                "representative_base": self.current_position, "before_first_prefill": self.current_position == 0})
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        self.setup_s += elapsed
        return {"new_graphs": len(missing) if self.cuda_graph else 0,
                "new_entries": len(missing), "setup_s": elapsed, "node_budget": self.node_budget}

    def _warm_compaction(self, entry, size):
        if self.commit_engine != "triton":
            return 0.0
        started = time.perf_counter()
        self.path_indices[:1].zero_()
        self.path_count.fill_(1)
        # Gather node0 and write it to its SAME tail slot; valid prefix stays
        # untouched. Use a scratch anchor so setup never changes real output.
        self._compact_triton(entry, size, 1, self._commit_anchor_scratch)
        torch.cuda.synchronize(self.device)
        return time.perf_counter() - started

    def _compact_triton(self, entry, size, count, anchor_output):
        block = 256
        grid = (triton.cdiv(count * self.kv_heads * self.head_dim, block), self.layers, 2)
        args = (self.cache_pointer_table, self.path_indices, self.path_count,
                entry.base, self.commit_temporary)
        constants = (self.layers, self.max_len, self.node_budget, size,
                     self.kv_heads, self.head_dim, block)
        next_ids = entry.next_ids if entry.next_ids is not None else self._dummy_next_ids
        _gather_path[grid](*args, *constants, num_warps=4)
        _store_path[grid](*args, next_ids, anchor_output, *constants, num_warps=4)

    @torch.no_grad()
    def prefill(self, ids):
        anchor = super().prefill(ids)
        self._pending_entry, self._pending_topology = None, None
        self.last_depths, self.last_ancestors, self.last_input_ids = None, None, None
        self.commit_calls, self.compacted_input_tokens = 0, 0
        return anchor

    @torch.no_grad()
    def verify_tree(self, input_ids, depths, ancestors, basep=None):
        if not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Use original target LongTensor[1,M] tree inputs")
        size = input_ids.shape[1]
        if self.current_position < 1 or self.anchor is None:
            raise RuntimeError("Prefill before tree verification")
        if self.pending is not None:
            raise RuntimeError("Commit/discard the preceding tree before verifying another")
        basep = self.current_position if basep is None else basep
        if type(basep) is not int or basep != self.current_position:
            raise ValueError("basep must equal the committed contiguous prefix length")
        if not 1 <= size <= self.node_budget or basep + size > self.max_len:
            raise ValueError("Tree exceeds node budget or physical cache capacity")
        if not isinstance(depths, torch.Tensor) or depths.shape != (size,) or depths.dtype not in (torch.int32, torch.int64):
            raise ValueError("Use integer depths[M]")
        if not isinstance(ancestors, torch.Tensor) or ancestors.shape != (size, size) or ancestors.dtype != torch.bool:
            raise ValueError("Use bool self-inclusive ancestors[M,M]")
        if any(value.device.type != "cpu" and value.device != self.device for value in (input_ids, depths, ancestors)):
            raise ValueError("Tree tensors must be CPU or on the original target device")
        topology = validate_tree_cpu(depths, ancestors) if self.validate_topology else None
        if self.validate_acceptance and int(input_ids[0, 0].item()) != int(self.anchor.item()):
            raise ValueError("Tree root input must equal the previous unprocessed target anchor")
        if size not in self.entries or (self.cuda_graph and size not in self.graphs):
            self.prepare(size)
        entry = self.entries[size]
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
            raise RuntimeError("Tree target returned an unexpected classifier shape")
        self.pending = {"basep": basep, "node_count": size}
        self._pending_entry, self._pending_topology = entry, topology
        self.last_logits, self.last_next_ids = entry.logits, entry.next_ids
        self.last_depths, self.last_ancestors, self.last_input_ids = entry.depths, entry.ancestors, entry.input_ids
        self.verify_calls += 1
        self.last_report = {"operation": "verify_tree", "basep": basep, "node_count": size,
            "topology_validated": topology is not None, "strict_serial_agreement_not_assumed": True}
        return self.last_next_ids

    def _validate_path(self, values):
        size = self.pending["node_count"]
        if not values:
            return
        if values[0] != 0 or any(type(node) is not int or not 0 <= node < size for node in values) or any(
                left >= right for left, right in zip(values, values[1:])):
            raise ValueError("Commit an increasing, unique path beginning at root0")
        topology = self._pending_topology
        if topology is not None and any(topology["parents"][node] != parent
                                        for parent, node in zip(values, values[1:])):
            raise ValueError("Commit path must contain every parent from root through its final node")
        if self.validate_acceptance:
            tokens, predictions = self.last_input_ids.detach().cpu().reshape(-1).tolist(), self.last_next_ids.detach().cpu().tolist()
            if any(tokens[node] != predictions[parent] for parent, node in zip(values, values[1:])):
                raise ValueError("A selected child input does not equal its parent's target prediction")

    @torch.no_grad()
    def commit_path(self, node_indices):
        if self.pending is None:
            raise RuntimeError("No pending target tree")
        if isinstance(node_indices, torch.Tensor):
            if node_indices.ndim != 1 or node_indices.dtype not in (torch.int32, torch.int64):
                raise ValueError("Use a one-dimensional integer root path")
            count = node_indices.numel()
            values = (node_indices.detach().cpu().tolist()
                      if self.validate_topology or self.validate_acceptance or node_indices.device.type == "cpu" else None)
        else:
            values = list(node_indices)
            count = len(values)
        if count > self.pending["node_count"]:
            raise ValueError("Commit path exceeds the verified tree")
        if values is not None:
            self._validate_path(values)
        base, size = self.pending["basep"], self.pending["node_count"]
        if count:
            if values is not None:
                self.path_host[:count].copy_(torch.tensor(values, device="cpu", dtype=torch.long))
                # Blocking tiny metadata copies let the permanent pinned host
                # buffers be safely reused without a later asynchronous race.
                self.path_indices[:count].copy_(self.path_host[:count], non_blocking=False)
            else:
                self.path_indices[:count].copy_(node_indices)
            self.count_host.fill_(count)
            self.path_count.copy_(self.count_host, non_blocking=False)
            if self.commit_engine == "triton":
                self._compact_triton(self._pending_entry, size, count, self._anchor_buffer)
            else:
                physical = self.path_indices[:count] + base
                for layer in self.cache.layers:
                    keys = layer.keys.index_select(2, physical)
                    values_tensor = layer.values.index_select(2, physical)
                    layer.keys[:, :, base:base + count, :].copy_(keys)
                    layer.values[:, :, base:base + count, :].copy_(values_tensor)
                self._anchor_buffer.copy_(self.last_next_ids.index_select(0, self.path_indices[count - 1:count]).view(1, 1))
            self.anchor = self._anchor_buffer
        self.current_position = base + count
        self.committed_input_tokens += count
        self.compacted_input_tokens += count
        self.commit_calls += 1
        self.pending, self._pending_entry, self._pending_topology = None, None, None
        self.last_report = {"operation": "commit_path", "path_length": count,
            "new_position": self.current_position, "commit_engine": self.commit_engine,
            "old_prefix_preserved": "writes start at previous base", "rerotated": False}
        return self.current_position

    def commit(self, acceptedinputcount):
        raise RuntimeError("Tree verification requires commit_path(node_indices), not a linear commit count")

    def verify(self, inputs, basep=None):
        raise RuntimeError("Use verify_tree(input_ids, depths, ancestors, basep) for this tree verifier")

    def discard(self):
        return self.commit_path([])

    def memory_report(self):
        return {**super().memory_report(), "node_budget": self.node_budget,
            "tree_graph_node_counts": sorted(self.graphs), "commit_engine": self.commit_engine,
            "compaction_temporary_bytes": self.commit_temporary.numel() * self.commit_temporary.element_size(),
            "pointer_table_bytes": self.cache_pointer_table.numel() * self.cache_pointer_table.element_size(),
            "commit_calls": self.commit_calls, "compacted_input_tokens": self.compacted_input_tokens}
