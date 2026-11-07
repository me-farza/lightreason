"""Private GPU tree selection/acceptance prototype for the original BF16 target.

Frozen CPU and append-only runners are untouched. One-CTA FP64 frontier
selection exactly follows (-score,depth,parent,token) priorities on unique
position-marginal top-K rows. Only original target argmax IDs authorize an
accepted path. Original target K/V is compacted with the frozen two-pass
gather/store kernels; a new kernel copies target-produced feature rows and
visible IDs. One pinned CPU receipt follows all queued cycle work.

Only private kernel-owned trees may disable the generic CPU topology checks.
Public mutable tree inputs are never admitted. Invalid device inputs are
clamped before target embedding access, set an error flag, commit zero K/V
rows, emit nothing, and require logical target/auxiliary reset. The frozen
store kernel's count-zero anchor read uses a permanent in-bounds sentinel;
the invalid request's anchor is discarded by logical reset.

CPU self-tests and a model-free synthetic GPU probe are provided. Successful
microprobes do not establish full model/cache/quality parity or a speedup.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
import time
from types import SimpleNamespace

try:
    import torch
    import triton
    import triton.language as tl
except ModuleNotFoundError as error:
    if error.name not in ("torch", "triton"):
        raise
    torch = triton = tl = None

from append_only_pard2_runtime3b_v1 import AppendOnlyPARD2ProposalAdapter
from append_only_tree3b_v1 import AppendOnlyTree3BRunner, feature_coordination_counts
from cost_aware_tree3b_v1 import PACKED_BUDGETS, capacity_budget

ERROR_BAD_ID = 1
ERROR_BAD_SCORE = 2
ERROR_DUPLICATE_ID = 4
ERROR_EMPTY_FRONTIER = 8
ERROR_BAD_PREDICTION = 16
ERROR_BAD_REMAINING = 32
ERROR_AMBIGUOUS_CHILD = 64
MAX_EOS_IDS = 8
RECEIPT_HEADER = 5


def checked_config(M, B, K, vocab_size):
    if type(M) is not int or M not in PACKED_BUDGETS:
        raise ValueError("Use physical packed M16/32/48/64/96/128")
    if type(B) is not int or not 1 <= B <= 32:
        raise ValueError("Use a checkpoint-admitted draft depth1..32")
    if type(K) is not int or not 1 <= K <= 16:
        raise ValueError("This private frontier prototype supports top-K1..16")
    if type(vocab_size) is not int or not 1 <= vocab_size < 2**31:
        raise ValueError("Use the actual positive original target vocabulary")
    total, level = 1, 1
    for _ in range(B):
        level *= K
        total += level
        if total >= M:
            break
    if total < M:
        raise ValueError("The supported draft lattice cannot fill this physical tree")
    return M, B, K, vocab_size


@lru_cache(maxsize=1)
def _frozen_tree_core():
    """Execute the exact existing dependency-free core without importing Torch."""
    path = Path(__file__).with_name("pard_tree.py")
    syntax = ast.parse(path.read_text())
    function = next(node for node in syntax.body if isinstance(node, ast.FunctionDef) and node.name == "_build_tree_lists")
    namespace = {"heapq": __import__("heapq"), "math": math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_build_tree_lists"]


def cpu_reference_tree(anchor, ids, logps, M, vocab_size=151936):
    """Frozen CPU reference with the private GPU controller's input admission."""
    B, K = len(ids), len(ids[0]) if ids else 0
    checked_config(M, B, K, vocab_size)
    if type(anchor) is not int or not 0 <= anchor < vocab_size:
        raise ValueError("Invalid original target anchor")
    if len(logps) != B or any(len(row) != K for row in ids + logps):
        raise ValueError("Use matching B×K marginal matrices")
    for row in ids:
        if any(type(value) is not int or not 0 <= value < vocab_size for value in row) or len(set(row)) != K:
            raise ValueError("Private top-K rows must contain unique in-vocabulary IDs")
    result = _frozen_tree_core()(anchor, ids, logps, M)
    if len(result["tokens"]) != M:
        raise ValueError("Finite frontier cannot fill the requested physical graph")
    return result


def cpu_reference_accept(tree, predicted, remaining, eos_ids=(), max_commit=33):
    if type(remaining) is not int or remaining < 1 or type(max_commit) is not int or max_commit < 1:
        raise ValueError("Use positive remaining and supported path capacities")
    children = [{} for _ in tree["tokens"]]
    for node in range(1, len(children)):
        parent, token = tree["parents"][node], tree["tokens"][node]
        if token in children[parent]:
            raise ValueError("Ambiguous duplicate child")
        children[parent][token] = node
    path, emitted, node, eos = [], [], 0, set(eos_ids)
    while True:
        path.append(node)
        value = predicted[node]
        emitted.append(value)
        if len(emitted) >= min(remaining, max_commit) or value in eos:
            break
        child = children[node].get(value)
        if child is None:
            break
        node = child
    return path, emitted


if triton is not None:
    @triton.jit
    def _tree_frontier(IDs, LogPs, Anchor, Tokens, Parents, Depths, Scores, Bits, Error,
                       M: tl.constexpr, B: tl.constexpr, K: tl.constexpr, VOCAB: tl.constexpr,
                       FRONT: tl.constexpr, NODES: tl.constexpr, LATTICE: tl.constexpr):
        lattice = tl.arange(0, LATTICE)
        valid_lattice = lattice < B * K
        ids = tl.load(IDs + lattice, valid_lattice, other=0)
        scores = tl.load(LogPs + lattice, valid_lattice, other=-float("inf"))
        bad_ids = valid_lattice & ((ids < 0) | (ids >= VOCAB))
        bad_scores = valid_lattice & ((scores != scores) | (scores == float("inf")) | (scores > 1.0e-5))
        duplicate = tl.full((LATTICE,), False, tl.int1)
        for delta in tl.static_range(1, K):
            other_rank = ((lattice % K) + delta) % K
            other = tl.load(IDs + (lattice // K) * K + other_rank, valid_lattice, other=-1)
            duplicate = duplicate | (valid_lattice & (ids == other))
        anchor_raw = tl.load(Anchor)
        error = tl.where((tl.sum(bad_ids.to(tl.int32), 0) > 0) | (anchor_raw < 0) | (anchor_raw >= VOCAB), 1, 0)
        error = error | tl.where(tl.sum(bad_scores.to(tl.int32), 0) > 0, 2, 0)
        error = error | tl.where(tl.sum(duplicate.to(tl.int32), 0) > 0, 4, 0)

        frontier = tl.arange(0, FRONT)
        rank = frontier % K
        frontier_parent = frontier // K
        first_score = tl.load(LogPs + rank, frontier < K, other=-float("inf"))
        first_id = tl.load(IDs + rank, frontier < K, other=0)
        front_scores = tl.where((frontier < K) & (error == 0), tl.minimum(first_score, 0.0).to(tl.float64), -float("inf"))
        front_tokens = tl.minimum(tl.maximum(first_id, 0), VOCAB - 1).to(tl.int64)
        front_depths = tl.full((FRONT,), 1, tl.int32)
        nodes = tl.arange(0, NODES)
        token_bank = tl.where(nodes == 0, tl.minimum(tl.maximum(anchor_raw, 0), VOCAB - 1), 0).to(tl.int64)
        parent_bank = tl.where(nodes == 0, -1, 0).to(tl.int64)
        depth_bank = tl.zeros((NODES,), tl.int64)
        score_bank = tl.where(nodes == 0, 0.0, -float("inf")).to(tl.float64)
        low = tl.where(nodes == 0, 1, 0).to(tl.uint64)
        high = tl.zeros((NODES,), tl.uint64)
        for node in range(1, M):
            best = tl.max(front_scores, 0)
            finite = best > -float("inf")
            chosen = (front_scores == best) & finite
            depth = tl.min(tl.where(chosen, front_depths, 2147483647), 0)
            chosen = chosen & (front_depths == depth)
            parent = tl.min(tl.where(chosen, frontier_parent, 2147483647), 0)
            chosen = chosen & (frontier_parent == parent)
            token = tl.min(tl.where(chosen, front_tokens, 2147483647), 0)
            chosen = chosen & (front_tokens == token)
            error = error | tl.where(finite, 0, 8)
            safe_parent = tl.where(finite, parent, 0)
            safe_depth = tl.where(finite, depth, 1)
            safe_token = tl.where(finite, token, 0)
            parent_low = tl.sum(tl.where(nodes == safe_parent, low, 0).to(tl.uint64), 0)
            parent_high = tl.sum(tl.where(nodes == safe_parent, high, 0).to(tl.uint64), 0)
            own_bit = tl.full((), 1, tl.uint64) << (node & 63)
            node_low = parent_low | tl.where(node < 64, own_bit, 0).to(tl.uint64)
            node_high = parent_high | tl.where(node >= 64, own_bit, 0).to(tl.uint64)
            low = tl.where(nodes == node, node_low, low)
            high = tl.where(nodes == node, node_high, high)
            token_bank = tl.where(nodes == node, safe_token, token_bank)
            parent_bank = tl.where(nodes == node, safe_parent, parent_bank)
            depth_bank = tl.where(nodes == node, safe_depth, depth_bank)
            score_bank = tl.where(nodes == node, best, score_bank)
            front_scores = tl.where(chosen, -float("inf"), front_scores)
            new_slots = (frontier_parent == node) & (frontier < M * K)
            can_expand = new_slots & finite & (safe_depth < B)
            next_score = tl.load(LogPs + safe_depth * K + rank, can_expand, other=-float("inf"))
            next_id = tl.load(IDs + safe_depth * K + rank, can_expand, other=0)
            expanded = best + tl.minimum(next_score, 0.0).to(tl.float64)
            front_scores = tl.where(new_slots, tl.where(can_expand, expanded, -float("inf")), front_scores)
            front_tokens = tl.where(new_slots, tl.minimum(tl.maximum(next_id, 0), VOCAB - 1).to(tl.int64), front_tokens)
            front_depths = tl.where(new_slots, safe_depth + 1, front_depths)
        tl.store(Tokens + nodes, token_bank, nodes < M)
        tl.store(Parents + nodes, parent_bank, nodes < M)
        tl.store(Depths + nodes, depth_bank, nodes < M)
        tl.store(Scores + nodes, score_bank, nodes < M)
        tl.store(Bits + nodes * 2, low, nodes < M)
        tl.store(Bits + nodes * 2 + 1, high, nodes < M)
        tl.store(Error, error)

    @triton.jit
    def _ancestor_matrix(Bits, Ancestors, M: tl.constexpr, BLOCK: tl.constexpr):
        flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row, column = flat // M, flat % M
        bits = tl.load(Bits + row * 2 + column // 64, flat < M * M, other=0)
        value = ((bits >> (column & 63)) & 1) != 0
        tl.store(Ancestors + flat, value, flat < M * M)

    @triton.jit
    def _accept_target_path(Parents, Tokens, NextIds, Error, Remaining, EOS, EOSCount,
                            Path, SafePath, Count, Emitted,
                            M: tl.constexpr, VOCAB: tl.constexpr,
                            MAX_COMMIT: tl.constexpr, NODES: tl.constexpr):
        nodes = tl.arange(0, NODES)
        parents = tl.load(Parents + nodes, nodes < M, other=-2)
        tokens = tl.load(Tokens + nodes, nodes < M, other=-1)
        eos_slots = tl.arange(0, 8)
        eos_count = tl.load(EOSCount)
        eos_values = tl.load(EOS + eos_slots)
        error = tl.load(Error)
        remaining = tl.load(Remaining)
        error = error | tl.where(remaining >= 1, 0, 32)
        active = error == 0
        node = tl.full((), 0, tl.int32)
        count = tl.full((), 0, tl.int32)
        for step in range(MAX_COMMIT):
            value = tl.load(NextIds + node, active & (node >= 0) & (node < M), other=0)
            valid_prediction = (value >= 0) & (value < VOCAB)
            error = error | tl.where(active & ~valid_prediction, 16, 0)
            write = active & valid_prediction
            tl.store(Path + step, node.to(tl.int64), write)
            tl.store(SafePath + step, node.to(tl.int64), write)
            tl.store(Emitted + step, value, write)
            count = tl.where(write, step + 1, count)
            is_eos = tl.sum(((eos_slots < eos_count) & (eos_values == value)).to(tl.int32), 0) > 0
            stop = (step + 1 >= remaining) | is_eos
            matches = (nodes < M) & (parents == node) & (tokens == value)
            match_count = tl.sum(matches.to(tl.int32), 0)
            error = error | tl.where(write & ~stop & (match_count > 1), 64, 0)
            child = tl.min(tl.where(matches, nodes, 2147483647), 0)
            active = write & ~stop & (match_count == 1) & (error == 0)
            node = tl.where(active, child, 0)
        tl.store(Count, tl.where(error == 0, count, 0))
        tl.store(Error, error)

    @triton.jit
    def _append_target_features_visible(NodeFeatures, History, Snapshot, Visible, Path, Count, Base, Emitted,
                                        M: tl.constexpr, CAPACITY: tl.constexpr,
                                        MAX_COMMIT: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        slot, column = flat // WIDTH, flat % WIDTH
        count, base = tl.load(Count), tl.load(Base)
        node = tl.load(Path + slot, slot < count, other=0)
        valid = (slot < count) & (node >= 0) & (node < M) & (base + slot < CAPACITY)
        features = tl.load(NodeFeatures.to(tl.pointer_type(tl.uint16)) + node * WIDTH + column, valid, other=0)
        tl.store(History.to(tl.pointer_type(tl.uint16)) + (base + slot) * WIDTH + column, features, valid)
        tl.store(Snapshot.to(tl.pointer_type(tl.uint16)) + (base + slot) * WIDTH + column, features, valid)
        write_id = valid & (column == 0) & (base + slot + 1 < CAPACITY)
        token = tl.load(Emitted + slot, write_id, other=0)
        tl.store(Visible + base + slot + 1, token, write_id)

    @triton.jit
    def _pack_receipt(Error, Count, Anchor, Base, Tokens, Emitted, Path, Receipt,
                       MAX_COMMIT: tl.constexpr, PACKET: tl.constexpr):
        index = tl.arange(0, PACKET)
        error, count = tl.load(Error), tl.load(Count)
        anchor, base, root = tl.load(Anchor), tl.load(Base), tl.load(Tokens)
        emit_slot = index - 5
        path_slot = index - 5 - MAX_COMMIT
        emit = tl.load(Emitted + emit_slot, (emit_slot >= 0) & (emit_slot < count), other=0)
        path = tl.load(Path + path_slot, (path_slot >= 0) & (path_slot < count), other=0)
        value = tl.where(index == 0, error,
                tl.where(index == 1, count,
                tl.where(index == 2, anchor,
                tl.where(index == 3, base + count,
                tl.where(index == 4, root,
                tl.where((emit_slot >= 0) & (emit_slot < MAX_COMMIT), emit, path)))))).to(tl.int64)
        tl.store(Receipt + index, value, index < 5 + 2 * MAX_COMMIT)


class GPUTreeController:
    """Preallocated fixed-physical-shape kernels, never a token authority."""
    def __init__(self, M, B, K, vocab_size, device):
        if torch is None or triton is None:
            raise RuntimeError("Use the pinned Triton/CUDA environment")
        checked_config(M, B, K, vocab_size)
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("The private GPU controller requires the original CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.M, self.B, self.K, self.vocab_size, self.device = M, B, K, vocab_size, device
        self.max_commit = B + 1
        with torch.inference_mode(False), torch.no_grad():
            self.parents = torch.empty((M,), device=device, dtype=torch.long)
            self.scores = torch.empty((M,), device=device, dtype=torch.float64)
            self.bits = torch.empty((M, 2), device=device, dtype=torch.uint64)
            self.error = torch.zeros((1,), device=device, dtype=torch.int32)
            self.remaining = torch.ones((1,), device=device, dtype=torch.int32)
            self.eos = torch.full((MAX_EOS_IDS,), -1, device=device, dtype=torch.long)
            self.eos_count = torch.zeros((1,), device=device, dtype=torch.int32)
            self.emitted = torch.empty((self.max_commit,), device=device, dtype=torch.long)
            self.safe_path_owner = torch.zeros((M + 1,), device=device, dtype=torch.long)
            self.safe_path = self.safe_path_owner[1:]
            packet_length = RECEIPT_HEADER + 2 * self.max_commit
            self.receipt_device = torch.empty((packet_length,), device=device, dtype=torch.long)
            self.receipt_host = torch.empty((packet_length,), device="cpu", dtype=torch.long, pin_memory=True)
            self.receipt_event = torch.cuda.Event()
        self.prepared = False
        self.last_receipt = None

    def configure_eos(self, eos_ids):
        values = sorted(set(eos_ids or ()))
        if len(values) > MAX_EOS_IDS or any(type(value) is not int or not 0 <= value < self.vocab_size for value in values):
            raise ValueError("Use at most eight original target native EOS IDs")
        padded = values + [-1] * (MAX_EOS_IDS - len(values))
        self.eos.copy_(torch.tensor(padded, device="cpu", dtype=torch.long))
        self.eos_count.fill_(len(values))

    def launch_tree(self, top_ids, logps, anchor, entry):
        if tuple(top_ids.shape) != (1, self.B, self.K) or top_ids.dtype != torch.long or logps.dtype != torch.float32 or tuple(logps.shape) != tuple(top_ids.shape):
            raise ValueError("Use the prepared original auxiliary lattice[1,B,K] long IDs/FP32 log masses")
        if not top_ids.is_contiguous() or not logps.is_contiguous() or top_ids.device != self.device or logps.device != self.device:
            raise ValueError("Use contiguous lattice buffers on the original target device")
        if tuple(entry.input_ids.shape) != (1, self.M) or tuple(entry.depths.shape) != (self.M,) or tuple(entry.ancestors.shape) != (self.M, self.M):
            raise ValueError("Bind the actual separately prepared M-row target entry")
        _tree_frontier[(1,)](top_ids, logps, anchor, entry.input_ids, self.parents, entry.depths,
            self.scores, self.bits, self.error, self.M, self.B, self.K, self.vocab_size,
            triton.next_power_of_2(self.M * self.K), triton.next_power_of_2(self.M),
            triton.next_power_of_2(self.B * self.K), num_warps=4, enable_fp_fusion=False)
        _ancestor_matrix[(triton.cdiv(self.M * self.M, 256),)](self.bits, entry.ancestors, self.M, 256, num_warps=4)

    def launch_accept(self, next_ids, owner, remaining):
        if type(remaining) is not int or remaining < 1:
            raise ValueError("A live target cycle must have positive remaining output capacity")
        self.remaining.fill_(remaining)
        _accept_target_path[(1,)](self.parents, owner._pending_entry.input_ids, next_ids, self.error,
            self.remaining, self.eos, self.eos_count, owner.path_indices, self.safe_path,
            owner.path_count, self.emitted, self.M, self.vocab_size, self.max_commit,
            triton.next_power_of_2(self.M), num_warps=4)

    def queue_kv_commit(self, owner, entry):
        from numerical3b_tree_verifier import _gather_path, _store_path
        block = 256
        grid = (triton.cdiv(self.max_commit * owner.kv_heads * owner.head_dim, block), owner.layers, 2)
        constants = (owner.layers, owner.max_len, owner.node_budget, self.M, owner.kv_heads, owner.head_dim, block)
        _gather_path[grid](owner.cache_pointer_table, owner.path_indices, owner.path_count,
            entry.base, owner.commit_temporary, *constants, num_warps=4)
        # Count0 reads safe_path[-1], which is the permanent in-bounds sentinel.
        # It writes no K/V rows. Its anchor is invalidated on the error receipt.
        _store_path[grid](owner.cache_pointer_table, self.safe_path, owner.path_count,
            entry.base, owner.commit_temporary, entry.next_ids, owner._anchor_buffer, *constants, num_warps=4)

    def queue_feature_visible_append(self, owner, entry, snapshot, visible):
        _append_target_features_visible[(triton.cdiv(self.max_commit * 8192, 256),)](
            entry.features, owner.committed_feature_history, snapshot, visible, owner.path_indices,
            owner.path_count, entry.base, self.emitted, self.M, owner.max_len, self.max_commit, 8192, 256,
            num_warps=4)

    def receive(self, owner, entry):
        _pack_receipt[(1,)](self.error, owner.path_count, owner._anchor_buffer, entry.base,
            entry.input_ids, self.emitted, owner.path_indices, self.receipt_device, self.max_commit,
            triton.next_power_of_2(RECEIPT_HEADER + 2 * self.max_commit), num_warps=4)
        self.receipt_host.copy_(self.receipt_device, non_blocking=True)
        self.receipt_event.record()
        self.receipt_event.synchronize()
        values = self.receipt_host.tolist()
        error, count, anchor, position, root = values[:RECEIPT_HEADER]
        if not 0 <= count <= self.max_commit:
            raise RuntimeError("Device receipt returned an invalid original-target path count")
        result = {"error": error, "count": count, "anchor": anchor, "position": position, "root": root,
            "emitted": values[RECEIPT_HEADER:RECEIPT_HEADER + count],
            "path": values[RECEIPT_HEADER + self.max_commit:RECEIPT_HEADER + self.max_commit + count]}
        self.last_receipt = result
        return result

    def prepare(self, owner, entry, snapshot, visible):
        """Synthetic-only kernel compilation before any true request prefix."""
        if self.prepared:
            return {"new_controller_kernels": False, "setup_s": 0.0}
        began = time.perf_counter()
        with torch.inference_mode(False), torch.no_grad():
            ids = torch.arange(self.B * self.K, device=self.device, dtype=torch.long).view(1, self.B, self.K)
            ids.remainder_(self.vocab_size)
            ranks = torch.arange(self.K, device=self.device, dtype=torch.float32)
            logps = (-0.25 - 0.125 * ranks).view(1, 1, self.K).expand(1, self.B, self.K).contiguous()
            anchor = torch.zeros((1,), device=self.device, dtype=torch.long)
            synthetic_next = torch.zeros((self.M,), device=self.device, dtype=torch.long)
            synthetic_features = torch.zeros((self.M, 8192), device=self.device, dtype=torch.bfloat16)
            self.configure_eos(())
            self.launch_tree(ids, logps, anchor, entry)
            # Scratch K/V is independent and entirely synthetic. No previous
            # request cache contents, genuine prefix, target answer, or feature
            # values enter these compilation calls. Full physical strides match
            # the original frozen compaction kernels' CAPACITY specialization.
            keys = [torch.zeros((1, owner.kv_heads, owner.max_len, owner.head_dim), device=self.device,
                                dtype=torch.bfloat16) for _ in range(owner.layers)]
            values = [torch.zeros_like(value) for value in keys]
            pointers = torch.tensor([value.data_ptr() for value in keys + values],
                                    device=self.device, dtype=torch.int64)
            scratch = SimpleNamespace(input_ids=entry.input_ids, depths=entry.depths, ancestors=entry.ancestors,
                base=torch.ones((1,), device=self.device, dtype=torch.long),
                next_ids=synthetic_next, features=synthetic_features)
            history_rows = self.max_commit + 1
            fake = SimpleNamespace(layers=owner.layers, max_len=owner.max_len, node_budget=owner.node_budget,
                kv_heads=owner.kv_heads, head_dim=owner.head_dim, cache_pointer_table=pointers,
                path_indices=torch.empty_like(owner.path_indices), path_count=torch.zeros_like(owner.path_count),
                commit_temporary=torch.empty_like(owner.commit_temporary),
                _anchor_buffer=torch.zeros_like(owner._anchor_buffer), _pending_entry=scratch,
                committed_feature_history=torch.zeros((1, history_rows, 8192), device=self.device, dtype=torch.bfloat16))
            synthetic_snapshot = torch.zeros_like(fake.committed_feature_history)
            synthetic_visible = torch.zeros((1, self.max_commit + 2), device=self.device, dtype=torch.long)
            self.launch_accept(synthetic_next, fake, self.max_commit)
            self.queue_kv_commit(fake, scratch)
            self.queue_feature_visible_append(fake, scratch, synthetic_snapshot, synthetic_visible)
            receipt = self.receive(fake, scratch)
            if receipt["error"] or receipt["count"] < 1:
                raise RuntimeError("Synthetic GPU controller preparation failed: " + repr(receipt))
        self.prepared = True
        return {"new_controller_kernels": True, "setup_s": time.perf_counter() - began,
                "synthetic_only": True, "target_forward_called": False}


class _GPUFeatureTailAdapter(AppendOnlyPARD2ProposalAdapter):
    """Only this private controller may acknowledge its fused target-row copy."""
    def __init__(self, runner):
        super().__init__(runner)
        self._fused_tail = None

    def acknowledge_queued_target_tail(self, start, end):
        if self._fused_tail is not None or not self.ledger.active or self.ledger.phase != "awaiting_original_target_commit":
            raise RuntimeError("No unique original-target GPU commit may authorize this feature tail")
        if start != self.ledger.position or not start < end <= self.runner.max_len:
            raise RuntimeError("GPU feature tail range differs from the owned original-target prefix")
        self._fused_tail = start, end

    def _feature_tail(self, start, end):
        if self._fused_tail is None:
            return super()._feature_tail(start, end)
        if self._fused_tail != (start, end):
            raise RuntimeError("Fused GPU feature copy does not cover the exact original-target append")
        self.copied_feature_rows += end - start
        self.owner._verified_feature_count = self.snapshot_count = end
        self._fused_tail = None

    def abort_request(self):
        self._fused_tail = None
        super().abort_request()


def acknowledge_original_target_commit(owner, entry, receipt, expected_root, M, max_commit, vocab_size):
    """CPU metadata only, after the single receipt; never authorizes proposals."""
    if receipt["error"]:
        raise RuntimeError("Private GPU tree controller rejected its input, device flags=" + str(receipt["error"]))
    count, path, emitted = receipt["count"], receipt["path"], receipt["emitted"]
    pending = owner.pending
    if pending is None or owner._pending_entry is not entry or pending["node_count"] != M:
        raise RuntimeError("No matching original-target packed verification is awaiting GPU commitment")
    base = pending["basep"]
    if (not 1 <= count <= min(max_commit, M) or len(path) != count or len(emitted) != count or
            path[0] != 0 or any(type(value) is not int or not 0 <= value < M for value in path) or
            any(left >= right for left, right in zip(path, path[1:]))):
        raise RuntimeError("Original target GPU receipt did not contain an increasing rooted path")
    if any(type(value) is not int or not 0 <= value < vocab_size for value in emitted):
        raise RuntimeError("Original target receipt emitted an out-of-vocabulary ID")
    if receipt["root"] != expected_root or receipt["anchor"] != emitted[-1] or receipt["position"] != base + count:
        raise RuntimeError("GPU root/anchor/contiguous-cache receipt invariant failed")
    if owner.current_position != base or base + count >= owner.max_len:
        raise RuntimeError("GPU receipt cannot extend the original target prefix at this position")
    owner.current_position = base + count
    owner.committed_input_tokens += count
    owner.compacted_input_tokens += count
    owner.commit_calls += 1
    owner.anchor = owner._anchor_buffer
    owner.pending, owner._pending_entry, owner._pending_topology = None, None, None
    owner.last_report = {"operation": "private_GPU_original_target_commit_path", "path_length": count,
        "new_position": owner.current_position, "commit_engine": "frozen_two_pass_triton",
        "old_prefix_preserved": "writes start at previous base", "rerotated": False,
        "original_target_only_KV_and_features": True, "single_CPU_receipt_after_all_cycle_GPU_work": True}
    return path, emitted, base, owner.current_position


class GPUTreeControl3BRunner(AppendOnlyTree3BRunner):
    """Actual packed target families with a private fixed-budget GPU controller."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configure_gpu_controller()

    @classmethod
    def from_control(cls, control, **kwargs):
        result = super().from_control(control, **kwargs)
        result._configure_gpu_controller()
        return result

    def _configure_gpu_controller(self):
        if self.ranking_policy != "original" or self.ranking_temperature != 1.0 or self.tree_policy != "best-first":
            raise ValueError("This first GPU controller preserves original best-first marginal ranking only")
        if self.cycle_costs_ms is not None:
            raise ValueError("This first GPU controller uses a development-selected fixed physical shape")
        for M in self.prepared_packed_budgets:
            checked_config(M, self.draft_tokens, self.top_k, self.model.config.vocab_size)
        owner = self.verifier._owner
        self._previous_generic_tree_validation = {"topology": owner.validate_topology, "acceptance": owner.validate_acceptance}
        # These flags affect CPU admission only, never model math. They apply
        # solely to THIS new cache owner and its private proven kernel outputs.
        owner.validate_topology = owner.validate_acceptance = False
        owner.metadata.update(private_GPU_tree_only=True,
            generic_CPU_tree_validation_disabled_for_private_GPU_outputs_only=True)
        self.proposal_adapter = _GPUFeatureTailAdapter(self)
        self._gpu_controllers = {}
        self.metadata = {**self.metadata,
            "variant": "original_BF16_target_private_GPU_tree_and_acceptance_control_v1",
            "private_GPU_controller": True,
            "controller_ranking": "FP64_prefix_sums_then_depth_parent_token_ties",
            "generic_mutable_tree_inputs_admitted": False,
            "CPU_topology_validation_scope": "disabled only for this privately owned GPU-built tree",
            "frozen_original_control_validation_flags_changed": False,
            "original_target_full_vocabulary_acceptance": True,
            "KV_commit_kernels": "unchanged frozen gather_path/store_path",
            "feature_visible_commit_source": "original verified target node features and argmax IDs only",
            "CPU_device_receipts_per_cycle": 1,
            "streaming_and_begin_finish_audits_timed": True,
            "GPU_input_errors_abort_without_output_and_reset_logical_caches": True,
            "full_ranking_trace_collected_during_GPU_decode": False,
            "performance_or_quality_improvement_established": False}

    def prepare(self, ids, n):
        prepared = super().prepare(ids, n)
        began = time.perf_counter()
        owner = self.verifier._owner
        receipts = {}
        for M in sorted({size for size, bucket in owner.entries}):
            if M not in self.prepared_packed_budgets:
                continue
            if M not in self._gpu_controllers:
                self._gpu_controllers[M] = GPUTreeController(M, self.draft_tokens, self.top_k,
                    self.model.config.vocab_size, self.model.device)
            entry = next(entry for (size, bucket), entry in owner.entries.items() if size == M)
            receipts[M] = self._gpu_controllers[M].prepare(owner, entry,
                self.proposer.conditioned._verified_feature_snapshot, self.visible)
        owner.reset_for_new_request()
        self.proposer.reset()
        torch.cuda.synchronize()
        extra = time.perf_counter() - began
        self.setup_s += extra
        return {**prepared, "setup_s": prepared["setup_s"] + extra,
                "private_GPU_controller_setup_s": extra, "private_GPU_controller_families": receipts}

    def execution_signature(self):
        result = super().execution_signature()
        result["private_GPU_controller_buffers"] = tuple((M,
            controller.parents.data_ptr(), controller.bits.data_ptr(), controller.error.data_ptr(),
            controller.receipt_device.data_ptr(), controller.receipt_host.data_ptr(), controller.prepared)
            for M, controller in sorted(self._gpu_controllers.items()))
        return result

    def run(self, ids, n, eos_ids=None, stream_callback=None):
        self._check(ids, n)
        prepared = self.prepare(ids, n)
        eos = set(eos_ids or [])
        owner, adapter = self.verifier._owner, self.proposal_adapter
        with torch.inference_mode(False), torch.no_grad():
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                self.proposer.reset()
                self.visible[:, :ids.shape[1]].copy_(ids)
                with self._stage("prefill"):
                    for controller in self._gpu_controllers.values():
                        controller.configure_eos(eos)
                    anchor = self.verifier.prefill(ids)
                    self.visible[:, ids.shape[1]:ids.shape[1] + 1].copy_(anchor)
                    tokens = [int(anchor.item())]
                    adapter.begin_request_from_owned_buffers(tokens[0], eos)
                if stream_callback is not None:
                    with self._stage("streaming"):
                        stream_callback(tokens.copy())
                torch.cuda.synchronize()
                prefill_end = time.perf_counter()
                cycles = []
                while len(tokens) < n and (not eos or tokens[-1] not in eos):
                    cycle_start = time.perf_counter()
                    visible_length = ids.shape[1] + len(tokens)
                    M, fallback = capacity_budget(self.fixed_packed_budget,
                        self.max_len - owner.current_position, self.capacity_fallback_budget)
                    bucket = owner.select_bucket(owner.current_position + M)
                    entry = owner.entries[(M, bucket)]
                    controller = self._gpu_controllers[M]
                    if not controller.prepared:
                        raise RuntimeError("Prepare all private GPU kernels with synthetic inputs before generation")
                    with self._stage("drafting"):
                        top_ids, logps = adapter.propose_lattice(self.visible[:, :visible_length], self.draft_tokens)
                    with self._stage("selection_tree"):
                        controller.launch_tree(top_ids, logps, owner.anchor, entry)
                    with self._stage("verification"):
                        predicted = self.verifier.verify_tree(entry.input_ids, entry.depths, entry.ancestors, owner.current_position)
                    if owner._pending_entry is not entry or owner.last_node_features is not entry.features:
                        raise RuntimeError("Only freshly verified original target node features may enter the GPU commit")
                    with self._stage("acceptance"):
                        controller.launch_accept(predicted, owner, n - len(tokens))
                    with self._stage("cache_commit"):
                        controller.queue_kv_commit(owner, entry)
                    with self._stage("visible_update"):
                        controller.queue_feature_visible_append(owner, entry,
                            self.proposer.conditioned._verified_feature_snapshot, self.visible)
                    with self._stage("receipt_sync"):
                        receipt = controller.receive(owner, entry)
                    with self._stage("anchor_sync"):
                        path, emitted, oldp, newp = acknowledge_original_target_commit(owner, entry, receipt,
                            tokens[-1], M, controller.max_commit, self.model.config.vocab_size)
                        adapter.acknowledge_queued_target_tail(oldp, newp)
                        adapter.observe_target_commit(emitted)
                        tokens.extend(emitted)
                    if stream_callback is not None:
                        with self._stage("streaming"):
                            stream_callback(tokens.copy())
                    cycles.append({"node_count": M, "committed_tokens": len(emitted),
                        "matched_draft_tokens": len(emitted) - 1, "accepted_node_indices": path,
                        "cycle_wall_s": time.perf_counter() - cycle_start, "capacity_fallback": fallback,
                        "device_error_flags": receipt["error"], "CPU_receipts": 1})
                with self._stage("final_audit"):
                    ownership_audit = adapter.finish_request()
                torch.cuda.synchronize()
                end = time.perf_counter()
            except Exception:
                if adapter.ledger.active:
                    adapter.abort_request()
                owner.reset_for_new_request()
                raise
        self.verifier.assert_target_unchanged()
        self.proposer.assert_draft_unchanged()
        decode_s = end - prefill_end
        self.last_cycles, self.last_ranking_trace = cycles, []
        return tokens, {"prefill_s": prefill_end - start, "decode_s": decode_s, "total_s": end - start,
            "decode_tokens_s": (len(tokens) - 1) / decode_s if len(tokens) > 1 else 0.0,
            "generated_tokens": len(tokens), "cycles": len(cycles),
            "mean_committed_tokens_per_cycle": (len(tokens) - 1) / len(cycles) if cycles else 0.0,
            "mean_cycle_ms": 1000.0 * decode_s / len(cycles) if cycles else 0.0,
            "verified_nodes": sum(cycle["node_count"] for cycle in cycles),
            "actual_packed_rows_used": sorted(set(cycle["node_count"] for cycle in cycles)),
            "capacity_fallback_cycles": sum(cycle["capacity_fallback"] for cycle in cycles),
            "excluded_setup_s": prepared["setup_s"], "cumulative_setup_s": self.setup_s,
            "stop_reason": "eos" if tokens[-1] in eos else "max_new_tokens" if eos else "fixed_budget",
            "all_output_tokens_target_verified": True, "final_acceptance_by_original_target": True,
            "streaming_callback_enabled": stream_callback is not None,
            "decode_includes_drafting_tree_building_verification_compaction_and_cpu_transfers": True,
            "variant": self.metadata["variant"], "ranking_policy": self.ranking_policy,
            "ranking_temperature": self.ranking_temperature, "tree_policy": self.tree_policy,
            "ranking_score_label": self.metadata["ranking_score_label"],
            "private_GPU_controller": True, "CPU_receipts": len(cycles),
            "full_lattice_or_tree_or_prediction_bank_copied_to_CPU_during_decode": False,
            "append_only_ownership_audit": ownership_audit,
            "append_only_begin_commit_final_audit_inside_generation_time": True,
            "feature_coordination_logical_bytes": feature_coordination_counts(ids.shape[1],
                [cycle["committed_tokens"] for cycle in cycles])}


def _float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def cpu_frontier_emulation(anchor, ids, logps, M, vocab_size=151936):
    """Independent fixed-frontier CPU emulation of the new kernel algorithm."""
    reference = cpu_reference_tree(anchor, ids, logps, M, vocab_size)
    B, K = len(ids), len(ids[0])
    tokens, parents, depths, scores, bitsets = [anchor], [-1], [0], [0.0], [1]
    frontier = [None] * (M * K)
    for rank in range(K):
        score = min(0.0, float(logps[0][rank]))
        if score > -math.inf:
            frontier[rank] = (score, 1, 0, ids[0][rank])
    for node in range(1, M):
        candidates = [(slot, value) for slot, value in enumerate(frontier) if value is not None]
        if not candidates:
            raise ValueError("Empty frontier")
        slot, (score, depth, parent, token) = min(candidates,
            key=lambda item: (-item[1][0], item[1][1], item[1][2], item[1][3]))
        frontier[slot] = None
        tokens.append(token)
        parents.append(parent)
        depths.append(depth)
        scores.append(score)
        bitsets.append(bitsets[parent] | (1 << node))
        if depth < B:
            for rank in range(K):
                value = min(0.0, float(logps[depth][rank]))
                if value > -math.inf:
                    frontier[node * K + rank] = (score + value, depth + 1, node, ids[depth][rank])
    result = {"tokens": tokens, "parents": parents, "depths": depths, "log_scores": scores,
              "ancestors": [[bool(bits & (1 << column)) for column in range(M)] for bits in bitsets]}
    if result != reference:
        raise AssertionError("New fixed-frontier CPU algorithm differs from the frozen original heap")
    return result


def _synthetic_owner(M, capacity, layers, kv_heads, head, device):
    keys = [torch.randn((1, kv_heads, capacity, head), device=device, dtype=torch.bfloat16) for _ in range(layers)]
    values = [torch.randn_like(value) for value in keys]
    pointers = torch.tensor([value.data_ptr() for value in keys + values], device=device, dtype=torch.long)
    owner = SimpleNamespace(layers=layers, max_len=capacity, node_budget=M, kv_heads=kv_heads, head_dim=head,
        cache_pointer_table=pointers, path_indices=torch.empty((M,), device=device, dtype=torch.long),
        path_count=torch.zeros((1,), device=device, dtype=torch.int32),
        commit_temporary=torch.empty((2, layers, M, kv_heads * head), device=device, dtype=torch.bfloat16),
        _anchor_buffer=torch.empty((1, 1), device=device, dtype=torch.long),
        committed_feature_history=torch.randn((1, capacity, 8192), device=device, dtype=torch.bfloat16))
    snapshot = torch.randn_like(owner.committed_feature_history)
    visible = torch.randint(0, 151936, (1, capacity), device=device, dtype=torch.long)
    return owner, keys, values, snapshot, visible


def _raw_equal(left, right):
    return bool(torch.equal(left.view(torch.int16), right.view(torch.int16)))


def gpu_probe(args):
    """Synthetic model-free proof; full-model parity remains a separate gate."""
    if torch is None or triton is None or not torch.cuda.is_available():
        raise RuntimeError("Use the existing pinned Ada CUDA runtime")
    if "4090" not in torch.cuda.get_device_name():
        raise RuntimeError("This experiment is scoped to the original RTX4090")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    shapes = tuple(int(value) for value in args.shapes.split(","))
    records, timing = [], []
    started = time.perf_counter()
    for M in shapes:
        checked_config(M, args.depth, args.top_k, args.vocab_size)
        if args.capacity < M + 16:
            raise ValueError("Probe capacity must leave at least16 prefix rows around the largest physical tree")
        controller = GPUTreeController(M, args.depth, args.top_k, args.vocab_size, device)
        owner, keys, values, snapshot, visible = _synthetic_owner(M, args.capacity, args.layers, args.kv_heads, args.head, device)
        entry = SimpleNamespace(input_ids=torch.empty((1, M), device=device, dtype=torch.long),
            depths=torch.empty((M,), device=device, dtype=torch.long),
            ancestors=torch.empty((M, M), device=device, dtype=torch.bool),
            base=torch.tensor([7], device=device, dtype=torch.long),
            next_ids=torch.empty((M,), device=device, dtype=torch.long),
            features=torch.randn((M, 8192), device=device, dtype=torch.bfloat16))
        owner._pending_entry = entry
        prep = controller.prepare(owner, entry, snapshot, visible)
        for case in range(args.cases):
            ids = [rng.sample(range(args.vocab_size - 8), args.top_k) for _ in range(args.depth)]
            if case % 4 == 0:
                logps = [[-0.25] * args.top_k for _ in range(args.depth)]
                kind = "exact_score_ties"
            elif case % 4 == 1:
                logps = [[0.0] * args.top_k for _ in range(args.depth)]
                kind = "zero_score_depth_parent_token_ties"
            elif case % 4 == 2:
                logps = [[_float32(-rng.uniform(0.001, 6.0)) for _ in range(args.top_k)] for _ in range(args.depth)]
                kind = "random_unordered_marginals"
            else:
                logps = [[_float32(-0.001 - rank * 0.01 - row * 0.0001)
                          for rank in range(args.top_k)] for row in range(args.depth)]
                kind = "deep_prefix_priorities"
            anchor = 7
            reference = cpu_reference_tree(anchor, ids, logps, M, args.vocab_size)
            gpu_ids = torch.tensor([ids], device=device, dtype=torch.long)
            gpu_logps = torch.tensor([logps], device=device, dtype=torch.float32)
            owner._anchor_buffer.fill_(anchor)
            controller.launch_tree(gpu_ids, gpu_logps, owner._anchor_buffer, entry)
            torch.cuda.synchronize()
            actual = {"tokens": entry.input_ids.reshape(-1).cpu().tolist(), "parents": controller.parents.cpu().tolist(),
                "depths": entry.depths.cpu().tolist(), "ancestors": entry.ancestors.cpu().tolist(),
                "log_scores": controller.scores.cpu().tolist()}
            if actual != reference or int(controller.error.item()) != 0:
                raise AssertionError("GPU tree differs from frozen CPU heap: M=" + str(M) + " case=" + str(case))
            if not torch.equal(controller.scores.view(torch.int64).cpu(), torch.tensor(reference["log_scores"], dtype=torch.float64).view(torch.int64)):
                raise AssertionError("GPU FP64 serial prefix score bits differ from the frozen CPU")
            predictions = [rng.randrange(args.vocab_size - 8) for _ in range(M)]
            node = 0
            if case % 3:
                for _ in range(args.depth):
                    children = [i for i in range(1, M) if reference["parents"][i] == node]
                    if not children:
                        break
                    child = children[0]
                    predictions[node] = reference["tokens"][child]
                    node = child
            eos = [args.vocab_size - 1, args.vocab_size - 2]
            if case % 3 == 2:
                predictions[node] = eos[0]
            remaining = 1 + case % (args.depth + 1)
            expected_path, expected_emitted = cpu_reference_accept(reference, predictions, remaining, eos, args.depth + 1)
            entry.next_ids.copy_(torch.tensor(predictions, device=device, dtype=torch.long))
            controller.configure_eos(eos)
            before_kv = [value.clone() for value in keys + values]
            before_history, before_snapshot, before_visible = owner.committed_feature_history.clone(), snapshot.clone(), visible.clone()
            controller.launch_accept(entry.next_ids, owner, remaining)
            controller.queue_kv_commit(owner, entry)
            controller.queue_feature_visible_append(owner, entry, snapshot, visible)
            receipt = controller.receive(owner, entry)
            if receipt["error"] or receipt["path"] != expected_path or receipt["emitted"] != expected_emitted or receipt["anchor"] != expected_emitted[-1]:
                raise AssertionError("GPU original-target acceptance receipt differs from CPU greedy reference: " + repr(receipt))
            count, base = len(expected_path), 7
            index = torch.tensor([base + value for value in expected_path], device=device, dtype=torch.long)
            for actual_kv, original_kv in zip(keys + values, before_kv):
                expected_kv = original_kv.clone()
                expected_kv[:, :, base:base + count].copy_(original_kv.index_select(2, index))
                if not _raw_equal(actual_kv, expected_kv):
                    raise AssertionError("Frozen GPU K/V compaction changed selected bits, old prefix, or unused tail")
            feature_index = torch.tensor(expected_path, device=device, dtype=torch.long)
            expected_features = entry.features.index_select(0, feature_index).unsqueeze(0)
            before_history[:, base:base + count].copy_(expected_features)
            before_snapshot[:, base:base + count].copy_(expected_features)
            before_visible[:, base + 1:base + count + 1].copy_(torch.tensor([expected_emitted], device=device, dtype=torch.long))
            if not _raw_equal(owner.committed_feature_history, before_history) or not _raw_equal(snapshot, before_snapshot) or not torch.equal(visible, before_visible):
                raise AssertionError("GPU original-target feature/visible append changed selected bits or an untouched row")
            records.append({"M": M, "case": case, "kind": kind, "target_path_count": count,
                "tree_and_FP64_score_bits_exact_frozen_CPU": True, "ancestor_mask_exact": True,
                "greedy_EOS_remaining_acceptance_exact": True, "all_KV_layers_entire_buffer_bits_exact": True,
                "target_feature_snapshot_and_visible_append_exact": True,
                "high_index_parent_nodes": sum(parent >= 64 for parent in reference["parents"]),
                "target_model_forward_called": False})
        # Invalid device values must be sanitized for embeddings and produce
        # zero accepted rows; no K/V/feature/visible changes may survive.
        for bad_kind in ("negative_ID", "out_of_vocabulary_ID", "NaN_score", "positive_infinity", "duplicate_ID", "empty_frontier"):
            bad_ids = [row[:] for row in ids]
            bad_logps = [row[:] for row in logps]
            if bad_kind == "negative_ID": bad_ids[0][0] = -1
            elif bad_kind == "out_of_vocabulary_ID": bad_ids[0][0] = args.vocab_size
            elif bad_kind == "NaN_score": bad_logps[0][0] = math.nan
            elif bad_kind == "positive_infinity": bad_logps[0][0] = math.inf
            elif bad_kind == "duplicate_ID":
                if args.top_k < 2: continue
                bad_ids[0][1] = bad_ids[0][0]
            elif bad_kind == "empty_frontier": bad_logps[0] = [-math.inf] * args.top_k
            bad_gpu_ids = torch.tensor([bad_ids], device=device, dtype=torch.long)
            bad_gpu_logps = torch.tensor([bad_logps], device=device, dtype=torch.float32)
            before_kv = [value.clone() for value in keys + values]
            before_history, before_snapshot, before_visible = owner.committed_feature_history.clone(), snapshot.clone(), visible.clone()
            owner._anchor_buffer.fill_(7)
            controller.launch_tree(bad_gpu_ids, bad_gpu_logps, owner._anchor_buffer, entry)
            controller.launch_accept(entry.next_ids, owner, args.depth + 1)
            controller.queue_kv_commit(owner, entry)
            controller.queue_feature_visible_append(owner, entry, snapshot, visible)
            receipt = controller.receive(owner, entry)
            if not receipt["error"] or receipt["count"] or receipt["emitted"]:
                raise AssertionError("Invalid device candidate escaped zero-count rejection")
            if any(not _raw_equal(a, b) for a, b in zip(keys + values, before_kv)) or not _raw_equal(owner.committed_feature_history, before_history) or not _raw_equal(snapshot, before_snapshot) or not torch.equal(visible, before_visible):
                raise AssertionError("Rejected invalid candidates wrote persistent K/V, features or visible output")
            if int(entry.input_ids.min().item()) < 0 or int(entry.input_ids.max().item()) >= args.vocab_size:
                raise AssertionError("Invalid candidate sanitization left an unsafe embedding ID")
            records.append({"M": M, "kind": bad_kind, "error_flags": receipt["error"], "zero_KV_feature_or_visible_writes": True,
                "zero_output_emitted": True, "embedding_IDs_sanitized": True, "error_anchor_must_be_discarded_by_logical_reset": True})
        # Measure only already compiled tree+mask kernels; not inference speed.
        owner._anchor_buffer.fill_(7)
        for _ in range(3):
            controller.launch_tree(gpu_ids, gpu_logps, owner._anchor_buffer, entry)
        torch.cuda.synchronize()
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        began = time.perf_counter()
        start_event.record()
        for _ in range(args.timing_repeats):
            controller.launch_tree(gpu_ids, gpu_logps, owner._anchor_buffer, entry)
        end_event.record()
        end_event.synchronize()
        timing.append({"M": M, "synthetic_preparation": prep,
            "tree_and_mask_GPU_mean_ms": start_event.elapsed_time(end_event) / args.timing_repeats,
            "tree_and_mask_synchronized_wall_mean_ms": 1000 * (time.perf_counter() - began) / args.timing_repeats,
            "repeats": args.timing_repeats, "target_forward_or_generation_included": False})
    return {"schema": "lightreason-private-GPU-tree-model-free-probe-v1", "arguments": vars(args),
            "gpu": torch.cuda.get_device_name(), "records": records, "timing": timing,
            "summary": {"all_requested_checks_passed": True, "records": len(records),
                "original_target_model_loaded_or_called": False, "full_model_cache_parity_established": False,
                "inference_speedup_or_quality_improvement_established": False,
                "wall_s": time.perf_counter() - started}}


def cpu_self_test():
    rng = random.Random(73901)
    count = 0
    for M in PACKED_BUDGETS:
        for mode in range(8):
            ids = [rng.sample(range(1000), 8) for _ in range(15)]
            scores = [[-0.25] * 8 for _ in range(15)] if mode % 2 else [
                [_float32(-rng.uniform(0.0001, 5)) for _ in range(8)] for _ in range(15)]
            tree = cpu_frontier_emulation(7, ids, scores, M, 1000)
            for node, row in enumerate(tree["ancestors"]):
                if not row[0] or not row[node] or sum(row) != tree["depths"][node] + 1:
                    raise AssertionError("128-bit rooted ancestor closure failed")
            count += 1
    return {"CPU_frozen_heap_vs_fixed_frontier_cases": count, "all_passed": True, "GPU_work_performed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--shapes", default="16,32,48,64,96,128")
    parser.add_argument("--depth", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--seed", type=int, default=73901)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head", type=int, default=128)
    parser.add_argument("--timing-repeats", type=int, default=30)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(cpu_self_test(), indent=2))
    elif args.probe:
        if not args.out or args.cases < 1 or args.timing_repeats < 1:
            parser.error("Use --out and positive case/timing counts for the synthetic GPU probe")
        try:
            result = gpu_probe(args)
        except Exception as error:
            result = {"schema": "lightreason-private-GPU-tree-model-free-probe-v1", "arguments": vars(args),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "summary": {"all_requested_checks_passed": False,
                    "fatal_error": type(error).__name__ + ": " + str(error),
                    "original_target_model_loaded_or_called": False,
                    "inference_speedup_or_quality_improvement_established": False}}
            Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
            raise
        result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["summary"], indent=2))
    else:
        parser.error("Use --self-test or --probe; inference factory is GPUTreeControl3BRunner.from_control")

