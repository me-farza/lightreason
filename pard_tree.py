"""PARD marginal lattices and a CPU best-first speculative token tree.

The frozen auxiliary model supplies position-wise marginal predictions from
parallel masks. A row does NOT condition on the particular earlier tokens a
tree branch chooses. Summed marginal log probabilities are proposal scores,
not a Markov model or authoritative target likelihoods. The unchanged target
must verify each committed token, causal branch, and EOS decision.

GraphPARDLatticeProposer(..., top_k=8).propose_lattice(visible_ids,B) returns
GPU IDs[1,B,K] and FP32 log probabilities[1,B,K]. Transfer these to CPU
explicitly before best_first_tree(anchor, ids, logps, node_budget=32).

The tree dict contains CPU tensors: tokens[1,M], parents[M], depths[M],
ancestors[M,M], and float64 log_scores[M]. Root0 is the public anchor, has
parent−1/depth0, and counts toward the budget. Ancestor masks INCLUDE SELF.
No expected answers, future target outputs, or stopping distance are inputs.
"""
from __future__ import annotations

import hashlib
import heapq
import math
from pathlib import Path

import torch

from graph_pard_proposer import GraphPARDProposer


class GraphPARDLatticeProposer(GraphPARDProposer):
    """Separate graph family retaining top-K PARD marginals in permanent buffers."""
    def __init__(self, *args, top_k=8, **kwargs):
        if type(top_k) is not int or not 1 <= top_k <= 64:
            raise ValueError("Use a fixed top_k between1 and64 for this graph family")
        self.top_k = top_k
        self.lattice_ids, self.lattice_logps = {}, {}
        super().__init__(*args, **kwargs)
        if top_k > self.target_vocab_size:
            raise ValueError("Lattice top_k exceeds the auxiliary vocabulary")
        self.metadata = {**self.metadata, "proposal_kind": "PARD position-marginal lattice",
            "top_k": top_k, "lattice_scores_dtype": "torch.float32",
            "branch_scores_are_approximate_marginal_products": True,
            "lattice_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    @torch.no_grad()
    def prepare(self, num_draft_tokens):
        block = num_draft_tokens
        if type(block) is not int or not 1 <= block <= 32:
            raise ValueError("Choose a public PARD proposal count between1 and32")
        # The superclass performs virtual _call warmups/captures, so every
        # output address must already exist before entering that preparation.
        with torch.inference_mode(False), torch.no_grad():
            if block not in self.lattice_ids:
                self.lattice_ids[block] = torch.empty((1, block, self.top_k),
                    device=self.device, dtype=torch.long)
                self.lattice_logps[block] = torch.empty((1, block, self.top_k),
                    device=self.device, dtype=torch.float32)
        result = super().prepare(block)
        return {**result, "lattice_top_k": self.top_k}

    def _call(self, entry, block):
        positions = self.gpu_base + entry["offsets"]
        allowed = self.key_positions.view(1, 1, 1, -1) <= positions.view(1, 1, -1, 1)
        output = self.forward(input_ids=entry["input_ids"], past_key_values=self.cache,
            cache_position=positions, position_ids=positions.view(1, -1),
            attention_mask=allowed, use_cache=True, logits_to_keep=block,
            output_attentions=False, output_hidden_states=False, return_dict=True)
        raw = output.logits[:, -block:]
        # Keep ordinary raw argmax independently: torch.topk tie ordering can
        # differ from torch.argmax's first-index tie rule. Both are proposals.
        self.outputs[block].copy_(raw.argmax(-1))
        logits = raw.float()
        values, ids = torch.topk(logits, self.top_k, dim=-1, largest=True, sorted=True)
        logps = values - torch.logsumexp(logits, dim=-1, keepdim=True)
        self.lattice_ids[block].copy_(ids)
        self.lattice_logps[block].copy_(logps)

    @torch.no_grad()
    def propose_lattice(self, visible_ids, num_draft_tokens):
        super().propose(visible_ids, num_draft_tokens)
        ids = self.lattice_ids[num_draft_tokens]
        logps = self.lattice_logps[num_draft_tokens]
        if self.last_report is not None:
            self.last_report.update({"proposal_kind": "position-marginal lattice", "top_k": self.top_k,
                                     "branch_conditioning_is_approximate": True})
        # As with the base argmax buffer, copies remain valid across subsequent
        # replays. copy_output=False explicitly exposes volatile graph buffers.
        return (ids.clone(), logps.clone()) if self.copy_output else (ids, logps)


def _build_tree_lists(anchor, top_ids, logps, node_budget):
    """Dependency-free CPU core, deliberately separate from tensor conversion."""
    if type(anchor) is not int or anchor < 0:
        raise ValueError("Anchor must be a public nonnegative integer token ID")
    if type(node_budget) is not int or node_budget < 1:
        raise ValueError("Node budget must be positive and includes the anchor root")
    if not top_ids or len(top_ids) != len(logps):
        raise ValueError("Use matching nonempty position-major ID/logprob rows")
    width = len(top_ids[0])
    if width < 1 or any(len(row) != width for row in top_ids) or any(len(row) != width for row in logps):
        raise ValueError("ID/logprob rows must form matching rectangular B×K matrices")
    rows = []
    for ids, scores in zip(top_ids, logps):
        unique = {}
        for token, score in zip(ids, scores):
            if type(token) is not int or token < 0:
                raise ValueError("Lattice IDs must be nonnegative integers")
            value = float(score)
            if math.isnan(value) or value == math.inf or value > 1e-5:
                raise ValueError("Use finite/nonpositive marginal log probabilities (negative infinity is allowed)")
            # Tiny positive roundoff must not make deeper prefixes outrank
            # their parents. Clamping changes proposal ranking only.
            value = min(0.0, value)
            unique[token] = max(unique.get(token, -math.inf), value)
        rows.append(sorted(((token, score) for token, score in unique.items() if score > -math.inf),
                           key=lambda item: (-item[1], item[0])))
    tokens, parents, depths, scores = [anchor], [-1], [0], [0.0]
    heap = []

    def enqueue(parent):
        row = depths[parent]
        if row >= len(rows):
            return
        for token, logp in rows[row]:
            score = scores[parent] + logp
            # Deterministic ties: shallower depth, existing parent index, ID.
            heapq.heappush(heap, (-score, row + 1, parent, token))

    enqueue(0)
    while heap and len(tokens) < node_budget:
        negative_score, depth, parent, token = heapq.heappop(heap)
        node = len(tokens)
        tokens.append(token)
        parents.append(parent)
        depths.append(depth)
        scores.append(-negative_score)
        enqueue(node)
    ancestors = []
    for node, parent in enumerate(parents):
        row = [False] * len(tokens) if parent < 0 else list(ancestors[parent])
        row[node] = True
        ancestors.append(row)
    return {"tokens": tokens, "parents": parents, "depths": depths,
            "ancestors": ancestors, "log_scores": scores}


def _cpu_matrix(value, name):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"Transfer {name} to CPU explicitly and include its synchronization in online timing")
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape[B,K] or[1,B,K]")
        if name == "top_ids" and value.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
            raise ValueError("top_ids must use an integer dtype")
        return value.tolist()
    rows = list(value)
    if rows and len(rows) == 1 and isinstance(rows[0], (list, tuple)) and rows[0] and isinstance(rows[0][0], (list, tuple)):
        rows = list(rows[0])
    return [list(row) for row in rows]


def best_first_tree(anchor, top_ids, logps, node_budget=32):
    """CPU prefix-closed tree from auxiliary marginals, including root/self."""
    if isinstance(anchor, torch.Tensor):
        if anchor.device.type != "cpu" or anchor.numel() != 1 or anchor.dtype not in (torch.int32, torch.int64):
            raise ValueError("A tensor anchor must be one CPU integer token")
        anchor = int(anchor.item())
    built = _build_tree_lists(anchor, _cpu_matrix(top_ids, "top_ids"),
                              _cpu_matrix(logps, "logps"), node_budget)
    count = len(built["tokens"])
    return {"tokens": torch.tensor([built["tokens"]], dtype=torch.long, device="cpu"),
            "parents": torch.tensor(built["parents"], dtype=torch.long, device="cpu"),
            "depths": torch.tensor(built["depths"], dtype=torch.long, device="cpu"),
            "ancestors": torch.tensor(built["ancestors"], dtype=torch.bool, device="cpu"),
            "log_scores": torch.tensor(built["log_scores"], dtype=torch.float64, device="cpu"),
            "node_count": count, "node_budget": node_budget,
            "root_index": 0, "ancestor_mask_includes_self": True,
            "proposal_scores_are_approximate": True,
            "target_verification_required": True}


def _self_test_core():
    """CPU structural checks; never loads a model, CUDA, or reference answers."""
    rows = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
    probs = [[-0.1, -0.8, -1.2], [-0.2, -0.7, -1.4], [-0.3, -0.9, -1.5]]
    tree = _build_tree_lists(7, rows, probs, 19)
    assert len(tree["tokens"]) == 19
    assert tree["parents"][0] == -1 and tree["depths"][0] == 0
    assert tree["tokens"][1] == 10
    for node in range(len(tree["tokens"])):
        parent = tree["parents"][node]
        assert parent < node
        assert tree["ancestors"][node][node] and tree["ancestors"][node][0]
        if node:
            assert tree["depths"][node] == tree["depths"][parent] + 1
            assert tree["tokens"][node] in rows[tree["depths"][node] - 1]
            assert sum(tree["ancestors"][node]) == tree["depths"][node] + 1
            assert tree["log_scores"][node] <= tree["log_scores"][parent]
        children = [tree["tokens"][index] for index, value in enumerate(tree["parents"]) if value == node]
        assert len(children) == len(set(children))
    all_scores, frontier = [0.0], [0.0]
    for row in probs:
        frontier = [parent + logp for parent in frontier for logp in row]
        all_scores.extend(frontier)
    expected_scores = sorted(all_scores, reverse=True)[:19]
    assert all(abs(left - right) < 1e-12 for left, right in
               zip(sorted(tree["log_scores"], reverse=True), expected_scores))
    assert _build_tree_lists(7, rows, probs, 1)["tokens"] == [7]
    deduplicated = _build_tree_lists(7, [[5, 5, 6]], [[-0.3, -0.1, -0.2]], 99)
    assert deduplicated["tokens"] == [7, 5, 6]
    zero_mass = _build_tree_lists(7, [[5]], [[-math.inf]], 32)
    assert zero_mass["tokens"] == [7]
    repeated = _build_tree_lists(7, [[7, 8], [7, 8]], [[0.0, 0.0], [0.0, 0.0]], 32)
    assert repeated["depths"] == [0, 1, 1, 2, 2, 2, 2]
    assert repeated == _build_tree_lists(7, [[7, 8], [7, 8]], [[0.0, 0.0], [0.0, 0.0]], 32)
    for bad_ids, bad_scores in (([[1]], [[float("nan")]]), ([[1]], [[0.1]]), ([[1, 2]], [[-1.0]])):
        try:
            _build_tree_lists(7, bad_ids, bad_scores, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid lattice was accepted")
    return {"passed": True, "tested_node_count": len(tree["tokens"]),
            "checks": ["parent-before-child", "prefix closure", "self/ancestor mask", "depth alignment",
                       "per-parent uniqueness", "global best-first ranking", "deterministic ties", "root-only budget", "invalid scores/shapes"]}


def self_test():
    """CPU tensor API plus structural tests; performs no GPU operation."""
    result = _self_test_core()
    ids = torch.tensor([[[1, 2], [3, 4], [5, 6]]], dtype=torch.long)
    logps = torch.tensor([[[-0.1, -0.9], [-0.2, -0.8], [-0.3, -0.7]]], dtype=torch.float32)
    tree = best_first_tree(torch.tensor([9], dtype=torch.long), ids, logps, 12)
    count = tree["node_count"]
    assert count == 12 and tree["tokens"].shape == (1, count)
    assert tree["parents"].shape == tree["depths"].shape == (count,)
    assert tree["ancestors"].shape == (count, count) and tree["ancestors"].dtype == torch.bool
    assert tree["tokens"].device.type == tree["ancestors"].device.type == "cpu"
    for bad_ids, bad_logps in ((ids.reshape(-1), logps), (ids[:, :2], logps), (ids.float(), logps)):
        try:
            best_first_tree(9, bad_ids, bad_logps, 12)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid tensor API shape/dtype was accepted")
    return {**result, "cpu_tensor_api_checked": True}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test()))
