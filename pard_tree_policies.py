"""Proposal-only PARD tree policies; the original target remains authoritative.

Mandatory-trunk policy reserves root+B nodes for a full top-1 marginal chain,
then spends the remaining budget on best-first side branches. This prevents
cumulative log-probability ranking from removing a useful deep linear draft.
The position marginals still do not condition on branch choices; scores are
approximate proposal priorities, never target likelihoods or acceptance rules.

trunk_best_first_tree returns the same CPU tensor schema as pard_tree:
tokens[1,M], parents[M] (root−1), depths[M] (root0), ancestors[M,M] INCLUDING
self, log_scores[M], plus explicit policy metadata. No future target tokens,
reference outputs, answers, or stopping distances are inputs.

Default trunk selection uses highest marginal logp and original input order
for ties. Pass the plain PARD argmax IDs as trunk_ids to guarantee that exact
linear chain; each supplied ID must be present in its lattice row. PyTorch's
topk and argmax tie rules need not choose the same ID, so this distinction is
reported rather than silently claiming linear-argmax coverage.
"""
from __future__ import annotations

import heapq
import math

import torch

from pard_tree import _cpu_matrix


def _trunk_tree_lists(anchor, top_ids, logps, node_budget, trunk_ids=None):
    """Dependency-free CPU construction, with parents allocated before children."""
    if type(anchor) is not int or anchor < 0:
        raise ValueError("Anchor must be a nonnegative public token ID")
    if not top_ids or len(top_ids) != len(logps):
        raise ValueError("Use matching nonempty position-major ID/logprob matrices")
    block = len(top_ids)
    if type(node_budget) is not int or node_budget < block + 1:
        raise ValueError("Mandatory trunk requires node_budget >= B+1, including root")
    width = len(top_ids[0])
    if width < 1 or any(len(row) != width for row in top_ids) or any(len(row) != width for row in logps):
        raise ValueError("ID/logprob rows must form matching rectangular B×K matrices")
    if trunk_ids is not None and len(trunk_ids) != block:
        raise ValueError("Explicit trunk_ids must contain exactly B IDs")
    normalised, trunk = [], []
    for depth, (ids, scores) in enumerate(zip(top_ids, logps)):
        unique = {}
        first_best, best_score = None, -math.inf
        for token, score in zip(ids, scores):
            if type(token) is not int or token < 0:
                raise ValueError("Lattice IDs must be nonnegative integers")
            score = float(score)
            if math.isnan(score) or score == math.inf or score > 1e-5:
                raise ValueError("Use nonpositive marginal log probabilities; negative infinity is allowed")
            score = min(score, 0.0)
            unique[token] = max(unique.get(token, -math.inf), score)
            if score > best_score:
                first_best, best_score = token, score
        if first_best is None:
            raise ValueError("Every mandatory-trunk position needs at least one finite marginal candidate")
        selected = first_best if trunk_ids is None else trunk_ids[depth]
        if type(selected) is not int or selected < 0 or selected not in unique or not math.isfinite(unique[selected]):
            raise ValueError("Every explicit trunk ID must be a finite-scored candidate in its lattice row")
        trunk.append(selected)
        normalised.append(unique)

    tokens, parents, depths, scores = [anchor], [-1], [0], [0.0]
    children = {}
    # Mandatory prefix is allocated first, guaranteeing a simple known path
    # 0->1->...->B independently of every alternative's joint score.
    for depth, token in enumerate(trunk):
        parent, node = len(tokens) - 1, len(tokens)
        tokens.append(token)
        parents.append(parent)
        depths.append(depth + 1)
        scores.append(scores[parent] + normalised[depth][token])
        children[(parent, token)] = node

    heap = []

    def enqueue(parent):
        row = depths[parent]
        if row >= block:
            return
        for token, logp in normalised[row].items():
            if not math.isfinite(logp) or (parent, token) in children:
                continue
            total = scores[parent] + logp
            heapq.heappush(heap, (-total, row + 1, parent, token))

    # Deep mandatory nodes already exist, so their alternative children are
    # eligible immediately, together with root and shallower side branches.
    for parent in range(len(tokens)):
        enqueue(parent)
    while heap and len(tokens) < node_budget:
        negative_score, depth, parent, token = heapq.heappop(heap)
        if (parent, token) in children:
            continue
        node = len(tokens)
        children[(parent, token)] = node
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
            "ancestors": ancestors, "log_scores": scores,
            "mandatory_trunk_ids": trunk, "mandatory_trunk_nodes": list(range(block + 1)),
            "trunk_depth": block, "alternative_nodes": len(tokens) - block - 1,
            "trunk_source": "explicit_plain_pard_argmax" if trunk_ids is not None else "marginal_top1_input_order_ties"}


def _cpu_trunk_ids(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.dtype not in (torch.int32, torch.int64):
            raise ValueError("Explicit trunk IDs must be CPU integer IDs")
        if value.ndim != 1 and not (value.ndim == 2 and value.shape[0] == 1):
            raise ValueError("Explicit trunk IDs must have shape[B] or[1,B]")
        return value.reshape(-1).tolist()
    result = list(value)
    if len(result) == 1 and isinstance(result[0], (list, tuple)):
        result = list(result[0])
    return result


def trunk_best_first_tree(anchor, top_ids, logps, node_budget=64, trunk_ids=None):
    """Reserve the full linear draft, then fill a prefix-closed best-first tree."""
    if isinstance(anchor, torch.Tensor):
        if anchor.device.type != "cpu" or anchor.numel() != 1 or anchor.dtype not in (torch.int32, torch.int64):
            raise ValueError("A tensor anchor must be one CPU integer token")
        anchor = int(anchor.item())
    built = _trunk_tree_lists(anchor, _cpu_matrix(top_ids, "top_ids"),
        _cpu_matrix(logps, "logps"), node_budget, _cpu_trunk_ids(trunk_ids))
    count = len(built["tokens"])
    policy = {"name": "mandatory_trunk_then_best_first_fill", "trunk_source": built["trunk_source"],
              "mandatory_trunk_ids": built["mandatory_trunk_ids"],
              "mandatory_trunk_nodes": built["mandatory_trunk_nodes"],
              "trunk_depth": built["trunk_depth"], "alternative_nodes": built["alternative_nodes"],
              "full_supplied_trunk_coverage_guaranteed": True,
              "target_verification_required": True}
    return {"tokens": torch.tensor([built["tokens"]], dtype=torch.long, device="cpu"),
            "parents": torch.tensor(built["parents"], dtype=torch.long, device="cpu"),
            "depths": torch.tensor(built["depths"], dtype=torch.long, device="cpu"),
            "ancestors": torch.tensor(built["ancestors"], dtype=torch.bool, device="cpu"),
            "log_scores": torch.tensor(built["log_scores"], dtype=torch.float64, device="cpu"),
            "node_count": count, "node_budget": node_budget, "root_index": 0,
            "ancestor_mask_includes_self": True, "proposal_scores_are_approximate": True,
            "target_verification_required": True, "policy": policy}


mandatory_trunk_tree = trunk_best_first_tree


def _self_test_core():
    rows = [list(range(depth * 8, depth * 8 + 8)) for depth in range(15)]
    logps = [[-1.0 - rank / 10 for rank in range(8)] for _ in rows]
    expected_trunk = [row[0] for row in rows]
    for budget in (16, 32, 64, 128):
        tree = _trunk_tree_lists(999, rows, logps, budget)
        assert len(tree["tokens"]) == budget
        assert tree["tokens"][1:16] == expected_trunk
        assert tree["parents"][:16] == [-1] + list(range(15))
        assert tree["depths"][:16] == list(range(16))
        assert max(tree["depths"]) == 15
        paths = set()
        for node, parent in enumerate(tree["parents"]):
            assert parent < node and tree["ancestors"][node][node]
            assert tree["ancestors"][node][0]
            assert sum(tree["ancestors"][node]) == tree["depths"][node] + 1
            if node:
                assert tree["depths"][node] == tree["depths"][parent] + 1
                assert tree["log_scores"][node] <= tree["log_scores"][parent]
            chain, current = [], node
            while current >= 0:
                chain.append(tree["tokens"][current])
                current = tree["parents"][current]
            path = tuple(reversed(chain))
            assert path not in paths
            paths.add(path)
            offspring = [tree["tokens"][i] for i, value in enumerate(tree["parents"]) if value == node]
            assert len(offspring) == len(set(offspring))
        assert tree == _trunk_tree_lists(999, rows, logps, budget)
    explicit = [row[1] for row in rows]
    supplied = _trunk_tree_lists(999, rows, logps, 32, explicit)
    assert supplied["tokens"][1:16] == explicit
    assert supplied["trunk_source"] == "explicit_plain_pard_argmax"
    ties = _trunk_tree_lists(9, [[2, 1], [4, 3]], [[-0.1, -0.1], [-0.2, -0.2]], 7)
    assert ties["tokens"][1:3] == [2, 4]
    duplicate = _trunk_tree_lists(9, [[1, 1, 2]], [[-0.3, -0.1, -0.2]], 9)
    assert duplicate["tokens"] == [9, 1, 2]
    for ids, values, budget, trunk in ((rows, logps, 15, None), ([[1]], [[float("nan")]], 2, None),
            ([[1]], [[-1.0]], 2, [2]), ([[1, 2]], [[-1.0]], 2, None), ([[1]], [[-math.inf]], 2, None)):
        try:
            _trunk_tree_lists(9, ids, values, budget, trunk)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid mandatory-trunk inputs were accepted")
    return {"passed": True, "tested_budgets": [16, 32, 64, 128],
            "checks": ["complete15-deep trunk", "parent-before-child", "prefix/self closure", "per-parent uniqueness",
                       "exact node count", "determinism", "explicit trunk IDs", "ties", "malformed inputs"]}


def self_test():
    result = _self_test_core()
    ids = torch.tensor([[[1, 2], [3, 4], [5, 6]]], dtype=torch.long)
    values = torch.tensor([[[-0.1, -0.8], [-0.2, -0.7], [-0.3, -0.6]]], dtype=torch.float32)
    tree = trunk_best_first_tree(torch.tensor([9]), ids, values, 12)
    assert tree["tokens"].shape == (1, 12)
    assert tree["ancestors"].shape == (12, 12) and tree["ancestors"].dtype == torch.bool
    assert tree["tokens"].tolist()[0][:4] == [9, 1, 3, 5]
    assert tree["policy"]["full_supplied_trunk_coverage_guaranteed"]
    return {**result, "cpu_tensor_schema_checked": True}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test()))
