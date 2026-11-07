"""Same original target weights/equations, FP-order-relaxed tree speculation.

PARD marginals rank a public candidate tree. The original trained target runs
every node with only its actual causal ancestors visible. Only the root path
matching target argmax predictions is committed, including a target correction.
Tree batching can change floating-point evaluation, so exact sequence agreement
is a reported metric and answer-quality validation remains separate.
"""
from __future__ import annotations

import time
import torch

from pard_proposer import PARDProposer
from pard_tree import GraphPARDLatticeProposer, best_first_tree
from pard_tree_policies import trunk_best_first_tree
from numerical3b_tree_verifier import NumericalTreeVerifier


class Numerical3BTreeRunner:
    def __init__(self, model, tokenizer, draft_path, *, max_len=1024,
                 node_budget=64, draft_tokens=15, top_k=8,
                 compile_static=False, compile_draft=False, cuda_graph=True,
                 share_pool=True, warmups=3, validate_topology=True,
                 commit_engine="triton", tree_policy="best-first"):
        if type(node_budget) is not int or not 2 <= node_budget <= 128:
            raise ValueError("Use a public target tree budget2..128")
        if type(draft_tokens) is not int or not 1 <= draft_tokens <= 32:
            raise ValueError("Use public parallel draft depth1..32")
        self.model, self.tokenizer = model, tokenizer
        self.max_len, self.node_budget = max_len, node_budget
        self.draft_tokens, self.top_k = draft_tokens, top_k
        if tree_policy not in ("best-first", "trunk"):
            raise ValueError("Use tree_policy best-first or trunk")
        self.tree_policy = tree_policy
        self.verifier = NumericalTreeVerifier(model, max_len=max_len,
            node_budget=node_budget, cuda_graph=cuda_graph,
            compile_static=compile_static, share_pool=share_pool, warmups=warmups,
            commit_engine=commit_engine, validate_topology=validate_topology)
        base = PARDProposer(draft_path, target_tokenizer=tokenizer,
            device=model.device, cache_mode="incremental", attn_implementation="sdpa",
            max_cache_len=max_len)
        self.proposer = GraphPARDLatticeProposer(base, max_len=max_len,
            top_k=top_k, compile_model=compile_draft,
            share_pool=share_pool, warmups=warmups)
        self.visible = torch.empty((1, max_len), device=model.device, dtype=torch.long)
        self.setup_s, self.prepared = 0.0, False
        self.metadata = {"variant": "original_target_weights_relaxed_fp_order_best_first_tree",
            "strict_token_matching_required": False,
            "exact_token_agreement_must_be_reported": True,
            "target_weights_and_equations_unchanged": True,
            "target_verifier": self.verifier.metadata,
            "proposer": self.proposer.metadata,
            "proposal_scores": "Position-marginal products; not target probabilities or branch-conditioned draft scores",
            "node_budget": node_budget, "draft_tokens": draft_tokens, "top_k": top_k,
            "tree_policy": tree_policy,
            "compile_static": compile_static, "compile_draft": compile_draft,
            "max_len": max_len,
            "generation": "Original target raw greedy argmax; original chat template; target EOS only"}

    def _check(self, ids, n):
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("Only one nonempty unpadded original-tokenized prompt is supported")
        if type(n) is not int or n < 1:
            raise ValueError("Use a positive public output budget")
        if ids.shape[1] + n + max(self.node_budget, self.draft_tokens) > self.max_len:
            raise ValueError("Prompt, full output budget and tree reserve exceed max_len")

    def prepare(self, ids, n):
        self._check(ids, n)
        if self.prepared:
            return {"setup_s": 0.0, "new_graphs": 0}
        torch.cuda.synchronize()
        start = time.perf_counter()
        target = self.verifier.prepare()
        draft = self.proposer.prepare(self.draft_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.setup_s += elapsed
        self.prepared = True
        return {"setup_s": elapsed, "target": target, "draft": draft,
                "new_graphs": len(self.verifier.graphs)}

    @staticmethod
    def accept_path(tree, predicted, remaining, eos):
        tokens = tree["tokens"].reshape(-1).tolist()
        parents = tree["parents"].tolist()
        children = [{} for _ in tokens]
        for node in range(1, len(tokens)):
            parent, token = parents[node], tokens[node]
            if token in children[parent]:
                raise RuntimeError("Candidate tree has ambiguous duplicate children")
            children[parent][token] = node
        node, path, emitted = 0, [], []
        while True:
            path.append(node)
            value = predicted[node]
            emitted.append(value)
            if len(emitted) >= remaining or (eos and value in eos):
                break
            child = children[node].get(value)
            if child is None:
                break
            node = child
        return path, emitted

    @torch.no_grad()
    def run(self, ids, n, eos_ids=None):
        self._check(ids, n)
        prepared = self.prepare(ids, n)
        eos = set(eos_ids or [])
        with torch.inference_mode(False), torch.no_grad():
            torch.cuda.synchronize()
            start = time.perf_counter()
            self.proposer.reset()
            self.visible[:, :ids.shape[1]].copy_(ids)
            anchor = self.verifier.prefill(ids)
            self.visible[:, ids.shape[1]:ids.shape[1] + 1].copy_(anchor)
            tokens = [int(anchor.item())]
            torch.cuda.synchronize()
            prefill_end = time.perf_counter()
            cycles = []
            while len(tokens) < n and (not eos or tokens[-1] not in eos):
                visible_length = ids.shape[1] + len(tokens)
                top_ids, logps = self.proposer.propose_lattice(self.visible[:, :visible_length], self.draft_tokens)
                builder = best_first_tree if self.tree_policy == "best-first" else trunk_best_first_tree
                tree = builder(tokens[-1], top_ids.detach().cpu(),
                    logps.detach().cpu(), node_budget=self.node_budget)
                predicted = self.verifier.verify_tree(tree["tokens"], tree["depths"],
                    tree["ancestors"], self.verifier.current_position)
                predictions = predicted.reshape(-1).cpu().tolist()
                path, emitted = self.accept_path(tree, predictions, n - len(tokens), eos)
                if not emitted or len(path) != len(emitted):
                    raise RuntimeError("Rooted verification failed to advance")
                committed = len(emitted)
                # The source predictions remain valid until the next replay.
                selected = torch.tensor(path, device=model_device(self.model), dtype=torch.long)
                self.visible[:, visible_length:visible_length + committed].copy_(
                    predicted.index_select(0, selected).view(1, -1))
                tokens.extend(emitted)
                self.verifier.commit_path(path)
                if int(self.verifier.anchor.item()) != tokens[-1]:
                    raise RuntimeError("Compacted target cache/anchor invariant failed")
                cycles.append({"node_count": tree["tokens"].numel(), "committed_tokens": committed,
                               "matched_draft_tokens": committed - 1,
                               "accepted_node_indices": path})
            torch.cuda.synchronize()
            end = time.perf_counter()
        self.verifier.assert_target_unchanged()
        self.proposer.assert_draft_unchanged()
        decode_s = end - prefill_end
        self.last_cycles = cycles
        return tokens, {"prefill_s": prefill_end - start, "decode_s": decode_s,
            "total_s": end - start, "decode_tokens_s": (len(tokens) - 1) / decode_s if len(tokens) > 1 else 0.0,
            "generated_tokens": len(tokens), "cycles": len(cycles),
            "mean_committed_tokens_per_cycle": (len(tokens) - 1) / len(cycles) if cycles else 0.0,
            "verified_nodes": sum(c["node_count"] for c in cycles),
            "excluded_setup_s": prepared["setup_s"], "cumulative_setup_s": self.setup_s,
            "stop_reason": "eos" if tokens[-1] in eos else "max_new_tokens" if eos else "fixed_budget",
            "all_output_tokens_target_verified": True,
            "decode_includes_drafting_tree_building_verification_compaction_and_cpu_transfers": True,
            "variant": self.metadata["variant"]}


def model_device(model):
    return model.get_input_embeddings().weight.device


def make_quality_runner(model, tokenizer, args):
    from numerical3b_runner import NumericalGenerationAdapter
    settings = {name: getattr(args, name) for name in ("node_budget", "draft_tokens", "top_k",
        "compile_static", "compile_draft", "cuda_graph", "share_pool", "warmups",
        "validate_topology", "commit_engine", "tree_policy") if hasattr(args, name)}
    runner = Numerical3BTreeRunner(model, tokenizer, args.draft_path,
        max_len=args.max_len, **settings)
    values = model.generation_config.eos_token_id
    eos = [values] if isinstance(values, int) else list(values or [])
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    return NumericalGenerationAdapter(runner, tokenizer, sorted(set(eos)))
