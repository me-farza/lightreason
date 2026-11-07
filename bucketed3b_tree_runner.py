"""Separately labeled original-weight tree runner with attention-length buckets.

Only static-cache attention views and graph selection change. The inherited
target acceptance, native EOS, prefix compaction and output controller remain
the same. All reachable graph families use only the public prompt and budget.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import torch

from bucketed3b_tree_verifier import BucketedNumericalTreeVerifier
from bucketed_pard_proposer import BucketedPARDLatticeProposer
from numerical3b_tree_runner import Numerical3BTreeRunner
from numerical3b_runner import NumericalGenerationAdapter


class BucketedNumerical3BTreeRunner(Numerical3BTreeRunner):
    def __init__(self, model, tokenizer, draft_path, *,
                 attention_buckets=(256, 512, 1024, 2048), **kwargs):
        compile_draft = bool(kwargs.pop("compile_draft", False))
        super().__init__(model, tokenizer, draft_path, compile_draft=False, **kwargs)
        attention_buckets = tuple(sorted({int(v) for v in attention_buckets
            if 1 <= int(v) <= self.max_len} | {self.max_len}))
        previous = self.proposer
        self.verifier = BucketedNumericalTreeVerifier(model, max_len=self.max_len,
            node_budget=self.node_budget, buckets=attention_buckets,
            cuda_graph=kwargs.get("cuda_graph", True),
            compile_static=kwargs.get("compile_static", False),
            share_pool=kwargs.get("share_pool", True), warmups=kwargs.get("warmups", 3),
            commit_engine=kwargs.get("commit_engine", "triton"),
            validate_topology=kwargs.get("validate_topology", True))
        self.proposer = BucketedPARDLatticeProposer(previous.base,
            max_len=self.max_len, attention_buckets=attention_buckets,
            top_k=self.top_k, compile_model=compile_draft,
            share_pool=kwargs.get("share_pool", True), warmups=kwargs.get("warmups", 3))
        if self.proposer.model is not previous.model or self.verifier.model is not model:
            raise RuntimeError("Cache-view construction replaced a trained model")
        self.metadata = {**self.metadata,
            "variant": "original_target_weights_relaxed_fp_order_bucketed_tree",
            "target_verifier": self.verifier.metadata, "proposer": self.proposer.metadata,
            "attention_buckets": list(self.verifier.attention_buckets),
            "compile_draft": compile_draft,
            "full_cache_storage_capacity": self.max_len,
            "target_controller_inherited_unchanged": True,
            "runtime_graph_capture_permitted": False,
            "preparation_uses_only_public_prompt_length_and_output_budget": True,
            "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    @staticmethod
    def _reachable(owner, first_end, last_end):
        first, last = owner.select_bucket(first_end), owner.select_bucket(last_end)
        return tuple(v for v in owner.attention_buckets if first <= v <= last)

    def prepare(self, ids, n):
        self._check(ids, n)
        if self.verifier.pending is not None:
            raise RuntimeError("Finish the preceding target verification before a new request")
        # The inherited run begins a fresh request after prepare. Invalidate
        # only logical metadata before any synthetic graph writes; genuine HF
        # prefill subsequently resets/copies the entire visible target prefix.
        self.verifier.reset_for_new_request()
        self.proposer.reset()
        if n == 1:
            return {"setup_s": 0.0, "new_graphs": 0}
        length = ids.shape[1]
        target_buckets = self._reachable(self.verifier,
            length + self.node_budget, length + n + self.node_budget - 2)
        draft_buckets = self._reachable(self.proposer,
            length + self.draft_tokens, length + n + self.draft_tokens - 2)
        torch.cuda.synchronize()
        began = time.perf_counter()
        target = self.verifier.prepare(self.node_budget, buckets=target_buckets)
        draft = self.proposer.prepare(self.draft_tokens, buckets=draft_buckets)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        self.setup_s += elapsed
        self.prepared = True
        return {"setup_s": elapsed, "target": target, "draft": draft,
            "target_buckets": list(target_buckets), "draft_buckets": list(draft_buckets),
            "new_graphs": target.get("new_graphs", 0) + draft.get("new_graphs", 0)}


def make_quality_runner(model, tokenizer, args):
    names = ("node_budget", "draft_tokens", "top_k", "compile_static", "compile_draft",
        "cuda_graph", "share_pool", "warmups", "validate_topology", "commit_engine",
        "tree_policy", "attention_buckets")
    settings = {name: getattr(args, name) for name in names if hasattr(args, name)}
    runner = BucketedNumerical3BTreeRunner(model, tokenizer, args.draft_path,
        max_len=args.max_len, **settings)
    configured = model.generation_config.eos_token_id
    eos = [configured] if isinstance(configured, int) else list(configured or [])
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    return NumericalGenerationAdapter(runner, tokenizer, sorted(set(eos)))
