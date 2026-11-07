"""Original full-target verification exposing HF taps2/18/33/36 (8192).

Indices2/18/33 are original residual states before those decoder layers;
index36 is the original final normalized state, already computed for the head.
History contains freshly verified COMMITTED input features only and excludes
exactly the current unprocessed anchor. No auxiliary feature reaches target.
"""
from __future__ import annotations

import time

import torch
from transformers.cache_utils import DynamicCache

from bucketed3b_tree_verifier import Bucketed3BTreeVerifier


class Feature4BTreeVerifier(Bucketed3BTreeVerifier):
    feature_indices = (2, 18, 33, 36)

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        with torch.inference_mode(False), torch.no_grad():
            self.committed_feature_history = torch.empty((1, self.max_len, 8192),
                device=self.device, dtype=torch.bfloat16)
        self.last_node_features = None
        self.metadata = {**self.metadata,
            "variant": "original_weight_relaxed_fp_bucketed_tree_with_four_feature_taps",
            "feature_indices_hf_tuple": list(self.feature_indices),
            "feature_sources": {"2":"pre_decoder2_residual","18":"pre_decoder18_residual",
                "33":"pre_decoder33_residual","36":"original_final_normalized_state"},
            "feature_width": 8192,
            "features_are_original_target_outputs": True,
            "auxiliary_features_committed_to_target": False,
            "feature_history_excludes_current_unprocessed_anchor": True,
            "extra_output_flag": "output_hidden_states=True",
            "new_quality_or_output_equivalence_claim": False}

    def _make_bucket_forward(self, view):
        model, indices = self.model, self.feature_indices
        def forward(input_ids, physical_positions, rotary_positions, mask):
            output = model(input_ids=input_ids, attention_mask=mask,
                position_ids=rotary_positions.view(1, -1), cache_position=physical_positions,
                past_key_values=view, use_cache=True, logits_to_keep=input_ids.shape[1],
                return_dict=True, output_attentions=False, output_hidden_states=True)
            logits = output.logits.squeeze(0)
            features = torch.cat(tuple(output.hidden_states[index] for index in indices), dim=-1).squeeze(0)
            return logits, logits.argmax(-1), features
        return forward

    def _tree_step(self, entry):
        torch.add(entry.base, entry.offsets, out=entry.physical_positions)
        torch.add(entry.base, entry.depths, out=entry.rotary_positions)
        torch.lt(entry.key_positions.expand_as(entry.mask), entry.base.view(1, 1, 1, 1), out=entry.mask)
        entry.mask.index_copy_(3, entry.physical_positions, entry.ancestors.view(1, 1, *entry.ancestors.shape))
        entry.logits, entry.next_ids, entry.features = entry.forward_fn(
            entry.input_ids, entry.physical_positions, entry.rotary_positions, entry.mask)
        return entry.next_ids

    def reset_for_new_request(self):
        report = super().reset_for_new_request()
        self.last_node_features = None
        return report

    @torch.no_grad()
    def prefill(self, ids):
        self._check_ids(ids)
        length = ids.shape[1]
        if length >= self.max_len:
            raise ValueError("Prompt must leave room for an unprocessed target input")
        self.assert_target_unchanged(); self.assert_storage_consistent()
        torch.cuda.synchronize(self.device)
        began = time.perf_counter()
        with torch.inference_mode(False), torch.no_grad():
            output = self.model(input_ids=ids, attention_mask=None,
                past_key_values=DynamicCache(config=self.model.config),
                use_cache=True, logits_to_keep=1, return_dict=True,
                output_attentions=False, output_hidden_states=True)
            self.cache.reset()
            if len(output.past_key_values.layers) != len(self.cache.layers):
                raise RuntimeError("Original target prefill/cache layer count mismatch")
            for dst, src in zip(self.cache.layers, output.past_key_values.layers):
                dst.keys[:, :, :length].copy_(src.keys)
                dst.values[:, :, :length].copy_(src.values)
            features = torch.cat(tuple(output.hidden_states[index] for index in self.feature_indices), dim=-1)
            if features.shape != (1, length, 8192) or features.dtype != torch.bfloat16:
                raise RuntimeError("Original target feature-tap geometry mismatch")
            self.committed_feature_history[:, :length].copy_(features)
            self.last_logits = output.logits[:, -1, :]
            self._anchor_buffer.copy_(self.last_logits.argmax(-1, keepdim=True))
            self.anchor, self.last_next_ids = self._anchor_buffer, self._anchor_buffer.reshape(-1)
        self.current_position, self.pending = length, None
        self._pending_entry, self._pending_topology = None, None
        self.verify_calls, self.committed_input_tokens = 0, 0
        self.commit_calls, self.compacted_input_tokens = 0, 0
        self.last_depths, self.last_ancestors, self.last_input_ids = None, None, None
        self.last_bucket, self.last_entry_key, self.last_node_features = None, None, None
        self.last_report = {"operation": "original_hf_prefill_with_feature_taps",
            "prompt_tokens": length, "cache_excludes_unprocessed_anchor": True,
            "features_hf_indices": list(self.feature_indices)}
        torch.cuda.synchronize(self.device)
        self.prefill_s = time.perf_counter() - began
        self.assert_storage_consistent()
        return self.anchor

    @torch.no_grad()
    def verify_tree(self, input_ids, depths, ancestors, basep=None):
        ids = super().verify_tree(input_ids, depths, ancestors, basep)
        self.last_node_features = self._pending_entry.features
        if self.last_node_features.shape != (input_ids.shape[1], 8192):
            raise RuntimeError("Verified node features have an unexpected shape")
        return ids

    @torch.no_grad()
    def commit_path(self, node_indices):
        if self.pending is None:
            raise RuntimeError("No pending target tree features")
        if not isinstance(node_indices, torch.Tensor):
            node_indices = list(node_indices)
        count = node_indices.numel() if isinstance(node_indices, torch.Tensor) else len(node_indices)
        base, entry = self.pending["basep"], self._pending_entry
        # Parent validation/compaction happens first: invalid paths must not
        # write feature history. Its stable path_indices are the same accepted
        # root path, and entry.features remain live until another graph replay.
        position = super().commit_path(node_indices)
        if count:
            selected = entry.features.index_select(0, self.path_indices[:count])
            self.committed_feature_history[:, base:base + count].copy_(selected.unsqueeze(0))
        return position

    def memory_report(self):
        return {**super().memory_report(), "feature_history_bytes":
            self.committed_feature_history.numel() * self.committed_feature_history.element_size(),
            "feature_indices_hf_tuple": list(self.feature_indices)}
