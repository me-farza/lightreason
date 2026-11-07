"""Separate actual-packed tree runner with timed append-only coordination.

The frozen CostAware/Calibrated runners remain untouched controls. This loop is
mechanically identical around original-target verify/accept/commit, native EOS,
visible updates, streaming and physical-shape selection. The optional private
proposal adapter replaces only repeated full-prefix coordination. Its initial
readback, feature-tail copies and required final byte audit are timed inside
the complete generation. It makes no speed or quality claim before GPU checks.
"""
from __future__ import annotations

import time

from append_only_pard2_runtime3b_v1 import AppendOnlyPARD2ProposalAdapter
from calibrated_tree_policy3b_v1 import CalibratedTreePolicy3BRunner


def feature_coordination_counts(prompt_length, committed_counts, width=8192, element_bytes=2):
    """Logical bytes for the SAME accepted path; not measured DRAM traffic.

    The frozen proposer snapshots the full p-row prefix at each draft and
    compares the previous proposal's verified prefix. The owned adapter copies
    each prompt/committed feature row once and compares the full prefix once at
    finalization. Include final commit rows even though the frozen proposer
    need not consume them: this is a conservative copy-cost comparison.
    """
    if type(prompt_length) is not int or prompt_length < 1:
        raise ValueError("Use the actual positive tokenized prompt length")
    counts = tuple(committed_counts)
    if any(type(value) is not int or value < 1 for value in counts):
        raise ValueError("Use actual positive original-target committed path lengths")
    position, previous, full_copy_rows, comparison_rows = prompt_length, None, 0, 0
    for count in counts:
        full_copy_rows += position
        comparison_rows += 0 if previous is None else previous
        previous = position
        position += count
    row_bytes = width * element_bytes
    return {
        "scope": "logical feature-coordinate bytes for this exact accepted path; not hardware DRAM traffic",
        "feature_row_bytes": row_bytes,
        "frozen_full_snapshot_rows_same_path": full_copy_rows,
        "frozen_full_snapshot_copy_logical_bytes_same_path": full_copy_rows * row_bytes,
        "frozen_prefix_comparison_rows_same_path": comparison_rows,
        "frozen_prefix_comparison_read_logical_bytes_same_path": 2 * comparison_rows * row_bytes,
        "append_only_snapshot_rows": position,
        "append_only_snapshot_copy_logical_bytes": position * row_bytes,
        "append_only_final_full_comparison_read_logical_bytes": 2 * position * row_bytes,
        "initial_prompt_and_final_commit_feature_rows_included": True,
    }


class AppendOnlyTree3BRunner(CalibratedTreePolicy3BRunner):
    """Experimental private-owner coordination, default original ranking."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configure_append_only()

    @classmethod
    def from_control(cls, control, **kwargs):
        result = super().from_control(control, **kwargs)
        result._configure_append_only()
        return result

    def _configure_append_only(self):
        self.proposal_adapter = AppendOnlyPARD2ProposalAdapter(self)
        self.metadata = {**self.metadata,
            "variant": "original_BF16_target_actual_packed_tree_append_only_coordination_v1",
            "append_only_coordination": dict(self.proposal_adapter.metadata),
            "all_append_only_readback_copy_and_final_audit_costs_inside_generation": True,
            "original_target_verify_accept_commit_and_EOS_unchanged": True,
            "separate_experimental_private_buffer_ownership_contract": True,
            "performance_or_quality_improvement_established": False}

    def run(self, ids, n, eos_ids=None, stream_callback=None):
        import torch
        self._check(ids, n)
        prepared = self.prepare(ids, n)
        eos = set(eos_ids or [])
        adapter = self.proposal_adapter
        with torch.inference_mode(False), torch.no_grad():
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                self.proposer.reset()
                self.visible[:, :ids.shape[1]].copy_(ids)
                with self._stage("prefill"):
                    anchor = self.verifier.prefill(ids)
                    self.visible[:, ids.shape[1]:ids.shape[1] + 1].copy_(anchor)
                    tokens = [int(anchor.item())]
                    # One initial device->host prompt+anchor observation. This
                    # and the initial feature snapshot remain in prefill time.
                    adapter.begin_request_from_owned_buffers(tokens[0], eos)
                if stream_callback is not None:
                    with self._stage("streaming"):
                        stream_callback(tokens.copy())
                torch.cuda.synchronize()
                prefill_end = time.perf_counter()
                cycles, ranking = [], []
                while len(tokens) < n and (not eos or tokens[-1] not in eos):
                    cycle_start = time.perf_counter()
                    visible_length = ids.shape[1] + len(tokens)
                    with self._stage("drafting"):
                        top_ids, logps = adapter.propose_lattice(self.visible[:, :visible_length], self.draft_tokens)
                    with self._stage("lattice_transfer"):
                        cpu_ids, cpu_logps = top_ids.detach().cpu(), logps.detach().cpu()
                    with self._stage("selection_tree"):
                        tree, cost_diagnostics = self._make_tree(tokens[-1], cpu_ids, cpu_logps,
                            self.max_len - self.verifier.current_position)
                    with self._stage("verification"):
                        predicted = self.verifier.verify_tree(tree["tokens"], tree["depths"],
                                                             tree["ancestors"], self.verifier.current_position)
                    with self._stage("prediction_transfer"):
                        predictions = predicted.reshape(-1).cpu().tolist()
                    with self._stage("acceptance"):
                        path, emitted = self.accept_path(tree, predictions, n - len(tokens), eos)
                    if not emitted or len(path) != len(emitted):
                        raise RuntimeError("Rooted original-target verification failed to advance")
                    committed = len(emitted)
                    with self._stage("visible_update"):
                        selected = torch.tensor(path, device=self.model.device, dtype=torch.long)
                        self.visible[:, visible_length:visible_length + committed].copy_(
                            predicted.index_select(0, selected).view(1, -1))
                        tokens.extend(emitted)
                    with self._stage("cache_commit"):
                        self.verifier.commit_path(path)
                        adapter.observe_target_commit(emitted)
                    with self._stage("anchor_sync"):
                        if int(self.verifier.anchor.item()) != tokens[-1]:
                            raise RuntimeError("Compacted target cache/anchor invariant failed")
                    if stream_callback is not None:
                        with self._stage("streaming"):
                            stream_callback(tokens.copy())
                    cycle = {"node_count": tree["node_count"], "committed_tokens": committed,
                             "matched_draft_tokens": committed - 1, "accepted_node_indices": path,
                             "cycle_wall_s": time.perf_counter() - cycle_start,
                             "capacity_fallback": tree["capacity_fallback"]}
                    if cost_diagnostics is not None:
                        cycle["cost_policy_candidates"] = cost_diagnostics
                    if self.trace_ranking:
                        accepted = set(path[1:])
                        ranking.append({"cycle": len(cycles), "packed_rows": tree["node_count"],
                            "depths": tree["depths"].tolist(), "log_prefix_masses": tree["log_scores"].tolist(),
                            "accepted_nonroot_nodes": [i in accepted for i in range(tree["node_count"])],
                            "actual_committed_tokens": committed,
                            "output_budget_or_EOS_can_censor_acceptance": len(tokens) >= n or tokens[-1] in eos})
                    cycles.append(cycle)
                with self._stage("final_audit"):
                    ownership_audit = adapter.finish_request()
                torch.cuda.synchronize()
                end = time.perf_counter()
            except Exception:
                if adapter.ledger.active:
                    adapter.abort_request()
                raise
        self.verifier.assert_target_unchanged()
        self.proposer.assert_draft_unchanged()
        decode_s = end - prefill_end
        self.last_cycles, self.last_ranking_trace = cycles, ranking
        for row in self.last_ranking_trace:
            row.update(ranking_policy=self.ranking_policy,
                       ranking_temperature=self.ranking_temperature,
                       tree_policy=self.tree_policy,
                       ranking_score_label=self.metadata["ranking_score_label"])
        feature_bytes = feature_coordination_counts(ids.shape[1], [cycle["committed_tokens"] for cycle in cycles])
        if feature_bytes["append_only_snapshot_rows"] != ownership_audit["feature_rows_copied_total"]:
            raise RuntimeError("Observed incremental feature copy count differs from exact accepted-path accounting")
        return tokens, {"prefill_s": prefill_end - start, "decode_s": decode_s,
            "total_s": end - start, "decode_tokens_s": (len(tokens) - 1) / decode_s if len(tokens) > 1 else 0.0,
            "generated_tokens": len(tokens), "cycles": len(cycles),
            "mean_committed_tokens_per_cycle": (len(tokens) - 1) / len(cycles) if cycles else 0.0,
            "mean_cycle_ms": 1000.0 * decode_s / len(cycles) if cycles else 0.0,
            "verified_nodes": sum(cycle["node_count"] for cycle in cycles),
            "actual_packed_rows_used": sorted(set(cycle["node_count"] for cycle in cycles)),
            "capacity_fallback_cycles": sum(cycle["capacity_fallback"] for cycle in cycles),
            "excluded_setup_s": prepared["setup_s"], "cumulative_setup_s": self.setup_s,
            "stop_reason": "eos" if tokens[-1] in eos else "max_new_tokens" if eos else "fixed_budget",
            "all_output_tokens_target_verified": True,
            "streaming_callback_enabled": stream_callback is not None,
            "decode_includes_drafting_tree_building_verification_compaction_and_cpu_transfers": True,
            "variant": self.metadata["variant"],
            "ranking_policy": self.ranking_policy,
            "ranking_temperature": self.ranking_temperature,
            "tree_policy": self.tree_policy,
            "ranking_score_label": self.metadata["ranking_score_label"],
            "final_acceptance_by_original_target": True,
            "append_only_ownership_audit": ownership_audit,
            "append_only_begin_commit_final_audit_inside_generation_time": True,
            "feature_coordination_logical_bytes": feature_bytes}

