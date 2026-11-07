"""Experimental private-owner append-only coordination for the frozen PARD2 query.

This adapter changes no target or auxiliary weights, forward function, attention
mask, draft ranking, or target acceptance rule. It reuses the exact prepared
private auxiliary queries. Only the CPU/GPU coordination around those queries
changes: an original-target host ledger replaces repeated full visible-ID
readback; verified-feature snapshots copy only freshly committed target rows.

Contract: one synchronous runner privately owns visible IDs, original-target
features, and both caches for the duration of a request. No caller, streaming
observer, thread, or external consumer may mutate those buffers while active.
The old arbitrary-mutable-input proposer remains the frozen control. This is
NOT a general replacement with identical per-call mutation detection. A final
full prefix audit detects surviving old-prefix changes; mutation followed by
restoration between checks cannot be detected under this ownership contract.

All begin/commit/final-audit costs must remain inside complete generation time.
Only the original target verifies/commits tokens and persistent target K/V.
"""
from __future__ import annotations

import time

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


class AppendOnlyLedger:
    """CPU state machine; token IDs originate from original-target observations."""

    def __init__(self, vocab_size):
        _require(type(vocab_size) is int and vocab_size > 0, "Use a positive original target vocabulary")
        self.vocab_size = vocab_size
        self.active = False
        self.phase = "closed"
        self.request_number = 0

    def _tokens(self, values):
        values = list(values)
        _require(bool(values) and all(type(value) is int and 0 <= value < self.vocab_size for value in values),
                 "Only actual original target token IDs are admitted")
        return values

    def begin(self, prompt_ids, anchor, eos_ids=()):
        _require(not self.active, "Finish or abort the prior request before another prefill")
        prompt = self._tokens(prompt_ids)
        anchor = self._tokens([anchor])[0]
        eos = set(eos_ids or ())
        _require(all(type(value) is int and 0 <= value < self.vocab_size for value in eos), "Invalid native EOS IDs")
        self.visible = prompt + [anchor]
        self.position = len(prompt)
        self.eos_ids = eos
        self.terminated = anchor in eos
        self.phase = "ready"
        self.active = True
        self.proposals = self.commits = 0
        self.last_B = None
        self.request_number += 1

    def ready(self, position, known_visible_ids=None):
        _require(self.active and self.phase == "ready", "Missing request or an uncommitted proposal is still active")
        _require(type(position) is int and position == self.position and len(self.visible) == position + 1,
                 "Original target cache must exclude exactly the current ledger anchor")
        _require(not self.terminated, "Native target EOS already ended this request")
        if known_visible_ids is not None:
            _require(tuple(known_visible_ids) == tuple(self.visible), "Host visible prefix differs from original-target ledger")

    def proposed(self, B):
        _require(self.active and self.phase == "ready" and not self.terminated, "Cannot replay an active/finished proposal")
        _require(type(B) is int and 1 <= B <= 32, "Use the existing public supported draft horizon1..32")
        self.last_B = B
        self.phase = "awaiting_original_target_commit"
        self.proposals += 1

    def validate_commit(self, emitted):
        _require(self.active and self.phase == "awaiting_original_target_commit", "No original-target proposal is awaiting commitment")
        values = self._tokens(emitted)
        _require(1 <= len(values) <= self.last_B + 1, "A target path may commit at most B+1 predictions")
        _require(not any(value in self.eos_ids for value in values[:-1]), "A commit continued after native target EOS")
        return values

    def commit(self, emitted):
        values = self.validate_commit(emitted)
        self.visible.extend(values)
        self.position += len(values)
        self.terminated = values[-1] in self.eos_ids
        self.commits += 1
        self.phase = "ready"

    def finish(self):
        _require(self.active and self.phase == "ready", "Cannot finish with an uncommitted proposal")
        self.active = False
        self.phase = "closed"

    def abort(self):
        self.active = False
        self.phase = "aborted"


def _identity(tensor):
    return (id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            str(tensor.dtype), str(tensor.device))


class AppendOnlyPARD2ProposalAdapter:
    """Narrow optional adapter for an actual owned original-target tree runner.

    Usage after original target prefill and writing its actual anchor:
      adapter.begin_request(prompt_ids_host, anchor_int, eos_ids)
      ids, logps = adapter.propose_lattice(visible_view, B)
      # Existing original target verification/acceptance/commit and visible write.
      adapter.observe_target_commit(emitted_ids)
      # At native EOS/cap, BEFORE the generation timer ends:
      audit = adapter.finish_request()

    The caller's first prompt GPU->CPU read, when necessary, is also timed. An
    authentic CPU tokenizer result may instead be supplied by the benchmark.
    """

    def __init__(self, runner):
        _require(torch is not None, "Use the pinned CUDA environment for this adapter")
        self.runner = runner
        self.owner = runner.proposer.conditioned
        self.verifier = runner.verifier
        self.visible = runner.visible
        self.history = self.verifier.committed_feature_history
        self.snapshot = self.owner._verified_feature_snapshot
        self.ledger = AppendOnlyLedger(self.owner.target_vocab_size)
        self._buffer_state = tuple(_identity(value) for value in (self.visible, self.history, self.snapshot))
        _require(self.history.dtype == self.snapshot.dtype == torch.bfloat16 and self.visible.dtype == torch.long,
                 "Original visible IDs must be long and verified feature buffers BF16")
        _require(self.history.shape[0] == self.snapshot.shape[0] == 1 and self.history.shape[2] == self.snapshot.shape[2] == 8192,
                 "Use the existing original-target Feature4 geometry")
        _require(self.history.shape[1] >= runner.max_len and self.snapshot.shape[1] >= runner.max_len,
                 "Feature prefix capacity is smaller than the original runner")
        self.metadata = {
            "variant": "private_owned_append_only_PARD2_coordination_v1",
            "target_or_auxiliary_forward_replaced": False,
            "target_or_auxiliary_weight_values_changed": False,
            "same_prepared_private_query_graphs": True,
            "same_true_and_mask_feature_alignment": True,
            "external_buffer_mutation_during_request_permitted": False,
            "per_cycle_arbitrary_input_mutation_detection_preserved": False,
            "final_full_visible_and_feature_prefix_audit_required": True,
            "all_begin_commit_and_finish_audit_costs_must_be_timed": True,
            "current_draft_horizon_changes_checkpoint_support": False,
        }

    @property
    def visible_ids_host(self):
        """Read-only value copy for diagnostics, never auxiliary-generated IDs."""
        return tuple(self.ledger.visible)

    def _buffers(self):
        _require(self.runner.proposer.conditioned is self.owner and self.runner.verifier is self.verifier,
                 "Target/auxiliary request owner was replaced")
        _require(self.runner.visible is self.visible and self.verifier.committed_feature_history is self.history and
                 self.owner._verified_feature_snapshot is self.snapshot,
                 "Owned visible/verified-feature buffers were replaced")
        _require(tuple(_identity(value) for value in (self.visible, self.history, self.snapshot)) == self._buffer_state,
                 "Owned visible/feature storage layout changed")

    def _counts(self):
        return self.verifier.verify_calls, self.verifier.commit_calls

    def _feature_tail(self, start, end):
        _require(type(start) is int and type(end) is int and 0 <= start <= end <= self.runner.max_len,
                 "New verified feature range exceeds original target prefix")
        if end > start:
            self.snapshot[:, start:end].copy_(self.history[:, start:end])
            self.copied_feature_rows += end - start
        self.owner._verified_feature_count = end
        self.snapshot_count = end

    def begin_request(self, prompt_ids_host, anchor_int, eos_ids=None):
        return self._begin_request(prompt_ids_host, anchor_int, eos_ids, visible_readback=None)

    def begin_request_from_owned_buffers(self, anchor_int, eos_ids=None):
        """One timed readback of the exact private prompt+target-anchor buffer.

        This avoids reading ids and visible separately when the caller has only
        CUDA tokenized inputs. It never accepts auxiliary-produced host IDs.
        """
        self._buffers()
        _require(not self.ledger.active, "Finish/abort the current private request before a new one")
        length = self.verifier.current_position + 1
        actual = self.visible[:, :length].detach().cpu().reshape(-1).tolist()
        return self._begin_request(actual[:-1], anchor_int, eos_ids, visible_readback=actual)

    def _begin_request(self, prompt_ids_host, anchor_int, eos_ids, visible_readback):
        self._buffers()
        _require(not self.ledger.active, "Finish/abort the current private request before a new one")
        _require(not self.owner.cached_visible_ids and self.owner._verified_feature_count == 0,
                 "Reset the auxiliary logical state before genuine original-target prefill")
        prompt = list(prompt_ids_host)
        _require(self.verifier.pending is None and self.verifier.current_position == len(prompt),
                 "Begin only after the original target genuinely prefills this prompt")
        _require(len(prompt) + 1 <= self.visible.shape[1], "Prompt anchor exceeds visible capacity")
        self.owner.assert_draft_unchanged()
        self.verifier.assert_target_unchanged()
        self.ledger.begin(prompt, anchor_int, eos_ids)
        self.copied_feature_rows = 0
        self.snapshot_count = 0
        self.last_known_counts = self._counts()
        self.last_cached_visible = ()
        self.prepared_graphs = self.owner.graphs
        self.prepared_entries = {key: (id(entry), id(entry["graph"])) for key, entry in self.owner.graphs.items()}
        self.prepared_output_ids = {B: (id(value), value.data_ptr()) for B, value in self.owner.outputs.items()}
        self.audit = None
        try:
            # One initial whole-prefix read is explicit and inside generation.
            actual = (self.visible[:, :len(prompt) + 1].detach().cpu().reshape(-1).tolist()
                      if visible_readback is None else visible_readback)
            _require(actual == self.ledger.visible, "Visible prompt/anchor bytes differ from the original-target host ledger")
            _require(int(self.verifier.anchor.item()) == anchor_int, "Original target anchor differs from the observed host anchor")
            self._feature_tail(0, len(prompt))
        except Exception:
            self.abort_request()
            raise

    def _entry(self, B, width, bucket):
        key = B, width, bucket
        _require(self.owner.graphs is self.prepared_graphs and key in self.prepared_entries and key in self.owner.graphs,
                 "An unprepared/replaced auxiliary graph family entered an active request")
        entry = self.owner.graphs[key]
        _require((id(entry), id(entry["graph"])) == self.prepared_entries[key],
                 "A prepared auxiliary graph entry was replaced during the request")
        output = self.owner.outputs[B]
        _require((id(output), output.data_ptr()) == self.prepared_output_ids[B], "Prepared auxiliary output storage changed")
        return entry

    def propose_lattice(self, visible_ids, num_draft_tokens, known_visible_ids=None):
        self._buffers()
        p, B = self.verifier.current_position, num_draft_tokens
        self.ledger.ready(p, known_visible_ids)
        _require(self.verifier.pending is None and self._counts() == self.last_known_counts,
                 "Original target verification/commit state changed outside the private controller")
        _require(self.owner.cached_visible_ids == self.last_cached_visible,
                 "Auxiliary cache reset/changed outside this request; a silent full-prefill fallback is forbidden")
        _require(self.snapshot_count == self.owner._verified_feature_count == p,
                 "Verified feature tail was not observed after original-target commitment")
        _require(type(B) is int and 1 <= B <= 32, "Use the existing supported draft horizon1..32")
        length = p + 1
        _require(visible_ids.dtype == torch.long and tuple(visible_ids.shape) == (1, length) and
                 visible_ids.device == self.visible.device and visible_ids.data_ptr() == self.visible.data_ptr() and
                 tuple(visible_ids.stride()) == tuple(self.visible.stride()),
                 "Only the privately owned complete visible prefix can enter the fast path")
        self.owner.assert_draft_unchanged()
        bucket = self.owner.select_bucket(length + B - 1)
        # Existing B..2B public preparation is mandatory. No graph capture or
        # stale prefix rebuilding is admitted while real values are active.
        for width in range(B, 2 * B + 1):
            key = B, width, bucket
            _require(key in self.prepared_entries and key in self.owner.graphs,
                     "An unprepared/replaced auxiliary graph family entered an active request")
        common = min(len(self.last_cached_visible), length - 1)
        used_prefill = False
        try:
            if not self.last_cached_visible:
                self.owner._prefill_conditioned_history(visible_ids, self.history[:, :p], p)
                common = length - 1
                used_prefill = True
            count = length - common
            _require(1 <= count <= B + 1, "Append-only genuine suffix is outside the prepared private query range")
            entry = self._entry(B, count + B - 1, bucket)
            entry["input_ids"][:, :count].copy_(visible_ids[:, common:])
            aligned = entry["feature_inputs"]
            if common == 0:
                aligned[:, :1].zero_()
                if count > 1:
                    aligned[:, 1:count].copy_(self.history[:, :p])
            else:
                aligned[:, :count].copy_(self.history[:, common - 1:p])
            if B > 1:
                aligned[:, count:].copy_(self.history[:, p - 1:p].expand(1, B - 1, 8192))
            self.owner.gpu_base.fill_(common)
            entry["graph"].replay()
            visible_host = tuple(self.ledger.visible)
            self.owner.cached_visible_ids = self.last_cached_visible = visible_host
            self.owner.assert_draft_unchanged()
            self.owner.last_report = {
                "visible_tokens": length, "verified_feature_prefix": p, "num_draft_tokens": B,
                "reused_true_prefix_tokens": common, "processed_true_tokens": count,
                "mask_tokens": B - 1, "last_verified_mask_feature_row": p - 1,
                "used_conditioned_full_prefill": used_prefill, "graph_key": [B, count + B - 1, bucket],
                "attention_bucket": bucket,
                "all_conditioning_features_already_verified": True,
                "auxiliary_never_final_authority": True,
                "prefix_validation": "private_original_target_append_ledger_plus_required_final_full_audit",
                "full_feature_snapshot_copied_this_proposal": False,
                "full_visible_IDs_read_back_this_proposal": False,
            }
            self.ledger.proposed(B)
            ids, logps = self.owner.lattice_ids[B], self.owner.lattice_logps[B]
            return (ids.clone(), logps.clone()) if self.owner.copy_output else (ids, logps)
        except Exception:
            self.abort_request()
            raise

    def observe_target_commit(self, emitted_ids):
        """Call only after the original verifier's accepted path is committed."""
        self._buffers()
        values = self.ledger.validate_commit(emitted_ids)
        oldp, newp = self.ledger.position, self.ledger.position + len(values)
        expected_counts = tuple(value + 1 for value in self.last_known_counts)
        _require(self.verifier.pending is None and self._counts() == expected_counts and
                 self.verifier.current_position == newp,
                 "Only one actual original-target verify/commit may authorize this exact append")
        _require(newp + 1 <= self.visible.shape[1], "Committed target anchor exceeds visible capacity")
        self.verifier.assert_target_unchanged()
        # The range includes the old root/anchor input and excludes the newly
        # emitted unprocessed anchor, exactly like the original target cache.
        self._feature_tail(oldp, newp)
        self.ledger.commit(values)
        self.last_known_counts = expected_counts

    def finish_request(self):
        """Required full audit, INCLUDED before complete generation timing ends."""
        began = time.perf_counter()
        self._buffers()
        _require(self.ledger.active and self.ledger.phase == "ready", "Cannot audit an incomplete proposal")
        p = self.ledger.position
        _require(self.verifier.pending is None and self.verifier.current_position == p and
                 self._counts() == self.last_known_counts and self.snapshot_count == p,
                 "Final target prefix/commit accounting differs from the owned ledger")
        actual = self.visible[:, :p + 1].detach().cpu().reshape(-1).tolist()
        _require(actual == self.ledger.visible, "Final visible token bytes differ from original-target observations")
        _require(torch.equal(self.history[:, :p].view(torch.int16), self.snapshot[:, :p].view(torch.int16)),
                 "A previously committed original-target feature row was mutated")
        _require(int(self.verifier.anchor.item()) == self.ledger.visible[-1], "Final original target anchor differs from accepted output")
        self.owner.assert_draft_unchanged()
        self.verifier.assert_target_unchanged()
        _require(self.owner.graphs is self.prepared_graphs and
                 {key: (id(entry), id(entry["graph"])) for key, entry in self.owner.graphs.items()} == self.prepared_entries,
                 "Auxiliary graph bank changed during the owned request")
        self.ledger.finish()
        self.audit = {
            "passed": True, "all_visible_token_IDs_match_original_target_host_ledger": True,
            "all_committed_original_target_feature_rows_match_snapshot": True,
            "feature_prefix_comparison_is_BF16_raw_bits": True,
            "original_target_anchor_and_verify_commit_counts_passed": True,
            "original_target_auxiliary_guards_passed": True,
            "no_new_or_replaced_auxiliary_graphs_during_request": True,
            "feature_rows_copied_total": self.copied_feature_rows,
            "committed_feature_prefix_rows": p,
            "feature_rows_copied_exactly_once": self.copied_feature_rows == p,
            "final_full_audit_s": time.perf_counter() - began,
            "final_full_audit_excluded_from_generation_time": False,
            "arbitrary_external_mutation_during_request_supported": False,
            "transient_mutation_then_restoration_detectable": False,
        }
        return dict(self.audit)

    def abort_request(self):
        self.ledger.abort()
        self.owner.reset()
        self.audit = {"passed": False, "aborted": True}

