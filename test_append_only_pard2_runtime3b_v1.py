"""CPU ownership/state/alignment contracts; not a substitute for GPU proof."""
from __future__ import annotations

import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import append_only_pard2_runtime3b_v1 as runtime


class LedgerTests(unittest.TestCase):
    def ledger(self):
        value = runtime.AppendOnlyLedger(100)
        value.begin([1, 2, 3], 4, [99])
        return value

    def test_monotonic_target_commits_and_exact_host_prefix(self):
        value = self.ledger()
        value.ready(3, [1, 2, 3, 4])
        value.proposed(2)
        value.commit([5, 6, 7])
        value.ready(6, [1, 2, 3, 4, 5, 6, 7])
        value.proposed(2)
        value.commit([99])
        self.assertTrue(value.terminated)
        value.finish()
        self.assertEqual(value.position, 7)

    def test_no_cross_request_or_proposal_replay(self):
        value = self.ledger()
        with self.assertRaises(RuntimeError):
            value.begin([1], 2)
        value.proposed(2)
        with self.assertRaises(RuntimeError):
            value.proposed(2)
        with self.assertRaises(RuntimeError):
            value.finish()
        value.abort()
        value.begin([8, 9], 10)
        self.assertEqual(value.visible, [8, 9, 10])

    def test_eos_and_length_guards(self):
        value = self.ledger()
        value.proposed(2)
        for invalid in ([], [1, 2, 3, 4], [99, 7], [-1], [100], [True]):
            with self.assertRaises(RuntimeError):
                value.validate_commit(invalid)
        value.commit([99])
        with self.assertRaises(RuntimeError):
            value.ready(4)

    def test_mismatched_position_and_host_values(self):
        value = self.ledger()
        with self.assertRaises(RuntimeError):
            value.ready(4)
        with self.assertRaises(RuntimeError):
            value.ready(3, [1, 2, 9, 4])

    def test_initial_native_eos_needs_no_auxiliary_query(self):
        value = runtime.AppendOnlyLedger(100)
        value.begin([1, 2], 99, [99])
        self.assertTrue(value.terminated)
        with self.assertRaises(RuntimeError):
            value.proposed(2)
        value.finish()


class FakeTensor:
    """Small sparse CPU fixture implementing only the actual adapter operations."""
    def __init__(self, shape, dtype="bf16", *, storage=None, offset=0, strides=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, "cuda:fixture"
        self.storage = {} if storage is None else storage
        self.offset = offset
        if strides is None:
            strides = []
            product = 1
            for size in reversed(shape):
                strides.insert(0, product)
                product *= size
        self.strides = tuple(strides)

    def stride(self):
        return self.strides

    def data_ptr(self):
        return id(self.storage) + self.offset

    def indexes(self):
        return itertools.product(*(range(size) for size in self.shape))

    def get(self, index):
        return self.storage.get(self.offset + sum(i * s for i, s in zip(index, self.strides)), 0)

    def put(self, index, value):
        self.storage[self.offset + sum(i * s for i, s in zip(index, self.strides))] = value

    def __getitem__(self, parts):
        if not isinstance(parts, tuple):
            parts = (parts,)
        parts = parts + (slice(None),) * (len(self.shape) - len(parts))
        offset, shape, strides = self.offset, [], []
        for part, size, stride in zip(parts, self.shape, self.strides):
            if isinstance(part, int):
                offset += part * stride
            else:
                start, stop, step = part.indices(size)
                offset += start * stride
                shape.append(len(range(start, stop, step)))
                strides.append(stride * step)
        return FakeTensor(shape, self.dtype, storage=self.storage, offset=offset, strides=strides)

    def copy_(self, other):
        if self.shape != other.shape:
            raise AssertionError((self.shape, other.shape))
        for index in self.indexes():
            self.put(index, other.get(index))
        return self

    def fill_(self, value):
        for index in self.indexes():
            self.put(index, value)
        return self

    def zero_(self):
        return self.fill_(0)

    def expand(self, *shape):
        strides = tuple(0 if before == 1 and after != 1 else stride
                        for before, after, stride in zip(self.shape, shape, self.strides))
        return FakeTensor(shape, self.dtype, storage=self.storage, offset=self.offset, strides=strides)

    def clone(self):
        return FakeTensor(self.shape, self.dtype).copy_(self)

    def view(self, dtype):
        return FakeTensor(self.shape, dtype, storage=self.storage, offset=self.offset, strides=self.strides)

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, size):
        if size != -1:
            raise AssertionError(size)
        values = [self.get(index) for index in self.indexes()]
        result = FakeTensor((len(values),), self.dtype)
        for index, value in enumerate(values):
            result.put((index,), value)
        return result

    def tolist(self):
        return [self.get(index) for index in self.indexes()]

    def item(self):
        values = self.tolist()
        if len(values) != 1:
            raise AssertionError("Not scalar")
        return values[0]


class FakeTorch:
    bfloat16 = "bf16"
    long = "long"
    int16 = "int16"

    @staticmethod
    def equal(left, right):
        return left.shape == right.shape and all(left.get(index) == right.get(index) for index in left.indexes())


def fixture():
    visible = FakeTensor((1, 24), "long")
    history, snapshot = FakeTensor((1, 24, 8192)), FakeTensor((1, 24, 8192))
    for row in range(3):
        history[:, row:row + 1].fill_(row + 20)
    for index, value in enumerate([2, 5, 9, 7]):
        visible.put((0, index), value)
    anchor = FakeTensor((1, 1), "long").fill_(7)
    verifier = SimpleNamespace(committed_feature_history=history, current_position=3, pending=None,
                               verify_calls=0, commit_calls=0, anchor=anchor,
                               assert_target_unchanged=lambda: None)
    owner = SimpleNamespace(target_vocab_size=100, cached_visible_ids=(), _verified_feature_count=0,
                            _verified_feature_snapshot=snapshot, device="cuda:fixture", copy_output=False,
                            gpu_base=FakeTensor((1,), "long"), outputs={2: FakeTensor((1, 2), "long")},
                            lattice_ids={2: FakeTensor((2, 2), "long")}, lattice_logps={2: FakeTensor((2, 2))},
                            select_bucket=lambda end: 24, assert_draft_unchanged=lambda: None)
    calls = []
    owner.graphs = {(2, width, 24): {"input_ids": FakeTensor((1, width), "long"),
                    "feature_inputs": FakeTensor((1, width, 8192)),
                    "graph": SimpleNamespace(replay=lambda: calls.append("query"))} for width in (2, 3, 4)}
    owner._prefill_conditioned_history = lambda ids, features, p: calls.append(("prefill", p))
    def reset():
        owner.cached_visible_ids = ()
        owner._verified_feature_count = 0
    owner.reset = reset
    runner = SimpleNamespace(proposer=SimpleNamespace(conditioned=owner), verifier=verifier,
                             visible=visible, max_len=24)
    return runner, calls


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.object(runtime, "torch", FakeTorch)
        self.patch.start()
        self.runner, self.calls = fixture()
        self.adapter = runtime.AppendOnlyPARD2ProposalAdapter(self.runner)
        self.adapter.begin_request([2, 5, 9], 7, [99])

    def tearDown(self):
        self.patch.stop()

    def target_commit(self, emitted):
        verifier = self.runner.verifier
        p = verifier.current_position
        for index, value in enumerate(emitted, start=p + 1):
            self.runner.visible.put((0, index), value)
        verifier.committed_feature_history[:, p:p + len(emitted)].fill_(p + 30)
        verifier.verify_calls += 1
        verifier.commit_calls += 1
        verifier.current_position += len(emitted)
        verifier.anchor.fill_(emitted[-1])
        self.adapter.observe_target_commit(emitted)

    def test_same_prepared_query_alignment_and_only_new_feature_tail(self):
        self.adapter.propose_lattice(self.runner.visible[:, :4], 2)
        self.assertEqual(self.calls, [("prefill", 3), "query"])
        first = self.runner.proposer.conditioned.graphs[(2, 2, 24)]
        self.assertEqual(first["input_ids"][0, 0].item(), 7)
        self.assertEqual(first["feature_inputs"][0, 0, 0].item(), 22)
        self.assertEqual(first["feature_inputs"][0, 1, 0].item(), 22)
        self.target_commit([11, 13])
        self.adapter.propose_lattice(self.runner.visible[:, :6], 2)
        second = self.runner.proposer.conditioned.graphs[(2, 3, 24)]
        self.assertEqual(second["input_ids"][:, :2].reshape(-1).tolist(), [11, 13])
        self.assertEqual(second["feature_inputs"][0, 0, 0].item(), 33)
        self.assertEqual(second["feature_inputs"][0, 1, 0].item(), 33)
        self.target_commit([99])
        audit = self.adapter.finish_request()
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["feature_rows_copied_total"], 6)
        self.assertTrue(audit["feature_rows_copied_exactly_once"])
        self.assertFalse(audit["final_full_audit_excluded_from_generation_time"])

    def test_surviving_old_feature_mutation_is_detected_at_final_audit(self):
        self.adapter.propose_lattice(self.runner.visible[:, :4], 2)
        self.target_commit([8])
        self.runner.verifier.committed_feature_history[0, 0, 0].fill_(999)
        with self.assertRaisesRegex(RuntimeError, "feature row was mutated"):
            self.adapter.finish_request()

    def test_surviving_old_visible_mutation_is_detected_at_final_audit(self):
        self.runner.visible[0, 0].fill_(90)
        with self.assertRaisesRegex(RuntimeError, "token bytes differ"):
            self.adapter.finish_request()

    def test_mutation_then_restore_is_explicitly_outside_contract(self):
        self.runner.verifier.committed_feature_history[0, 0, 0].fill_(999)
        self.runner.verifier.committed_feature_history[0, 0, 0].fill_(20)
        audit = self.adapter.finish_request()
        self.assertFalse(audit["transient_mutation_then_restoration_detectable"])
        self.assertFalse(audit["arbitrary_external_mutation_during_request_supported"])

    def test_unobserved_or_non_original_commit_is_rejected(self):
        self.adapter.propose_lattice(self.runner.visible[:, :4], 2)
        with self.assertRaises(RuntimeError):
            self.adapter.observe_target_commit([8])
        with self.assertRaises(RuntimeError):
            self.adapter.propose_lattice(self.runner.visible[:, :4], 2)

    def test_cache_reset_and_missing_graph_do_not_silently_fallback(self):
        self.adapter.propose_lattice(self.runner.visible[:, :4], 2)
        self.target_commit([8])
        self.runner.proposer.conditioned.reset()
        with self.assertRaisesRegex(RuntimeError, "cache reset"):
            self.adapter.propose_lattice(self.runner.visible[:, :5], 2)
        self.adapter.abort_request()
        self.runner, self.calls = fixture()
        self.adapter = runtime.AppendOnlyPARD2ProposalAdapter(self.runner)
        self.adapter.begin_request([2, 5, 9], 7)
        self.runner.proposer.conditioned.graphs.pop((2, 3, 24))
        with self.assertRaisesRegex(RuntimeError, "graph family"):
            self.adapter.propose_lattice(self.runner.visible[:, :4], 2)

    def test_owner_or_storage_replacement_rejected(self):
        self.runner.visible = self.runner.visible.clone()
        with self.assertRaisesRegex(RuntimeError, "buffers were replaced"):
            self.adapter.propose_lattice(self.runner.visible[:, :4], 2)

    def test_no_active_request_reuse_without_reset(self):
        with self.assertRaises(RuntimeError):
            self.adapter.begin_request([2, 5, 9], 7)
        self.adapter.finish_request()
        with self.assertRaisesRegex(RuntimeError, "Reset the auxiliary"):
            self.adapter.begin_request([2, 5, 9], 7)
        self.adapter.abort_request()
        self.adapter.begin_request([2, 5, 9], 7)
        self.assertEqual(self.adapter.visible_ids_host, (2, 5, 9, 7))


if __name__ == "__main__":
    unittest.main()
