"""CPU reference, admission, receipt, and fused-tail ownership tests."""
import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import append_only_pard2_runtime3b_v1 as adapter_runtime
from gpu_tree_control3b_v1 import (cpu_self_test, cpu_frontier_emulation, cpu_reference_tree,
    cpu_reference_accept, checked_config, acknowledge_original_target_commit, _GPUFeatureTailAdapter)
from test_append_only_pard2_runtime3b_v1 import FakeTensor, FakeTorch, fixture


class CPUReferenceTests(unittest.TestCase):
    def test_all_physical_shapes_random_and_exact_ties_match_frozen_heap(self):
        result = cpu_self_test()
        self.assertEqual(result["CPU_frozen_heap_vs_fixed_frontier_cases"], 48)
        self.assertTrue(result["all_passed"])
        self.assertFalse(result["GPU_work_performed"])

    def test_single_position_and_single_branch_depth_boundaries(self):
        tree = cpu_frontier_emulation(7, [list(range(16))], [[-0.2] * 16], 16, 100)
        self.assertEqual(tree["depths"], [0] + [1] * 15)
        chain = cpu_frontier_emulation(7, [[row + 20] for row in range(15)], [[-0.1] for _ in range(15)], 16, 100)
        self.assertEqual(chain["parents"], list(range(-1, 15)))
        self.assertEqual(chain["depths"], list(range(16)))

    def test_private_unique_in_vocabulary_finite_inputs_required(self):
        ids = [list(range(8)) for _ in range(15)]
        scores = [[-0.2] * 8 for _ in range(15)]
        for kind in ("duplicate", "negative", "too_large", "nan", "infinity", "positive", "empty"):
            bad_ids, bad_scores = copy.deepcopy(ids), copy.deepcopy(scores)
            if kind == "duplicate": bad_ids[0][1] = bad_ids[0][0]
            elif kind == "negative": bad_ids[0][0] = -1
            elif kind == "too_large": bad_ids[0][0] = 100
            elif kind == "nan": bad_scores[0][0] = math.nan
            elif kind == "infinity": bad_scores[0][0] = math.inf
            elif kind == "positive": bad_scores[0][0] = 0.1
            elif kind == "empty": bad_scores[0] = [-math.inf] * 8
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                cpu_reference_tree(7, bad_ids, bad_scores, 64, 100)

    def test_acceptance_has_native_eos_remaining_and_target_bonus(self):
        tree = {"tokens": [7, 11, 13, 17], "parents": [-1, 0, 1, 0]}
        self.assertEqual(cpu_reference_accept(tree, [11, 13, 19, 23], 10, []), ([0, 1, 2], [11, 13, 19]))
        self.assertEqual(cpu_reference_accept(tree, [11, 13, 19, 23], 1, []), ([0], [11]))
        self.assertEqual(cpu_reference_accept(tree, [11, 13, 19, 23], 10, [13]), ([0, 1], [11, 13]))
        self.assertEqual(cpu_reference_accept(tree, [31, 13, 19, 23], 10, []), ([0], [31]))

    def test_configuration_cannot_extend_supported_physical_or_draft_range(self):
        self.assertEqual(checked_config(64, 15, 8, 151936), (64, 15, 8, 151936))
        for values in ((63, 15, 8, 151936), (64, 33, 8, 151936), (64, 15, 17, 151936),
                       (64, 1, 1, 151936), (64, True, 8, 151936)):
            with self.assertRaises(ValueError):
                checked_config(*values)


def metadata_fixture():
    entry = object()
    owner = SimpleNamespace(pending={"basep": 10, "node_count": 64}, _pending_entry=entry,
        _pending_topology=None, current_position=10, max_len=128, committed_input_tokens=0,
        compacted_input_tokens=0, commit_calls=0, _anchor_buffer=object(), anchor=object())
    receipt = {"error": 0, "count": 3, "path": [0, 1, 9], "emitted": [11, 13, 17],
               "root": 7, "anchor": 17, "position": 13}
    return owner, entry, receipt


class ReceiptTests(unittest.TestCase):
    def test_only_the_matching_original_target_path_advances_metadata(self):
        owner, entry, receipt = metadata_fixture()
        path, emitted, oldp, newp = acknowledge_original_target_commit(owner, entry, receipt, 7, 64, 16, 100)
        self.assertEqual((path, emitted, oldp, newp), ([0, 1, 9], [11, 13, 17], 10, 13))
        self.assertEqual(owner.current_position, 13)
        self.assertEqual(owner.committed_input_tokens, 3)
        self.assertEqual(owner.commit_calls, 1)
        self.assertIsNone(owner.pending)
        self.assertIs(owner.anchor, owner._anchor_buffer)

    def test_invalid_receipts_never_advance_CPU_persistent_state(self):
        for field, value in (("error", 1), ("count", 0), ("path", [1, 2, 3]),
                             ("path", [0, 1, 1]), ("root", 8), ("anchor", 99),
                             ("position", 14), ("emitted", [11, -1, 17])):
            owner, entry, receipt = metadata_fixture()
            receipt[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                acknowledge_original_target_commit(owner, entry, receipt, 7, 64, 16, 100)
            self.assertEqual(owner.current_position, 10)
            self.assertEqual(owner.commit_calls, 0)
            self.assertIs(owner._pending_entry, entry)

    def test_no_cross_entry_or_replayed_commit(self):
        owner, entry, receipt = metadata_fixture()
        with self.assertRaises(RuntimeError):
            acknowledge_original_target_commit(owner, object(), receipt, 7, 64, 16, 100)
        acknowledge_original_target_commit(owner, entry, receipt, 7, 64, 16, 100)
        with self.assertRaises(RuntimeError):
            acknowledge_original_target_commit(owner, entry, receipt, 7, 64, 16, 100)


class FusedFeatureOwnershipTests(unittest.TestCase):
    def test_fused_original_target_tail_is_acknowledged_without_another_copy(self):
        with patch.object(adapter_runtime, "torch", FakeTorch):
            runner, calls = fixture()
            adapter = _GPUFeatureTailAdapter(runner)
            adapter.begin_request([2, 5, 9], 7)
            adapter.propose_lattice(runner.visible[:, :4], 2)
            # Emulate completed original-target GPU writes and the one receipt.
            runner.verifier.committed_feature_history[:, 3:5].fill_(33)
            runner.proposer.conditioned._verified_feature_snapshot[:, 3:5].fill_(33)
            runner.visible.put((0, 4), 11)
            runner.visible.put((0, 5), 13)
            runner.verifier.current_position = 5
            runner.verifier.verify_calls = runner.verifier.commit_calls = 1
            runner.verifier.anchor.fill_(13)
            adapter.acknowledge_queued_target_tail(3, 5)
            copies = []
            original = FakeTensor.copy_
            def observe_copy(destination, source):
                copies.append(destination.shape)
                return original(destination, source)
            with patch.object(FakeTensor, "copy_", observe_copy):
                adapter.observe_target_commit([11, 13])
            self.assertEqual(copies, [])
            self.assertEqual(adapter.copied_feature_rows, 5)
            self.assertTrue(adapter.finish_request()["passed"])

    def test_fused_tail_cannot_be_authorized_without_original_target_cycle(self):
        with patch.object(adapter_runtime, "torch", FakeTorch):
            runner, calls = fixture()
            adapter = _GPUFeatureTailAdapter(runner)
            adapter.begin_request([2, 5, 9], 7)
            with self.assertRaises(RuntimeError):
                adapter.acknowledge_queued_target_tail(3, 5)
            adapter.propose_lattice(runner.visible[:, :4], 2)
            with self.assertRaises(RuntimeError):
                adapter.acknowledge_queued_target_tail(2, 5)
            adapter.acknowledge_queued_target_tail(3, 5)
            with self.assertRaises(RuntimeError):
                adapter.acknowledge_queued_target_tail(3, 5)
            adapter.abort_request()
            self.assertIsNone(adapter._fused_tail)


if __name__ == "__main__":
    unittest.main()
