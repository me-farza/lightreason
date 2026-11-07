"""CPU-only contract tests for the live comparison observers and metrics.

Run from the workspace root: python3 -m unittest demo_chat.test_inference
No Torch import, CUDA device, model checkpoint, or HTTP service is required.
"""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from demo_chat.inference import _instance_observers, compare_outputs


class _VerifierOwner:
    def __init__(self):
        self.anchor = SimpleNamespace(item=lambda: 7)
        self.commits = []
        self.fail_commit = False

    def prefill(self, ids):
        return self.anchor

    def commit_path(self, path):
        if self.fail_commit:
            raise RuntimeError("Original commit failed")
        self.commits.append(list(path))
        return self.commits[-1]


class _ForwardingVerifierProxy:
    """Match the audited proxy's consequential attribute-forwarding behavior."""
    def __init__(self, owner):
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name):
        return getattr(self._owner, name)

    def __setattr__(self, name, value):
        setattr(self._owner, name, value)

    def prefill(self, ids):
        return self._owner.prefill(ids)


class _Runner:
    def __init__(self):
        self.verifier = _ForwardingVerifierProxy(_VerifierOwner())

    @staticmethod
    def accept_path(tree, predicted, remaining, eos):
        return tree["accepted_result"]


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.runner = _Runner()
        self.proxy = self.runner.verifier
        self.owner = self.proxy._owner
        self.proxy_before = vars(self.proxy).copy()
        self.runner_before = vars(self.runner).copy()
        self.owner_prefill_function = self.owner.prefill.__func__
        self.owner_commit_function = self.owner.commit_path.__func__
        self.accepted_result = ([0, 3], [11, 12])
        self.tree = {"accepted_result": self.accepted_result}

    def assert_restored(self):
        self.assertEqual(vars(self.proxy), self.proxy_before)
        self.assertEqual(vars(self.runner), self.runner_before)
        # Forwarded attribute writes must never have installed observer methods
        # on the original owner retained by the feature-conditioned proposer.
        self.assertNotIn("prefill", vars(self.owner))
        self.assertNotIn("commit_path", vars(self.owner))
        self.assertIs(self.owner.prefill.__func__, self.owner_prefill_function)
        self.assertIs(self.owner.commit_path.__func__, self.owner_commit_function)

    def test_emits_only_original_anchor_and_committed_target_tokens(self):
        events = []
        with _instance_observers(self.runner, events.append) as observed:
            self.assertIs(self.proxy.prefill("actual-ids"), self.owner.anchor)
            returned = self.runner.accept_path(self.tree, "actual-predictions", 15, {99})
            self.assertIs(returned, self.accepted_result)
            self.assertEqual(events, [[7]])  # Proposed tokens are not published.
            committed = self.proxy.commit_path(returned[0])
            self.assertIs(committed, self.owner.commits[-1])
            self.assertEqual(observed, [7, 11, 12])
            self.assertEqual(events, [[7], [7, 11, 12]])
            self.assertIsNot(events[0], events[1])
        self.assertEqual(self.owner.commits, [[0, 3]])
        self.assert_restored()

    def test_prefill_callback_failure_restores_every_observer(self):
        def disconnected(_):
            raise BrokenPipeError("Client disconnected")

        with self.assertRaisesRegex(BrokenPipeError, "Client disconnected"):
            with _instance_observers(self.runner, disconnected):
                self.proxy.prefill("actual-ids")
        self.assertEqual(self.owner.commits, [])
        self.assert_restored()

    def test_commit_callback_failure_preserves_original_commit_and_restores(self):
        events = []

        def disconnected_after_commit(tokens):
            events.append(tokens)
            if len(tokens) > 1:
                raise BrokenPipeError("Client disconnected after commit")

        with self.assertRaisesRegex(BrokenPipeError, "after commit"):
            with _instance_observers(self.runner, disconnected_after_commit):
                self.proxy.prefill("actual-ids")
                path, _ = self.runner.accept_path(self.tree, [], 15, set())
                self.proxy.commit_path(path)
        # The observer must not undo, replace, or retry the target's commit.
        self.assertEqual(self.owner.commits, [[0, 3]])
        self.assertEqual(events, [[7], [7, 11, 12]])
        self.assert_restored()

    def test_original_commit_failure_publishes_no_uncommitted_tokens(self):
        self.owner.fail_commit = True
        events = []
        with self.assertRaisesRegex(RuntimeError, "Original commit failed"):
            with _instance_observers(self.runner, events.append):
                self.proxy.prefill("actual-ids")
                path, _ = self.runner.accept_path(self.tree, [], 15, set())
                self.proxy.commit_path(path)
        self.assertEqual(self.owner.commits, [])
        self.assertEqual(events, [[7]])
        self.assert_restored()

    def test_existing_instance_method_overrides_restored_by_identity(self):
        custom_prefill = lambda ids: self.owner.anchor
        custom_commit = lambda path: ("original-override", path)
        custom_accept = lambda *args: self.accepted_result
        object.__setattr__(self.proxy, "prefill", custom_prefill)
        object.__setattr__(self.proxy, "commit_path", custom_commit)
        object.__setattr__(self.runner, "accept_path", custom_accept)
        proxy_before, runner_before = vars(self.proxy).copy(), vars(self.runner).copy()
        with _instance_observers(self.runner, lambda tokens: None):
            self.proxy.prefill("actual-ids")
            path, _ = self.runner.accept_path(self.tree, [], 15, set())
            self.assertEqual(self.proxy.commit_path(path), ("original-override", path))
        self.assertEqual(vars(self.proxy), proxy_before)
        self.assertEqual(vars(self.runner), runner_before)
        self.assertIs(self.proxy.prefill, custom_prefill)
        self.assertIs(self.proxy.commit_path, custom_commit)
        self.assertIs(self.runner.accept_path, custom_accept)
        self.assertNotIn("prefill", vars(self.owner))
        self.assertNotIn("commit_path", vars(self.owner))


def _result(tokens, seconds, eos):
    return {"token_ids": tokens, "metrics": {
        "seconds": seconds, "tokens_per_second": len(tokens) / seconds,
        "finished_eos": eos}}


class ComparisonMetricTests(unittest.TestCase):
    def test_shorter_different_output_has_distinct_latency_and_token_rate_ratios(self):
        result = compare_outputs(_result([1, 2, 3, 99], 4.0, True),
                                 _result([1, 99], 1.0, True))
        self.assertEqual(result["speedup"], 4.0)
        self.assertEqual(result["token_rate_speedup"], 2.0)
        self.assertFalse(result["exact_match"])
        self.assertEqual(result["prefix_match_tokens"], 1)
        self.assertEqual(result["baseline_tokens"], 4)
        self.assertEqual(result["candidate_tokens"], 2)
        self.assertIsNone(result["same_output_latency_speedup"])
        self.assertTrue(result["both_finished_eos"])
        self.assertFalse(result["quality_equivalence_certified"])

    def test_exact_output_enables_same_output_latency_metric(self):
        result = compare_outputs(_result([1, 2, 99], 3.0, True),
                                 _result([1, 2, 99], 1.0, True))
        self.assertTrue(result["exact_match"])
        self.assertEqual(result["prefix_match_tokens"], 3)
        self.assertEqual(result["same_output_latency_speedup"], 3.0)
        self.assertEqual(result["speedup"], result["token_rate_speedup"])
        self.assertFalse(result["quality_equivalence_certified"])

    def test_identical_bounded_prefix_is_not_a_complete_response_certificate(self):
        result = compare_outputs(_result([1, 2, 3], 2.0, False),
                                 _result([1, 2, 3], 1.0, False))
        self.assertTrue(result["exact_match"])
        self.assertFalse(result["baseline_finished_eos"])
        self.assertFalse(result["candidate_finished_eos"])
        self.assertFalse(result["both_finished_eos"])
        self.assertFalse(result["quality_equivalence_certified"])

    def test_eos_status_is_reported_independently_for_each_runner(self):
        result = compare_outputs(_result([5, 99], 2.0, True),
                                 _result([5, 6], 1.0, False))
        self.assertTrue(result["baseline_finished_eos"])
        self.assertFalse(result["candidate_finished_eos"])
        self.assertFalse(result["both_finished_eos"])
        self.assertFalse(result["exact_match"])


if __name__ == "__main__":
    unittest.main()
