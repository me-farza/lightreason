"""CPU tests for the constrained vector-only coaching optimizer."""

from importlib.util import find_spec
import unittest


@unittest.skipUnless(find_spec("torch"), "torch is required for neural memory coaching")
class DreamTrainTests(unittest.TestCase):
    def make_problem(self):
        import torch
        from .dream_train import DreamExample
        from .rl_policy import MemoryPolicy

        torch.manual_seed(5)
        policy = MemoryPolicy(8)
        with torch.no_grad():
            policy.admit.bias.copy_(torch.tensor([1., -1.]))
        features = torch.randn(1, 4, 8)
        mask = torch.ones(1, 4, dtype=torch.bool)
        keys = torch.eye(4)[None]
        values = keys.clone()
        focus = DreamExample(features, mask, keys, values, True, 2, 1)
        ignore = DreamExample(features, mask, keys, values, False)
        return torch, policy, focus, ignore

    def test_verified_write_really_updates_copy_inside_inference_mode(self):
        from .dream_train import coach

        torch, policy, focus, _ = self.make_problem()
        state_before = {key: value.clone() for key, value in policy.state_dict().items()}
        with torch.inference_mode():
            result = coach(policy, focus, max_steps=80, timeout_s=20)
        self.assertTrue(result.accepted, result.stats)
        self.assertEqual(result.stats["reason"], "accepted")
        self.assertGreater(result.stats["steps"], 0)
        self.assertFalse(result.stats["focus_before"])
        self.assertTrue(result.stats["focus_after"])
        self.assertTrue(all(torch.equal(value, policy.state_dict()[key]) for key, value in state_before.items()))
        self.assertTrue(any(not torch.equal(value, result.candidate_state[key]) for key, value in state_before.items()))
        policy.load_state_dict(result.candidate_state)
        repeated = coach(policy, focus)
        self.assertFalse(repeated.accepted)
        self.assertIsNone(repeated.candidate_state)
        self.assertEqual(repeated.stats["reason"], "already_correct")

    def test_conflicting_guard_preserves_prior_behavior_and_blocks_promotion(self):
        from .dream_train import coach

        _, policy, focus, ignore = self.make_problem()
        result = coach(policy, focus, guard=(ignore,), max_steps=40, timeout_s=20)
        self.assertFalse(result.accepted)
        self.assertIsNone(result.candidate_state)
        self.assertEqual(result.stats["reason"], "behavior_gate_failed")
        self.assertEqual(result.stats["guard_before_correct"], 1)
        self.assertGreater(result.stats["behavior_checks"], 0)
        self.assertEqual(result.stats["guard_regressions"], 1)

    def test_invalid_and_expired_feedback_cannot_promote(self):
        from dataclasses import replace
        from .dream_train import coach

        _, policy, focus, _ = self.make_problem()
        with self.assertRaises(ValueError):
            coach(policy, replace(focus, key_position=999))
        result = coach(policy, focus, max_steps=80, timeout_s=1e-9)
        self.assertFalse(result.accepted)
        self.assertEqual(result.stats["reason"], "timeout")
        self.assertEqual(result.stats["steps"], 0)

    def test_verified_read_feedback_can_improve_slot_selection(self):
        from .dream_train import DreamExample, coach
        from .rl_policy import Bank

        torch, policy, write, _ = self.make_problem()
        bank = Bank.empty(1, 4, 4, "cpu")
        bank.keys[0, 0] = write.keys[0, 0]
        bank.values[0, 0] = write.values[0, 3]
        bank.occupied[0, 0] = True
        with torch.no_grad():
            policy.match_scale.fill_(5.)
            policy.abstain.bias.zero_()
            policy.abstain.weight.zero_()
            policy.query.weight.zero_()
            policy.query.bias.zero_()
        focus = DreamExample(write.features, write.mask, write.keys, bank=bank, slot=0, key_position=0)
        result = coach(policy, focus, max_steps=100, timeout_s=20)
        self.assertTrue(result.accepted, result.stats)


if __name__ == "__main__":
    unittest.main()
