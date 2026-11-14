"""Exercise imitation and outcome-based optimization on a small discrete policy."""

from importlib.util import find_spec
import unittest


@unittest.skipUnless(find_spec("torch"), "torch is required for the memory-policy experiment")
class RLTrainTests(unittest.TestCase):
    def make_problem(self):
        import torch

        policy = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(policy.weight)
        target = torch.tensor([0, 1] * 20)
        data = {"features": torch.nn.functional.one_hot(target).float(), "target": target}
        return torch, policy, data

    @staticmethod
    def rollout(policy, batch, *, mode, generator=None):
        import torch
        from .rl_train import Trajectory

        logits = policy(batch["features"])
        scores = torch.log_softmax(logits, dim=-1)
        probabilities = scores.exp()
        target = batch["target"]
        if mode == "teacher":
            chosen = target
        elif mode == "sample":
            chosen = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        else:
            chosen = logits.argmax(dim=-1)
        rewards = chosen.eq(target).float()
        return Trajectory(
            scores.gather(1, chosen[:, None]),
            -(probabilities * scores).sum(dim=-1, keepdim=True),
            rewards,
            {"recall": (int(rewards.sum()), target.numel())},
        )

    def test_reinforcement_learns_without_teacher_loss(self):
        from .rl_train import evaluate, train

        torch, policy, data = self.make_problem()
        train_data = {key: value[:30] for key, value in data.items()}
        dev_data = {key: value[30:] for key, value in data.items()}
        states, report = train(
            policy, train_data, dev_data, self.rollout, warmstart_steps=0, rl_steps=12,
            learning_rate=0.05, rl_learning_rate=0.05, batch_size=14, seed=183, cpu_threads=1,
        )
        self.assertEqual(set(states), {"untrained", "supervised", "rl"})
        self.assertTrue(torch.equal(states["untrained"]["weight"], states["supervised"]["weight"]))
        self.assertFalse(torch.equal(states["untrained"]["weight"], states["rl"]["weight"]))
        self.assertEqual(report["stages"]["untrained"]["development"]["mean_reward"], 0.5)
        self.assertEqual(report["stages"]["rl"]["development"]["mean_reward"], 1.0)
        aggregate = evaluate(policy, dev_data, self.rollout, batch_size=3)["metrics"]["recall"]
        self.assertEqual(aggregate, {"correct": 10, "total": 10, "accuracy": 1.0})

    def test_imitation_stage_and_mode_restoration(self):
        from .rl_train import train

        torch, policy, data = self.make_problem()
        train_data = {key: value[:30] for key, value in data.items()}
        dev_data = {key: value[30:] for key, value in data.items()}
        policy.eval()
        states, report = train(
            policy, train_data, dev_data, self.rollout, warmstart_steps=4, rl_steps=0,
            learning_rate=0.05, batch_size=14, cpu_threads=1,
        )
        self.assertFalse(policy.training)
        self.assertTrue(torch.equal(states["supervised"]["weight"], states["rl"]["weight"]))
        self.assertEqual(report["stages"]["supervised"]["development"]["metrics"]["recall"]["accuracy"], 1.0)

    def test_matched_teacher_and_reinforcement_share_batches_and_start_state(self):
        from .rl_train import train

        torch, policy, data = self.make_problem()
        data["id"] = torch.arange(len(data["target"]))
        train_data = {key: value[:30] for key, value in data.items()}
        dev_data = {key: value[30:] for key, value in data.items()}
        batches = {"teacher": [], "sample": []}
        initials = {mode: [] for mode in batches}

        def rollout(policy, batch, *, mode, generator=None):
            if mode in batches:
                batches[mode].append(batch["id"].tolist())
                initials[mode].append(policy.weight.detach().clone())
            return self.rollout(policy, batch, mode=mode, generator=generator)

        states, report = train(
            policy, train_data, dev_data, rollout, warmstart_steps=2, rl_steps=3, batch_size=12,
            matched_teacher=True, cpu_threads=1,
        )
        self.assertEqual(report["stages"]["matched_teacher"]["steps"], 3)
        self.assertEqual(batches["teacher"][2:], batches["sample"])
        self.assertTrue(torch.equal(initials["teacher"][2], initials["sample"][0]))
        self.assertTrue(torch.equal(initials["sample"][0], states["supervised"]["weight"]))

    def test_expired_evaluation_does_not_run_or_change_policy_mode(self):
        from .rl_train import evaluate

        _, policy, data = self.make_problem()
        absent = lambda *args, **kwargs: self.fail("An expired evaluation must not call the policy.")
        with self.assertRaises(TimeoutError):
            evaluate(policy, data, absent, deadline=0)
        self.assertTrue(policy.training)
