"""Exercise policy correction, restart, and rejected promotion on the CPU."""

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless


@skipUnless(
    find_spec("torch") and find_spec("safetensors"), "skill memory uses torch and safetensors",
)
class SkillMemoryTests(TestCase):
    def test_corrected_policy_restores_vectors_and_retains_earlier_skill(self):
        import torch

        from lightreason_memory.latent_freetoken_hook import HIDDEN_SIZE, MAX_ROWS, embedding_digest
        from lightreason_memory.rl_memory import FEATURE_WIDTH
        from scientist_ui.skill_memory import SkillMemory

        class Encoder:
            def request(self, question, *, encode_values):
                topic = ("Drift", "Scoring", "Unrelated").index(question.split()[0])
                feature = torch.zeros((1, 1, FEATURE_WIDTH))
                feature[0, 0, topic] = 1
                feature[0, 0, topic + FEATURE_WIDTH // 2] = 1
                return feature, torch.ones((1, 1), dtype=torch.bool), None, None

        actions = {"original": "Original reference check", "revised": "Sealed reference check",
                   "retained": "Blind the judges"}

        def rows(lesson):
            value = list(actions.values()).index(lesson) + 1
            return torch.full((MAX_ROWS, HIDDEN_SIZE), value, dtype=torch.bfloat16), 2

        def case(kind, question, action, *, retain=False):
            return {"id": kind, "question": question, "action": action, "retain": retain}

        def batch(action):
            return [
                case("transfer", "Drift in training apparatus", action),
                case("retention", "Scoring in training trial", "retained"),
                case("abstain", "Unrelated training question", None),
            ], [
                case("transfer", "Drift in unfamiliar apparatus", action),
                case("retention", "Scoring new samples", "retained", retain=True),
                case("abstain", "Unrelated follow-up", None),
            ]

        encoder = Encoder()
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "skills.safetensors"
            memory = SkillMemory(path, rows)
            self.assertIsNone(memory.recall(encoder, "Drift in unfamiliar apparatus"))
            training, checks = batch("original")
            self.assertTrue(memory.train(encoder, labels=actions, training=training, checks=checks)["promoted"])
            training, checks = batch("revised")
            update = memory.train(encoder, labels=actions, training=training, checks=checks)
            self.assertTrue(update["promoted"], update)
            self.assertEqual([row["before"] for row in update["checks"]], ["original", "retained", None])
            self.assertGreater(update["optimizer_steps"], 0)

            memory = SkillMemory(path, None)
            restored = memory.evaluate(encoder, checks, labels=actions)
            self.assertEqual([row["actual"] for row in restored], ["revised", "retained", None])
            vectors = memory.recall(encoder, checks[0]["question"])
            self.assertTrue(torch.equal(vectors, rows(actions["revised"])[0][:2]))
            self.assertEqual(restored[0]["vector_rows"], len(vectors))
            self.assertEqual(restored[0]["vector_sha256"], embedding_digest(vectors))
            self.assertEqual(restored[2]["vector_rows"], 0)
            self.assertIsNone(restored[2]["vector_sha256"])
            self.assertIsNone(memory.recall(encoder, checks[2]["question"]))
            accepted_state = path.read_bytes()
            for lesson in actions.values():
                self.assertNotIn(lesson.encode(), accepted_state)

            training, checks = batch("original")
            checks.append(case("contradictory", "Drift on a third apparatus", None))
            refused = memory.train(encoder, labels=actions, training=training, checks=checks)
            self.assertFalse(refused["promoted"], refused)
            self.assertEqual(path.read_bytes(), accepted_state)
