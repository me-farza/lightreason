"""Exercise the optional per-user memory coaching session without loading Qwen."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path
import tempfile
from threading import Barrier
import unittest
from unittest.mock import Mock, patch


_HAS_TENSOR_RUNTIME = all(find_spec(name) is not None for name in ("torch", "safetensors"))
FIRST = "My synthetic lantern is now violet."
CORRECTED = "Correction: my synthetic lantern is now navy."
OTHER = "My synthetic notebook is orange."
QUERY = "What is the synthetic lantern color?"
DISTRACTOR = "Suppose the synthetic lantern were white."
UNVERIFIED = "Unverified advice about a synthetic sailboat."


@unittest.skipUnless(_HAS_TENSOR_RUNTIME, "torch and safetensors are required for dream-session tests")
class DreamTests(unittest.TestCase):
    def setUp(self):
        import torch
        from . import rl_memory
        from .neural import _atomic_write
        from .rl_policy import MemoryPolicy

        self.torch = torch
        self.runtime = rl_memory
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)
        self.codec = self.directory / "codec.safetensors"
        self.checkpoint = self.directory / "base.safetensors"
        self.state = self.directory / "state.safetensors"
        self.codec.write_bytes(b"synthetic codec checkpoint")
        self.rows = self._make_encoder()
        encoder = patch.object(rl_memory, "FrozenQwenFeatures", return_value=self.rows)
        encoder.start()
        self.addCleanup(encoder.stop)

        # Produce a predictable real policy. It admits every observation and
        # uses the first token as a key, so each synthetic value sits on that row.
        policy = MemoryPolicy(rl_memory.FEATURE_WIDTH)
        for parameter in policy.parameters():
            parameter.detach().zero_()
        policy.admit.bias.detach()[1] = 1.0
        policy.match_scale.detach().fill_(18.0)
        _atomic_write(policy.state_dict(), self.checkpoint, {
            "version": rl_memory.FORMAT, "backbone": rl_memory.MODEL_VERSION,
            "codec_sha256": sha256(self.codec.read_bytes()).hexdigest(),
        })

    def _make_encoder(self):
        from .neural import KEY_DIM, VALUE_DIM
        from .rl_memory import FEATURE_WIDTH

        torch = self.torch
        data = {}
        for index, text in enumerate((FIRST, OTHER, QUERY, DISTRACTOR, UNVERIFIED, CORRECTED)):
            feats = torch.full((1, 2, FEATURE_WIDTH), (index + 1) / 16)
            mask = torch.ones(1, 2, dtype=torch.bool)
            keys = torch.zeros(1, 2, KEY_DIM)
            keys[0, 0, 1 if text == OTHER else 0] = 1.0
            values = torch.zeros(1, 2, VALUE_DIM)
            values[0, :, index] = 1.0
            data[text] = (feats, mask, keys, values)
        encoder = Mock()
        def request(text, *, encode_values=True):
            features, mask, keys, values = data[text]
            return features, mask, keys, values if encode_values else None

        encoder.request.side_effect = request
        encoder.decode.side_effect = lambda value: "violet" if int(value.argmax().item()) == 0 else "orange"
        ids = {}
        names = {101: "lantern", 102: "violet", 103: "notebook", 104: "orange", 105: "navy"}
        for text, token_pair in ((FIRST, [101, 102]), (OTHER, [103, 104]), (CORRECTED, [101, 105])):
            ids[text] = {
                "input_ids": token_pair,
                "offset_mapping": [(text.index(names[token]), text.index(names[token]) + len(names[token]))
                                   for token in token_pair],
            }
        encoder.ids.side_effect = ids.__getitem__
        encoder.tokenizer.decode.side_effect = lambda tokens: names[tokens[0]]
        return encoder

    def _memory(self, *, every=2, state=None, destination=None):
        return self.runtime.PolicyMemory(
            self.directory / "model", self.codec, self.checkpoint, state or self.state, "cpu", dream_every=every,
            destination_model_path=destination,
        )

    def _bank(self):
        from safetensors.torch import load_file

        state = load_file(self.state, device="cpu")
        return {key: state[key].clone() for key in ("keys", "values", "occupied")}

    def _assert_same_bank(self, original):
        for key, value in self._bank().items():
            self.torch.testing.assert_close(value, original[key], rtol=0, atol=0)

    def _coach(self, **options):
        def result(policy, focus, *args, **kwargs):
            from .dream_train import DreamResult

            candidate = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
            candidate["match_scale"].fill_(19.0)
            accepted = options.get("accepted", True)
            return DreamResult(accepted, candidate if accepted else None, {
                "reason": options.get("reason", "accepted" if accepted else "behavior_gate_failed"),
                "steps": 1 if accepted else 0,
            })
        return patch("lightreason_memory.dream_state.coach", side_effect=result)

    def _assert_active_scale_after_restart(self, expected):
        originals = []

        def inspect(policy, *args, **kwargs):
            from .dream_train import DreamResult

            originals.append(float(policy.match_scale.item()))
            return DreamResult(False, None, {"reason": "already_correct", "steps": 0})

        with patch("lightreason_memory.dream_state.coach", side_effect=inspect):
            memory = self._memory()
            memory.absorb(OTHER, expected_key="notebook", expected_value="orange")
            result = self._memory().recall(QUERY)
        self.assertEqual(result["dream"]["messages"], 4)
        self.assertEqual(result["dream"]["status"], "unchanged")
        self.assertEqual(originals, [expected])

    def test_opt_in_and_cadence_survive_restart(self):
        with self._coach() as coach:
            initial = self._memory(every=0)
            initial.absorb(FIRST)
            initial.recall(QUERY)
            coach.assert_not_called()

            enabled = self._memory()
            self.assertEqual(enabled.dream_status()["messages"], 0)
            first = enabled.absorb(FIRST, expected_key="lantern", expected_value="violet")
            self.assertEqual(first["dream"]["messages"], 1)
            self.assertEqual(first["dream"]["pending"], 1)
            coach.assert_not_called()
            bank = self._bank()
            checkpoint = self.checkpoint.read_bytes()
            reply = self._memory().recall(QUERY)
            self.assertEqual(reply["answer"], "violet")
            self.assertEqual(reply["dream"]["status"], "promoted")
            self.assertEqual(reply["dream"]["messages"], 2)
            self.assertEqual(reply["dream"]["pending"], 0)
            self.assertEqual(self._memory().dream_status()["messages"], 2)
            self.assertEqual(self.checkpoint.read_bytes(), checkpoint)
            self._assert_same_bank(bank)
            coach.assert_called_once()
            focus = coach.call_args.args[1]
            self.assertIs(focus.write, True)
            self.assertEqual((focus.key_position, focus.value_position), (0, 1))
        self._assert_active_scale_after_restart(19.0)

    def test_unverified_messages_do_not_become_training_targets(self):
        with self._coach() as coach:
            memory = self._memory()
            memory.absorb(FIRST)
            reply = memory.recall(QUERY)
            self.assertEqual(reply["dream"]["status"], "awaiting_feedback")
            self.assertEqual(reply["dream"]["pending"], 0)
            coach.assert_not_called()
            memory.absorb(DISTRACTOR, ignore=True)
            reply = self._memory().recall(QUERY)
            self.assertEqual(reply["dream"]["status"], "promoted")
            coach.assert_called_once()
            focus = coach.call_args.args[1]
            self.assertIs(focus.write, False)
            self.assertIsNone(focus.key_position)
            self.assertIsNone(focus.value_position)
            self.assertNotIn(FIRST.encode(), self.state.read_bytes())
            self.assertNotIn(DISTRACTOR.encode(), self.state.read_bytes())
            self.assertNotIn(QUERY.encode(), self.state.read_bytes())
            self.assertNotIn(b"input_ids", self.state.read_bytes())
            self.assertEqual(self.state.stat().st_mode & 0o777, 0o600)

    def test_invalid_verification_neither_admits_nor_counts(self):
        memory = self._memory()
        for feedback in (
            {"expected_key": "lantern"},
            {"expected_key": "lantern", "expected_value": "missing"},
            {"expected_key": "lantern", "expected_value": "violet", "ignore": True},
        ):
            with self.subTest(feedback=feedback), self.assertRaises(ValueError):
                memory.absorb(FIRST, **feedback)
            self.assertEqual(memory.dream_status()["messages"], 0)

    def test_vectors_preserve_shape_and_trigger_pending_feedback(self):
        with self._coach() as coach:
            memory = self._memory()
            memory.absorb(FIRST, expected_key="lantern", expected_value="violet")
            vector = self._memory().vectors(QUERY)
            self.assertEqual(vector.dtype, self.torch.float16)
            self.assertEqual(vector.shape, (1, self.runtime.VALUE_DIM))
            self.rows.request.assert_called_with(QUERY, encode_values=False)
            self.rows.decode.assert_not_called()
            self.assertEqual(self._memory().dream_status()["status"], "promoted")
            coach.assert_called_once()

    def test_aligned_vectors_follow_corrections_and_text_only_writes(self):
        from .neural import TARGET_WIDTH

        torch = self.torch
        destination = self.directory / "35b"
        tokenizer = Mock(all_special_ids=[])
        tokenizer.encode.side_effect = lambda value, **kwargs: {" violet": [0, 1], " navy": [2]}[value]
        table = torch.arange(1, 4, dtype=torch.bfloat16)[:, None].repeat(1, TARGET_WIDTH)

        with patch("lightreason_memory.latent_bridge.load_destination", return_value=(tokenizer, table)) as load:
            writer = self._memory(every=20, destination=destination)
            writer.absorb(FIRST, expected_key="lantern", expected_value="violet")
            first = self._memory(every=0, destination=destination).vectors(QUERY, target_aligned=True)
            torch.testing.assert_close(first, table[:2], rtol=0, atol=0)

            writer.absorb(CORRECTED, expected_key="lantern", expected_value="navy")
            reader = self._memory(every=0, destination=destination)
            corrected = reader.vectors(QUERY, target_aligned=True)
            torch.testing.assert_close(corrected, table[2:], rtol=0, atol=0)
            corrected_native = reader.vectors(QUERY)

            # The compact value is unchanged. Dropping the aligned row must
            # still persist when the same observation is written without 35B.
            self.assertFalse(self._memory(every=0).absorb(CORRECTED)["updated"])
            reader = self._memory(every=0, destination=destination)
            self.assertIsNone(reader.vectors(QUERY, target_aligned=True))
            torch.testing.assert_close(reader.vectors(QUERY), corrected_native, rtol=0, atol=0)
            self.rows.decode.assert_not_called()
            load.assert_called_once_with(destination)

    def test_rejected_candidate_preserves_bank_and_policy(self):
        with self._coach(accepted=False) as coach:
            memory = self._memory()
            memory.absorb(FIRST, expected_key="lantern", expected_value="violet")
            before = self._bank()
            reply = self._memory().recall(QUERY)
            self.assertEqual(reply["dream"]["status"], "rejected")
            self.assertEqual(reply["dream"]["pending"], 0)
            self.assertEqual(reply["answer"], "violet")
            coach.assert_called_once()
            self._assert_same_bank(before)
        self._assert_active_scale_after_restart(18.0)

    def test_feedback_already_satisfied_reports_unchanged(self):
        with self._coach(accepted=False, reason="already_correct") as coach:
            memory = self._memory()
            memory.absorb(FIRST, expected_key="lantern", expected_value="violet")
            reply = memory.recall(QUERY)
            self.assertEqual(reply["dream"]["status"], "unchanged")
            coach.assert_called_once()

    def test_real_coach_changes_a_later_unverified_action(self):
        from safetensors import safe_open
        from safetensors.torch import load_file
        from .neural import _atomic_write

        # Give the synthetic writer a small preference for admitting observations.
        with safe_open(self.checkpoint, framework="pt", device="cpu") as file:
            metadata = file.metadata()
        policy = load_file(self.checkpoint)
        policy["admit.bias"][1] = 0.025
        _atomic_write(policy, self.checkpoint, metadata)

        memory = self._memory()
        memory.absorb(DISTRACTOR, ignore=True)
        before = self._bank()
        reply = self._memory().recall(QUERY)
        self.assertEqual(reply["dream"]["status"], "promoted")
        self.assertGreater(reply["dream"]["steps"], 0)
        self._assert_same_bank(before)
        later = self._memory().absorb(DISTRACTOR)
        self.assertEqual(later["status"], "unknown")
        self.assertFalse(later["updated"])

    def test_failed_coach_keeps_recall_available(self):
        with patch("lightreason_memory.dream_state.coach", side_effect=RuntimeError("mock optimizer failure")):
            memory = self._memory()
            memory.absorb(FIRST, expected_key="lantern", expected_value="violet")
            before = self._bank()
            reply = self._memory().recall(QUERY)
            self.assertEqual(reply["dream"]["status"], "training_failed")
            self.assertEqual(reply["answer"], "violet")
            self._assert_same_bank(before)

    def test_simultaneous_messages_commit_once_each(self):
        with self._coach() as coach:
            left = self._memory()
            right = self._memory()
            gate = Barrier(2, timeout=5)

            def send(memory):
                gate.wait()
                return memory.recall(QUERY)

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(send, memory) for memory in (left, right)]
                answers = [future.result(timeout=15) for future in futures]
            self.assertEqual(sorted(reply["dream"]["messages"] for reply in answers), [1, 2])
            self.assertEqual(self._memory().dream_status()["messages"], 2)
            coach.assert_not_called()


if __name__ == "__main__":
    unittest.main()
