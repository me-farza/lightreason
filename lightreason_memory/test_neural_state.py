"""CPU checks for tensor-only persistence and protected state handling."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

try:
    import torch
    from . import neural
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "install PyTorch and safetensors to run tensor-state tests")
class StateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.memory = object.__new__(neural.NeuralMemory)
        self.memory.state_path = Path(directory.name) / "state.safetensors"
        self.memory.binding = torch.arange(32, dtype=torch.uint8)

    def test_restores_fixed_shape_latents_without_text(self):
        initial = self.memory._load_state()
        self.assertFalse(self.memory.state_path.exists())
        key = torch.rand((1, neural.KEY_DIM))
        initial["keys"][0] = key
        initial["lengths"][0] = 2
        initial["updated"][0] = 1
        initial["values"][0, :2] = torch.rand((2, neural.VALUE_DIM))
        neural._atomic_write(initial, self.memory.state_path, {"version": neural.VERSION})
        restored = self.memory._load_state()
        self.assertEqual(neural.NeuralMemory._matching(restored, key), 0)
        torch.testing.assert_close(restored["values"], initial["values"])
        self.assertEqual(self.memory.state_path.stat().st_mode & 0o777, 0o600)
        self.assertNotIn(b"experience", self.memory.state_path.read_bytes())

    def test_empty_bank_cannot_match(self):
        empty = self.memory._load_state()
        self.assertIsNone(self.memory._matching(empty, torch.ones((1, neural.KEY_DIM))))

    def test_recall_vectors_never_decodes_the_value(self):
        self.memory.destination_model_path = self.memory.state_path.parent / "35b"
        state = self.memory._load_state()
        key = torch.ones((1, neural.KEY_DIM))
        state["keys"][0] = key[0]
        state["lengths"][0] = 1
        state["target_lengths"][0] = 2
        state["updated"][0] = 1
        state["values"][0, :1].fill_(.125)
        state["target_values"][0, :2].fill_(.25)
        self.memory._tokens = Mock(return_value=(torch.tensor([[1]]), torch.tensor([1])))
        self.memory.model = Mock()
        self.memory.model.key.return_value = key
        self.memory.model.decode.side_effect = AssertionError("vector recall cannot decode")
        neural._atomic_write(state, self.memory.state_path, {"version": neural.VERSION})

        raw = self.memory.recall_vectors("What is the access phrase?")
        aligned = self.memory.recall_vectors("What is the access phrase?", target_aligned=True)
        torch.testing.assert_close(raw, state["values"][0, :1])
        torch.testing.assert_close(aligned, state["target_values"][0, :2])
        self.memory.model.decode.assert_not_called()

    def test_refuses_wrong_checkpoint_and_public_state(self):
        state = self.memory._load_state()
        state["binding"] = torch.zeros_like(self.memory.binding)
        neural._atomic_write(state, self.memory.state_path, {"version": neural.VERSION})
        with self.assertRaisesRegex(ValueError, "checkpoint binding"):
            self.memory._load_state()
        os.chmod(self.memory.state_path, 0o644)
        with self.assertRaises(PermissionError):
            self.memory._load_state()

    def test_refuses_public_state_directory(self):
        directory = self.memory.state_path.parent
        os.chmod(directory, 0o755)
        with self.assertRaises(PermissionError), self.memory._lock(write=True):
            self.fail("a public state directory must not be admitted")
