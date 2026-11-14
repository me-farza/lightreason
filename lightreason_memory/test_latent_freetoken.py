"""CPU tests for the 35B prefill row substitution and request boundary."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
    from safetensors.torch import save_file

    from . import latent_freetoken_hook as hook
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch and safetensors are required")
class LatentPrefillTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.pending = self.root / "pending.safetensors"
        self.receipt = self.root / "injected.json"
        env = patch.dict("os.environ", {hook.INGRESS_ENV: str(self.root)})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(hook._read_sidecar.cache_clear)
        self.ids = [11, hook.MARKER_ID, hook.MARKER_ID, 13]
        self.payload = torch.stack([
            torch.full((hook.HIDDEN_SIZE,), 1.5, dtype=torch.bfloat16),
            torch.full((hook.HIDDEN_SIZE,), -2.25, dtype=torch.bfloat16),
        ])
        save_file({"embeddings": self.payload}, self.pending, metadata={
            "request_id": "test-payload", "prompt_ids": json.dumps(self.ids),
        })

    def batch(self, start, stop, *, prefill=True, uid=1, ids=None):
        ids = self.ids if ids is None else ids
        return SimpleNamespace(is_prefill=prefill, reqs=[SimpleNamespace(
            cached_len=start, device_len=stop, input_ids=torch.tensor(ids[:stop]),
            prompt_len=len(ids), uid=uid,
        )])

    @staticmethod
    def base(rows):
        return torch.full((rows, hook.HIDDEN_SIZE), 7, dtype=torch.bfloat16)

    def test_substitutes_prompt_rows_only_and_consumes_payload(self):
        out = hook.inject_memory_embeddings(self.base(4), self.batch(0, 4))
        torch.testing.assert_close(out[1:3], self.payload, atol=0, rtol=0)
        torch.testing.assert_close(out[[0, 3]], self.base(2), atol=0, rtol=0)
        self.assertFalse(self.pending.exists())
        self.assertEqual(json.loads(self.receipt.read_text()), {
            "request_id": "test-payload", "uid": 1, "positions": [1, 2],
            "sha256": hook.embedding_digest(self.payload),
        })

    def test_prefill_without_memory_marker_is_identity(self):
        native = self.base(2)
        self.assertIs(hook.inject_memory_embeddings(native, self.batch(0, 2, ids=[11, 13])), native)
        self.assertTrue(self.pending.exists())
        self.assertFalse(self.receipt.exists())

    def test_chunked_prefill_before_decode(self):
        first = hook.inject_memory_embeddings(self.base(2), self.batch(0, 2))
        self.assertTrue(self.pending.exists())
        torch.testing.assert_close(first[1], self.payload[0], atol=0, rtol=0)
        decode = self.base(2)
        self.assertIs(hook.inject_memory_embeddings(decode, self.batch(0, 2, prefill=False)), decode)
        second = hook.inject_memory_embeddings(self.base(2), self.batch(2, 4))
        torch.testing.assert_close(second[0], self.payload[1], atol=0, rtol=0)
        self.assertFalse(self.pending.exists())

    def test_rejects_another_request_and_skipped_rows(self):
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            hook.inject_memory_embeddings(self.base(4), self.batch(0, 4, ids=[12, *self.ids[1:]]))
        with self.assertRaisesRegex(RuntimeError, "replayed or skipped"):
            hook.inject_memory_embeddings(self.base(2), self.batch(2, 4))
        hook.inject_memory_embeddings(self.base(2), self.batch(0, 2))
        with self.assertRaisesRegex(RuntimeError, "different request"):
            hook.inject_memory_embeddings(self.base(2), self.batch(2, 4, uid=2))
