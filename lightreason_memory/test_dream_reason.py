"""Exercise the memory controller's request routing without loading a GPU model."""

from contextlib import nullcontext
from importlib.util import find_spec
import unittest
from unittest.mock import patch

from demo_chat.messages import PromptText, chat_messages

from .dream_reason import answer_with_memory


_HAS_TENSORS = all(find_spec(name) is not None for name in ("torch", "safetensors"))
_PATHS = dict(model_3b="3b", codec="codec", policy="policy", state="state", model_35b="35b", stage_35b="stage")


@unittest.skipUnless(_HAS_TENSORS, "PyTorch and safetensors are required")
class DreamReasonTests(unittest.TestCase):
    def setUp(self):
        def start_patch(target, **kwargs):
            patcher = patch(target, **kwargs)
            self.addCleanup(patcher.stop)
            return patcher.start()

        start_patch("lightreason_memory.remote.comparison_lease", return_value=nullcontext())
        self.memory = start_patch("lightreason_memory.dream_reason._memory_vectors")
        self.normal = start_patch("lightreason_memory.remote.Reasoner35B")
        self.aligned = start_patch("lightreason_memory.latent_freetoken.LatentReasoner35B")

    def test_miss_passes_original_prompt_and_budget_to_ordinary_runner(self):
        request = PromptText("Explain this fresh problem.\n", "Use an example from arithmetic.")
        self.memory.return_value = ({"status": "unknown"}, None, None, .1)
        response = answer_with_memory(request, **_PATHS, reasoner_tokens=37)

        self.assertFalse(response["memory_injected"])
        self.assertEqual(self.memory.call_args.kwargs["dream_every"], 0)
        forwarded, budget = self.normal.return_value.answer.call_args.args
        self.assertIs(forwarded, request)
        self.assertEqual(chat_messages(forwarded), chat_messages(request))
        self.assertEqual(budget, 37)
        self.aligned.assert_not_called()
        self.normal.return_value.close.assert_called_once()

    def test_hit_passes_rows_and_preserves_caller_instructions(self):
        from .latent_freetoken import PLACEHOLDER

        request = PromptText("What color was the lantern?", "Answer with a single word.")
        rows = [object()]
        self.memory.return_value = ({"status": "supported"}, rows, "receipt", .1)
        self.aligned.return_value.answer_with_vectors.return_value = {
            "injected_rows": 1, "memory_embedding_sha256": "receipt",
        }
        result = answer_with_memory(request, **_PATHS, memory_query="lantern color")

        self.assertTrue(result["memory_injected"])
        self.assertEqual(self.memory.call_args.args, ("lantern color",))
        prompt, budget, handed_off = self.aligned.return_value.answer_with_vectors.call_args.args
        self.assertEqual(prompt.system_prompt, request.system_prompt)
        self.assertTrue(prompt.endswith(request))
        self.assertEqual(prompt.count(PLACEHOLDER), 1)
        self.assertEqual(budget, 128)
        self.assertIs(handed_off, rows)
        self.normal.assert_not_called()
