"""Tests for the local-only saved-memory inventory reply."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from scientist_ui.chat import LocalChat


def message(content):
    return [{"role": "user", "content": content}]


class ChatInventoryTests(unittest.TestCase):
    def setUp(self):
        working = TemporaryDirectory()
        self.addCleanup(working.cleanup)
        root = Path(working.name)
        self.dream = Mock()
        self.gemini = Mock(model="fake-gemini")
        self.readout = Mock(return_value=None)
        self.vectors = Mock()
        self.chat = LocalChat(
            root / "memory.db", dream=self.dream, gemini=self.gemini, memory_readout=self.readout,
            model_35b=root, stage_35b=root, vectors=self.vectors, openrouter_model="model", openrouter_api_key="dummy",
        )
        self.addCleanup(self.chat.close)
        self.chat._answer_35b = Mock()
        self.chat._answer_remote = Mock()
        self.dream.memory_inventory.return_value = {
            "active_count": 2, "superseded_count": 1, "pending_count": 1,
            "notes": [
                {"id": "secret-opaque-id", "title": "Calibration protocol", "summary": "Use a sealed blank first.",
                 "status": "active", "source": "/private/lab/calibration.txt"},
                {"id": "old-opaque-id", "title": "Old calibration", "summary": "Outdated step.", "status": "superseded"},
            ],
        }

    def test_exact_typo_reports_inventory_without_inference_or_disclosure_in_local_routes(self):
        for mode in ("gemini", "vector", "text"):
            with self.subTest(mode=mode):
                result = self.chat.reply(message("tell me whaich memories do you have?"), mode)
                self.assertEqual(result["provider"], "local")
                self.assertEqual(result["model"], "Vicena")
                self.assertTrue(result["local_only"])
                self.assertTrue(result["memory_used"])
                self.assertEqual(result["injected_rows"], 0)
                self.assertIsNone(result["memory_readout"])
                self.assertIn("2 active saved notes, 1 older version, and 1 waiting for Dream.", result["answer"])
                self.assertIn("Showing 1 of 2", result["answer"])
                self.assertIn("Use a sealed blank first.", result["answer"])
                for fragment in ("Old calibration", "opaque-id", "/private/"):
                    self.assertNotIn(fragment, result["answer"])
        self.assertEqual(self.dream.memory_inventory.call_count, 3)
        self.gemini.answer.assert_not_called()
        self.readout.assert_not_called()
        self.vectors.assert_not_called()
        self.chat._answer_35b.assert_not_called()
        self.chat._answer_remote.assert_not_called()

    def test_explicit_inventory_aliases_report_empty_and_pending(self):
        self.dream.memory_inventory.return_value = {
            "active_count": 0, "superseded_count": 0, "pending_count": 2, "notes": [],
        }
        for question in (
            "What memories do you have?", "Which memories do you have?", "Memory inventory",
            "Could you list my memories, please?", "Show my memories", "What do you remember about me?",
        ):
            with self.subTest(question=question):
                result = self.chat.reply(message(question), "gemini")
                self.assertTrue(result["local_only"])
                self.assertFalse(result["memory_used"])
                self.assertIn("0 active saved notes, 0 older versions, and 2 waiting for Dream.", result["answer"])
        self.gemini.answer.assert_not_called()

    def test_specific_research_questions_still_use_gemini(self):
        self.gemini.answer.return_value = "Normal research response."
        for question in (
            "What do you remember about baseline drift?",
            "Which memories do you have about the zinc oxide run?",
            "List my memories about the test before we review the method.",
        ):
            with self.subTest(question=question):
                result = self.chat.reply(message(question), "gemini")
                self.assertEqual(result["provider"], "gemini")
                self.assertEqual(result["answer"], "Normal research response.")
                self.gemini.answer.assert_called_with(message(question), context=None)
        self.dream.memory_inventory.assert_not_called()

    def test_memory_off_and_openrouter_keep_existing_reply_paths(self):
        question = message("Which memories do you have?")
        self.chat._answer_35b.return_value = ("Normal memory-off response.", [], None)
        self.chat._answer_remote.return_value = {"answer": "Normal OpenRouter response."}
        self.assertEqual(self.chat.reply(question, "off")["answer"], "Normal memory-off response.")
        self.assertEqual(self.chat.reply(question, "remote")["answer"], "Normal OpenRouter response.")
        self.chat._answer_35b.assert_called_once_with(question, [], None)
        self.chat._answer_remote.assert_called_once_with(question)
        self.dream.memory_inventory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
