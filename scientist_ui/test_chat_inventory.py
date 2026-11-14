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

    def test_natural_wiki_overview_limits_context_to_current_note_excerpts(self):
        notes = [
            {"id": f"local-id-{index}", "title": "Research note", "content": "Fictional context " * 100,
             "status": "active", "source": "/private/source.txt"}
            for index in range(5)
        ]
        self.dream.memory_inventory.return_value["notes"] = [
            {"status": "superseded", "content": "Old material"}, *notes,
        ]
        responses = []
        self.gemini.answer.side_effect = lambda _messages, *, context, wiki_search: (
            responses.append(wiki_search("")) or "Here are the research themes."
        )
        result = self.chat.reply([
            {"role": "user", "content": "OK based on the researchers that we have done togheer what should we learn?"},
            {"role": "assistant", "content": "Which works are you referring to?"},
            {"role": "user", "content": "es the works that we have done in many different areas are the research that we have done look at your wiki"},
        ], "gemini")
        self.assertEqual(result["provider"], "gemini")
        self.assertEqual(result["memory_mode"], "Saved wiki evidence")
        self.assertTrue(result["memory_used"])
        self.assertEqual(len(result["sources"]), 3)
        self.assertEqual(result["sources"][0]["memory_id"], "local-id-0")
        self.assertIn({"step": "Search local wiki", "outcome": "3 matches"}, result["actions"])
        self.assertEqual(responses[0]["notes"][0], {
            "citation": 1, "title": notes[0]["title"], "excerpt": notes[0]["content"][:400],
        })
        self.assertNotIn("/private/", str(responses))
        self.assertNotIn("local-id", str(responses))
        self.assertNotIn("Old material", str(responses))
        self.dream.find_wiki_notes.assert_not_called()

    def test_natural_wiki_lookup_allows_a_topic_from_recent_user_turn(self):
        self.dream.find_wiki_notes.return_value = []
        captured = []
        self.gemini.answer.side_effect = lambda _messages, *, context, wiki_search: (
            captured.append(wiki_search("baseline drift")) or "No matching notes."
        )
        result = self.chat.reply([
            {"role": "user", "content": "We should check the baseline drift."},
            {"role": "assistant", "content": "What about it?"},
            {"role": "user", "content": "Can you look at your wiki for evidence?"},
        ], "gemini")
        self.dream.find_wiki_notes.assert_called_once_with("baseline drift")
        self.dream.memory_inventory.assert_not_called()
        self.assertEqual(captured, [{"status": "no_matches", "notes": []}])
        self.assertIn({"step": "Search local wiki", "outcome": "No matching notes"}, result["actions"])

    def test_ungrounded_topic_or_blank_named_query_cannot_open_general_inventory(self):
        for topic, question in (
            ("something invented", "Look at your wiki and tell me about ZnO"),
            ("", "Look at your wiki and tell me about ZnO"),
            ("", "Look at your wiki and tell me more"),
        ):
            with self.subTest(topic=topic, question=question):
                captured = []
                self.gemini.answer.side_effect = lambda _messages, *, context, wiki_search: (
                    captured.append(wiki_search(topic)) or "Search unavailable."
                )
                self.chat.reply([
                    {"role": "user", "content": "We should check ZnO calibration."},
                    {"role": "assistant", "content": "What about it?"},
                    {"role": "user", "content": question},
                ], "gemini")
                self.assertEqual(captured, [{"status": "unavailable", "notes": []}])
        self.dream.find_wiki_notes.assert_not_called()
        self.dream.memory_inventory.assert_not_called()

    def test_unrelated_question_never_exposes_the_wiki_tool(self):
        self.gemini.answer.return_value = "The sky scatters blue light."
        question = message("Why is the sky blue?")
        self.assertEqual(self.chat.reply(question, "gemini")["answer"], self.gemini.answer.return_value)
        self.gemini.answer.assert_called_once_with(question, context=None)
        self.dream.find_wiki_notes.assert_not_called()
        self.dream.memory_inventory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
