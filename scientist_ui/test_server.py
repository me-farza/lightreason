"""Regression tests for the scoped, persistent notebook data behind the UI."""

from http import HTTPStatus
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock
from uuid import uuid4

from lightreason_memory.scientist_fixtures import TIO2_UNSEEN_VARIATION, ZNO_WORKFLOW

from scientist_ui.chat import LocalChat
from scientist_ui.dream import Dream
from scientist_ui.server import Notebook, NotebookHandler, _verified_restart_checks
from scientist_ui.skill_rehearsal import CHECKS


class NotebookTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "notebook.sqlite3"
        self.notebook = Notebook(self.path)
        self.zno = ZNO_WORKFLOW.workflow_id
        self.tio2 = TIO2_UNSEEN_VARIATION.workflow_id

    def test_correction_is_idempotent_scoped_and_persistent(self):
        for workflow, old_temp, new_temp in ((self.zno, "350 C", "425 C"), (self.tio2, "480 C", "500 C")):
            with self.subTest(workflow=workflow):
                initial = self.notebook.snapshot(workflow)
                initial_id = initial["protocol"]["memory_id"]
                self.assertFalse(initial["correction"]["applied"])
                self.assertEqual([step["number"] for step in initial["steps"]], list(range(1, 6)))
                self.assertIn(old_temp, initial["steps"][3]["text"])
                current = self.notebook.correct(workflow)
                current_id = current["protocol"]["memory_id"]
                self.assertTrue(current["correction"]["applied"])
                self.assertEqual(current["protocol"]["supersedes_id"], initial_id)
                self.assertIn(new_temp, current["steps"][3]["text"])
                self.assertEqual(self.notebook.correct(workflow), current)
                reopened = Notebook(self.path)
                self.assertEqual(reopened.snapshot(workflow), current)
                self.assertEqual(reopened.snapshot(workflow, "initial")["protocol"]["memory_id"], initial_id)
                with self.assertRaises(KeyError):
                    reopened.evidence(workflow, "current", initial_id)
                with self.assertRaises(KeyError):
                    reopened.evidence(workflow, "initial", current_id)
        with self.assertRaises(KeyError):
            self.notebook.evidence(self.tio2, "current", self.notebook.snapshot(self.zno)["protocol"]["memory_id"])

    def test_lookup_distinguishes_evidence_from_lexical_matches(self):
        preset = "explicit_lookup_current"
        before = self.notebook.recall(self.zno, "current", query=preset)
        self.assertEqual(before["status"], "insufficient")
        self.assertFalse(before["sources"])

        current = self.notebook.correct(self.zno)
        result = self.notebook.recall(self.zno, "current", query=preset)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["sources"], [current["protocol"]])
        self.assertIn("425 C", result["answer"])
        historic = self.notebook.recall(self.zno, "initial", query="explicit_lookup_historical")
        self.assertIn("350 C", historic["answer"])
        self.assertNotIn("425 C", historic["answer"])
        with self.assertRaises(ValueError):
            self.notebook.recall(self.zno, "initial", query=preset)

        missing = self.notebook.recall(self.zno, "current", query="absent_hall_mobility")
        self.assertEqual(missing["status"], "insufficient")
        self.assertFalse(missing["sources"])
        search = self.notebook.recall(self.zno, "current", question="pinhole")
        self.assertEqual(search["status"], "matches")
        unique_miss = self.notebook.recall(self.zno, "current", question="absentmeasurementkeyword")
        self.assertEqual(unique_miss["status"], "insufficient")
        self.assertFalse(unique_miss["sources"])


class ChatCaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "chat.sqlite3"
        self.dream = Dream(self.path)
        self.chat = LocalChat(self.path, model_path="unused", dream=self.dream)
        self.addCleanup(self.chat.close)
        self.chat.reply = Mock(side_effect=RuntimeError("inference timeout"))
        self.handler = NotebookHandler.__new__(NotebookHandler)
        self.handler.server = Mock(chat=self.chat, run_id=uuid4().hex)
        self.handler._json = Mock()
        self.handler.log_error = Mock()
        self.request_id = str(uuid4())
        self.request = {"messages": [{"role": "user", "content": "Approved rehearsal lesson."}],
                        "mode": "text", "save": True, "capture_request_id": self.request_id}

    def test_validated_capture_survives_failed_reply_and_restart_retry(self):
        def fail_after_save(messages, *, mode):
            self.assertEqual(self.dream.status()["counts"]["pending"], 1)
            raise RuntimeError("inference timeout")

        self.chat.reply.side_effect = fail_after_save
        self.handler._chat(self.request)
        response, status = self.handler._json.call_args.args
        self.assertEqual(status, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertEqual(response["capture_id"], self.request_id)
        self.assertIn("saved for Dream", response["error"])

        self.chat.dream = Dream(self.path)
        self.chat.reply.side_effect = None
        self.chat.reply.return_value = {"answer": "Ready."}
        self.handler._chat(self.request)
        self.assertEqual(self.handler._json.call_args.args[0]["capture_id"], self.request_id)
        self.assertEqual(self.chat.dream.status()["counts"]["pending"], 1)

    def test_invalid_chat_does_not_capture_or_generate(self):
        with self.assertRaises(ValueError):
            self.handler._chat({**self.request, "messages": [{"role": "assistant", "content": "Invalid"}]})
        with self.assertRaises(ValueError):
            self.handler._chat({**self.request, "mode": "vector"})
        with self.assertRaises(ValueError):
            self.handler._chat({**self.request, "capture_request_id": "malformed"})
        with self.assertRaises(ValueError):
            self.handler._chat({**self.request, "save": False})
        with self.assertRaises(ValueError):
            self.handler._chat({**self.request, "save": "true"})
        self.chat.reply.assert_not_called()
        self.assertEqual(self.dream.status()["counts"]["pending"], 0)

    def test_capture_defaults_on_in_all_chat_modes_with_explicit_opt_out(self):
        self.chat.remote_available = True
        self.chat.reply.side_effect = None
        self.chat.reply.return_value = {"answer": "Ready."}
        for index, mode in enumerate(("text", "remote"), start=1):
            request = {"messages": self.request["messages"], "mode": mode}
            self.handler._chat(request)
            self.assertTrue(self.handler._json.call_args.args[0]["capture_id"])
            self.assertEqual(self.dream.wiki_snapshot(), ([], index))
            self.handler._chat({**request, "save": False})
            self.assertIsNone(self.handler._json.call_args.args[0]["capture_id"])
            self.assertEqual(self.dream.status()["counts"]["pending"], index)

    def test_direct_note_capture_is_local_and_idempotent(self):
        note = {"text": "When aligning the demo optics, check the reference first.",
                "source": "Synthetic Dream sample", "capture_request_id": self.request_id}
        self.handler._save_for_dream(note)
        self.assertEqual(self.handler._json.call_args.args[0]["memory_id"], self.request_id)
        self.chat.dream = Dream(self.path)
        self.handler._save_for_dream(note)
        self.assertEqual(self.handler._json.call_args.args[0]["memory_id"], self.request_id)
        with self.assertRaises(ValueError):
            self.handler._save_for_dream({**note, "text": "A different approved note."})
        with self.assertRaises(ValueError):
            self.handler._save_for_dream({**note, "capture_request_id": "bad"})
        self.assertEqual(self.chat.dream.status()["counts"]["pending"], 1)
        self.chat.reply.assert_not_called()


class RehearsalProofTests(unittest.TestCase):
    def test_proof_checks_restarted_service_checkpoint_and_vector_receipts(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "rehearsal.json"
            checkpoint, vector = "1" * 64, "2" * 64
            rows = []
            for case in CHECKS:
                action = case["day_two"]
                count = int(action is not None)
                rows.append({"id": case["id"], "question": case["question"], "expected": action,
                             "actual": action, "passed": True, "answer": "Synthetic fixture reply.",
                             "injected_rows": count, "vector_rows": count, "vector_sha256": vector if count else None,
                             "receipt": {"rows": count, "sha256": vector} if count else None})
            before, after = uuid4().hex, uuid4().hex
            proof = {"verified": True, "checkpoint_sha256": checkpoint, "before_run_id": before,
                     "after_run_id": after, "restart_checks": rows}
            path.write_text(json.dumps(proof))
            self.assertEqual(_verified_restart_checks(path, checkpoint, CHECKS), rows)
            self.assertIsNone(_verified_restart_checks(path, "3" * 64, CHECKS))
            path.write_text(json.dumps({**proof, "after_run_id": before}))
            self.assertIsNone(_verified_restart_checks(path, checkpoint, CHECKS))
            rows[0]["receipt"]["sha256"] = "3" * 64
            path.write_text(json.dumps(proof))
            self.assertIsNone(_verified_restart_checks(path, checkpoint, CHECKS))
            path.write_text("{invalid")
            self.assertIsNone(_verified_restart_checks(path, checkpoint, CHECKS))


class GeminiChatTests(unittest.TestCase):
    def test_gemini_reads_only_the_current_question_and_never_starts_35b(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "gemini.sqlite3"
            reasoner = Mock(model="gemini-test")
            reasoner.answer.return_value = "Try a sealed blank first."
            reader = Mock(side_effect=["Use a sealed blank.", None])
            chat = LocalChat(path, dream=Dream(path), model_35b=path.parent, stage_35b=path.parent,
                             vectors=Mock(side_effect=AssertionError("Unexpected vector handoff")),
                             gemini=reasoner, memory_readout=reader)
            self.addCleanup(chat.close)
            self.assertEqual(chat.default_mode, "gemini")
            messages = [{"role": "user", "content": "My setup is a humidity rig."},
                        {"role": "assistant", "content": "Understood."},
                        {"role": "user", "content": "Which control should I use?"}]
            reply = chat.reply(messages, mode="gemini")
            reader.assert_called_once_with(messages[-1]["content"])
            reasoner.answer.assert_called_once_with(messages, context="Use a sealed blank.")
            self.assertEqual(reply["provider"], "gemini")
            self.assertEqual(reply["memory_readout"], "Use a sealed blank.")
            self.assertTrue(reply["memory_used"])
            self.assertEqual(reply["sources"], [])
            self.assertEqual(reply["injected_rows"], 0)
            self.assertIsNone(reply["vector_receipt"])
            reply = chat.reply(messages, mode="gemini")
            reasoner.answer.assert_called_with(messages, context=None)
            self.assertIsNone(reply["memory_readout"])
            self.assertFalse(reply["memory_used"])
            self.assertEqual(chat.local_status, "cold")
            self.assertIsNone(chat._reasoner)


    def test_wiki_intent_executes_locally_and_includes_the_receipt(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "gemini.sqlite3"
            dream = Mock(spec=Dream)
            note = {"id": "note-id", "title": "Baseline drift", "source": "Lab note",
                    "content": "For baseline drift, use a sealed blank first.", "summary": "Use a sealed blank first."}
            dream.find_wiki_notes.side_effect = [[note], []]
            reasoner, reader = Mock(model="gemini-test"), Mock()
            chat = LocalChat(path, dream=dream, gemini=reasoner, memory_readout=reader)
            try:
                result = chat.reply([{"role": "user", "content": "Search the wiki for baseline drift."}], mode="gemini")
                self.assertTrue(result["local_only"])
                self.assertTrue(result["memory_used"])
                self.assertEqual(result["actions"], [{"step": "Search local wiki", "outcome": "1 match"}])
                self.assertEqual(result["sources"][0]["memory_id"], note["id"])
                dream.find_wiki_notes.assert_called_once_with("baseline drift")
                result = chat.reply([{"role": "user", "content": "Show saved notes about no match"}], mode="gemini")
                self.assertTrue(result["local_only"])
                self.assertFalse(result["memory_used"])
                self.assertEqual(result["sources"], [])
                self.assertIn("no current wiki notes", result["answer"])
                reader.assert_not_called()
                reasoner.answer.assert_not_called()
            finally:
                chat.close()
