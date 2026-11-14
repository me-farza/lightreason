"""Dream surfaces sparse captures without producing training examples."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
import unittest
from unittest.mock import Mock

from scientist_ui.dream import Dream, PolicyTrainer
from scientist_ui.skill_rehearsal import DAYS


class DreamIntakeTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = Path(temporary.name) / "memory.db"

    def finish(self, dream):
        dream.start()
        deadline = monotonic() + 4
        while monotonic() < deadline:
            report = dream.status()
            if report["status"] != "running":
                self.assertEqual(report["status"], "complete", report.get("error"))
                return report
            sleep(.01)
        self.fail("Dream did not finish")

    def test_sparse_note_gets_follow_up_and_remains_searchable_after_reopen(self):
        trainer = Mock(spec=[])
        dream = Dream(self.db, trainer=trainer)
        note = "Meeting about spectrometer work."
        dream.capture(note, "lab.md", "document")
        report = self.finish(dream)
        self.assertEqual(report["intake"],
                         {"status": "needs_detail", "captured_count": 1, "recognized_lessons": 0,
                          "trainable_lessons": 0})
        follow_up, = report["clarifications"]
        self.assertEqual(follow_up["reason"], "unrecognized")
        self.assertEqual(follow_up["source"], "lab.md")
        self.assertIn("experiment outcome", follow_up["question"])
        self.assertNotIn(note, follow_up["question"])
        self.assertFalse(report["questions"])
        self.assertEqual(report["training"]["optimizer_steps"], 0)
        trainer.assert_not_called()
        self.assertIn("saved the approved notes as evidence", report["events"][-1]["message"])
        reopened = Dream(self.db)
        self.assertEqual(reopened.start(), report)
        self.assertEqual(reopened.recall("spectrometer")[0]["content"], note)

    def test_noncompact_lesson_gets_a_targeted_follow_up(self):
        dream = Dream(self.db)
        dream.capture("The calibration protocol specifies inspecting the cables and then resetting the sensor.", "protocol.md")
        report = self.finish(dream)
        self.assertEqual(report["intake"],
                         {"status": "evidence_only", "captured_count": 1, "recognized_lessons": 1,
                          "trainable_lessons": 0})
        self.assertEqual(report["clarifications"], [{
            "question": "For this procedure, what is the first action you want Vicena to learn?",
            "source": "protocol.md",
            "reason": "not_trainable",
        }])
        self.assertEqual(len(report["questions"]), 1)
        self.assertFalse(report["questions"][0]["neural_eligible"])

    def test_mixed_notes_train_only_from_grounded_examples(self):
        trainer = Mock(spec=[], return_value={"eligible": 1, "optimizer_steps": 4, "promotions": 1})
        dream = Dream(self.db, trainer=trainer)
        dream.capture("For baseline drift, use a fresh reference first.", "procedure.md")
        dream.capture("We talked about the next project.", "meeting.md")
        report = self.finish(dream)
        self.assertEqual(report["intake"],
                         {"status": "ready", "captured_count": 2, "recognized_lessons": 1,
                          "trainable_lessons": 1})
        self.assertEqual([item["source"] for item in report["clarifications"]], ["meeting.md"])
        self.assertEqual([item["source"] for item in trainer.call_args.args[0]], ["procedure.md"])
        self.assertEqual(report["training"]["optimizer_steps"], 4)

        dream.capture("For visual scoring, blind the scorer first.", "procedure.md")
        report = self.finish(dream)
        self.assertEqual(report["clarifications"], [])

    def test_auto_capture_filters_chat_and_keeps_latest_grounded_lessons(self):
        trainer = Mock(spec=[], return_value={"eligible": 1, "promotions": 1})
        planner = Mock(spec=["plan"])
        planner.plan.return_value = {"questions": {}, "wiki": [], "skip": [0, 1]}
        dream = Dream(self.db, trainer=trainer, planner=planner)
        texts = ["Thanks!", "Search the wiki for baseline drift",
                 "Experiment Quartz status is awaiting shipment.",
                 "For baseline drift, use a fresh reference first.",
                 "Correction: for baseline drift, use a sealed blank first."]
        for text in texts:
            dream.capture(text, "Saved chat message")
        report = self.finish(dream)
        self.assertEqual(report["selection"], {"reviewed": 5, "conversation_only": 2,
                                              "excluded_lessons": 2, "training_lessons": 1})
        self.assertEqual(report["counts"]["pending"], 0)
        self.assertEqual(report["counts"]["promoted"], 3)
        sent, _ = planner.plan.call_args.args
        self.assertEqual([entry["text"] for entry in sent], texts[2:])
        self.assertEqual([fact["answer"] for fact in trainer.call_args.args[0] if fact["neural_eligible"]],
                         ["use a sealed blank first"])
        reopened = Dream(self.db)
        self.assertEqual({entry["content"] for entry in reopened.wiki()}, set(texts[2:]))
        self.assertEqual(reopened.status()["selection"], report["selection"])
        current, = reopened.find_wiki_notes("baseline drift")
        self.assertEqual(current["content"], texts[-1])
        self.assertEqual(reopened.find_wiki_notes("unknown microscope"), [])

    def test_planner_proposals_require_a_verbatim_decision_from_a_queued_note(self):
        planner = Mock(spec=["plan"])
        trainer = Mock(spec=[], return_value={"eligible": 1, "promotions": 1})
        dream = Dream(self.db, planner=planner, trainer=trainer)
        note = "Our standard microscope warmup practice is to wait ten minutes."
        capture = dream.capture(note, "Training notes", "document")
        tentative = "For camera calibration we might always use a gray card."
        unconfirmed = dream.capture(tentative, "Open questions", "document")
        planner.plan.return_value = {"questions": {}, "wiki": [], "lessons": [
            {"capture_id": capture, "category": "process", "key": "microscope warmup",
             "answer": "wait ten minutes", "evidence": note},
            {"capture_id": unconfirmed, "category": "process", "key": "camera calibration",
             "answer": "use a gray card", "evidence": tentative},
        ]}
        report = self.finish(dream)
        question, = report["questions"]
        self.assertEqual(question["answer"], "wait ten minutes")
        self.assertEqual(question["observation"], note)
        self.assertEqual(question["origin"], "gemini_extraction")
        self.assertEqual([row["source"] for row in report["clarifications"]], ["Open questions"])
        self.assertEqual({note["content"] for note in dream.wiki()}, {note, tentative})
        self.assertTrue(all(entry["may_extract"] for entry in planner.plan.call_args.args[0]))

    def test_chat_only_batch_never_calls_planner_or_trainer(self):
        trainer, planner = Mock(spec=[]), Mock(spec=["plan"])
        dream = Dream(self.db, trainer=trainer, planner=planner)
        dream.capture("Hello!", "Saved chat message")
        report = self.finish(dream)
        self.assertEqual(report["selection"]["conversation_only"], 1)
        self.assertEqual(report["counts"]["pending"], 0)
        self.assertEqual(report["counts"]["promoted"], 0)
        self.assertFalse(report["clarifications"])
        self.assertEqual(dream.wiki(), [])
        trainer.assert_not_called()
        planner.plan.assert_not_called()

    def test_empty_and_scripted_rehearsal_do_not_suggest_a_clarification(self):
        dream = Dream(self.db)
        self.assertEqual(dream.start()["night"], 0)
        self.assertIsNone(dream.status()["intake"])
        self.assertEqual(dream.status()["clarifications"], [])

        dream.capture("\n".join(DAYS[0]), "scripted")
        report = self.finish(dream)
        self.assertIsNone(report["intake"])
        self.assertEqual(report["clarifications"], [])

    def test_legacy_report_gets_guidance_without_reprocessing(self):
        dream = Dream(self.db)
        dream.capture("Meeting to discuss the reference.", "older-note.md", "document")
        legacy = self.finish(dream)
        legacy.pop("intake")
        legacy.pop("clarifications")
        with dream.store._connect(write=True) as db:
            db.execute("UPDATE dream_run SET payload = ? WHERE id = 1", (json.dumps(legacy),))

        reopened = Dream(self.db)
        report = reopened.start()
        self.assertEqual(report["intake"]["status"], "needs_detail")
        self.assertEqual(report["clarifications"][0]["source"], "older-note.md")
        self.assertEqual(report["counts"]["pending"], 0)
        self.assertEqual(report["night"], legacy["night"])
        self.assertEqual(report["questions"], [])
        with dream.store._connect() as db:
            self.assertEqual(json.loads(db.execute("SELECT payload FROM dream_run WHERE id = 1").fetchone()[0]), legacy)

    def test_live_policy_snapshots_replace_the_same_candidate(self):
        observed = []

        class Trainer(PolicyTrainer):
            def __call__(self, examples, progress, *, on_policy_run):
                run = {"candidate_id": 1, "key": "baseline drift", "status": "training",
                       "optimizer_steps": 0, "plan": [{"question": examples[0]["question"]}]}
                on_policy_run(run)
                observed.append(dream.status()["training"])
                on_policy_run({**run, "optimizer_steps": 8})
                observed.append(dream.status()["training"])
                finish = {**run, "optimizer_steps": 9, "status": "held"}
                on_policy_run(finish)
                observed.append(dream.status()["training"])
                return {"eligible": 1, "optimizer_steps": 9, "promotions": 0, "policy_runs": [finish]}

            def __init__(self):
                pass

            def skill_checkpoint_digest(self):
                return None

        dream = Dream(self.db, trainer=Trainer())
        dream.capture("For baseline drift, use a fresh reference first.", "lesson.md")
        report = self.finish(dream)
        self.assertEqual([snapshot["optimizer_steps"] for snapshot in observed], [0, 8, 9])
        self.assertTrue(all(len(snapshot["policy_runs"]) == 1 for snapshot in observed))
        self.assertTrue(all(snapshot["promotions"] == 0 for snapshot in observed))
        self.assertEqual(report["training"]["policy_runs"][0]["status"], "held")
        self.assertEqual(len(report["questions"]), 1)


if __name__ == "__main__":
    unittest.main()
