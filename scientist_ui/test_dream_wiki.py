"""Keep Gemini augmentation and local evidence provenance linked through Dream."""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from unittest import TestCase
from unittest.mock import Mock

from scientist_ui.dream import Dream


class WikiTests(TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "memory.sqlite3"

    def finish(self, dream):
        report = dream.start()
        deadline = monotonic() + 4
        while report["status"] == "running" and monotonic() < deadline:
            sleep(.01)
            report = dream.status()
        self.assertEqual(report["status"], "complete", report)
        return report

    def test_wiki_keeps_exact_text_and_links_corrections_across_restart(self):
        original = "For baseline drift, use a fresh reference first."
        revised = "Correction: for baseline drift, use a sealed blank first."
        dream = Dream(self.path)
        dream.capture(original, "Day 1")
        self.finish(dream)
        self.assertEqual(dream.wiki()[0]["organized_by"], "local")

        planner, trainer = Mock(spec=["plan"]), Mock(spec=[])
        trainer.return_value = {"eligible": 1, "promotions": 0}
        dream = Dream(self.path, trainer=trainer, planner=planner)
        capture = dream.capture(revised, "Day 2")
        pair = ["Which control should I choose first for baseline drift?",
                "On a new detector with baseline drift, what would you run first?"]
        planner.plan.return_value = {"questions": {0: pair}, "wiki": [
            {"capture_id": capture, "title": "baseline drift", "summary": revised, "tags": ["correction"]},
        ]}
        report = self.finish(dream)
        self.assertEqual(report["planning"]["proposed_questions"], 2)
        self.assertEqual([entry["text"] for entry in planner.plan.call_args.args[0]], [revised])
        self.assertEqual(trainer.call_args.args[0][0]["question_variants"], pair)
        self.assertEqual(trainer.call_args.args[0][0]["answer"], "use a sealed blank first")

        current, previous = Dream(self.path).wiki()
        self.assertEqual(current["organized_by"], "Gemini")
        self.assertEqual(current["content"], revised)
        self.assertEqual(current["source"], "Day 2")
        self.assertEqual(previous["content"], original)
        self.assertEqual(current["supersedes_id"], previous["id"])
        self.assertEqual(previous["superseded_by_id"], current["id"])
        self.assertEqual(previous["status"], "superseded")
        self.assertEqual(set(current["tags"]) & set(previous["tags"]), {"baseline drift", "process"})
        self.assertEqual(dream.start()["night"], 2)
        planner.plan.assert_called_once()

    def test_provider_failure_still_indexes_local_evidence(self):
        planner = Mock(spec=["plan"])
        planner.plan.side_effect = RuntimeError("provider temporarily unavailable")
        dream = Dream(self.path, planner=planner)
        text = "Our experiment camera response status is awaiting review."
        dream.capture(text, "Camera log", "document")
        with self.assertLogs("scientist_ui.dream", level="WARNING"):
            report = self.finish(dream)
        self.assertEqual(report["planning"]["status"], "fallback")
        note, = dream.wiki()
        self.assertEqual(note["content"], text)
        self.assertEqual(note["organized_by"], "local")
        self.assertEqual(note["category"], "experiment")

    def test_memory_inventory_counts_all_versions_and_lists_current_notes_only(self):
        dream = Dream(self.path)
        self.assertEqual(dream.memory_inventory(), {
            "active_count": 0, "superseded_count": 0, "pending_count": 0, "notes": [],
        })
        for number in range(10):
            dream.capture(f"For procedure {number}, use a fresh reference first.", "Lab note")
        self.finish(dream)
        dream.capture("Correction: for procedure 2, use a sealed blank first.", "Correction")
        self.finish(dream)
        dream.capture("For procedure 11, check the reference first.", "Waiting note")

        inventory = Dream(self.path).memory_inventory()
        self.assertEqual(inventory["active_count"], 10)
        self.assertEqual(inventory["superseded_count"], 1)
        self.assertEqual(inventory["pending_count"], 1)
        self.assertEqual(len(inventory["notes"]), 8)
        self.assertTrue(all(note["status"] == "active" for note in inventory["notes"]))
        self.assertEqual(inventory["notes"][0]["source"], "Correction")
        self.assertIn("sealed blank", inventory["notes"][0]["summary"])
