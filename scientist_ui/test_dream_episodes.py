"""Regression checks for local Dream lessons distilled from approved notes."""

import unittest

from scientist_ui.dream import _CORRECTION, _episodes


def episodes(text):
    return list(_episodes({"id": "capture", "source": "lab.md", "text": text}))


class DreamEpisodeTests(unittest.TestCase):
    def test_correction_matches_the_named_procedure_and_keeps_independent_action(self):
        drift, scoring = episodes(
            "For baseline drift, use a fresh reference first.\n"
            "For visual scoring, blind the scorer first."
        )
        corrected, = episodes("- Correction: when baseline drift occurs, use a sealed blank first.")
        trailing_scope, = episodes("Correction: use a sealed blank first for baseline drift.")
        self.assertEqual(drift["key"], corrected["key"])
        self.assertEqual(corrected["key"], trailing_scope["key"])
        self.assertEqual(corrected["answer"], trailing_scope["answer"])
        self.assertNotEqual(drift["key"], scoring["key"])
        self.assertEqual(drift["answer"], "use a fresh reference first")
        self.assertEqual(corrected["answer"], "use a sealed blank first")
        self.assertEqual(scoring["answer"], "blind the scorer first")
        for lesson in (drift, corrected, scoring):
            self.assertEqual(lesson["capture_id"], "capture")
            self.assertEqual(lesson["source"], "lab.md")
            self.assertEqual(lesson["training_kind"], "process")
            self.assertTrue(lesson["neural_eligible"])

    def test_typed_markdown_headings_and_plain_text_can_scope_facts(self):
        status, anneal, preference = episodes(
            "## Experiment: ZnO Run A ###\n"
            "- Status: awaiting XRD review\n"
            "- Waiting on the beamline team to assign a slot.\n"
            "## Notes on meeting\n"
            "- Status: delayed\n"
            "Procedure: ZnO anneal\n"
            "1. Use 425 C.\n"
            "# Preferences\n"
            "* I prefer short checklists for lab notes.\n"
            "- Use 500 C.\n"
        )
        self.assertEqual(
            [(x["category"], x["key"], x["answer"]) for x in (status, anneal, preference)],
            [("experiment", "ZnO Run A", "awaiting XRD review"),
             ("process", "ZnO anneal", "Use 425 C"),
             ("preference", "lab notes", "short checklists")],
        )
        self.assertEqual(status["question"], "What is the status of experiment ZnO Run A?")
        self.assertTrue(all(x["neural_eligible"] for x in (status, anneal, preference)))

    def test_incomplete_ambiguous_or_code_content_does_not_propose_lessons(self):
        self.assertEqual(episodes(
            "```text\n"
            "For visual scoring, blind the scorer first.\n"
            "```\n"
            "~~~~\n"
            "My lantern is gray.\n"
            "```\n"
            "For baseline drift, use a fresh reference first.\n"
            "~~~~\n"
            "For baseline drift, the reference may fail first.\n"
            "For baseline drift, maybe use a different sensor first.\n"
            "For example, use a fresh reference first.\n"
            "For instance, use a different probe.\n"
            "If we wanted to test a different approach, use a blank.\n"
            "For baseline drift, should we use a fresh reference first?\n"
            "For baseline drift, use a fresh reference first. Measure the check twice.\n"
            "# Procedure: multistep setup\n"
            "1. Set 50 C.\n"
            "2. Cover the samples.\n"
            "# Experiment: conflicting statuses\n"
            "Status: cancelled\n"
            "Status: on track\n"
        ), [])

    def test_natural_unscoped_preference_and_named_experiment_status(self):
        preference, specific, status = episodes(
            "I prefer short checklists.\n"
            "I prefer handwritten reminders for lab notes.\n"
            "The ZnO experiment is complete."
        )
        self.assertEqual(
            [(lesson["key"], lesson["answer"]) for lesson in (preference, specific, status)],
            [("general", "short checklists"), ("lab notes", "handwritten reminders"), ("ZnO", "complete")],
        )
        self.assertEqual(preference["question"], "What is my general preference?")
        self.assertEqual(status["question"], "What is the status of experiment ZnO?")
        self.assertTrue(all(lesson["neural_eligible"] for lesson in (preference, specific, status)))
        self.assertEqual(episodes(
            "I measured the ZnO film thickness at 15 nm.\n"
            "The ZnO experiment is something we could try.\n"
            "The ZnO experiment is complete nonsense."
        ), [])

    def test_natural_procedure_records_a_clear_intent_or_single_directive(self):
        declared, always, markdown = episodes(
            "When the baseline drifts, we use a sealed blank first.\n"
            "For baseline drift, always use a sealed blank first.\n"
            "# Procedure: visual scoring\n"
            "- Blind the scorer first.\n"
            "This prevents expectation bias."
        )
        self.assertEqual(
            [(lesson["key"], lesson["answer"]) for lesson in (declared, always, markdown)],
            [("baseline drift", "we use a sealed blank first"),
             ("baseline drift", "always use a sealed blank first"), ("visual scoring", "Blind the scorer first")],
        )
        self.assertTrue(all(lesson["neural_eligible"] for lesson in (declared, always, markdown)))
        for note in (
            "# Procedure: probe clean-up\n- Use a clean wipe.\n- Do not wipe the calibration mark.",
            "# Procedure: probe clean-up\n- Use a clean wipe.\nThis only works after calibration.",
            "# Procedure: probe clean-up\n- Use a clean wipe.\nThis helps and then we check the baseline.",
            "For baseline drift, we might use a sealed blank first.",
            "For baseline drift, we never use a sealed blank first.",
        ):
            with self.subTest(note=note):
                self.assertEqual(episodes(note), [])

    def test_preserves_color_absorb_and_explicit_legacy_procedure(self):
        color, anneal, profile = episodes(
            "Correction: my lantern is gray.\n"
            "The ZnO anneal schedule specifies 425C at 3C/min.\n"
            "My role is lab manager."
        )
        self.assertEqual((color["question"], color["observation"], color["training_kind"]),
                         ("What color is my lantern?", "my lantern is gray.", "color"))
        self.assertEqual(anneal["answer"], "425C at 3C/min")
        self.assertTrue(anneal["neural_eligible"])
        self.assertFalse(profile["neural_eligible"])

    def test_habitual_first_person_note_and_natural_correction_share_topic(self):
        original, = episodes("When the baseline drifts, I always use a fresh reference first.")
        revision = "Actually, I now start baseline drift checks by using a sealed blank."
        corrected, = episodes(revision)
        self.assertEqual(original["key"], "baseline drift")
        self.assertEqual(original["key"], corrected["key"])
        self.assertEqual(original["answer"], "I always use a fresh reference first")
        self.assertEqual(corrected["answer"], "using a sealed blank")
        self.assertTrue(original["neural_eligible"] and corrected["neural_eligible"])
        self.assertIsNotNone(_CORRECTION.search(revision))

    def test_natural_preference_and_finished_experiment_can_share_one_note(self):
        observation = "We finished the ZnO experiment today. I now prefer short checklists for lab notes."
        status, preference = episodes(observation)
        self.assertEqual(
            [(lesson["category"], lesson["key"], lesson["answer"]) for lesson in (status, preference)],
            [("experiment", "ZnO", "finished"), ("preference", "lab notes", "short checklists")],
        )
        self.assertTrue(status["neural_eligible"] and preference["neural_eligible"])
        for lesson in (status, preference):
            self.assertIn(lesson["answer"], observation)

        scoped, = episodes("For lab reports, I usually prefer one-page summaries.")
        self.assertEqual((scoped["key"], scoped["answer"]), ("lab reports", "one-page summaries"))

    def test_natural_expansion_leaves_ambiguous_or_dependent_notes_untrained(self):
        for note in (
            "When the baseline drifts, I might use a sealed blank.",
            "When that happens, I always use a sealed blank.",
            "I now start baseline drift checks by using a sealed blank. Then discard it.",
            "I now start baseline drift checks by guessing.",
            "I now start this by using a sealed blank.",
            "We may have finished the ZnO experiment today.",
            "We finished the experiment today.",
            "Hypothetically. I now start baseline drift checks by using a sealed blank.",
        ):
            with self.subTest(note=note):
                self.assertFalse(any(lesson["neural_eligible"] for lesson in episodes(note)))

    def test_long_and_tentative_explicit_values_remain_evidence_only(self):
        tentative, lengthy = episodes(
            "Our experiment detector status is maybe ready.\n"
            "I prefer running through a carefully structured report with a separate line for every "
            "instrument reading we made that day for assessment."
        )
        self.assertFalse(tentative["neural_eligible"])
        self.assertFalse(lengthy["neural_eligible"])


if __name__ == "__main__":
    unittest.main()
