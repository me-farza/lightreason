"""Check Dream's conservative selection of unstructured chat turns."""

import unittest

from scientist_ui.dream_selection import is_conversation_only


class DreamSelectionTests(unittest.TestCase):
    def test_inventory_questions_and_small_talk_are_conversation_only(self):
        for turn in (
            "tell me whaich memories do you have?",
            "Memory inventory",
            "Could you show me how the interface works?",
            "What should we do next",
            "Why does it reset?",
            "Is the interface still running?",
            "Any ideas?",
            "Hi, Vicena!",
            "OK, thanks.",
            "Got it. Thank you!\nWhat's up?",
        ):
            with self.subTest(turn=turn):
                self.assertTrue(is_conversation_only(turn))

    def test_unstructured_notes_survive_conservative_filtering(self):
        for turn in (
            "The green sample darkened at the edges after the second pass.",
            "We raised the chamber pressure to 4 psi and got the same result.",
            "Maybe the warm-up accounts for the color change.",
            "The oxygen line was noisy, right?",
            "How light scattering works in this setup",
            "Why the detector drifts is still unclear.",
            "Air pressure?",
            "Check the reference before the next run.",
            "# Notes\nWe will come back to this after review.",
        ):
            with self.subTest(turn=turn):
                self.assertFalse(is_conversation_only(turn))

    def test_mixed_turns_stay_available_as_evidence(self):
        for turn in (
            "Thanks. The second sample darkened.\nWhy did that happen?",
            "Can you explain why? We changed the clamp.",
            "Can you explain, I changed the clamp?",
            "How should we proceed given yesterday's failed calibration?\nUse a blank first.",
            "What is the new threshold? Keep the older one for comparison.",
        ):
            with self.subTest(turn=turn):
                self.assertFalse(is_conversation_only(turn))


if __name__ == "__main__":
    unittest.main()
