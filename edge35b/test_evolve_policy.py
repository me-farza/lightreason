import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from edge35b.evolve_policy import CONTROL, Policy, evolve, replay_archive
from edge35b.width_autotune import DEFAULT_CPU_PAIR_TUNING, NATIVE_PACKED_TRANSPORT, SHARED_CPU_TRANSPORT, WidthTrial


RESULTS = Path(__file__).with_name("results") / "native50"


class EvolutionTests(unittest.TestCase):
    def score_archive(self, name):
        archive = json.loads((RESULTS / (name + ".json")).read_text())
        return replay_archive(archive, "fixed-8", ["studio"], ["known-math", "coding", "chat"], 2)

    def test_measured_narrower_width_is_rejected_for_token_drift(self):
        result = self.score_archive("width-trials")
        combined = result["development"]["combined"]
        self.assertGreater(combined["median_decode_ratio"], 1.10)
        self.assertEqual(combined["exact_pairs"], 0)
        self.assertEqual(combined["mismatches"][0]["first_different_token"], 31)
        self.assertIsNone(result["promoted"])

    def test_measured_cpu_pairing_passes_the_held_out_cases(self):
        result = self.score_archive("width-pair-retest")
        self.assertEqual(result["promoted"], "paired-1")
        self.assertEqual(result["held_out"]["exact_pairs"], 6)
        self.assertGreater(result["held_out"]["median_decode_ratio"], 1.01)

    def test_search_tests_only_the_frozen_winner_on_held_out_inputs(self):
        fastest = Policy((4,) * 4, 0)
        slower = Policy((6,) * 4, 0)
        calls = []

        def measure(case, stream, _label, policy, repeat):
            calls.append((case, stream, policy, repeat))
            rate = {CONTROL: 10, fastest: 12, slower: 11}[policy]
            tokens = [1, 7, 3] if case == "holdout" and policy == fastest else [1, 2, 3]
            return {"case": case, "stream": stream, "repeat": repeat, "cap": 3,
                    "cached_prompt_tokens": 0, "finish_reason": "length", "token_ids": tokens,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                    "timings": {"predicted_per_second": rate}, "http_wall_s": 3 / rate}

        result = evolve(measure, ["dev"], ["holdout"], 2, 2, [fastest, slower])
        self.assertEqual(result["selected"]["policy"], fastest.as_dict())
        self.assertIsNone(result["promoted"])
        self.assertEqual(result["held_out"]["mismatches"][0]["first_different_token"], 2)
        self.assertEqual({policy for case, _, policy, _ in calls if case == "holdout"}, {CONTROL, fastest})
        for stream in (True, False):
            pairs = [policy for case, mode, policy, _ in calls if case == "holdout" and mode == stream]
            self.assertEqual(pairs[:2], pairs[2:][::-1])

    def test_cpu_kernel_choice_is_recorded_and_resets_between_requests(self):
        with TemporaryDirectory() as root:
            config = Path(root) / "width.json"
            config.write_text(json.dumps({"width": 8, "min_pairs": 1, "cpu_pair_tuning": [3, 2, 5]}))
            trial = WidthTrial(config)
            self.assertTrue(trial.begin(1))
            width = trial.choose()
            trial.observe(width, 4, True)
            self.assertEqual(json.loads(trial.log.read_text())["cpu_pair_tuning"], [3, 2, 5])
            config.write_text(json.dumps({"width": 8, "min_pairs": 1}))
            self.assertTrue(trial.begin(2))
            self.assertEqual(trial.cpu_pair_tuning, DEFAULT_CPU_PAIR_TUNING)

    def test_cpu_read_policy_requires_the_native_verifier_and_resets(self):
        with TemporaryDirectory() as root:
            config = Path(root) / "width.json"
            settings = {"width": 8, "min_pairs": 1, "transport": NATIVE_PACKED_TRANSPORT}
            candidate = {**settings, "cpu_read_policy": "packed-both-prefetch"}
            trial = WidthTrial(config)
            config.write_text(json.dumps({**candidate, "transport": "raw"}))
            with self.assertRaises(ValueError):
                trial.begin(1)
            config.write_text(json.dumps(candidate))
            self.assertTrue(trial.begin(1))
            trial.observe(trial.choose(), 4, True)
            self.assertEqual(json.loads(trial.log.read_text())["cpu_read_policy"], "packed-both-prefetch")
            config.write_text(json.dumps(settings))
            self.assertTrue(trial.begin(2))
            self.assertEqual(trial.cpu_read_policy, "stock")

    def test_draft_budget_respects_the_cap_and_resets_for_the_control(self):
        costs = {str(w): {"pass_ms": ms, "host_bytes": ms * 1e7}
                 for w, ms in ((4, 60), (6, 85), (8, 100))}
        budget = {"probabilities": [[0.15] + [0.99] * 7 for _ in range(7)],
                  "costs": {"cold": costs, "steady": costs}}
        settings = {"width": "draft-budget", "min_pairs": 1, "transport": SHARED_CPU_TRANSPORT,
                    "cpu_read_policy": "packed-both-prefetch", "draft_budget": budget}
        with TemporaryDirectory() as root, patch.dict("os.environ", {"LIGHTREASON_DRAFT_BUDGET": "1"}):
            config = Path(root) / "width.json"
            config.write_text(json.dumps(settings))
            trial = WidthTrial(config)
            self.assertTrue(trial.begin(1))
            self.assertTrue(trial.needs_draft_confidence)
            self.assertEqual(trial.choose(confidence=[0.99] * 7, remaining=9), 8)
            self.assertEqual(trial.choose(confidence=[0.05] * 7, remaining=9), 4)
            self.assertEqual(trial.choose(confidence=[0.99] * 7, remaining=2), 4)
            settings.pop("draft_budget")
            config.write_text(json.dumps({**settings, "width": 8}))
            self.assertTrue(trial.begin(2))
            self.assertFalse(trial.needs_draft_confidence)
            self.assertEqual(trial.choose(), 8)


if __name__ == "__main__":
    unittest.main()
