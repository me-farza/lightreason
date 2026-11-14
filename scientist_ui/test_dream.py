"""Checks for captured evidence, Dream promotion and verified coaching inputs."""

from math import exp, log
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from importlib.util import find_spec
from types import SimpleNamespace
from time import monotonic, sleep
import unittest
from unittest.mock import patch

from scientist_ui.dream import Dream, PolicyTrainer, ProcedureMemory
from scientist_ui.skill_rehearsal import DAYS


class DreamTestCase(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = Path(temporary.name) / "memory.db"

    def finish(self, dream):
        deadline = monotonic() + 4
        while monotonic() < deadline:
            report = dream.status()
            if report["status"] != "running":
                self.assertEqual(report["status"], "complete", report.get("error"))
                return report
            sleep(.01)
        self.fail("Dream did not finish")

class DreamTests(DreamTestCase):
    def test_promotion_and_correction_survive_reopen(self):
        dream = Dream(self.db)
        dream.capture("My lantern is purple.", "Approved chat")
        dream.capture("For my lab notes, I prefer a short checklist.", "notes.md", "document")
        self.assertEqual(dream.recall("lantern"), [])
        dream.start()
        report = self.finish(dream)
        self.assertEqual(report["counts"], {"captures": 2, "promoted": 2, "pending": 0,
                                            "questions": 2, "neural_examples": 2})
        self.assertFalse(report["training"]["configured"])
        self.assertEqual({item["category"] for item in report["questions"]}, {"profile", "preference"})
        self.assertEqual(dream.recall("lantern")[0]["source"], "Approved chat")
        self.assertEqual(len(dream.recall("What do you know about me?")), 2)

        reopened = Dream(self.db)
        self.assertEqual(reopened.status(), report)
        reopened.capture("Correction: my lantern is gray.", "Approved correction")
        self.assertIn("purple", reopened.recall("lantern")[0]["content"])
        reopened.start()
        self.finish(reopened)
        self.assertEqual([row["content"] for row in reopened.recall("lantern")],
                         ["Correction: my lantern is gray."])

    def test_compact_procedure_is_a_neural_training_candidate(self):
        observed = []

        def coach(examples, _):
            observed.extend(examples)
            return {"configured": True, "eligible": len(examples)}

        dream = Dream(self.db, coach)
        dream.capture("The ZnO anneal schedule specifies 425C at 3C/min.", "zno.md", "document")
        dream.start()
        report = self.finish(dream)
        self.assertEqual(report["counts"]["neural_examples"], 1)
        self.assertEqual(observed[0]["training_kind"], "process")
        self.assertEqual(observed[0]["answer"], "425C at 3C/min")

    def test_coaching_reports_only_real_trainer_result_and_preserves_queue(self):
        entered = Event()
        allow_finish = Event()

        trained = []

        def coach(examples, progress):
            candidates = [item for item in examples if item["neural_eligible"]]
            trained.append([(item["training_kind"], item["answer"]) for item in candidates])
            entered.set()
            if not allow_finish.wait(4):
                raise RuntimeError("Test did not release trainer")
            progress("Ran one verified coaching case.")
            return {"configured": True, "eligible": 1, "attempts": 1, "promotions": 0,
                    "optimizer_steps": 0, "outcomes": ["unchanged"]}

        dream = Dream(self.db, trainer=coach)
        dream.capture("My lantern is gray.", "Approved chat")
        dream.start()
        try:
            self.assertTrue(entered.wait(4))
            self.assertEqual(dream.start()["status"], "running")
            dream.capture("Our experiment ZnO status is awaiting XRD review.", "notes.md", "document")
        finally:
            allow_finish.set()
        report = self.finish(dream)
        self.assertEqual(report["training"]["outcomes"], ["unchanged"])
        self.assertEqual(report["counts"]["pending"], 1)
        self.assertEqual(dream.recall("ZnO"), [])
        dream.start()
        report = self.finish(dream)
        self.assertEqual(report["counts"]["pending"], 0)
        self.assertIn("awaiting XRD review", dream.recall("ZnO")[0]["content"])
        self.assertEqual(trained, [[("color", "gray")], [("experiment", "awaiting XRD review")]])
        self.assertEqual(
            [row["source"] for row in Dream(self.db).recall("what expriemnts did we do?")],
            ["notes.md"],
        )
        self.assertEqual(
            [row["source"] for row in Dream(self.db).recall("what do you know about me?")],
            ["Approved chat"],
        )


class RehearsalTests(DreamTestCase):
    def setUp(self):
        super().setUp()
        self.trainer = RehearsalTrainer()

    def open(self):
        return Dream(self.db, trainer=self.trainer)

    def run_dream(self, dream):
        dream.start()
        return self.finish(dream)

    def test_split_lessons_and_correction_retry(self):
        dream = self.open()
        dream.capture(DAYS[0][0], "chat")
        first = self.run_dream(dream)
        self.assertEqual(first["skill"]["status"], "awaiting_lessons")
        self.assertEqual(first["counts"]["pending"], 0)
        self.assertEqual(self.trainer.stages, [])

        dream = self.open()
        dream.capture(DAYS[0][1], "chat")
        dream.capture(DAYS[1][0], "chat")
        second = self.run_dream(dream)
        self.assertEqual(second["skill"]["stage"], "day1")
        self.assertEqual(self.trainer.stages, ["day1"])

        self.trainer.reject = True
        held = self.run_dream(dream)
        self.assertEqual(held["skill"]["status"], "held")
        self.assertTrue(held["skill"]["retryable"])
        self.assertEqual(self.trainer.digest, "day1")
        self.assertFalse(dream.rehearsal_report()["day2"]["promoted"])

        self.trainer.reject = False
        dream = self.open()
        promoted = self.run_dream(dream)
        self.assertEqual(promoted["skill"]["status"], "promoted")
        self.assertEqual(promoted["skill"]["stage"], "day2")
        report = dream.rehearsal_report()
        self.assertEqual(report["day2"]["checkpoint_sha256"], "day2")
        self.assertEqual(self.run_dream(dream), promoted)
        self.assertEqual(dream.rehearsal_report(), report)
        self.assertEqual(self.trainer.stages, ["day1", "day2", "day2"])

    def test_active_checkpoint_mismatch_blocks_new_skill_update(self):
        dream = self.open()
        dream.capture("\n".join(DAYS[0]), "chat")
        self.run_dream(dream)
        dream.capture(DAYS[1][0], "chat")
        self.trainer.digest = "mismatched"
        report = self.run_dream(dream)
        self.assertEqual(report["skill"]["status"], "inconsistent")
        self.assertEqual(self.trainer.stages, ["day1"])
        self.assertNotIn("day2", dream.rehearsal_report())

    def test_report_is_recovered_after_checkpoint_publication(self):
        dream = self.open()
        dream.capture("\n".join(DAYS[0]), "chat")
        self.run_dream(dream)
        dream.capture(DAYS[1][0], "chat")
        self.trainer.interrupt = True
        with self.assertLogs("scientist_ui.dream", level="ERROR"):
            dream.start()
            deadline = monotonic() + 4
            while (state := dream.status())["status"] == "running" and monotonic() < deadline:
                sleep(.01)
            self.assertEqual(state["status"], "failed")
        self.assertNotIn("day2", dream.rehearsal_report())
        self.assertEqual(self.trainer.digest, "day2")

        dream = self.open()
        recovered = self.run_dream(dream)
        self.assertEqual(recovered["skill"]["status"], "up_to_date")
        self.assertEqual(dream.rehearsal_report()["day2"]["checkpoint_sha256"], "day2")
        self.assertEqual(self.trainer.stages, ["day1", "day2"])


class NightCadenceTests(DreamTestCase):
    def test_empty_entry_keeps_results_and_reopen_advances_on_new_material(self):
        dream = Dream(self.db)
        self.assertEqual(dream.start(), dream.status())
        self.assertEqual(dream.status()["night"], 0)

        dream.capture("The camera protocol specifies a dark reference.", "notes.md", "document")
        dream.start()
        night_one = self.finish(dream)
        self.assertEqual(night_one["night"], 1)
        self.assertEqual(night_one["history"], [])
        self.assertEqual(dream.start(), night_one)

        dream = Dream(self.db)
        self.assertEqual(dream.start(), night_one)
        dream.capture("Correction: the camera protocol specifies a sealed reference.", "chat")
        dream.start()
        night_two = self.finish(dream)
        self.assertEqual(night_two["night"], 2)
        self.assertEqual(night_two["history"], [{
            "night": 1, "status": "complete", "questions": 1, "captures": 1, "neural_updates": 0,
        }])
        self.assertEqual(dream.start(), night_two)

    def test_failed_night_retries_same_snapshot(self):
        calls = []

        def trainer(examples, _):
            calls.append([item["answer"] for item in examples])
            if len(calls) == 1:
                raise RuntimeError("synthetic interruption")
            return {"eligible": len(examples), "promotions": 0}

        dream = Dream(self.db, trainer)
        dream.capture("The camera protocol specifies a dark reference.", "chat")
        with self.assertLogs("scientist_ui.dream", level="ERROR"):
            dream.start()
            deadline = monotonic() + 4
            while (report := dream.status())["status"] == "running" and monotonic() < deadline:
                sleep(.01)
        self.assertEqual(report["status"], "failed")
        dream.capture("The scorer procedure specifies hidden labels.", "chat")

        dream = Dream(self.db, trainer)
        dream.start()
        retried = self.finish(dream)
        self.assertEqual(retried["night"], 1)
        self.assertEqual(retried["counts"]["pending"], 1)
        dream.start()
        self.assertEqual(self.finish(dream)["night"], 2)
        self.assertEqual(calls, [["a dark reference"], ["a dark reference"], ["hidden labels"]])

    def test_held_policy_can_retry_or_start_a_new_night_with_new_material(self):
        calls = []

        def trainer(examples, _):
            calls.append(examples[0]["answer"])
            return {"eligible": 1, "promotions": 0, "policy_runs": [{"status": "held"}]}

        dream = Dream(self.db, trainer)
        dream.capture("The camera protocol specifies a dark reference.", "chat")
        dream.start()
        original = self.finish(dream)
        self.assertEqual(dream.start(), original)
        dream.start(retry=True)
        self.assertEqual(self.finish(dream)["night"], 1)
        dream.capture("Correction: the camera protocol specifies a sealed reference.", "chat")
        dream.start()
        self.assertEqual(self.finish(dream)["night"], 2)
        self.assertEqual(calls, ["a dark reference", "a dark reference", "a sealed reference"])


class RehearsalTrainer:
    def __init__(self):
        self.digest = None
        self.reject = False
        self.interrupt = False
        self.stages = []

    def skill_checkpoint_digest(self):
        return self.digest

    def train_skills(self, batch, progress, prepare):
        stage = "day2" if batch["supersedes"] else "day1"
        self.stages.append(stage)
        report = {"promoted": not self.reject, "checkpoint_sha256": None if self.reject else stage}
        if not self.reject:
            prepare(report)
            self.digest = stage
            if self.interrupt:
                raise OSError("interrupted after publication")
        return report


@unittest.skipUnless(find_spec("torch") and find_spec("safetensors"), "requires PyTorch and safetensors")
class ProcedureMemoryTests(unittest.TestCase):
    def test_optimizer_updates_persist_and_return_exact_native_rows(self):
        import torch

        class Encoder:
            def ids(self, text):
                return text

            def request(self, text, *, encode_values):
                self.features_called = True
                assert not encode_values
                features = torch.zeros((1, 1, 768))
                lane = 0 if "zno anneal schedule" in text.casefold() else 1
                features[0, 0, lane] = features[0, 0, 384 + lane] = 1
                return features, torch.ones((1, 1), dtype=torch.bool), None, None

        class Tokenizer:
            all_special_ids = [0]

            def __len__(self):
                return 6

            def encode(self, text, *, add_special_tokens):
                assert not add_special_tokens
                return [2, 3] if "425" in text else [4, 5]

            def decode(self, ids, *, clean_up_tokenization_spaces):
                assert not clean_up_tokenization_spaces
                return {(2, 3): " 425C at 3C/min", (4, 5): " 435C at 3C/min"}[tuple(ids)]

        with TemporaryDirectory() as folder:
            path = Path(folder) / "memory.procedure.safetensors"
            state = path.with_name("memory.safetensors")
            encoder = Encoder()
            table = torch.arange(6, dtype=torch.bfloat16).expand(2048, 6).T.contiguous()
            original = ProcedureMemory(path, "pinned-test-model")
            holdout = "For the ZnO anneal schedule, what should I set or do?"
            self.assertIsNone(original.recall(encoder, holdout))
            example = {"key": "ZnO anneal schedule", "answer": "425C at 3C/min",
                       "question": "What does the ZnO anneal schedule specify?"}
            with patch("lightreason_memory.latent_bridge.load_destination", return_value=(Tokenizer(), table)) as load:
                fresh = PolicyTrainer(None, None, None, state, "pinned-test-model")
                self.assertIsNone(fresh.readout(holdout))
                load.assert_not_called()
                trace = {}
                checkpoint, steps, checks = original.train(encoder, example, lambda _: None, trace=trace)
                self.assertTrue(checkpoint)
                self.assertGreater(steps, 0)
                self.assertEqual(trace["status"], "promoted")
                self.assertGreater(trace["sampling"]["first"]["sample_count"], 0)
                self.assertEqual(trace["sampling"]["last"]["step"], steps)
                self.assertGreater(trace["sampling"]["last"]["mean_reward"], trace["sampling"]["first"]["mean_reward"])
                self.assertEqual([check["reward"] for check in checks if check["split"] == "held out"], [1, 1])
                self.assertLess(checks[0]["score_before"], .8)
                self.assertGreaterEqual(checks[0]["score_after"], .8)
                restart = ProcedureMemory(path, "pinned-test-model")
                torch.testing.assert_close(restart.recall(encoder, holdout), table[[2, 3]])
                self.assertIsNone(restart.recall(encoder, "What settings does the unrelated reference procedure use?"))
                example["answer"] = "435C at 3C/min"
                revision, correction_steps, correction = restart.train(encoder, example, lambda _: None)
                self.assertGreater(correction_steps, 0)
                self.assertEqual(correction[0]["before"], "previous value")
                self.assertEqual(correction[0]["reward_before"], -1)
                self.assertEqual(correction[0]["reward_after"], 1)
                unchanged_trace = {}
                unchanged, unchanged_steps, _ = restart.train(encoder, example, lambda _: None, trace=unchanged_trace)
                self.assertIsNone(unchanged)
                self.assertEqual(unchanged_steps, 0)
                self.assertEqual(unchanged_trace["status"], "unchanged")
                restored = PolicyTrainer(None, None, None, state, "pinned-test-model")
                with patch.object(restored, "_memory", return_value=SimpleNamespace(encoder=encoder)):
                    self.assertIsNone(restored.readout("What settings does the unrelated reference procedure use?"))
                    self.assertEqual(load.call_count, 2)
                    self.assertEqual(restored.readout(holdout), "435C at 3C/min")
                    torch.testing.assert_close(restored.vectors(holdout), table[[4, 5]])
                    self.assertEqual(restored.readout(holdout), "435C at 3C/min")
                    self.assertEqual(load.call_count, 3)
            self.assertNotEqual(checkpoint, revision)
            torch.testing.assert_close(ProcedureMemory(path, None).recall(encoder, holdout), table[[4, 5]])
            self.assertNotIn(b"ZnO anneal schedule", path.read_bytes())

    def test_readout_requires_an_exact_unique_regular_vocabulary_row(self):
        import torch

        class Tokenizer:
            all_special_ids = [0]

            def __len__(self):
                return 5

            def decode(self, ids, *, clean_up_tokenization_spaces):
                return " ".join(str(i) for i in ids)

        memory = ProcedureMemory(None, "pinned-test-model")
        table = torch.zeros((6, 2048), dtype=torch.bfloat16)
        table[1:, -1] = torch.tensor([1, 2, 2, 4, 4], dtype=torch.bfloat16)
        with patch("lightreason_memory.latent_bridge.load_destination", return_value=(Tokenizer(), table)):
            self.assertEqual(memory.decode_rows(table[[4, 1]]), "4 1")
            for rows in (table[[0]], table[[2]], torch.ones_like(table[[1]])):
                with self.assertRaisesRegex(ValueError, "unique non-special"):
                    memory.decode_rows(rows)
            with self.assertRaisesRegex(ValueError, "Invalid native memory rows"):
                memory.decode_rows(table[[1]].float())

    def test_sampled_policy_retains_other_procedures_and_holds_an_ambiguous_candidate(self):
        import torch

        def features(_, question):
            # The ambiguous candidate's held-out phrasing collides only with the
            # retained probe. Training examples still identify a distinct key.
            lane = 2
            if "gas inlet" in question or question.startswith("For the ambiguous"):
                lane = 0
            elif any(key in question for key in ("phosphate rinse", "ambiguous")):
                lane = 1
            vector = torch.zeros(768)
            vector[lane] = vector[384 + lane] = 1
            return vector

        def answer_rows(word):
            rows = torch.zeros((12, 2048), dtype=torch.bfloat16)
            rows[0, 0] = float(word)
            return rows, 1

        def candidate(key, answer, kind="process"):
            return {"key": key, "question": f"What does the {key} specify?", "answer": answer,
                    "source": "test note", "training_kind": kind, "neural_eligible": True}

        with TemporaryDirectory() as folder:
            path = Path(folder) / "memory.safetensors"
            trainer = PolicyTrainer(None, None, None, path)
            memory = SimpleNamespace(encoder=None, dream_status=lambda: {"generation": 0, "attempts": 0})
            published = []

            def checked_answer_rows(word):
                if word in {"40", "50"}:
                    self.assertEqual(published[-1]["status"], "training")
                    self.assertEqual(published[-1]["optimizer_steps"], 0)
                    self.assertEqual(published[-1]["checks"], [])
                    self.assertEqual(len(published[-1]["plan"]), 6)
                return answer_rows(word)

            with (patch.object(trainer, "_memory", return_value=memory),
                  patch.object(trainer.procedure, "_encode", side_effect=features),
                  patch.object(trainer.procedure, "_answer_rows", side_effect=checked_answer_rows)):
                report = trainer([candidate("gas inlet procedure", "40"),
                                  candidate("phosphate rinse", "50", "preference")], lambda _: None,
                                 on_policy_run=published.append)
                final_runs = report["policy_runs"]
                self.assertEqual([run for run in published if run["status"] == "promoted"], final_runs)
                self.assertEqual([run["candidate_id"] for run in final_runs], [1, 2])
                for run in final_runs:
                    history = [event for event in published if event["candidate_id"] == run["candidate_id"]]
                    self.assertEqual(history[0]["status"], "training")
                    self.assertEqual(history[1]["sampling"]["last"]["step"], 1)
                    self.assertEqual(len(history[1]["sampling"]["points"]), 1)
                    self.assertEqual(history[-2]["status"], "validating")
                    self.assertEqual(history[-2]["optimizer_steps"], run["optimizer_steps"])
                    self.assertEqual(history[-2]["checks"], [])
                    points = run["sampling"]["points"]
                    self.assertLessEqual(len(points), 20)
                    self.assertEqual(run["sampling"]["first"], points[0])
                    self.assertEqual(run["sampling"]["last"], points[-1])
                    self.assertEqual(run["optimizer_steps"], points[-1]["step"])
                    self.assertTrue(all(point["step"] % 8 == 0 for point in points[1:-1]))
                    # Initial logits are (0, -2) and sampled at temperature 2.
                    p = 1 / (1 + exp(1))
                    initial_entropy = -(p * log(p) + (1 - p) * log(1 - p))
                    self.assertAlmostEqual(points[0]["mean_entropy_nats"], initial_entropy, places=6)
                    for point in points:
                        self.assertEqual(point["sample_count"], 96)
                        self.assertGreaterEqual(point["mean_entropy_nats"], 0)
                        self.assertLessEqual(point["mean_entropy_nats"], log(2) + 1e-7)
                        self.assertGreaterEqual(point["success_rate"], 0)
                        self.assertLessEqual(point["success_rate"], 1)
                        self.assertAlmostEqual(point["mean_reward"], 2 * point["success_rate"] - 1, places=6)
                        successes = point["success_rate"] * point["sample_count"]
                        self.assertAlmostEqual(successes, round(successes), places=6)
                self.assertEqual([run["algorithm"] for run in final_runs], ["REINFORCE", "REINFORCE"])
                self.assertEqual(final_runs[1]["before_passed"], 2)
                self.assertEqual(final_runs[1]["after_passed"], 3)
                self.assertEqual(final_runs[0]["checks"][0]["question"],
                                 "For the gas inlet procedure, what should I set or do?")
                self.assertEqual(final_runs[1]["checks"][0]["question"],
                                 "Which option do I usually choose for phosphate rinse?")
                self.assertIn("What is my preference for phosphate rinse?",
                              [check["question"] for check in final_runs[1]["checks"] if check["split"] == "training"])
                saved = trainer.procedure.path.read_bytes()

                report = trainer([candidate("ambiguous procedure", "130")], lambda _: None)
                run, = report["policy_runs"]
                self.assertEqual(run["status"], "held")
                self.assertGreater(run["optimizer_steps"], 0)
                self.assertLess(run["after_passed"], len([x for x in run["checks"] if x["split"] != "training"]))
                self.assertEqual(saved, trainer.procedure.path.read_bytes())
                torch.testing.assert_close(trainer.procedure.recall(None, "gas inlet"), answer_rows("40")[0][:1])

    def test_status_and_general_preference_use_category_specific_paraphrases(self):
        import torch

        examples = (
            ("experiment", "Quartz", "Where does the Quartz experiment stand?",
             "How is the Quartz experiment progressing?", "What is the latest update on the Quartz experiment?"),
            ("preference", "general", "What is my general preference?",
             "What do I usually prefer?", "What is my usual choice?"),
        )

        def features(_, question):
            negative = any(word in question for word in ("unrelated", "separate"))
            vector = torch.zeros(768)
            lane = int(negative)
            vector[lane] = vector[384 + lane] = 1
            return vector

        for kind, key, prompt, training, held_out in examples:
            with self.subTest(kind=kind), TemporaryDirectory() as folder:
                memory = ProcedureMemory(Path(folder) / "procedure.safetensors", None)
                example = {"training_kind": kind, "key": key, "question": prompt, "answer": "verified result"}
                rows = torch.ones((12, 2048), dtype=torch.bfloat16)
                with (patch.object(memory, "_encode", side_effect=features),
                      patch.object(memory, "_answer_rows", return_value=(rows, 1))):
                    digest, _, checks = memory.train(None, example, lambda _: None)
                self.assertTrue(digest)
                positive = [(check["split"], check["question"]) for check in checks if check["expected"] == "recall"]
                self.assertEqual(positive, [("held out", held_out), ("training", prompt), ("training", training)])

    def test_optional_gemini_questions_are_checked_before_promotion(self):
        import torch

        practice = "How should I set up the gas inlet before this run?"
        transfer = "What gas inlet setting should I use on a fresh setup?"
        example = {"training_kind": "process", "key": "gas inlet", "answer": "40 sccm",
                   "question": "What does the gas inlet specify?", "question_variants": [practice, transfer]}
        rows = torch.ones((12, 2048), dtype=torch.bfloat16)

        for scenario in ("validated", "feature_bound", "failed_check"):
            with self.subTest(scenario=scenario), TemporaryDirectory() as folder:
                calls = []
                memory = ProcedureMemory(Path(folder) / "procedure.safetensors", None)

                def features(_, question):
                    calls.append(question)
                    if question == transfer and scenario == "feature_bound":
                        raise ValueError("too many tokens")
                    gis = "gas inlet" in question and not (question == transfer and scenario == "failed_check")
                    vector = torch.zeros(768)
                    lane = int(gis)
                    vector[lane] = vector[384 + lane] = 1
                    return vector

                trace = {}
                progress = []
                with (patch.object(memory, "_encode", side_effect=features),
                      patch.object(memory, "_answer_rows", return_value=(rows, 1))):
                    digest, steps, checks = memory.train(None, example, lambda _: None, trace=trace,
                                                         on_progress=lambda **update: progress.append(update))
                self.assertTrue(digest)
                self.assertEqual(trace["status"], "promoted")
                # Held-out cases never contribute to the optimizer batch, including on fallback.
                sampled_per_step = 120 if scenario == "validated" else 96
                self.assertTrue(all(point["sample_count"] == sampled_per_step for point in trace["sampling"]["points"]))
                self.assertCountEqual(calls, [case["question"] for case in memory.question_plan(example)])
                gemini = [(row["split"], row["question"], row["reward_after"])
                          for row in checks if row["origin"] == "gemini"]
                if scenario == "validated":
                    self.assertEqual(gemini, [("held out", transfer, 1), ("training", practice, 1)])
                    self.assertNotIn("planner_fallback", trace)
                else:
                    self.assertFalse(gemini)
                    self.assertTrue(trace["planner_fallback"])
                    self.assertTrue(all(case["origin"] == "local" for case in trace["plan"]))
                    discarded = trace["discarded_optimizer_steps"]
                    self.assertEqual(discarded > 0, scenario == "failed_check")
                    self.assertEqual(steps, discarded + trace["sampling"]["last"]["step"])
                    self.assertEqual(progress[-1]["optimizer_steps"], steps)
                    self.assertEqual(progress[-1]["planner_fallback"], trace["planner_fallback"])

        plan = ProcedureMemory.question_plan({**example, "question_variants": [example["question"].upper(), transfer]})
        self.assertEqual(len(plan), 6)
        self.assertTrue(all(case["origin"] == "local" for case in plan))


if __name__ == "__main__":
    unittest.main()
