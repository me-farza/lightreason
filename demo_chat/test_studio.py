"""CPU-only contracts for single-runner generation and explicit page context."""
from __future__ import annotations

from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

from demo_chat import studio
from demo_chat.prepared import PreparedLiveInference
from demo_chat.messages import PromptText


class PromptTests(unittest.TestCase):
    def test_first_instruction_is_not_rewritten(self):
        value = "  Build a blue button in one HTML file.\n"
        self.assertEqual(studio.build_prompt(value), value)
        self.assertEqual(studio.build_prompt(value, ""), value)

    def test_update_explicitly_contains_selected_code_and_instruction(self):
        page = "<!doctype html><button>Old</button>"
        prompt = studio.build_prompt("Make the button green.", page)
        self.assertIn("Current HTML document:\n```html\n" + page + "\n```", prompt)
        self.assertIn("Requested update:\nMake the button green.", prompt)
        self.assertIn("Return the complete updated HTML document", prompt)

    def test_invalid_or_oversized_context_is_not_silently_truncated(self):
        for instruction, code in ((None, None), (" ", None), ("x", {}), ("x" * 8001, None),
                                  ("Make it green", "x" * 8000)):
            with self.subTest(instruction=str(instruction)[:20]), self.assertRaises(ValueError):
                studio.build_prompt(instruction, code)

    def test_explicit_page_update_retains_request_local_system_instructions(self):
        instruction = PromptText("Make the button green.", "Build accessible responsive pages.")
        prompt = studio.build_prompt(instruction, "<button>Old</button>")
        self.assertIsInstance(prompt, PromptText)
        self.assertEqual(prompt.system_prompt, instruction.system_prompt)
        self.assertIn("<button>Old</button>", prompt)


class _Tokenizer:
    def __init__(self, trail):
        self.trail = trail
        self.length = 3

    def apply_chat_template(self, messages, **kwargs):
        self.trail.append(("tokenize", messages, kwargs))
        return [11] * self.length

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))


class _Torch:
    long = "int64"
    cuda = SimpleNamespace(synchronize=lambda: None)

    @staticmethod
    def tensor(values, **kwargs):
        return SimpleNamespace(shape=(1, len(values[0])), values=values)


class _Engine(studio.StudioLiveInference):
    def __init__(self, turbo=True):
        paths = {"draft_path": "test-draft", "checkpoint_path": "test-checkpoint"} if turbo else {}
        super().__init__(**paths)
        self.trail = []
        self.ready = True
        self.model = SimpleNamespace(device="fake-cuda")
        self.tokenizer = _Tokenizer(self.trail)
        self.fast = SimpleNamespace(signature=["existing-fast"]) if turbo else None
        self.strict = SimpleNamespace(signature=["existing-exact"]) if turbo else None
        self.eos = {99}
        self.outputs = [7, 8, 99]
        self.admitted = True
        self.mutate_graph = False

    def _thread(self):
        self.trail.append(("thread",))

    def _request_guards(self):
        self.trail.append(("identity",))

    def _source_guards(self):
        self.trail.append(("sources",))

    def is_prepared(self, prompt, budget, mode):
        self.trail.append(("admission", prompt, budget, mode))
        if self.fast is None:
            return super().is_prepared(prompt, budget, mode)
        return {"prepared": self.admitted, "detail": "Prepare this demo first", "setup_s": 12.0}

    def run_fake(self, name, ids, budget, emit):
        self.trail.append(("generation", name, ids, budget))
        for index in range(1, len(self.outputs) + 1):
            emit(self.outputs[:index])
        if self.mutate_graph:
            self.fast.signature.append("unexpected-capture")
        return list(self.outputs), {"test_only": True}

    def _run_baseline(self, ids, budget, emit):
        return self.run_fake("original", ids, budget, emit)

    def _run_fast(self, ids, budget, emit):
        return self.run_fake("optimized", ids, budget, emit)

    def _run_strict(self, ids, budget, emit):
        return self.run_fake("strict", ids, budget, emit)


class SingleGenerationTests(unittest.TestCase):
    def setUp(self):
        self.engine = _Engine()
        self.events = []

    def invoke(self, mode="optimized", budget=64, prompt="Build an HTML button", **kwargs):
        with patch.dict("sys.modules", {"torch": _Torch()}), \
                patch.object(studio, "_all_graphs", lambda owner, mode: tuple(owner.signature) if owner else ()), \
                patch("time.sleep", side_effect=AssertionError("No artificial pauses")):
            return self.engine.generate(prompt, budget, mode, self.events.append, **kwargs)

    def test_exactly_one_selected_runner_generates_live_tokens(self):
        for mode in ("original", "optimized", "strict"):
            with self.subTest(mode=mode):
                self.setUp()
                result = self.invoke(mode)
                calls = [item for item in self.engine.trail if item[0] == "generation"]
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][1], mode)
                self.assertEqual([e["type"] for e in self.events],
                                 ["status", "started", "token_ids", "token_ids", "token_ids", "completed", "generation"])
                self.assertTrue(all(e["runner"] == "single" for e in self.events if "runner" in e))
                self.assertEqual(result["generation"]["token_ids"], [7, 8, 99])
                self.assertTrue(result["generation"]["metrics"]["finished_eos"])
                self.assertTrue(result["generation"]["integrity_passed"])
                self.assertFalse(result["generation"]["answers_cached"])
                self.assertEqual(result["generation"]["new_gpu_graphs_during_generation"], 0)
                self.assertFalse(any("speedup" in event for event in self.events))

    def test_qwen_only_standard_is_admitted_and_streams_without_exact_graphs(self):
        self.engine = _Engine(turbo=False)
        status = self.engine.prepared_status("HTML request", 64, "original")
        self.assertTrue(status["prepared"])
        self.assertEqual(status["prepared_modes"], ["original"])
        self.assertFalse(status["demo_ready"])
        self.assertFalse(self.engine.prepared_status("HTML request", 64, "strict")["prepared"])
        receipt = self.engine.prepare_demo("HTML request", 64, "original", lambda event: None)
        self.assertEqual(receipt["receipts"], {})
        result = self.invoke("original")["generation"]
        self.assertEqual(result["token_ids"], [7, 8, 99])
        self.assertTrue(result["integrity_passed"])
        self.assertFalse(result["demo_prepared"])
        self.assertEqual(result["new_gpu_graphs_during_generation"], 0)
        self.assertIsNone(self.engine.strict)

    def test_unprepared_shape_stops_before_any_generation(self):
        self.engine.admitted = False
        with self.assertRaisesRegex(RuntimeError, "Prepare this demo"):
            self.invoke()
        self.assertFalse(self.events)
        self.assertFalse(any(item[0] == "generation" for item in self.engine.trail))

    def test_internal_opt_out_cannot_cold_compile_turbo(self):
        self.engine.admitted = False
        with self.assertRaisesRegex(RuntimeError, "Prepare this demo"):
            self.invoke(require_prepared=False)
        self.assertFalse(self.events)

    def test_original_template_receives_exact_effective_prompt(self):
        self.invoke("original")
        tokenization = next(item for item in self.engine.trail if item[0] == "tokenize")
        self.assertEqual(tokenization[1], [{"role": "user", "content": "Build an HTML button"}])
        self.assertEqual(tokenization[2], {"add_generation_prompt": True, "tokenize": True})

    def test_each_runner_receives_the_same_actual_system_role(self):
        prompt = PromptText("Build an HTML button", "Use a responsive layout and working controls.")
        expected = [{"role": "system", "content": prompt.system_prompt},
                    {"role": "user", "content": str(prompt)}]
        for mode in ("original", "optimized", "strict"):
            with self.subTest(mode=mode):
                self.setUp()
                self.invoke(mode, prompt=prompt)
                tokenization = next(item for item in self.engine.trail if item[0] == "tokenize")
                self.assertEqual(tokenization[1], expected)

    def test_queue_delay_is_separate_from_generation_first_token(self):
        result = self.invoke(request_started_s=time.perf_counter() - 2)
        metrics = result["generation"]["metrics"]
        self.assertGreaterEqual(metrics["request_to_first_token_s"], 2)
        self.assertGreater(metrics["request_to_first_token_s"], metrics["time_to_first_token_s"])

    def test_unexpected_capture_prevents_accepted_completed_or_generation_event(self):
        self.engine.mutate_graph = True
        with self.assertRaisesRegex(RuntimeError, "unexpected GPU graph capture"):
            self.invoke()
        self.assertFalse(any(event["type"] in ("completed", "generation") for event in self.events))

    def test_native_eos_continuation_is_rejected(self):
        self.engine.outputs = [99, 7]
        with self.assertRaisesRegex(RuntimeError, "continued after native EOS"):
            self.invoke()
        self.assertFalse(any(event["type"] == "generation" for event in self.events))

    def test_until_finished_reports_real_context_cap_without_shortening_tokens(self):
        self.engine.outputs = [7, 8]
        result = self.invoke(budget=2, until_finished=True)
        event = result["generation"]
        self.assertEqual(event["token_ids"], [7, 8])
        self.assertEqual(event["metrics"]["stop_reason"], "context_limit")
        self.assertIsNone(event["metrics"]["output_limit_requested"])
        self.assertFalse(event["metrics"]["finished_eos"])

    def test_until_finished_retains_native_eos_when_it_occurs(self):
        result = self.invoke(until_finished=True)
        self.assertEqual(result["generation"]["metrics"]["stop_reason"], "eos")
        self.assertIsNone(result["generation"]["metrics"]["output_limit_requested"])

    def test_context_overflow_rejected_before_cuda_generation(self):
        self.engine.tokenizer.length = studio.MAX_LEN - 8
        with self.assertRaisesRegex(RuntimeError, "must fit in " + str(studio.MAX_LEN)):
            self.invoke()
        self.assertFalse(any(item[0] == "generation" for item in self.engine.trail))

    def test_source_guard_failure_rejected_before_output(self):
        self.engine._source_guards = lambda: (_ for _ in ()).throw(RuntimeError("source changed"))
        with self.assertRaisesRegex(RuntimeError, "source changed"):
            self.invoke()
        self.assertFalse(self.events)


class StudioPolicyTests(unittest.TestCase):
    def test_original_readiness_maps_to_the_same_prepared_shapes(self):
        instance = studio.StudioLiveInference()
        with patch.object(PreparedLiveInference, "_shape", return_value=33) as shape:
            self.assertEqual(instance._shape("HTML request", 128, "original"), 33)
        shape.assert_called_once_with("HTML request", 128, "strict")
        self.assertEqual(instance.prepared_status("HTML request", 128, "original")["runner_name"], "Original Qwen")

    def test_comparison_until_finished_annotations_preserve_original_tokens(self):
        events = []
        def compare(engine, prompt, budget, mode, callback, **kwargs):
            records = {}
            for name in ("baseline", "candidate"):
                event = {"type": "completed", "runner": name, "token_ids": [7, 8],
                         "metrics": {"finished_eos": False, "stop_reason": "max_new_tokens"}}
                records[name] = event
                callback(event)
            comparison = {"type": "comparison", "exact_match": True}
            callback(comparison)
            return {**records, "comparison": comparison}
        with patch.object(PreparedLiveInference, "compare", compare):
            result = studio.StudioLiveInference().compare("HTML", 2, "strict", events.append, until_finished=True)
        for name in ("baseline", "candidate"):
            self.assertEqual(result[name]["token_ids"], [7, 8])
            self.assertEqual(result[name]["metrics"]["stop_reason"], "context_limit")
        self.assertEqual(events[-1]["runner_names"]["candidate"], "Lightreason Exact")
        self.assertIsNone(result["comparison"]["output_limit_requested"])

    def test_invalid_mode_or_budget_is_rejected_before_torch(self):
        instance = studio.StudioLiveInference()
        for mode, budget in (("bad", 64), ("original", True),
                             ("strict", studio.MAX_LEN - studio.TREE_RESERVE)):
            with self.subTest(mode=mode, budget=budget), self.assertRaises(RuntimeError):
                instance.generate("HTML", budget, mode, lambda e: None)
        with self.assertRaisesRegex(RuntimeError, "available Lightreason runner"):
            instance.compare("HTML", 64, "original", lambda e: None)


if __name__ == "__main__":
    unittest.main()
