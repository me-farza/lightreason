"""CPU-only checks for original-first scheduling and truthful wait metrics."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

from demo_chat.interactive import InteractiveLiveInference, PROTOCOL
from demo_chat.inference import MAX_LEN
from demo_chat.messages import PromptText


class _Tokenizer:
    def __init__(self, trail, length=3):
        self.trail, self.length = trail, length

    def apply_chat_template(self, messages, **kwargs):
        self.trail.append(("tokenize", messages, kwargs))
        return [11] * self.length

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))


class _Runner:
    def __init__(self, trail):
        self.trail = trail

    def prepare(self, ids, budget):
        self.trail.append(("prepare", ids, budget))
        return {"setup_s": 0.0, "new_graphs": 0}


class _Torch:
    long = "int64"

    def __init__(self, trail):
        self.trail = trail
        self.cuda = SimpleNamespace(synchronize=lambda: None)

    def tensor(self, values, **kwargs):
        self.trail.append(("real_ids", values))
        return SimpleNamespace(kind="real", shape=(1, len(values[0])))

    def zeros(self, shape, **kwargs):
        value = SimpleNamespace(kind="synthetic", shape=shape)
        self.trail.append(("synthetic", shape))
        return value

    def inference_mode(self, *args):
        return nullcontext()

    def no_grad(self):
        return nullcontext()


class _Engine(InteractiveLiveInference):
    def __init__(self, trail):
        super().__init__(draft_path="test-draft", checkpoint_path="test-checkpoint")
        self.trail = trail
        self.ready = True
        self.model = SimpleNamespace(device="test-cuda")
        self.tokenizer = _Tokenizer(trail)
        self.fast = _Runner(trail)
        self.eos = {99}
        self.baseline_outputs = [7, 99]
        self.candidate_outputs = [7, 99]

    def _thread(self):
        pass

    def _request_guards(self):
        self.trail.append(("guard",))

    def _source_guards(self):
        self.trail.append(("source_guard",))

    def _ensure_strict(self, callback):
        self.trail.append(("construct_strict",))
        self.strict = _Runner(self.trail)

    def _run_baseline(self, ids, budget, emit):
        self.trail.append(("baseline_generate", ids))
        for index in range(1, len(self.baseline_outputs) + 1):
            emit(self.baseline_outputs[:index])
        return self.baseline_outputs[:], {"original_hf_test_only": True}

    def _run_fast(self, ids, budget, emit):
        self.trail.append(("candidate_generate", ids))
        for index in range(1, len(self.candidate_outputs) + 1):
            emit(self.candidate_outputs[:index])
        return self.candidate_outputs[:], {"original_controller_test_only": True}

    def _run_strict(self, ids, budget, emit):
        self.trail.append(("strict_generate", ids))
        for index in range(1, len(self.candidate_outputs) + 1):
            emit(self.candidate_outputs[:index])
        return self.candidate_outputs[:], {}


class InteractiveSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.trail = []
        self.engine = _Engine(self.trail)
        self.events = []
        self.torch = _Torch(self.trail)

    def emit(self, event):
        self.events.append(event)
        self.trail.append(("event", event["type"], event.get("runner")))

    def invoke(self, mode="optimized", prompt="Actual user prompt", **kwargs):
        with patch.dict("sys.modules", {"torch": self.torch}), \
                patch("time.sleep", side_effect=AssertionError("No artificial pauses")):
            return self.engine.compare(prompt, 64, mode, self.emit, **kwargs)

    def test_original_finishes_before_any_synthetic_prepare_even_for_odd_request(self):
        result = self.invoke(request_index=1)
        baseline_done = self.trail.index(("event", "completed", "baseline"))
        prepare_index = next(i for i, item in enumerate(self.trail) if item[0] == "prepare")
        synthetic_index = next(i for i, item in enumerate(self.trail) if item[0] == "synthetic")
        candidate_start = self.trail.index(("event", "started", "candidate"))
        self.assertLess(baseline_done, synthetic_index)
        self.assertLess(synthetic_index, prepare_index)
        self.assertLess(prepare_index, candidate_start)
        prepared = self.trail[prepare_index][1]
        self.assertEqual(prepared.kind, "synthetic")
        self.assertEqual(prepared.shape, (1, 3))
        self.assertEqual(result["comparison"]["order"], ["baseline", "candidate"])
        self.assertEqual(result["comparison"]["cooldown_s_per_run"], 0.0)
        self.assertFalse(result["comparison"]["per_request_full_byte_audit"])
        self.assertEqual(result["comparison"]["benchmark_protocol"], PROTOCOL)

    def test_strict_construction_waits_until_after_original_answer(self):
        self.invoke(mode="strict")
        self.assertLess(self.trail.index(("event", "completed", "baseline")),
                        self.trail.index(("construct_strict",)))
        self.assertTrue(any(item[0] == "strict_generate" for item in self.trail))
        self.assertFalse(any(item[0] == "candidate_generate" for item in self.trail))

    def test_queue_to_first_token_is_separate_from_generation_prefill(self):
        result = self.invoke(request_started_s=time.perf_counter() - 2.0)
        for side in ("baseline", "candidate"):
            metrics = result[side]["metrics"]
            self.assertGreaterEqual(metrics["request_to_first_token_s"], 2.0)
            self.assertGreater(metrics["request_to_first_token_s"], metrics["time_to_first_token_s"])
            self.assertTrue(metrics["finished_eos"])
        self.assertEqual(result["comparison"]["request_to_first_token_s"],
                         result["baseline"]["metrics"]["request_to_first_token_s"])

    def test_same_original_template_and_prompt_reach_encoding(self):
        self.invoke()
        tokenization = next(item for item in self.trail if item[0] == "tokenize")
        self.assertEqual(tokenization[1], [{"role": "user", "content": "Actual user prompt"}])
        self.assertEqual(tokenization[2], {"add_generation_prompt": True, "tokenize": True})
        real = next(item[1] for item in self.trail if item[0] == "baseline_generate")
        candidate = next(item[1] for item in self.trail if item[0] == "candidate_generate")
        self.assertIs(real, candidate)
        self.assertEqual(real.kind, "real")

    def test_comparison_tokenization_uses_actual_request_local_system_role(self):
        prompt = PromptText("Actual user prompt", "Design a responsive page.")
        self.invoke(prompt=prompt)
        tokenization = next(item for item in self.trail if item[0] == "tokenize")
        self.assertEqual(tokenization[1], [{"role": "system", "content": prompt.system_prompt},
                                           {"role": "user", "content": str(prompt)}])

    def test_outputs_are_checked_and_never_assumed_identical(self):
        self.engine.candidate_outputs = [8, 99]
        result = self.invoke()
        self.assertFalse(result["comparison"]["exact_match"])
        self.assertFalse(result["comparison"]["quality_equivalence_certified"])
        self.assertIsNone(result["comparison"]["same_output_latency_speedup"])

    def test_source_guard_failure_prevents_accepted_comparison(self):
        def changed_source():
            raise RuntimeError("Source changed")
        self.engine._source_guards = changed_source
        with self.assertRaisesRegex(RuntimeError, "Source changed"):
            self.invoke()
        self.assertFalse(any(e["type"] == "comparison" for e in self.events))

    def test_invalid_eos_continuation_stops_before_candidate_setup(self):
        self.engine.baseline_outputs = [99, 7]
        with self.assertRaisesRegex(RuntimeError, "continued after native EOS"):
            self.invoke()
        self.assertFalse(any(item[0] == "prepare" for item in self.trail))
        self.assertFalse(any(e["type"] == "comparison" for e in self.events))

    def test_oversized_context_rejected_before_generation_or_graph_preparation(self):
        self.engine.tokenizer.length = MAX_LEN - 8
        with self.assertRaisesRegex(RuntimeError, "must fit in " + str(MAX_LEN)):
            self.invoke()
        self.assertFalse(any(item[0] in ("baseline_generate", "prepare") for item in self.trail))

    def test_simultaneous_execution_is_rejected_before_gpu_import(self):
        with self.assertRaisesRegex(RuntimeError, "original first"):
            self.engine.compare("Prompt", 64, "optimized", self.emit, execution="simultaneous")
        self.assertEqual(self.trail, [])


if __name__ == "__main__":
    unittest.main()
