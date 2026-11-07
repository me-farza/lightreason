"""CPU-only tests for synthetic demo preparation and live graph admission."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from demo_chat.prepared import PreparedLiveInference, _buckets
from demo_chat.interactive import InteractiveLiveInference
from demo_chat.inference import MAX_LEN, TREE_RESERVE
from demo_chat.messages import PromptText


class _Graph:
    pass


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        # The test prompt only selects a public length; no inference occurs.
        length = int(messages[-1]["content"])
        length += sum(len(message["content"]) for message in messages if message["role"] == "system")
        return [9] * length


class _Torch:
    long = "long"
    cuda = SimpleNamespace(synchronize=lambda: None)

    @staticmethod
    def zeros(shape, **kwargs):
        return SimpleNamespace(shape=shape, kind="synthetic-zero")

    @staticmethod
    def inference_mode(*args):
        return nullcontext()

    @staticmethod
    def no_grad():
        return nullcontext()


class _Fast:
    node_budget = 64
    draft_tokens = 15

    def __init__(self, trail):
        self.trail = trail
        choices = tuple(sorted({256, 512, 1024, 2048, MAX_LEN}))
        self.verifier = SimpleNamespace(attention_buckets=choices, graphs={})
        self.target_prefill_component = SimpleNamespace(graph=SimpleNamespace(entries={}))
        self.proposer = SimpleNamespace(conditioned=SimpleNamespace(
            attention_buckets=choices, graphs={},
            prefill_component=SimpleNamespace(graph=SimpleNamespace(entries={}))))
        self.lengths = OrderedDict()

    def prepare(self, ids, budget):
        self.trail.append(("fast_prepare", ids.kind, ids.shape[1], budget))
        length = ids.shape[1]
        aux = self.proposer.conditioned
        count = 0
        for bucket in _buckets(self.verifier.attention_buckets, length + 64, length + budget + 62):
            if (64, bucket) not in self.verifier.graphs:
                self.verifier.graphs[(64, bucket)] = _Graph()
                count += 1
        for bucket in _buckets(aux.attention_buckets, length + 15, length + budget + 13):
            for width in range(15, 31):
                key = (15, width, bucket)
                if key not in aux.graphs:
                    aux.graphs[key] = {"graph": _Graph()}
                    count += 1
        if length not in self.lengths:
            if len(self.lengths) >= 4:
                evicted, _ = self.lengths.popitem(last=False)
                self.target_prefill_component.graph.entries.pop(evicted)
                aux.prefill_component.graph.entries.pop(evicted + 1)
            self.lengths[length] = None
            self.target_prefill_component.graph.entries[length] = SimpleNamespace(graph=_Graph())
            aux.prefill_component.graph.entries[length + 1] = SimpleNamespace(graph=_Graph())
            count += 2
        else:
            self.lengths.move_to_end(length)
        return {"new_graphs": count, "setup_s": 0.0}


class _Strict:
    def __init__(self, trail):
        self.trail = trail
        self.graphs = {}

    def prepare(self, ids, budget):
        self.trail.append(("strict_prepare", ids.kind, ids.shape[1], budget))
        wanted = {((p // 128) + 1) * 128 for p in range(ids.shape[1], ids.shape[1] + budget - 1)}
        missing = wanted - self.graphs.keys()
        self.graphs.update({key: _Graph() for key in missing})
        return {"new_graphs": len(missing), "setup_s": 0.0}


class _Engine(PreparedLiveInference):
    def __init__(self, turbo=True):
        paths = {"draft_path": "test-draft", "checkpoint_path": "test-checkpoint"} if turbo else {}
        super().__init__(**paths)
        self.trail = []
        self.ready = True
        self.tokenizer = _Tokenizer()
        self.model = SimpleNamespace(device="test-cuda")
        self.fast = _Fast(self.trail) if turbo else None
        self.strict = None
        self._demo_warm_lengths = {}
        self._demo_admitted_graphs = {}
        self._demo_preparing = False

    def _thread(self):
        pass

    def _request_guards(self):
        self.trail.append(("guard",))

    def _source_guards(self):
        self.trail.append(("source_guard",))

    def _warm_original(self, synthetic):
        self.trail.append(("original_warm", synthetic.kind, synthetic.shape[1]))

    def _ensure_strict(self, callback):
        if self.strict is None:
            self.trail.append(("construct_strict",))
            self.strict = _Strict(self.trail)


class PreparedDemoTests(unittest.TestCase):
    def setUp(self):
        self.engine = _Engine()
        self.events = []

    def prepare(self, length=44, budget=256, mode="optimized"):
        with patch.dict("sys.modules", {"torch": _Torch()}):
            return self.engine.prepare_demo(str(length), budget, mode, self.events.append)

    def test_qwen_only_preparation_admits_exact_without_auxiliary_graphs(self):
        self.engine = _Engine(turbo=False)
        result = self.prepare(mode="strict")
        self.assertTrue(result["prepared"])
        self.assertEqual(result["prepared_modes"], ["strict"])
        self.assertEqual(set(result["receipts"]), {"strict"})
        self.assertIn(("original_warm", "synthetic-zero", 44), self.engine.trail)
        self.assertIsNone(self.engine.fast)
        with self.assertRaisesRegex(RuntimeError, "available Lightreason runner"):
            self.engine.is_prepared("44", 256, "optimized")

    def test_both_candidates_and_original_are_prepared_with_synthetic_values_only(self):
        result = self.prepare()
        self.assertTrue(result["prepared"])
        self.assertEqual(result["prepared_modes"], ["optimized", "strict"])
        self.assertEqual(result["receipts"]["optimized"]["new_graphs"], 36)
        self.assertFalse(result["answers_cached"])
        self.assertTrue(result["synthetic_only"])
        self.assertIn(("original_warm", "synthetic-zero", 44), self.engine.trail)
        self.assertIn(("fast_prepare", "synthetic-zero", 44, 256), self.engine.trail)
        self.assertIn(("strict_prepare", "synthetic-zero", 44, 256), self.engine.trail)

    def test_smaller_budgets_reuse_families_but_larger_unprepared_family_is_rejected(self):
        self.prepare(budget=128)
        self.assertTrue(self.engine.is_prepared("44", 64, "strict")["prepared"])
        self.assertTrue(self.engine.is_prepared("44", 128, "optimized")["prepared"])
        self.assertFalse(self.engine.is_prepared("44", 256, "optimized")["prepared"])

    def test_preparation_counts_the_actual_system_role_in_public_shape(self):
        prompt = PromptText("44", "style")
        with patch.dict("sys.modules", {"torch": _Torch()}):
            result = self.engine.prepare_demo(prompt, 256, "optimized", self.events.append)
        self.assertEqual(result["prompt_tokens"], 49)
        self.assertIn(("original_warm", "synthetic-zero", 49), self.engine.trail)
        self.assertIn(("fast_prepare", "synthetic-zero", 49, 256), self.engine.trail)
        self.assertTrue(self.engine.is_prepared(prompt, 256, "optimized")["prepared"])
        self.assertFalse(self.engine.is_prepared("44", 256, "optimized")["prepared"])

    def test_context_beyond_old_2048_capacity_can_prepare_the_extended_family(self):
        if MAX_LEN <= 2048:
            self.skipTest("Extended context is not configured")
        self.prepare(length=44, budget=2048)
        self.assertIn((64, MAX_LEN), self.engine.fast.verifier.graphs)
        self.assertTrue(self.engine.is_prepared("44", 2048, "optimized")["prepared"])

    def test_eviction_invalidates_readiness_without_retaining_old_graph_objects(self):
        self.prepare(length=44)
        for length in (45, 46, 47, 48):
            self.prepare(length=length)
        self.assertEqual(len(self.engine.fast.lengths), 4)
        self.assertFalse(self.engine.is_prepared("44", 256, "optimized")["prepared"])
        self.assertTrue(self.engine.is_prepared("48", 256, "optimized")["prepared"])

    def test_replacing_live_graph_or_runner_invalidates_admission(self):
        self.prepare()
        entry = self.engine.fast.proposer.conditioned.graphs[(15, 15, 256)]
        entry["graph"] = _Graph()
        self.assertFalse(self.engine.is_prepared("44", 256, "optimized")["prepared"])
        self.prepare()
        self.assertTrue(self.engine.is_prepared("44", 256, "optimized")["prepared"])
        self.engine.strict = _Strict(self.engine.trail)
        self.assertFalse(self.engine.is_prepared("44", 256, "strict")["prepared"])

    def test_compare_rejects_unprepared_shape_before_any_generation(self):
        with patch.object(InteractiveLiveInference, "compare", side_effect=AssertionError("Generation must not start")):
            with self.assertRaisesRegex(RuntimeError, "Prepare this demo"):
                self.engine.compare("44", 256, "optimized", self.events.append)
        self.assertFalse(self.events)

    def test_prepared_compare_keeps_original_outputs_and_reports_zero_new_graphs(self):
        self.prepare()
        self.events.clear()
        token_ids = [7, 8, 99]

        def original_compare(engine, prompt, budget, mode, callback, **kwargs):
            comparison = {"type": "comparison", "prepared": {"new_graphs": 0}, "exact_match": True}
            callback({"type": "status", "state": "preparing"})
            callback(comparison)
            return {"baseline": {"token_ids": token_ids}, "candidate": {"token_ids": token_ids},
                    "comparison": comparison, "integrity_gates": {}}

        with patch.object(InteractiveLiveInference, "compare", original_compare):
            result = self.engine.compare("44", 256, "optimized", self.events.append)
        self.assertIs(result["baseline"]["token_ids"], token_ids)
        self.assertTrue(result["comparison"]["demo_prepared"])
        self.assertEqual(result["comparison"]["new_gpu_graphs_during_comparison"], 0)
        self.assertFalse(result["comparison"]["answers_cached"])
        self.assertEqual(self.events[0]["state"], "demo_ready")
        self.assertEqual(self.events[-1]["type"], "comparison")

    def test_unexpected_capture_prevents_an_accepted_comparison(self):
        self.prepare()
        self.events.clear()

        def captured(engine, prompt, budget, mode, callback, **kwargs):
            engine.fast.verifier.graphs[(64, 1024)] = _Graph()
            comparison = {"type": "comparison", "prepared": {"new_graphs": 1}}
            callback(comparison)
            return {"comparison": comparison, "integrity_gates": {}}

        with patch.object(InteractiveLiveInference, "compare", captured):
            with self.assertRaisesRegex(RuntimeError, "unexpected GPU graph capture"):
                self.engine.compare("44", 256, "optimized", self.events.append)
        self.assertFalse(any(e["type"] == "comparison" for e in self.events))

    def test_invalid_public_shape_is_rejected_before_any_gpu_work(self):
        for prompt, budget in (("0", 256), (str(MAX_LEN - TREE_RESERVE), 128),
                               ("44", True), ("44", MAX_LEN - TREE_RESERVE)):
            with self.subTest(prompt=prompt, budget=budget), self.assertRaises(RuntimeError):
                self.engine.prepare_demo(prompt, budget, "optimized", self.events.append)
        self.assertFalse(any(item[0] in ("original_warm", "fast_prepare", "strict_prepare")
                             for item in self.engine.trail))

    def test_failed_preparation_does_not_claim_demo_ready(self):
        self.engine.fast.prepare = lambda *args: (_ for _ in ()).throw(RuntimeError("expected preparation failure"))
        with self.assertRaisesRegex(RuntimeError, "expected preparation failure"):
            self.prepare()
        self.assertFalse(self.engine._demo_preparing)
        self.assertFalse(self.engine.is_prepared("44", 256, "optimized")["prepared"])
        self.assertFalse(any(e["type"] == "prepared" for e in self.events))


if __name__ == "__main__":
    unittest.main()
