import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from demo_chat import freetoken_inference
from demo_chat.messages import PromptText
from edge35b import freetoken_bench


class LiveComparisonTests(unittest.TestCase):
    def setUp(self):
        self.runner = freetoken_inference.FreeTokenStudioInference("model")
        self.prompt = PromptText("Build a weather app.", "Use HTML only.")
        self.events = []
        self.input_ids = [101, 202, 303]
        self.runner._input_ids = Mock(return_value=self.input_ids)
        self.runner.close = Mock()
        self.runner.initialize = Mock()

    def test_uses_one_prompt_and_budget_for_both_runners(self):
        order = Mock()
        baseline = dict(token_ids=[10, 20], metrics=dict(seconds=2, tokens_per_second=1,
                                                        decode_tokens_per_s=1, finished_eos=False))
        candidate = dict(token_ids=[10, 30], metrics=dict(seconds=1, tokens_per_second=2,
                                                         decode_tokens_per_s=2, finished_eos=False))
        order.attach_mock(self.runner.close, "close")
        order.attach_mock(self.runner.initialize, "initialize")
        self.runner._packed_replay = order.packed
        self.runner._stream = order.freetoken
        order.packed.return_value = baseline
        order.freetoken.return_value = candidate
        with patch.object(freetoken_inference, "perf_counter", side_effect=[100.0, 109.0]):
            result = self.runner.compare(self.prompt, 64, "optimized", self.events.append, 99.0)
        self.assertEqual([call[0] for call in order.mock_calls], ["close", "packed", "initialize", "freetoken"])
        order.packed.assert_called_once_with(self.prompt, self.input_ids, 64, False, 99.0, self.events.append)
        order.freetoken.assert_called_once_with(self.prompt, 64, "optimized", self.events.append, "candidate", 99.0, False)
        comparison = result["comparison"]
        self.assertEqual(comparison["setup_s"], 7.0)
        self.assertEqual(comparison["decode_rate_speedup"], 2)
        self.assertFalse(comparison["exact_match"])
        self.assertEqual(comparison["prefix_match_tokens"], 1)
        self.assertIs(self.events[-1], comparison)

    def test_reference_failure_restores_freetoken(self):
        self.runner._packed_replay = Mock(side_effect=RuntimeError("reference failed"))
        self.runner._stream = Mock()
        with self.assertRaisesRegex(RuntimeError, "reference failed"):
            self.runner.compare(self.prompt, 64, "optimized", self.events.append, 99.0)
        self.runner.initialize.assert_called_once_with(self.events.append)
        self.runner._stream.assert_not_called()

    def test_resident_comparison_hands_off_and_restores_packed(self):
        self.runner.resident = True
        order = Mock()
        self.runner.prepare_demo = order.prepare
        self.runner._select_runner = order.select
        self.runner._packed_replay = order.packed
        self.runner._stream = order.freetoken
        complete = dict(token_ids=[10, 20], metrics=dict(seconds=2, tokens_per_second=1,
                                                        decode_tokens_per_s=1, finished_eos=False))
        order.packed.return_value = complete
        order.freetoken.return_value = complete
        def on_event(event):
            self.events.append(event)
            if event["type"] == "comparison":
                order.comparison()

        result = self.runner.compare(self.prompt, 64, "optimized", on_event, 99.0)
        self.assertEqual([call[0] for call in order.mock_calls],
                         ["prepare", "select", "packed", "select", "freetoken", "comparison", "select"])
        self.assertEqual([call.args[0] for call in order.select.call_args_list],
                         ["packed", "freetoken", "packed"])
        self.runner.close.assert_not_called()
        self.runner.initialize.assert_not_called()
        self.assertTrue(result["comparison"]["exact_match"])
        self.assertIn("handoff_s", result["comparison"])

    def test_shared_cpu_rate_requires_the_configured_verifier_trace(self):
        self.runner.expert_cache_dir = Path("stage")
        self.runner.shared_transport = "native-shared-cpu"
        self.runner.shared_env = {"LIGHTREASON_CPU_READ_POLICY": "packed-both-prefetch"}
        with TemporaryDirectory() as directory:
            self.runner.control = Mock(name="control")
            self.runner.control.name = directory

            def complete(*_, **__):
                path = self.runner.width_policy_path
                settings = json.loads(path.read_text())
                self.assertEqual(settings["transport"], "native-shared-cpu")
                self.assertEqual(settings["cpu_read_policy"], "packed-both-prefetch")
                with path.with_suffix(".passes.jsonl").open("a") as log:
                    log.write(json.dumps(dict(settings, width="8", passes=[[8, 5, 200]])) + "\n")
                return dict(usage={"completion_tokens": 6}, cached_prompt_tokens=0, token_ids=list(range(6)),
                            timings={"predicted_ms": 200})

            with patch.object(freetoken_inference, "completion", side_effect=complete):
                run = self.runner._complete("optimized", "prompt", 3, 64, True)
            self.assertEqual(run["decode_tokens_per_s"], 25.0)

    def test_memory_auth_uses_child_environment_and_completion_header(self):
        runner = freetoken_inference.FreeTokenStudioInference("model", api_key="temporary-key")
        runner.load_tokenizer = Mock()
        runner._completion_prompt = Mock(return_value=("prompt", 3))
        child = Mock()
        child.poll.return_value = None
        child.pid = -1
        runner.child = child
        runner._start_monitored = Mock(return_value=child)
        with patch.object(Path, "glob", return_value=[Path(__file__)]), patch.object(
            freetoken_inference, "available_memory", return_value=100 * freetoken_inference.GIB,
        ), patch.object(freetoken_inference, "track_worker_processes"), patch.object(runner, "_complete") as complete:
            runner._initialize_freetoken(lambda _event: None)
        call = runner._start_monitored.call_args
        self.assertNotIn("temporary-key", call.args[0])
        self.assertEqual(call.kwargs["env"]["FREETOKEN_API_KEY"], "temporary-key")
        complete.assert_called_once_with("optimized", "prompt", 3, 3, False)

        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps({
            "usage": {"prompt_tokens": 3, "completion_tokens": 1}, "timings": {"cache_n": 0},
            "choices": [{"finish_reason": "stop", "text": "answer"}],
        })
        with patch.object(freetoken_bench, "urlopen", return_value=response) as request:
            self.runner._complete("optimized", "prompt", 3, 1, False)
            self.assertIsNone(request.call_args.args[0].get_header("Authorization"))
            runner._complete("optimized", "prompt", 3, 1, False)
            self.assertEqual(request.call_args.args[0].get_header("Authorization"), "Bearer temporary-key")


if __name__ == "__main__":
    unittest.main()
