"""CPU tests for the private worker transport and prompt admission."""
import sys
from time import perf_counter
import unittest
from unittest.mock import patch

from lightreason_memory.backend import BackendError, QwenMemoryBackend, _Worker


class WorkerTest(unittest.TestCase):
    def test_request_can_contain_newlines_and_unicode(self):
        worker = _Worker([sys.executable, "-u", "-c", "import sys; print(sys.stdin.readline().strip())"])
        self.addCleanup(worker.close)
        payload = {"prompt": "First line\nSecond line. 漢字"}
        self.assertEqual(worker.request(payload, perf_counter() + 2), payload)

    def test_deadline_applies_while_waiting_for_worker(self):
        worker = _Worker([sys.executable, "-u", "-c", "import time; time.sleep(20)"])
        self.addCleanup(worker.close)
        with self.assertRaises(BackendError) as context:
            worker.request({"prompt": "irrelevant"}, perf_counter() + 0.05)
        self.assertEqual(context.exception.code, "timeout")

    def test_non_protocol_stdout_is_unavailable(self):
        worker = _Worker([sys.executable, "-u", "-c", "print('unexpected worker output')"])
        self.addCleanup(worker.close)
        with self.assertRaises(BackendError) as context:
            worker.request({}, perf_counter() + 2)
        self.assertEqual(context.exception.code, "unavailable")


class AdmissionTest(unittest.TestCase):
    def test_prompt_over_capacity_never_starts_cuda(self):
        backend = QwenMemoryBackend("unused-checkpoint")
        with patch.object(backend, "_ask", return_value={"tokens": 4000}) as ask:
            with self.assertRaises(BackendError) as context:
                backend.generate("Some evidence", "Follow the spec.", 33)
        self.assertEqual(context.exception.code, "capacity")
        self.assertEqual([call.args[0] for call in ask.call_args_list], ["tokenize"])

    def test_raw_counter_never_starts_cuda(self):
        backend = QwenMemoryBackend("unused-checkpoint")
        with patch.object(backend, "_ask", return_value={"tokens": 2}) as ask:
            self.assertEqual(backend.count_text_tokens("two words"), 2)
        self.assertEqual(ask.call_args.args[:2], ("tokenize", {"text": "two words"}))

    def test_missing_checkpoint_is_reported_without_gpu(self):
        backend = QwenMemoryBackend("unused-checkpoint", timeout_s=5)
        self.addCleanup(backend.close)
        with self.assertRaises(BackendError) as context:
            backend.count_tokens("Test prompt", "Use supplied evidence.")
        self.assertEqual(context.exception.code, "unavailable")
        self.assertEqual(backend._workers, {})


if __name__ == "__main__":
    unittest.main()
