"""CPU-only protocol tests. FakeEngine is test data, never a product runner."""
from __future__ import annotations

from argparse import Namespace
import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

try:
    from demo_chat import server
except ModuleNotFoundError:
    import server
from demo_chat.inference import MAX_LEN, TREE_RESERVE
from demo_chat.messages import PromptText, chat_messages, DEFAULT_DESIGN_PROMPT


def fake_token_count(prompt):
    return sum(len(message["content"]) for message in chat_messages(prompt))


class RequestValidationTests(unittest.TestCase):
    def test_qwen_only_worker_becomes_ready_without_startup_preparation(self):
        class LoadedEngine:
            def __init__(self, **kwargs):
                self.loaded = False

            def initialize(self, callback):
                self.loaded = True

        args = Namespace(model_path="TEST_ONLY", draft_path=None, checkpoint_path=None, cooldown_seconds=0)
        app = server.Application(args, engine_factory=LoadedEngine)

        def stop_worker():
            raise RuntimeError("worker startup finished")

        app.jobs.get = stop_worker
        with self.assertRaisesRegex(RuntimeError, "worker startup finished"):
            app.worker()
        self.assertTrue(app.engine.loaded)
        self.assertTrue(app.snapshot()["ready"])
        self.assertTrue(app.snapshot()["models_loaded"])
        self.assertFalse(app.snapshot()["demo_ready"])
        self.assertIsNone(app.snapshot()["startup_prepared"])

    def test_budget_and_prompt_boundaries(self):
        for budget in (64, 128, 256, 512, 1024):
            self.assertEqual(server.validate_request({"prompt": "  hello  ", "max_new_tokens": budget}),
                             ("hello", budget, "optimized", "sequential"))
        self.assertEqual(server.validate_request({"prompt": "x" * 8000, "mode": "strict"}),
                         ("x" * 8000, 128, "strict", "sequential"))
        self.assertEqual(server.validate_request({"prompt": "hello", "execution": "sequential"}),
                         ("hello", 128, "optimized", "sequential"))
        for body in (None, [], {}, {"prompt": " "}, {"prompt": 1}, {"prompt": "x" * 8001},
                     {"prompt": "x", "max_new_tokens": True}, {"prompt": "x", "max_new_tokens": 63},
                     {"prompt": "x", "max_new_tokens": 2048}, {"prompt": "x", "mode": "unknown"},
                     {"prompt": "x", "execution": "simultaneous"},
                     {"prompt": "x", "execution": "parallel"}, {"prompt": "x", "execution": None},
                     {"prompt": "x", "execution": True}):
            with self.subTest(body=str(body)[:70]), self.assertRaises(ValueError):
                server.validate_request(body)

    def test_submit_rejects_unready_and_busy(self):
        app = server.Application(Namespace())
        self.assertEqual(app.snapshot()["available_modes"], ["original", "strict"])
        self.assertEqual(app.default_mode, "strict")
        validate = lambda body: server.validate_request(body, default_mode=app.default_mode,
                                                        available_modes=app.available_modes)
        self.assertEqual(validate({"prompt": "hello"})[2], "strict")
        with self.assertRaisesRegex(ValueError, "available Lightreason runner"):
            validate({"prompt": "hello", "mode": "optimized"})
        with self.assertRaises(RuntimeError):
            app.submit(("x", 128, "strict", "sequential"))
        with app.lock:
            app.state.update(ready=True, busy=True)
        with self.assertRaises(BlockingIOError):
            app.submit(("x", 128, "strict", "sequential"))
        self.assertTrue(app.jobs.empty())
        self.assertEqual(app.request_index, 0)

    def test_new_app_discards_old_page_and_edit_requires_explicit_context(self):
        prompt, _, _, _ = server.validate_request({"prompt": "Build a calculator", "app_action": "new",
                                                   "previous_code": "<h1>Old weather dashboard</h1>",
                                                   "system_prompt": "Use responsive grids."})
        self.assertIsInstance(prompt, PromptText)
        self.assertEqual(str(prompt), "Build a calculator")
        self.assertEqual(prompt.system_prompt, "Use responsive grids.")
        self.assertNotIn("Old weather", prompt)
        for body in ({"prompt": "Edit it", "app_action": "edit"},
                     {"prompt": "Edit it", "app_action": "edit", "previous_code": " "},
                     {"prompt": "Edit it", "app_action": "unknown"}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                server.validate_request(body)

    def test_system_instructions_have_a_validated_actual_role(self):
        prompt = server.validate_request({"prompt": "Build a shop", "system_prompt": "Use responsive grids."})[0]
        self.assertEqual(chat_messages(prompt), [{"role": "system", "content": "Use responsive grids."},
                                                 {"role": "user", "content": "Build a shop"}])
        for system in (False, 3, {}, [], "x" * 4001):
            with self.subTest(system=str(system)[:30]), self.assertRaises(ValueError):
                server.validate_request({"prompt": "Build a shop", "system_prompt": system})


class HTTPProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp.name)
        (self.temp_path / "static").mkdir()
        (self.temp_path / "static" / "index.html").write_text("<main>live test</main>", encoding="utf-8")
        self.old_root, server.ROOT = server.ROOT, self.temp_path
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.prepare_calls = []
        self.readiness_calls = []
        self.allow_all_prepared = True
        self.request_started_times = []
        self.generation_calls = []
        self.generation_started_times = []
        self.worker_trail = []
        self.output_policies = []
        self.tokenization_messages = []
        self.role_calls = []
        owner = self

        class FakeTokenizer:
            @staticmethod
            def apply_chat_template(messages, **kwargs):
                owner.tokenization_messages.append([dict(message) for message in messages])
                return [1] * sum(len(message["content"]) for message in messages)

            @staticmethod
            def decode(ids, skip_special_tokens):
                assert skip_special_tokens
                return "".join(chr(value) for value in ids)

        class FakeEngine:
            tokenizer = FakeTokenizer()

            def __init__(self, **kwargs):
                self.prepared_shapes = set()

            def initialize(self, callback):
                callback({"type": "status", "state": "loading", "detail": "test initialization"})

            def is_prepared(self, prompt, budget, mode):
                return owner.allow_all_prepared or (fake_token_count(prompt), budget) in self.prepared_shapes

            def prepared_status(self, prompt, budget, mode):
                owner.readiness_calls.append((prompt, budget, mode))
                ready = self.is_prepared(prompt, budget, mode)
                return {"prepared": ready, "demo_ready": ready,
                        "prepared_modes": ["optimized", "strict"] if ready else [],
                        "detail": "Both test runners warmed" if ready else "Prepare this configuration"}

            def prepare_demo(self, prompt, budget, mode, callback):
                owner.role_calls.append(("prepare", chat_messages(prompt)))
                owner.worker_trail.append(("prepare_started", prompt, budget, mode, time.perf_counter()))
                owner.prepare_calls.append((prompt, budget, mode))
                callback({"type": "status", "state": "preparing", "detail": "Preparing both test runners"})
                if prompt == "hold preparation":
                    owner.entered.set()
                    if not owner.release.wait(2):
                        raise TimeoutError("test preparation barrier was not released")
                if prompt == "fail preparation":
                    raise ValueError("expected test preparation failure")
                self.prepared_shapes.add((fake_token_count(prompt), budget))
                owner.worker_trail.append(("prepare_finished", prompt, budget, mode, time.perf_counter()))
                result = {"type": "prepared", "ready": True, "prepared": True, "demo_ready": True,
                          "prepared_modes": ["optimized", "strict"], "prompt_tokens": fake_token_count(prompt),
                          "budget": budget, "setup_s": 0.0, "synthetic_only": True,
                          "detail": "Both test runners warmed"}
                callback(result)
                return result

            def compare(self, prompt, budget, mode, callback, request_index, execution, request_started_s,
                        require_prepared=False, until_finished=False):
                owner.role_calls.append(("compare", chat_messages(prompt)))
                owner.output_policies.append(until_finished)
                owner.assertTrue(require_prepared, "HTTP comparisons must require previously prepared graphs")
                owner.assertTrue(self.is_prepared(prompt, budget, mode))
                owner.worker_trail.append(("compare_started", prompt, budget, mode, time.perf_counter()))
                owner.calls.append((prompt, budget, mode, request_index, execution))
                owner.request_started_times.append(request_started_s)
                owner.assertIsInstance(request_started_s, (int, float))
                owner.assertGreater(request_started_s, 0)
                owner.assertLessEqual(request_started_s, time.perf_counter())
                owner.assertEqual(execution, "sequential")
                if prompt == "hold":
                    owner.entered.set()
                    if not owner.release.wait(2):
                        raise TimeoutError("test barrier was not released")
                if prompt == "fail":
                    raise ValueError("expected test inference failure")
                def started(runner):
                    callback({"type": "started", "runner": runner, "execution": execution})

                def token(runner, ids):
                    callback({"type": "token_ids", "runner": runner, "token_ids": ids,
                              "tokens": len(ids), "elapsed_s": len(ids) * 0.01})

                def completed(runner):
                    callback({"type": "completed", "runner": runner, "text": "AB<script>",
                              "token_ids": [65, 66], "metrics": {"seconds": 0.02, "tokens": 2,
                              "tokens_per_second": 100, "finished_eos": False}})

                for runner in ("baseline", "candidate"):
                    started(runner)
                    token(runner, [65])
                    token(runner, [65, 66])
                    completed(runner)
                comparison = {"type": "comparison", "exact_match": True, "speedup": 1.0,
                              "token_rate_speedup": 1.0, "execution": execution,
                              "isolated_benchmark": execution == "sequential"}
                callback(comparison)
                return {"comparison": comparison}

            def generate(self, prompt, budget, mode, callback, request_started_s, require_prepared,
                         until_finished=False):
                owner.role_calls.append(("generate", chat_messages(prompt)))
                owner.assertTrue(require_prepared)
                owner.assertTrue(self.is_prepared(prompt, budget, mode))
                owner.worker_trail.append(("generate_started", prompt, budget, mode, time.perf_counter()))
                owner.generation_calls.append((prompt, budget, mode))
                owner.generation_started_times.append(request_started_s)
                owner.output_policies.append(until_finished)
                callback({"type": "started", "runner": "single", "mode": mode})
                callback({"type": "token_ids", "runner": "single", "token_ids": [65],
                          "tokens": 1, "elapsed_s": 0.01})
                result = {"type": "generation", "runner": "single", "mode": mode,
                          "text": "A", "token_ids": [65], "metrics": {"seconds": 0.01,
                          "tokens": 1, "finished_eos": True, "stop_reason": "eos"}}
                callback(dict(result, type="completed"))
                callback(result)
                return {"generation": result}

        args = Namespace(model_path="TEST_ONLY", draft_path="TEST_ONLY", checkpoint_path="TEST_ONLY",
                         cooldown_seconds=0, log_dir=str(self.temp_path / "runs"))
        self.app = server.Application(args, engine_factory=FakeEngine)
        threading.Thread(target=self.app.worker, daemon=True).start()
        deadline = time.monotonic() + 2
        while not self.app.snapshot()["ready"] and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertTrue(self.app.snapshot()["ready"])
        self.assertTrue(self.prepare_calls, "Startup must prepare both default demo runners")
        self.assertEqual(len(self.prepare_calls), 2)
        self.startup_prepare_count = len(self.prepare_calls)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        self.httpd.app = self.app
        self.httpd.daemon_threads = True
        self.http_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.http_thread.start()
        self.port = self.httpd.server_address[1]
        self.host = f"127.0.0.1:{self.port}"

    def tearDown(self):
        self.release.set()
        deadline = time.monotonic() + 2
        while self.app.snapshot()["busy"] and time.monotonic() < deadline:
            time.sleep(0.005)
        self.httpd.shutdown()
        self.httpd.server_close()
        self.http_thread.join(timeout=1)
        server.ROOT = self.old_root
        self.temp.cleanup()

    def request(self, method, path, payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        body = json.dumps(payload).encode() if payload is not None else None
        request_headers = {"Host": self.host}
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        request_headers.update(headers or {})
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def test_status_and_static_allowlist(self):
        status, headers, body = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        state = json.loads(body)
        self.assertTrue(state["ready"])
        self.assertFalse(state["supports_simultaneous_streaming"])
        self.assertEqual(state["default_execution"], "sequential")
        self.assertFalse(state["preparation_required"])
        self.assertTrue(state["automatic_setup"])
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        status, headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"<main>live test</main>")
        self.assertIn("frame-src about:", headers["Content-Security-Policy"])
        self.assertIn("geolocation=()", headers["Permissions-Policy"])
        for path in ("/../server.py", "/%2e%2e/server.py", "/static/../server.py", "/unknown"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path)[0], 404)

    def test_host_and_cross_origin_requests_rejected(self):
        for headers in ({"Host": "attacker.invalid"}, {"Host": "localhost.attacker.invalid"},
                        {"Origin": "https://attacker.invalid"}, {"Origin": "null"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request("GET", "/api/status", headers=headers)[0], 403)
                for path in ("/api/compare", "/api/generate", "/api/prepare", "/api/check-ready"):
                    self.assertEqual(self.request("POST", path, {"prompt": "hello"}, headers)[0], 403)
        self.assertEqual(self.request("GET", "/api/status", headers={"Origin": f"http://{self.host}"})[0], 200)

    def test_invalid_post_rejected_before_gpu_queue(self):
        for payload in ({"prompt": ""}, {"prompt": "x", "max_new_tokens": True},
                        {"prompt": "x", "execution": "unknown"},
                        {"prompt": "x", "execution": "simultaneous"}, ["x"]):
            for path in ("/api/compare", "/api/generate", "/api/prepare", "/api/check-ready"):
                self.assertEqual(self.request("POST", path, payload)[0], 400)
        self.assertEqual(self.request("POST", "/api/compare", {"prompt": "x"},
                                      {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/unknown", {"prompt": "x"})[0], 404)
        self.assertEqual(self.calls, [])
        self.assertEqual(len(self.prepare_calls), self.startup_prepare_count)

    def test_cold_compare_automatically_warms_before_any_generation(self):
        self.allow_all_prepared = False
        prompt = "unprepared test prompt"
        status, _, body = self.request("POST", "/api/compare", {"prompt": prompt})
        self.assertEqual(status, 200)
        events = [json.loads(line) for line in body.splitlines()]
        self.assertEqual(events[-1], {"type": "done", "success": True})
        self.assertEqual(self.prepare_calls[-1], (prompt, 128, "optimized"))
        self.assertEqual(self.calls, [(prompt, 128, "optimized", 0, "sequential")])
        trail = [entry for entry in self.worker_trail if entry[1] == prompt]
        self.assertEqual([entry[0] for entry in trail],
                         ["prepare_started", "prepare_finished", "compare_started"])
        self.assertLessEqual(self.request_started_times[0], trail[0][-1],
                             "Automatic setup must remain part of request-to-first-token time")
        self.assertFalse(any(e["type"] == "prepared" for e in events),
                         "Automatic setup must not expose a manual preparation step")
        first_started = next(i for i, e in enumerate(events) if e["type"] == "started")
        starting = [e for e in events[:first_started] if e["type"] == "status"]
        self.assertTrue(starting)
        self.assertTrue(all(e["detail"] == "Starting Qwen… setting up GPU execution" for e in starting))
        comparison = next(e for e in events if e["type"] == "comparison")
        self.assertTrue(comparison["auto_setup_required"])
        self.assertGreaterEqual(comparison["auto_setup_s"], 0)
        record = json.loads((self.temp_path / "runs" / "comparisons.jsonl").read_text().splitlines()[-1])
        self.assertEqual(record["result"]["auto_setup_s"], comparison["auto_setup_s"])
        self.assertTrue(record["result"]["comparison"]["auto_setup_required"])
        self.assertEqual(len(self.prepare_calls), self.startup_prepare_count + 1)
        self.assertFalse(self.app.snapshot()["busy"])

        # The same warm configuration immediately reuses its admission.
        count = len(self.prepare_calls)
        status, _, body = self.request("POST", "/api/compare", {"prompt": prompt})
        self.assertEqual(status, 200)
        events = [json.loads(line) for line in body.splitlines()]
        comparison = next(e for e in events if e["type"] == "comparison")
        self.assertFalse(comparison["auto_setup_required"])
        self.assertEqual(comparison["auto_setup_s"], 0.0)
        self.assertEqual(len(self.prepare_calls), count)

    def test_cold_generate_automatically_prepares_same_update_context_and_resolved_budget(self):
        self.allow_all_prepared = False
        body = {"prompt": "Make it blue", "previous_code": "<h1>Hello</h1>",
                "mode": "original", "max_new_tokens": None, "system_prompt": "Use responsive grids."}
        effective_prompt = server.validate_request(body, allow_original=True)[0]
        capacity = MAX_LEN - TREE_RESERVE - fake_token_count(effective_prompt)
        status, _, wire = self.request("POST", "/api/generate", body)
        self.assertEqual(status, 200)
        events = [json.loads(line) for line in wire.splitlines()]
        self.assertEqual(events[-1], {"type": "done", "success": True})
        self.assertEqual(self.prepare_calls[-1], (effective_prompt, capacity, "original"))
        self.assertEqual(self.generation_calls, [(effective_prompt, capacity, "original")])
        self.assertEqual(self.prepare_calls[-1][0].system_prompt, body["system_prompt"])
        self.assertEqual(self.generation_calls[-1][0].system_prompt, body["system_prompt"])
        self.assertEqual(self.output_policies, [True])
        trail = [entry for entry in self.worker_trail if entry[1] == effective_prompt]
        self.assertEqual([entry[0] for entry in trail],
                         ["prepare_started", "prepare_finished", "generate_started"])
        self.assertLessEqual(self.generation_started_times[0], trail[0][-1])
        generation = next(e for e in events if e["type"] == "generation")
        self.assertTrue(generation["auto_setup_required"])
        self.assertGreaterEqual(generation["auto_setup_s"], 0)
        self.assertFalse(any(e["type"] == "prepared" for e in events))
        self.assertEqual([e["runner"] for e in events if e["type"] == "started"], ["single"])
        record = json.loads((self.temp_path / "runs" / "generations.jsonl").read_text().splitlines()[-1])
        self.assertEqual(record["prompt"], effective_prompt)
        self.assertEqual(record["max_new_tokens"], capacity)
        self.assertTrue(record["output_policy"]["until_finished"])
        self.assertTrue(record["result"]["generation"]["auto_setup_required"])

    def test_automatic_setup_failure_unlocks_without_generating(self):
        self.allow_all_prepared = False
        with self.assertLogs(level="ERROR"):
            status, _, body = self.request("POST", "/api/generate", {"prompt": "fail preparation"})
        self.assertEqual(status, 200)
        events = [json.loads(line) for line in body.splitlines()]
        self.assertEqual(events[-1], {"type": "done", "success": False})
        self.assertTrue(any(e["type"] == "error" for e in events))
        self.assertFalse(any(e["type"] in ("started", "token", "completed", "generation") for e in events))
        self.assertEqual(self.generation_calls, [])
        self.assertFalse(self.app.snapshot()["busy"])

    def test_prepare_warms_both_modes_without_generating_answers(self):
        self.allow_all_prepared = False
        request = {"prompt": "prepared shape", "max_new_tokens": 64, "mode": "optimized"}
        status, _, body = self.request("POST", "/api/check-ready", request)
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["prepared"])
        status, headers, body = self.request("POST", "/api/prepare", request)
        self.assertEqual(status, 200)
        self.assertIn("application/x-ndjson", headers["Content-Type"])
        events = [json.loads(line) for line in body.splitlines()]
        self.assertEqual(events[-1], {"type": "done", "success": True})
        prepared = [e for e in events if e["type"] == "prepared"]
        self.assertEqual(len(prepared), 1, "Preparation result must not be emitted twice")
        self.assertTrue(prepared[0].get("prepared", prepared[0].get("demo_ready")))
        self.assertEqual(prepared[0]["prepared_modes"], ["optimized", "strict"])
        self.assertTrue(prepared[0]["synthetic_only"])
        self.assertFalse(any(e["type"] in ("token", "completed", "comparison", "started") for e in events))
        self.assertEqual(self.calls, [])
        for mode in ("optimized", "strict"):
            status, _, body = self.request("POST", "/api/check-ready", dict(request, mode=mode))
            self.assertEqual(status, 200)
            self.assertTrue(json.loads(body)["prepared"])
        for changed in (dict(request, prompt="a different length"), dict(request, max_new_tokens=128)):
            status, _, body = self.request("POST", "/api/check-ready", changed)
            self.assertEqual(status, 200)
            self.assertFalse(json.loads(body)["prepared"])
        self.assertFalse(self.app.snapshot()["busy"])
        self.assertFalse((self.temp_path / "runs" / "comparisons.jsonl").exists())
        self.assertTrue((self.temp_path / "runs" / "preparations.jsonl").exists())

    def test_preparation_and_comparison_share_busy_lock(self):
        first = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        first.request("POST", "/api/prepare", body=json.dumps({"prompt": "hold preparation"}),
                      headers={"Host": self.host, "Content-Type": "application/json"})
        response = first.getresponse()
        self.assertEqual(response.status, 200)
        self.assertTrue(self.entered.wait(1))
        before_reads = len(self.readiness_calls)
        for path in ("/api/compare", "/api/generate", "/api/prepare"):
            self.assertEqual(self.request("POST", path, {"prompt": "second"})[0], 409)
        status, _, body = self.request("POST", "/api/check-ready", {"prompt": "second"})
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["prepared"])
        self.assertEqual(len(self.readiness_calls), before_reads,
                         "Readiness must not inspect mutable graph banks while the GPU worker is busy")
        self.assertEqual(self.calls, [])
        self.release.set()
        events = [json.loads(line) for line in response.read().splitlines()]
        first.close()
        self.assertEqual(events[-1], {"type": "done", "success": True})
        self.assertFalse(self.app.snapshot()["busy"])

    def test_readiness_unavailable_during_initialization_avoids_engine_reads(self):
        with self.app.lock:
            self.app.state.update(ready=False, state="loading")
        before_reads = len(self.readiness_calls)
        status, _, body = self.request("POST", "/api/check-ready", {"prompt": "any prompt"})
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["prepared"])
        self.assertEqual(len(self.readiness_calls), before_reads)
        with self.app.lock:
            self.app.state.update(ready=True, state="ready")

    def test_startup_does_not_report_ready_before_default_preparation_finishes(self):
        preparing = threading.Event()
        release = threading.Event()
        calls = []

        class StartupEngine:
            tokenizer = self.app.engine.tokenizer
            def __init__(self, **kwargs):
                self.prepared = False

            def initialize(self, callback):
                callback({"type": "status", "state": "ready", "detail": "Models loaded"})

            def prepare_demo(self, prompt, budget, mode, callback):
                calls.append((prompt, budget, mode))
                preparing.set()
                if not release.wait(2):
                    raise TimeoutError("startup preparation test barrier was not released")
                self.prepared = True
                return {"type": "prepared", "prepared": True, "ready": True,
                        "demo_ready": True, "prepared_modes": ["optimized", "strict"],
                        "synthetic_only": True, "setup_s": 0.0}

            def prepared_status(self, prompt, budget, mode):
                return {"prepared": self.prepared}

        args = Namespace(model_path="TEST_ONLY", draft_path="TEST_ONLY", checkpoint_path="TEST_ONLY",
                         cooldown_seconds=0, log_dir=str(self.temp_path / "startup-runs"))
        app = server.Application(args, engine_factory=StartupEngine)
        worker = threading.Thread(target=app.worker, daemon=True)
        worker.start()
        try:
            self.assertTrue(preparing.wait(1))
            self.assertFalse(app.snapshot()["ready"], "Loaded models alone are not demo-ready")
        finally:
            release.set()
        deadline = time.monotonic() + 2
        while not app.snapshot()["ready"] and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertTrue(app.snapshot()["ready"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(str(calls[0][0]), "x")
        self.assertIsNone(calls[0][0].system_prompt)
        self.assertEqual(calls[0][1:], (MAX_LEN - TREE_RESERVE - 1, "optimized"))
        self.assertEqual(str(calls[1][0]), server.DEFAULT_PROMPT)
        self.assertEqual(calls[1][0].system_prompt, DEFAULT_DESIGN_PROMPT)
        self.assertEqual(calls[1][1:], (MAX_LEN - TREE_RESERVE - fake_token_count(calls[1][0]), "optimized"))

    def test_preparation_failure_unlocks_without_generating_or_claiming_ready(self):
        self.allow_all_prepared = False
        with self.assertLogs(level="ERROR"):
            status, _, body = self.request("POST", "/api/prepare", {"prompt": "fail preparation"})
        self.assertEqual(status, 200)
        events = [json.loads(line) for line in body.splitlines()]
        self.assertEqual(events[-1], {"type": "done", "success": False})
        self.assertTrue(any(e["type"] == "error" for e in events))
        self.assertFalse(any(e["type"] in ("prepared", "token", "completed", "comparison") for e in events))
        self.assertFalse(self.app.snapshot()["busy"])
        status, _, body = self.request("POST", "/api/check-ready", {"prompt": "fail preparation"})
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["prepared"])
        self.assertEqual(self.calls, [])

    def test_ndjson_stream_decodes_cumulative_tokens_and_logs_result(self):
        status, headers, body = self.request("POST", "/api/compare",
                                             {"prompt": "hello", "max_new_tokens": 64, "mode": "strict"})
        self.assertEqual(status, 200)
        self.assertIn("application/x-ndjson", headers["Content-Type"])
        events = [json.loads(line) for line in body.splitlines()]
        self.assertEqual(events[-1], {"type": "done", "success": True})
        baseline = [event for event in events if event.get("runner") == "baseline"]
        tokens = [event for event in baseline if event["type"] == "token"]
        self.assertEqual([event["text"] for event in tokens], ["A", "AB"])
        self.assertTrue(all(event["cumulative"] for event in tokens))
        self.assertTrue(all("token_ids" not in event for event in tokens))
        self.assertEqual(next(event for event in baseline if event["type"] == "completed")["text"], "AB<script>")
        self.assertEqual(self.calls, [("hello", 64, "strict", 0, "sequential")])
        self.assertEqual(len(self.request_started_times), 1)
        logs = (self.temp_path / "runs" / "comparisons.jsonl").read_text().splitlines()
        self.assertEqual(len(logs), 1)
        self.assertEqual(json.loads(logs[0])["prompt"], "hello")
        self.assertEqual(json.loads(logs[0])["execution"], "sequential")
        self.assertFalse(self.app.snapshot()["busy"])

    def test_original_completes_before_candidate_starts_for_both_request_indices(self):
        expected = [("started", "baseline"), ("token", "baseline"),
                    ("token", "baseline"), ("completed", "baseline"),
                    ("started", "candidate"), ("token", "candidate"),
                    ("token", "candidate"), ("completed", "candidate")]
        for index in range(2):
            status, _, body = self.request("POST", "/api/compare",
                                          {"prompt": f"request {index}"})
            self.assertEqual(status, 200)
            events = [json.loads(line) for line in body.splitlines()]
            self.assertEqual([(e["type"], e["runner"]) for e in events if "runner" in e], expected)
            comparison = next(e for e in events if e["type"] == "comparison")
            self.assertEqual(comparison["execution"], "sequential")
            self.assertTrue(comparison["isolated_benchmark"])
        self.assertEqual([call[3] for call in self.calls], [0, 1])

    def test_both_runner_modes_are_forwarded_and_logged(self):
        for mode in ("optimized", "strict"):
            status, _, body = self.request("POST", "/api/compare",
                                          {"prompt": mode, "mode": mode, "execution": "sequential"})
            self.assertEqual(status, 200)
            events = [json.loads(line) for line in body.splitlines()]
            comparison = next(e for e in events if e["type"] == "comparison")
            self.assertEqual(comparison["execution"], "sequential")
        self.assertEqual([call[2] for call in self.calls], ["optimized", "strict"])
        self.assertEqual([call[4] for call in self.calls], ["sequential", "sequential"])
        records = [json.loads(line) for line in
                   (self.temp_path / "runs" / "comparisons.jsonl").read_text().splitlines()]
        self.assertEqual([record["execution"] for record in records], ["sequential", "sequential"])
        self.assertEqual([record["mode"] for record in records], ["optimized", "strict"])

    def test_busy_request_is_rejected_and_unlocks_after_completion(self):
        first = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        first.request("POST", "/api/compare", body=json.dumps({"prompt": "hold"}),
                      headers={"Host": self.host, "Content-Type": "application/json"})
        response = first.getresponse()
        self.assertEqual(response.status, 200)
        self.assertTrue(self.entered.wait(1))
        status, _, body = self.request("POST", "/api/compare", {"prompt": "second"})
        self.assertEqual(status, 409)
        self.assertIn("already using", json.loads(body)["error"])
        self.assertEqual(len(self.calls), 1)
        self.release.set()
        events = [json.loads(line) for line in response.read().splitlines()]
        first.close()
        self.assertEqual(events[-1], {"type": "done", "success": True})
        status, _, _ = self.request("POST", "/api/compare", {"prompt": "second"})
        self.assertEqual(status, 200)
        self.assertEqual([call[3] for call in self.calls], [0, 1])

    def test_error_event_unlocks_worker_for_next_request(self):
        with self.assertLogs(level="ERROR"):
            status, _, body = self.request("POST", "/api/compare", {"prompt": "fail"})
        self.assertEqual(status, 200)
        events = [json.loads(line) for line in body.splitlines()]
        self.assertEqual(events[-2], {"type": "error", "message": "expected test inference failure"})
        self.assertEqual(events[-1], {"type": "done", "success": False})
        self.assertFalse(self.app.snapshot()["busy"])
        self.assertTrue(self.app.snapshot()["ready"])
        self.assertEqual(self.request("POST", "/api/compare", {"prompt": "works"})[0], 200)

    def test_until_finished_resolves_remaining_capacity_for_all_runners(self):
        for mode in ("original", "optimized", "strict"):
            status, _, body = self.request("POST", "/api/check-ready",
                                          {"prompt": "hello", "mode": mode, "max_new_tokens": None})
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertTrue(data["until_finished"])
            self.assertIsNone(data["output_limit_requested"])
            self.assertEqual(data["resolved_max_new_tokens"], MAX_LEN - TREE_RESERVE - 5)

    def test_single_generation_runs_one_selected_runner_and_logs_no_speedup(self):
        for mode in ("original", "optimized", "strict"):
            status, _, body = self.request("POST", "/api/generate",
                                          {"prompt": "hello", "mode": mode, "max_new_tokens": None})
            self.assertEqual(status, 200)
            events = [json.loads(line) for line in body.splitlines()]
            self.assertEqual(events[-1], {"type": "done", "success": True})
            self.assertEqual([e["runner"] for e in events if e["type"] == "started"], ["single"])
            self.assertFalse(any(e["type"] == "comparison" or "speedup" in e for e in events))
            self.assertEqual(next(e for e in events if e["type"] == "token")["text"], "A")
        self.assertEqual([row[2] for row in self.generation_calls], ["original", "optimized", "strict"])
        self.assertTrue(all(row[1] == MAX_LEN - TREE_RESERVE - 5 for row in self.generation_calls))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.output_policies, [True, True, True])
        self.assertEqual(len((self.temp_path / "runs" / "generations.jsonl").read_text().splitlines()), 3)

    def test_original_is_single_only_and_exhausted_context_is_rejected(self):
        self.assertEqual(self.request("POST", "/api/compare", {"prompt": "hello", "mode": "original"})[0], 400)
        self.assertEqual(self.request("POST", "/api/generate",
                                      {"prompt": "x" * (MAX_LEN - TREE_RESERVE), "max_new_tokens": None})[0], 400)
        self.assertEqual(self.generation_calls, [])

    def test_previous_code_uses_identical_visible_context_for_preparation_and_generation(self):
        data = {"prompt": "Make it blue", "previous_code": "<h1>Hello</h1>", "max_new_tokens": 128}
        self.assertEqual(self.request("POST", "/api/prepare", data)[0], 200)
        self.assertEqual(self.request("POST", "/api/generate", data)[0], 200)
        self.assertEqual(self.prepare_calls[-1][0], self.generation_calls[-1][0])
        self.assertIn("<h1>Hello</h1>", self.generation_calls[-1][0])
        self.assertIn("Make it blue", self.generation_calls[-1][0])

    def test_system_roles_and_capacity_are_identical_for_ready_prepare_generate_and_compare(self):
        self.allow_all_prepared = False
        data = {"prompt": "Build a calculator", "app_action": "new",
                "previous_code": "<h1>Old weather dashboard</h1>",
                "system_prompt": "Use responsive grids.", "max_new_tokens": None}
        expected_messages = [{"role": "system", "content": data["system_prompt"]},
                             {"role": "user", "content": data["prompt"]}]
        capacity = MAX_LEN - TREE_RESERVE - len(data["system_prompt"]) - len(data["prompt"])
        status, _, body = self.request("POST", "/api/check-ready", data)
        self.assertEqual(status, 200)
        ready = json.loads(body)
        self.assertFalse(ready["prepared"])
        self.assertEqual(ready["resolved_max_new_tokens"], capacity)
        self.assertEqual(ready["system_prompt"], data["system_prompt"])
        self.assertEqual(ready["effective_prompt"], data["prompt"])
        self.assertEqual(self.tokenization_messages[-1], expected_messages)
        status, _, body = self.request("POST", "/api/generate", data)
        self.assertEqual(status, 200)
        self.assertEqual(self.generation_calls[-1][1], capacity)
        self.assertEqual(self.tokenization_messages[-1], expected_messages)
        self.assertEqual(self.role_calls[-2:], [("prepare", expected_messages), ("generate", expected_messages)])
        status, _, body = self.request("POST", "/api/compare", data)
        self.assertEqual(status, 200)
        self.assertEqual(self.calls[-1][1], capacity)
        self.assertEqual(self.tokenization_messages[-1], expected_messages)
        self.assertEqual(self.role_calls[-1], ("compare", expected_messages))
        self.assertNotIn("Old weather", self.generation_calls[-1][0])

    def test_malformed_system_or_missing_edit_context_is_http400_before_gpu_queue(self):
        before = len(self.prepare_calls)
        for bad in ({"system_prompt": {}}, {"system_prompt": "x" * 4001},
                    {"app_action": "edit"}, {"app_action": "edit", "previous_code": " "}):
            for path in ("/api/compare", "/api/generate", "/api/prepare", "/api/check-ready"):
                status, _, _ = self.request("POST", path, {"prompt": "Build a page", **bad})
                self.assertEqual(status, 400)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.generation_calls, [])
        self.assertEqual(len(self.prepare_calls), before)


if __name__ == "__main__":
    unittest.main()
