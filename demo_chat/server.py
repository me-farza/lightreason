"""Loopback-only original-first comparison UI with one CUDA worker."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import queue
import sys
import threading
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
DEFAULT_PROMPT = "Build a compact weather dashboard as one complete HTML file with inline CSS and JavaScript. Use a dark theme, a city search field and sample weather data. Keep it concise. Return only the HTML."
DEFAULT_BUDGET = None

if __package__:
    from .inference import configured_modes
    from .studio import RUNNER_NAMES
else:
    from inference import configured_modes
    from studio import RUNNER_NAMES


def validate_request(body, allow_original=False, default_mode="optimized", available_modes=RUNNER_NAMES):
    if not isinstance(body, dict):
        raise ValueError("Send a JSON object.")
    prompt = body.get("prompt")
    budget = body.get("max_new_tokens", 128)
    mode = body.get("mode", default_mode)
    execution = body.get("execution", "sequential")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a prompt first.")
    if len(prompt) > 8000:
        raise ValueError("Use a prompt shorter than 8,000 characters.")
    if budget is not None and (type(budget) is not int or budget not in (64, 128, 256, 512, 1024)):
        raise ValueError("Choose Until finished, or a fixed output limit.")
    if not isinstance(mode, str) or mode not in available_modes or (mode == "original" and not allow_original):
        raise ValueError("Choose an available Lightreason runner.")
    if execution != "sequential":
        raise ValueError("Refresh this page to use original-first sequential streaming.")
    if __package__:
        from .studio import build_prompt
        from .messages import PromptText
    else:
        from studio import build_prompt
        from messages import PromptText
    previous = body.get("previous_code")
    action = body.get("app_action", "edit" if previous else "new")
    if action not in ("new", "edit"):
        raise ValueError("Choose New app or Edit current app.")
    if action == "new":
        previous = None
    elif not isinstance(previous, str) or not previous.strip():
        raise ValueError("Generate an app before choosing Edit current app.")
    return PromptText(build_prompt(prompt.strip(), previous), body.get("system_prompt", "")), budget, mode, execution


class Application:
    def __init__(self, args, engine_factory=None):
        self.args = args
        self.available_modes = configured_modes(getattr(args, "draft_path", None), getattr(args, "checkpoint_path", None))
        self.default_mode = self.available_modes[1]
        self.jobs = queue.Queue()
        self.lock = threading.Lock()
        self.engine = None
        self.request_index = 0
        self.started = time.time()
        self.state = dict(application="lightreason-live-v1", ready=False, models_loaded=False,
                          demo_ready=False, busy=False, state="loading",
                          detail="Loading the original Qwen model on the GPU…",
                          model="Qwen/Qwen2.5-3B-Instruct · BF16",
                          gpu="RTX 4090", same_target_weights=True,
                          session_expires_at=self.started + 3600,
                          supports_simultaneous_streaming=False, default_execution="sequential",
                          baseline_first=True, preparation_required=False, automatic_setup=True,
                          supports_single_generation=True, supports_web_preview=True,
                          runner_names={mode: RUNNER_NAMES[mode] for mode in self.available_modes},
                          available_modes=list(self.available_modes), default_mode=self.default_mode,
                          context_capacity_tokens=4096,
                          timing_scope="Execution graphs warm automatically when needed; generation timing excludes automatic setup.")
        self.engine_factory = engine_factory

    def update(self, event):
        if event.get("type") == "status":
            with self.lock:
                self.state.update({k: v for k, v in event.items() if k != "type"})

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def resolve_request(self, request):
        prompt, budget, mode, execution = request
        if mode not in self.available_modes:
            raise ValueError("Choose an available Lightreason runner.")
        until_finished = budget is None
        if until_finished:
            if __package__:
                from .inference import MAX_LEN, TREE_RESERVE
                from .messages import chat_messages
            else:
                from inference import MAX_LEN, TREE_RESERVE
                from messages import chat_messages
            ids = self.engine.tokenizer.apply_chat_template(
                chat_messages(prompt), add_generation_prompt=True, tokenize=True)
            budget = MAX_LEN - TREE_RESERVE - len(ids)
            if budget < 1:
                raise ValueError("The prompt and current page fill the model context. Shorten the page or start a new one.")
        return (prompt, budget, mode, execution), {
            "until_finished": until_finished,
            "output_limit_requested": None if until_finished else budget,
            "resolved_max_new_tokens": budget,
        }

    def check_ready(self, request):
        with self.lock:
            if not self.state["ready"]:
                return dict(prepared=False, detail=self.state["detail"])
            if self.state["busy"]:
                return dict(prepared=False, busy=True, detail="The GPU is working. Readiness will update when it finishes.")
            resolved, policy = self.resolve_request(request)
            prompt, budget, mode, _ = resolved
            # A readiness read runs only while idle, under the same lock used
            # by submit. It inspects CPU shape/cache metadata; never CUDA.
            return dict(self.engine.prepared_status(prompt, budget, mode), **policy, effective_prompt=prompt,
                        system_prompt=getattr(prompt, "system_prompt", ""))

    def submit(self, request, request_started_s=None, action="compare"):
        if request_started_s is None:
            request_started_s = time.perf_counter()
        if action not in ("compare", "prepare", "generate"):
            raise ValueError("Unknown GPU action.")
        if action == "compare" and request[2] == "original":
            raise ValueError("Choose an available Lightreason runner.")
        with self.lock:
            if not self.state["ready"]:
                raise RuntimeError(self.state["detail"])
            if self.state["busy"]:
                raise BlockingIOError("A comparison is already using this GPU. Wait for it to finish.")
            request, policy = self.resolve_request(request)
            self.state.update(busy=True, state="preparing",
                              detail="Starting Qwen… setting up GPU execution" if action == "prepare" else "Starting Qwen…")
            index = self.request_index
            if action in ("compare", "generate"):
                self.request_index += 1
        events = queue.Queue()
        self.jobs.put((action, request, events, index, request_started_s, policy))
        return events

    def _prepare_startup(self):
        if __package__:
            from .messages import PromptText, DEFAULT_DESIGN_PROMPT
        else:
            from messages import PromptText, DEFAULT_DESIGN_PROMPT
        # Warm the smallest input as well, so short new-app requests do not
        # encounter a missing 256-token target family after startup.
        mode = self.default_mode
        bootstrap, _ = self.resolve_request((PromptText("x", ""), None, mode, "sequential"))
        self.engine.prepare_demo(bootstrap[0], bootstrap[1], mode, self.update)
        default_prompt = PromptText(DEFAULT_PROMPT, DEFAULT_DESIGN_PROMPT)
        default_request, default_policy = self.resolve_request((default_prompt, DEFAULT_BUDGET, mode, "sequential"))
        _, default_budget, _, _ = default_request
        self.engine.prepare_demo(default_prompt, default_budget, mode, self.update)
        admission = dict(self.engine.prepared_status(default_prompt, default_budget, mode), **default_policy)
        if not admission.get("prepared"):
            raise RuntimeError("Default demo preparation did not establish readiness.")
        return admission

    def worker(self):
        try:
            if self.engine_factory is None:
                if __package__:
                    from .studio import StudioLiveInference
                else:
                    from studio import StudioLiveInference
                factory = StudioLiveInference
            else:
                factory = self.engine_factory
            self.engine = factory(model_path=self.args.model_path, draft_path=self.args.draft_path,
                                  checkpoint_path=self.args.checkpoint_path, cooldown_s=self.args.cooldown_seconds)
            self.engine.initialize(self.update)
            admission = None
            if "optimized" in self.available_modes:
                with self.lock:
                    self.state.update(models_loaded=True, state="preparing",
                                      detail="Starting Qwen… optimizing GPU execution")
                admission = self._prepare_startup()
            with self.lock:
                self.state.update(ready=True, models_loaded=True, demo_ready=admission is not None,
                                  state="ready", detail="Qwen is ready. Enter a prompt.", startup_prepared=admission)
        except Exception as exc:
            logging.exception("GPU initialization failed")
            with self.lock:
                self.state.update(ready=False, state="error", detail=f"GPU initialization failed: {exc}")
            return
        while True:
            action, request, events, index, request_started_s, policy = self.jobs.get()
            prompt, budget, mode, execution = request
            emitted_prepared = False
            auto_preparing = False
            auto_setup_s = 0.0
            auto_setup_required = False
            def emit(event):
                nonlocal emitted_prepared
                if event.get("type") == "prepared":
                    emitted_prepared = True
                    if auto_preparing:
                        # Automatic shape setup is internal, not a separate
                        # manual-preparation action in the live interface.
                        return
                    event = dict(event, ready=True, prepared=True, effective_prompt=prompt, **policy)
                if auto_preparing and event.get("type") == "status":
                    event = dict(event, state="starting", detail="Starting Qwen… setting up GPU execution")
                if action in ("compare", "generate") and event.get("type") in ("comparison", "generation"):
                    event = dict(event, auto_setup_s=auto_setup_s,
                                 auto_setup_required=auto_setup_required)
                self.update(event)
                events.put(event)
            try:
                if action in ("compare", "generate"):
                    admission = self.engine.prepared_status(prompt, budget, mode)
                    if not admission.get("prepared"):
                        auto_setup_required = True
                        setup_started = time.perf_counter()
                        auto_preparing = True
                        try:
                            emit(dict(type="status", state="starting",
                                      detail="Starting Qwen… setting up GPU execution"))
                            self.engine.prepare_demo(prompt, budget, mode, emit)
                            if not self.engine.prepared_status(prompt, budget, mode).get("prepared"):
                                raise RuntimeError("Automatic GPU setup finished without a valid execution shape.")
                        finally:
                            auto_setup_s = time.perf_counter() - setup_started
                            auto_preparing = False
                if action == "prepare":
                    result = self.engine.prepare_demo(prompt, budget, mode, emit)
                    if not self.engine.prepared_status(prompt, budget, mode).get("prepared"):
                        raise RuntimeError("Preparation finished without a valid warmed configuration.")
                    if not emitted_prepared:
                        emit(dict(result, type="prepared", ready=True, prepared=True))
                elif action == "generate":
                    result = self.engine.generate(prompt, budget, mode, emit,
                                                  request_started_s=request_started_s, require_prepared=True,
                                                  until_finished=policy["until_finished"])
                else:
                    result = self.engine.compare(prompt, budget, mode, emit, request_index=index,
                                                 execution=execution, request_started_s=request_started_s,
                                                 require_prepared=True, until_finished=policy["until_finished"])
                if action in ("compare", "generate"):
                    result["auto_setup_s"] = auto_setup_s
                    result["auto_setup_required"] = auto_setup_required
                    event_key = "generation" if action == "generate" else "comparison"
                    if isinstance(result.get(event_key), dict):
                        result[event_key].update(auto_setup_s=auto_setup_s,
                                                 auto_setup_required=auto_setup_required)
                log_dir = Path(self.args.log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                record = dict(timestamp=datetime.now(timezone.utc).isoformat(), prompt=prompt,
                              max_new_tokens=budget, mode=mode, execution=execution, request_index=index,
                              action=action, output_policy=policy,
                              system_prompt=getattr(prompt, "system_prompt", ""), result=result)
                filename = {"prepare": "preparations.jsonl", "generate": "generations.jsonl", "compare": "comparisons.jsonl"}[action]
                with (log_dir / filename).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                emit(dict(type="done", success=True))
            except Exception as exc:
                logging.exception("Comparison failed")
                emit(dict(type="error", message=str(exc)))
                emit(dict(type="done", success=False))
            finally:
                with self.lock:
                    self.state.update(busy=False, state="ready", detail="Qwen is ready. Enter a prompt.")
                self.jobs.task_done()


class Handler(BaseHTTPRequestHandler):
    server_version = "LightreasonLive/1"

    @property
    def app(self):
        return self.server.app

    def log_message(self, fmt, *args):
        logging.info("%s %s", self.address_string(), fmt % args)

    def local_request(self):
        # Neither a public bind nor cross-origin browser access is needed.
        host = self.headers.get("Host", "")
        try:
            host_name = urlsplit("http://" + host).hostname
        except ValueError:
            return False
        if host_name not in ("127.0.0.1", "localhost", "::1"):
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://{host}", f"https://{host}"):
            return False
        return True

    def send_bytes(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if content_type.startswith("text/html"):
            self.send_header("Content-Security-Policy", "frame-src about:; object-src 'none'; base-uri 'self'; form-action 'self'")
            self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=(), usb=()")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, code, value):
        self.send_bytes(code, json.dumps(value, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        if not self.local_request():
            return self.send_json(403, {"error": "Use this app from its localhost address."})
        path = urlsplit(self.path).path
        if path == "/api/status":
            return self.send_json(200, self.app.snapshot())
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/index.html": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/preview.js": ("preview.js", "text/javascript; charset=utf-8"),
                 "/lightreason.svg": ("lightreason.svg", "image/svg+xml"),
                 "/style.css": ("style.css", "text/css; charset=utf-8"),
                 "/styles.css": ("styles.css", "text/css; charset=utf-8")}
        if path not in files or not (ROOT / "static" / files[path][0]).is_file():
            return self.send_json(404, {"error": "Not found"})
        name, mime = files[path]
        self.send_bytes(200, (ROOT / "static" / name).read_bytes(), mime)

    def do_POST(self):
        request_started_s = time.perf_counter()
        if not self.local_request():
            return self.send_json(403, {"error": "Cross-origin requests are disabled."})
        path = urlsplit(self.path).path
        if path not in ("/api/compare", "/api/generate", "/api/prepare", "/api/check-ready"):
            return self.send_json(404, {"error": "Not found"})
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.send_json(415, {"error": "Send application/json."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                raise ValueError("Request must be at most 64 KB.")
            request = validate_request(json.loads(self.rfile.read(size)), allow_original=path != "/api/compare",
                                       default_mode=self.app.default_mode, available_modes=self.app.available_modes)
            if path == "/api/check-ready":
                return self.send_json(200, self.app.check_ready(request))
            action = {"/api/prepare": "prepare", "/api/generate": "generate", "/api/compare": "compare"}[path]
            events = self.app.submit(request, request_started_s=request_started_s, action=action)
        except (ValueError, UnicodeDecodeError) as exc:
            return self.send_json(400, {"error": str(exc)})
        except BlockingIOError as exc:
            return self.send_json(409, {"error": str(exc)})
        except RuntimeError as exc:
            return self.send_json(503, {"error": str(exc)})
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        connected = True
        while True:
            try:
                event = events.get(timeout=1)
            except queue.Empty:
                event = dict(type="heartbeat", state=self.app.snapshot()["state"])
            if event.get("type") == "token_ids":
                # Decoding and socket writes happen off the CUDA worker thread.
                event = dict(event, type="token", cumulative=True,
                             text=self.app.engine.tokenizer.decode(event["token_ids"], skip_special_tokens=True))
                event.pop("token_ids", None)
            if connected:
                try:
                    self.wfile.write(json.dumps(event, ensure_ascii=False, allow_nan=False).encode() + b"\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    connected = False
            if event.get("type") == "done":
                break


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model-path", default="model-3b")
    p.add_argument("--draft-path", help="Pinned local auxiliary model directory for optional Lightreason Turbo")
    p.add_argument("--checkpoint-path", help="Authorized local adaptation checkpoint for optional Lightreason Turbo")
    p.add_argument("--cooldown-seconds", type=float, default=0.0)
    p.add_argument("--log-dir", default="demo_chat/runs")
    args = p.parse_args()
    if bool(args.draft_path) != bool(args.checkpoint_path):
        p.error("Provide both --draft-path and --checkpoint-path to enable Lightreason Turbo.")
    if not 1024 <= args.port <= 65535 or not 0 <= args.cooldown_seconds <= 15:
        p.error("Use a valid unprivileged port and cooldown between 0 and 15 seconds.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = Application(args)
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    import os
    (runtime / "service.pid").write_text(str(os.getpid()))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.app = app
    server.daemon_threads = True
    threading.Thread(target=app.worker, name="single-gpu-worker", daemon=True).start()
    logging.info("Live comparison listening on http://127.0.0.1:%s", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
