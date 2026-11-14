"""Serve the measured seven-proposal BF16 runner through Studio."""

from contextlib import redirect_stdout
import json
import logging
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
from time import perf_counter, sleep, time_ns
from urllib.error import URLError

from edge35b.download import GIB, REVISION, available_memory
from edge35b.freetoken_bench import ModelLoading, completion

if __package__:
    from .edge_inference import CONTEXT_CAPACITY, MODEL_NAME, RUNNER_NAMES as EDGE_RUNNERS, EdgeLiveInference
    from .inference import compare_outputs
    from .messages import PromptText, chat_messages
else:
    from edge_inference import CONTEXT_CAPACITY, MODEL_NAME, RUNNER_NAMES as EDGE_RUNNERS, EdgeLiveInference
    from inference import compare_outputs
    from messages import PromptText, chat_messages


RUNNER_NAMES = {"optimized": "FreeToken · 7 proposals"}
EXPERT_CACHE_RUNNER_NAMES = {
    "original": "Frozen eight-position runner",
    "optimized": "Lightreason · Tuned expert cache",
}
EXPERT_CACHE_BASELINES = {
    "scheduled": EDGE_RUNNERS["original"],
    "frozen": EXPERT_CACHE_RUNNER_NAMES["original"],
}
SHARED_CPU_RUNNER_NAMES = {"optimized": "Lightreason · 34-layer CPU/CUDA"}
PORT = 1922
EVENT_PREFIX = "studio-event "


def shared_cpu_settings(stage):
    from edge35b.width_autotune import SHARED_CPU_TRANSPORT

    report = json.loads((stage / "cpu-shared-coverage.json").read_text())
    transport = report["transport_settings"]
    cpu = report["cpu_read_settings"]
    summary = report["transport_summary"]
    if (report["revision"] != REVISION or report["dtype"] != "bfloat16" or not summary["exact"]
            or summary["candidate"] != SHARED_CPU_TRANSPORT
            or transport["width"] != 8 or transport["cpu_pair_threshold"] != 1
            or transport["expert_cache_gib"] != 15 or transport["hot_expert_count"] != 128
            or cpu["candidate_policy"] != "packed-both-prefetch"
            or len(cpu["layers"]) != 34 or cpu["layers"] != transport["shared_cpu_layers"]
            or cpu["control_layers"] != transport["shared_cpu_control_layers"]):
        raise ValueError("Use the exact 34-layer BF16 benchmark runtime.")
    return SHARED_CPU_TRANSPORT, dict(
        LIGHTREASON_NATIVE_PACKED_GATHER=transport["native_geometry"],
        LIGHTREASON_SHARED_CPU_GATHER=transport["shared_cpu_geometry"],
        LIGHTREASON_CPU_READ_POLICY=cpu["candidate_policy"],
        LIGHTREASON_CPU_READ_LAYERS=",".join(map(str, cpu["layers"])),
        LIGHTREASON_SHARED_CPU_LAYERS=",".join(map(str, transport["shared_cpu_layers"])),
        LIGHTREASON_CPU_READ_CONTROL_LAYERS=",".join(map(str, cpu["control_layers"])),
        LIGHTREASON_SHARED_CPU_CONTROL_LAYERS=",".join(map(str, transport["shared_cpu_control_layers"])),
        LIGHTREASON_STUDIO_CACHE_PARK="1",
    )


def track_worker_processes(root_pid, handles):
    """Record stable Linux handles for this server's children, including thread-spawned workers."""
    pending = [root_pid]
    seen = {root_pid}
    while pending:
        parent = pending.pop()
        for entry in Path(f"/proc/{parent}/task").glob("*/children"):
            try:
                children = map(int, entry.read_text().split())
            except (FileNotFoundError, ProcessLookupError):
                continue
            for pid in children:
                if pid in seen:
                    continue
                seen.add(pid)
                if pid not in handles:
                    try:
                        handles[pid] = os.pidfd_open(pid)
                    except ProcessLookupError:
                        continue
                pending.append(pid)


def workers_still_running(handles, timeout):
    remaining = set(handles)
    poller = select.poll()
    for handle in remaining:
        poller.register(handle, select.POLLIN)
    deadline = perf_counter() + timeout
    while remaining:
        wait_ms = max(0, int((deadline - perf_counter()) * 1000))
        exited = poller.poll(wait_ms)
        if not exited:
            break
        for handle, _ in exited:
            poller.unregister(handle)
            remaining.remove(handle)
    return remaining


def stream_reference(model_path, shared_experts=None, reference="packed"):
    output = sys.stdout

    def emit(event):
        print(EVENT_PREFIX + json.dumps(event, allow_nan=False), file=output, flush=True)

    with redirect_stdout(sys.stderr):
        try:
            request = None if shared_experts else json.loads(sys.stdin.readline())
            packed = reference == "packed"
            mode = "optimized" if packed else "original"
            runner = EdgeLiveInference(model_path, packed=packed, packed_replay=packed)
            runner.runner_names = {mode: runner.runner_names[mode]}
            use_checkpoint = shared_experts and reference == "scheduled" and os.getenv("LIGHTREASON_STUDIO_PAGEABLE_HANDOFF") == "1"
            views = {} if use_checkpoint else None
            runner.initialize(emit, shared_experts=shared_experts, gpu_weight_views=views)

            def run(request):
                prompt = PromptText(request["prompt"], request["system_prompt"])
                if runner._input_ids(prompt) != request["input_ids"]:
                    raise RuntimeError("The reference and FreeToken produced different input token IDs.")
                runner._stream(prompt, request["budget"], mode, emit, "baseline",
                               request["request_started_s"], request["until_finished"])

            if shared_experts:
                from demo_chat.resident_runners import CheckpointResidency, Residency

                residency = CheckpointResidency(runner.model, views) if use_checkpoint else Residency()
                emit(dict(type="resident_ready", **runner.metadata))
                for line in sys.stdin:
                    request = json.loads(line)
                    if "residency" in request:
                        emit(dict(type="residency", **residency.transition(request["residency"])))
                    elif residency.parked:
                        raise RuntimeError("The reference runner needs the GPU before generation.")
                    else:
                        run(request)
                return
            run(request)
        except Exception as exc:
            logging.exception("Reference comparison failed")
            emit(dict(type="error", message=str(exc)))
            if shared_experts:
                return
        # Studio stops the watchdog after the stream drains. The worker stays idle while
        # the supervisor is in charge of its CUDA context and process group.
        while True:
            signal.pause()


class FreeTokenStudioInference(EdgeLiveInference):
    def __init__(self, model_path, *, resident=False, expert_cache_dir=None, shared_cpu=False, api_key=None):
        self.model_path = Path(model_path)
        self.expert_cache_dir = Path(expert_cache_dir) if expert_cache_dir else None
        self.runner_names = EXPERT_CACHE_RUNNER_NAMES if self.expert_cache_dir else RUNNER_NAMES
        self.shared_transport, self.shared_env = None, {}
        self.pageable_library = None
        if shared_cpu:
            if not (resident and self.expert_cache_dir):
                raise ValueError("The 34-layer runner needs the staged resident comparison runtime.")
            self.shared_transport, self.shared_env = shared_cpu_settings(self.expert_cache_dir)
            self.runner_names = SHARED_CPU_RUNNER_NAMES
            library = self.expert_cache_dir / "tms-pageable/torch_memory_saver_hook_mode_preload_cu13.abi3.so"
            if library.is_file() and os.getenv("LIGHTREASON_STUDIO_DISK_HANDOFF") != "1":
                self.pageable_library = library
        self.url = f"http://127.0.0.1:{PORT}"
        self.api_key = api_key
        self.child = None
        self.worker_handles = {}
        self.resident = resident
        self.resident_reference = "scheduled" if self.expert_cache_dir else "packed"
        self.replay_child = None
        self.replay_handles = {}
        self.control = None
        self.disk_backup = None
        self.active_runner = None
        self.startup_setup_s = 0.0
        self.process_lock = threading.Lock()
        self.shutting_down = False

    def _start_monitored(self, command, *, replay=False, **kwargs):
        directory = self.model_path.parent / "studio-telemetry"
        directory.mkdir(exist_ok=True)
        watchdog = Path(__file__).resolve().parents[1] / "hardware_exact10.py"
        reporting = []
        if self.shared_transport:
            watchdog = self.expert_cache_dir / "hardware_exact10.py"
            reporting = ["--final-report-only", "--safety-sampling"]
        report = directory / f"runner-{time_ns()}.json"
        monitor = ["taskset", "-c", "2-3", sys.executable, "-u", str(watchdog),
                   "--gpu-stop-c", "70", "--cpu-stop-c", "75", "--command-timeout-s", "3600",
                   "--out", str(report), *reporting, "--monitor-command", "taskset", "-c", "8-15", *command]
        with self.process_lock:
            slot = "replay_child" if replay else "child"
            if self.shutting_down or getattr(self, slot) is not None:
                raise RuntimeError("Studio is unable to start another GPU runner.")
            child = subprocess.Popen(monitor, **kwargs)
            setattr(self, slot, child)
            return child

    def load_tokenizer(self):
        from transformers import AutoTokenizer

        manifest = json.loads((self.model_path / "lightreason-download.json").read_text())
        if manifest["revision"] != REVISION:
            raise RuntimeError("Use the pinned Qwen3.6 BF16 checkpoint.")
        if not hasattr(self, "tokenizer"):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)

    def initialize(self, callback):
        self.close()
        started = perf_counter()
        try:
            if self.resident or self.expert_cache_dir:
                self.control = TemporaryDirectory(prefix="lightreason-", dir="/tmp")
            if self.shared_transport:
                self.disk_backup = TemporaryDirectory(prefix="studio-vram-", dir=self.model_path.parent)
                for name in ("lightreason", "scheduled"):
                    (Path(self.disk_backup.name) / name).mkdir(mode=0o700)
            self._initialize_freetoken(callback)
            if self.resident:
                self.active_runner = "freetoken"
                self._start_resident_reference(callback)
                # Prepare the first backups and verify both resume paths
                # before accepting a user's first prompt.
                self._select_runner("freetoken", callback)
                self._select_runner(self.resident_reference, callback)
            self.startup_setup_s = perf_counter() - started
        except BaseException:
            self.close()
            raise

    def _initialize_freetoken(self, callback):
        self.load_tokenizer()
        root = self.model_path.parent
        sizes = [p.stat().st_size for p in self.model_path.glob("model-*.safetensors")]
        if not sizes or available_memory() < sum(sizes) + max(sizes) + 8 * GIB:
            raise RuntimeError("The pinned BF16 expert bank needs more free host RAM.")
        command = [
            str(root / "freetoken/bin/ft"), "serve", "--model", str(self.model_path),
            "--served-model-name", "qwen36-bf16", "--host", "127.0.0.1", "--port", str(PORT),
            "--dtype", "bfloat16", "--text-model-only", "--max-running-requests", "1",
            "--max-seq-len-override", str(CONTEXT_CAPACITY), "--cache-type", "naive",
            "--moe-strategy", "hybrid", "--moe-cache-mem", "15GiB", "--moe-cpu-threads", "8",
            "--cuda-graph-max-bs", "1", "--attn", "triton", "--enable-timings-report",
            "--moe-collect-stats", "--decode-log-interval", "8192" if self.expert_cache_dir else "64",
            "--moe-prefill-hit-d2d",
            "--speculative-algorithm", "dflash", "--speculative-num-steps", "7",
            "--speculative-draft-model-path", str(root / "dflash2-51ef7b6923ad6c14cb1bb41c37a9041446496ab6"),
            "--embed-weights", "host", "--expert-load", "parallel",
        ]
        env = {
            **os.environ, "PYTHONPATH": str(root / "freetoken-selected-bench/python"),
            "XDG_CACHE_HOME": str(root / "freetoken-dflash2-cache"),
            "TRITON_CACHE_DIR": str(root / "freetoken-triton"), "TMPDIR": str(root / "tmp"),
            "HF_HUB_OFFLINE": "1", "MALLOC_ARENA_MAX": "2", "LIGHTREASON_BENCH_TOKEN_IDS": "1",
            "LIGHTREASON_ROUTED_PREFILL_MAX_TOKENS": "80",
            "LIGHTREASON_SELECTED_PREFILL_MAX_TOKENS": "192",
        }
        if self.api_key is not None:
            env["FREETOKEN_API_KEY"] = self.api_key
        if self.resident:
            from demo_chat.resident_runners import allocator_env

            disk = Path(self.disk_backup.name) / "lightreason" if self.disk_backup else None
            env.update(allocator_env(root, 13, root / "freetoken", disk_backup=disk,
                                     preload_library=self.pageable_library))
            if self.pageable_library:
                env["LIGHTREASON_STUDIO_PAGEABLE_HANDOFF"] = "1"
            env.update(
                PYTHONPATH=os.pathsep.join((str(root / "freetoken-studio-resident/python"),
                                           str(root / "studio-resident-tools"), str(Path.cwd()))),
                LIGHTREASON_STUDIO_HANDOFF=str(self.control_path),
            )
        if self.expert_cache_dir:
            paths = [self.expert_cache_dir / "benchmark", self.expert_cache_dir / "python-tune"]
            if self.resident:
                paths.extend((root / "studio-resident-tools", Path.cwd()))
            env.update(
                PYTHONPATH=os.pathsep.join(map(str, paths)),
                LIGHTREASON_WIDTH_TRIAL_PATH=str(self.width_policy_path),
                LIGHTREASON_PASS_DIAGNOSTICS="1",
                LIGHTREASON_PACKED_BF16_TRANSFER="1",
                LIGHTREASON_PACKED_BF16_WIDTH="8",
                LIGHTREASON_PACKED_BF16_HOT_PLAN=str(self.expert_cache_dir / "hot-128.json"),
            )
            env.update(self.shared_env)
            self._set_expert_cache_policy(self.shared_transport or "raw")
        detail = ("Loading Lightreason’s 34-layer BF16 expert bank…" if self.shared_transport
                  else "Loading FreeToken’s BF16 expert bank…")
        callback(dict(type="status", state="loading", detail=detail))
        child = self._start_monitored(command, env=env, stdin=subprocess.DEVNULL)
        try:
            text, count = self._completion_prompt("Count to three.")
            deadline = perf_counter() + 600
            while perf_counter() < deadline and child.poll() is None:
                with self.process_lock:
                    if self.child is child:
                        track_worker_processes(child.pid, self.worker_handles)
                try:
                    if self.expert_cache_dir:
                        if not self.shared_transport:
                            self._complete("original", text, count, 3, False)
                        self._complete("optimized", text, count, 3, False)
                    else:
                        self._complete("optimized", text, count, 3, False)
                    detail = ("Lightreason is warm. Preparing Scheduled…" if self.shared_transport else
                              "Both Qwen runners are warm." if self.expert_cache_dir else "FreeToken is warm.")
                    callback(dict(type="status", state="starting", detail=detail,
                                  models_loaded=not self.resident, demo_ready=not self.resident))
                    return
                except (ModelLoading, URLError):
                    sleep(2)
            raise RuntimeError("FreeToken did not finish loading. Check the Studio session log.")
        finally:
            with self.process_lock:
                if self.child is child and child.poll() is None:
                    track_worker_processes(child.pid, self.worker_handles)

    @property
    def control_path(self):
        return Path(self.control.name) / "freetoken.sock"

    @property
    def width_policy_path(self):
        return Path(self.control.name) / "width.cfg"

    def _set_expert_cache_policy(self, transport):
        policy = self.width_policy_path
        staged = policy.with_suffix(".part")
        settings = dict(width=8, min_pairs=1, diagnostics="off", transport=transport)
        if self.shared_transport:
            settings["cpu_read_policy"] = self.shared_env["LIGHTREASON_CPU_READ_POLICY"]
        staged.write_text(json.dumps(settings))
        staged.replace(policy)

    def _complete(self, mode, text, count, budget, stream, *, on_text=None):
        auth = {"api_key": self.api_key} if self.api_key is not None else {}
        if not self.expert_cache_dir:
            return completion(self.url, text, count, budget, stream, on_text=on_text, **auth)

        transport = self.shared_transport or ("raw" if mode == "original" else "hot")
        self._set_expert_cache_policy(transport)
        trace_path = self.width_policy_path.with_suffix(".passes.jsonl")
        offset = trace_path.stat().st_size if trace_path.exists() else 0
        result = completion(self.url, text, count, budget, stream, on_text=on_text, **auth)
        tokens = result["usage"]["completion_tokens"]
        if result["cached_prompt_tokens"] != 0 or not result.get("token_ids"):
            raise RuntimeError("The live comparison requires uncached input and output token IDs.")
        if tokens == 1:
            result["decode_tokens_per_s"] = None
            return result
        with trace_path.open("rb") as trace_file:
            trace_file.seek(offset)
            records = trace_file.readlines()
        if len(records) != 1:
            raise RuntimeError("Expected one verifier trace for this live request.")
        trace = json.loads(records[0])
        if (trace["width"] != "8" or trace["min_pairs"] != 1 or trace["transport"] != transport
                or trace.get("cpu_read_policy", "stock") != self.shared_env.get("LIGHTREASON_CPU_READ_POLICY", "stock")
                or any(width != 8 for width, _, _ in trace["passes"])
                or sum(committed for _, committed, _ in trace["passes"]) != tokens - 1):
            raise RuntimeError("The live request did not use the configured eight-position runner.")
        result["decode_tokens_per_s"] = (tokens - 1) * 1000 / result["timings"]["predicted_ms"]
        return result

    def _start_resident_reference(self, callback):
        from demo_chat.resident_runners import allocator_env, switch_freetoken

        name = EXPERT_CACHE_BASELINES["scheduled"] if self.expert_cache_dir else "Packed Replay"
        callback(dict(type="status", state="loading", detail=f"Auditing and warming {name}…"))
        switch_freetoken(self.control_path, "park")
        root = self.model_path.parent
        reference_env = Path.cwd() / ".venv" if self.expert_cache_dir else Path(sys.prefix)
        disk = (Path(self.disk_backup.name) / "scheduled"
                if self.disk_backup and os.getenv("LIGHTREASON_STUDIO_DISK_HANDOFF") == "1" else None)
        env = {**os.environ, **allocator_env(root, 12, reference_env, disk_backup=disk)}
        if self.shared_transport and disk is None:
            # Park only the smaller Scheduled state in RAM. The inactive
            # Lightreason state is disk-backed while Scheduled audits its weights.
            env.update(TMS_INIT_CPU_BACKUP_BACKEND="mmap", TMS_RETAIN_CPU_BACKUP="0")
        if self.pageable_library:
            env["LIGHTREASON_STUDIO_PAGEABLE_HANDOFF"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(map(str, (root / "studio-resident-tools", root / "packed-replay-deps")))
        command = [str(reference_env / "bin/python"), "-u", "-m", "demo_chat.freetoken_inference",
                   str(self.model_path), str(self.control_path), self.resident_reference]
        child = self._start_monitored(command, replay=True, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        audit = self._read_replay("resident_ready", callback)
        with self.process_lock:
            if self.replay_child is child:
                track_worker_processes(child.pid, self.replay_handles)
        logging.info("Shared BF16 runtime audit compared %s parameters in %.2f s.",
                     audit["target_parameters"], audit["checkpoint_audit_s"])
        self.active_runner = self.resident_reference

    def _read_replay(self, terminal, callback):
        for line in self.replay_child.stdout:
            if not line.startswith(EVENT_PREFIX):
                sys.stderr.write(line)
                continue
            event = json.loads(line[len(EVENT_PREFIX):])
            if event["type"] == "error":
                raise RuntimeError(event["message"])
            if event["type"] == terminal:
                return event
            callback(event)
        raise RuntimeError("The reference worker exited. Check the Studio session log.")

    def _replay_command(self, request, terminal, callback):
        self.replay_child.stdin.write(json.dumps(request) + "\n")
        self.replay_child.stdin.flush()
        return self._read_replay(terminal, callback)

    def _select_runner(self, runner, callback):
        if self.active_runner == runner:
            return
        from demo_chat.resident_runners import switch_freetoken

        def switch(name, command):
            if name == self.resident_reference:
                result = self._replay_command(dict(residency=command), "residency", callback)
            else:
                result = switch_freetoken(self.control_path, command)
            if result["parked"] != (command == "park"):
                raise RuntimeError(f"{name} did not confirm the GPU handoff.")
            logging.info("%s %s in %.3f s with %.2f GiB free on the GPU.",
                         name, command, result["seconds"], result["free_bytes"] / GIB)
            if command == "park" and "pageable_bytes" in result:
                logging.info("%s parked %.2f GiB in RAM and %.2f GiB on disk.", name,
                             result["pageable_bytes"] / GIB, result["disk_bytes"] / GIB)

        try:
            switch(self.active_runner, "park")
            switch(runner, "activate")
            self.active_runner = runner
        except BaseException:
            self.close()
            raise

    def close(self):
        with self.process_lock:
            for child, handles in ((self.replay_child, self.replay_handles), (self.child, self.worker_handles)):
                if child is not None and child.poll() is None:
                    track_worker_processes(child.pid, handles)
                    child.terminate()
            try:
                self._close_runner("replay_child", self.replay_handles)
            finally:
                self._close_runner("child", self.worker_handles)
            if self.control is not None:
                self.control.cleanup()
                self.control = None
            if self.disk_backup is not None:
                self.disk_backup.cleanup()
                self.disk_backup = None
            self.active_runner = None

    def shutdown(self):
        self.shutting_down = True
        self.close()

    def _close_runner(self, slot, handles):
        child = getattr(self, slot)
        if child is None:
            return
        if child.poll() is None:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        remaining = workers_still_running(handles.values(), timeout=5)
        for sig, timeout in ((signal.SIGTERM, 5), (signal.SIGKILL, 60)):
            if not remaining:
                break
            for handle in remaining:
                try:
                    signal.pidfd_send_signal(handle, sig)
                except ProcessLookupError:
                    pass
            remaining = workers_still_running(remaining, timeout=timeout)
        if remaining:
            raise RuntimeError("Studio workers are still using the GPU after shutdown.")
        for handle in handles.values():
            os.close(handle)
        handles.clear()
        for pipe in (child.stdin, child.stdout):
            if pipe is not None:
                pipe.close()
        setattr(self, slot, None)

    def prepared_status(self, prompt, budget, mode):
        if self.child is None or self.child.poll() is not None:
            return dict(prepared=False, mode=mode, detail="FreeToken needs to load on the RTX 4090.")
        if self.resident:
            if self.replay_child is None or self.replay_child.poll() is not None:
                return dict(prepared=False, mode=mode, detail="The reference runner needs to load on the RTX 4090.")
            if self.shared_transport:
                return dict(super().prepared_status(prompt, budget, mode), startup_setup_s=self.startup_setup_s,
                            detail="Scheduled and Lightreason 34-layer are ready for live comparisons.")
            if self.expert_cache_dir:
                return dict(super().prepared_status(prompt, budget, mode), startup_setup_s=self.startup_setup_s,
                            detail="Scheduled reference and both FreeToken runners are ready.")
            return dict(super().prepared_status(prompt, budget, mode), startup_setup_s=self.startup_setup_s,
                        detail="Packed Replay and FreeToken are ready for live comparisons.")
        if self.expert_cache_dir:
            return dict(super().prepared_status(prompt, budget, mode), startup_setup_s=self.startup_setup_s,
                        detail="Both Qwen runners are ready. Enter a prompt to compare.")
        return super().prepared_status(prompt, budget, mode)

    def prepare_demo(self, prompt, budget, mode, callback):
        if not self.prepared_status(prompt, budget, mode)["prepared"]:
            self.initialize(callback)
        return self.prepared_status(prompt, budget, mode)

    def _packed_replay(self, prompt, input_ids, budget, until_finished, request_started_s, callback):
        request = dict(prompt=prompt, system_prompt=prompt.system_prompt, input_ids=input_ids,
                       budget=budget, until_finished=until_finished, request_started_s=request_started_s)
        if self.resident:
            try:
                event = self._replay_command(request, "completed", callback)
            except BaseException:
                self.close()
                raise
            callback(event)
            return event
        command = [sys.executable, "-u", "-m", "demo_chat.freetoken_inference", str(self.model_path)]
        child = self._start_monitored(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        completed = None
        try:
            with child.stdin:
                child.stdin.write(json.dumps(request) + "\n")
            with child.stdout:
                for line in child.stdout:
                    if not line.startswith(EVENT_PREFIX):
                        sys.stderr.write(line)
                        continue
                    event = json.loads(line[len(EVENT_PREFIX):])
                    if event["type"] == "error":
                        raise RuntimeError(event["message"])
                    callback(event)
                    if event["type"] == "completed":
                        completed = event
                        with self.process_lock:
                            if self.child is child and child.poll() is None:
                                track_worker_processes(child.pid, self.worker_handles)
                                child.terminate()
            if completed is None:
                raise RuntimeError("Packed Replay did not complete. Check the Studio session log.")
            return completed
        finally:
            self.close()

    def compare(self, prompt, budget, mode, callback, request_started_s, until_finished=False, baseline=None, **_):
        if self.shared_transport and baseline not in (None, "scheduled"):
            raise ValueError("Choose the Scheduled baseline for the 34-layer comparison.")
        scheduled = self.expert_cache_dir and self.resident and baseline in (None, "scheduled")
        if not self.expert_cache_dir or scheduled:
            input_ids = self._input_ids(prompt)
        setup_started = perf_counter()
        timing = {}
        if self.resident and (not self.expert_cache_dir or scheduled):
            self.prepare_demo(prompt, budget, mode, callback)
            self._select_runner(self.resident_reference, callback)
            setup_s = perf_counter() - setup_started
            baseline = self._packed_replay(prompt, input_ids, budget, until_finished, request_started_s, callback)
            handoff_started = perf_counter()
            callback(dict(type="status", state="switching", detail=f"Switching the RTX 4090 to {self.runner_names[mode]}…"))
            self._select_runner("freetoken", callback)
            timing["handoff_s"] = perf_counter() - handoff_started
            setup_s += timing["handoff_s"]
            timing["startup_setup_s"] = self.startup_setup_s
            try:
                candidate = self._stream(prompt, budget, mode, callback, "candidate", request_started_s, until_finished)
                result = self._comparison_result(baseline, candidate, setup_s, timing, callback)
                callback(dict(type="status", state="restoring",
                              detail="Results are ready. Switching the GPU back for the next comparison…"))
                return result
            finally:
                self._select_runner(self.resident_reference, callback)
        elif self.expert_cache_dir:
            self.prepare_demo(prompt, budget, mode, callback)
            if self.resident:
                self._select_runner("freetoken", callback)
            setup_s = perf_counter() - setup_started
            baseline = self._stream(prompt, budget, "original", callback, "baseline", request_started_s, until_finished)
            candidate = self._stream(prompt, budget, mode, callback, "candidate", request_started_s, until_finished)
            timing["startup_setup_s"] = self.startup_setup_s
        else:
            callback(dict(type="status", state="loading", detail="Switching the GPU to Packed Replay…"))
            self.close()
            try:
                callback(dict(type="status", state="loading", models_loaded=False,
                              detail="Loading Packed Replay’s BF16 weights…"))
                baseline = self._packed_replay(prompt, input_ids, budget, until_finished, request_started_s, callback)
            finally:
                self.initialize(callback)
            setup_s = max(0.0, perf_counter() - setup_started - baseline["metrics"]["seconds"])
            candidate = self._stream(prompt, budget, mode, callback, "candidate", request_started_s, until_finished)
        return self._comparison_result(baseline, candidate, setup_s, timing, callback)

    def _comparison_result(self, baseline, candidate, setup_s, timing, callback):
        if not candidate["token_ids"]:
            raise RuntimeError("FreeToken did not return the output token IDs required for comparison.")
        if self.expert_cache_dir:
            timing["stop_reason_match"] = baseline["metrics"]["stop_reason"] == candidate["metrics"]["stop_reason"]
        event = dict(compare_outputs(baseline, candidate), type="comparison", model=MODEL_NAME,
                     execution="sequential", setup_s=setup_s, **timing)
        callback(event)
        return dict(comparison=event, baseline=baseline, candidate=candidate)

    def generate(self, prompt, budget, mode, callback, request_started_s, require_prepared=True, until_finished=False):
        if self.resident:
            self._select_runner("freetoken", callback)
        try:
            return super().generate(prompt, budget, mode, callback, request_started_s, require_prepared, until_finished)
        finally:
            if self.resident:
                self._select_runner(self.resident_reference, callback)

    def _completion_prompt(self, prompt):
        text = self.tokenizer.apply_chat_template(chat_messages(prompt), add_generation_prompt=True,
                                                  tokenize=False, enable_thinking=True)
        ids = self._input_ids(prompt)
        if self.tokenizer.encode(text, add_special_tokens=True) != ids:
            raise RuntimeError("The FreeToken prompt does not round trip to the original token IDs.")
        return text, len(ids)

    def _stream(self, prompt, budget, mode, callback, runner, request_started_s, until_finished):
        label = self.runner_names[mode]
        text, count = self._completion_prompt(prompt)
        first_request_s = None

        def observe(piece, elapsed, tokens):
            nonlocal first_request_s
            if first_request_s is None:
                first_request_s = perf_counter() - request_started_s
            callback(dict(type="token", runner=runner, mode=mode, text=piece, elapsed_s=elapsed, tokens=tokens))

        callback(dict(type="started", runner=runner, mode=mode, runner_name=label))
        result = self._complete(mode, text, count, budget, True, on_text=observe)
        tokens = result["usage"]["completion_tokens"]
        seconds = result["http_wall_s"]
        finished = result["finish_reason"] == "stop"
        metrics = dict(
            seconds=seconds, tokens=tokens, tokens_per_second=tokens / seconds,
            decode_tokens_per_s=result.get("decode_tokens_per_s", result["timings"]["predicted_per_second"]),
            time_to_first_token_s=result["http_first_text_s"], request_to_first_token_s=first_request_s,
            finished_eos=finished, stop_reason="eos" if finished else
            ("context_limit" if until_finished else "max_new_tokens"),
            until_finished=until_finished, output_limit_requested=None if until_finished else budget,
            cached_prompt_tokens=result["cached_prompt_tokens"], backend_timings=result["timings"],
        )
        event = dict(type="completed", runner=runner, mode=mode, runner_name=label,
                     token_ids=result.get("token_ids"), text=result["text"], metrics=metrics)
        callback(event)
        return event


if __name__ == "__main__":
    stream_reference(Path(sys.argv[1]), *sys.argv[2:])
