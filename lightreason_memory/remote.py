"""Use the staged, single-worker Qwen3.6 reasoner on the GPU host."""

from contextlib import contextmanager
import argparse
import fcntl
import json
import os
from pathlib import Path
from secrets import token_urlsafe
import signal
import socket
import stat
import subprocess
import sys
import threading
from time import perf_counter
from urllib.request import getproxies, proxy_bypass

from demo_chat.edge_inference import CONTEXT_CAPACITY
from demo_chat.freetoken_inference import PORT, FreeTokenStudioInference, shared_cpu_settings
from demo_chat.messages import chat_messages
from edge35b.download import REVISION


REASONER_VERSION = f"Qwen/Qwen3.6-35B-A3B@{REVISION}"


def nvidia_smi(query):
    try:
        return subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("The NVIDIA driver did not answer. Check the staged driver environment.") from exc


def require_idle_gpu():
    inventory = nvidia_smi("--query-gpu=name").splitlines()
    if len(inventory) != 1 or "RTX 4090" not in inventory[0]:
        raise RuntimeError("This reasoner stage was validated for one RTX 4090.")
    if nvidia_smi("--query-compute-apps=pid"):
        raise RuntimeError("The RTX 4090 already has a compute workload. Retry after it exits.")


def _lease(suffix):
    path = f"/tmp/lightreason-memory-{os.getuid()}-{suffix}.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RuntimeError("Memory GPU lease files must be private and owned by the host account.")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A Memory worker or comparison is already using this GPU host.") from exc
        require_idle_gpu()
        return fd
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def comparison_lease():
    """Serialize Memory comparisons by this host account and leave other workloads alone."""
    fd = _lease("comparison")
    try:
        yield
    finally:
        os.close(fd)


def _memory_sample():
    gpu_mib = int(nvidia_smi("--query-gpu=memory.used"))
    host = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, values = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable"}:
            host[name] = int(values.split()[0]) * 1024
    return gpu_mib * (1 << 20), host["MemTotal"] - host["MemAvailable"]


class SampledMemory:
    """Sample whole-device VRAM and system RAM pressure once per second."""

    def __init__(self):
        self.peak = None
        self.samples = 0
        self.failed_samples = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        try:
            sample = _memory_sample()
        except (RuntimeError, ValueError, KeyError, OSError):
            self.failed_samples += 1
            return
        self.peak = sample if self.peak is None else tuple(map(max, zip(self.peak, sample)))
        self.samples += 1

    def _run(self):
        while not self.stop.wait(1):
            self._sample()

    def __enter__(self):
        self._sample()
        self.baseline = self.peak
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self._sample()

    def report(self):
        return dict(
            interval_s=1, samples=self.samples, failed_samples=self.failed_samples,
            baseline_vram_bytes=self.baseline[0] if self.baseline else None,
            baseline_host_used_bytes=self.baseline[1] if self.baseline else None,
            peak_vram_bytes=self.peak[0] if self.peak else None,
            peak_host_used_bytes=self.peak[1] if self.peak else None,
        )


class Reasoner35B(FreeTokenStudioInference):
    """The audited Studio runner's optimized arm, without its Scheduled worker."""

    def __init__(self, model_path, stage, *, worker_logs=True):
        super().__init__(model_path, expert_cache_dir=stage, api_key=token_urlsafe(32))
        self.gpu_lease = None
        self.worker_logs = worker_logs
        self.shared_transport, self.shared_env = shared_cpu_settings(self.expert_cache_dir)
        # No second resident runner needs to park or restore this cache. The
        # staged allocator builds the same cache directly when parking is off.
        self.shared_env["LIGHTREASON_STUDIO_CACHE_PARK"] = "0"
        self.runner_names = {"optimized": "Lightreason · 34-layer CPU/CUDA"}

    def start(self, callback):
        if getproxies().get("http") and not proxy_bypass("127.0.0.1"):
            raise RuntimeError("Exclude 127.0.0.1 from HTTP proxying before running the local reasoner.")
        require_idle_gpu()
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", PORT))
            except OSError as exc:
                raise RuntimeError("The staged FreeToken loopback port is already in use.") from exc
        self.initialize(callback)
        return self.startup_setup_s

    def _initialize_freetoken(self, callback):
        # Use the 3B backend's own GPU lock, acquired only after its process exits.
        self.gpu_lease = _lease("gpu")

        def status(event):
            if event.get("detail") == "Lightreason is warm. Preparing Scheduled…":
                event = dict(event, detail="The staged 35B reasoner is ready.")
            callback(event)

        super()._initialize_freetoken(status)

    def _start_monitored(self, command, **options):
        if not self.worker_logs:
            options.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return super()._start_monitored(command, **options)

    def close(self):
        super().close()
        if self.gpu_lease is not None:
            os.close(self.gpu_lease)
            self.gpu_lease = None

    def _input_ids(self, prompt):
        return self.tokenizer.apply_chat_template(chat_messages(prompt), add_generation_prompt=True,
                                                  tokenize=True, return_dict=True, enable_thinking=False)["input_ids"]

    def _completion_prompt(self, prompt):
        # FreeToken's raw endpoint accepts the rendered checkpoint template. The
        # template provides an empty think section so its response starts in the final answer.
        rendered = self.tokenizer.apply_chat_template(chat_messages(prompt), add_generation_prompt=True,
                                                      tokenize=False, enable_thinking=False)
        ids = self._input_ids(prompt)
        if self.tokenizer.encode(rendered, add_special_tokens=True) != ids:
            raise RuntimeError("The FreeToken prompt does not round trip to the original token IDs.")
        return rendered, len(ids)

    def answer(self, prompt, token_budget):
        request_started = perf_counter()
        prompt_tokens = len(self._input_ids(prompt))
        if prompt_tokens + token_budget > CONTEXT_CAPACITY:
            raise ValueError("The complete reasoner prompt and output exceed its staged 4,096-token context.")
        self.resolve_budget(prompt, token_budget)
        first_final = None

        def observe(event):
            nonlocal first_final
            if event["type"] == "token" and first_final is None and event["text"].strip():
                first_final = perf_counter()

        event = self.generate(prompt, token_budget, "optimized", observe, request_started)["generation"]
        completed = perf_counter()
        metrics = event["metrics"]
        backend = metrics["backend_timings"]
        answer = event["text"].strip()
        if "<think>" in answer or "</think>" in answer:
            # Exclude any unexpectedly emitted thinking from the response.
            _, boundary, answer = answer.partition("</think>")
            answer = answer.strip() if boundary else ""
            first_final = None
        queue_ms = backend.get("queue_ms")
        return dict(
            answer=answer, final_answer_present=bool(answer),
            prompt_tokens=prompt_tokens, generated_tokens=metrics["tokens"], stop_reason=metrics["stop_reason"],
            model_version=REASONER_VERSION, reasoner_request_s=completed - request_started,
            reasoner_prefill_s=backend["prompt_ms"] / 1000, reasoner_decode_s=backend["predicted_ms"] / 1000,
            reasoner_queue_s=queue_ms / 1000 if queue_ms is not None else None,
            reasoner_first_streamed_text_s=metrics["request_to_first_token_s"],
            reasoner_first_final_text_s=first_final - request_started if first_final is not None else None,
        )


def main():
    from .__main__ import _read_stdin, add_options, make_service
    from .reasoning import pre_recall
    from .service import MAX_QUESTION

    parser = argparse.ArgumentParser(description="Read a question from stdin, recall once, and run the local 35B reasoner.")
    add_options(parser)
    parser.add_argument("--model-35b", type=Path, required=True)
    parser.add_argument("--stage-35b", type=Path, required=True)
    parser.add_argument("--memory-tokens", type=int, default=256, choices=range(1, 513), metavar="1..512")
    parser.add_argument("--reasoner-tokens", type=int, default=384, choices=range(1, 2049), metavar="1..2048")
    parser.add_argument("--deadline", type=int, default=900, help="total request deadline in seconds")
    parser.add_argument("--as-of")
    args = parser.parse_args()
    if not 1 <= args.deadline <= 1800:
        parser.error("--deadline must be between 1 and 1800 seconds")

    def stop(*_):
        raise KeyboardInterrupt

    def expire(*_):
        raise TimeoutError("The memory reasoning request timed out. Its workers were released.")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, expire)
    try:
        question = _read_stdin(MAX_QUESTION)
        signal.setitimer(signal.ITIMER_REAL, args.deadline)
        with comparison_lease():
            reasoner = Reasoner35B(args.model_35b, args.stage_35b, worker_logs=False)
            service = make_service(args)

            def generate(prompt):
                try:
                    setup_s = reasoner.start(lambda _event: None)
                    answer = reasoner.answer(prompt, args.reasoner_tokens)
                except TimeoutError as exc:
                    raise RuntimeError("The reasoner timed out. Its worker will be released.") from exc
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise RuntimeError("The reasoner ran out of memory. Its worker will be released.") from exc
                    raise RuntimeError("The reasoner could not complete this request.") from exc
                return dict(answer, reasoner_setup_s=setup_s)

            try:
                started = perf_counter()
                result = pre_recall(service, generate, question, args.memory_tokens, args.as_of)
                result["end_to_end_s"] = perf_counter() - started
            finally:
                reasoner.close()
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, KeyError, RuntimeError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Stopped the memory reasoning request.\n")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
