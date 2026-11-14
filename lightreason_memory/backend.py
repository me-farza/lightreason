"""Private CPU tokenization and single-owner CUDA readout for Memory v1."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import selectors
import signal
import stat
import subprocess
import sys
import threading
from time import perf_counter


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
MODEL_VERSION = f"{MODEL_ID}@{REVISION}"
CONTEXT_CAPACITY = 4096
RESERVE_TOKENS = 64
MAX_PROMPT_CHARACTERS = 8000
MAX_SYSTEM_CHARACTERS = 4000
MAX_WIRE_BYTES = 256 * 1024
_ROOT = Path(__file__).resolve().parents[1]
_NESTED_RUNTIME = _ROOT / "release" / "qwen25-3b"
DEFAULT_RUNTIME = _NESTED_RUNTIME if _NESTED_RUNTIME.is_dir() else _ROOT


class BackendError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _remaining(deadline):
    seconds = deadline - perf_counter()
    if seconds <= 0:
        raise BackendError("timeout", "The memory backend exceeded its request deadline.")
    return seconds


class _Worker:
    def __init__(self, argv):
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"}
        try:
            self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, env=env, bufsize=0,
                                            start_new_session=True)
        except OSError as exc:
            raise BackendError("unavailable", "The configured 3B Python worker could not start.") from exc
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)

    def request(self, payload, deadline):
        encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
        if len(encoded) > MAX_WIRE_BYTES:
            raise BackendError("capacity", "The request exceeds the worker transport limit.")
        input_fd = self.process.stdin.fileno()
        output_fd = self.process.stdout.fileno()
        try:
            with selectors.DefaultSelector() as poll:
                poll.register(input_fd, selectors.EVENT_WRITE)
                sent = 0
                while sent < len(encoded):
                    if not poll.select(_remaining(deadline)):
                        _remaining(deadline)
                        continue
                    sent += os.write(input_fd, encoded[sent:])
                poll.unregister(input_fd)
                poll.register(output_fd, selectors.EVENT_READ)
                response = bytearray()
                while not response.endswith(b"\n"):
                    if not poll.select(_remaining(deadline)):
                        _remaining(deadline)
                        continue
                    chunk = os.read(output_fd, 65536)
                    if not chunk or len(response) + len(chunk) > MAX_WIRE_BYTES:
                        raise BackendError("unavailable", "The memory worker exited or returned an invalid response.")
                    response.extend(chunk)
            result = json.loads(response)
            if not isinstance(result, dict):
                raise ValueError("Worker response must be an object")
        except (OSError, ValueError) as exc:
            raise BackendError("unavailable", "The memory worker connection failed.") from exc
        if "error" in result:
            raise BackendError(result["error"], result["message"])
        return result

    def close(self):
        process = self.process
        process.stdin.close()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            for action in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, action)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            process.stdout.close()
        if process.poll() is None:
            raise BackendError("unavailable", "The private memory worker did not release its resources.")


class QwenMemoryBackend:
    """Use the pinned, target-only Lightreason Studio generation entry point."""

    model_version = MODEL_VERSION
    context_capacity = CONTEXT_CAPACITY
    reserve_tokens = RESERVE_TOKENS
    max_prompt_characters = MAX_PROMPT_CHARACTERS

    def __init__(self, model_path, *, runtime_path=None, python_executable=None, timeout_s=120):
        self.model_path = Path(model_path).expanduser().resolve()
        self.runtime_path = Path(runtime_path or DEFAULT_RUNTIME).expanduser().resolve()
        # A virtual environment's Python can be a symlink. Preserve its argv[0].
        self.python_executable = os.path.abspath(os.path.expanduser(python_executable or sys.executable))
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 300:
            raise ValueError("Backend timeout must be in (0, 300] seconds.")
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._workers = {}

    def _ask(self, mode, data, deadline):
        _remaining(deadline)
        current = self._workers.get(mode)
        if current is not None and current.process.poll() is not None:
            self._release(mode)
        if mode not in self._workers:
            self._workers[mode] = _Worker([self.python_executable, "-I", str(Path(__file__).resolve()),
                                           mode, str(self.model_path), str(self.runtime_path), str(os.getpid())])
        worker = self._workers[mode]
        try:
            return worker.request(data, deadline)
        except BackendError as exc:
            if exc.code not in {"capacity", "invalid_prompt"}:
                self._release(mode)
            raise

    def _release(self, mode):
        worker = self._workers.get(mode)
        if worker is not None:
            worker.close()
            del self._workers[mode]

    def _acquire(self, deadline):
        if not self._lock.acquire(timeout=_remaining(deadline)):
            raise BackendError("timeout", "The memory backend request expired in its queue.")

    def count_tokens(self, prompt, system_prompt=None):
        _validate_prompt(prompt, system_prompt)
        deadline = perf_counter() + self.timeout_s
        self._acquire(deadline)
        try:
            return self._ask("tokenize", {"prompt": prompt, "system": system_prompt}, deadline)["tokens"]
        finally:
            self._lock.release()

    def count_text_tokens(self, text):
        if not isinstance(text, str) or len(text) > MAX_PROMPT_CHARACTERS:
            raise BackendError("capacity", "The text exceeds the token counter's character limit.")
        deadline = perf_counter() + self.timeout_s
        self._acquire(deadline)
        try:
            return self._ask("tokenize", {"text": text}, deadline)["tokens"]
        finally:
            self._lock.release()

    def generate(self, prompt, system_prompt, max_new_tokens):
        _validate_prompt(prompt, system_prompt)
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens < CONTEXT_CAPACITY - RESERVE_TOKENS:
            raise BackendError("capacity", "Invalid memory worker output limit.")
        received = perf_counter()
        deadline = received + self.timeout_s
        self._acquire(deadline)
        try:
            queue_s = perf_counter() - received
            data = {"prompt": prompt, "system": system_prompt}
            counting_started = perf_counter()
            prompt_tokens = self._ask("tokenize", data, deadline)["tokens"]
            if prompt_tokens + max_new_tokens + RESERVE_TOKENS > CONTEXT_CAPACITY:
                raise BackendError("capacity", "Templated prompt, output limit and runner reserve exceed 4096 tokens.")
            start = perf_counter()
            result = self._ask("generate", {**data, "max_new_tokens": max_new_tokens,
                                            "prompt_tokens": prompt_tokens, "request_started_s": received}, deadline)
            result["timings"].update(queue_s=queue_s, token_count_s=start - counting_started,
                                     worker_round_trip_s=perf_counter() - start,
                                     elapsed_s=perf_counter() - received)
            return result
        finally:
            self._lock.release()

    def release_gpu(self):
        """Wait for this backend's CUDA process to exit before loading a reasoner."""
        with self._lock:
            start = perf_counter()
            self._release("generate")
            return perf_counter() - start

    def close(self):
        """Release all private processes. The backend can be used again afterwards."""
        with self._lock:
            start = perf_counter()
            try:
                self._release("generate")
            finally:
                self._release("tokenize")
            return perf_counter() - start


def _validate_prompt(prompt, system_prompt):
    if (not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARACTERS
            or (system_prompt is not None and (not isinstance(system_prompt, str)
                                               or len(system_prompt) > MAX_SYSTEM_CHARACTERS))):
        raise BackendError("capacity", "Invalid memory prompt or system-instruction length.")


def _check_runtime(runtime_path):
    if not (runtime_path / "demo_chat" / "studio.py").is_file():
        raise BackendError("unavailable", "Provide the public, target-only Lightreason 3B runtime directory.")
    sys.path.insert(0, str(runtime_path))
    from demo_chat.inference import MAX_LEN, TREE_RESERVE
    if (MAX_LEN, TREE_RESERVE) != (CONTEXT_CAPACITY, RESERVE_TOKENS):
        raise BackendError("unavailable", "The Studio runner context settings do not match this memory adapter.")


def _check_checkpoint(model_path, runtime_path):
    """Pin the small config and tokenizer files as well as Studio's live BF16 hash."""
    checked = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "merges.txt"}
    try:
        plan = json.loads((runtime_path / "results3b" / "download-plan.json").read_text())
        if (plan["model_id"], plan["revision"]) != (MODEL_ID, REVISION):
            raise ValueError("Wrong download plan revision")
        for entry in plan["files"]:
            name = entry["rfilename"]
            if name in checked:
                with (model_path / name).open("rb") as handle:
                    data = handle.read(entry["size"] + 1)
                digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                if len(data) != entry["size"] or digest != entry["blobId"]:
                    raise ValueError("Checkpoint metadata failed its official hash")
                checked.remove(name)
        if checked:
            raise ValueError("Incomplete checkpoint metadata plan")
    except (KeyError, OSError, ValueError) as exc:
        raise BackendError("unavailable", "The pinned Qwen 3B checkpoint or tokenizer audit failed.") from exc


def _prompt_ids(tokenizer, request):
    from demo_chat.messages import PromptText, chat_messages

    _validate_prompt(request["prompt"], request["system"])
    prompt = PromptText(request["prompt"], request["system"])
    special = {item.content for item in tokenizer.added_tokens_decoder.values() if item.special}
    special.update(tokenizer.all_special_tokens)
    if any(marker and (marker in prompt or marker in (prompt.system_prompt or "")) for marker in special):
        raise BackendError("invalid_prompt", "The memory prompt contains a literal tokenizer control token.")
    ids = tokenizer.apply_chat_template(chat_messages(prompt), add_generation_prompt=True, tokenize=True)
    return prompt, ids


def _gpu_lease():
    path = f"/tmp/lightreason-memory-{os.getuid()}-gpu.lock"
    lease = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(lease)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise BackendError("unavailable", "The GPU lease file must be private and owned by the current user.")
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BackendError("busy", "Another Memory worker already owns the GPU lease.") from exc
        try:
            result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, check=True, timeout=5)
        except (OSError, subprocess.SubprocessError) as exc:
            raise BackendError("unavailable", "NVIDIA compute process telemetry is unavailable.") from exc
        if result.stdout.strip():
            raise BackendError("busy", "The GPU already has an active compute workload.")
        return lease
    except BaseException:
        os.close(lease)
        raise


def _generate(engine, request, startup_s, model_load_s):
    import torch

    prompt, ids = _prompt_ids(engine.tokenizer, request)
    if len(ids) != request["prompt_tokens"]:
        raise BackendError("unavailable", "CPU and CUDA workers disagreed about the templated token count.")
    result = engine.generate(prompt, request["max_new_tokens"], "original", lambda _event: None,
                             request_started_s=request["request_started_s"],
                             require_prepared=False)
    generation = result["generation"]
    if not generation["integrity_passed"] or not generation["same_original_target_weights"]:
        raise BackendError("generation_failed", "The original target integrity gates did not pass.")
    metrics = generation["metrics"]
    prefill = metrics["time_to_first_token_s"]
    timing = {"startup_s": startup_s, "model_load_s": model_load_s,
              "inference_s": metrics["seconds"], "prefill_s": prefill, "decode_s": metrics["seconds"] - prefill,
              "tokenization_s": generation["tokenization_s"],
              "request_to_first_token_s": metrics["request_to_first_token_s"]}
    report = {"text": generation["text"], "prompt_tokens": len(ids), "generated_tokens": metrics["tokens"],
              "stop_reason": metrics["stop_reason"], "timings": timing,
              "peak_vram_bytes": torch.cuda.max_memory_reserved(),
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    torch.cuda.reset_peak_memory_stats()
    return report


def _worker(mode, model_path, runtime_path, output):
    tokenizer = engine = lease = None
    try:
        for line in iter(lambda: sys.stdin.buffer.readline(MAX_WIRE_BYTES + 1), b""):
            try:
                if len(line) > MAX_WIRE_BYTES or not line.endswith(b"\n"):
                    raise BackendError("capacity", "Invalid worker request length.")
                request = json.loads(line)
                startup_s = model_load_s = 0.0
                if tokenizer is None:
                    startup_started = perf_counter()
                    _check_runtime(runtime_path)
                    _check_checkpoint(model_path, runtime_path)
                    if mode == "generate":
                        lease = _gpu_lease()
                        from demo_chat.studio import StudioLiveInference
                        candidate = StudioLiveInference(model_path=model_path, cooldown_s=0)
                        if candidate.available_modes != ("original", "strict"):
                            raise BackendError("unavailable", "Use the target-only public Lightreason runtime.")
                        metadata = candidate.initialize(lambda _event: None)
                        engine = candidate
                        tokenizer = engine.tokenizer
                        model_load_s = metadata["model_load_s"]
                    else:
                        from transformers import AutoTokenizer, __version__
                        if __version__ != "4.57.1":
                            raise BackendError("unavailable", "Use the pinned Transformers 4.57.1 3B environment.")
                        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                    startup_s = perf_counter() - startup_started
                if mode == "generate":
                    result = _generate(engine, request, startup_s, model_load_s)
                elif "text" in request:
                    result = {"tokens": len(tokenizer.encode(request["text"], add_special_tokens=False))}
                else:
                    _, ids = _prompt_ids(tokenizer, request)
                    result = {"tokens": len(ids)}
            except BackendError as exc:
                result = {"error": exc.code, "message": str(exc)}
            except Exception as exc:
                if "out of memory" in str(exc).lower():
                    result = {"error": "oom", "message": "The 3B worker ran out of memory and was released."}
                elif "exceeded its 120-second limit" in str(exc):
                    result = {"error": "timeout", "message": "The 3B runner exceeded its generation limit."}
                elif tokenizer is None:
                    result = {"error": "unavailable", "message": "The pinned Lightreason 3B worker failed to initialize."}
                else:
                    result = {"error": "generation_failed", "message": "The Lightreason 3B worker failed to complete the request."}
            output.write(json.dumps(result, ensure_ascii=True, allow_nan=False) + "\n")
            output.flush()
            if result.get("error") not in {None, "capacity", "invalid_prompt"}:
                break
    finally:
        if lease is not None:
            os.close(lease)


def _exit_with_parent(parent_pid):
    if sys.platform == "linux":
        import ctypes

        # The supervisor can be force-stopped while CUDA is in an active call.
        # Kernel-enforced parent-death cleanup also covers that case.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), "Could not register worker cleanup")
    if os.getppid() != parent_pid:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tokenize", "generate"))
    parser.add_argument("model_path", type=Path)
    parser.add_argument("runtime_path", type=Path)
    parser.add_argument("parent_pid", type=int)
    args = parser.parse_args()
    _exit_with_parent(args.parent_pid)
    # Keep native stdout writes from libraries out of the private protocol as well.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="ascii") as protocol:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        with redirect_stdout(sys.stderr):
            _worker(args.mode, args.model_path, args.runtime_path, protocol)
