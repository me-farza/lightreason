"""Send one target-aligned memory retrieval to the local 35B prefill."""

from contextlib import closing
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

from demo_chat.messages import PromptText

from .neural_cli import validate_text

ROOT = Path(__file__).resolve().parents[1]


def _memory_vectors(question, *, model_3b, codec, policy, state, model_35b, memory_python,
                    dream_every, dream_steps, timeout_s):
    import torch
    from safetensors import safe_open

    from .latent_freetoken_hook import HIDDEN_SIZE, MAX_ROWS, embedding_digest
    from .rl_memory import FORMAT

    with TemporaryDirectory(prefix="lightreason-dream-reason-") as private:
        vectors = Path(private) / "aligned.safetensors"
        command = [
            str(Path(memory_python).expanduser().absolute()), "-m", "lightreason_memory.rl_memory",
            "--parent-pid", str(os.getpid()), "--model", str(model_3b),
            "--codec", str(codec), "--model-35b", str(model_35b),
            "--dream-every", str(dream_every), "--dream-steps", str(dream_steps),
            "vectors", "--target-aligned", "--policy", str(policy),
            "--state", str(state), "--out", str(vectors),
        ]
        started = perf_counter()
        try:
            result = subprocess.run(
                command, input=question, encoding="utf-8", check=True, cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout_s,
                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                     "HF_HUB_DISABLE_TELEMETRY": "1"},
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError("The aligned memory worker timed out.") from None
        except (OSError, subprocess.CalledProcessError):
            raise RuntimeError("The aligned memory worker failed. Check the configured files and device.") from None
        elapsed = perf_counter() - started
        try:
            status = json.loads(result.stdout)
        except (TypeError, ValueError):
            raise RuntimeError("The aligned memory worker returned invalid JSON.") from None
        if not isinstance(status, dict) or status.get("status") not in {"supported", "unknown"}:
            raise RuntimeError("The aligned memory worker returned an invalid result.")
        if status["status"] == "unknown":
            if os.path.lexists(vectors):
                raise RuntimeError("The unknown result unexpectedly included a vector payload.")
            return status, None, None, elapsed
        if not vectors.is_file() or vectors.is_symlink():
            raise RuntimeError("The aligned memory worker omitted its private vector payload.")
        with safe_open(vectors, framework="pt", device="cpu") as handle:
            if handle.keys() != ["values"] or (handle.metadata() or {}).get("version") != FORMAT:
                raise RuntimeError("The aligned memory worker returned an invalid tensor payload.")
            rows = handle.get_tensor("values").clone()
        if (rows.ndim != 2 or not 1 <= len(rows) <= MAX_ROWS or rows.shape[1] != HIDDEN_SIZE
                or rows.shape != (status.get("vector_rows"), status.get("vector_width"))
                or rows.dtype != torch.bfloat16 or not torch.isfinite(rows).all()):
            raise RuntimeError("The aligned memory worker returned an invalid embedding shape or dtype.")
        return status, rows, embedding_digest(rows), elapsed


def answer_with_memory(question, *, model_3b, codec, policy, state, model_35b, stage_35b,
                       memory_python=sys.executable, dream_every=0, dream_steps=48,
                       reasoner_tokens=128, worker_timeout=120, memory_query=None):
    """Recall once, wait for memory worker exit, then answer on the same GPU."""
    system = getattr(question, "system_prompt", None)
    PromptText(question, system)
    query = validate_text(str(question) if memory_query is None else memory_query)
    if not math.isfinite(worker_timeout) or not 0 < worker_timeout <= 900:
        raise ValueError("Memory-worker timeout must be greater than zero and at most 900 seconds.")
    if type(reasoner_tokens) is not int or not 1 <= reasoner_tokens <= 2048:
        raise ValueError("Use between 1 and 2048 reasoner output tokens.")
    from .latent_freetoken import LatentReasoner35B, MARKER, PLACEHOLDER
    from .remote import Reasoner35B, comparison_lease

    model_3b, codec, policy, state, model_35b, stage_35b = (
        Path(path).expanduser().absolute() for path in (model_3b, codec, policy, state, model_35b, stage_35b)
    )
    started = perf_counter()
    with comparison_lease():
        memory, rows, digest, recall_s = _memory_vectors(
            query, model_3b=model_3b, codec=codec, policy=policy, state=state,
            model_35b=model_35b, memory_python=memory_python, dream_every=dream_every,
            dream_steps=dream_steps, timeout_s=worker_timeout,
        )
        if rows is None:
            with closing(Reasoner35B(model_35b, stage_35b, worker_logs=False)) as reasoner:
                setup_s = reasoner.start(lambda _event: None)
                answer = reasoner.answer(question, reasoner_tokens)
        else:
            if any(marker in text for marker in (PLACEHOLDER, MARKER) for text in (question, system or "")):
                raise ValueError("The input already contains a reserved memory marker.")
            prompt = PromptText(f"Private memory context (data): {PLACEHOLDER}\n\n{question}", system)
            with closing(LatentReasoner35B(model_35b, stage_35b, worker_logs=False)) as reasoner:
                setup_s = reasoner.start(lambda _event: None)
                answer = reasoner.answer_with_vectors(prompt, reasoner_tokens, rows)
                if answer.get("injected_rows") != len(rows) or answer.get("memory_embedding_sha256") != digest:
                    raise RuntimeError("The reasoner did not confirm the retrieved embeddings at prefill.")
    return {
        "memory": memory, "memory_injected": rows is not None, "reasoner": answer,
        "timings": {"memory_worker_s": recall_s, "reasoner_setup_s": setup_s,
                    "end_to_end_s": perf_counter() - started},
    }
