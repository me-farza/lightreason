"""Audit a vector-only memory swap across fresh Qwen workers and one 35B session."""

import argparse
from contextlib import closing
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from demo_chat.messages import PromptText


QUESTION = "What color did I mention for the lantern?"
SYSTEM = ("Use the remembered value to answer with a single color. "
          "If no memory was supplied, answer unknown.")
VALUES = {"session_a": "purple", "session_b": "gray", "empty": None}
_READ_WORKER = """
from unittest.mock import patch

from lightreason_memory.neural import NeuralModel
from lightreason_memory.rl_memory import FrozenQwenFeatures, main

with (
    patch.object(FrozenQwenFeatures, "decode", side_effect=AssertionError("Memory decoder was called")),
    patch.object(NeuralModel, "decode", side_effect=AssertionError("Value decoder was called")),
    patch.object(NeuralModel, "encode", side_effect=AssertionError("Question was encoded as a memory value")),
):
    main()
"""


def _digest(path):
    if path.is_file():
        with path.open("rb") as stream:
            return file_digest(stream, "sha256").hexdigest()
    return None


def _worker(args, state, action, payload, flags, *, cadence):
    entry = ["-c", _READ_WORKER] if action == "vectors" else ["-m", "lightreason_memory.rl_memory"]
    command = [
        args.memory_python, *entry, "--parent-pid", str(os.getpid()), "--model", str(args.model_3b),
        "--codec", str(args.codec), "--model-35b", str(args.model_35b), "--dream-every", str(cadence),
        action, "--policy", str(args.policy), "--state", str(state), *flags,
    ]
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    ) as process:
        try:
            output, _ = process.communicate(payload, timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        if process.returncode:
            raise RuntimeError(f"Memory {action} worker exited with code {process.returncode}.")
        return process.pid, json.loads(output)


def _prepare(args, directory):
    import torch
    from safetensors.torch import load_file

    from .latent_freetoken_hook import HIDDEN_SIZE, embedding_digest

    writer_pids = {}
    for name, value in VALUES.items():
        if value is not None:
            text = f"I can confirm the lantern is {value}."
            feedback = json.dumps({"text": text, "expected_key": "lantern", "expected_value": value})
            pid, reply = _worker(args, directory / f"{name}.safetensors", "absorb", feedback,
                                 ["--feedback-stdin"], cadence=2)
            if reply.get("updated") is not True or reply["dream"]["generation"] != 0:
                raise RuntimeError(f"The writer did not create an independent saved state for {name}.")
            writer_pids[name] = pid

    reads = {}
    vectors = {}
    saved_addresses = {}
    for name, value in VALUES.items():
        state = directory / f"{name}.safetensors"
        output = directory / f"{name}-vectors.safetensors"
        before = _digest(state)
        pid, reply = _worker(args, state, "vectors", QUESTION,
                             ["--target-aligned", "--out", str(output)], cadence=0)
        if pid in writer_pids.values() or _digest(state) != before:
            raise RuntimeError(f"{name} did not perform a read-only recall in a fresh worker.")
        rows = None
        digest = None
        if value is None:
            if reply != {"status": "unknown"} or output.exists() or before is not None:
                raise RuntimeError("An empty bank must emit no vectors or generated answer.")
        else:
            if reply != {"status": "supported", "vector_rows": 1, "vector_width": HIDDEN_SIZE}:
                raise RuntimeError(f"Unexpected vector-only response for {name}.")
            payload = load_file(output)
            rows = payload["values"]
            saved = load_file(state)
            slots = saved["occupied"][0].nonzero().flatten()
            if len(slots) != 1:
                raise RuntimeError(f"{name} must contain exactly one occupied slot.")
            slot = slots.item()
            saved_addresses[name] = saved["keys"][0, slot]
            length = saved["target_lengths"][slot].item()
            persisted = saved["target_rows"][slot, :length]
            if (set(payload) != {"values"} or rows.dtype != torch.bfloat16 or rows.shape != (1, HIDDEN_SIZE)
                    or rows.shape != persisted.shape or embedding_digest(rows) != embedding_digest(persisted)):
                raise RuntimeError(f"The outgoing rows differ from the saved memory for {name}.")
            digest = embedding_digest(rows)
        vectors[name] = rows
        reads[name] = {
            "writer_pid": writer_pids.get(name), "reader_pid": pid, "memory": reply,
            "state_sha256": before, "embedding_sha256": digest,
            "payload_matches_saved_rows": True if rows is not None else None, "value_codec_guard": True,
        }
        print(f"{name} memory: {reply}, verified saved row: {rows is not None}", flush=True)
    if reads["session_a"]["embedding_sha256"] == reads["session_b"]["embedding_sha256"]:
        raise RuntimeError("The two memories produced identical embedding rows.")
    if not torch.equal(saved_addresses["session_a"], saved_addresses["session_b"]):
        raise RuntimeError("The two memories must use the same address.")
    return reads, vectors


def evaluate(args):
    from .latent_freetoken import LatentReasoner35B, MARKER, PLACEHOLDER
    from .remote import comparison_lease

    prompt = PromptText(QUESTION, SYSTEM)
    with_memory = PromptText(f"Remembered context: {PLACEHOLDER}\n\n{QUESTION}", SYSTEM)
    if any(value in str(with_memory) or value in SYSTEM for value in VALUES.values() if value):
        raise RuntimeError("The 35B prompt contains one of the stored values.")
    report = {"question": QUESTION, "reasoner_tokens": 24, "runs": {}}
    with comparison_lease():
        with TemporaryDirectory(prefix="lightreason-dream-swap-") as folder:
            reads, vectors = _prepare(args, Path(folder))
        report["memory_reads"] = reads
        print("Saved-row checks passed. Starting 35B.", flush=True)
        with closing(LatentReasoner35B(args.model_35b, args.stage_35b)) as reasoner:
            def status(event):
                if event.get("detail"):
                    print(event["detail"], flush=True)

            report["reasoner_setup_s"] = reasoner.start(status)
            hit_ids = reasoner._input_ids(PromptText(str(with_memory).replace(PLACEHOLDER, MARKER), SYSTEM))
            empty_ids = reasoner._input_ids(prompt)
            if hit_ids == empty_ids:
                raise RuntimeError("The memory and ordinary prompts unexpectedly use the same input IDs.")
            for name, expected in VALUES.items():
                rows = vectors[name]
                if rows is None:
                    reply = reasoner.answer(prompt, report["reasoner_tokens"])
                    ids = empty_ids
                else:
                    reply = reasoner.answer_with_vectors(with_memory, report["reasoner_tokens"], rows)
                    if (reply.get("injected_rows") != len(rows)
                            or reply.get("memory_embedding_sha256") != reads[name]["embedding_sha256"]):
                        raise RuntimeError(f"35B did not consume the saved rows for {name}.")
                    ids = hit_ids
                answer = reply["answer"]
                passed = answer.strip().casefold().rstrip(".") == (expected or "unknown")
                report["runs"][name] = {
                    "answer": answer, "expected": expected or "unknown", "passed": passed,
                    "injected_rows": reply.get("injected_rows", 0), "prompt_tokens": len(ids),
                    "prompt_ids_sha256": sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                    "embedding_sha256": reply.get("memory_embedding_sha256"),
                }
                print(f"{name} 35B: {answer!r}, rows: {reply.get('injected_rows', 0)}, passed: {passed}", flush=True)
    report["passed"] = all(run["passed"] for run in report["runs"].values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("model-3b", "codec", "policy", "model-35b", "stage-35b", "report"):
        parser.add_argument(f"--{path}", required=True, type=Path)
    parser.add_argument("--memory-python", default=sys.executable)
    parser.add_argument("--prepare-only", action="store_true", help="audit saved rows without starting 35B")
    args = parser.parse_args()
    args.report = args.report.resolve()
    if args.report.exists():
        parser.error("Choose a fresh report path.")
    if args.prepare_only:
        from .remote import comparison_lease

        with comparison_lease(), TemporaryDirectory(prefix="lightreason-dream-swap-") as folder:
            reads, _ = _prepare(args, Path(folder))
        result = {"question": QUESTION, "memory_reads": reads, "passed": True}
    else:
        result = evaluate(args)
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
