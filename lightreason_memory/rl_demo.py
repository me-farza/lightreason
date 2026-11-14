"""Replay a held-out RL memory correction across fresh local processes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

from .neural_demo import _digest, _tensor_layout, _write_report
from .rl_episodes import make_episodes


ROOT = Path(__file__).resolve().parents[1]
_WORKER = """
import contextlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from lightreason_memory.backend import _exit_with_parent, _gpu_lease
from lightreason_memory.neural_demo import _digest

model_path, codec, policy, state, device, parent_pid = sys.argv[1:]
_exit_with_parent(int(parent_pid))
operations = json.load(sys.stdin)
state = Path(state)
with contextlib.redirect_stdout(sys.stderr):
    import torch
    from lightreason_memory.rl_memory import PolicyMemory

    torch.set_num_threads(1)
    lease = _gpu_lease() if device == "cuda" else None
    try:
        started = perf_counter()
        memory = PolicyMemory(Path(model_path), Path(codec), Path(policy), state, torch.device(device))
        _ = memory.encoder
        load_s = perf_counter() - started
        operations_report = []
        for action, text in operations:
            if action not in {"absorb", "recall"}:
                raise ValueError("Unexpected synthetic demo operation")
            before = _digest(state)
            started = perf_counter()
            result = getattr(memory, action)(text)
            operations_report.append({
                "response": result, "operation_s": perf_counter() - started,
                "state_before_sha256": before, "state_after_sha256": _digest(state),
            })
        close = getattr(memory, "close", None)
        if close is not None:
            close()
    finally:
        if lease is not None:
            os.close(lease)
print(json.dumps({"pid": os.getpid(), "load_s": load_s, "operations": operations_report}, allow_nan=False))
"""


def _run_worker(args, state, operations):
    command = [
        sys.executable, "-B", "-c", _WORKER, str(args.model), str(args.codec), str(args.policy), str(state),
        args.device, str(os.getpid()),
    ]
    # The parent strips expected answers and labels before the subprocess.
    payload = [[action, text] for action, text, _ in operations]
    started = perf_counter()
    process = subprocess.run(
        command, input=json.dumps(payload), capture_output=True, text=True,
        cwd=ROOT, timeout=args.worker_timeout,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"},
    )
    if process.returncode:
        raise RuntimeError(f"Memory worker exited {process.returncode}: {process.stderr[-3000:]}")
    result = json.loads(process.stdout)
    result["process_wall_s"] = perf_counter() - started
    for (action, text, expected), observed in zip(operations, result["operations"], strict=True):
        response = observed["response"]
        changed = observed["state_before_sha256"] != observed["state_after_sha256"]
        if action == "recall":
            status = "unknown" if expected is None else "supported"
            passed = not changed and response.get("status") == status and response.get("answer", "") == (expected or "")
        else:
            status = "supported" if expected else "unknown"
            passed = changed == expected and response.get("status") == status and response.get("updated") == expected
        observed.update(action=action, text=text, expected=expected, passed=passed)
    return result


def _plan(seed, transient=False):
    for episode in make_episodes(seed, 32, "test", transient=transient):
        observations, queries = episode["observations"], episode["queries"]
        correction_index = next(i for i, row in enumerate(observations) if row["is_correction"])
        correction = observations[correction_index]
        if correction["key"] == "lantern":
            break
    else:
        raise ValueError("The selected seed did not provide a lantern correction in the first 32 episodes")

    pre = next(row for row in reversed(queries) if row["key"] == "lantern" and row["after_write"] <= correction_index)
    post = next(row for row in queries if row["key"] == "lantern" and row["kind"] == "correction")
    final_length = len(observations)
    controls = [row for row in queries if row["kind"] in ("retention", "unknown", "transient")
                and row["after_write"] == final_length]
    post_reads = [("recall", pre["text"], post["value"])]
    if post["text"] != pre["text"]:
        post_reads.append(("recall", post["text"], post["value"]))
    post_reads.extend(("recall", row["text"], row["value"]) for row in controls)
    writes = [("absorb", row["text"], not (row["is_distractor"] or row.get("is_transient")))
              for row in observations]
    return episode, (
        ("initialize", [("recall", pre["text"], None), *writes[:correction_index]]),
        ("restore_and_correct", [("recall", pre["text"], pre["value"]), *writes[correction_index:]]),
        ("restore_after_correction", post_reads),
    )


def evaluate(args, report):
    episode, phases = _plan(args.seed, args.transient)
    report["episode_id"] = episode["episode_id"]
    policy_digest = _digest(args.policy)
    codec_digest = _digest(args.codec)
    with TemporaryDirectory(prefix="lightreason-rl-demo-") as directory:
        state = Path(directory) / "state.safetensors"
        for name, operations in phases:
            phase = _run_worker(args, state, operations)
            phase["phase"] = name
            phase["state_sha256"] = _digest(state)
            report["phases"].append(phase)
            _write_report(args.report, report)

        workers = report["phases"]
        report["writer_and_readers_are_distinct_processes"] = len({row["pid"] for row in workers}) == len(phases)
        report["restore_digest_matches"] = all(
            right["operations"][0]["state_before_sha256"] == left["state_sha256"]
            for left, right in zip(workers, workers[1:])
        )
        report["policy_and_codec_unchanged"] = _digest(args.policy) == policy_digest and _digest(args.codec) == codec_digest
        if state.is_file():
            report["state_bytes"] = state.stat().st_size
            report["state_tensor_layout"] = _tensor_layout(state)
            state_bytes = state.read_bytes()
            observation_texts = (row["text"].encode("utf-8") for row in episode["observations"])
            report["no_full_observation_text_in_state"] = not any(text in state_bytes for text in observation_texts)
        report["passed"] = (
            report["writer_and_readers_are_distinct_processes"] and report["restore_digest_matches"]
            and report["policy_and_codec_unchanged"] and report.get("no_full_observation_text_in_state", False)
            and all(operation["passed"] for row in workers for operation in row["operations"])
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--transient", action="store_true", help="include a true temporary observation that should not persist")
    parser.add_argument("--worker-timeout", type=int, default=180)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    for name in ("model", "codec", "policy", "report"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.expanduser().resolve())
    if args.worker_timeout <= 0 or (args.report is not None and args.report.exists()):
        parser.error("Choose a positive worker timeout and a fresh report path")
    report = {
        "passed": False, "seed": args.seed, "device": args.device, "transient": args.transient, "phases": [],
        "protocol": (
            "Replay deterministic held-out observations as chronological text. Each phase starts in a fresh process "
            "with the same policy and state path. The benchmark parent keeps the expected answers and never supplies "
            "them or prior observations to a recall operation."
        ),
    }
    try:
        evaluate(args, report)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        report["error"] = str(exc) or exc.__class__.__name__
    _write_report(args.report, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
