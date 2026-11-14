"""Evaluate state-only neural recall in fresh processes using unseen synthetic facts."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import perf_counter


DEFAULT_SEED = 90101
_WORDS = (
    "coral amber saffron zircon hazel cedar opal mango velvet basalt walnut onyx jasper ochre "
    "cobalt ivy satin iris cotton jade rattan pewter rust birch violet ivory lilac mahogany pearl linen"
).split()
_SPLIT_SEEDS = {"acceptance": DEFAULT_SEED, "development": 41041, "alternate": 90102}
_MODEL_ROOT = Path(__file__).resolve().parents[1]

# Import only the implementation in the child. The parent owns the experiences
# and the answer key. A recall child is started with a question-only payload.
_WORKER = """
import contextlib
import json
import os
import sys
from time import perf_counter

model_path, checkpoint_path, state_path, device, action = sys.argv[1:]
items = json.load(sys.stdin)
with contextlib.redirect_stdout(sys.stderr):
    from lightreason_memory.neural import NeuralMemory
    started = perf_counter()
    memory = NeuralMemory(model_path, checkpoint_path, state_path, device=device)
    load_s = perf_counter() - started
    results = []
    try:
        operation = memory.recall if action == "recall" else memory.absorb
        for item in items:
            started = perf_counter()
            result = operation(item)
            results.append({"response": result, "operation_s": perf_counter() - started})
    finally:
        close = getattr(memory, "close", None)
        if close is not None:
            close()
print(json.dumps({"pid": os.getpid(), "load_s": load_s, "results": results}, allow_nan=False))
"""


def _run_worker(args, state, action, items):
    items = list(items)
    command = [
        sys.executable, "-B", "-c", _WORKER, str(args.model_path), str(args.checkpoint),
        str(state), args.device, action,
    ]
    started = perf_counter()
    # Captured output is synthetic. Keeping it out of argv also exercises how
    # a launcher should submit private observations or questions.
    process = subprocess.run(
        command, input=json.dumps(items), capture_output=True, text=True,
        cwd=_MODEL_ROOT, timeout=args.worker_timeout,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    if process.returncode != 0:
        raise RuntimeError(f"{action} worker exited {process.returncode}: {process.stderr[-3000:]}")
    result = json.loads(process.stdout)
    if type(result.get("results")) is not list or len(result["results"]) != len(items):
        raise RuntimeError(f"{action} worker returned the wrong number of responses")
    result["process_wall_s"] = perf_counter() - started
    return result


def _digest(path):
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _tensor_layout(path):
    """Inspect the safetensors header without importing the inference runtime."""
    with path.open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        if not 1 <= length <= 128 * 1024:
            raise ValueError("Invalid persisted neural-state header size")
        header = json.loads(stream.read(length))
    header.pop("__metadata__", None)
    return {name: {"dtype": entry["dtype"], "shape": entry["shape"]} for name, entry in header.items()}


def _write_report(path, report):
    if path is None:
        return
    with NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", encoding="utf-8", delete=False,
    ) as output:
        temporary = Path(output.name)
        try:
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _fixture(seed, split="acceptance"):
    vocabulary = _WORDS
    if split == "development":
        vocabulary = "fig anise cocoa oat sunbeam sunflower dandelion aspen".split()
        entities = ("trophy", "canteen", "harp", "violin")
        writers = (
            "I learned the access phrase for the {entity} is {value}.",
            "For future use, the {entity}'s access phrase will be {value}.",
            "I updated the {entity}'s access phrase to {value}.",
            "For future use, the {entity}'s access phrase will be {value}.",
        )
        primary, revised, satchel, compass, missing = (
            "Do you know the current access phrase for the {entity}?",
            "Remind me which access phrase is used for the {entity}.",
            "Remind me which access phrase is used for the {entity}.",
            "Do you know the current access phrase for the {entity}?",
            "Do you know the current access phrase for the {entity}?",
        )
    elif split == "alternate":
        entities = ("binoculars", "sextant", "rucksack", "sundial")
        writers = (
            "For this expedition, I set the access phrase for the {entity} to {value}.",
            "I realized that the access phrase for the {entity} is {value}.",
            "After packing, I set the access phrase for the {entity} to {value}.",
            "I changed the access phrase for the {entity} to {value}.",
        )
        primary, revised, satchel, compass, missing = (
            "What access phrase should I use for the {entity}?",
            "Which access phrase does the {entity} need now?",
            "Can you remind me of the access phrase for the {entity}?",
            "What is the entry code for the {entity}?",
            "Which access phrase does the {entity} need now?",
        )
    else:
        entities = ("lantern", "satchel", "compass", "telescope")
        writers = (
            "During the field test, I set the access phrase for the {entity} to {value}.",
            "I found that the access phrase for the {entity} is {value}.",
            "During the field test, I set the access phrase for the {entity} to {value}.",
            "I changed the access phrase for the {entity} to {value}.",
        )
        primary, revised, satchel, compass, missing = (
            "What access phrase did I choose for the {entity}?",
            "Which access phrase does the {entity} use now?",
            "What phrase unlocks the {entity}?",
            "What access phrase did I choose for the {entity}?",
            "Which access phrase does the {entity} use now?",
        )
    words = random.Random(seed).sample(vocabulary, 8)
    first, second, third, correction = (
        " ".join(words[start:start + 2]) for start in range(0, len(words), 2)
    )
    obs_first, obs_second, obs_third, obs_corrected = (
        template.format(entity=entity, value=value)
        for template, entity, value in zip(writers, (*entities[:3], entities[0]), (first, second, third, correction), strict=True)
    )
    primary, revised, satchel, compass, missing = (
        template.format(entity=entity)
        for template, entity in zip((primary, revised, satchel, compass, missing), (*entities[:1], *entities), strict=True)
    )
    return (
        (
            "first_experience", (obs_first,),
            (("unseen_value", primary, first), ("unseen_entity", missing, None)),
        ),
        (
            "more_experiences", (obs_second, obs_third),
            (("retained_first", primary, first), ("new_satchel", satchel, second),
             ("new_compass", compass, third), ("unseen_entity", missing, None)),
        ),
        (
            "correction", (obs_corrected,),
            (("corrected_value", revised, correction), ("old_question_gets_correction", primary, correction),
             ("retained_satchel", satchel, second),
             ("retained_compass", compass, third), ("unseen_entity", missing, None),
             ("unrelated_same_entity", f"What color is the {entities[0]}?", None)),
        ),
    )


def _score(cases, results):
    checks = []
    for (label, question, expected), result in zip(cases, results, strict=True):
        answer = result["response"].get("answer", "")
        status = result["response"].get("status")
        expected_status = "unknown" if expected is None else "supported"
        expected_answer = expected or ""
        checks.append({
            "case": label, "question": question, "expected_status": expected_status,
            "expected_answer": expected_answer, "status": status, "answer": answer,
            "passed": status == expected_status and answer == expected_answer,
        })
    return checks


def _recall_stage(args, path, cases, values):
    before = _digest(path)
    questions = [question for _, question, _ in cases]
    if any(value in question for value in values for question in questions):
        raise ValueError("A recall question contains a supplied answer")
    data = _run_worker(args, path, "recall", questions)
    return {
        **data, "state_sha256_before": before, "state_sha256_after": _digest(path),
        "no_answer_values_in_questions": True,
        "checks": _score(cases, data["results"]),
    }


def evaluate(args, report):
    fixture = _fixture(args.seed, args.split)
    values = {value for _, _, cases in fixture for _, _, value in cases if value}
    originals = {text.encode("utf-8") for _, experiences, _ in fixture for text in experiences}
    originals.update(value.encode("utf-8") for value in values)
    checkpoint_digest = _digest(args.checkpoint)
    if checkpoint_digest is None:
        raise ValueError("A trained neural-memory checkpoint is required")
    with TemporaryDirectory(prefix="lightreason-neural-accept-") as directory:
        state = Path(directory) / "memory.safetensors"
        for label, experiences, cases in fixture:
            before = _digest(state)
            write = _run_worker(args, state, "absorb", experiences)
            after = _digest(state)
            state_bytes = state.read_bytes() if after else b""
            stage = {
                "phase": label, "write": write, "state_size_bytes": state.stat().st_size if after else 0,
                "state_updated": after is not None and after != before,
                "state_tensor_layout": _tensor_layout(state) if after else None,
                "no_plaintext_experiences_or_values": bool(state_bytes) and not any(
                    phrase in state_bytes for phrase in originals
                ),
            }
            report["stages"].append(stage)
            stage["recall"] = _recall_stage(args, state, cases, values)
            recall = stage["recall"]
            writes = [row["response"] for row in write["results"]]
            stage["passed"] = (
                stage["state_updated"] and stage["no_plaintext_experiences_or_values"]
                and recall["state_sha256_before"] == recall["state_sha256_after"]
                and recall["pid"] != write["pid"] and all(row["passed"] for row in recall["checks"])
                and all(row.get("status") == "supported" and row.get("updated") is True for row in writes)
                and all(row.get("replaced") == (label == "correction") for row in writes)
            )
            _write_report(args.report, report)

        # Exactly the same final questions and checkpoint as the loaded-state
        # arm, but with a path that has never received a write.
        cases = [(label, question, None) for label, question, _ in fixture[-1][2]]
        empty_state = Path(directory) / "empty.safetensors"
        control = _recall_stage(args, empty_state, cases, values)
        report["empty_state"] = control
        report["checkpoint_unchanged"] = _digest(args.checkpoint) == checkpoint_digest
        state_sizes = {stage["state_size_bytes"] for stage in report["stages"]}
        report["constant_state_size"] = len(state_sizes) == 1 and 0 not in state_sizes
        allowed = {entry.name for entry in (state, empty_state)}
        allowed |= {name + ".lock" for name in allowed}
        report["state_directory_only_tensors_and_locks"] = all(
            entry.name in allowed for entry in state.parent.iterdir()
        )
        for label, checks in (
            ("loaded_state", [check for stage in report["stages"] for check in stage["recall"]["checks"]]),
            ("empty_state", control["checks"]),
        ):
            report.setdefault("counts", {})[label] = {"passed": sum(row["passed"] for row in checks), "total": len(checks)}
        report["passed"] = (
            all(stage["passed"] for stage in report["stages"]) and report["checkpoint_unchanged"]
            and report["constant_state_size"] and report["state_directory_only_tensors_and_locks"]
            and control["state_sha256_before"] is None and control["state_sha256_after"] is None
            and all(row["passed"] for row in control["checks"])
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=tuple(_SPLIT_SEEDS), default="acceptance")
    parser.add_argument("--seed", type=int, help="override the split's fixed value seed")
    parser.add_argument("--worker-timeout", type=int, default=180)
    args = parser.parse_args(argv)
    if args.seed is None:
        args.seed = _SPLIT_SEEDS[args.split]
    args.model_path = args.model_path.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.report is not None:
        args.report = args.report.expanduser().resolve()
        if args.report.exists():
            parser.error(f"Choose a fresh report path, {str(args.report)!r} already exists.")
    if args.worker_timeout <= 0:
        parser.error("The worker timeout must be positive.")
    report = {
        "passed": False, "split": args.split, "seed": args.seed, "device": args.device, "stages": [],
        "protocol": (
            "Synthetic experiences are generated in the parent and sent only to write workers. "
            "Recall workers are separate processes and receive only question lists, the model and "
            "checkpoint paths, and one state path. Original experience text is never written by this runner. "
            "Workers use the host process's ordinary filesystem permissions."
        ),
    }
    try:
        evaluate(args, report)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        report["error"] = str(exc)
    _write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
