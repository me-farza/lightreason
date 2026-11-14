"""Command-line access to Lightreason Memory and its runnable demonstration."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from .backend import DEFAULT_RUNTIME, QwenMemoryBackend
from .demo import (
    DEMO_AS_OF, DEMO_CORRECTED_AT, DEMO_CORRECTION_QUESTION, DEMO_QUESTION, DEMO_UNKNOWN,
    WEATHER, seed_demo,
)
from .service import MAX_QUESTION, MemoryService
from .store import MAX_CONTENT_LENGTH, Store

ROOT = Path(__file__).resolve().parent.parent


def add_options(parser, *, require_db=True):
    parser.add_argument("--db", type=Path, required=require_db)
    parser.add_argument("--user", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--model-path", type=Path, default=ROOT / "model-3b")
    parser.add_argument("--runtime-path", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--inference-python", type=Path)
    parser.add_argument("--timeout", type=float, default=120, help="seconds per worker operation")


def make_service(args, db=None):
    return MemoryService(
        Store(db if db is not None else args.db, args.user, args.workspace),
        QwenMemoryBackend(
            args.model_path, runtime_path=args.runtime_path, python_executable=args.inference_python,
            timeout_s=args.timeout,
        ),
    )


def _read_stdin(limit):
    value = sys.stdin.read(limit + 1)
    if len(value) > limit:
        raise ValueError(f"stdin exceeds {limit} characters")
    return value.rstrip("\n")


def _demo(args):
    with tempfile.TemporaryDirectory(prefix="lightreason-memory-demo-") as directory:
        db = Path(directory) / "benchmarks.sqlite3"
        # The ingesting process exits before the service opens the same database.
        seeded = subprocess.run(
            [sys.executable, "-m", "lightreason_memory", "--db", str(db),
             "--user", args.user, "--workspace", args.workspace,
             "seed-demo", "--source-root", str(args.source_root)],
            cwd=ROOT, check=True, capture_output=True, text=True, timeout=30,
        )
        records = json.loads(seeded.stdout)
        service = make_service(args, db=db)
        try:
            initial = service.recall(DEMO_QUESTION)
            weather = records[2]
            try:
                cap = json.loads((args.source_root / WEATHER).read_text())["output_limit"]
            except FileNotFoundError:
                cap = None
            addition = (
                f"The Studio weather comparison used a {cap}-token output cap." if cap is not None else
                "The Studio weather comparison's output cap is absent from the available artifacts."
            )
            correction = service.correct(
                weather["memory_id"],
                weather["content"] + " " + addition,
                weather["source"], DEMO_CORRECTED_AT,
            )
            current = service.recall(DEMO_CORRECTION_QUESTION)
            historical = service.recall(DEMO_CORRECTION_QUESTION, as_of=DEMO_AS_OF)
            unknown = service.recall(DEMO_UNKNOWN)
        finally:
            service.close()
    return {
        "seed_process_exited": True, "initial": initial, "correction": correction["memory_id"],
        "current": current, "as_of": DEMO_AS_OF, "historical": historical, "absent": unknown,
        "checks": {
            "all_agreement_sources": set(initial["source_ids"]) == {record["memory_id"] for record in records},
            "correction_cited": correction["memory_id"] in current["source_ids"],
            "correction_hidden_before_event_time": correction["memory_id"] not in historical["source_ids"],
            "absent_abstained": unknown["status"] == "unknown" and not unknown["claims"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_options(parser, require_db=False)
    actions = parser.add_subparsers(dest="action", required=True)
    observe = actions.add_parser("observe")
    for name in ("event-id", "source", "timestamp", "kind"):
        observe.add_argument("--" + name, required=True)
    body = observe.add_mutually_exclusive_group(required=True)
    body.add_argument("--content")
    body.add_argument("--stdin", action="store_true")
    correct = actions.add_parser("correct")
    for name in ("memory-id", "source", "timestamp"):
        correct.add_argument("--" + name, required=True)
    body = correct.add_mutually_exclusive_group(required=True)
    body.add_argument("--replacement")
    body.add_argument("--stdin", action="store_true")
    recall = actions.add_parser("recall")
    recall.add_argument("question", nargs="?")
    recall.add_argument("--stdin", action="store_true")
    recall.add_argument("--token-budget", type=int, default=256)
    recall.add_argument("--as-of")
    for action in ("seed-demo", "demo"):
        actions.add_parser(action).add_argument("--source-root", type=Path, default=ROOT)

    args = parser.parse_args()
    if args.action != "demo" and args.db is None:
        parser.error("--db is required for this command")
    try:
        if args.action == "demo":
            result = _demo(args)
        elif args.action == "seed-demo":
            result = seed_demo(Store(args.db, args.user, args.workspace), args.source_root)
        elif args.action == "observe":
            result = Store(args.db, args.user, args.workspace).observe(
                args.event_id, _read_stdin(MAX_CONTENT_LENGTH) if args.stdin else args.content,
                args.source, args.timestamp, args.kind,
            )
        elif args.action == "correct":
            result = Store(args.db, args.user, args.workspace).correct(
                args.memory_id, _read_stdin(MAX_CONTENT_LENGTH) if args.stdin else args.replacement,
                args.source, args.timestamp,
            )
        else:
            if args.stdin == (args.question is not None):
                parser.error("use a question argument or --stdin")
            service = make_service(args)
            try:
                question = _read_stdin(MAX_QUESTION) if args.stdin else args.question
                result = service.recall(question, args.token_budget, args.as_of)
            finally:
                service.close()
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
