"""Read or update a personal neural state with the trained Qwen writer and reader."""

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys


MAX_TEXT = 1024


def add_options(parser):
    parser.add_argument("--model-path", type=Path, required=True, help="pinned local Qwen checkpoint")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="trained writer and reader checkpoint")
    parser.add_argument("--state-path", type=Path, required=True, help="private state path for one user")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))


def validate_text(text):
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT or "\0" in text:
        raise ValueError(f"Enter nonblank text of at most {MAX_TEXT} characters.")
    return text


def run_operation(args, text):
    with redirect_stdout(sys.stderr):
        from .neural import NeuralMemory

        model = NeuralMemory(
            args.model_path.expanduser(), args.checkpoint_path.expanduser(), args.state_path.expanduser(),
            device=args.device,
        )
        try:
            return getattr(model, args.action)(text)
        finally:
            model.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_options(parser)
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("absorb", help="read an approved experience from stdin and update the state")
    actions.add_parser("recall", help="read a question from stdin and answer from the state")
    args = parser.parse_args(argv)
    try:
        if args.parent_pid is not None:
            from .backend import _exit_with_parent

            _exit_with_parent(args.parent_pid)
        text = validate_text(sys.stdin.read(MAX_TEXT + 1))
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        result = run_operation(args, text)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
