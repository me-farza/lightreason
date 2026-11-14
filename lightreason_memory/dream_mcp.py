"""Expose one private dream-enabled memory policy over MCP stdio."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
from time import monotonic
from typing import Any

from .neural_cli import validate_text


ROOT = Path(__file__).resolve().parent.parent


class DreamWorker:
    """Serialize bounded RL CLI invocations and keep all evidence on stdin."""

    def __init__(self, args):
        if not math.isfinite(args.timeout) or not 0 < args.timeout <= 900:
            raise ValueError("Timeout must be greater than zero and at most 900 seconds.")
        cadence = args.dream_every if args.dream_every is not None else (2 if args.allow_absorb else 0)
        if cadence < 0 or (cadence > 0 and not args.allow_absorb):
            raise ValueError("A positive dream cadence requires --allow-absorb because recall can update policy state.")
        if not 1 <= args.dream_steps <= 128:
            raise ValueError("Use between 1 and 128 dream steps.")
        self.timeout = args.timeout
        self.lock = Lock()
        self.allow_absorb = args.allow_absorb
        self.command = [
            os.path.abspath(os.path.expanduser(args.inference_python)),
            "-m", "lightreason_memory.rl_memory",
            "--parent-pid", str(os.getpid()),
            "--model", str(args.model.expanduser().absolute()),
            "--codec", str(args.codec.expanduser().absolute()),
            "--device", args.device,
            "--dream-every", str(cadence),
            "--dream-steps", str(args.dream_steps),
        ]
        if args.model_35b is not None:
            self.command.extend(("--model-35b", str(args.model_35b.expanduser().absolute())))
        self.target = ["--policy", str(args.policy.expanduser().absolute()),
                       "--state", str(args.state.expanduser().absolute())]

    def __call__(self, action, payload=""):
        if action not in ("recall", "dream-status", "absorb"):
            raise ValueError("Unknown policy memory operation.")
        if action == "absorb" and not self.allow_absorb:
            raise ValueError("This server does not allow new observations.")
        deadline = monotonic() + self.timeout
        if not self.lock.acquire(timeout=self.timeout):
            raise RuntimeError("The policy memory worker is busy.")
        try:
            return self._call(action, payload, deadline)
        finally:
            self.lock.release()

    def _call(self, action, payload, deadline):
        timeout = deadline - monotonic()
        if timeout <= 0:
            raise RuntimeError("The policy memory worker is busy.")
        command = [*self.command, action, *self.target]
        if action == "absorb":
            command.append("--feedback-stdin")
        try:
            completed = subprocess.run(
                command, input=payload, encoding="utf-8", check=True, cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout,
                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                     "HF_HUB_DISABLE_TELEMETRY": "1"},
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("The policy memory worker timed out.") from None
        except (OSError, subprocess.CalledProcessError):
            raise RuntimeError("The policy memory worker failed. Check the configured files and device.") from None
        try:
            result = json.loads(completed.stdout)
        except ValueError:
            raise RuntimeError("The policy memory worker returned invalid JSON.") from None
        if not isinstance(result, dict):
            raise RuntimeError("The policy memory worker returned an invalid result.")
        return result


def create_server(invoke, allow_absorb=False):
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("Lightreason Dream Memory", log_level="WARNING")

    @server.tool()
    def recall(question: str) -> dict[str, Any]:
        """Ask the configured private memory a question. In learning mode this counts toward replay."""
        return invoke("recall", validate_text(question))

    @server.tool()
    def dream_status() -> dict[str, Any]:
        """Inspect automatic replay status without advancing its cadence."""
        return invoke("dream-status")

    if allow_absorb:
        @server.tool()
        def observe(experience: str, expected_key: str | None = None,
                    expected_value: str | None = None, ignore: bool = False) -> dict[str, Any]:
            """Absorb approved experience. Add exact verified labels or mark a verified ignore control."""
            if (expected_key is None) != (expected_value is None) or (ignore and expected_key is not None):
                raise ValueError("Provide both expected key and value, or mark the experience as ignored.")
            payload = {
                "text": validate_text(experience),
                "expected_key": None if expected_key is None else validate_text(expected_key),
                "expected_value": None if expected_value is None else validate_text(expected_value),
                "ignore": ignore,
            }
            return invoke("absorb", json.dumps(payload, ensure_ascii=False))

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("model", "codec", "policy", "state"):
        parser.add_argument(f"--{path}", type=Path, required=True)
    parser.add_argument("--model-35b", type=Path, help="pinned checkpoint for saving native 35B memory embeddings")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dream-every", type=int,
                        help="run replay every N operations (default 2 when updates are enabled, otherwise 0)")
    parser.add_argument("--dream-steps", type=int, default=48, help="maximum optimizer steps per replay")
    parser.add_argument("--allow-absorb", action="store_true",
                        help="enable observations and automatic policy updates for a trusted client")
    parser.add_argument("--inference-python", default=sys.executable, help="Python with the pinned Qwen runtime")
    parser.add_argument("--timeout", type=float, default=120, help="maximum seconds for a tool call")
    args = parser.parse_args()
    try:
        worker = DreamWorker(args)
    except ValueError as exc:
        parser.error(str(exc))
    create_server(worker, args.allow_absorb).run(transport="stdio")


if __name__ == "__main__":
    main()
