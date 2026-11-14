"""Expose the trained neural-memory backend to an agent over MCP stdio."""

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

from mcp.server.fastmcp import FastMCP

from .neural_cli import add_options, validate_text


ROOT = Path(__file__).resolve().parent.parent


class NeuralWorker:
    """Own one bounded inference child per call and release it before responding."""

    def __init__(self, args):
        if not math.isfinite(args.timeout) or not 0 < args.timeout <= 900:
            raise ValueError("Timeout must be greater than zero and at most 900 seconds.")
        self.timeout = args.timeout
        self.lock = Lock()
        self.command = [
            os.path.abspath(os.path.expanduser(args.inference_python)),
            "-m", "lightreason_memory.neural_cli",
            "--parent-pid", str(os.getpid()),
            "--device", args.device,
            "--model-path", str(args.model_path.expanduser().absolute()),
            "--checkpoint-path", str(args.checkpoint_path.expanduser().absolute()),
            "--state-path", str(args.state_path.expanduser().absolute()),
        ]

    def __call__(self, action, value):
        deadline = monotonic() + self.timeout
        if not self.lock.acquire(timeout=self.timeout):
            raise RuntimeError("The neural memory worker is busy.")
        try:
            return self._call(action, value, deadline)
        finally:
            self.lock.release()

    def _call(self, action, value, deadline):
        timeout = deadline - monotonic()
        if timeout <= 0:
            raise RuntimeError("The neural memory worker is busy.")
        try:
            completed = subprocess.run(
                [*self.command, action], input=value, encoding="utf-8", check=True, cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout,
                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                     "HF_HUB_DISABLE_TELEMETRY": "1"},
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("The neural memory worker timed out.") from None
        except (OSError, subprocess.CalledProcessError):
            raise RuntimeError("The neural memory worker failed. Check the configured files and device.") from None
        try:
            result = json.loads(completed.stdout)
        except ValueError:
            raise RuntimeError("The neural memory worker returned invalid JSON.") from None
        if not isinstance(result, dict):
            raise RuntimeError("The neural memory worker returned an invalid result.")
        return result


def create_server(invoke, allow_absorb=False):
    server = FastMCP("Lightreason Neural Memory")

    @server.tool()
    def recall(question: str) -> dict[str, Any]:
        """Ask a question of the configured personal neural state."""
        return invoke("recall", validate_text(question))

    if allow_absorb:
        @server.tool()
        def absorb(experience: str) -> dict[str, Any]:
            """Update the personal neural state with one approved experience or correction."""
            return invoke("absorb", validate_text(experience))

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_options(parser)
    parser.add_argument("--allow-absorb", action="store_true", help="enable state updates for a trusted client")
    parser.add_argument("--inference-python", default=sys.executable, help="Python with the pinned Qwen runtime")
    parser.add_argument("--timeout", type=float, default=120, help="maximum seconds for a tool call")
    args = parser.parse_args()
    try:
        worker = NeuralWorker(args)
    except ValueError as exc:
        parser.error(str(exc))
    create_server(worker, args.allow_absorb).run(transport="stdio")


if __name__ == "__main__":
    main()
