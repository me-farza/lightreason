"""Smoke-test the transport with synthetic workers. No CUDA or checkpoints required."""

import argparse
from contextlib import redirect_stdout
from importlib.util import find_spec
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from . import neural_cli


class NeuralCLITests(unittest.TestCase):
    def test_cli_uses_stdin_and_closes_model(self):
        backend = ModuleType("lightreason_memory.neural")
        backend.NeuralMemory = Mock()
        model = backend.NeuralMemory.return_value
        model.absorb.return_value = {"status": "updated"}
        with patch.dict(sys.modules, {"lightreason_memory.neural": backend}), patch(
            "sys.stdin", io.StringIO("I adopted a dog named Canopy.")
        ), patch.dict(os.environ), redirect_stdout(io.StringIO()) as output:
            neural_cli.main([
                "--model-path", "model", "--checkpoint-path", "trained",
                "--state-path", "personal", "--device", "cpu", "absorb",
            ])
        backend.NeuralMemory.assert_called_once_with(
            Path("model"), Path("trained"), Path("personal"), device="cpu",
        )
        model.absorb.assert_called_once_with("I adopted a dog named Canopy.")
        model.close.assert_called_once()
        self.assertEqual(json.loads(output.getvalue()), {"status": "updated"})

    def test_empty_and_overlong_input_fails_before_loading_model(self):
        base_args = [
            "--model-path", "model", "--checkpoint-path", "trained", "--state-path", "personal", "recall",
        ]
        for bad_text in (" \n", "x" * (neural_cli.MAX_TEXT + 1), "\0"):
            with self.subTest(length=len(bad_text)), patch("sys.stdin", io.StringIO(bad_text)), patch(
                "sys.stderr", io.StringIO()
            ), patch.object(neural_cli, "run_operation") as run, self.assertRaises(SystemExit):
                neural_cli.main(base_args)
            run.assert_not_called()


@unittest.skipUnless(find_spec("mcp"), "install requirements-mcp.txt to exercise stdio")
class NeuralMCPTests(unittest.IsolatedAsyncioTestCase):
    def test_worker_keeps_text_out_of_argv_and_handles_worker_timeout(self):
        from .neural_mcp import NeuralWorker

        worker = NeuralWorker(argparse.Namespace(
            model_path=Path("model"), checkpoint_path=Path("trained"), state_path=Path("one-user"),
            device="cpu", inference_python=sys.executable, timeout=10,
        ))
        question = "Where did I meet Élodie?"
        with patch("subprocess.run", return_value=subprocess.CompletedProcess(
            worker.command, 0, stdout='{"status":"unknown","answer":""}'
        )) as run:
            self.assertEqual(worker("recall", question)["status"], "unknown")
        self.assertNotIn(question, run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["input"], question)
        self.assertEqual(run.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(run.call_args.args[0][-1], "recall")
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(worker.command, 10)):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                worker("recall", question)

    async def test_mcp_default_is_read_only_and_trusted_client_can_absorb(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "owner-state"
            # Stand in for the inference interpreter only. FastMCP and its
            # client both run normally over a real stdio connection.
            worker_path = state_path.parent / "synthetic-python"
            worker_path.write_text(
                f"#!{sys.executable}\n"
                "import json, sys\n"
                "from pathlib import Path\n"
                "state_path = Path(sys.argv[sys.argv.index('--state-path') + 1])\n"
                "value = sys.stdin.read()\n"
                "if sys.argv[-1] == 'absorb':\n"
                "    state_path.write_text('enabled')\n"
                "    print(json.dumps({'status': 'updated'}))\n"
                "else:\n"
                "    print(json.dumps({'status': 'supported' if state_path.is_file() else 'unknown', "
                "'answer': 'Canopy' if state_path.is_file() else ''}))\n",
                encoding="utf-8",
            )
            worker_path.chmod(0o700)
            command = [
                "-m", "lightreason_memory.neural_mcp", "--model-path", "unused",
                "--checkpoint-path", "unused", "--state-path", str(state_path),
                "--device", "cpu", "--inference-python", str(worker_path),
            ]
            readonly = StdioServerParameters(command=sys.executable, args=command, cwd=root)
            async with stdio_client(readonly) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = (await session.list_tools()).tools
                    self.assertEqual([tool.name for tool in tools], ["recall"])
                    self.assertEqual(set(tools[0].inputSchema["properties"]), {"question"})
                    result = await session.call_tool("recall", {"question": "What is my dog's name?"})
                    self.assertEqual(result.structuredContent, {"status": "unknown", "answer": ""})
            trusted = StdioServerParameters(
                command=sys.executable, args=[*command, "--allow-absorb"], cwd=root,
            )
            async with stdio_client(trusted) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.assertEqual({tool.name for tool in (await session.list_tools()).tools}, {"recall", "absorb"})
                    absorbed = await session.call_tool("absorb", {"experience": "My dog is named Canopy."})
                    self.assertEqual(absorbed.structuredContent, {"status": "updated"})
                    result = await session.call_tool("recall", {"question": "What is my dog's name?"})
                    self.assertEqual(result.structuredContent, {"status": "supported", "answer": "Canopy"})
