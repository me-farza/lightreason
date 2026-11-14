"""MCP integration tests with a synthetic inference process. No model is loaded."""

import argparse
from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from .dream_mcp import DreamWorker, ROOT


def _args(allow_absorb=False, dream_every=None, model_35b=None):
    return argparse.Namespace(
        model=Path("model"), codec=Path("codec"), policy=Path("policy"), state=Path("state"),
        device="cpu", inference_python=sys.executable, timeout=10, dream_every=dream_every,
        dream_steps=48, allow_absorb=allow_absorb, model_35b=model_35b,
    )


class DreamWorkerTests(unittest.TestCase):
    def test_dreams_cannot_run_when_observations_are_disabled(self):
        worker = DreamWorker(_args())
        self.assertEqual(worker.command[worker.command.index("--dream-every") + 1], "0")
        with self.assertRaisesRegex(ValueError, "do not allow|does not allow"):
            worker("absorb", "")
        with self.assertRaisesRegex(ValueError, "positive dream cadence"):
            DreamWorker(_args(dream_every=2))

    def test_configured_destination_is_forwarded_before_the_operation(self):
        destination = Path("pinned-35b")
        worker = DreamWorker(_args(allow_absorb=True, model_35b=destination))
        with patch("subprocess.run", return_value=subprocess.CompletedProcess(
            worker.command, 0, stdout='{"status":"unknown"}'
        )) as run:
            worker("recall", "Synthetic question")
        command = run.call_args.args[0]
        flag = command.index("--model-35b")
        self.assertEqual(command[flag + 1], str(destination.absolute()))
        self.assertLess(flag, command.index("recall"))

    def test_private_payload_stays_on_stdin_and_out_of_failures(self):
        worker = DreamWorker(_args(allow_absorb=True))
        secret = "This synthetic private experience should stay off argv."
        feedback = json.dumps({"text": secret, "expected_key": "key", "expected_value": "value", "ignore": False})
        with patch("subprocess.run", return_value=subprocess.CompletedProcess(
            worker.command, 0, stdout='{"updated":true}'
        )) as run:
            self.assertEqual(worker("absorb", feedback), {"updated": True})
        self.assertFalse(any(secret in element for element in run.call_args.args[0]))
        self.assertEqual(run.call_args.args[0][-1], "--feedback-stdin")
        self.assertEqual(run.call_args.kwargs["input"], feedback)
        self.assertEqual(run.call_args.kwargs["stderr"], subprocess.DEVNULL)
        self.assertEqual(run.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, [secret], stderr=secret)):
            with self.assertRaises(RuntimeError) as error:
                worker("absorb", feedback)
        self.assertNotIn(secret, str(error.exception))


@unittest.skipUnless(find_spec("mcp"), "install requirements-mcp.txt to exercise stdio")
class DreamMCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_stdio_observe_recall_and_non_advancing_status(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        with TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "synthetic-python"
            executable.write_text(
                f"#!{sys.executable}\n"
                "import json, sys\n"
                "from pathlib import Path\n"
                "args = sys.argv\n"
                "path = Path(args[args.index('--state') + 1])\n"
                "cadence = int(args[args.index('--dream-every') + 1])\n"
                "name = args[args.index('--dream-steps') + 2]\n"
                "input_text = sys.stdin.read()\n"
                "state = json.loads(path.read_text()) if path.exists() else {'count': 0, 'value': ''}\n"
                "if name != 'dream-status' and cadence:\n"
                "    state['count'] += 1\n"
                "    if name == 'absorb':\n"
                "        assert args[-1] == '--feedback-stdin'\n"
                "        feedback = json.loads(input_text)\n"
                "        assert feedback['text'] not in args\n"
                "        state['value'] = feedback['expected_value']\n"
                "    path.write_text(json.dumps(state))\n"
                "print(json.dumps({'count': state['count'], 'value': state['value'], 'cadence': cadence}))\n",
                encoding="utf-8",
            )
            executable.chmod(0o700)
            base = [
                "-m", "lightreason_memory.dream_mcp", "--model", "unused", "--codec", "unused",
                "--policy", "unused", "--state", str(root / "private-state"),
                "--device", "cpu", "--inference-python", str(executable),
            ]
            async with stdio_client(StdioServerParameters(command=sys.executable, args=base, cwd=ROOT)) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.assertEqual({tool.name for tool in (await session.list_tools()).tools}, {"recall", "dream_status"})
                    result = await session.call_tool("recall", {"question": "What is my dog's name?"})
                    self.assertEqual(result.structuredContent, {"cadence": 0, "count": 0, "value": ""})
            trusted = StdioServerParameters(command=sys.executable, args=[*base, "--allow-absorb"], cwd=ROOT)
            async with stdio_client(trusted) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.assertEqual({tool.name for tool in (await session.list_tools()).tools},
                                     {"recall", "dream_status", "observe"})
                    invalid = await session.call_tool("observe", {"experience": "Example only", "expected_key": "dog"})
                    self.assertTrue(invalid.isError)
                    observation = await session.call_tool("observe", {
                        "experience": "My dog is named Canopy.", "expected_key": "dog", "expected_value": "Canopy",
                    })
                    self.assertEqual(observation.structuredContent, {"cadence": 2, "count": 1, "value": "Canopy"})
                    status = await session.call_tool("dream_status", {})
                    self.assertEqual(status.structuredContent, observation.structuredContent)
                    recall = await session.call_tool("recall", {"question": "What is my dog's name?"})
                    self.assertEqual(recall.structuredContent, {"cadence": 2, "count": 2, "value": "Canopy"})


if __name__ == "__main__":
    unittest.main()
