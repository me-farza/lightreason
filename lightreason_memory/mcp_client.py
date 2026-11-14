"""Example client that starts the stdio server and calls its recall tool."""

import argparse
import asyncio
from datetime import timedelta
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .__main__ import ROOT, _read_stdin, add_options
from .demo import DEMO_QUESTION
from .service import MAX_QUESTION


async def run(args):
    command = [
        "-m", "lightreason_memory.mcp_server", "--db", str(args.db.resolve()),
        "--user", args.user, "--workspace", args.workspace, "--model-path",
        str(args.model_path.resolve()), "--runtime-path", str(args.runtime_path.resolve()),
        "--timeout", str(args.timeout),
    ]
    if args.inference_python:
        command.extend(("--inference-python", str(args.inference_python.absolute())))
    env = {key: os.environ[key] for key in ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES") if key in os.environ}
    params = StdioServerParameters(command=sys.executable, args=command, cwd=ROOT, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=3 * args.timeout)) as session:
            await session.initialize()
            listed = await session.list_tools()
            result = await session.call_tool("recall", {
                "question": args.question, "token_budget": args.token_budget, "as_of": args.as_of,
            })
            if result.isError:
                raise RuntimeError("MCP recall call failed")
            payload = result.structuredContent
            if payload is None and len(result.content) == 1 and result.content[0].type == "text":
                payload = json.loads(result.content[0].text)
            if not isinstance(payload, dict) or "readout_mode" not in payload:
                raise RuntimeError("MCP recall returned no valid response object")
            print(json.dumps({
                "tools": [tool.name for tool in listed.tools],
                "result": payload,
            }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_options(parser)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--stdin", action="store_true")
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--as-of")
    args = parser.parse_args()
    if args.stdin and args.question is not None:
        parser.error("use a question argument or --stdin")
    try:
        args.question = _read_stdin(MAX_QUESTION) if args.stdin else args.question or DEMO_QUESTION
    except ValueError as exc:
        parser.error(str(exc))
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
