"""Optional integration test using the real MCP client and stdio subprocess."""

from importlib.util import find_spec
from pathlib import Path
import sys
import tempfile
import unittest


@unittest.skipUnless(find_spec("mcp"), "install requirements-mcp.txt to exercise stdio")
class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_configured_scope_and_trusted_write_tools(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "memory.sqlite3"
            command = [
                "-m", "lightreason_memory.mcp_server", "--db", str(db), "--user", "synthetic",
                "--workspace", "fixture", "--model-path", str(db.parent / "missing-checkpoint"),
            ]
            trusted = StdioServerParameters(command=sys.executable, args=[*command, "--trusted-writes"], cwd=root)
            async with stdio_client(trusted) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.assertEqual({item.name for item in (await session.list_tools()).tools},
                                     {"recall", "observe", "correct"})
                    payload = {
                        "event_id": "fixture-1", "content": "The synthetic flag is red.", "source": "fixture",
                        "timestamp": "2026-11-12T00:00:00Z", "kind": "tool_observation",
                    }
                    first = await session.call_tool("observe", payload)
                    duplicate = await session.call_tool("observe", payload)
                    self.assertFalse(first.isError)
                    self.assertEqual(first.structuredContent, duplicate.structuredContent)
                    old_id = first.structuredContent["memory_id"]
                    corrected = await session.call_tool("correct", {
                        "memory_id": old_id, "replacement": "The synthetic flag is blue.",
                        "source": "fixture", "timestamp": "2026-11-13T00:00:00Z",
                    })
                    self.assertFalse(corrected.isError)
                    self.assertEqual(corrected.structuredContent["supersedes_id"], old_id)

            read_only = StdioServerParameters(command=sys.executable, args=command, cwd=root)
            async with stdio_client(read_only) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.assertEqual([item.name for item in (await session.list_tools()).tools], ["recall"])
                    schema = (await session.list_tools()).tools[0].inputSchema
                    self.assertNotIn("user", schema["properties"])
                    self.assertNotIn("workspace", schema["properties"])
                    result = await session.call_tool("recall", {"question": "What is the synthetic flag color?"})
                    self.assertFalse(result.isError)
                    self.assertEqual(result.structuredContent["readout_mode"], "extractive_fallback")
                    self.assertEqual(result.structuredContent["fallback_excerpts"][0]["excerpt"],
                                     "The synthetic flag is blue.")
