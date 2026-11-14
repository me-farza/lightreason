"""MCP stdio transport. Scope and write authority belong to the launcher."""

import argparse
from typing import Any

from mcp.server.fastmcp import FastMCP

from .__main__ import add_options, make_service


def create_server(service, trusted_writes=False):
    server = FastMCP("Lightreason Memory")

    @server.tool()
    def recall(question: str, token_budget: int = 256, as_of: str | None = None) -> dict[str, Any]:
        """Read scoped memory, with optional inclusive RFC 3339 event-time cutoff."""
        try:
            result = service.recall(question, token_budget, as_of)
        finally:
            release_s = service.release_gpu()
        result["timings"]["memory_release_s"] = release_s
        return result

    if trusted_writes:
        @server.tool()
        def observe(event_id: str, content: str, source: str, timestamp: str, kind: str) -> dict[str, Any]:
            """Record an observation with provenance. Use assistant_proposal or inference for guesses."""
            return service.observe(event_id, content, source, timestamp, kind)

        @server.tool()
        def correct(memory_id: str, replacement: str, source: str, timestamp: str) -> dict[str, Any]:
            """Supersede a scoped record from the replacement's event time."""
            return service.correct(memory_id, replacement, source, timestamp)

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_options(parser)
    parser.add_argument("--trusted-writes", action="store_true", help="enable writes for a trusted client")
    args = parser.parse_args()
    service = make_service(args)
    try:
        create_server(service, args.trusted_writes).run(transport="stdio")
    finally:
        service.close()


if __name__ == "__main__":
    main()
