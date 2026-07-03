#!/usr/bin/env python3
"""Smoke test del servidor MCP PCM (FastMCP)."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pcm.compressor import PromptCompressor
from pcm.mcp_server import create_mcp_server
from pcm.tools import PCMServerTools


async def main() -> None:
    tools = PCMServerTools(PromptCompressor())
    server = create_mcp_server(tools)

    registered = await server.list_tools()
    print(f"Herramientas registradas ({len(registered)}):")
    for tool in registered:
        print(f"  - {tool.name}: {tool.description[:60]}...")

    result = await server.call_tool("estimate_tokens", {"text": "Hola mundo"})
    if isinstance(result, tuple):
        result = result[1]
    print(f"\nestimate_tokens('Hola mundo') -> {result}")


if __name__ == "__main__":
    asyncio.run(main())
