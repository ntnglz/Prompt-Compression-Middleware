"""
Tests del servidor MCP PCM.
"""

import pytest

from pcm.compressor import PromptCompressor, CompressorConfig
from pcm.mcp_server import MCP_HTTP_PATH, SERVER_NAME, create_mcp_server
from pcm.tools import PCMServerTools

EXPECTED_TOOLS = {
    "compress_prompt",
    "batch_compress",
    "compare_prompts",
    "estimate_tokens",
    "get_compression_stats",
    "health_check",
}


@pytest.fixture
def mcp_server():
    compressor = PromptCompressor(
        config=CompressorConfig(model="granite4.1:3b", temperature=0.1, timeout=30)
    )
    tools = PCMServerTools(compressor)
    return create_mcp_server(tools)


@pytest.mark.asyncio
async def test_list_tools_registers_all_tools(mcp_server):
    tools = await mcp_server.list_tools()
    names = {tool.name for tool in tools}

    assert names == EXPECTED_TOOLS


@pytest.mark.asyncio
async def test_tool_schemas_have_required_fields(mcp_server):
    tools = await mcp_server.list_tools()
    by_name = {tool.name: tool for tool in tools}

    compress = by_name["compress_prompt"]
    assert compress.description
    assert "prompt" in compress.inputSchema.get("properties", {})
    assert "prompt" in compress.inputSchema.get("required", [])

    batch = by_name["batch_compress"]
    assert "prompts" in batch.inputSchema.get("properties", {})

    health = by_name["health_check"]
    assert health.inputSchema.get("properties", {}) == {}


@pytest.mark.asyncio
async def test_estimate_tokens_tool_call(mcp_server):
    result = await mcp_server.call_tool("estimate_tokens", {"text": "Hola mundo"})

    if isinstance(result, tuple):
        result = result[1]

    assert result["text"] == "Hola mundo"
    assert result["tokens"] > 0


@pytest.mark.asyncio
async def test_unknown_tool_raises(mcp_server):
    with pytest.raises(Exception):
        await mcp_server.call_tool("nonexistent_tool", {})


def test_server_metadata(mcp_server):
    assert mcp_server.name == SERVER_NAME
    assert mcp_server.instructions


def test_streamable_http_app_exposes_mcp_endpoint(mcp_server):
    app = mcp_server.streamable_http_app()
    paths = {getattr(route, "path", None) for route in app.routes}

    assert MCP_HTTP_PATH in paths
