"""
Servidor MCP para PCM usando FastMCP.
"""

import logging
from typing import Any, List, Optional

from mcp.server.fastmcp import FastMCP

from .compressor import PromptCompressor
from .tools import PCMServerTools

logger = logging.getLogger(__name__)

SERVER_NAME = "pcm-mcp-server"
SERVER_VERSION = "0.1.0"
MCP_HTTP_PATH = "/mcp"
SERVER_INSTRUCTIONS = (
    "Prompt Compression Middleware: comprime prompts en lenguaje natural "
    "en representaciones compactas optimizadas para LLM, preservando la intención semántica."
)


def create_mcp_server(
    tools: PCMServerTools,
    *,
    host: str = "0.0.0.0",
    port: int = 8001,
) -> FastMCP:
    """Crea y registra las herramientas MCP sobre una instancia FastMCP."""
    mcp = FastMCP(
        SERVER_NAME,
        instructions=SERVER_INSTRUCTIONS,
        host=host,
        port=port,
        streamable_http_path=MCP_HTTP_PATH,
    )

    @mcp.tool()
    async def compress_prompt(
        prompt: str,
        strategy: Optional[str] = None,
        target_model: Optional[str] = None,
    ) -> dict[str, Any]:
        """Comprime un prompt preservando su intención semántica."""
        return await tools.compress_prompt(prompt, strategy, target_model)

    @mcp.tool()
    async def batch_compress(
        prompts: List[str],
        strategy: Optional[str] = None,
    ) -> dict[str, Any]:
        """Comprime múltiples prompts en batch."""
        return await tools.batch_compress(prompts, strategy)

    @mcp.tool()
    async def compare_prompts(
        original: str,
        compressed: Optional[str] = None,
    ) -> dict[str, Any]:
        """Compara un prompt original con su versión comprimida."""
        return await tools.compare_prompts(original, compressed)

    @mcp.tool()
    async def estimate_tokens(text: str) -> dict[str, Any]:
        """Estima el número de tokens de un texto."""
        return await tools.estimate_tokens(text)

    @mcp.tool()
    async def get_compression_stats(prompts: List[str]) -> dict[str, Any]:
        """Obtiene estadísticas de compresión para una lista de prompts."""
        return await tools.get_compression_stats(prompts)

    @mcp.tool()
    async def health_check() -> dict[str, Any]:
        """Verifica el estado del servidor y la disponibilidad del modelo."""
        return await tools.health_check()

    return mcp


def _public_url(host: str, port: int, path: str = MCP_HTTP_PATH) -> str:
    display_host = "localhost" if host in ("0.0.0.0", "::") else host
    return f"http://{display_host}:{port}{path}"


def run_mcp_server(tools: PCMServerTools) -> None:
    """Ejecuta el servidor MCP en modo stdio."""
    mcp = create_mcp_server(tools)
    logger.info("Iniciando servidor MCP (stdio)...")
    logger.info("Conecta con: python run.py --stdio")
    mcp.run(transport="stdio")


def run_mcp_http_server(
    tools: PCMServerTools,
    host: str = "0.0.0.0",
    port: int = 8001,
) -> None:
    """Ejecuta el servidor MCP con transporte streamable-http."""
    mcp = create_mcp_server(tools, host=host, port=port)
    url = _public_url(host, port)

    logger.info("Iniciando servidor MCP (streamable-http)...")
    logger.info("Endpoint MCP: %s", url)
    logger.info("Configura Vibe con transport=streamable-http y url=%s", url)
    mcp.run(transport="streamable-http")
