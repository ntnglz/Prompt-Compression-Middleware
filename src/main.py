#!/usr/bin/env python3
"""
Prompt Compression Middleware (PCM) - Servidor MCP y HTTP

Este servidor expone herramientas para comprimir prompts de manera semántica,
optimizando la comunicación con LLM.

Uso:
    python src/main.py          # Inicia el servidor MCP en stdio
    python src/main.py --http   # Inicia el servidor HTTP (FastAPI)

Conectar con cliente MCP (Cursor, Claude Desktop, etc.):
    python run.py --stdio
    # o: python src/main.py --stdio
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv

# Cargar variables de entorno desde .env (si existe)
load_dotenv(Path(__file__).parent.parent / ".env")

# Configuración de logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuración por defecto (sobrescribible vía .env o CLI)
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "granite4.1:3b")
DEFAULT_PORT = int(os.getenv("HTTP_PORT", "8080"))
DEFAULT_MCP_HTTP_HOST = os.getenv("MCP_HTTP_HOST", "0.0.0.0")
DEFAULT_MCP_HTTP_PORT = int(os.getenv("MCP_HTTP_PORT", "8001"))
DEFAULT_TEMPERATURE = float(os.getenv("COMPRESSOR_TEMPERATURE", "0.1"))
DEFAULT_TIMEOUT = int(os.getenv("COMPRESSOR_TIMEOUT", "120"))

# Importar después de configurar logging
try:
    from mcp.server.fastmcp import FastMCP  # noqa: F401
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP no disponible. El modo stdio no funcionará.")

from pcm.compressor import PromptCompressor, CompressorConfig
from pcm.tools import PCMServerTools


# ============================================================================
# CONFIGURACIÓN DEL COMPRESOR Y HERRAMIENTAS
# ============================================================================

compressor = PromptCompressor(
    config=CompressorConfig(
        model=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        timeout=DEFAULT_TIMEOUT
    )
)
tools = PCMServerTools(compressor)


# ============================================================================
# SERVIDOR MCP (modo stdio)
# ============================================================================

def run_mcp_server():
    """Ejecuta el servidor MCP en modo stdio"""
    if not MCP_AVAILABLE:
        logger.error("MCP no está disponible. Instala con: pip install mcp")
        sys.exit(1)

    from pcm.mcp_server import run_mcp_server as _run_mcp_stdio

    _run_mcp_stdio(tools)


def run_mcp_http_server(
    host: str = DEFAULT_MCP_HTTP_HOST,
    port: int = DEFAULT_MCP_HTTP_PORT,
):
    """Ejecuta el servidor MCP con transporte streamable-http en /mcp"""
    if not MCP_AVAILABLE:
        logger.error("MCP no está disponible. Instala con: pip install mcp")
        sys.exit(1)

    from pcm.mcp_server import run_mcp_http_server as _run_mcp_http

    _run_mcp_http(tools, host=host, port=port)


# ============================================================================
# SERVIDOR HTTP (FastAPI)
# ============================================================================

def run_http_server(port: int = 8080):
    """Ejecuta el servidor HTTP con FastAPI"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    
    fastapi_app = FastAPI(
        title="PCM MCP Server",
        description="Prompt Compression Middleware - API HTTP",
        version="0.1.0"
    )
    
    # Configurar CORS
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Endpoints API
    from fastapi import Body, Query
    from pydantic import BaseModel
    
    # Modelos para request bodies
    class CompressRequest(BaseModel):
        prompt: str
        strategy: Optional[str] = None
        target_model: Optional[str] = None
    
    class BatchCompressRequest(BaseModel):
        prompts: List[str]
        strategy: Optional[str] = None
    
    class CompareRequest(BaseModel):
        original: str
        compressed: Optional[str] = None
    
    class EstimateTokensRequest(BaseModel):
        text: str
    
    class StatsRequest(BaseModel):
        prompts: List[str]
    
    @fastapi_app.post("/compress")
    async def api_compress(request: CompressRequest):
        return await tools.compress_prompt(
            request.prompt,
            request.strategy,
            request.target_model,
        )
    
    @fastapi_app.post("/batch-compress")
    async def api_batch_compress(request: BatchCompressRequest):
        return await tools.batch_compress(request.prompts, request.strategy)
    
    @fastapi_app.post("/compare")
    async def api_compare(request: CompareRequest):
        return await tools.compare_prompts(request.original, request.compressed)
    
    @fastapi_app.post("/estimate-tokens")
    async def api_estimate_tokens(request: EstimateTokensRequest):
        return await tools.estimate_tokens(request.text)
    
    @fastapi_app.post("/stats")
    async def api_stats(request: StatsRequest):
        return await tools.get_compression_stats(request.prompts)
    
    @fastapi_app.get("/health")
    async def api_health():
        return await tools.health_check()
    
    @fastapi_app.get("/")
    async def api_root():
        return {
            "name": "PCM MCP Server",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
            "endpoints": [
                {"path": "/compress", "method": "POST", "description": "Comprimir un prompt"},
                {"path": "/batch-compress", "method": "POST", "description": "Comprimir múltiples prompts"},
                {"path": "/compare", "method": "POST", "description": "Comparar prompts"},
                {"path": "/estimate-tokens", "method": "POST", "description": "Contar tokens"},
                {"path": "/stats", "method": "POST", "description": "Estadísticas de compresión"},
                {"path": "/health", "method": "GET", "description": "Estado del servidor"}
            ]
        }
    
    # Iniciar servidor FastAPI
    logger.info(f"Iniciando servidor HTTP en http://localhost:{port}")
    logger.info(f"Documentación API: http://localhost:{port}/docs")
    logger.info(f"Health check: http://localhost:{port}/health")
    
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================

def main():
    """Punto de entrada principal"""
    global DEFAULT_MODEL

    parser = argparse.ArgumentParser(
        description="PCM MCP Server - Prompt Compression Middleware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python src/main.py --http             # API REST (FastAPI)
  python src/main.py --stdio            # MCP por stdio
  python src/main.py --mcp-http         # MCP por HTTP en /mcp
  python src/main.py --mcp-http --port 8001
        """
    )
    
    # Argumentos
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--http", action="store_true", help="Iniciar API REST (FastAPI)")
    group.add_argument("--stdio", action="store_true", help="Iniciar servidor MCP (stdio)")
    group.add_argument("--mcp-http", action="store_true", help="Iniciar servidor MCP (streamable-http en /mcp)")
    
    parser.add_argument("--port", type=int, default=None, help="Puerto (8080 REST, 8001 MCP HTTP por defecto)")
    parser.add_argument("--host", type=str, default=None, help="Host para MCP HTTP (por defecto: 0.0.0.0)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Modelo Ollama a usar")
    
    args = parser.parse_args()
    
    DEFAULT_MODEL = args.model
    compressor.config.model = args.model
    logger.info(f"Usando modelo: {args.model}")
    
    if args.http:
        run_http_server(args.port or DEFAULT_PORT)
    elif args.stdio:
        run_mcp_server()
    elif args.mcp_http:
        run_mcp_http_server(
            host=args.host or DEFAULT_MCP_HTTP_HOST,
            port=args.port or DEFAULT_MCP_HTTP_PORT,
        )
    else:
        run_http_server(args.port or DEFAULT_PORT)


if __name__ == "__main__":
    main()
