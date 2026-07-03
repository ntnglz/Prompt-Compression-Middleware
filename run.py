#!/usr/bin/env python3
"""
Script de ejecución principal para el PCM MCP Server

Este script simplifica la ejecución del servidor.

Uso:
    python run.py                    # Inicia API REST (por defecto)
    python run.py --http             # Inicia API REST (explícito)
    python run.py --http --port 8080 # REST en puerto 8080
    python run.py --stdio            # Inicia servidor MCP (stdio)
    python run.py --mcp-http         # Inicia servidor MCP HTTP en /mcp
    python run.py --mcp-http --port 8001
    python run.py --test             # Ejecuta tests
    python run.py --check-deps       # Verificar dependencias
    python run.py --benchmark        # Benchmark sobre example_prompts.json
    python run.py --benchmark --semantic  # Benchmark con similitud semántica
"""

import sys
import subprocess
from pathlib import Path

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_http_server(port=8080):
    """Ejecuta el servidor HTTP (FastAPI)"""
    from main import run_http_server as _run_http_server
    _run_http_server(port)


def run_mcp_http_server(host=None, port=None):
    """Ejecuta el servidor MCP (streamable-http)"""
    from main import run_mcp_http_server as _run_mcp_http_server, DEFAULT_MCP_HTTP_HOST, DEFAULT_MCP_HTTP_PORT
    _run_mcp_http_server(
        host=host or DEFAULT_MCP_HTTP_HOST,
        port=port or DEFAULT_MCP_HTTP_PORT,
    )


def run_mcp_server():
    """Ejecuta el servidor MCP (stdio)"""
    from main import run_mcp_server as _run_mcp_server
    _run_mcp_server()


def run_benchmark(extra_args=None):
    """Ejecuta el benchmark de compresión"""
    cmd = [sys.executable, str(Path(__file__).parent / "scripts" / "benchmark.py")]
    if extra_args:
        cmd.extend(extra_args)
    sys.exit(subprocess.call(cmd))


def run_tests():
    """Ejecuta los tests"""
    print("=" * 60)
    print("PCM MCP Server - Ejecutando Tests")
    print("=" * 60)
    print()
    
    # Verificar que pytest está instalado
    try:
        import pytest
    except ImportError:
        print("pytest no está instalado. Instalando...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest", "pytest-asyncio", "httpx", "--break-system-packages"])
    
    # Ejecutar tests
    exit_code = pytest.main(["-v", "tests/"])
    sys.exit(exit_code)


def check_dependencies():
    """Verifica que las dependencias están instaladas"""
    print("Verificando dependencias...")
    
    required = [
        ("mcp", "Model Context Protocol SDK"),
        ("ollama", "Cliente Ollama para modelos locales"),
        ("fastapi", "Framework HTTP"),
        ("uvicorn", "Servidor ASGI"),
        ("tiktoken", "Tokenizador para conteo de tokens"),
        ("numpy", "Librería numérica para cálculos"),
    ]
    missing = []
    
    for package, description in required:
        try:
            __import__(package)
            print(f"  ✓ {package:15} ({description})")
        except ImportError:
            print(f"  ✗ {package:15} ({description}) - FALTANTE")
            missing.append(package)
    
    if missing:
        print()
        print("Instalando dependencias faltantes...")
        for package in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--break-system-packages", "--quiet"])
        print("✅ Dependencias instaladas!")
    else:
        print()
        print("✅ Todas las dependencias están instaladas!")
    
    # Verificar Ollama
    print()
    print("Verificando Ollama...")
    try:
        import ollama
        models = ollama.list()
        default_model = "granite4.1:3b"
        model_available = any(default_model in str(m) for m in models.get("models", []))
        if model_available:
            print(f"  ✓ Ollama está instalado y {default_model} está disponible")
        else:
            print(f"  ✗ {default_model} no está descargado")
            print()
            print(f"Descargando {default_model}...")
            subprocess.check_call(["ollama", "pull", default_model])
            print(f"✅ {default_model} descargado!")
    except Exception as e:
        print(f"  ✗ Error con Ollama: {e}")


def main():
    """Punto de entrada principal"""
    import argparse
    from main import DEFAULT_MODEL, DEFAULT_PORT, DEFAULT_MCP_HTTP_HOST, DEFAULT_MCP_HTTP_PORT
    
    parser = argparse.ArgumentParser(
        description="PCM MCP Server - Script de Ejecución",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python run.py                    # API REST (por defecto)
  python run.py --http             # API REST en puerto 8080
  python run.py --stdio            # MCP stdio
  python run.py --mcp-http         # MCP HTTP en http://localhost:8001/mcp
  python run.py --test             # Ejecutar tests
  python run.py --benchmark        # Benchmark de compresión
  python run.py --check-deps       # Verificar dependencias
        """
    )
    
    # Argumentos
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--http", action="store_true", help="Iniciar API REST (FastAPI)")
    group.add_argument("--stdio", action="store_true", help="Iniciar servidor MCP (stdio)")
    group.add_argument("--mcp-http", action="store_true", help="Iniciar servidor MCP (streamable-http en /mcp)")
    group.add_argument("--test", action="store_true", help="Ejecutar tests")
    group.add_argument("--benchmark", action="store_true", help="Ejecutar benchmark de compresión")
    group.add_argument("--check-deps", action="store_true", help="Verificar dependencias")
    
    parser.add_argument("--port", type=int, default=None, help="Puerto (8080 REST, 8001 MCP HTTP por defecto)")
    parser.add_argument("--host", type=str, default=None, help="Host para MCP HTTP (por defecto: 0.0.0.0)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Modelo Ollama a usar")
    
    args = parser.parse_args()

    from main import compressor
    compressor.config.model = args.model
    
    if args.check_deps:
        check_dependencies()
        return
    
    if args.test:
        run_tests()
        return

    if args.benchmark:
        benchmark_args = []
        if args.model != DEFAULT_MODEL:
            benchmark_args.extend(["--model", args.model])
        run_benchmark(benchmark_args)
        return
    
    if args.stdio:
        run_mcp_server()
        return

    if args.mcp_http:
        run_mcp_http_server(
            host=args.host or DEFAULT_MCP_HTTP_HOST,
            port=args.port or DEFAULT_MCP_HTTP_PORT,
        )
        return
    
    # Por defecto o --http: API REST
    run_http_server(args.port or DEFAULT_PORT)


if __name__ == "__main__":
    main()
