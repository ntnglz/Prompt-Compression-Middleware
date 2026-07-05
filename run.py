#!/usr/bin/env python3
"""
PCM entry point — visitor-friendly demos and maintainer tooling.

Visitor:
    python run.py --demo              # canonical compression (Ollama or stub fallback)
    python run.py --demo-stub         # deterministic demo without Ollama
    python run.py --quickstart        # demo + copy-paste OpenAI SDK snippet
    python run.py --proxy             # OpenAI-compatible proxy on :8090

Maintainer:
    python run.py --help-all          # benchmark, validate, CI, all flags
    python run.py --ci                # fast local CI (no Ollama)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

VISITOR_EPILOG = """
Try it now:
  python run.py --demo-stub           # no Ollama required
  python run.py --quickstart          # demo + integration snippet
  python run.py --proxy               # http://localhost:8090/v1/chat/completions

Docs: README.md · docs/getting-started.md · docs/FAQ.md
"""

MAINTAINER_EPILOG = """
Maintainer commands (also in visitor --help):
  python run.py --test-fast           # pytest without Ollama integration tests
  python run.py --test                # full pytest (needs Ollama)
  python run.py --ci                  # same as scripts/ci-local.sh
  python run.py --benchmark           # compressor benchmark
  python run.py --check-deps          # verify Ollama + Python deps
  python scripts/validate_granite_v2.py --semantic --e2e
"""


def _ollama_available() -> bool:
    try:
        import ollama

        ollama.list()
        return True
    except Exception:
        return False


def run_demo(*, stub: bool = False) -> int:
    from pcm.canonical import (
        CANONICAL_COMPRESSED_MESSAGE,
        CANONICAL_INSTRUCTION,
        CANONICAL_PCM,
        CANONICAL_PAYLOAD,
        CANONICAL_USER_MESSAGE,
        COE_REPO,
        CURSOR_DEV_INSTRUCTION,
        CURSOR_DEV_PCM,
        CURSOR_DEV_PAYLOAD,
        canonical_metrics,
        cursor_dev_metrics,
        format_token_metrics,
    )
    from pcm.prompt_utils import join_instruction_and_payload

    mode = "stub"
    compressed = CANONICAL_PCM

    if not stub and _ollama_available():
        try:
            from pcm.compressor import PromptCompressor

            result = PromptCompressor().compress(CANONICAL_INSTRUCTION)
            compressed = result.compressed_prompt.strip()
            if compressed.startswith("TASK="):
                mode = "ollama"
            else:
                mode = "stub (unexpected model output)"
                compressed = CANONICAL_PCM
        except Exception as exc:
            mode = f"stub (Ollama error: {exc})"
            compressed = CANONICAL_PCM
    elif not stub:
        mode = "stub (Ollama not available)"

    print("=" * 60)
    print("PCM — canonical demo")
    print("=" * 60)
    print(f"Mode: {mode}")
    print()
    print("BEFORE (natural-language instruction):")
    print(CANONICAL_INSTRUCTION)
    print()
    print("AFTER (PCM instruction):")
    print(compressed)
    print()
    metrics = canonical_metrics()
    if compressed != CANONICAL_PCM:
        from pcm.canonical import token_metrics

        metrics = token_metrics(CANONICAL_INSTRUCTION, compressed, CANONICAL_PAYLOAD)
    print(format_token_metrics(metrics, label="Token savings (tiktoken gpt-4 estimate)"))
    print()
    print("Payload (unchanged, excerpt):")
    print(CANONICAL_PAYLOAD.splitlines()[0])
    print("...")
    print()
    print("Full proxy message BEFORE:")
    print(CANONICAL_USER_MESSAGE[:120] + "…")
    print()
    print("Full proxy message AFTER:")
    preview = join_instruction_and_payload(compressed, CANONICAL_PAYLOAD)
    if compressed == CANONICAL_PCM:
        print(CANONICAL_COMPRESSED_MESSAGE[:80] + "…")
    else:
        print(preview[:80] + "…")

    print()
    print("=" * 60)
    print("Long Cursor-style instruction (anonymized dev session)")
    print("=" * 60)
    print("Derived from COE benchmark corpus — real agent triage pattern.")
    print()
    print("BEFORE (excerpt):")
    print(CURSOR_DEV_INSTRUCTION[:220] + "…")
    print()
    print("AFTER (PCM):")
    print(CURSOR_DEV_PCM)
    print()
    print(format_token_metrics(cursor_dev_metrics(), label="Token savings"))
    print()
    print("Payload excerpt:")
    print(CURSOR_DEV_PAYLOAD.splitlines()[1])
    print("...")
    print()
    print(f"Context / chat history → optimize with COE: {COE_REPO}")
    print("Examples: data/examples/ · REST: POST /compress · Proxy: POST /v1/chat/completions")
    return 0


def run_quickstart() -> int:
    run_demo(stub=not _ollama_available())
    port = 8090
    print("=" * 60)
    print("Quickstart — OpenAI SDK against PCM proxy")
    print("=" * 60)
    print(
        f"""
1. cp .env.example .env   # set MISTRAL_API_KEY
2. python run.py --proxy  # listens on :{port}

Python (pip install openai):

    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:{port}/v1",
        api_key="dummy",
        default_headers={{"x-pcm-provider": "mistral"}},
    )

    with open("data/examples/proxy_chat.json") as f:
        body = json.load(f)

    response = client.chat.completions.create(**body)
    print(response.choices[0].message.content)

curl (compression headers only):

    curl -s http://localhost:{port}/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d @data/examples/proxy_chat.json -D - -o /dev/null | grep -i x-pcm

MCP (Cursor): python scripts/mcp/print_cursor_config.py
"""
    )
    return 0


def run_http_server(port=8080):
    from main import run_http_server as _run_http_server

    _run_http_server(port)


def run_mcp_http_server(host=None, port=None):
    from main import (
        DEFAULT_MCP_HTTP_HOST,
        DEFAULT_MCP_HTTP_PORT,
        run_mcp_http_server as _run_mcp_http_server,
    )

    _run_mcp_http_server(
        host=host or DEFAULT_MCP_HTTP_HOST,
        port=port or DEFAULT_MCP_HTTP_PORT,
    )


def run_mcp_server():
    from main import run_mcp_server as _run_mcp_server

    _run_mcp_server()


def run_benchmark(extra_args=None):
    cmd = [sys.executable, str(ROOT / "scripts" / "benchmark.py")]
    if extra_args:
        cmd.extend(extra_args)
    sys.exit(subprocess.call(cmd))


def run_proxy_server(port=None):
    from main import DEFAULT_PROXY_PORT, run_proxy_server as _run_proxy_server

    _run_proxy_server(port or DEFAULT_PROXY_PORT)


def run_tests(fast: bool = False):
    print("=" * 60)
    print("PCM — running tests")
    print("=" * 60)
    print()

    try:
        import pytest
    except ImportError:
        print("pytest not installed. Run: pip install -e \".[dev]\"")
        sys.exit(1)

    args = ["-v", "tests/"]
    if fast:
        args = ["-m", "not integration", "-q", "--tb=short", "tests/"]
        print("Fast mode: skipping Ollama integration tests")
        print()

    sys.exit(pytest.main(args))


def run_ci():
    script = ROOT / "scripts" / "ci-local.sh"
    if not script.is_file():
        run_tests(fast=True)
        return
    sys.exit(subprocess.call([str(script)]))


def check_dependencies():
    print("Checking dependencies...")

    required = [
        ("ollama", "Ollama Python client"),
        ("fastapi", "HTTP framework (proxy/REST)"),
        ("uvicorn", "ASGI server"),
        ("tiktoken", "Token counting"),
        ("numpy", "Numeric utilities"),
    ]
    missing = []

    for package, description in required:
        try:
            __import__(package)
            print(f"  ✓ {package:15} ({description})")
        except ImportError:
            print(f"  ✗ {package:15} ({description}) — missing")
            missing.append(package)

    if missing:
        print()
        print("Install with: pip install -e \".[dev]\"")
        sys.exit(1)

    print()
    print("Checking Ollama...")
    if _ollama_available():
        print("  ✓ Ollama reachable")
    else:
        print("  ✗ Ollama not reachable (OK for --demo-stub and --test-fast)")


def build_parser(*, maintainer: bool = False) -> argparse.ArgumentParser:
    from main import (
        DEFAULT_MCP_HTTP_HOST,
        DEFAULT_MCP_HTTP_PORT,
        DEFAULT_MODEL,
        DEFAULT_PORT,
        DEFAULT_PROXY_PORT,
    )

    description = (
        "Prompt Compression Middleware (PCM)"
        if not maintainer
        else "PCM — full maintainer CLI"
    )
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=MAINTAINER_EPILOG if maintainer else VISITOR_EPILOG,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--demo", action="store_true", help="Canonical compression demo")
    group.add_argument(
        "--demo-stub",
        action="store_true",
        help="Canonical demo without Ollama (deterministic)",
    )
    group.add_argument(
        "--quickstart",
        action="store_true",
        help="Demo plus OpenAI SDK integration snippet",
    )
    group.add_argument("--http", action="store_true", help="REST API on :8080")
    group.add_argument("--stdio", action="store_true", help="MCP stdio transport")
    group.add_argument("--mcp-http", action="store_true", help="MCP HTTP on :8001/mcp")
    group.add_argument(
        "--proxy",
        action="store_true",
        help="OpenAI-compatible proxy on :8090",
    )
    group.add_argument("--test", action="store_true", help="Run all pytest suites")
    group.add_argument(
        "--test-fast",
        action="store_true",
        help="Run pytest without Ollama integration tests",
    )
    group.add_argument("--ci", action="store_true", help="Local CI (scripts/ci-local.sh)")
    group.add_argument("--benchmark", action="store_true", help="Compressor benchmark")
    group.add_argument("--check-deps", action="store_true", help="Check dependencies")

    parser.add_argument("--port", type=int, default=None, help="Port override")
    parser.add_argument("--host", type=str, default=None, help="MCP HTTP host")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model")

    parser.set_defaults(
        _default_model=DEFAULT_MODEL,
        _default_port=DEFAULT_PORT,
        _default_proxy_port=DEFAULT_PROXY_PORT,
        _default_mcp_host=DEFAULT_MCP_HTTP_HOST,
        _default_mcp_port=DEFAULT_MCP_HTTP_PORT,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    if argv is not None:
        # Used by tests
        pass
    else:
        argv = sys.argv[1:]

    if "--help-all" in argv:
        build_parser(maintainer=True).print_help()
        return

    parser = build_parser(maintainer=False)
    args = parser.parse_args(argv)

    from main import compressor

    compressor.config.model = args.model

    if args.check_deps:
        check_dependencies()
        return

    if args.demo_stub:
        sys.exit(run_demo(stub=True))

    if args.demo:
        sys.exit(run_demo(stub=False))

    if args.quickstart:
        sys.exit(run_quickstart())

    if args.ci:
        run_ci()
        return

    if args.test_fast:
        run_tests(fast=True)
        return

    if args.test:
        run_tests(fast=False)
        return

    if args.benchmark:
        benchmark_args = []
        if args.model != args._default_model:
            benchmark_args.extend(["--model", args.model])
        run_benchmark(benchmark_args)
        return

    if args.stdio:
        run_mcp_server()
        return

    if args.mcp_http:
        run_mcp_http_server(
            host=args.host or args._default_mcp_host,
            port=args.port or args._default_mcp_port,
        )
        return

    if args.proxy:
        run_proxy_server(args.port or args._default_proxy_port)
        return

    if args.http or not any(
        [
            args.demo,
            args.demo_stub,
            args.quickstart,
            args.stdio,
            args.mcp_http,
            args.proxy,
            args.test,
            args.test_fast,
            args.ci,
            args.benchmark,
            args.check_deps,
        ]
    ):
        # Default: show help for visitors instead of silently starting HTTP
        if len(argv) == 0:
            parser.print_help()
            return
        run_http_server(args.port or args._default_port)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
