#!/usr/bin/env python3
"""Print Cursor MCP config JSON with absolute paths (no symlink resolve)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    venv_python = root / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.is_file() else sys.executable
    run_py = str(root / "run.py")

    config = {
        "mcpServers": {
            "pcm": {
                "command": python,
                "args": [run_py, "--stdio"],
                "env": {
                    "OLLAMA_HOST": "http://localhost:11434",
                    "OLLAMA_MODEL": "granite4.1:3b",
                },
            }
        }
    }

    print(json.dumps(config, indent=2))
    print(
        "\n# Paste into Cursor MCP settings (stdio). "
        "Requires: pip install -e \".[mcp]\" and Ollama with a compressor model.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
