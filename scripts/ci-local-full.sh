#!/usr/bin/env bash
# CI local completo — incluye tests de integración (requiere Ollama)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

export PYTHONPATH="${ROOT}/src"

echo "==> PCM CI local (completo)"
echo "    Requiere: Ollama con granite4.1:3b (u modelo del fixture)"
echo

"$PYTHON" -m pytest -q --tb=short "$@"

echo
echo "==> CI completo OK"
