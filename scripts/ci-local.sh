#!/usr/bin/env bash
# CI local rápido — sin Ollama ni APIs externas
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

# Prefer editable install; fall back to PYTHONPATH for bare checkouts
if ! "$PYTHON" -c "import pcm" 2>/dev/null; then
  export PYTHONPATH="${ROOT}/src"
fi

echo "==> PCM CI local (rápido)"
echo "    Excluye: integration (Ollama)"
echo

"$PYTHON" -m pytest -m "not integration" -q --tb=short "$@"

if [[ -f "${ROOT}/data/training/v2/train.jsonl" ]]; then
  echo
  echo "==> Leakage check (v2 train)"
  "$PYTHON" scripts/check_dataset_leakage.py
fi

echo
echo "==> CI local OK"
