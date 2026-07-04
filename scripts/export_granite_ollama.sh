#!/usr/bin/env bash
# Exporta adapter granite fine-tuned a Ollama como pcm-granite
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT="${ROOT}/data/training/v2/checkpoints"
ADAPTER="${CHECKPOINT}/granite-lora"
MODELFILE="${CHECKPOINT}/Modelfile"
MODEL_NAME="pcm-granite"

if [[ ! -d "${ADAPTER}" ]]; then
  echo "No existe adapter en ${ADAPTER}"
  echo "Descarga el adapter desde RunPod y colócalo ahí."
  echo "Ver: docs/fase3b-granite-cloud.md"
  exit 1
fi

if [[ ! -f "${MODELFILE}" ]]; then
  cat > "${MODELFILE}" <<EOF
FROM ./granite-merged
PARAMETER temperature 0.1
SYSTEM ""
EOF
  echo "Modelfile creado en ${MODELFILE} — ajusta FROM si usas GGUF distinto."
fi

cd "${CHECKPOINT}"
echo "Creando modelo Ollama: ${MODEL_NAME}"
ollama create "${MODEL_NAME}" -f Modelfile

echo ""
echo "Smoke test (glossary, sin few-shots en runtime):"
ollama run "${MODEL_NAME}" "Prompt a comprimir: Resume este informe trimestral en lista con métricas clave" || true
