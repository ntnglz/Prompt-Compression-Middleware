#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FUSED="${ROOT}/data/training/checkpoints/pcm-fused"
GGUF="${ROOT}/data/training/checkpoints/pcm-compressor.gguf"
MODELFILE="${ROOT}/data/training/Modelfile"

if [[ ! -d "$FUSED" ]]; then
  echo "No existe $FUSED. Ejecuta primero: python scripts/train_compressor.py"
  exit 1
fi

python -m mlx_lm.convert \
  --hf-path "$FUSED" \
  --mlx-path "$FUSED" \
  -q \
  --q-bits 4 \
  --q-group-size 64 \
  --quantize-mlx \
  --quantize-gguf \
  --gguf-path "$GGUF"

cd "${ROOT}/data/training"
ollama create pcm-compressor -f Modelfile

echo "Modelo Ollama creado: pcm-compressor"
ollama run pcm-compressor "Prompt a comprimir: Analiza este código Python" || true
