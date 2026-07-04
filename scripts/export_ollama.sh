#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FUSED="${ROOT}/data/training/checkpoints/pcm-fused"
ADAPTER="${ROOT}/data/training/checkpoints/pcm-lora"
MODELFILE="${ROOT}/data/training/Modelfile"
MODEL="mlx-community/Qwen2.5-3B-Instruct-4bit"

if [[ ! -f "${ADAPTER}/adapters.safetensors" ]]; then
  echo "No existe adapter en ${ADAPTER}"
  echo "Ejecuta primero: python scripts/train_compressor.py"
  exit 1
fi

# Ollama no importa safetensors MLX 4-bit (dtype U32). Re-fundir sin cuantizar.
if [[ ! -f "${FUSED}/model.safetensors" ]] || grep -q '"bits": 4' "${FUSED}/config.json" 2>/dev/null; then
  echo "Fusionando adapter sin cuantización (compatible con Ollama, ~6 GB)..."
  rm -rf "${FUSED}"
  python -m mlx_lm fuse \
    --model "${MODEL}" \
    --adapter-path "${ADAPTER}" \
    --save-path "${FUSED}" \
    --dequantize
fi

cd "${ROOT}/data/training"
echo "Importando modelo fused a Ollama como pcm-compressor..."
ollama create pcm-compressor -f Modelfile

echo ""
echo "Modelo Ollama creado: pcm-compressor"
echo "Smoke test:"
ollama run pcm-compressor "Prompt a comprimir: Analiza este código Python buscando bugs de concurrencia" || true
