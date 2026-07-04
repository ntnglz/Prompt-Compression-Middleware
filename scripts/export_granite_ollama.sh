#!/usr/bin/env bash
# Exporta granite fine-tuned a Ollama como pcm-granite (vía GGUF).
# Ollama no importa safetensors GraniteForCausalLM directamente.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT="${ROOT}/data/training/v2/checkpoints"
MERGED="${CHECKPOINT}/granite-merged"
GGUF="${CHECKPOINT}/pcm-granite-f16.gguf"
MODELFILE="${CHECKPOINT}/Modelfile"
MODEL_NAME="pcm-granite"
CONVERT="${CONVERT_HF_TO_GGUF:-/opt/homebrew/Cellar/llama.cpp/9100/bin/convert_hf_to_gguf.py}"

if [[ ! -d "${MERGED}" ]]; then
  echo "No existe modelo fusionado en ${MERGED}"
  echo "Ejecuta primero: python scripts/merge_granite_lora.py"
  exit 1
fi

if [[ ! -f "${GGUF}" ]]; then
  echo "Convirtiendo HF → GGUF (f16, ~5 GB)..."
  python -c "import gguf" 2>/dev/null || pip install gguf
  python "${CONVERT}" "${MERGED}" --outfile "${GGUF}" --outtype f16
fi

cat > "${MODELFILE}" <<EOF
FROM ./pcm-granite-f16.gguf
PARAMETER temperature 0.1
SYSTEM ""
EOF

cd "${CHECKPOINT}"
echo "Creando modelo Ollama: ${MODEL_NAME}"
ollama create "${MODEL_NAME}" -f Modelfile

echo ""
echo "Smoke test:"
curl -s http://127.0.0.1:11434/api/generate \
  -d '{"model":"'"${MODEL_NAME}"'","prompt":"Prompt a comprimir: Resume este informe trimestral","stream":false,"options":{"num_predict":80}}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin).get('response','')[:300])"
