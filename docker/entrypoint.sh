#!/bin/sh
set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
MODEL="${OLLAMA_MODEL:-granite4.1:3b}"
PROXY_PORT="${PCM_PROXY_PORT:-8090}"

echo "Esperando Ollama en ${OLLAMA_HOST}..."
until curl -sf "${OLLAMA_HOST}/api/tags" >/dev/null; do
  sleep 2
done

echo "Comprobando modelo ${MODEL}..."
if ! curl -sf "${OLLAMA_HOST}/api/tags" | grep -q "\"name\":\"${MODEL}\""; then
  echo "Descargando ${MODEL} (puede tardar varios minutos)..."
  curl -sf "${OLLAMA_HOST}/api/pull" -d "{\"name\":\"${MODEL}\"}" >/dev/null
fi

case "$1" in
  proxy)
    exec python run.py --proxy --port "${PROXY_PORT}"
    ;;
  api)
    exec python run.py --http --port "${HTTP_PORT:-8080}"
    ;;
  mcp)
    exec python run.py --mcp-http --port "${MCP_HTTP_PORT:-8001}"
    ;;
  *)
    exec python run.py "$@"
    ;;
esac
