# Prompt Compression Middleware (PCM)

Middleware que comprime instrucciones en lenguaje natural a formato **PCM** (`TASK=review INPUT=python...`) antes de enviarlas a un LLM destino, preservando payloads (código, documentos) sin modificar.

```
Cliente → PCM Proxy (:8090) → Mistral / OpenAI-compatible API
              ↓
         Ollama (compresor local)
```

## Requisitos

- [Docker](https://docs.docker.com/get-docker/) y Docker Compose
- Clave API de [Mistral](https://console.mistral.ai/) (u otro proveedor compatible)

## Inicio rápido con Docker

```bash
git clone https://github.com/ntnglz/Prompt-Compression-Middleware.git
cd Prompt-Compression-Middleware

cp .env.example .env
# Edita .env y añade MISTRAL_API_KEY=...

docker compose up --build
```

La primera vez descargará el modelo `granite4.1:3b` en Ollama (puede tardar).

**Proxy listo en:** `http://localhost:8090/v1/chat/completions`

### Health check

```bash
curl http://localhost:8090/health
```

### Ejemplo

```bash
# Prompt largo → se comprime
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": "Analiza este código Python buscando race conditions.\n\n```python\nCACHE = {}\n```"
    }]
  }' -D - -o /dev/null | grep -i x-pcm

# Prompt corto (< 12 tokens) → no se comprime
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hola, responde en una línea."}]}' \
  -D - -o /dev/null | grep -i x-pcm
```

### Headers de respuesta PCM

| Header | Descripción |
|--------|-------------|
| `X-PCM-Messages-Compressed` | Mensajes comprimidos en la petición |
| `X-PCM-Compression-Ratio` | Ratio de ahorro (0 si omitido) |
| `X-PCM-Tokens-Saved` | Tokens ahorrados en input |
| `X-PCM-Compression-Time-Ms` | Tiempo de compresión (Ollama) |

Omitir compresión en una petición: `x-pcm-disable: true`

## Desarrollo local (sin Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ollama en el host con granite4.1:3b
ollama pull granite4.1:3b

cp .env.example .env
# MISTRAL_API_KEY=...

python run.py --proxy          # Proxy → :8090
python run.py --http           # API REST → :8080
python run.py --mcp-http       # MCP HTTP → :8001/mcp
python run.py --stdio          # MCP stdio
```

## Modos de ejecución

| Comando | Puerto | Uso |
|---------|--------|-----|
| `python run.py --proxy` | 8090 | **Producción** — proxy OpenAI-compatible |
| `python run.py --http` | 8080 | API REST de compresión directa |
| `python run.py --mcp-http` | 8001 | Servidor MCP |
| `python run.py --benchmark` | — | Benchmark compresor |
| `scripts/e2e_benchmark.py` | — | Benchmark E2E con Mistral API |

## Configuración (`.env`)

| Variable | Default | Descripción |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | — | Clave API upstream (requerida para proxy) |
| `OLLAMA_MODEL` | `granite4.1:3b` | Modelo compresor local |
| `OLLAMA_HOST` | `http://localhost:11434` | URL Ollama (`http://ollama:11434` en Docker) |
| `PCM_PROXY_PORT` | `8090` | Puerto del proxy |
| `PCM_UPSTREAM_URL` | `https://api.mistral.ai/v1` | API destino |
| `PCM_UPSTREAM_MODEL` | `mistral-medium-3.5` | Modelo destino |
| `PCM_REASONING_EFFORT` | `none` | `none` o `high` (mistral-medium-3.5) |
| `PCM_MIN_INSTRUCTION_TOKENS` | `12` | Umbral mínimo para comprimir |

## OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8090/v1",
    api_key="dummy",  # el proxy usa MISTRAL_API_KEY del entorno
)

response = client.chat.completions.create(
    model="mistral-medium-3.5",
    messages=[{"role": "user", "content": "Resume este informe anual..."}],
)
print(response.choices[0].message.content)
```

## Tests

```bash
pytest tests/test_compression_policy.py tests/test_proxy.py tests/test_e2e_benchmark.py -q
```

## Estructura

```
src/pcm/
  compressor.py      # Compresor Ollama → PCM
  proxy.py           # Proxy HTTP
  compression_policy.py  # Umbral mínimo / ahorro neto
  mcp_server.py      # Servidor MCP
scripts/
  benchmark.py       # Benchmark compresor
  e2e_benchmark.py   # Benchmark E2E Mistral
data/
  e2e_prompts.json   # Prompts E2E con payload
```

## Licencia

Prototipo de investigación — Fase 1/2 completadas.
