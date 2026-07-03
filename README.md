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

## Multi-upstream

El proxy soporta varios proveedores OpenAI-compatible:

| Proveedor | Variable API key | URL por defecto | Modelo por defecto |
|-----------|------------------|-----------------|-------------------|
| `mistral` | `MISTRAL_API_KEY` | `https://api.mistral.ai/v1` | `mistral-medium-3.5` |
| `openai` | `OPENAI_API_KEY` | `https://api.openai.com/v1` | `gpt-4o-mini` |
| `openrouter` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` | `openai/gpt-4o-mini` |
| `custom` | `PCM_UPSTREAM_API_KEY` | `PCM_UPSTREAM_URL` | `PCM_UPSTREAM_MODEL` |

### Selección de proveedor

Prioridad:
1. Header `x-pcm-provider: openai` (por petición)
2. Auto-detección por modelo (`gpt-*` → OpenAI, `mistral-*` → Mistral)
3. `PCM_UPSTREAM_PROVIDER` en `.env` (por defecto: `mistral`)

```bash
# Mistral (por defecto)
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hola"}]}'

# OpenAI explícito
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-pcm-provider: openai" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hola"}]}'

# Auto-ruta a OpenAI por nombre de modelo
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hola"}]}'
```

Headers extra en respuesta: `X-PCM-Upstream-Provider`, `X-PCM-Upstream-Model`

`reasoning_effort` solo se envía a Mistral (no a OpenAI).

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
| `MISTRAL_API_KEY` | — | Clave Mistral |
| `OPENAI_API_KEY` | — | Clave OpenAI |
| `OPENROUTER_API_KEY` | — | Clave OpenRouter (opcional) |
| `PCM_UPSTREAM_PROVIDER` | `mistral` | Proveedor por defecto |
| `OLLAMA_MODEL` | `granite4.1:3b` | Modelo compresor local |
| `OLLAMA_HOST` | `http://localhost:11434` | URL Ollama (`http://ollama:11434` en Docker) |
| `PCM_PROXY_PORT` | `8090` | Puerto del proxy |
| `PCM_UPSTREAM_MODEL` | — | Modelo destino (vacío = default del proveedor) |
| `PCM_REASONING_EFFORT` | `none` | Solo Mistral: `none` o `high` |
| `PCM_MIN_INSTRUCTION_TOKENS` | `12` | Umbral mínimo para comprimir |

## OpenAI SDK (Python)

```python
from openai import OpenAI

# Mistral vía proxy
client = OpenAI(
    base_url="http://localhost:8090/v1",
    api_key="dummy",
    default_headers={"x-pcm-provider": "mistral"},
)

# OpenAI vía proxy (auto-ruta si model=gpt-4o-mini)
client_openai = OpenAI(
    base_url="http://localhost:8090/v1",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="mistral-medium-3.5",
    messages=[{"role": "user", "content": "Resume este informe anual..."}],
)
print(response.choices[0].message.content)
```

## Fase 3 — Fine-tuning local (Mac Apple Silicon)

Ver [docs/fase3-finetuning.md](docs/fase3-finetuning.md).

## Tests

### CI local (rápido, sin Ollama)

```bash
./scripts/ci-local.sh
# o
python run.py --test-fast
```

### CI local completo (requiere Ollama)

```bash
./scripts/ci-local-full.sh
# o
python run.py --test
```

### Tests concretos

```bash
pytest tests/test_compression_policy.py tests/test_proxy.py -q
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
