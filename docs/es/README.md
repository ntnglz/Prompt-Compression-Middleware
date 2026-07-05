> **Legacy (ES)** — README anterior a la migración de adopción visitante (v1.0.0).  
> Documentación actual en inglés: [README en inglés](../README.md) · [getting-started](../getting-started.md) · [FAQ](../FAQ.md).

# Prompt Compression Middleware (PCM)

Middleware que comprime **instrucciones** en lenguaje natural a formato **PCM** compacto (`TASK=review INPUT=python...`) antes de enviarlas a un LLM destino, preservando payloads (código, documentos) sin modificar.

```
Cliente → PCM Proxy (:8090) → Mistral / OpenAI-compatible API
              ↓
         Ollama (compresor local: granite4.1:3b | pcm-granite)
```

> **Estado del proyecto (julio 2026):** experimento PCM **cerrado con éxito**. Fine-tune `pcm-granite` validado E2E (94.50% similitud Mistral). Ver [conclusiones del experimento](docs/experimento-pcm-conclusiones.md).

## Requisitos

- [Docker](https://docs.docker.com/get-docker/) y Docker Compose, **o** Python 3.11+ + [Ollama](https://ollama.com)
- Clave API de [Mistral](https://console.mistral.ai/) (u otro proveedor compatible)
- Compresor en Ollama: `granite4.1:3b` (baseline) o `pcm-granite` (fine-tuned, recomendado)

## Inicio rápido con Docker

```bash
git clone https://github.com/ntnglz/Prompt-Compression-Middleware.git
cd Prompt-Compression-Middleware

cp .env.example .env
# Edita .env: MISTRAL_API_KEY=...
# Opcional fine-tuned: OLLAMA_MODEL=pcm-granite

docker compose up --build
```

La primera vez descargará el modelo compresor en Ollama (puede tardar).

**Proxy:** `http://localhost:8090/v1/chat/completions`

### Health check

```bash
curl http://localhost:8090/health
```

### Ejemplo con compresión

```bash
# Prompt largo → se comprime (ver headers X-PCM-*)
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
| `X-PCM-Upstream-Provider` | Proveedor destino (mistral, openai, …) |
| `X-PCM-Upstream-Model` | Modelo destino usado |

Omitir compresión: `x-pcm-disable: true`

## Modelos compresor

| Modelo Ollama | Origen | Cuándo usar |
|---------------|--------|-------------|
| `granite4.1:3b` | Hub Ollama (default) | Baseline, sin fine-tune |
| **`pcm-granite`** | Fase 3b cloud (RunPod) | **Recomendado** — E2E 94.50% |
| `pcm-compressor` | Fase 3a MLX (Mac) | Experimental, solo Apple Silicon |

```bash
# Baseline
ollama pull granite4.1:3b

# Fine-tuned (tras export_granite_ollama.sh — ver docs/fase3b-granite-cloud.md)
ollama list | grep pcm-granite
```

En `.env`:

```bash
OLLAMA_MODEL=pcm-granite
```

## Multi-upstream

| Proveedor | Variable API key | Modelo por defecto |
|-----------|------------------|-------------------|
| `mistral` | `MISTRAL_API_KEY` | `mistral-medium-3.5` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `openrouter` | `OPENROUTER_API_KEY` | `openai/gpt-4o-mini` |
| `custom` | `PCM_UPSTREAM_API_KEY` | `PCM_UPSTREAM_MODEL` |

Selección: header `x-pcm-provider` → auto-detección por nombre de modelo → `PCM_UPSTREAM_PROVIDER` en `.env`.

## Desarrollo local (sin Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull granite4.1:3b   # o usar pcm-granite si ya exportado

cp .env.example .env
# MISTRAL_API_KEY=...
# OLLAMA_MODEL=pcm-granite

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
| `scripts/validate_granite_v2.py` | — | Validación A/B/C post fine-tune |

## Configuración (`.env`)

| Variable | Default | Descripción |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | — | Clave Mistral |
| `OPENAI_API_KEY` | — | Clave OpenAI |
| `PCM_UPSTREAM_PROVIDER` | `mistral` | Proveedor por defecto |
| `OLLAMA_MODEL` | `granite4.1:3b` | Modelo compresor (`pcm-granite` recomendado) |
| `OLLAMA_HOST` | `http://localhost:11434` | URL Ollama |
| `PCM_PROXY_PORT` | `8090` | Puerto del proxy |
| `PCM_MIN_INSTRUCTION_TOKENS` | `12` | Umbral mínimo para comprimir |
| `PCM_REASONING_EFFORT` | `none` | Solo Mistral: `none` o `high` |

## OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8090/v1",
    api_key="dummy",
    default_headers={"x-pcm-provider": "mistral"},
)

response = client.chat.completions.create(
    model="mistral-medium-3.5",
    messages=[{"role": "user", "content": "Resume este informe anual..."}],
)
print(response.choices[0].message.content)
```

## Fine-tuning y experimento PCM

El experimento de compresión de instrucciones está **cerrado** (julio 2026).

| Documento | Contenido |
|-----------|-----------|
| [**Conclusiones del experimento**](docs/experimento-pcm-conclusiones.md) | Resultados, lecciones, cómo repetir |
| [Fase 3b — Granite cloud](docs/fase3b-granite-cloud.md) | RunPod, `pcm-granite`, guía operativa |
| [Fase 3 — MLX Mac](docs/fase3-finetuning.md) | `pcm-compressor` local |
| [Benchmark 3b](data/benchmarks/fase3b_validation.md) | Métricas finales |

**Validación rápida:**

```bash
python scripts/validate_granite_v2.py --semantic --e2e   # requiere Ollama + MISTRAL_API_KEY
python scripts/check_dataset_leakage.py                  # CI: 0 leakage
./scripts/ci-local.sh                                    # tests sin Ollama
```

**Próximo horizonte:** [Context Optimization Engine (COE)](Context%20Optimization%20Engine%20(COE).md) — compresión de contextos completos y memoria semántica del chat.

## Tests

```bash
./scripts/ci-local.sh              # rápido, sin Ollama
./scripts/ci-local-full.sh         # completo, requiere Ollama
pytest tests/test_compression_policy.py tests/test_proxy.py -q
```

## Estructura

```
src/pcm/
  compressor.py           # Compresor Ollama → PCM
  compression_prompts.py  # GLOSSARY vs FULL system prompts
  proxy.py                # Proxy HTTP multi-upstream
  compression_policy.py   # Umbral mínimo / ahorro neto
  mcp_server.py           # Servidor MCP
  training/dataset.py     # Pipeline dataset fine-tuning
scripts/
  validate_granite_v2.py  # Benchmark post fine-tune 3b
  export_granite_ollama.sh
  build_dataset_v2.py
  check_dataset_leakage.py
data/
  eval/                   # Holdouts congelados (nunca en train)
  e2e_prompts.json        # Prompts E2E con payload
  benchmarks/             # Informes de validación
docs/
  experimento-pcm-conclusiones.md
  fase3b-granite-cloud.md
```

## Roadmap

| Fase | Estado |
|------|--------|
| 1 Prototipo (proxy + compresor) | ✅ |
| 2 Especialización (system prompt) | ✅ |
| 3a Fine-tune MLX (`pcm-compressor`) | ✅ |
| 3b Fine-tune cloud (`pcm-granite`) | ✅ |
| 3b-B Dataset ampliado (pre-producción) | Opcional |
| COE (contextos completos) | Planificado |
| 3 RL / 4 LLM IR | Investigación |

## Licencia

Prototipo de investigación — uso bajo responsabilidad del operador.
