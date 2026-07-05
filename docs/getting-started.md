# Getting started

Install PCM, pick an integration path, and run the canonical example in under 15 minutes.

## Prerequisites

| Component | Docker proxy | Local proxy | REST only | MCP |
|-----------|-------------|-------------|-----------|-----|
| Python 3.11+ | in image | required | required | required |
| Docker + Compose | required | — | — | — |
| Ollama + compressor model | in compose | required | required | required |
| Upstream API key | `MISTRAL_API_KEY` in `.env` | same | not needed | optional |

Compressor models (Ollama): `granite4.1:3b` (default) or `pcm-granite` (fine-tuned, recommended if available).

## Install

```bash
git clone https://github.com/ntnglz/Prompt-Compression-Middleware.git
cd Prompt-Compression-Middleware
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Verify:

```bash
python -c "from pcm import PromptCompressor; print('ok')"
python run.py --demo-stub
```

No `PYTHONPATH` required after editable install.

## Path 1 — Docker proxy (recommended for visitors)

```bash
cp .env.example .env
# Edit .env: MISTRAL_API_KEY=...

docker compose up --build
```

- Proxy: `http://localhost:8090/v1/chat/completions`
- Health: `curl http://localhost:8090/health`
- First start pulls the Ollama compressor model (may take several minutes).

Compress and forward:

```bash
curl -s http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @data/examples/proxy_chat.json -D - -o /dev/null | grep -i x-pcm
```

## Path 2 — Local proxy

```bash
cp .env.example .env
ollama pull granite4.1:3b

python run.py --proxy
```

OpenAI Python SDK:

```python
from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:8090/v1",
    api_key="dummy",
    default_headers={"x-pcm-provider": "mistral"},
)

with open("data/examples/proxy_chat.json") as f:
    body = json.load(f)

response = client.chat.completions.create(**body)
print(response.choices[0].message.content)
```

Or run `python run.py --quickstart` for the full snippet.

## Path 3 — REST compression API

No upstream key required.

```bash
python run.py --http    # listens on :8080
```

```bash
curl -s http://localhost:8080/compress \
  -H "Content-Type: application/json" \
  -d @data/examples/canonical_compress.json | jq .
```

Expected compressed instruction:

```text
TASK=review INPUT=python CHECK=race,leak,perf FORMAT=markdown ORDER=severity
```

API docs: `http://localhost:8080/docs`

## Path 4 — MCP

### stdio (Cursor)

```bash
pip install -e ".[mcp]"
python scripts/mcp/print_cursor_config.py
```

Paste the JSON into Cursor MCP settings. Requires Ollama with a compressor model.

### HTTP

```bash
python run.py --mcp-http   # http://localhost:8001/mcp
```

## Mode reference

| Flag | Port | Protocol |
|------|------|----------|
| `python run.py --proxy` | 8090 | OpenAI-compatible HTTP |
| `python run.py --http` | 8080 | PCM REST (`/compress`, …) |
| `python run.py --mcp-http` | 8001 | MCP streamable HTTP |
| `python run.py --stdio` | — | MCP stdio |

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | — | Mistral upstream |
| `OPENAI_API_KEY` | — | OpenAI upstream |
| `PCM_UPSTREAM_PROVIDER` | `mistral` | Default provider |
| `OLLAMA_MODEL` | `granite4.1:3b` | Compressor model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama URL |
| `PCM_PROXY_PORT` | `8090` | Proxy port |
| `PCM_MIN_INSTRUCTION_TOKENS` | `12` | Skip compression below this |

Provider selection: header `x-pcm-provider` → model auto-detect → `.env` default.

## PCM + COE

PCM compresses **instructions**; [Context Optimization Engine](https://github.com/ntnglz/Context-Optimization-Engine) optimizes **full context** and chat memory.

Suggested pipeline:

`User → PCM proxy → COE → upstream LLM`

- PCM repo: https://github.com/ntnglz/Context-Optimization-Engine
- COE getting started: https://github.com/ntnglz/Context-Optimization-Engine/blob/master/docs/getting-started.md
- Combined guide: [pcm-and-coe.md](pcm-and-coe.md)

Example: a long Cursor CI-triage ask shrinks ~81% with PCM; the attached pytest log is unchanged until COE optimizes context. See [`cursor_dev_triage.json`](../data/examples/cursor_dev_triage.json).

## Without Ollama

- `python run.py --demo-stub` — deterministic canonical example
- `python run.py --test-fast` / `./scripts/ci-local.sh` — unit tests
- Real compression and proxy require Ollama (or Docker Compose with the `ollama` service)

## CI (local)

No GitHub Actions in this repo yet. Run:

```bash
pip install -e ".[dev]"
python run.py --ci
# or: ./scripts/ci-local.sh
```

## Next steps

- [FAQ](FAQ.md)
- [Examples](../data/examples/README.md)
- [README](../README.md)
