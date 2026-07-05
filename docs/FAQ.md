# FAQ

## What does PCM compress?

PCM compresses the **instruction** part of a user message (natural language → `TASK=… INPUT=…`). Fenced code blocks, attached documents, and similar **payloads are not modified**.

## Does PCM summarize or truncate payloads?

No. Only the instruction text before the first fenced code block (or the whole message if there is no fence) is compressed. See the canonical example in [`data/examples/proxy_chat.json`](../data/examples/proxy_chat.json).

## What is the 12-token threshold?

`PCM_MIN_INSTRUCTION_TOKENS` (default **12**) skips compression when the instruction is too short to benefit. Short prompts like “Hi, reply in one line.” pass through unchanged. Response headers show `X-PCM-Messages-Compressed: 0`.

## Can I try PCM without Ollama?

Yes. Run `python run.py --demo-stub` for a deterministic before/after example. Unit tests: `python run.py --test-fast`. Real compression requires Ollama (local or via Docker Compose).

## Do I need an API key?

- **Proxy mode** (`--proxy`, Docker): yes — set `MISTRAL_API_KEY` (or another provider) in `.env` for upstream calls.
- **REST `/compress`**: no upstream key; Ollama still required for live compression.
- **`--demo-stub`**: neither Ollama nor API key.

## PCM vs COE — which do I need?

| | PCM | [COE](https://github.com/ntnglz/Context-Optimization-Engine) |
|---|-----|-----|
| Compresses | NL **instructions** | Full **context** / chat memory |
| Typical input | “Review this code…”, long Cursor asks | Long RAG context, tool logs, history |
| Ollama for demo | Yes (or stub) | Library demo works without API key |
| Example savings | 81% on a long dev-session instruction ([metrics](../data/examples/README.md)) | Depends on context size and COE level |

They are complementary: `User → PCM → COE → LLM`. See [PCM + COE guide](pcm-and-coe.md).

## MCP vs proxy — which should I use?

- **Proxy** — drop-in for existing OpenAI SDK clients; compresses on the way to Mistral/OpenAI.
- **MCP (stdio)** — native tools in Cursor; use `scripts/mcp/print_cursor_config.py`.
- **MCP (HTTP)** — remote agents on `:8001/mcp`.
- **REST** — compress-only microservice on `:8080`.

Use proxy for production-shaped integrations; MCP for IDE tooling; REST if you only need compression.

## What if my prompt is already `TASK=…`?

PCM leaves messages that already use PCM format unchanged.

## How do I disable compression for one request?

Send header `x-pcm-disable: true` on proxy requests.

## Which compressor model should I use?

| Ollama model | When |
|--------------|------|
| `granite4.1:3b` | Default baseline |
| `pcm-granite` | Fine-tuned (recommended if exported) |
| `pcm-compressor` | Experimental MLX export (Apple Silicon) |

Set `OLLAMA_MODEL` in `.env`.

## What are the `X-PCM-*` response headers?

They report how many messages were compressed, token savings, compression time, and which upstream provider/model was used. See [README](../README.md#response-headers-x-pcm).

## Does Docker include Ollama?

Yes. `docker compose up` starts `ollama` and `pcm-proxy`. The entrypoint waits for Ollama and pulls the compressor model on first run.

## How do I run tests locally?

```bash
pip install -e ".[dev]"
./scripts/ci-local.sh
pytest tests/test_run_demo.py -v
```

Integration tests need Ollama: `python run.py --test`.

## Where is fine-tuning documented?

Maintainer docs (Spanish): [experiment conclusions](experimento-pcm-conclusiones.md), [Fase 3b cloud](fase3b-granite-cloud.md). Not required for adoption.

## Is PCM published on PyPI?

Not yet. Install from source: `pip install -e ".[dev]"`.
