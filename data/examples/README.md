# PCM example payloads

Canonical examples used in the README, `python run.py --demo`, and curl snippets.

| File | Use with | Description |
|------|----------|-------------|
| [`canonical_compress.json`](canonical_compress.json) | `POST http://localhost:8080/compress` | Instruction-only body for the REST compression API |
| [`proxy_chat.json`](proxy_chat.json) | `POST http://localhost:8090/v1/chat/completions` | OpenAI-compatible chat body (instruction + Python payload) |

## Quick try

```bash
# REST API (compression only, no upstream LLM)
curl -s http://localhost:8080/compress \
  -H "Content-Type: application/json" \
  -d @data/examples/canonical_compress.json | jq .

# Proxy (compress + forward to upstream — needs MISTRAL_API_KEY in .env)
curl -s http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @data/examples/proxy_chat.json -D - -o /dev/null | grep -i x-pcm
```

Expected compressed instruction:

```text
TASK=review INPUT=python CHECK=race,leak,perf FORMAT=markdown ORDER=severity
```

The fenced Python block in `proxy_chat.json` is passed through unchanged.
