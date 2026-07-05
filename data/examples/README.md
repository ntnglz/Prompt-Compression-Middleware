# PCM example payloads

Canonical examples used in the README, `python run.py --demo`, and curl snippets.

| File | Use with | Description |
|------|----------|-------------|
| [`canonical_compress.json`](canonical_compress.json) | `POST http://localhost:8080/compress` | Short instruction — code review |
| [`proxy_chat.json`](proxy_chat.json) | `POST http://localhost:8090/v1/chat/completions` | Short instruction + Python payload |
| [`cursor_dev_compress.json`](cursor_dev_compress.json) | `POST /compress` | **Long** Cursor-style CI triage instruction |
| [`cursor_dev_triage.json`](cursor_dev_triage.json) | `POST /v1/chat/completions` | Long instruction + anonymized pytest log |

The long examples follow patterns from the [COE dev_agent benchmark corpus](https://github.com/ntnglz/Context-Optimization-Engine/tree/master/data/benchmarks/cases/dev_agent) (anonymized `ExampleService` / `ExampleApp` names). PCM compresses the **instruction**; attach [COE](https://github.com/ntnglz/Context-Optimization-Engine) when you need to optimize **context** blocks or chat history.

## Token savings (`python run.py --demo-stub`)

| Example | Instruction | Payload | Total input |
|---------|-------------|---------|-------------|
| `proxy_chat.json` | 24 → 20 (17%) | 147 unchanged | 171 → 167 (2%) |
| `cursor_dev_triage.json` | 160 → 31 (**81%**) | 95 unchanged | 255 → 126 (**51%**) |

Counts use tiktoken `gpt-4` encoding (same estimate as the compressor).

## Quick try

```bash
# REST API (compression only, no upstream LLM)
curl -s http://localhost:8080/compress \
  -H "Content-Type: application/json" \
  -d @data/examples/canonical_compress.json | jq .

# Long instruction metrics
curl -s http://localhost:8080/compress \
  -H "Content-Type: application/json" \
  -d @data/examples/cursor_dev_compress.json | jq .

# Proxy (compress + forward to upstream — needs MISTRAL_API_KEY in .env)
curl -s http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @data/examples/cursor_dev_triage.json -D - -o /dev/null | grep -i x-pcm
```

Expected PCM lines:

```text
# canonical
TASK=review INPUT=python CHECK=race,leak,perf FORMAT=markdown ORDER=severity

# cursor dev triage
TASK=triage INPUT=pytest_ci CHECK=auth_401,missing_token,billing_rounding FORMAT=priority_list ORDER=user_impact START=auth
```

Payload fences are passed through unchanged.
