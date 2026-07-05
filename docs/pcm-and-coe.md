# PCM + COE — saving tokens end-to-end

PCM and [Context Optimization Engine (COE)](https://github.com/ntnglz/Context-Optimization-Engine) solve **different** parts of the same problem: paying less for LLM input tokens without losing task fidelity.

```
User message
    │
    ▼
┌─────────────────────────────────────┐
│  PCM — instruction compression      │  NL → TASK=… (Ollama compressor)
│  "Review this code carefully…"      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  COE — context optimization         │  logs, RAG, chat history, tool output
│  levels 1–5, session memory         │
└─────────────────────────────────────┘
    │
    ▼
Upstream LLM (Mistral, OpenAI, …)
```

## What each project does

| Layer | Project | Compresses | Typical savings |
|-------|---------|------------|-----------------|
| **Instruction** | **PCM** (this repo) | Verbose natural-language *asks* | High on long Cursor-style prompts (see metrics below) |
| **Context** | **[COE](https://github.com/ntnglz/Context-Optimization-Engine)** | Attached logs, RAG blocks, multi-turn history | High on long sessions and tool output |

PCM **does not** rewrite fenced code, documents, or CI logs — it only shrinks the instruction that precedes them. COE **does** optimize those context blocks and chat memory.

## Example metrics (anonymized Cursor dev session)

Instruction derived from the [COE benchmark corpus](https://github.com/ntnglz/Context-Optimization-Engine/tree/master/data/benchmarks/cases/dev_agent) (real agent triage pattern, names anonymized to `ExampleService`).

| | Tokens (tiktoken gpt-4 estimate) |
|---|----------------------------------|
| Instruction before | 160 |
| Instruction after PCM | 31 (**81% saved** on instruction) |
| CI log payload | 95 (unchanged by PCM) |
| **Full message input** | 255 → 126 (**51% saved**) |

PCM line:

```text
TASK=triage INPUT=pytest_ci CHECK=auth_401,missing_token,billing_rounding FORMAT=priority_list ORDER=user_impact START=auth
```

To shrink the **95-token log block** and multi-turn history, add COE downstream of PCM.

Files: [`data/examples/cursor_dev_triage.json`](../data/examples/cursor_dev_triage.json)

## Quick start both

1. **PCM** — `docker compose up` or `python run.py --proxy` ([getting-started](getting-started.md))
2. **COE** — [COE getting started](https://github.com/ntnglz/Context-Optimization-Engine/blob/master/docs/getting-started.md)

Pipeline wiring is integration-specific; the intended order is always **PCM first, COE second** on the path to the upstream model.

## When you only need one

- Long instructions, modest payloads → **PCM alone**
- Short asks, huge RAG/history → **COE alone**
- Cursor agent with verbose asks *and* long tool logs → **both**

See also [FAQ — PCM vs COE](FAQ.md#pcm-vs-coe--which-do-i-need).
