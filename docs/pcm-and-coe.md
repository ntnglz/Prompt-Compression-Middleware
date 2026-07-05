# PCM + COE — saving tokens end-to-end

PCM and [Context Optimization Engine (COE)](https://github.com/ntnglz/Context-Optimization-Engine) solve **different** parts of the same problem: paying less for LLM input tokens without losing task fidelity.

```
User message
    │
    ▼
┌─────────────────────────────────────┐
│  PCM — instruction compression      │  NL → TASK=… (Ollama compressor)
│  "Review this code carefully…"      │
│                                     │
│  Outbound system prompt includes:   │
│  • compressed instruction (PCM)     │
│  • RESPONSE block (output_style)    │
│  • optional PCM interpretation hint │
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

`output_style` is `normal` (user-facing) or `concise` (agent loops); the RESPONSE rules live in the system prompt.

## What each project does

| Layer | Project | Compresses | Typical savings |
|-------|---------|------------|-----------------|
| **Instruction** | **PCM** (this repo) | Verbose natural-language *asks* | High on long Cursor-style prompts (see metrics below) |
| **Context** | **[COE](https://github.com/ntnglz/Context-Optimization-Engine)** | Attached logs, RAG blocks, multi-turn history | High on long sessions and tool output |
| **Output style** | **PCM** (`output_style`) | `normal` (user-facing) vs `concise` (agent loops) — RESPONSE rules in system | Fewer output tokens on agent/tool chains |

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

## Output directives and turn cost

| Env var | Default | Purpose |
|---------|---------|---------|
| `PCM_OUTPUT_STYLE` | `normal` | `normal` or `concise` — selects the RESPONSE block in the outbound system prompt |
| `PCM_RESPONSE_LANG` | `en` | Language line inside the RESPONSE block |

Proxy responses also expose per-turn cost (input + output tokens, priced separately):

| Header | Meaning |
|--------|---------|
| `X-PCM-Input-Tokens` | Upstream prompt tokens for the turn |
| `X-PCM-Output-Tokens` | Upstream completion tokens for the turn |
| `X-PCM-Cost-Total-USD` | `input_price × input_tokens + output_price × output_tokens` |

## E2E: output directives (July 2026)

Three-arm benchmark on [`data/e2e_prompts.json`](../data/e2e_prompts.json) (4 prompts) and [`data/e2e_prompts_extensive.json`](../data/e2e_prompts_extensive.json) (8 prompts from anonymized COE transcripts). Compressor: **`pcm-granite`** (`glossary_only=True`, auto in CLI). Target: `mistral-medium-3.5` with **`reasoning=none`** (required — `reasoning=high` inflates `completion_tokens` with hidden reasoning).

### Short corpus (4 prompts)

| Arm | Instruction | `output_style` | Total cost |
|-----|-------------|----------------|------------|
| Baseline | Natural | `normal` | **$0.0204** |
| PCM | Compressed | `normal` | **$0.0092** (−55%) |
| PCM + concise | Compressed | `concise` | **$0.0074** (−64%) |

| Metric | Value |
|--------|-------|
| Response similarity (PCM normal vs baseline) | **93.25%** |
| Avg output token savings (concise vs PCM normal) | **26.4%** |
| Avg cost delta (concise vs baseline, per prompt) | **−$0.00325** |

### Extensive corpus (8 prompts)

| Arm | Total cost |
|-----|------------|
| Baseline | **$0.0444** |
| PCM | **$0.0352** (−21%) |
| PCM + concise | **$0.0132** (−70%) |

| Metric | Value |
|--------|-------|
| Response similarity | **90.62%** |
| Avg output token savings (concise vs PCM) | **56.7%** |

Reproduce:

```bash
PYTHONPATH=src .venv/bin/python scripts/e2e_benchmark.py \
  --compressor-model pcm-granite --reasoning-effort none -q

PYTHONPATH=src .venv/bin/python scripts/e2e_benchmark.py \
  --compressor-model pcm-granite \
  --prompts data/e2e_prompts_extensive.json --reasoning-effort none -q
```

Full report: [`data/benchmarks/output_directives_e2e.md`](../data/benchmarks/output_directives_e2e.md) · JSON (short): [`e2e_mistral_medium_3_5_20260705_205226.json`](../data/e2e/runs/e2e_mistral_medium_3_5_20260705_205226.json) · JSON (extensive): [`e2e_mistral_medium_3_5_20260705_205454.json`](../data/e2e/runs/e2e_mistral_medium_3_5_20260705_205454.json)
