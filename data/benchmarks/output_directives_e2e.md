# E2E — Output directives (`output_style`) + coste por turno

> **Fecha:** 2026-07-05 (actualizado con `pcm-granite`)  
> **Spec:** [docs/superpowers/specs/2026-07-05-rcm-v0-design.md](../../docs/superpowers/specs/2026-07-05-rcm-v0-design.md)

Benchmark de tres brazos:

| Brazo | Instrucción | `output_style` | Qué mide |
|-------|-------------|----------------|----------|
| **A — baseline** | Natural, sin PCM | `normal` (system hint PCM solo) | Techo de referencia |
| **B — pcm** | Comprimida | `normal` | Ahorro en ida (input) |
| **C — pcm+concise** | Comprimida | `concise` (`RESPONSE:` en system) | Ahorro en ida + vuelta (output) |

**Compresor:** `pcm-granite` (Ollama, fine-tune Fase 3b) con `glossary_only=True`  
**LLM destino:** `mistral-medium-3.5`  
**Reasoning:** `none` (obligatorio para medir output tokens con fidelidad)  
**Precios Mistral (API):** $1.50 / M input · $7.50 / M output

> `granite4.1:3b` queda como baseline de compresión sin fine-tune. No usarlo para informes E2E de producción — ver [comparativa histórica](#comparativa-granite41-vs-pcm-granite).

## Comando de reproducción

```bash
cd Prompt\ Compression\ Middleware
source .env   # MISTRAL_API_KEY; opcional OLLAMA_MODEL=pcm-granite

# Corpus corto (4 prompts)
PYTHONPATH=src .venv/bin/python scripts/e2e_benchmark.py \
  --compressor-model pcm-granite \
  --reasoning-effort none -q

# Corpus extenso (8 prompts desde transcripts COE anonimizados)
PYTHONPATH=src .venv/bin/python scripts/e2e_benchmark.py \
  --compressor-model pcm-granite \
  --prompts data/e2e_prompts_extensive.json \
  --reasoning-effort none -q
```

El script activa `glossary_only` automáticamente para modelos `pcm-*` (como en `validate_granite_v2.py`).

---

## Corpus corto — `data/e2e_prompts.json` (4 prompts)

**Run JSON:** [../e2e/runs/e2e_mistral_medium_3_5_20260705_205226.json](../e2e/runs/e2e_mistral_medium_3_5_20260705_205226.json)

| Métrica | Baseline (A) | PCM normal (B) | PCM concise (C) |
|---------|--------------|----------------|-----------------|
| Input tokens (total) | 1 233 | 1 204 (−29) | ~1 601* |
| Output tokens (media) | 620 | 247 | **168** |
| **Coste total (4 prompts)** | **$0.0204** | **$0.0092** (−55%) | **$0.0074** (−64%) |
| Similitud respuesta B vs A | — | **93.25%** | — |
| Ahorro output C vs B | — | — | **26.4%** |
| Δ coste C vs A (media/prompt) | — | — | **−$0.00325** |
| Ratio compresión (media) | — | **49.54%** | — |
| t_compresión (media) | — | 3 163 ms | — |

\*El brazo concise añade el bloque `RESPONSE:` al system (~+20–30 tokens de input por prompt). El ahorro neto sigue siendo dominante en output.

### Detalle por prompt (corpus corto)

| ID | Categoría | Out baseline | Out PCM | Out concise | $ baseline | $ PCM | $ concise | Similitud |
|----|-----------|-------------|---------|-------------|------------|-------|-----------|-----------|
| e2e_001 | code_review | 1 345 | 445 | **283** | 0.0106 | 0.0038 | **0.0028** | 100% |
| e2e_002 | code_review | 852 | 286 | **197** | 0.0069 | 0.0026 | **0.0021** | 80% |
| e2e_003 | translation | 56 | 60 | 55 | 0.0008 | 0.0008 | 0.0009 | 98% |
| e2e_004 | summarization | 227 | 195 | **137** | 0.0022 | 0.0019 | **0.0017** | 95% |

---

## Corpus extenso — `data/e2e_prompts_extensive.json` (8 prompts)

Prompts largos anonimizados desde transcripts COE (logs pytest, stack traces, xcodebuild, etc.). Ver [docs/e2e-prompts-corpus.md](../../docs/e2e-prompts-corpus.md).

**Run JSON:** [../e2e/runs/e2e_mistral_medium_3_5_20260705_205454.json](../e2e/runs/e2e_mistral_medium_3_5_20260705_205454.json)

| Métrica | Baseline (A) | PCM normal (B) | PCM concise (C) |
|---------|--------------|----------------|-----------------|
| Input tokens (total) | 2 489 | 1 840 (−649) | ~2 654* |
| Output tokens (media) | 679 | 541 | **153** |
| **Coste total (8 prompts)** | **$0.0444** | **$0.0352** (−21%) | **$0.0132** (−70%) |
| Similitud respuesta B vs A | — | **90.62%** | — |
| Ahorro output C vs B | — | — | **56.7%** |
| Δ coste C vs A (media/prompt) | — | — | **−$0.00391** |
| Ratio compresión (media) | — | **80.24%** | — |
| t_compresión (media) | — | 699 ms | — |

### Detalle por prompt (corpus extenso)

| ID | Categoría | Out baseline | Out PCM | Out concise | $ baseline | $ PCM | $ concise | Similitud |
|----|-----------|-------------|---------|-------------|------------|-------|-----------|-----------|
| ext_001 | dev_agent | 257 | 514 | **133** | 0.0026 | 0.0043 | **0.0016** | 90% |
| ext_002 | dev_agent | 1 185 | 1 066 | **262** | 0.0094 | 0.0083 | **0.0024** | 90% |
| ext_003 | dev_agent | 571 | 412 | **219** | 0.0047 | 0.0034 | **0.0021** | 90% |
| ext_004 | refactor | 954 | **110** | 180 | 0.0076 | **0.0011** | 0.0017 | 90% |
| ext_005 | debug | 678 | 376 | **63** | 0.0055 | 0.0031 | **0.0009** | 90% |
| ext_006 | ux | 1 004 | 826 | **98** | 0.0079 | 0.0064 | **0.0011** | 90% |
| ext_007 | dev_agent | 431 | 580 | **192** | 0.0038 | 0.0049 | **0.0021** | 95% |
| ext_008 | bugfix | 348 | 443 | **77** | 0.0030 | 0.0036 | **0.0010** | 90% |

---

## Interpretación

1. **`pcm-granite`** es el compresor de referencia para E2E y producción (`OLLAMA_MODEL=pcm-granite`). Mantiene similitud ≥90% y reduce coste total de forma consistente.
2. **PCM en instrucción** ahorra input; en el corpus extenso el output domina el coste, por lo que el ahorro PCM solo (−21%) es menor que en el corpus corto (−55%).
3. **`output_style=concise`** es el brazo con mayor impacto económico: **−64%** (corto) y **−70%** (extenso) vs baseline.
4. **La métrica a optimizar es `cost_total`** (input + output con precios distintos), no solo tokens de entrada.
5. Con **`reasoning=high`**, Mistral infla `completion_tokens` con razonamiento interno — **no usar para benchmark de output directives** (ver § abajo).
6. El baseline Mistral **no es determinista** entre runs; comparar brazos A/B/C **dentro del mismo run**.

## Comparativa `granite4.1:3b` vs `pcm-granite`

Runs históricos con `granite4.1:3b`: [203803.json](../e2e/runs/e2e_mistral_medium_3_5_20260705_203803.json) (corto), [204905.json](../e2e/runs/e2e_mistral_medium_3_5_20260705_204905.json) (extenso).

| Corpus | Métrica | `granite4.1:3b` | `pcm-granite` |
|--------|---------|-----------------|---------------|
| Corto (4) | Similitud | 94.5% | 93.2% |
| Corto (4) | Coste PCM vs baseline | −18% | **−55%** |
| Corto (4) | Coste concise vs baseline | −56% | **−64%** |
| Extenso (8) | Similitud | 90.0% | **90.6%** |
| Extenso (8) | Coste PCM vs baseline | −12% | **−21%** |
| Extenso (8) | Coste concise vs baseline | −55% | **−70%** |
| Extenso (8) | t_compresión | 893 ms | **699 ms** |

`pcm-granite` evita el output inflado que `granite4.1:3b` provocaba en algunos prompts extensos (p. ej. compresión demasiado agresiva → respuestas más largas en Mistral).

## Nota: `reasoning=high` distorsiona output tokens

Run de control (1 prompt, `reasoning=high`):

| Brazo | Output tokens reportados | Coste | Texto visible |
|-------|------------------------|-------|---------------|
| concise | **2 837** | $0.0219 | 3 viñetas (~40 palabras) |

Mismo prompt con `reasoning=none`: **138** output tokens, **$0.00167**.

**Conclusión:** benchmarks de `output_style` deben ejecutarse con `--reasoning-effort none`. El script solo admite `high` o `none` (no `auto`).

## Umbrales spec v0 (orientativos)

Evaluados con `pcm-granite`, corpus corto:

| KPI | Umbral | Resultado |
|-----|--------|-----------|
| `cost_delta` pcm+concise vs baseline | ≤ −15% coste total medio | **−64%** ✅ |
| Comprensión E2E (similitud) | ≥ 0.90 | **0.932** ✅ |
| `output_tokens` concise vs normal | ≤ −20% medio | **−26.4%** ✅ |

Corpus extenso: similitud **0.906**, coste concise **−70%** ✅.

## Integración

| Componente | Parámetro |
|------------|-----------|
| Proxy | `OLLAMA_MODEL=pcm-granite`, `PCM_OUTPUT_STYLE=concise` |
| COE compose | `output_style="concise"` en `build_pcm_messages` / `optimize_with_pcm` |
| Headers proxy | `X-PCM-Input-Tokens`, `X-PCM-Output-Tokens`, `X-PCM-Cost-Total-USD` |

Ver también [docs/pcm-and-coe.md](../../docs/pcm-and-coe.md#e2e-output-directives-july-2026).
