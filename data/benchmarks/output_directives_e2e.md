# E2E — Output directives (`output_style`) + coste por turno

> **Fecha:** 2026-07-05  
> **Spec:** [docs/superpowers/specs/2026-07-05-rcm-v0-design.md](../../docs/superpowers/specs/2026-07-05-rcm-v0-design.md)  
> **Run JSON:** [../e2e/runs/e2e_mistral_medium_3_5_20260705_203803.json](../e2e/runs/e2e_mistral_medium_3_5_20260705_203803.json)

Benchmark de tres brazos sobre `data/e2e_prompts.json` (4 prompts con payload adjunto):

| Brazo | Instrucción | `output_style` | Qué mide |
|-------|-------------|----------------|----------|
| **A — baseline** | Natural, sin PCM | `normal` (system hint PCM solo) | Techo de referencia |
| **B — pcm** | Comprimida (`granite4.1:3b`) | `normal` | Ahorro en ida (input) |
| **C — pcm+concise** | Comprimida | `concise` (`RESPONSE:` en system) | Ahorro en ida + vuelta (output) |

**Compresor:** `granite4.1:3b` (Ollama)  
**LLM destino:** `mistral-medium-3.5`  
**Reasoning:** `none` (obligatorio para medir output tokens con fidelidad)  
**Precios Mistral (API):** $1.50 / M input · $7.50 / M output

## Comando de reproducción

```bash
cd Prompt\ Compression\ Middleware
source .env   # MISTRAL_API_KEY
PYTHONPATH=src .venv/bin/python scripts/e2e_benchmark.py --reasoning-effort none -q
```

## Resultados agregados (2026-07-05)

| Métrica | Baseline (A) | PCM normal (B) | PCM concise (C) |
|---------|--------------|----------------|-----------------|
| Input tokens (total) | 1 233 | 1 207 (−26) | ~1 280* |
| Output tokens (media) | 461 | 371 | **150** |
| **Coste total (4 prompts)** | **$0.0157** | **$0.0129** (−18%) | **$0.0069** (−56%) |
| Similitud respuesta B vs A | — | **94.50%** | — |
| Ahorro output C vs B | — | — | **42.6%** |
| Δ coste C vs A (media/prompt) | — | — | **−$0.00219** |

\*El brazo concise añade el bloque `RESPONSE:` al system (~+20–30 tokens de input por prompt). El ahorro neto sigue siendo dominante en output.

### Interpretación

1. **PCM en instrucción** reduce input y coste total (~18% vs baseline) sin perder calidad (94.5% similitud).
2. **`output_style=concise`** reduce fuertemente tokens de salida (~43% vs PCM normal) al pedir al modelo respuestas mínimas en el prompt de ida.
3. **La métrica a optimizar es `cost_total`** (input + output con precios distintos), no solo tokens de entrada.
4. Con **`reasoning=high`**, Mistral infla `completion_tokens` con razonamiento interno aunque el texto visible sea corto — **no usar para benchmark de output directives** (ver § abajo).

## Detalle por prompt

| ID | Categoría | Out baseline | Out PCM | Out concise | $ baseline | $ PCM | $ concise |
|----|-----------|-------------|---------|-------------|------------|-------|-----------|
| e2e_001 | code_review | 747 | 629 | **138** | 0.00610 | 0.00520 | **0.00167** |
| e2e_002 | code_review | 805 | 601 | **269** | 0.00654 | 0.00500 | **0.00266** |
| e2e_003 | translation | 61 | 60 | 55 | 0.00081 | 0.00080 | 0.00091 |
| e2e_004 | summarization | 231 | 192 | **137** | 0.00222 | 0.00192 | **0.00166** |

## Nota: `reasoning=high` distorsiona output tokens

Run de control (1 prompt, `reasoning=high`):

| Brazo | Output tokens reportados | Coste | Texto visible |
|-------|------------------------|-------|---------------|
| concise | **2 837** | $0.0219 | 3 viñetas (~40 palabras) |

Mismo prompt con `reasoning=none`: **138** output tokens, **$0.00167**.

**Conclusión:** benchmarks de `output_style` deben ejecutarse con `--reasoning-effort none`. El script solo admite `high` o `none` (no `auto`).

## Umbrales spec v0 (orientativos)

| KPI | Umbral | Resultado |
|-----|--------|-----------|
| `cost_delta` pcm+concise vs baseline | ≤ −15% coste total medio | **−56%** ✅ |
| Comprensión E2E (similitud) | ≥ 0.90 | **0.945** ✅ |
| `output_tokens` concise vs normal | ≤ −20% medio | **−42.6%** ✅ |

## Integración

| Componente | Parámetro |
|------------|-----------|
| Proxy | `PCM_OUTPUT_STYLE=concise` |
| COE compose | `output_style="concise"` en `build_pcm_messages` / `optimize_with_pcm` |
| Headers proxy | `X-PCM-Input-Tokens`, `X-PCM-Output-Tokens`, `X-PCM-Cost-Total-USD` |

Ver también [docs/pcm-and-coe.md](../../docs/pcm-and-coe.md#e2e-output-directives-july-2026).
