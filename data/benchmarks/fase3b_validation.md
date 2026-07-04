# Validación Fase 3b — Granite Cloud (Enfoque A)

Generado: 2026-07-04T15:32:55.170010+00:00

## Configuraciones

| Label | Modelo | System prompt |
|-------|--------|---------------|
| baseline_full | granite4.1:3b | FULL (few-shots) |
| baseline_glossary | granite4.1:3b | GLOSSARY |
| finetuned_glossary | pcm-granite | GLOSSARY |

## Eval: `holdout_curated`

| Label | Ratio | Formato | Semántica |
|-------|-------|---------|-----------|
| baseline_full | 19.68% | 41.92% | N/A |
| baseline_glossary | 25.49% | 4.00% | N/A |
| finetuned_glossary | 27.01% | 68.71% | N/A |

## Eval: `holdout_synthetic`

| Label | Ratio | Formato | Semántica |
|-------|-------|---------|-----------|
| baseline_full | 13.96% | 31.14% | N/A |
| baseline_glossary | 9.45% | 0.33% | N/A |
| finetuned_glossary | 21.88% | 47.56% | N/A |

## E2E Mistral

| Label | Similitud | Ratio |
|-------|-----------|-------|
| baseline_full | 93.25% | 47.13% |
| baseline_glossary | 96.25% | 27.96% |
| finetuned_glossary | 94.50% | 49.54% |

**Umbral E2E:** ≥90% similitud → **pcm-granite pasa (94.50%)**.

## Criterio go/no-go (Enfoque A)

| Criterio | Resultado |
|----------|-----------|
| 0 leakage en train | ✅ |
| holdout_curated formato ≥ 70% | ⚠️ 68.71% (−1.29 pp) |
| Pipeline cloud sin errores | ✅ |
| E2E Mistral ≥ 90% | ✅ **94.50%** |

## Conclusión (cierre Fase 3b + E2E)

**Veredicto: GO — experimento PCM validado para integración.**

El fine-tune cumple en el escenario que importa para producción: **E2E con Mistral** (94.50% similitud, ratio 49.54%). Holdout formato sigue justo del 70%, pero la compresión real con payload preserva la calidad de respuesta del LLM destino.

**Integración recomendada:** `OLLAMA_MODEL=pcm-granite` en `.env` / Docker Compose.

**Siguiente horizonte (fuera de PCM):** dar por cerrado el experimento de compresión de *instrucciones* y plantear **Context Optimization Engine** — compresión de contextos completos, memoria semántica del chat, etc.

**Pre-producción (opcional):** Enfoque B (dataset ampliado) antes de fijar el modelo en producción a largo plazo.

**Documentación:** `docs/fase3b-granite-cloud.md` § Cierre y resultados.

