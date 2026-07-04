# Benchmark E2E PCM → Mistral

- **Generado:** 2026-07-04T05:51:13.021807+00:00
- **Compresor:** `granite4.1:3b`
- **Modelo destino:** `mistral-medium-3.5`
- **Reasoning effort:** `none`

## Resumen

| Métrica | Valor |
|---------|-------|
| Prompts | 4 |
| Ratio compresión (media) | 47.13% |
| Tiempo compresión (media) | 359 ms |
| Tiempo Mistral original (media) | 4376 ms |
| Tiempo Mistral comprimido (media) | 3869 ms |
| Similitud respuestas (media) | 93.25% |
| Tokens input original (total) | 1233 |
| Tokens input comprimido (total) | 1207 |
| Tokens input ahorrados | 26 |
| Coste original (total) | $0.0200 |
| Coste comprimido (total) | $0.0119 |

## Detalle por prompt

| ID | Categoría | Payload | Compresión | Similitud | Truncado | t_comp | t_llm_orig | t_llm_pcm |
|----|-----------|---------|------------|-----------|----------|--------|------------|-----------|
| e2e_001 | code_review | 541 ch | 50% | 90% | no | 419 ms | 8980 ms | 4913 ms |
| e2e_002 | code_review | 529 ch | 48% | 90% | no | 367 ms | 6019 ms | 5970 ms |
| e2e_003 | translation | 393 ch | 42% | 98% | no | 305 ms | 718 ms | 804 ms |
| e2e_004 | summarization | 474 ch | 48% | 95% | no | 343 ms | 1786 ms | 3790 ms |
