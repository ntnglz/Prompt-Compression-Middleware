# Benchmark E2E PCM → Mistral

- **Generado:** 2026-07-04T05:51:52.824423+00:00
- **Compresor:** `pcm-compressor`
- **Modelo destino:** `mistral-medium-3.5`
- **Reasoning effort:** `none`

## Resumen

| Métrica | Valor |
|---------|-------|
| Prompts | 4 |
| Ratio compresión (media) | 47.13% |
| Tiempo compresión (media) | 632 ms |
| Tiempo Mistral original (media) | 5258 ms |
| Tiempo Mistral comprimido (media) | 3400 ms |
| Similitud respuestas (media) | 92.00% |
| Tokens input original (total) | 1233 |
| Tokens input comprimido (total) | 1207 |
| Tokens input ahorrados | 26 |
| Coste original (total) | $0.0143 |
| Coste comprimido (total) | $0.0106 |

## Detalle por prompt

| ID | Categoría | Payload | Compresión | Similitud | Truncado | t_comp | t_llm_orig | t_llm_pcm |
|----|-----------|---------|------------|-----------|----------|--------|------------|-----------|
| e2e_001 | code_review | 541 ch | 50% | 90% | no | 734 ms | 4258 ms | 7815 ms |
| e2e_002 | code_review | 529 ch | 48% | 90% | no | 641 ms | 11883 ms | 3684 ms |
| e2e_003 | translation | 393 ch | 42% | 98% | no | 542 ms | 1223 ms | 631 ms |
| e2e_004 | summarization | 474 ch | 48% | 90% | no | 614 ms | 3666 ms | 1471 ms |
