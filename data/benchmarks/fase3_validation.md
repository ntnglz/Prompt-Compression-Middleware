# Validación fine-tune PCM (holdout + E2E)

Generado: 2026-07-04T05:52:33.916651+00:00

## Holdout (`valid.jsonl`, sin overlap con gold benchmark)

Prompts: **10**

| Modelo | Ratio | Formato | Semántica |
|--------|-------|---------|-----------|
| granite4.1:3b | 19.77% | 57.75% | 94.00% |
| pcm-compressor | 21.02% | 60.00% | 92.00% |

- Salidas idénticas granite vs pcm: **8/10**
- pcm gana en formato: **2**
- pcm gana en ratio: **2**

### Diferencias de salida

- **holdout_000**: granite `TASK=review INPUT=typescript CHECK=logic_errors,anti_patterns,perf FORMAT=list` | pcm `TASK=review INPUT=typescript CHECK=logic,anti_patterns,perf FORMAT=list`
- **holdout_002**: granite `TASK=translate INPUT=document STYLE=formal DOMAIN=messaging FROM=en TO=es` | pcm `TASK=translate FROM=en TO=es STYLE=formal DOMAIN=marketing`

## E2E Mistral (`e2e_prompts.json`)

| Compresor | Similitud respuesta | Ratio compresión | Tokens ahorrados |
|-----------|---------------------|------------------|------------------|
| granite4.1:3b | 93.25% | 47.13% | 26 |
| pcm-compressor | 92.00% | 47.13% | 26 |
