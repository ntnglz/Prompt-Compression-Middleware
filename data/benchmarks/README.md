# Benchmarks PCM

Resultados de comparación de modelos compresores y validación post fine-tuning.

## Informes principales

| Archivo | Descripción |
|---------|-------------|
| [output_directives_e2e.md](output_directives_e2e.md) | **Output directives** — E2E 3 brazos (baseline / PCM / concise), `reasoning=none` |
| [fase3b_validation.md](fase3b_validation.md) | **Informe final** — holdout + E2E Mistral (3b) |
| [fase3_comparison.md](fase3_comparison.md) | Comparativa 3a MLX (`pcm-compressor` vs granite) |
| [fase3_validation.md](fase3_validation.md) | Validación 3a |

## Estructura

```
data/benchmarks/
├── fase3b_validation.md          # Resumen ejecutivo Fase 3b
├── fase3b_results_*.json         # Resultados agregados JSON
├── fase3b_*_holdout_*/           # Runs detallados por config/eval set
├── index.json                    # Índice (benchmarks genéricos)
└── runs/                         # Benchmarks por modelo (gitignored)
```

## Comandos

```bash
# Benchmark compresor genérico
python scripts/benchmark.py --model granite4.1:3b --semantic -q
python scripts/benchmark.py --model pcm-granite --semantic -q

# Validación completa Fase 3b (holdout + E2E)
python scripts/validate_granite_v2.py --semantic --e2e

# Comparativa 3a MLX
python scripts/compare_finetune.py --semantic

# E2E standalone (usar reasoning=none para métricas de output_style)
python scripts/e2e_benchmark.py --reasoning-effort none -q
```

## Resultados Fase 3b (referencia)

| Métrica | pcm-granite |
|---------|-------------|
| E2E similitud Mistral | **94.50%** |
| E2E ratio compresión | **49.54%** |
| Holdout formato (glossary) | 69.12% |
| Leakage train/eval | 0 |

Ver [docs/experimento-pcm-conclusiones.md](../../docs/experimento-pcm-conclusiones.md).
