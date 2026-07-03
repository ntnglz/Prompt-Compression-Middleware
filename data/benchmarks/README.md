# Benchmarks PCM

Resultados de comparación de modelos compresores.

## Estructura

```
data/benchmarks/
├── index.json              # Índice de todas las ejecuciones
├── leaderboard.md          # Ranking comparativo (auto-generado)
└── runs/
    ├── qwen3_4b/
    │   ├── qwen3_4b_YYYYMMDD_HHMMSS.json
    │   ├── qwen3_4b_YYYYMMDD_HHMMSS.md
    │   ├── qwen3_4b_latest.json    # última ejecución de este modelo
    │   └── qwen3_4b_latest.md
    ├── qwen3_1.7b/
    └── gemma3_4b/
```

## Comandos

```bash
# Ejecutar benchmark (guarda automáticamente en runs/{modelo}/)
python3 scripts/benchmark.py --model granite4.1:3b --semantic -q

# Reconstruir índice desde archivos existentes
python3 scripts/benchmark.py --rebuild-index
```

## Ronda de modelos sugerida

```bash
for model in qwen3:4b qwen3:1.7b gemma3:4b granite4.1:3b; do
  python3 scripts/benchmark.py --model "$model" --semantic -q
done
```

Tras cada ejecución, consulta `leaderboard.md` para comparar.
