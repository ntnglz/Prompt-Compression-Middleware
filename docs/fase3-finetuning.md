# Fase 3 — Fine-tuning del compresor PCM (Mac local)

Guía operativa para entrenar un compresor PCM especializado con LoRA/MLX, exportarlo a Ollama como `pcm-compressor` e integrarlo en el stack existente.

**Hardware de referencia:** Mac M4 Pro, 64 GB RAM (entrenamiento ~1–3 h).

---

## 1. Requisitos

| Requisito | Detalle |
|-----------|---------|
| **Mac Apple Silicon** | M1/M2/M3/M4 (MLX requiere chip Apple) |
| **RAM** | 32 GB mínimo; **64 GB recomendado** para evitar OOM |
| **Ollama** | Instalado y en ejecución (`ollama serve` o app Ollama) |
| **Python venv** | Entorno virtual del proyecto activado |
| **Baseline** | `ollama pull granite4.1:3b` (comparativa A/B) |

```bash
cd Prompt-Compression-Middleware
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Dependencias de entrenamiento

Instalar solo en la Mac donde se entrenará (MLX no es necesario en CI ni en Docker):

```bash
pip install -r requirements-training.txt
```

Incluye `mlx-lm` y `mlx` para LoRA sobre `mlx-community/Qwen2.5-3B-Instruct-4bit`.

La configuración LoRA (`rank`, `mask_prompt`, etc.) está en `data/training/lora_config.yaml` y es consumida por `scripts/train_compressor.py` vía `mlx_lm lora -c`.

---

## 3. Generar dataset

```bash
python scripts/generate_dataset.py
```

**Salida esperada:**

- `data/training/train.jsonl` — ~90% ejemplos
- `data/training/valid.jsonl` — ~10% holdout
- `data/training/manifest.json` — metadatos del split

Verificar que `manifest.json` muestre **`total` ≥ 80** (objetivo de la spec: ~100 pares; el generador actual produce ~83):

```bash
cat data/training/manifest.json
# Ejemplo: {"total": 83, "train": 73, "valid": 10, ...}
```

---

## 4. Entrenar LoRA

```bash
python scripts/train_compressor.py
```

**Duración estimada:** 1–3 horas en M4 Pro 64 GB (3 épocas, batch efectivo 8).

El script ejecuta:

1. `mlx_lm.lora` → `data/training/checkpoints/pcm-lora/`
2. `mlx_lm.fuse` → `data/training/checkpoints/pcm-fused/`

Opciones útiles: `--epochs`, `--batch-size`, `--fuse-only` (solo fusión si LoRA ya existe).

---

## 5. Exportar a Ollama

```bash
./scripts/export_ollama.sh
```

Pasos del script:

1. Convierte pesos fusionados a GGUF (`data/training/checkpoints/pcm-compressor.gguf`)
2. Crea el modelo Ollama `pcm-compressor` desde `data/training/Modelfile`
3. Smoke test con `ollama run pcm-compressor`

Confirmar:

```bash
ollama list | grep pcm-compressor
```

---

## 6. Benchmark comparativo (baseline vs fine-tuned)

Requiere Ollama con `granite4.1:3b` y `pcm-compressor` disponibles:

```bash
python scripts/compare_finetune.py --semantic
```

Genera informes en `data/benchmarks/` y un resumen en `data/benchmarks/fase3_comparison.md` con ratio, formato y similitud semántica para ambos modelos.

---

## 7. Benchmark E2E con Mistral

Configurar el compresor fine-tuned y ejecutar el benchmark end-to-end (requiere `MISTRAL_API_KEY` en `.env`):

```bash
OLLAMA_MODEL=pcm-compressor python scripts/e2e_benchmark.py
```

O de forma persistente en `.env`:

```bash
# OLLAMA_MODEL=pcm-compressor
```

---

## 8. Criterios de éxito

| Métrica | Umbral | Script |
|---------|--------|--------|
| **Ratio de compresión** | >50% media | `compare_finetune.py` / `benchmark.py` |
| **Similitud de formato (campos PCM)** | ≥85% (≥0.85) | `compare_finetune.py` |
| **Similitud semántica** | ≥85% (≥0.85) | `compare_finetune.py --semantic` |
| **E2E Mistral** | ≥90% (≥0.90) | `e2e_benchmark.py` con `OLLAMA_MODEL=pcm-compressor` |

**Baseline de referencia (`granite4.1:3b`):** ~40% ratio, ~90% semántica, ~93% E2E.

El modelo fine-tuned debe **superar el ratio del baseline** (>50%) manteniendo formato y calidad semántica dentro de los umbrales.

---

## 9. Integración en runtime

Tras validar métricas:

1. Actualizar `.env`: `OLLAMA_MODEL=pcm-compressor`
2. Reiniciar proxy o contenedor Docker
3. Opcional: `./scripts/ci-local-full.sh` (requiere Ollama + modelo nuevo)

No se modifican `proxy.py`, MCP ni el protocolo PCM: solo cambia el modelo del compresor vía variable de entorno.

---

## 10. Estructura de artefactos

```
data/training/
  train.jsonl, valid.jsonl, manifest.json   # generados
  Modelfile                                   # plantilla Ollama
  checkpoints/                                # gitignored
    pcm-lora/
    pcm-fused/
    pcm-compressor.gguf
data/benchmarks/
  fase3_comparison.md                         # comparativa A/B
```

---

## 11. Troubleshooting

### `AttributeError: 'str' object has no attribute '__module__'` al importar mlx-lm

Incompatibilidad entre `mlx-lm` y `transformers` 5.x. Reinstala con el pin del proyecto:

```bash
pip install 'transformers>=4.43,<5.0' -r requirements-training.txt
```

### `unrecognized arguments: --lora-rank` o `--val-data`

`mlx_lm` 0.31+ cambió la CLI: el rank LoRA va en `data/training/lora_config.yaml` y `valid.jsonl` se lee automáticamente del directorio `--data`. Actualiza el repo y usa el `train_compressor.py` más reciente.

---

## Referencias

- Spec: `docs/superpowers/specs/2026-07-03-fase3-finetuning-design.md`
- Plan de implementación: `docs/superpowers/plans/2026-07-03-fase3-finetuning.md`
- Glosario PCM: `src/pcm/compressor.py` (`COMPRESSION_SYSTEM_PROMPT`)
