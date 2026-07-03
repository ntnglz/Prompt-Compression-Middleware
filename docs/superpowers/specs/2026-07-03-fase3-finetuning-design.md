# Fase 3 — Fine-tuning del compresor PCM (Enfoque A)

> **Estado:** Aprobado  
> **Fecha:** 2026-07-03  
> **Hardware objetivo:** Mac M4 Pro, 64 GB RAM (entrenamiento local)  
> **Enfoque:** LoRA con MLX → export GGUF → Ollama `pcm-compressor`

---

## 1. Objetivo

Entrenar un compresor PCM especializado que supere al baseline `granite4.1:3b` en ratio de compresión (>50%) manteniendo calidad semántica (≥85% benchmark, ≥90% E2E Mistral).

El modelo resultante se integra en el stack existente cambiando solo `OLLAMA_MODEL=pcm-compressor`; el proxy, MCP y benchmarks no requieren cambios estructurales.

---

## 2. Contexto del sistema actual

| Componente | Rol |
|------------|-----|
| `src/pcm/compressor.py` | Compresor vía Ollama; `COMPRESSION_SYSTEM_PROMPT` define el glosario PCM |
| `data/example_prompts.json` | 10 pares gold (texto → PCM esperado) |
| `scripts/benchmark.py` | Benchmark ratio + similitud de formato/campos |
| `scripts/e2e_benchmark.py` | Validación E2E con Mistral upstream |
| `src/pcm/proxy.py` | Proxy producción; usa el mismo compresor |

**Baseline actual (granite4.1:3b):**
- Ratio compresión: ~40%
- Similitud semántica benchmark: ~90%
- E2E Mistral: ~93%

---

## 3. Decisiones de diseño

### 3.1 Modelo base para fine-tuning

**Elección:** `mlx-community/Qwen2.5-3B-Instruct-4bit`

| Criterio | Qwen2.5-3B-Instruct | granite4.1:3b |
|----------|---------------------|---------------|
| Pesos MLX oficiales | ✅ mlx-community | ❌ sin port estable |
| Tamaño en 64 GB | ~2 GB (4-bit) | N/A para MLX |
| Calidad instrucción | Alta | Alta |
| Integración Ollama | GGUF post-fusión | Ya en Ollama |

> **Nota:** El baseline de comparación sigue siendo `granite4.1:3b`. El modelo entrenado se publica como `pcm-compressor` en Ollama. Si en el futuro hay pesos MLX de Granite 4, se puede repetir el pipeline.

### 3.2 Método de entrenamiento

- **LoRA** (no full fine-tuning): rank 16, alpha 32, dropout 0.05
- **Framework:** `mlx-lm` (Apple MLX, optimizado para Apple Silicon)
- **Épocas:** 3 (ajustable según overfitting en holdout)
- **Learning rate:** 1e-5
- **Batch size:** 2 (gradient accumulation 4 → batch efectivo 8)
- **Tiempo estimado:** 1–3 h para ~100 pares en M4 Pro 64 GB

### 3.3 Dataset

**Fase inicial:** ~100 pares (10 gold + ~90 sintéticos)

| Fuente | Cantidad | Método |
|--------|----------|--------|
| `example_prompts.json` | 10 | Conversión directa a JSONL chat |
| Plantillas por categoría | ~60 | Variaciones ES/EN de 10 categorías |
| Teacher granite (opcional) | ~30 | Paráfrasis de prompts + validación de campos PCM |

**Formato JSONL (mlx-lm chat):**
```json
{
  "messages": [
    {"role": "system", "content": "<COMPRESSION_SYSTEM_PROMPT>"},
    {"role": "user", "content": "Prompt a comprimir:\n<texto natural>"},
    {"role": "assistant", "content": "TASK=review INPUT=python CHECK=race,leak,perf FORMAT=markdown ORDER=severity"}
  ]
}
```

**Split:** 90% train (`train.jsonl`), 10% valid (`valid.jsonl`) — estratificado por categoría.

**Criterios de calidad del dataset:**
- Toda salida `assistant` empieza por `TASK=`
- Solo claves del glosario PCM (`compressor.COMPRESSION_SYSTEM_PROMPT`)
- Una sola línea por salida
- Sin duplicados exactos de `user` content

### 3.4 Pipeline de exportación a Ollama

```
train.jsonl
    ↓ mlx_lm.lora
checkpoints/pcm-lora/
    ↓ mlx_lm.fuse
checkpoints/pcm-fused/
    ↓ mlx_lm.convert (q4_k_m)
checkpoints/pcm-compressor.gguf
    ↓ ollama create -f Modelfile
pcm-compressor (Ollama local)
```

**Modelfile:**
```
FROM ./checkpoints/pcm-compressor.gguf
PARAMETER temperature 0.1
PARAMETER num_predict 256
```

### 3.5 Evaluación

| Métrica | Script | Umbral éxito |
|---------|--------|--------------|
| Ratio compresión | `scripts/benchmark.py` | >50% media |
| Similitud campos PCM | `scripts/benchmark.py` | ≥0.85 score medio |
| Similitud semántica | `scripts/benchmark.py --semantic` | ≥0.85 |
| E2E Mistral | `scripts/e2e_benchmark.py` | ≥0.90 |
| Latencia inferencia | benchmark `processing_time_ms` | <1000 ms |

**Comparativa obligatoria:** baseline `granite4.1:3b` vs `pcm-compressor` en el mismo run; informe en `data/benchmarks/fase3_comparison.md`.

---

## 4. Estructura de archivos nueva

```
data/training/
  templates.json           # Plantillas por categoría para sintéticos
  train.jsonl              # Dataset entrenamiento (generado)
  valid.jsonl              # Holdout (generado)
  manifest.json            # Metadatos: conteos, split, versión

src/pcm/training/
  __init__.py
  dataset.py               # Conversión prompts → JSONL, validación, split

scripts/
  generate_dataset.py      # CLI: gold + sintéticos → train/valid
  train_compressor.py      # Wrapper mlx_lm.lora + fuse
  export_ollama.sh           # convert GGUF + ollama create
  compare_finetune.py        # Benchmark A/B granite vs pcm-compressor

data/training/checkpoints/   # Gitignored; pesos locales
  pcm-lora/
  pcm-fused/
  pcm-compressor.gguf

requirements-training.txt    # mlx-lm, mlx; separado del runtime PCM

docs/
  fase3-finetuning.md        # Guía operativa para el usuario
```

---

## 5. Integración con el runtime existente

1. Tras exportar: `ollama run pcm-compressor` (smoke test manual)
2. Actualizar `.env`: `OLLAMA_MODEL=pcm-compressor`
3. Re-ejecutar `./scripts/ci-local-full.sh` (requiere Ollama + modelo nuevo)
4. Proxy y MCP heredan el modelo vía `CompressorConfig.model`

No se modifica `COMPRESSION_SYSTEM_PROMPT` en runtime: el fine-tune internaliza el formato; el system prompt se mantiene para consistencia y fallback.

---

## 6. Criterios de aceptación (Fase 3)

- [ ] `data/training/train.jsonl` con ≥100 ejemplos válidos
- [ ] Tests unitarios de `dataset.py` pasan en CI local (`./scripts/ci-local.sh`)
- [ ] Entrenamiento completa sin OOM en M4 Pro 64 GB
- [ ] Modelo `pcm-compressor` disponible en Ollama local
- [ ] Ratio medio >50% en `example_prompts.json`
- [ ] Score formato ≥0.85 y semántica ≥0.85 con `--semantic`
- [ ] E2E ≥0.90 con Mistral
- [ ] Documentación `docs/fase3-finetuning.md` con pasos reproducibles

---

## 7. Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Overfitting en 100 pares | Holdout 10%, early stopping por val loss |
| Base Qwen ≠ baseline Granite | Comparar A/B; si no mejora, ampliar dataset antes de más épocas |
| Export GGUF incompatible | Probar `ollama run` antes de integrar proxy |
| MLX no instalado | `requirements-training.txt` + verificación en `train_compressor.py` |
| Teacher genera PCM inválido | Validador de claves en `dataset.py`; descartar ejemplos inválidos |

---

## 8. Fuera de alcance (Fase 3)

- Reinforcement Learning (Fase 3 avanzada del doc de viabilidad)
- Entrenamiento en cloud / multi-GPU
- Full fine-tuning de modelos >7B
- GitHub Actions para entrenamiento
- Cambios al protocolo PCM o glosario de claves

---

## 9. Referencias

- `src/pcm/compressor.py` — `COMPRESSION_SYSTEM_PROMPT`, glosario PCM
- `data/example_prompts.json` — dataset gold
- `Analisis de viabilidad y plan de implementacion.md` — Opción B (LoRA 4B)
- [mlx-lm LoRA](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- [Ollama Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
