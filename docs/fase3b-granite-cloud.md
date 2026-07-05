# Fase 3b — Fine-tune Granite en Cloud (Enfoque A conservador)

> **Maintainers (ES)** — Guía operativa de fine-tuning. Visitantes: [README en inglés](../README.md).

> **Estado: CERRADA** (2026-07-04) — Pipeline cloud validado; modelo `pcm-granite` publicado en Ollama. Ver [Cierre y resultados](#cierre-y-resultados-2026-07-04).

Guía paso a paso para quien nunca ha usado RunPod ni fine-tuning en cloud.

**Objetivo:** entrenar granite con ~400 pares (plantillas v2) y publicar `pcm-granite` en Ollama.

**Presupuesto:** ~2 USD por experimento (RTX 4090 spot, ~30 min). **Coste real:** ~0.18 USD.

---

## 1. Preparación local (Mac)

```bash
cd "/Volumes/DevSSD/XcodeProjects/Ideas/Prompt Compression Middleware"
source .venv/bin/activate

# Generar eval synthetic + manifest de exclusión
python scripts/build_dataset_v2.py --eval-output data/eval/holdout_synthetic.json
python scripts/update_excluded_manifest.py

# Dataset train/valid (sin teacher = rápido; con teacher = labels granite)
python scripts/build_dataset_v2.py
python scripts/check_dataset_leakage.py   # debe imprimir OK: 0 leakage
```

Opcional (más lento, requiere Ollama con granite):

```bash
python scripts/build_dataset_v2.py --with-teacher
```

Archivos generados (gitignored):

- `data/training/v2/train.jsonl` (~360 ejemplos)
- `data/training/v2/valid.jsonl` (~40 ejemplos)

---

## 2. Crear cuenta RunPod

1. Ir a [runpod.io](https://www.runpod.io) y registrarse.
2. Añadir **10 USD** de crédito (Settings → Billing).
3. Crear un **Network Volume** de 20 GB (Storage → Network Volumes).

---

## 3. Subir dataset al volume

1. Crear un pod **CPU** barato con el volume montado en `/workspace`.
2. Subir vía `scp` o el file browser de RunPod:

```
data/training/v2/train.jsonl
data/training/v2/valid.jsonl
notebooks/train_granite_unsloth.ipynb
```

---

## 4. Entrenar en GPU

1. Crear pod **GPU** → RTX 3090/A5000 On-Demand o 4090 spot.
2. Template: **Runpod PyTorch 2.8+** con CUDA.
3. Volume montado en `/workspace` (puede crearse vacío).
4. **Connect → Jupyter Lab** (SSH no necesario).
5. Subir `train.jsonl`, `valid.jsonl`, notebook.
6. Ejecutar notebook (celda pip → imports → train). Ver [Lecciones aprendidas](#lecciones-aprendidas-2026-07-04).
7. Descargar `granite-lora/` y **Stop Pod** inmediatamente.

---

## 5. Exportar a Ollama (Mac)

### 5.1 Descargar modelo base (HuggingFace)

El adapter LoRA no basta; hay que fusionar con `ibm-granite/granite-3.3-2b-instruct` (~5 GB).

**Importante: desactivar Xet** — en Mac la descarga con Xet se queda al 1% durante horas sin progreso real.

```bash
cd "/Volumes/DevSSD/XcodeProjects/Ideas/Prompt Compression Middleware"
source .venv/bin/activate
pip install peft transformers accelerate

export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
hf download ibm-granite/granite-3.3-2b-instruct --max-workers 8
```

Comprobar que existen **2 shards** (la barra de progreso puede no refrescarse; mirar tamaño en disco):

```bash
ls -lh ~/.cache/huggingface/hub/models--ibm-granite--granite-3.3-2b-instruct/snapshots/*/model*.safetensors
```

Si la descarga se atasca: ver [Lecciones aprendidas](#lecciones-aprendidas-2026-07-04).

### 5.2 Fusionar adapter + base

Coloca el adapter descargado de RunPod en `data/training/v2/checkpoints/granite-lora/` y ejecuta:

```bash
python scripts/merge_granite_lora.py
```

Salida esperada: `data/training/v2/checkpoints/granite-merged/` (~5 GB).

### 5.3 Crear modelo Ollama

Ollama **no importa safetensors** con arquitectura `GraniteForCausalLM`. El script convierte primero a GGUF (`pcm-granite-f16.gguf`, ~4.7 GB) y luego ejecuta `ollama create`.

```bash
chmod +x scripts/export_granite_ollama.sh
./scripts/export_granite_ollama.sh
ollama list | grep pcm-granite   # esperado: ~5.1 GB
```

Requisito: `convert_hf_to_gguf.py` de llama.cpp (Homebrew: `brew install llama.cpp`) y `pip install gguf`.

---

## 6. Validar

```bash
python scripts/validate_granite_v2.py --semantic
python scripts/validate_granite_v2.py --semantic --e2e   # requiere MISTRAL_API_KEY
```

Informe: `data/benchmarks/fase3b_validation.md`

---

## Criterio go/no-go (Enfoque A)

| Criterio | Umbral |
|----------|--------|
| Leakage train/eval | 0 |
| Holdout formato PCM | ≥ 70% |
| Pipeline cloud | Completado sin errores |
| Mejora vs baseline | Bonus (no bloqueante) |

Si pasa pero el ratio apenas mejora → escalar al **Enfoque B** (más datos, corpus real).

---

## Cierre y resultados (2026-07-04)

### Veredicto: **GO — experimento PCM validado**

El objetivo del Enfoque A era demostrar que el flujo RunPod → merge → Ollama → eval funciona de punta a punta. **E2E Mistral confirma que el fine-tune es usable en producción.**

| Criterio | Resultado | Estado |
|----------|-----------|--------|
| Leakage train/eval | 0 (367 train + 35 valid, `check_dataset_leakage.py`) | ✅ |
| Pipeline cloud | RunPod → adapter → merge → GGUF → `pcm-granite` | ✅ |
| Holdout curated formato | **69.12%** (umbral ≥ 70%) | ⚠️ −0.88 pp |
| Mejora vs baseline glossary | Formato 5.67% → **69.12%**; semántica 66.5% → **88.0%** | ✅ |
| **E2E Mistral** | **94.50%** similitud, **49.54%** ratio | ✅ |

El umbral de formato holdout queda justo del 70%, pero E2E ≥90% desbloquea la integración. **No se requiere repetir RunPod** para usar `pcm-granite` en el proxy.

### Entrenamiento (RunPod)

| Parámetro | Valor |
|-----------|-------|
| GPU | RTX 3090 On-Demand ($0.46/h) |
| Duración | ~25 min |
| Coste cloud | **~$0.18** |
| Base HF | `ibm-granite/granite-3.3-2b-instruct` |
| Dataset | 367 train / 35 valid (plantillas v2, glossary) |
| Epochs / steps | 2 / 92 |
| Loss final | ~0.135 → ~0.017 |
| Adapter | `data/training/v2/checkpoints/granite-lora/` |

### Validación (Mac, 2026-07-04)

Benchmark A/B/C con `scripts/validate_granite_v2.py --semantic`. Informe: `data/benchmarks/fase3b_validation.md`.

**holdout_curated** (20 prompts):

| Config | Ratio | Formato | Semántica |
|--------|-------|---------|-----------|
| baseline_full (`granite4.1:3b`) | 19.43% | 46.58% | 93.50% |
| baseline_glossary | 29.84% | 5.67% | 66.50% |
| **finetuned_glossary (`pcm-granite`)** | **26.51%** | **69.12%** | **88.00%** |

**holdout_synthetic** (30 prompts): formato **51.31%**, semántica **86.33%**, ratio **22.46%**.

**Nota metodológica:** el baseline compara contra `granite4.1:3b` (Ollama) mientras el fine-tune partió de `granite-3.3-2b-instruct` (HF). Documentar este delta al interpretar ratios.

**E2E Mistral** (`validate_granite_v2.py --e2e`, 4 prompts con payload):

| Config | Similitud | Ratio E2E |
|--------|-----------|-----------|
| baseline_full | 93.25% | 47.13% |
| baseline_glossary | 96.25% | 27.96% |
| **pcm-granite** | **94.50%** | **49.54%** |

### Integración

```bash
# .env
OLLAMA_MODEL=pcm-granite

python run.py --proxy
```

### Artefactos finales

```
data/training/v2/checkpoints/
  granite-lora/          # adapter RunPod (gitignored)
  granite-merged/        # pesos fusionados HF (~5 GB, gitignored)
  pcm-granite-f16.gguf   # export GGUF (~4.7 GB, gitignored)
  Modelfile              # FROM ./pcm-granite-f16.gguf

Ollama: pcm-granite:latest (~5.1 GB)
```

### Próximos pasos (post-E2E)

| Fase | Estado | Acción |
|------|--------|--------|
| Pipeline cloud + Ollama | ✅ | — |
| E2E Mistral | ✅ 94.50% | — |
| **Integración proxy** | ✅ | `OLLAMA_MODEL=pcm-granite` verificado |
| Enfoque B | Opcional | Pre-producción a largo plazo (dataset 1k–1.5k pares) |
| **COE** | Horizonte | Compresión de contextos completos, BD semántica del chat |

**Cierre del experimento PCM:** con E2E validado, el middleware de compresión de *instrucciones* puede darse por demostrado. El siguiente salto ambicioso es **[Context Optimization Engine](https://github.com/ntnglz/Context-Optimization-Engine)** — compresión de contextos completos, memoria semántica del chat, etc.

---

## Costes típicos

| Paso | Coste estimado | Coste real (1.er experimento) |
|------|----------------|-------------------------------|
| Pod GPU RTX 3090 On-Demand ~25 min | ~0.15–0.35 USD | **~0.18 USD** |
| Pod CPU (subir archivos) | ~0.05 USD | (omitido: Jupyter directo) |
| Reintentos / debugging | +0.50–1.50 USD | 0 USD |
| **Total cloud** | **~2 USD** (tope) | **~0.18 USD** |

Tope recomendado: **10 USD** antes de replantear el enfoque.

---

## Lecciones aprendidas (2026-07-04)

Registro operativo del primer experimento Enfoque A. Actualizar si cambian versiones de herramientas.

### RunPod / cloud

| Lección | Detalle |
|---------|---------|
| **GPU suficiente y barata** | RTX 3090 On-Demand ($0.46/h) o A5000 ($0.27/h) bastan para granite 3B QLoRA. No hace falta 4090/H100. |
| **Coste real << estimado** | Entrenamiento completo (pip + train + descarga adapter): **~$0.18** y ~25 min. |
| **Parar el pod** | Running = cobro aunque GPU esté al 0%. `Stop Pod` al terminar; `Terminate` + borrar volume cuando tengas el zip. |
| **SSH opcional** | Jupyter Lab basta para subir archivos, entrenar y descargar. |
| **Un solo pod** | Se puede crear volume vacío y subir todo por Jupyter sin pod CPU previo. |
| **Barra de progreso RunPod** | Utilization 0% con pod Running es normal antes de entrenar; igual se cobra. |

### Notebook / entrenamiento

| Lección | Detalle |
|---------|---------|
| **Instalar antes de importar** | Celda `pip install unsloth...` obligatoria antes de `from unsloth import ...`. |
| **PicklingError al guardar** | Unsloth + TRL falla con `save_strategy="epoch"`. Usar `save_strategy="no"` y `eval_strategy="no"`; guardar solo con `model.save_pretrained()` al final. |
| **Loss esperada** | 0.13 → ~0.02 en 92 steps (2 epochs, 367 ejemplos) = aprendizaje sano. |
| **Base HF ≠ Ollama tag** | Entrenamos `ibm-granite/granite-3.3-2b-instruct`; producción usa `granite4.1:3b`. Documentar delta en validación. |

### Descarga HuggingFace en Mac

| Lección | Detalle |
|---------|---------|
| **No usar Xet** | `HF_HUB_ENABLE_HF_TRANSFER=0` y `HF_HUB_DISABLE_XET=1`. Con Xet activo la barra se queda al 1% durante horas; el `.incomplete` apenas crece. |
| **Barra vs disco** | `huggingface-cli` a veces no refresca la UI; Ctrl+C puede mostrar el % real de golpe. Monitorizar con `ls -lh .../*.incomplete` o `du -sh` en otra terminal. |
| **Un proceso a la vez** | Varios `merge_granite_lora.py` / `hf download` en paralelo bloquean locks en `~/.cache/huggingface/hub/.locks/`. |
| **Shards del modelo** | `model-00002` (~64 MB) suele completarse pronto; `model-00001` (~4.7 GB) es el cuello de botella. |
| **Limpiar locks** | Si queda colgado: borrar `*.lock` y `*.incomplete` rotos, relanzar con flags Xet desactivados. |
| **Alternativa lenta** | `curl -C -` al URL directo del shard grande, o fusionar en RunPod (~$0.05, red rápida). |

### Dataset / eval (Enfoque A)

| Lección | Detalle |
|---------|---------|
| **402 pares bastan** | Para validar pipeline cloud; no para maximizar métricas. |
| **0 leakage** | `check_dataset_leakage.py` en CI evita eval circular. |
| **Glossary en train** | JSONL usa `PCM_SYSTEM_GLOSSARY` (sin few-shots) para eval justa post fine-tune. |

### Exportación Ollama

| Lección | Detalle |
|---------|---------|
| **Safetensors no soportado** | `ollama create` desde `granite-merged/` falla con `unsupported architecture "GraniteForCausalLM"`. |
| **Ruta correcta** | `convert_hf_to_gguf.py` (llama.cpp) → `pcm-granite-f16.gguf` → Modelfile → `ollama create`. |
| **Automatizado** | `scripts/export_granite_ollama.sh` incluye conversión GGUF + smoke test vía API. |

### Comandos de referencia (descarga HF)

```bash
# Correcto (Mac)
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
hf download ibm-granite/granite-3.3-2b-instruct --max-workers 8

# Monitorizar progreso real
watch -n 10 'ls -lh ~/.cache/huggingface/hub/models--ibm-granite--granite-3.3-2b-instruct/blobs/*.incomplete 2>/dev/null; du -sh ~/.cache/huggingface/hub/models--ibm-granite--granite-3.3-2b-instruct'

# Limpiar bloqueo (solo si atascado)
rm -f ~/.cache/huggingface/hub/.locks/models--ibm-granite--granite-3.3-2b-instruct/*.lock
```

---

## Troubleshooting

| Problema | Solución |
|----------|----------|
| `LEAKAGE DETECTADO` | Regenerar manifest y dataset; no mezclar eval en train |
| `ModuleNotFoundError: unsloth` | Ejecutar celda pip del notebook antes de imports |
| `PicklingError` SFTConfig | `save_strategy="no"`, `eval_strategy="no"` en TrainingArguments |
| Descarga HF al 1% horas | Desactivar Xet (`HF_HUB_ENABLE_HF_TRANSFER=0`, `HF_HUB_DISABLE_XET=1`) |
| `Still waiting to acquire lock` | Matar procesos HF duplicados; borrar `*.lock` en cache HF |
| OOM en GPU | Reducir batch size en notebook; usar QLoRA 4-bit |
| `unsupported architecture GraniteForCausalLM` | Convertir a GGUF; ver `export_granite_ollama.sh` |
| `pcm-granite` no existe en Ollama | Completar merge → `export_granite_ollama.sh` |
| Formato PCM bajo en holdout | Normal en A; validar pipeline antes de escalar dataset |

---

## Referencias

- Spec: `docs/superpowers/specs/2026-07-04-fase3b-granite-cloud-design.md`
- Plan: `docs/superpowers/plans/2026-07-04-fase3b-granite-cloud-conservative.md`
- System prompt glossary: `src/pcm/compression_prompts.py`
