# Experimento PCM — Conclusiones y cierre (julio 2026)

> **Maintainers (ES)** — Documentación de experimento y fine-tuning. Visitantes: [README en inglés](../README.md) · [getting-started](../getting-started.md).

Documento de cierre del experimento **Prompt Compression Middleware**: qué se demostró, qué aprendimos y cómo repetirlo.

---

## 1. Resumen ejecutivo

**PCM** comprime la *instrucción* del usuario (no el payload) de lenguaje natural a formato estructurado compacto (`TASK=review INPUT=python...`) antes de enviarla a un LLM destino (Mistral, OpenAI, etc.).

| Pregunta | Respuesta |
|----------|-----------|
| ¿Funciona el concepto? | **Sí** — E2E Mistral 94.50% similitud con `pcm-granite` |
| ¿El fine-tune aporta valor? | **Sí** — formato PCM 5.67% → 69.12% vs baseline glossary |
| ¿Listo para producción? | **Integrable** — `OLLAMA_MODEL=pcm-granite`; Enfoque B opcional para pulir holdout |
| ¿Qué sigue? | **COE** — compresión de contextos completos, memoria semántica del chat |

**Veredicto:** experimento **cerrado con éxito**. El middleware de compresión de instrucciones está demostrado.

---

## 2. Fases completadas

| Fase | Alcance | Estado | Artefacto principal |
|------|---------|--------|---------------------|
| **1** | Prototipo proxy + compresor Ollama | ✅ | `src/pcm/proxy.py`, `compressor.py` |
| **2** | Especialización (system prompt + few-shots) | ✅ | `COMPRESSION_SYSTEM_PROMPT` |
| **3a** | Fine-tune local MLX (Qwen) | ✅ | `pcm-compressor` (Ollama) |
| **3b-A** | Fine-tune cloud Unsloth (Granite) | ✅ | `pcm-granite` (Ollama) |

Fases **3 RL** y **4 LLM IR** del roadmap original quedan como investigación futura (COE).

---

## 3. Resultados clave

### 3.1 Fase 3b — `pcm-granite` (recomendado)

Fuente: `data/benchmarks/fase3b_validation.md` (2026-07-04).

**Holdout curated** (20 prompts, eval congelado):

| Config | Formato | Semántica | Ratio |
|--------|---------|-----------|-------|
| granite4.1:3b + glossary | 5.67% | 66.50% | 29.84% |
| **pcm-granite + glossary** | **69.12%** | **88.00%** | 26.51% |

**E2E Mistral** (4 prompts con código/documento adjunto):

| Config | Similitud respuesta | Ratio compresión |
|--------|---------------------|------------------|
| baseline_full | 93.25% | 47.13% |
| **pcm-granite** | **94.50%** | **49.54%** |

### 3.4 Output directives — `output_style=concise` (2026-07-05)

Tres brazos E2E (`mistral-medium-3.5`, `reasoning=none`). Fuente: [data/benchmarks/output_directives_e2e.md](../data/benchmarks/output_directives_e2e.md).

| Brazo | Coste total (4 prompts) |
|-------|-------------------------|
| Baseline (NL) | $0.0157 |
| PCM (`normal`) | $0.0129 (−18%) |
| **PCM + concise** | **$0.0069 (−56%)** |

Similitud PCM normal vs baseline: **94.50%**. Ahorro output concise vs PCM: **42.6%**.

**Coste cloud RunPod:** ~$0.18 (RTX 3090, ~25 min).

### 3.2 Fase 3a — `pcm-compressor` (Mac MLX)

Fine-tune local sobre Qwen; útil como referencia y para Mac sin GPU cloud. Ver `data/benchmarks/fase3_comparison.md`.

### 3.3 Modelos disponibles en Ollama

| Modelo | Origen | Uso recomendado |
|--------|--------|-----------------|
| `granite4.1:3b` | Ollama hub | Baseline / fallback |
| `pcm-granite` | Fase 3b cloud | **Producción / integración** |
| `pcm-compressor` | Fase 3a MLX | Mac-only, experimental |

---

## 4. Conclusiones

1. **La compresión semántica para LLM es viable:** el LLM destino responde igual con instrucción comprimida (94.50% E2E).
2. **El fine-tune internaliza el glosario PCM** sin depender de few-shots en runtime (`PCM_SYSTEM_GLOSSARY`).
3. **Eval desacoplado es obligatorio:** leakage check + holdouts congelados evitan autoengaño (lección de 3a).
4. **El pipeline cloud es reproducible y barato:** <$0.20 por experimento con granite 3B QLoRA.
5. **Holdout formato (~69%) vs E2E (94%)** — en producción importa más E2E; holdout mide adherencia estricta al esquema.
6. **Base HF ≠ Ollama tag** — entrenamos `granite-3.3-2b-instruct`, baseline `granite4.1:3b`; documentar al comparar ratios.

---

## 5. Lecciones aprendidas (consolidado)

### Producto / ML

- No mezclar few-shots del system prompt con el eval set.
- Glossary-only en train para eval justa post fine-tune.
- ~400 pares bastan para validar pipeline; 1k–1.5k (Enfoque B) para maximizar holdout.
- Teacher labels con validación semántica mejoran calidad del dataset.

### RunPod / cloud

- RTX 3090 On-Demand suficiente para granite 3B (~$0.18/experimento).
- Parar el pod al terminar; Jupyter basta (SSH opcional).
- `save_strategy="no"` en Unsloth/TRL evita PicklingError.

### Mac / exportación

- Desactivar Xet en descargas HF (`HF_HUB_ENABLE_HF_TRANSFER=0`, `HF_HUB_DISABLE_XET=1`).
- Ollama no importa Granite safetensors → convertir a GGUF con llama.cpp.
- Un solo proceso HF a la vez (locks en cache).

### Integración

- `OLLAMA_MODEL=pcm-granite` en `.env` activa el compresor fine-tuned en proxy/MCP.
- Umbral `PCM_MIN_INSTRUCTION_TOKENS=12` evita comprimir saludos cortos.

---

## 6. Cómo repetir el experimento

### 6.1 Uso diario (sin reentrenar)

```bash
ollama list | grep pcm-granite   # debe existir (~5.1 GB)
cp .env.example .env
# OLLAMA_MODEL=pcm-granite
# MISTRAL_API_KEY=...
python run.py --proxy
```

### 6.2 Reentrenar Fase 3b (Enfoque A)

Guía completa: [docs/fase3b-granite-cloud.md](fase3b-granite-cloud.md).

```bash
# 1. Dataset
python scripts/build_dataset_v2.py
python scripts/check_dataset_leakage.py   # debe: OK 0 leakage

# 2. RunPod — notebook notebooks/train_granite_unsloth.ipynb
#    Descargar adapter → data/training/v2/checkpoints/granite-lora/

# 3. Mac — merge + Ollama
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
hf download ibm-granite/granite-3.3-2b-instruct
python scripts/merge_granite_lora.py
./scripts/export_granite_ollama.sh

# 4. Validar
python scripts/validate_granite_v2.py --semantic --e2e
```

### 6.3 Reentrenar Fase 3a (Mac MLX)

Guía: [docs/fase3-finetuning.md](fase3-finetuning.md).

```bash
pip install -r requirements-training.txt
python scripts/generate_dataset.py
python scripts/train_compressor.py
./scripts/export_ollama.sh
python scripts/compare_finetune.py --semantic
```

### 6.4 Escalar a Enfoque B (pre-producción)

Ver spec § Enfoque B: `docs/superpowers/specs/2026-07-04-fase3b-granite-cloud-design.md` — 1.000–1.500 pares, corpus real, ~3–5 días prep, ~$0.30–0.60 cloud.

---

## 7. Roadmap post-PCM

| Prioridad | Iniciativa | Descripción |
|-----------|------------|-------------|
| Opcional | Enfoque B | Pulir holdout antes de producción a largo plazo |
| **Siguiente** | **COE** | Context Optimization Engine — contextos completos, BD semántica, memoria de chat |
| Investigación | Fase 3 RL | Optimización por recompensa (ratio + similitud) |
| Investigación | Fase 4 LLM IR | Representación intermedia óptima |

Referencia COE: [github.com/ntnglz/Context-Optimization-Engine](https://github.com/ntnglz/Context-Optimization-Engine).

---

## 8. Índice de documentación

| Documento | Contenido |
|-----------|-----------|
| [README.md](../README.md) | Inicio rápido, configuración, arquitectura |
| [docs/fase3b-granite-cloud.md](fase3b-granite-cloud.md) | RunPod paso a paso + cierre 3b |
| [docs/fase3-finetuning.md](fase3-finetuning.md) | Fine-tune MLX Mac (3a) |
| [data/benchmarks/fase3b_validation.md](../data/benchmarks/fase3b_validation.md) | Métricas finales 3b |
| [MCP de compresion de prompts para LLM v2.md](../MCP%20de%20compresion%20de%20prompts%20para%20LLM%20v2.md) | Roadmap original 4 fases |

---

*Última actualización: 2026-07-04 — Experimento PCM cerrado.*
