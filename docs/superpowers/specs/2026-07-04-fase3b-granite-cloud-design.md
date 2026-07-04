# Fase 3b — Fine-tuning Granite en cloud (diseño corregido)

> **Estado:** Aprobado (Enfoque A conservador — primera iteración)  
> **Fecha:** 2026-07-04  
> **Precedente:** Fase 3a (Qwen MLX local) — ver lecciones en §2  
> **Objetivo:** Entrenar **granite4.1:3b** en cloud con dataset desacoplado de evaluación

---

## 1. Objetivo

Producir `pcm-granite` (Ollama) que supere a `granite4.1:3b` + system prompt en:

| Métrica | Baseline | Objetivo |
|---------|----------|----------|
| Ratio compresión (holdout) | ~20% | **>35%** |
| Ratio compresión (E2E) | ~47% | **>50%** |
| Formato PCM (holdout) | ~58% | **≥80%** |
| E2E similitud Mistral | ~93% | **≥90%** |

> El objetivo de ratio >50% en gold (`example_prompts.json`) se abandona como métrica primaria: esas labels fijan un techo ~40% y están en el system prompt.

---

## 2. Lecciones de Fase 3a (no repetir)

| Problema | Causa | Corrección en 3b |
|----------|-------|------------------|
| Benchmark idéntico granite vs pcm | `example_prompts.json` = few-shots del system prompt | Eval con sets **congelados y excluidos del train** |
| Fine-tune sin efecto E2E | System prompt hace el trabajo; Qwen ≠ granite | Misma base **granite**; inferencia con **system prompt reducido** |
| Ratio no sube de 40% | Labels gold poco agresivas | Teacher con modo agresivo + validación semántica |
| 83 pares insuficientes | Plantillas repetitivas | **500–1.500 pares** diversos y más largos |
| Validación circular | Train ∩ eval ≠ ∅ | Registro de procedencia + hashes; CI que falla si hay overlap |

---

## 3. Principio rector: triple separación

```
┌─────────────────────────────────────────────────────────────┐
│  TRAIN SET (cloud)          │  Nunca aparece en eval       │
│  data/training/v2/train/    │                              │
├─────────────────────────────────────────────────────────────┤
│  DEV SET (cloud, early stop)│  valid.jsonl — solo métricas │
│  data/training/v2/valid/    │  durante entrenamiento       │
├─────────────────────────────────────────────────────────────┤
│  EVAL SETS (congelados)     │  Nunca en train ni en system │
│  data/eval/*                │  prompt few-shots            │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 Conjuntos de evaluación congelados (NUNCA en train)

| Archivo | Uso | Prompts |
|---------|-----|---------|
| `data/example_prompts.json` | Benchmark legacy | 10 |
| `data/e2e_prompts.json` | E2E Mistral + payload | 4 (+ ampliar a 12) |
| `data/eval/holdout_curated.json` | Holdout curado a mano | 20 (nuevo) |
| `data/eval/holdout_synthetic.json` | Plantillas **distintas** a train | 50 (nuevo) |
| `data/eval/long_prompts.json` | Instrucciones largas (>100 tokens) | 15 (nuevo) |

**Regla dura:** ningún `text` de estos archivos ni subcadena normalizada puede estar en `train.jsonl`. Script `scripts/check_dataset_leakage.py` obligatorio en CI.

### 3.2 System prompt en entrenamiento e inferencia

**Fase 3a:** `COMPRESSION_SYSTEM_PROMPT` incluye 10 few-shots = `example_prompts.json`.

**Fase 3b:** dos variantes:

| Variante | Contenido | Uso |
|----------|-----------|-----|
| `PCM_SYSTEM_GLOSSARY` | Reglas + glosario PCM, **sin few-shots** | Train + inferencia + eval |
| `PCM_SYSTEM_FULL` | Glosario + few-shots (actual) | Solo baseline granite en comparativa A/B |

El fine-tune debe internalizar el glosario; la evaluación justa usa `PCM_SYSTEM_GLOSSARY`.

---

## 4. Dataset de entrenamiento — tamaño y calidad

### 4.1 ¿Cuántos pares?

| Tamaño | Coste cloud | Calidad esperada | Recomendación |
|--------|-------------|------------------|---------------|
| 200–300 | ~$1 | Bajo — riesgo overfit plantillas | ❌ Insuficiente |
| **500–800** | ~$2–5 | Buen punto de partida LoRA 3B | ✅ **MVP** |
| **1.000–1.500** | ~$5–10 | Mejor generalización | ✅ **Objetivo** |
| 3.000+ | ~$15–25 | Rendimientos decrecientes en 3B | Opcional fase posterior |

**Recomendación:** empezar con **800 train + 100 valid**; escalar a 1.500 si holdout <80% formato.

### 4.2 ¿Qué tipo de prompts?

Priorizar **variedad y realismo** sobre cantidad de plantillas:

| Tipo | % del train | Características | Ejemplo |
|------|-------------|-----------------|---------|
| **Instrucciones largas** | 30% | 80–300 tokens, cortesía, matices | Peticiones de agente, RAG, legal |
| **Código + contexto** | 20% | "Revisa este snippet…" sin el código en la label | Solo comprimir la instrucción |
| **Multi-idioma** | 25% | ES 50%, EN 40%, mix 10% | |
| **Dominio específico** | 25% | legal, médico, DevOps, ventas, educación | |
| **Ruido conversacional** | 10% | "Por favor podrías…", "Necesito que…" | Entrena eliminación de relleno |

**No incluir en train:**
- Los 10 `example_prompts.json`
- Los 4+ `e2e_prompts.json` (instruction field)
- Los 10 valid.jsonl de Fase 3a
- Ningún few-shot del system prompt actual

### 4.3 Fuentes de prompts brutos (solo entrada, no labels)

| Fuente | Método | Volumen objetivo |
|--------|--------|------------------|
| **Plantillas v2** | Nuevas categorías y paráfrasis (seed distinto a eval) | 300 |
| **Prompts reales sintéticos** | LLM genera instrucciones largas estilo usuario | 200 |
| **Corpus público filtrado** | ShareGPT / WildChat — solo turnos user cortos filtrados por longitud | 200 |
| **Casos proxy** | Instrucciones extraídas de `e2e` **reescritas** (misma intención, texto distinto) | 100 |

### 4.4 Generación de labels (salida PCM)

Pipeline en tres pasos:

```
prompt bruto
    ↓ ① Teacher (granite4.1:3b o mistral-small, strategy=aggressive)
PCM candidato
    ↓ ② validate_pcm_output() — claves glosario
    ↓ ③ compare_prompts() — semántica ≥0.85 vs original
train pair (si pasa)
```

**Teacher agresivo:** system prompt que prioriza mínimos tokens manteniendo intención (no copiar gold de example_prompts).

**Rechazar** pares con:
- Semántica <0.85
- Ratio <15% (no aporta valor)
- Claves PCM inválidas
- Duplicado exacto (hash user content)

### 4.5 Formato de almacenamiento

```
data/training/v2/
  manifest.json           # versiones, conteos, hashes, seed
  sources/                # prompts brutos por fuente (JSONL)
  labels/                 # pares validados (JSONL)
  train.jsonl             # chat format, glossary-only system
  valid.jsonl             # 100 ejemplos, estratificado
  excluded_hashes.txt     # hashes de todos los eval sets
```

Formato chat (Unsloth/HuggingFace):

```json
{
  "messages": [
    {"role": "system", "content": "<PCM_SYSTEM_GLOSSARY>"},
    {"role": "user", "content": "Prompt a comprimir:\n<texto>"},
    {"role": "assistant", "content": "TASK=..."}
  ],
  "meta": {"source": "template_v2", "category": "legal", "lang": "es", "tokens_in": 142}
}
```

---

## 5. Entrenamiento en cloud

### 5.1 Stack recomendado

| Componente | Elección | Motivo |
|------------|----------|--------|
| Plataforma | **RunPod** (spot RTX 4090 o L4) | ~$0.20–0.60/h, factura por segundo |
| Framework | **Unsloth** + PEFT LoRA | Granite en HuggingFace, CUDA maduro |
| Modelo base | **`ibm-granite/granite-3.3-2b-instruct`** o **`granite4.1:3b`** vía HF | Alinear con Ollama `granite4.1:3b` |
| Export | GGUF → Ollama `pcm-granite` | Mismo flujo que producción |

> Verificar equivalencia exacta HF ↔ Ollama antes de entrenar. Si `granite4.1:3b` no está en HF, usar el checkpoint IBM más cercano y documentar delta.

### 5.2 Hiperparámetros iniciales

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| LoRA rank | 16 | Igual que 3a |
| LoRA alpha | 32 | |
| Epochs | 2–3 | Early stop por valid loss |
| Learning rate | 2e-4 | Unsloth suele tolerar LR más alto que MLX |
| Batch size | 4 | 4090 24GB |
| Max seq length | 2048 | Cubrir instrucciones largas |
| mask_prompt | true | Solo aprende assistant |
| Quantization | 4-bit QLoRA | Reduce VRAM |

### 5.3 Coste estimado

| Fase | GPU | Tiempo | Coste |
|------|-----|--------|-------|
| Setup (CPU pod) | — | 30 min | ~$0.05 |
| Dataset generation (teacher local) | Ollama Mac | 1–2 h | $0 |
| LoRA train 800 pares | RTX 4090 spot | 30–60 min | **$0.20–0.50** |
| LoRA train 1.500 pares | RTX 4090 spot | 45–90 min | **$0.30–0.90** |
| 3–5 experimentos | — | — | **$2–5 total** |
| Export + prueba Ollama | Mac local | — | $0 |

**Presupuesto recomendado:** **$10** (margen para reintentos).

### 5.4 Artefactos de salida

```
data/training/v2/checkpoints/
  pcm-granite-lora/         # adapter safetensors
  pcm-granite-merged/       # modelo fusionado
  training_log.json         # loss curves

Ollama local:
  pcm-granite               # modelo de producción candidato
```

---

## 6. Evaluación (post-entrenamiento)

### 6.1 Protocolo obligatorio

1. `scripts/check_dataset_leakage.py` → PASS
2. Benchmark holdout (`data/eval/*`) con `PCM_SYSTEM_GLOSSARY`
3. E2E Mistral (`e2e_prompts.json`) — ampliado
4. Comparativa A/B:
   - `granite4.1:3b` + `PCM_SYSTEM_FULL` (baseline actual)
   - `pcm-granite` + `PCM_SYSTEM_GLOSSARY` (fine-tuned)
5. Comparativa adicional (sanity):
   - `granite4.1:3b` + `PCM_SYSTEM_GLOSSARY` (techo sin fine-tune)

### 6.2 Criterios de éxito (go / no-go)

| Criterio | Go | No-go |
|----------|-----|-------|
| Holdout formato | ≥80% | <70% |
| Holdout ratio vs granite+full | **+5 pp** mínimo | ≤0 pp |
| E2E similitud | ≥90% | <85% |
| E2E ratio | ≥50% | <45% |
| Leakage check | 0 overlap | cualquier overlap |

### 6.3 Script unificado

Extender `scripts/validate_finetune.py` → `scripts/validate_granite_v2.py` con:
- Eval sets en `data/eval/`
- System prompt configurable (`--system glossary|full`)
- Informe `data/benchmarks/fase3b_validation.md`

---

## 7. Ampliación de E2E (recomendado)

Ampliar `data/e2e_prompts.json` de 4 a **12 casos**:

| Categoría nueva | Por qué |
|-----------------|---------|
| Instrucción larga (>150 tokens) | Mide compresión real en proxy |
| Prompt con ruido / cortesía extrema | Valida eliminación de relleno |
| Instrucción ES mezclada con términos EN | Realismo |
| Agente multi-paso | "Primero X, luego Y, finalmente Z" |

Estos casos se escriben a mano y **nunca** entran en train.

---

## 8. Enfoques de dataset (decisión)

### A. Teacher-only sintético (rápido)

- 800 prompts plantilla v2 + teacher granite agresivo
- **Pros:** 1–2 días, <$5
- **Contras:** Techo de diversidad

### B. Híbrido teacher + corpus real filtrado (recomendado)

- 40% plantillas v2, 30% LLM-generado largo, 30% corpus filtrado
- Teacher + validación semántica
- **Pros:** Mejor generalización, prompts más largos
- **Contras:** 3–5 días de preparación

### C. Curación manual mayoritaria

- >50% pares revisados por humano
- **Pros:** Máxima calidad de labels
- **Contras:** Lento, no escala

**Decisión (2026-07-04):** **Enfoque A** para la primera iteración — el operador no tiene experiencia previa en fine-tuning cloud; minimizar riesgo y coste antes de escalar a B.

| Enfoque A (elegido) | Enfoque B (fase posterior si A tiene go) |
|---------------------|------------------------------------------|
| ~500 pares train, ~50 valid | 1.000–1.500 pares |
| Solo plantillas v2 + teacher granite | + corpus real + prompts LLM largos |
| Coste ~$2, prep ~1–2 días | Coste ~$5–10, prep ~3–5 días |
| Valida pipeline cloud sin sorpresas | Maximiza generalización |

**Criterio para escalar a B:** Enfoque A pasa leakage check + holdout formato ≥70% pero ratio <+5 pp vs baseline → ampliar dataset.

---

## 9. Estructura de archivos nueva

```
data/
  eval/                              # NUNCA en train
    holdout_curated.json
    holdout_synthetic.json
    long_prompts.json
    excluded_manifest.json             # hashes de todos los eval texts
  e2e_prompts.json                   # ampliar a 12
  training/v2/
    sources/
    train.jsonl
    valid.jsonl
    manifest.json

src/pcm/
  compression_prompts.py               # PCM_SYSTEM_GLOSSARY vs FULL

scripts/
  build_dataset_v2.py                # pipeline completo
  check_dataset_leakage.py
  train_granite_cloud.py               # wrapper Unsloth (ejecutar en RunPod)
  export_granite_ollama.sh
  validate_granite_v2.py

docs/
  fase3b-granite-cloud.md              # guía operativa RunPod
```

---

## 10. Fuera de alcance (Fase 3b)

- Full fine-tuning (solo LoRA/QLoRA)
- Cambiar glosario PCM
- Entrenar en Mac MLX
- RL / optimización por ratio
- GitHub Actions para GPU

---

## 11. Roadmap de implementación (orden)

1. Extraer `PCM_SYSTEM_GLOSSARY` de `compressor.py`
2. Crear `data/eval/*` (holdout congelado)
3. Implementar `check_dataset_leakage.py`
4. Implementar `build_dataset_v2.py` (enfoque B, 1.000 pares)
5. Ampliar `e2e_prompts.json` a 12 casos
6. Documentar RunPod + notebook Unsloth en `docs/fase3b-granite-cloud.md`
7. Entrenar en cloud (~$2–5)
8. Export `pcm-granite` → Ollama en Mac
9. `validate_granite_v2.py` — decisión go/no-go

---

## 12. Referencias

- Fase 3a spec: `docs/superpowers/specs/2026-07-03-fase3-finetuning-design.md`
- Validación 3a: `data/benchmarks/fase3_validation.md`
- Unsloth: https://github.com/unslothai/unsloth
- IBM Granite HuggingFace: `ibm-granite/*`
- RunPod pricing: ~$0.22–0.60/h spot (RTX 3090/4090)
