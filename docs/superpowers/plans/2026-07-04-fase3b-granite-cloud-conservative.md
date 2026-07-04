# Fase 3b Granite Cloud (Enfoque A conservador) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entrenar `granite4.1:3b` en RunPod con ~500 pares (solo plantillas v2), evaluación desacoplada, y publicar `pcm-granite` en Ollama — validando el pipeline cloud antes de escalar.

**Architecture:** Eval sets congelados en `data/eval/` → plantillas v2 generan prompts → teacher granite (local Ollama) produce labels agresivas → validación PCM + semántica → `train.jsonl` con `PCM_SYSTEM_GLOSSARY` → Unsloth QLoRA en RunPod → GGUF/Ollama `pcm-granite` → `validate_granite_v2.py` go/no-go.

**Tech Stack:** Python, Unsloth, PEFT LoRA, RunPod (RTX 4090 spot), Ollama granite4.1:3b, Mistral API (E2E), pytest

**Spec:** `docs/superpowers/specs/2026-07-04-fase3b-granite-cloud-design.md`

**Presupuesto cloud:** ~$2 por experimento | **Dataset:** ~500 train + ~50 valid (solo plantillas)

---

## Riesgos del Enfoque A (para el operador)

| Riesgo | Probabilidad | Mitigación |
|--------|--------------|------------|
| Overfitting a plantillas | Media | Holdout con plantillas distintas; early stopping |
| No superar baseline E2E | Media | Es **aceptable** en A — objetivo es validar pipeline |
| Granite HF ≠ Ollama | Baja | Verificar checkpoint en Task 8 antes de entrenar |
| Coste cloud imprevisto | Baja | Spot 4090, parar pod al terminar, tope $10 |
| Leakage train/eval | Baja si CI pasa | `check_dataset_leakage.py` obligatorio |

**Criterio de éxito Enfoque A (más blando que B):** pipeline completo sin errores + holdout formato ≥70% + 0 leakage. Mejora sobre baseline es bonus, no bloqueante.

---

## File map

| File | Responsibility |
|------|----------------|
| `src/pcm/compression_prompts.py` | `PCM_SYSTEM_GLOSSARY`, `PCM_SYSTEM_FULL` |
| `src/pcm/compressor.py` | Usar prompts importados; flag `use_glossary_only` |
| `data/eval/holdout_curated.json` | 20 prompts curados (nunca en train) |
| `data/eval/holdout_synthetic.json` | 30 prompts plantilla eval-only |
| `data/eval/excluded_manifest.json` | Hashes de todos los textos prohibidos en train |
| `data/training/v2/templates_train.json` | Plantillas SOLO para train (seed 100) |
| `data/training/v2/templates_eval.json` | Plantillas SOLO para eval holdout (seed 200) |
| `scripts/check_dataset_leakage.py` | Detecta overlap train vs eval |
| `scripts/build_dataset_v2.py` | Genera train/valid con teacher granite |
| `scripts/validate_granite_v2.py` | Benchmark A/B post-entrenamiento |
| `tests/test_dataset_leakage.py` | Tests del checker |
| `notebooks/train_granite_unsloth.ipynb` | Notebook RunPod (copiar al pod) |
| `docs/fase3b-granite-cloud.md` | Guía paso a paso para principiantes |
| `requirements-cloud.txt` | unsloth, transformers, peft, datasets |

---

### Task 1: Separar system prompts (glossary vs full)

**Files:**
- Create: `src/pcm/compression_prompts.py`
- Modify: `src/pcm/compressor.py`

- [ ] **Step 1: Crear compression_prompts.py**

Mover `COMPRESSION_SYSTEM_PROMPT` actual a `PCM_SYSTEM_FULL`.

Crear `PCM_SYSTEM_GLOSSARY` — mismas REGLAS + GLOSARIO + PROHIBIDO, **sin sección EJEMPLOS** (líneas desde `EJEMPLOS:` hasta el final del bloque de ejemplos).

```python
# src/pcm/compression_prompts.py
"""Variantes del system prompt PCM."""

PCM_SYSTEM_GLOSSARY = '''Eres un compresor de prompts para LLM. Transforma lenguaje natural en formato PCM compacto.

REGLAS:
1. Preserva EXACTAMENTE la intención semántica
2. Maximiza reducción de tokens
...
PROHIBIDO:
...
'''

# Importar o duplicar el prompt actual completo con few-shots
from pcm.compressor import COMPRESSION_SYSTEM_PROMPT as PCM_SYSTEM_FULL
```

> Refactor limpio: mover el string completo a `compression_prompts.py` como `PCM_SYSTEM_FULL`; `compressor.py` re-exporta para compatibilidad.

- [ ] **Step 2: Añadir parámetro en CompressorConfig**

```python
# compressor.py — CompressorConfig
glossary_only: bool = False  # True → PCM_SYSTEM_GLOSSARY
```

En `_get_compression_prompt`, si `glossary_only`: usar `PCM_SYSTEM_GLOSSARY`.

- [ ] **Step 3: Test**

```python
# tests/test_compressor.py
def test_glossary_only_excludes_few_shot_examples():
    c = PromptCompressor(CompressorConfig(glossary_only=True))
    prompt = c._get_compression_prompt("balanced")
    assert "EJEMPLOS:" not in prompt
    assert "TASK:" in prompt  # glosario presente
```

- [ ] **Step 4: CI**

```bash
./scripts/ci-local.sh
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: system prompt glossary-only para eval fair fine-tune"
```

---

### Task 2: Eval sets congelados

**Files:**
- Create: `data/eval/holdout_curated.json`
- Create: `data/eval/holdout_synthetic.json`
- Create: `data/eval/excluded_manifest.json`
- Create: `data/eval/.gitkeep`

- [ ] **Step 1: Crear holdout_curated.json (20 prompts)**

Hand-crafted, **distintos** de `example_prompts.json`. Incluir:
- 4 code_review (python, rust, sql injection context)
- 3 translation (pares idioma variados)
- 3 summarization (papers, tickets, emails)
- 2 explanation (audiencia variada)
- 2 writing (email, README)
- 2 comparison
- 2 legal/planning
- 2 database/design

Formato igual que `example_prompts.json`:
```json
{
  "id": "eval_001",
  "text": "...",
  "category": "code_review",
  "expected_compression": "TASK=...",
  "language": "es"
}
```

`expected_compression` = label **agresiva** curada a mano (referencia para formato score, no para ratio techo).

- [ ] **Step 2: Crear data/eval/holdout_synthetic.json (30 prompts)**

Usar `data/training/v2/templates_eval.json` (Task 3) — generar en Task 5.

Inicialmente: placeholder `[]` o generar 30 con script.

- [ ] **Step 3: Crear excluded_manifest.json**

Script inline o manual — lista de SHA256 de:
- Todos los `text` de `example_prompts.json`
- Todos los `instruction` de `e2e_prompts.json`
- Todos los `text` de holdout_curated.json
- Todos los user texts de Fase 3a `valid.jsonl` (si existen)

- [ ] **Step 4: Commit**

```bash
git add data/eval/
git commit -m "feat: eval sets congelados Fase 3b"
```

---

### Task 3: Plantillas v2 (train ≠ eval)

**Files:**
- Create: `data/training/v2/templates_train.json`
- Create: `data/training/v2/templates_eval.json`

- [ ] **Step 1: templates_train.json**

Copiar estructura de `data/training/templates.json` pero:
- Nuevas paráfrasis en cada `prompts[]`
- Nuevos `topics`, `inputs`, `pairs` (no repetir combinaciones de eval)
- Seed combinatorio distinto (más variaciones por categoría)
- Objetivo: ~550 combinaciones únicas brutas → ~500 tras dedup

Ejemplo categoría nueva en train (no en eval):
```json
"refactoring": {
  "prompts": [
    "Refactoriza este módulo {lang} eliminando deuda técnica y mejorando testabilidad. Salida en lista priorizada.",
    "Suggest refactoring steps for this legacy {lang} service. Focus on coupling and test coverage."
  ],
  "pcm_templates": [
    "TASK=review INPUT={input} CHECK=tech_debt,testability FORMAT=list ORDER=priority",
    "TASK=review INPUT={input} CHECK=coupling,test_coverage FORMAT=list"
  ],
  "inputs": ["python", "java", "csharp"]
}
```

Añadir 2–3 categorías nuevas vs Fase 3a: `refactoring`, `api_design`, `data_analysis`.

- [ ] **Step 2: templates_eval.json**

10 categorías con 3 prompts cada una — **textos distintos** a train. Solo para `holdout_synthetic.json`.

- [ ] **Step 3: Commit**

```bash
git add data/training/v2/templates_*.json
git commit -m "feat: plantillas v2 train/eval separadas"
```

---

### Task 4: Leakage checker + tests

**Files:**
- Create: `scripts/check_dataset_leakage.py`
- Create: `tests/test_dataset_leakage.py`

- [ ] **Step 1: Test que falla**

```python
# tests/test_dataset_leakage.py
from scripts.check_dataset_leakage import normalize_text, load_excluded_hashes, check_leakage

def test_normalize_strips_prompt_prefix():
    assert normalize_text("Prompt a comprimir:\nhola") == "hola"

def test_detects_exact_overlap(tmp_path):
    excluded = {"abc123"}
    train = [{"user_text": "hola", "user_hash": "abc123"}]
    leaks = check_leakage(train, excluded)
    assert len(leaks) == 1
```

- [ ] **Step 2: Implementar check_dataset_leakage.py**

```python
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def normalize_text(text: str) -> str:
    t = text.strip()
    if t.startswith("Prompt a comprimir:"):
        t = t.split("\n", 1)[-1].strip()
    return " ".join(t.split())

def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()

def load_excluded_hashes() -> set[str]:
    hashes = set()
    # example_prompts, e2e instructions, eval holdout, manifest
    ...
    return hashes

def check_leakage(train_items: list[dict], excluded: set[str]) -> list[dict]:
    return [item for item in train_items if item["user_hash"] in excluded]

def main() -> int:
    # Lee train.jsonl, reporta leaks, exit 1 si any
    ...
```

- [ ] **Step 3: Integrar en ci-local.sh**

Añadir al final (si existe train.jsonl):
```bash
if [ -f data/training/v2/train.jsonl ]; then
  python scripts/check_dataset_leakage.py || exit 1
fi
```

- [ ] **Step 4: CI + commit**

---

### Task 5: build_dataset_v2.py (Enfoque A)

**Files:**
- Create: `scripts/build_dataset_v2.py`

- [ ] **Step 1: Implementar generador**

Flujo:
1. Expandir `templates_train.json` combinatoriamente (reutilizar lógica de `generate_dataset.py` v2)
2. Dedup por hash user text
3. Excluir cualquier hash en `load_excluded_hashes()`
4. Para cada prompt sin label: llamar `PromptCompressor` con `granite4.1:3b`, `strategy=aggressive`, `glossary_only=False` (teacher usa full prompt)
5. Validar con `validate_pcm_output()` + `compare_prompts()` semántica ≥0.85
6. Split 90/10 → `train.jsonl` / `valid.jsonl` (~500/50)
7. Escribir `manifest.json` con conteos y versión `v2a-templates-only`
8. System en JSONL: `PCM_SYSTEM_GLOSSARY` (lo que aprenderá el modelo)

CLI:
```bash
python scripts/build_dataset_v2.py
python scripts/build_dataset_v2.py --skip-teacher  # solo si labels precalculadas
python scripts/check_dataset_leakage.py
```

- [ ] **Step 2: Generar dataset local**

```bash
python scripts/build_dataset_v2.py
```

Expected: manifest `train` ≈ 500, `valid` ≈ 50, leakage 0.

- [ ] **Step 3: Commit** (solo scripts + templates; train.jsonl gitignored)

---

### Task 6: Generar holdout_synthetic desde templates_eval

**Files:**
- Modify: `scripts/build_dataset_v2.py` (añadir `--eval-output`)

- [ ] **Step 1: Flag --eval-output**

Genera `data/eval/holdout_synthetic.json` desde `templates_eval.json` con labels del teacher (o curadas).

- [ ] **Step 2: Regenerar excluded_manifest.json**

- [ ] **Step 3: Commit**

---

### Task 7: validate_granite_v2.py

**Files:**
- Create: `scripts/validate_granite_v2.py`

- [ ] **Step 1: Script comparativo**

Compara 4 configuraciones:

| Label | Compresor | System |
|-------|-----------|--------|
| baseline_full | granite4.1:3b | FULL |
| baseline_glossary | granite4.1:3b | glossary |
| finetuned_glossary | pcm-granite | glossary |

Eval sets: `holdout_curated`, `holdout_synthetic`, `e2e_prompts.json`

Salida: `data/benchmarks/fase3b_validation.md`

- [ ] **Step 2: Commit**

---

### Task 8: Documentación RunPod (principiante)

**Files:**
- Create: `docs/fase3b-granite-cloud.md`
- Create: `requirements-cloud.txt`
- Create: `notebooks/train_granite_unsloth.ipynb`

- [ ] **Step 1: docs/fase3b-granite-cloud.md**

Secciones para quien nunca ha usado cloud:
1. Crear cuenta RunPod
2. Añadir $10 crédito
3. Crear Network Volume 20GB
4. CPU pod: subir `train.jsonl`, `valid.jsonl`
5. GPU pod RTX 4090 spot: template PyTorch 2.1
6. Instalar Unsloth (`pip install unsloth`)
7. Abrir notebook `train_granite_unsloth.ipynb`
8. Entrenar (~30 min)
9. Descargar adapter a Mac
10. Fusionar + `ollama create pcm-granite`
11. `python scripts/validate_granite_v2.py --semantic --e2e`
12. **Parar el pod** (importante — evita costes)

Incluir captura de coste esperado y checklist go/no-go.

- [ ] **Step 2: Notebook mínimo**

Cells:
- Load granite from HF (`ibm-granite/granite-3.3-2b-instruct` o verificar `granite-4.1`)
- Load JSONL dataset
- Unsloth FastLanguageModel + LoRA
- Train 2 epochs, save adapter
- Zip y download

- [ ] **Step 3: requirements-cloud.txt**

```
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
torch>=2.1.0
transformers>=4.43.0
peft>=0.10.0
datasets>=2.18.0
trl>=0.8.0
```

- [ ] **Step 4: Commit**

```bash
git add docs/fase3b-granite-cloud.md notebooks/ requirements-cloud.txt
git commit -m "docs: guía RunPod Fase 3b granite para principiantes"
```

---

### Task 9: Export Ollama pcm-granite

**Files:**
- Create: `scripts/export_granite_ollama.sh`

- [ ] **Step 1: Script export**

Similar a `export_ollama.sh` pero:
- Model name: `pcm-granite`
- Path: `data/training/v2/checkpoints/`
- Verificar modelo HF compatible

- [ ] **Step 2: Commit**

---

### Task 10: Ejecución cloud (manual — operador)

**No automatizable en CI.**

- [ ] **Step 1:** `python scripts/build_dataset_v2.py` + leakage check
- [ ] **Step 2:** Subir dataset a RunPod volume
- [ ] **Step 3:** Ejecutar notebook (~30 min, ~$0.30)
- [ ] **Step 4:** Descargar adapter, **parar pod**
- [ ] **Step 5:** `./scripts/export_granite_ollama.sh`
- [ ] **Step 6:** `python scripts/validate_granite_v2.py --semantic --e2e`
- [ ] **Step 7:** Decisión go/no-go según spec §6.2 (criterios Enfoque A §Riesgos)

---

## Self-review (spec coverage)

| Requisito spec Enfoque A | Task |
|--------------------------|------|
| granite cloud | Task 8, 10 |
| ~500 pares plantillas | Task 3, 5 |
| eval desacoplado | Task 2, 4, 6 |
| PCM_SYSTEM_GLOSSARY | Task 1 |
| teacher granite | Task 5 |
| leakage CI | Task 4 |
| validate A/B | Task 7 |
| docs principiante | Task 8 |
| coste ~$2 | Task 8 doc |
| 0 overlap | Task 4, 5 |

---

## Execution Handoff

Plan guardado en `docs/superpowers/plans/2026-07-04-fase3b-granite-cloud-conservative.md`.

**Opciones:**

1. **Subagent-Driven** — Un subagente por task, revisión entre tasks
2. **Inline Execution** — Implementar en sesión (Tasks 1–7 local; Task 10 manual cloud contigo)

¿Cuál prefieres?
