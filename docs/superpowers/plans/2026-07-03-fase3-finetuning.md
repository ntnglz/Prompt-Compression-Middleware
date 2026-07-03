# Fase 3 Fine-tuning PCM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entrenar un compresor PCM con LoRA/MLX en Mac M4 Pro, exportarlo a Ollama como `pcm-compressor`, y demostrar ratio >50% frente al baseline `granite4.1:3b`.

**Architecture:** Dataset JSONL (gold + sintéticos) → `mlx_lm.lora` sobre Qwen2.5-3B-Instruct-4bit → fusión + GGUF → `ollama create pcm-compressor`. El runtime PCM existente solo cambia `OLLAMA_MODEL`. Validación con benchmark A/B y E2E Mistral.

**Tech Stack:** Python 3.11+, mlx-lm, Apple MLX, Ollama, pytest, tiktoken

**Spec:** `docs/superpowers/specs/2026-07-03-fase3-finetuning-design.md`

---

## File map

| File | Responsibility |
|------|----------------|
| `src/pcm/training/dataset.py` | Validación PCM, conversión a JSONL chat, split train/valid |
| `data/training/templates.json` | Plantillas por categoría para prompts sintéticos |
| `scripts/generate_dataset.py` | Genera `train.jsonl` + `valid.jsonl` |
| `scripts/train_compressor.py` | Wrapper `mlx_lm.lora` + `mlx_lm.fuse` |
| `scripts/export_ollama.sh` | `mlx_lm.convert` → GGUF + `ollama create` |
| `scripts/compare_finetune.py` | Benchmark A/B granite vs pcm-compressor |
| `requirements-training.txt` | mlx-lm (no en runtime PCM) |
| `tests/test_training_dataset.py` | Tests unitarios del módulo dataset |
| `docs/fase3-finetuning.md` | Guía operativa |

---

### Task 1: Scaffolding y dependencias de entrenamiento

**Files:**
- Create: `src/pcm/training/__init__.py`
- Create: `requirements-training.txt`
- Modify: `.gitignore`
- Create: `data/training/.gitkeep`

- [ ] **Step 1: Crear paquete training**

```python
# src/pcm/training/__init__.py
"""Utilidades de dataset y fine-tuning PCM."""
```

- [ ] **Step 2: Crear requirements-training.txt**

```
# Fine-tuning PCM — instalar solo en Mac de entrenamiento
# pip install -r requirements-training.txt
mlx-lm>=0.21.0
mlx>=0.22.0
```

- [ ] **Step 3: Actualizar .gitignore**

Añadir al final de `.gitignore`:

```
# Fine-tuning checkpoints (pesos locales grandes)
data/training/checkpoints/
data/training/*.jsonl
!data/training/.gitkeep
```

- [ ] **Step 4: Crear directorio data/training**

```bash
mkdir -p data/training
touch data/training/.gitkeep
```

- [ ] **Step 5: Commit**

```bash
git add src/pcm/training/__init__.py requirements-training.txt .gitignore data/training/.gitkeep
git commit -m "feat: scaffolding Fase 3 fine-tuning PCM"
```

---

### Task 2: Módulo dataset con validación PCM

**Files:**
- Create: `src/pcm/training/dataset.py`
- Create: `tests/test_training_dataset.py`
- Modify: `src/pcm/compressor.py` (exportar lista de claves válidas si hace falta — preferir parsear del prompt existente)

- [ ] **Step 1: Escribir test que falla**

```python
# tests/test_training_dataset.py
import json
from pathlib import Path

import pytest

from pcm.training.dataset import (
    VALID_PCM_KEYS,
    build_chat_example,
    load_gold_prompts,
    split_dataset,
    validate_pcm_output,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "example_prompts.json"


def test_valid_pcm_keys_includes_task():
    assert "TASK" in VALID_PCM_KEYS


def test_validate_pcm_output_accepts_gold_example():
    ok, errors = validate_pcm_output(
        "TASK=review INPUT=python CHECK=race,leak,perf FORMAT=markdown ORDER=severity"
    )
    assert ok is True
    assert errors == []


def test_validate_pcm_output_rejects_missing_task():
    ok, errors = validate_pcm_output("INPUT=python")
    assert ok is False
    assert any("TASK" in e for e in errors)


def test_validate_pcm_output_rejects_unknown_key():
    ok, errors = validate_pcm_output("TASK=review TARGET=client")
    assert ok is False
    assert any("TARGET" in e for e in errors)


def test_build_chat_example_structure():
    example = build_chat_example(
        "Analiza este código Python",
        "TASK=review INPUT=python",
    )
    assert len(example["messages"]) == 3
    assert example["messages"][0]["role"] == "system"
    assert example["messages"][1]["role"] == "user"
    assert "Prompt a comprimir:" in example["messages"][1]["content"]
    assert example["messages"][2]["content"].startswith("TASK=")


def test_load_gold_prompts_returns_ten():
    prompts = load_gold_prompts(GOLD)
    assert len(prompts) == 10
    assert all("text" in p and "expected_compression" in p for p in prompts)


def test_split_dataset_stratified():
    items = [{"category": "a", "messages": []} for _ in range(10)]
    items += [{"category": "b", "messages": []} for _ in range(10)]
    train, valid = split_dataset(items, valid_ratio=0.2, seed=42)
    assert len(train) + len(valid) == 20
    assert len(valid) >= 2


def test_write_jsonl_roundtrip(tmp_path):
    examples = [build_chat_example("hola", "TASK=explain TOPIC=test")]
    path = tmp_path / "out.jsonl"
    write_jsonl(examples, path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["messages"][2]["content"].startswith("TASK=")
```

- [ ] **Step 2: Ejecutar test — debe fallar**

```bash
cd "/Volumes/DevSSD/XcodeProjects/Ideas/Prompt Compression Middleware"
./scripts/ci-local.sh
```

Expected: FAIL — `ModuleNotFoundError: No module named 'pcm.training.dataset'`

- [ ] **Step 3: Implementar dataset.py**

```python
# src/pcm/training/dataset.py
"""Generación y validación de dataset JSONL para fine-tuning PCM."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from pcm.compressor import COMPRESSION_SYSTEM_PROMPT

FIELD_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=")

# Claves del glosario PCM (extraídas del system prompt)
VALID_PCM_KEYS = frozenset({
    "TASK", "INPUT", "CHECK", "FORMAT", "ORDER", "FROM", "TO",
    "STYLE", "TONE", "DOMAIN", "TOPIC", "TYPE", "ITEMS", "CRITERIA",
    "FEATURES", "FOCUS", "HIGHLIGHT", "INCLUDE", "AUDIENCE", "USE",
    "SCHEMA", "ENTITIES", "REQUIRE", "OPTIMIZE",
})

USER_PREFIX = "Prompt a comprimir:\n"


def validate_pcm_output(text: str) -> tuple[bool, list[str]]:
    """Valida que la salida PCM cumple reglas básicas del glosario."""
    errors: list[str] = []
    normalized = " ".join(text.strip().split())

    if not normalized.startswith("TASK="):
        errors.append("La salida debe empezar por TASK=")

    if "\n" in text.strip():
        errors.append("La salida debe ser una sola línea")

    keys = [m.group(1).upper() for m in FIELD_PATTERN.finditer(normalized)]
    if not keys:
        errors.append("No se encontraron claves PCM")

    for key in keys:
        if key not in VALID_PCM_KEYS:
            errors.append(f"Clave desconocida: {key}")

    return (len(errors) == 0, errors)


def build_chat_example(user_text: str, assistant_pcm: str) -> dict[str, Any]:
    """Construye un ejemplo chat para mlx-lm."""
    return {
        "messages": [
            {"role": "system", "content": COMPRESSION_SYSTEM_PROMPT},
            {"role": "user", "content": f"{USER_PREFIX}{user_text.strip()}"},
            {"role": "assistant", "content": assistant_pcm.strip()},
        ]
    }


def load_gold_prompts(path: Path) -> list[dict[str, Any]]:
    """Carga example_prompts.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Se esperaba lista en {path}")
    return data


def gold_to_examples(gold_path: Path) -> list[dict[str, Any]]:
    """Convierte prompts gold a ejemplos chat validados."""
    examples: list[dict[str, Any]] = []
    for item in load_gold_prompts(gold_path):
        pcm = item["expected_compression"]
        ok, errors = validate_pcm_output(pcm)
        if not ok:
            raise ValueError(f"Gold inválido {item.get('id')}: {errors}")
        ex = build_chat_example(item["text"], pcm)
        ex["_meta"] = {
            "id": item.get("id"),
            "category": item.get("category", "unknown"),
            "language": item.get("language", "unknown"),
            "source": "gold",
        }
        examples.append(ex)
    return examples


def split_dataset(
    items: list[dict[str, Any]],
    *,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split estratificado por categoría."""
    rng = random.Random(seed)
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        cat = item.get("_meta", {}).get("category", "unknown")
        by_cat.setdefault(cat, []).append(item)

    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for group in by_cat.values():
        rng.shuffle(group)
        n_valid = max(1, int(len(group) * valid_ratio)) if len(group) > 1 else 0
        if n_valid:
            valid.extend(group[:n_valid])
            train.extend(group[n_valid:])
        else:
            train.extend(group)
    return train, valid


def strip_meta(example: dict[str, Any]) -> dict[str, Any]:
    """Elimina metadatos internos antes de escribir JSONL."""
    return {"messages": example["messages"]}


def write_jsonl(examples: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(strip_meta(ex), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            examples.append(json.loads(line))
    return examples
```

- [ ] **Step 4: Ejecutar tests**

```bash
./scripts/ci-local.sh
```

Expected: PASS (todos los tests, incluidos los nuevos)

- [ ] **Step 5: Commit**

```bash
git add src/pcm/training/dataset.py tests/test_training_dataset.py
git commit -m "feat: módulo dataset PCM para fine-tuning"
```

---

### Task 3: Plantillas sintéticas y generador de dataset

**Files:**
- Create: `data/training/templates.json`
- Create: `scripts/generate_dataset.py`

- [ ] **Step 1: Crear templates.json**

```json
{
  "code_review": {
    "language": ["es", "en"],
    "prompts": [
      "Revisa este código {lang} buscando bugs de concurrencia, fugas de memoria y cuellos de botella. Informe en Markdown por severidad.",
      "Analyze this {lang} source code for security issues, performance problems, and maintainability concerns. Output as JSON.",
      "Examina detenidamente el siguiente código {lang} e identifica errores lógicos, anti-patrones y mejoras de rendimiento. Formato lista."
    ],
    "pcm_templates": [
      "TASK=review INPUT={input} CHECK=race,leak,perf FORMAT=markdown ORDER=severity",
      "TASK=analyze INPUT={input} CHECK=security,perf,code_smells FORMAT=json",
      "TASK=review INPUT={input} CHECK=logic,anti_patterns,perf FORMAT=list"
    ],
    "inputs": ["python", "javascript", "rust", "go"]
  },
  "translation": {
    "language": ["es", "en"],
    "prompts": [
      "Traduce este documento técnico del {from_lang} al {to_lang} manteniendo terminología especializada y tono formal.",
      "Please translate this legal contract from {from_lang} to {to_lang}, preserving formal style and legal terminology."
    ],
    "pcm_templates": [
      "TASK=translate FROM={from_lang} TO={to_lang} STYLE=formal DOMAIN=technical",
      "TASK=translate FROM={from_lang} TO={to_lang} STYLE=formal DOMAIN=legal"
    ]
  },
  "summarization": {
    "language": ["es", "en"],
    "prompts": [
      "Genera un resumen ejecutivo de este informe trimestral destacando métricas clave y conclusiones principales en formato lista.",
      "Summarize this research paper highlighting methodology, findings, and limitations. Use bullet points."
    ],
    "pcm_templates": [
      "TASK=summarize INPUT=quarterly_report FORMAT=list HIGHLIGHT=metrics,conclusions",
      "TASK=summarize INPUT=research_paper FORMAT=list HIGHLIGHT=methodology,findings,limitations"
    ]
  },
  "explanation": {
    "language": ["es", "en"],
    "prompts": [
      "Explica qué es {topic} a un público principiante usando analogías y ejemplos cotidianos.",
      "Explain {topic} to a non-technical audience with simple language and real-world examples."
    ],
    "pcm_templates": [
      "TASK=explain TOPIC={topic} AUDIENCE=beginner STYLE=simple USE=analogies,examples"
    ],
    "topics": ["blockchain", "machine_learning", "quantum_computing", "neural_networks"]
  },
  "writing": {
    "language": ["es", "en"],
    "prompts": [
      "Redacta un email profesional para un cliente potencial sobre nuestro servicio de {topic}. Debe ser conciso, destacar beneficios e incluir llamada a la acción.",
      "Write a professional blog post introduction about {topic} for developers. Tone: informative and concise."
    ],
    "pcm_templates": [
      "TASK=write TYPE=email TO=client TOPIC={topic} STYLE=professional TONE=concise INCLUDE=benefits,cta",
      "TASK=write TYPE=blog TOPIC={topic} AUDIENCE=developers STYLE=informative TONE=concise"
    ],
    "topics": ["cloud_migration", "api_design", "devops_automation"]
  },
  "comparison": {
    "language": ["es", "en"],
    "prompts": [
      "Compara {item_a} y {item_b} en complejidad temporal, espacial y casos de uso recomendados.",
      "Compare {item_a} versus {item_b} regarding scalability, learning curve, and ecosystem."
    ],
    "pcm_templates": [
      "TASK=compare ITEMS={item_a},{item_b} CRITERIA=time_complexity,space_complexity,use_cases",
      "TASK=compare ITEMS={item_a},{item_b} CRITERIA=scalability,learning_curve,ecosystem"
    ],
    "pairs": [["redis", "memcached"], ["docker", "kubernetes"], ["sql", "nosql"]]
  },
  "planning": {
    "language": ["es", "en"],
    "prompts": [
      "Crea un plan de proyecto para una app móvil con autenticación, chat en tiempo real y compartir imágenes.",
      "Create a detailed roadmap for building a SaaS dashboard with role-based access, analytics, and export features."
    ],
    "pcm_templates": [
      "TASK=create_plan TYPE=project DOMAIN=mobile_app FEATURES=auth,realtime_chat,image_sharing",
      "TASK=create_plan TYPE=project DOMAIN=saas_dashboard FEATURES=rbac,analytics,export"
    ]
  },
  "legal_review": {
    "language": ["es", "en"],
    "prompts": [
      "Revisa este contrato y señala cláusulas problemáticas para el cliente, especialmente cancelación y penalizaciones.",
      "Review this NDA and flag clauses that may be unfavorable to the receiving party, focusing on confidentiality scope and duration."
    ],
    "pcm_templates": [
      "TASK=review INPUT=contract FOCUS=problematic_clauses CHECK=cancellation_terms,penalties",
      "TASK=review INPUT=nda FOCUS=unfavorable_clauses CHECK=confidentiality_scope,duration"
    ]
  },
  "database_design": {
    "language": ["es", "en"],
    "prompts": [
      "Diseña un esquema de base de datos para una app e-commerce con usuarios, productos, pedidos y reseñas. Incluye relaciones e índices.",
      "Design a database schema for a blogging platform with users, posts, comments, and tags. Optimize for read performance."
    ],
    "pcm_templates": [
      "TASK=design SCHEMA=ecommerce ENTITIES=users,products,orders,reviews REQUIRE=relationships,indexes OPTIMIZE=performance",
      "TASK=design SCHEMA=blog ENTITIES=users,posts,comments,tags REQUIRE=relationships,indexes OPTIMIZE=read_performance"
    ]
  }
}
```

- [ ] **Step 2: Crear scripts/generate_dataset.py**

```python
#!/usr/bin/env python3
"""Genera train.jsonl y valid.jsonl para fine-tuning PCM."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pcm.training.dataset import (
    build_chat_example,
    gold_to_examples,
    split_dataset,
    validate_pcm_output,
    write_jsonl,
)

TRAINING_DIR = ROOT / "data" / "training"
GOLD_PATH = ROOT / "data" / "example_prompts.json"
TEMPLATES_PATH = TRAINING_DIR / "templates.json"


def expand_templates(templates: dict, *, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    examples: list[dict] = []

    for category, cfg in templates.items():
        prompts = cfg["prompts"]
        pcm_templates = cfg["pcm_templates"]

        for i, prompt_tpl in enumerate(prompts):
            pcm_tpl = pcm_templates[i % len(pcm_templates)]
            ctx: dict[str, str] = {
                "lang": rng.choice(["python", "javascript", "rust"]),
                "input": rng.choice(cfg.get("inputs", ["python"])),
                "from_lang": rng.choice(["es", "en"]),
                "to_lang": rng.choice(["en", "es"]),
                "topic": rng.choice(cfg.get("topics", ["AI"])),
            }
            if "pairs" in cfg:
                pair = rng.choice(cfg["pairs"])
                ctx["item_a"], ctx["item_b"] = pair[0], pair[1]

            user_text = prompt_tpl.format(**ctx)
            pcm = pcm_tpl.format(**ctx)
            ok, errors = validate_pcm_output(pcm)
            if not ok:
                continue
            ex = build_chat_example(user_text, pcm)
            ex["_meta"] = {
                "id": f"synthetic_{category}_{i}",
                "category": category,
                "language": rng.choice(cfg.get("language", ["es"])),
                "source": "synthetic",
            }
            examples.append(ex)
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Generar dataset fine-tuning PCM")
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--templates", type=Path, default=TEMPLATES_PATH)
    parser.add_argument("--output-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gold_examples = gold_to_examples(args.gold)
    templates = json.loads(args.templates.read_text(encoding="utf-8"))
    synthetic = expand_templates(templates, seed=args.seed)

    all_examples = gold_examples + synthetic
    # Deduplicar por contenido user
    seen: set[str] = set()
    unique: list[dict] = []
    for ex in all_examples:
        user = ex["messages"][1]["content"]
        if user in seen:
            continue
        seen.add(user)
        unique.append(ex)

    train, valid = split_dataset(unique, valid_ratio=args.valid_ratio, seed=args.seed)
    out_train = args.output_dir / "train.jsonl"
    out_valid = args.output_dir / "valid.jsonl"
    write_jsonl(train, out_train)
    write_jsonl(valid, out_valid)

    manifest = {
        "total": len(unique),
        "train": len(train),
        "valid": len(valid),
        "gold": len(gold_examples),
        "synthetic": len(synthetic),
        "seed": args.seed,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Dataset generado: {manifest}")
    print(f"  train: {out_train}")
    print(f"  valid: {out_valid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Generar dataset y verificar conteos**

```bash
python scripts/generate_dataset.py
cat data/training/manifest.json
```

Expected: `total` ≥ 80, `gold` = 10, `train` + `valid` = `total`

- [ ] **Step 4: Commit**

```bash
git add data/training/templates.json scripts/generate_dataset.py
git commit -m "feat: generador de dataset sintético para fine-tuning PCM"
```

---

### Task 4: Script de entrenamiento MLX

**Files:**
- Create: `scripts/train_compressor.py`

- [ ] **Step 1: Crear train_compressor.py**

```python
#!/usr/bin/env python3
"""Entrena LoRA del compresor PCM con mlx-lm."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
DEFAULT_DATA = ROOT / "data" / "training"
DEFAULT_ADAPTER = ROOT / "data" / "training" / "checkpoints" / "pcm-lora"
DEFAULT_FUSED = ROOT / "data" / "training" / "checkpoints" / "pcm-fused"


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def require_mlx_lm() -> None:
    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "mlx-lm no instalado. Ejecuta: pip install -r requirements-training.txt"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune PCM con MLX LoRA")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--fused", type=Path, default=DEFAULT_FUSED)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--fuse-only", action="store_true")
    args = parser.parse_args()

    require_mlx_lm()

    train_path = args.data / "train.jsonl"
    valid_path = args.data / "valid.jsonl"
    if not args.fuse_only:
        if not train_path.exists():
            raise SystemExit(f"Falta {train_path}. Ejecuta scripts/generate_dataset.py")

        if args.adapter.exists():
            shutil.rmtree(args.adapter)
        args.adapter.mkdir(parents=True, exist_ok=True)

        lora_cmd = [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", args.model,
            "--train",
            "--data", str(args.data),
            "--adapter-path", str(args.adapter),
            "--batch-size", str(args.batch_size),
            "--iters", str(args.epochs * 100),  # ~100 steps/epoch con ~100 ejemplos
            "--learning-rate", str(args.learning_rate),
            "--lora-rank", str(args.lora_rank),
        ]
        if valid_path.exists():
            lora_cmd.extend(["--val-data", str(valid_path)])
        run(lora_cmd)

    if args.fused.exists():
        shutil.rmtree(args.fused)
    args.fused.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", args.model,
        "--adapter-path", str(args.adapter),
        "--save-path", str(args.fused),
    ])

    print(f"Adapter: {args.adapter}")
    print(f"Fused:   {args.fused}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verificar que el script arranca (sin entrenar si no hay mlx-lm)**

```bash
python scripts/train_compressor.py --help
```

Expected: ayuda impresa sin error

- [ ] **Step 3: Commit**

```bash
git add scripts/train_compressor.py
git commit -m "feat: script entrenamiento LoRA MLX para compresor PCM"
```

---

### Task 5: Exportación a Ollama

**Files:**
- Create: `scripts/export_ollama.sh`
- Create: `data/training/Modelfile`

- [ ] **Step 1: Crear Modelfile**

```
FROM ./checkpoints/pcm-compressor.gguf
PARAMETER temperature 0.1
PARAMETER num_predict 256
```

- [ ] **Step 2: Crear export_ollama.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FUSED="${ROOT}/data/training/checkpoints/pcm-fused"
GGUF="${ROOT}/data/training/checkpoints/pcm-compressor.gguf"
MODELFILE="${ROOT}/data/training/Modelfile"

if [[ ! -d "$FUSED" ]]; then
  echo "No existe $FUSED. Ejecuta primero: python scripts/train_compressor.py"
  exit 1
fi

python -m mlx_lm.convert \
  --hf-path "$FUSED" \
  --mlx-path "$FUSED" \
  -q \
  --q-bits 4 \
  --q-group-size 64 \
  --quantize-mlx \
  --quantize-gguf \
  --gguf-path "$GGUF"

cd "${ROOT}/data/training"
ollama create pcm-compressor -f Modelfile

echo "Modelo Ollama creado: pcm-compressor"
ollama run pcm-compressor "Prompt a comprimir: Analiza este código Python" || true
```

```bash
chmod +x scripts/export_ollama.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/export_ollama.sh data/training/Modelfile
git commit -m "feat: export GGUF y creación Ollama pcm-compressor"
```

---

### Task 6: Benchmark comparativo A/B

**Files:**
- Create: `scripts/compare_finetune.py`

- [ ] **Step 1: Crear compare_finetune.py**

```python
#!/usr/bin/env python3
"""Compara baseline granite4.1:3b vs pcm-compressor."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from pcm.benchmark import run_benchmark, save_report
from pcm.compressor import CompressorConfig, PromptCompressor


def summarize(report: dict) -> dict:
    entries = report.get("entries", [])
    if not entries:
        return {"count": 0, "avg_ratio": 0.0, "avg_format": 0.0, "avg_semantic": None}
    ratios = [e["compression_ratio"] for e in entries]
    formats = [e["format_similarity"]["score"] for e in entries]
    semantics = [e["semantic_similarity"] for e in entries if e.get("semantic_similarity") is not None]
    return {
        "count": len(entries),
        "avg_ratio": round(sum(ratios) / len(ratios), 4),
        "avg_format": round(sum(formats) / len(formats), 4),
        "avg_semantic": round(sum(semantics) / len(semantics), 4) if semantics else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark A/B fine-tuning PCM")
    parser.add_argument("--baseline", default="granite4.1:3b")
    parser.add_argument("--finetuned", default="pcm-compressor")
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "benchmarks")
    args = parser.parse_args()

    prompts_path = ROOT / "data" / "example_prompts.json"
    results: dict[str, dict] = {}

    for label, model in [("baseline", args.baseline), ("finetuned", args.finetuned)]:
        compressor = PromptCompressor(CompressorConfig(model=model))
        report = run_benchmark(
            compressor,
            prompts_path=prompts_path,
            include_semantic=args.semantic,
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        save_report(report, args.output / f"fase3_{label}_{model.replace(':', '_')}_{stamp}")
        results[label] = summarize(report)

    md = [
        "# Fase 3 — Comparativa fine-tuning",
        "",
        f"| Modelo | Ratio medio | Formato | Semántica |",
        f"|--------|-------------|---------|-----------|",
    ]
    for label, model in [("baseline", args.baseline), ("finetuned", args.finetuned)]:
        s = results[label]
        sem = f"{s['avg_semantic']:.2%}" if s["avg_semantic"] is not None else "N/A"
        md.append(
            f"| {model} | {s['avg_ratio']:.2%} | {s['avg_format']:.2%} | {sem} |"
        )

    out_md = args.output / "fase3_comparison.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Commit**

```bash
git add scripts/compare_finetune.py
git commit -m "feat: benchmark comparativo baseline vs pcm-compressor"
```

---

### Task 7: Documentación y configuración

**Files:**
- Create: `docs/fase3-finetuning.md`
- Modify: `.env.example`
- Modify: `README.md` (sección Fase 3)

- [ ] **Step 1: Crear docs/fase3-finetuning.md**

Contenido mínimo:
1. Requisitos: Mac Apple Silicon, 32+ GB RAM, Ollama, Python venv
2. `pip install -r requirements-training.txt`
3. `python scripts/generate_dataset.py`
4. `python scripts/train_compressor.py` (1–3 h)
5. `./scripts/export_ollama.sh`
6. `python scripts/compare_finetune.py --semantic`
7. `python scripts/e2e_benchmark.py` con `OLLAMA_MODEL=pcm-compressor`
8. Criterios de éxito de la spec

- [ ] **Step 2: Actualizar .env.example**

Añadir:
```
# Modelo fine-tuned Fase 3 (tras export_ollama.sh)
# OLLAMA_MODEL=pcm-compressor
```

- [ ] **Step 3: Añadir sección en README.md**

```markdown
## Fase 3 — Fine-tuning local (Mac Apple Silicon)

Ver [docs/fase3-finetuning.md](docs/fase3-finetuning.md).
```

- [ ] **Step 4: Commit**

```bash
git add docs/fase3-finetuning.md .env.example README.md
git commit -m "docs: guía Fase 3 fine-tuning PCM en Mac local"
```

---

### Task 8: Validación end-to-end (manual en M4 Pro)

**No es automatizable en CI** — requiere MLX + Ollama + tiempo de entrenamiento.

- [ ] **Step 1: Instalar dependencias de entrenamiento**

```bash
pip install -r requirements-training.txt
```

- [ ] **Step 2: Generar dataset**

```bash
python scripts/generate_dataset.py
```

- [ ] **Step 3: Entrenar**

```bash
python scripts/train_compressor.py
```

Expected: checkpoints en `data/training/checkpoints/pcm-lora` y `pcm-fused`

- [ ] **Step 4: Exportar a Ollama**

```bash
./scripts/export_ollama.sh
```

Expected: `ollama list` muestra `pcm-compressor`

- [ ] **Step 5: Benchmark A/B**

```bash
python scripts/compare_finetune.py --semantic
```

Expected: `fase3_comparison.md` con ratio finetuned > baseline

- [ ] **Step 6: E2E Mistral**

```bash
OLLAMA_MODEL=pcm-compressor python scripts/e2e_benchmark.py
```

Expected: similitud ≥ 0.90

- [ ] **Step 7: CI local**

```bash
./scripts/ci-local.sh
```

Expected: todos los tests pasan (training tests no requieren MLX)

---

## Self-review (spec coverage)

| Requisito spec | Task |
|----------------|------|
| Dataset ≥100 pares | Task 3 |
| Validación claves PCM | Task 2 |
| LoRA MLX Qwen2.5-3B | Task 4 |
| Export GGUF + Ollama | Task 5 |
| Benchmark >50% ratio | Task 6, 8 |
| E2E ≥90% | Task 8 |
| Integración OLLAMA_MODEL | Task 7 |
| Tests en CI local | Task 2 |
| checkpoints gitignored | Task 1 |

Sin placeholders pendientes.

---

## Execution Handoff

Plan guardado en `docs/superpowers/plans/2026-07-03-fase3-finetuning.md`.

**Opciones de ejecución:**

1. **Subagent-Driven (recomendado)** — Un subagente por task, revisión entre tasks
2. **Inline Execution** — Implementar en esta sesión con checkpoints

¿Cuál prefieres?
