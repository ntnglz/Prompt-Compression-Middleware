"""Generación y validación de dataset JSONL para fine-tuning PCM."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from pcm.compression_prompts import PCM_SYSTEM_FULL, PCM_SYSTEM_GLOSSARY

FIELD_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=")

# Claves del glosario PCM (extraídas del system prompt)
VALID_PCM_KEYS = frozenset({
    "TASK", "INPUT", "CHECK", "FORMAT", "ORDER", "FROM", "TO",
    "STYLE", "TONE", "DOMAIN", "TOPIC", "TYPE", "ITEMS", "CRITERIA",
    "FEATURES", "FOCUS", "HIGHLIGHT", "INCLUDE", "AUDIENCE", "USE",
    "SCHEMA", "ENTITIES", "REQUIRE", "OPTIMIZE",
})

USER_PREFIX = "Prompt a comprimir:\n"


def normalize_user_text(text: str) -> str:
    """Normaliza texto de usuario para dedup y leakage checks."""
    t = text.strip()
    if t.startswith(USER_PREFIX.strip()):
        t = t.removeprefix(USER_PREFIX).strip()
    if t.startswith("Prompt a comprimir:"):
        t = t.split("\n", 1)[-1].strip()
    return " ".join(t.split())


def user_text_hash(text: str) -> str:
    return hashlib.sha256(normalize_user_text(text).encode()).hexdigest()


def extract_user_text(example: dict[str, Any]) -> str:
    """Extrae el texto usuario de un ejemplo chat."""
    for msg in example.get("messages", []):
        if msg.get("role") == "user":
            return normalize_user_text(msg["content"])
    raise ValueError("Ejemplo sin mensaje user")


def load_excluded_hashes(root: Path | None = None) -> set[str]:
    """Carga hashes de textos prohibidos en train (eval + gold + e2e)."""
    root = root or Path(__file__).resolve().parents[3]
    hashes: set[str] = set()

    manifest = root / "data" / "eval" / "excluded_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        hashes.update(data.get("hashes", []))
        return hashes

    for path in (
        root / "data" / "example_prompts.json",
        root / "data" / "eval" / "holdout_curated.json",
        root / "data" / "eval" / "holdout_synthetic.json",
    ):
        if not path.exists():
            continue
        items = json.loads(path.read_text(encoding="utf-8"))
        for item in items:
            text = item.get("text") or item.get("instruction", "")
            if text:
                hashes.add(user_text_hash(text))

    e2e = root / "data" / "e2e_prompts.json"
    if e2e.exists():
        for item in json.loads(e2e.read_text(encoding="utf-8")):
            instruction = item.get("instruction", "")
            if instruction:
                hashes.add(user_text_hash(instruction))

    return hashes


def check_leakage(train_items: list[dict], excluded: set[str]) -> list[dict]:
    """Devuelve ejemplos de train cuyo hash está en excluded."""
    leaks: list[dict] = []
    for item in train_items:
        text = extract_user_text(item)
        h = user_text_hash(text)
        if h in excluded:
            leaks.append({"user_text": text[:120], "user_hash": h})
    return leaks


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


def build_chat_example(
    user_text: str,
    assistant_pcm: str,
    *,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Construye un ejemplo chat para mlx-lm."""
    system = system_prompt or PCM_SYSTEM_FULL
    return {
        "messages": [
            {"role": "system", "content": system},
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
