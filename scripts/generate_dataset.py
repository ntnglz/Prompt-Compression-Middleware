#!/usr/bin/env python3
"""Genera train.jsonl y valid.jsonl para fine-tuning PCM."""

from __future__ import annotations

import argparse
import json
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


def _translation_pairs() -> list[tuple[str, str]]:
    return [("es", "en"), ("en", "es")]


def _context_variants(category: str, cfg: dict) -> list[dict[str, str]]:
    """Genera combinaciones deterministas para alcanzar ~90 sintéticos únicos."""
    if category == "summarization" and "documents" in cfg:
        return [{}]

    if category == "translation":
        variants: list[dict[str, str]] = []
        for domain in cfg.get("domains", ["technical"]):
            for from_lang, to_lang in _translation_pairs():
                variants.append({"from_lang": from_lang, "to_lang": to_lang, "domain": domain})
        return variants

    if "pairs" in cfg:
        return [
            {"item_a": pair[0], "item_b": pair[1]}
            for pair in cfg["pairs"]
        ]

    if "topics" in cfg:
        return [{"topic": topic} for topic in cfg["topics"]]

    if "inputs" in cfg:
        return [{"input": inp, "lang": inp} for inp in cfg["inputs"]]

    return [{}]


def expand_templates(templates: dict, *, seed: int = 42) -> list[dict]:
    examples: list[dict] = []

    for category, cfg in templates.items():
        if category == "summarization" and "documents" in cfg:
            for doc_idx, doc in enumerate(cfg["documents"]):
                pcm = doc["pcm"]
                ok, errors = validate_pcm_output(pcm)
                if not ok:
                    continue
                ex = build_chat_example(doc["prompt"], pcm)
                ex["_meta"] = {
                    "id": f"synthetic_{category}_{doc_idx}",
                    "category": category,
                    "language": "es" if doc_idx % 2 == 0 else "en",
                    "source": "synthetic",
                }
                examples.append(ex)
            continue

        prompts = cfg["prompts"]
        pcm_templates = cfg["pcm_templates"]

        for variant_idx, base_ctx in enumerate(_context_variants(category, cfg)):
            for i, prompt_tpl in enumerate(prompts):
                pcm_tpl = pcm_templates[i % len(pcm_templates)]
                ctx = {
                    "lang": base_ctx.get("lang", "python"),
                    "input": base_ctx.get("input", "python"),
                    "from_lang": base_ctx.get("from_lang", "es"),
                    "to_lang": base_ctx.get("to_lang", "en"),
                    "domain": base_ctx.get("domain", "technical"),
                    "topic": base_ctx.get("topic", "AI"),
                    "item_a": base_ctx.get("item_a", "item_a"),
                    "item_b": base_ctx.get("item_b", "item_b"),
                }

                user_text = prompt_tpl.format(**ctx)
                pcm = pcm_tpl.format(**ctx)
                ok, errors = validate_pcm_output(pcm)
                if not ok:
                    continue
                ex = build_chat_example(user_text, pcm)
                ex["_meta"] = {
                    "id": f"synthetic_{category}_{variant_idx}_{i}",
                    "category": category,
                    "language": "es" if i % 2 == 0 else "en",
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
