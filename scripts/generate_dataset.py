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
