#!/usr/bin/env python3
"""Genera train/valid v2 (Enfoque A) con plantillas y teacher granite opcional."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pcm.compressor import CompressorConfig, PromptCompressor
from pcm.compression_prompts import PCM_SYSTEM_GLOSSARY
from pcm.training.dataset import (
    build_chat_example,
    load_excluded_hashes,
    split_dataset,
    user_text_hash,
    validate_pcm_output,
    write_jsonl,
)

V2_DIR = ROOT / "data" / "training" / "v2"
TRAIN_TEMPLATES = V2_DIR / "templates_train.json"
EVAL_TEMPLATES = V2_DIR / "templates_eval.json"
EVAL_SYNTHETIC = ROOT / "data" / "eval" / "holdout_synthetic.json"


def _translation_pairs() -> list[tuple[str, str]]:
    return [("es", "en"), ("en", "es")]


def _context_variants(category: str, cfg: dict) -> list[dict[str, str]]:
    if category == "summarization" and "documents" in cfg:
        return [{}]

    if category == "translation":
        variants: list[dict[str, str]] = []
        for domain in cfg.get("domains", ["technical"]):
            for from_lang, to_lang in _translation_pairs():
                variants.append({"from_lang": from_lang, "to_lang": to_lang, "domain": domain})
        return variants

    if "pairs" in cfg:
        return [{"item_a": p[0], "item_b": p[1]} for p in cfg["pairs"]]

    if "topics" in cfg:
        return [{"topic": t} for t in cfg["topics"]]

    if "service_types" in cfg:
        return [{"service_type": s} for s in cfg["service_types"]]

    if "datasets" in cfg:
        return [{"dataset": d} for d in cfg["datasets"]]

    if "frameworks" in cfg:
        return [{"framework": f} for f in cfg["frameworks"]]

    if "doc_types" in cfg:
        return [{"doc_type": d} for d in cfg["doc_types"]]

    if "targets" in cfg:
        return [{"target": t} for t in cfg["targets"]]

    if "severities" in cfg:
        return [{"severity": s} for s in cfg["severities"]]

    if "inputs" in cfg:
        return [{"input": inp, "lang": inp} for inp in cfg["inputs"]]

    return [{}]


def _default_ctx(base_ctx: dict[str, str]) -> dict[str, str]:
    return {
        "lang": base_ctx.get("lang", "python"),
        "input": base_ctx.get("input", "python"),
        "from_lang": base_ctx.get("from_lang", "es"),
        "to_lang": base_ctx.get("to_lang", "en"),
        "domain": base_ctx.get("domain", "technical"),
        "topic": base_ctx.get("topic", "AI"),
        "item_a": base_ctx.get("item_a", "item_a"),
        "item_b": base_ctx.get("item_b", "item_b"),
        "service_type": base_ctx.get("service_type", "rest_api"),
        "dataset": base_ctx.get("dataset", "sales_data"),
        "framework": base_ctx.get("framework", "pytest"),
        "doc_type": base_ctx.get("doc_type", "api_reference"),
        "target": base_ctx.get("target", "api_latency"),
        "severity": base_ctx.get("severity", "p1"),
    }


def expand_templates(
    templates: dict,
    *,
    eval_mode: bool = False,
    source: str = "synthetic",
    id_prefix: str = "v2",
) -> list[dict]:
    examples: list[dict] = []

    for category, cfg in templates.items():
        if category == "summarization" and "documents" in cfg:
            for doc_idx, doc in enumerate(cfg["documents"]):
                pcm = doc["pcm"]
                ok, _ = validate_pcm_output(pcm)
                if not ok:
                    continue
                ex = build_chat_example(
                    doc["prompt"],
                    pcm,
                    system_prompt=PCM_SYSTEM_GLOSSARY,
                )
                ex["_meta"] = {
                    "id": f"{id_prefix}_{category}_{doc_idx}",
                    "category": category,
                    "language": "es" if doc_idx % 2 == 0 else "en",
                    "source": source,
                }
                examples.append(ex)
            continue

        prompts = cfg["prompts"]
        pcm_templates = cfg["pcm_templates"]
        variants = [{}] if eval_mode else _context_variants(category, cfg)

        for variant_idx, base_ctx in enumerate(variants):
            ctx = _default_ctx(base_ctx)
            for i, prompt_tpl in enumerate(prompts):
                pcm_tpl = pcm_templates[i % len(pcm_templates)]
                user_text = prompt_tpl.format(**ctx)
                pcm = pcm_tpl.format(**ctx)
                ok, _ = validate_pcm_output(pcm)
                if not ok:
                    continue
                ex = build_chat_example(
                    user_text,
                    pcm,
                    system_prompt=PCM_SYSTEM_GLOSSARY,
                )
                ex["_meta"] = {
                    "id": f"{id_prefix}_{category}_{variant_idx}_{i}",
                    "category": category,
                    "language": "es" if i % 2 == 0 else "en",
                    "source": source,
                }
                examples.append(ex)
    return examples


def dedupe_and_filter(
    examples: list[dict],
    excluded: set[str],
) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for ex in examples:
        user = ex["messages"][1]["content"]
        h = user_text_hash(user)
        if h in seen or h in excluded:
            continue
        seen.add(h)
        ex["user_hash"] = h
        unique.append(ex)
    return unique


def apply_teacher(
    examples: list[dict],
    *,
    model: str = "granite4.1:3b",
    min_semantic: float = 0.85,
) -> list[dict]:
    compressor = PromptCompressor(
        CompressorConfig(
            model=model,
            strategy="aggressive",
            glossary_only=False,
            min_instruction_tokens=0,
        )
    )
    accepted: list[dict] = []
    for ex in examples:
        user_text = ex["messages"][1]["content"].split("\n", 1)[-1].strip()
        meta = ex.get("_meta", {})
        try:
            result = compressor.compress(user_text, strategy="aggressive")
            pcm = result.compressed_prompt
            ok, _ = validate_pcm_output(pcm)
            if not ok:
                continue
            comparison = compressor.compare_prompts(user_text, pcm)
            if comparison.semantic_similarity < min_semantic:
                continue
            new_ex = build_chat_example(
                user_text,
                pcm,
                system_prompt=PCM_SYSTEM_GLOSSARY,
            )
            new_ex["_meta"] = meta
            new_ex["user_hash"] = ex.get("user_hash", user_text_hash(user_text))
            accepted.append(new_ex)
        except Exception:
            continue
    return accepted


def examples_to_eval_json(examples: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for ex in examples:
        user = ex["messages"][1]["content"].split("\n", 1)[-1].strip()
        pcm = ex["messages"][2]["content"]
        meta = ex.get("_meta", {})
        rows.append(
            {
                "id": meta.get("id", "eval_unknown"),
                "text": user,
                "category": meta.get("category", "unknown"),
                "language": meta.get("language", "unknown"),
                "expected_compression": pcm,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset v2 Enfoque A (plantillas)")
    parser.add_argument("--templates", type=Path, default=TRAIN_TEMPLATES)
    parser.add_argument("--output-dir", type=Path, default=V2_DIR)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--with-teacher",
        action="store_true",
        help="Regenera labels con granite (requiere Ollama, lento)",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=None,
        help="Genera holdout_synthetic.json desde templates_eval",
    )
    parser.add_argument("--eval-templates", type=Path, default=EVAL_TEMPLATES)
    args = parser.parse_args()

    if args.eval_output is not None:
        eval_templates = json.loads(args.eval_templates.read_text(encoding="utf-8"))
        eval_examples = expand_templates(
            eval_templates,
            eval_mode=True,
            source="eval_synthetic",
            id_prefix="eval_syn",
        )
        rows = examples_to_eval_json(eval_examples)
        args.eval_output.parent.mkdir(parents=True, exist_ok=True)
        args.eval_output.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Eval synthetic: {args.eval_output} ({len(rows)} prompts)")
        return 0

    if not args.templates.exists():
        print(f"Plantillas no encontradas: {args.templates}", file=sys.stderr)
        return 1

    excluded = load_excluded_hashes(ROOT)
    templates = json.loads(args.templates.read_text(encoding="utf-8"))
    raw = expand_templates(templates, source="synthetic", id_prefix="v2a")
    unique = dedupe_and_filter(raw, excluded)

    if args.with_teacher:
        print(f"Teacher granite: procesando {len(unique)} ejemplos…")
        unique = apply_teacher(unique)
        print(f"Teacher aceptó {len(unique)} ejemplos")

    train, valid = split_dataset(unique, valid_ratio=args.valid_ratio, seed=args.seed)
    out_train = args.output_dir / "train.jsonl"
    out_valid = args.output_dir / "valid.jsonl"
    write_jsonl(train, out_train)
    write_jsonl(valid, out_valid)

    manifest = {
        "version": "v2a-templates-only",
        "total": len(unique),
        "train": len(train),
        "valid": len(valid),
        "excluded_hashes": len(excluded),
        "teacher": args.with_teacher,
        "seed": args.seed,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Dataset v2: {manifest}")
    print(f"  train: {out_train}")
    print(f"  valid: {out_valid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
