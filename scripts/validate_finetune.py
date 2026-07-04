#!/usr/bin/env python3
"""Validación real del fine-tune: holdout (valid.jsonl) + E2E Mistral A/B."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from pcm.benchmark import format_similarity, run_benchmark, save_report
from pcm.compressor import CompressorConfig, PromptCompressor
from pcm.e2e_benchmark import MistralClient, run_e2e_benchmark, save_report as save_e2e_report
from pcm.training.dataset import USER_PREFIX, read_jsonl

VALID_JSONL = ROOT / "data" / "training" / "valid.jsonl"
E2E_PROMPTS = ROOT / "data" / "e2e_prompts.json"
OUTPUT = ROOT / "data" / "benchmarks"


def valid_jsonl_to_prompts(path: Path) -> list[dict]:
    """Convierte valid.jsonl (chat MLX) a formato example_prompts.json."""
    prompts: list[dict] = []
    for i, example in enumerate(read_jsonl(path)):
        messages = example["messages"]
        user = next(m["content"] for m in messages if m["role"] == "user")
        expected = next(m["content"] for m in messages if m["role"] == "assistant")
        text = user.removeprefix(USER_PREFIX).strip()
        meta = example.get("_meta", {})
        prompts.append(
            {
                "id": meta.get("id", f"holdout_{i:03d}"),
                "text": text,
                "category": meta.get("category", "holdout"),
                "language": meta.get("language", "unknown"),
                "expected_compression": expected,
            }
        )
    return prompts


def write_holdout_json(prompts: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_benchmark(report) -> dict:
    entries = report.entries
    if not entries:
        return {"count": 0, "avg_ratio": 0.0, "avg_format": 0.0, "avg_semantic": None}
    return {
        "count": len(entries),
        "avg_ratio": sum(e.compression_ratio for e in entries) / len(entries),
        "avg_format": sum(e.format_similarity["score"] for e in entries) / len(entries),
        "avg_semantic": (
            sum(e.semantic_similarity for e in entries if e.semantic_similarity is not None)
            / max(1, sum(1 for e in entries if e.semantic_similarity is not None))
        )
        if any(e.semantic_similarity is not None for e in entries)
        else None,
    }


def compare_entries(base_entries, fine_entries) -> dict:
    identical = 0
    format_wins = 0
    ratio_wins = 0
    details: list[str] = []
    for b, f in zip(base_entries, fine_entries):
        same = b.compressed_prompt == f.compressed_prompt
        if same:
            identical += 1
        if f.format_similarity["score"] > b.format_similarity["score"]:
            format_wins += 1
        if f.compression_ratio > b.compression_ratio:
            ratio_wins += 1
        if not same:
            details.append(
                f"- **{b.id}**: granite `{b.compressed_prompt}` | pcm `{f.compressed_prompt}`"
            )
    n = len(base_entries)
    return {
        "identical": identical,
        "total": n,
        "format_wins_pcm": format_wins,
        "ratio_wins_pcm": ratio_wins,
        "diff_details": details,
    }


def run_holdout(args) -> tuple[dict, dict, dict]:
    prompts = valid_jsonl_to_prompts(args.valid)
    holdout_path = OUTPUT / "holdout_prompts.json"
    write_holdout_json(prompts, holdout_path)

    results: dict[str, dict] = {}
    reports: dict[str, object] = {}

    for label, model in [("granite", args.baseline), ("pcm", args.finetuned)]:
        compressor = PromptCompressor(config=CompressorConfig(model=model))
        report = run_benchmark(
            compressor,
            prompts_path=holdout_path,
            include_semantic=args.semantic,
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        save_report(report, OUTPUT / f"holdout_{label}_{model.replace(':', '_')}_{stamp}")
        results[label] = summarize_benchmark(report)
        reports[label] = report

    diff = compare_entries(reports["granite"].entries, reports["pcm"].entries)
    return results, diff, {"holdout_path": str(holdout_path), "count": len(prompts)}


def run_e2e_pair(args) -> dict | None:
    try:
        mistral = MistralClient(
            model=args.target_model,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_tokens,
        )
    except RuntimeError as exc:
        print(f"E2E omitido: {exc}", file=sys.stderr)
        return None

    out: dict[str, dict] = {}
    for label, model in [("granite", args.baseline), ("pcm", args.finetuned)]:
        compressor = PromptCompressor(config=CompressorConfig(model=model))
        report = run_e2e_benchmark(
            compressor,
            mistral,
            args.e2e_prompts,
            limit=args.limit,
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        save_e2e_report(report, ROOT / "data" / "e2e" / f"validate_{label}")
        s = report.summary
        out[label] = {
            "avg_compression_ratio": s["avg_compression_ratio"],
            "avg_response_similarity": s["avg_response_similarity"],
            "input_tokens_saved": s["input_tokens_saved"],
            "total_cost_original_usd": s["total_cost_original_usd"],
            "total_cost_compressed_usd": s["total_cost_compressed_usd"],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Validación holdout + E2E fine-tune PCM")
    parser.add_argument("--valid", type=Path, default=VALID_JSONL)
    parser.add_argument("--e2e-prompts", type=Path, default=E2E_PROMPTS)
    parser.add_argument("--baseline", default="granite4.1:3b")
    parser.add_argument("--finetuned", default="pcm-compressor")
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--e2e", action="store_true", help="Ejecutar E2E Mistral (requiere API key)")
    parser.add_argument("--target-model", default="mistral-medium-3.5")
    parser.add_argument("--reasoning-effort", default="none", choices=["high", "none"])
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.valid.exists():
        print(f"No existe {args.valid}. Ejecuta scripts/generate_dataset.py", file=sys.stderr)
        return 1

    print("=== Holdout (valid.jsonl) ===")
    holdout_results, diff, meta = run_holdout(args)
    print(f"Prompts holdout: {meta['count']}")
    for label, model in [("granite", args.baseline), ("pcm", args.finetuned)]:
        r = holdout_results[label]
        sem = f"{r['avg_semantic']:.2%}" if r.get("avg_semantic") is not None else "N/A"
        print(
            f"  {model}: ratio={r['avg_ratio']:.2%} "
            f"formato={r['avg_format']:.2%} semántica={sem}"
        )
    print(
        f"  Salidas idénticas: {diff['identical']}/{diff['total']} | "
        f"pcm gana formato: {diff['format_wins_pcm']} | pcm gana ratio: {diff['ratio_wins_pcm']}"
    )

    e2e_results = None
    if args.e2e:
        print("\n=== E2E Mistral ===")
        e2e_results = run_e2e_pair(args)
        if e2e_results:
            for label, model in [("granite", args.baseline), ("pcm", args.finetuned)]:
                e = e2e_results[label]
                print(
                    f"  {model}: similitud={e['avg_response_similarity']:.2%} "
                    f"ratio={e['avg_compression_ratio']:.2%} "
                    f"tokens_ahorrados={e['input_tokens_saved']}"
                )

    md = [
        "# Validación fine-tune PCM (holdout + E2E)",
        "",
        f"Generado: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Holdout (`valid.jsonl`, sin overlap con gold benchmark)",
        "",
        f"Prompts: **{meta['count']}**",
        "",
        "| Modelo | Ratio | Formato | Semántica |",
        "|--------|-------|---------|-----------|",
    ]
    for label, model in [("granite", args.baseline), ("pcm", args.finetuned)]:
        r = holdout_results[label]
        sem = f"{r['avg_semantic']:.2%}" if r.get("avg_semantic") is not None else "N/A"
        md.append(
            f"| {model} | {r['avg_ratio']:.2%} | {r['avg_format']:.2%} | {sem} |"
        )
    md.extend(
        [
            "",
            f"- Salidas idénticas granite vs pcm: **{diff['identical']}/{diff['total']}**",
            f"- pcm gana en formato: **{diff['format_wins_pcm']}**",
            f"- pcm gana en ratio: **{diff['ratio_wins_pcm']}**",
        ]
    )
    if diff["diff_details"]:
        md.extend(["", "### Diferencias de salida", ""] + diff["diff_details"][:15])

    if e2e_results:
        md.extend(
            [
                "",
                "## E2E Mistral (`e2e_prompts.json`)",
                "",
                "| Compresor | Similitud respuesta | Ratio compresión | Tokens ahorrados |",
                "|-----------|---------------------|------------------|------------------|",
            ]
        )
        for label, model in [("granite", args.baseline), ("pcm", args.finetuned)]:
            e = e2e_results[label]
            md.append(
                f"| {model} | {e['avg_response_similarity']:.2%} | "
                f"{e['avg_compression_ratio']:.2%} | {e['input_tokens_saved']} |"
            )

    out_md = OUTPUT / "fase3_validation.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nInforme: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
