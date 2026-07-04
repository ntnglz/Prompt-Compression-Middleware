#!/usr/bin/env python3
"""Benchmark A/B/C post fine-tune granite (Fase 3b)."""

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

from pcm.benchmark import BenchmarkReport, run_benchmark, save_report
from pcm.compressor import CompressorConfig, PromptCompressor
from pcm.e2e_benchmark import MistralClient, run_e2e_benchmark, save_report as save_e2e_report

OUTPUT = ROOT / "data" / "benchmarks"
EVAL_CURATED = ROOT / "data" / "eval" / "holdout_curated.json"
EVAL_SYNTHETIC = ROOT / "data" / "eval" / "holdout_synthetic.json"
E2E_PROMPTS = ROOT / "data" / "e2e_prompts.json"

CONFIGS = [
    ("baseline_full", "granite4.1:3b", False),
    ("baseline_glossary", "granite4.1:3b", True),
    ("finetuned_glossary", "pcm-granite", True),
]


def summarize(report: BenchmarkReport) -> dict:
    entries = report.entries
    if not entries:
        return {"count": 0, "avg_ratio": 0.0, "avg_format": 0.0, "avg_semantic": None}
    ratios = [e.compression_ratio for e in entries]
    formats = [e.format_similarity["score"] for e in entries]
    semantics = [e.semantic_similarity for e in entries if e.semantic_similarity is not None]
    return {
        "count": len(entries),
        "avg_ratio": round(sum(ratios) / len(ratios), 4),
        "avg_format": round(sum(formats) / len(formats), 4),
        "avg_semantic": round(sum(semantics) / len(semantics), 4) if semantics else None,
    }


def run_eval_set(
    label: str,
    model: str,
    glossary_only: bool,
    prompts_path: Path,
    *,
    include_semantic: bool,
    stamp: str,
) -> dict:
    compressor = PromptCompressor(
        config=CompressorConfig(model=model, glossary_only=glossary_only)
    )
    report = run_benchmark(
        compressor,
        prompts_path=prompts_path,
        include_semantic=include_semantic,
    )
    save_report(
        report,
        OUTPUT / f"fase3b_{label}_{prompts_path.stem}_{stamp}",
    )
    return summarize(report)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validación Fase 3b granite cloud")
    parser.add_argument("--finetuned", default="pcm-granite")
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--e2e", action="store_true")
    parser.add_argument("--target-model", default="mistral-medium-3.5")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    eval_sets = [
        ("holdout_curated", EVAL_CURATED),
        ("holdout_synthetic", EVAL_SYNTHETIC),
    ]
    for _, path in eval_sets:
        if not path.exists():
            print(f"Falta eval set: {path}", file=sys.stderr)
            print("Ejecuta: python scripts/build_dataset_v2.py --eval-output data/eval/holdout_synthetic.json", file=sys.stderr)
            return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    configs = [
        ("baseline_full", "granite4.1:3b", False),
        ("baseline_glossary", "granite4.1:3b", True),
        ("finetuned_glossary", args.finetuned, True),
    ]

    results: dict[str, dict[str, dict]] = {}
    for eval_name, eval_path in eval_sets:
        results[eval_name] = {}
        for label, model, glossary in configs:
            print(f"=== {eval_name} / {label} ({model}) ===")
            results[eval_name][label] = run_eval_set(
                label,
                model,
                glossary,
                eval_path,
                include_semantic=args.semantic,
                stamp=stamp,
            )

    e2e_results = None
    if args.e2e and E2E_PROMPTS.exists():
        try:
            mistral = MistralClient(model=args.target_model)
        except RuntimeError as exc:
            print(f"E2E omitido: {exc}", file=sys.stderr)
        else:
            e2e_results = {}
            for label, model, glossary in configs:
                compressor = PromptCompressor(
                    config=CompressorConfig(model=model, glossary_only=glossary)
                )
                report = run_e2e_benchmark(
                    compressor,
                    mistral,
                    E2E_PROMPTS,
                    limit=args.limit,
                )
                save_e2e_report(report, ROOT / "data" / "e2e" / f"fase3b_{label}")
                e2e_results[label] = report.summary

    md = [
        "# Validación Fase 3b — Granite Cloud (Enfoque A)",
        "",
        f"Generado: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Configuraciones",
        "",
        "| Label | Modelo | System prompt |",
        "|-------|--------|---------------|",
        "| baseline_full | granite4.1:3b | FULL (few-shots) |",
        "| baseline_glossary | granite4.1:3b | GLOSSARY |",
        f"| finetuned_glossary | {args.finetuned} | GLOSSARY |",
        "",
    ]

    for eval_name, _ in eval_sets:
        md.extend([f"## Eval: `{eval_name}`", "", "| Label | Ratio | Formato | Semántica |", "|-------|-------|---------|-----------|"])
        for label, _, _ in configs:
            r = results[eval_name][label]
            sem = f"{r['avg_semantic']:.2%}" if r.get("avg_semantic") is not None else "N/A"
            md.append(f"| {label} | {r['avg_ratio']:.2%} | {r['avg_format']:.2%} | {sem} |")
        md.append("")

    if e2e_results:
        md.extend([
            "## E2E Mistral",
            "",
            "| Label | Similitud | Ratio |",
            "|-------|-----------|-------|",
        ])
        for label in e2e_results:
            s = e2e_results[label]
            md.append(
                f"| {label} | {s['avg_response_similarity']:.2%} | "
                f"{s['avg_compression_ratio']:.2%} |"
            )
        md.append("")

    md.extend([
        "## Criterio go/no-go (Enfoque A)",
        "",
        "- 0 leakage en train (check_dataset_leakage.py)",
        "- holdout_curated formato ≥ 70%",
        "- Pipeline cloud completado sin errores",
        "",
    ])

    out_md = OUTPUT / "fase3b_validation.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUTPUT / f"fase3b_results_{stamp}.json").write_text(
        json.dumps({"eval": results, "e2e": e2e_results}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nInforme: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
