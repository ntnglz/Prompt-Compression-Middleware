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

from pcm.benchmark import BenchmarkReport, run_benchmark, save_report
from pcm.compressor import CompressorConfig, PromptCompressor


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark A/B fine-tuning PCM")
    parser.add_argument("--baseline", default="granite4.1:3b")
    parser.add_argument("--finetuned", default="pcm-compressor")
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "benchmarks")
    args = parser.parse_args()

    prompts_path = ROOT / "data" / "example_prompts.json"
    if not prompts_path.exists():
        print(f"Dataset no encontrado: {prompts_path}", file=sys.stderr)
        return 1

    results: dict[str, dict] = {}

    for label, model in [("baseline", args.baseline), ("finetuned", args.finetuned)]:
        compressor = PromptCompressor(config=CompressorConfig(model=model))
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
        "| Modelo | Ratio medio | Formato | Semántica |",
        "|--------|-------------|---------|-----------|",
    ]
    for label, model in [("baseline", args.baseline), ("finetuned", args.finetuned)]:
        s = results[label]
        sem = f"{s['avg_semantic']:.2%}" if s["avg_semantic"] is not None else "N/A"
        md.append(
            f"| {model} | {s['avg_ratio']:.2%} | {s['avg_format']:.2%} | {sem} |"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    out_md = args.output / "fase3_comparison.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
