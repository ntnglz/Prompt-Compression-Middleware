#!/usr/bin/env python3
"""
Benchmark PCM sobre data/example_prompts.json

Uso:
    python scripts/benchmark.py
    python scripts/benchmark.py --semantic
    python scripts/benchmark.py --limit 3
    python scripts/benchmark.py --output data/benchmarks
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from pcm.benchmark import model_slug, rebuild_index, run_benchmark, save_report
from pcm.compressor import CompressorConfig, PromptCompressor


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark de compresión PCM")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=ROOT / "data" / "example_prompts.json",
        help="Ruta al dataset de prompts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "benchmarks",
        help="Directorio de salida para informes",
    )
    parser.add_argument("--model", default="granite4.1:3b", help="Modelo Ollama")
    parser.add_argument(
        "--strategy",
        default="balanced",
        choices=["aggressive", "balanced", "conservative"],
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Incluir similitud semántica vía LLM (más lento)",
    )
    parser.add_argument(
        "--think",
        choices=["auto", "true", "false"],
        default="auto",
        help="Modo thinking de Ollama (auto: Qwen3 usa thinking, resto no)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limitar número de prompts")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Reconstruir index.json y leaderboard.md desde resultados guardados",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Menos logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.rebuild_index:
        count = rebuild_index(args.output)
        print(f"Índice reconstruido: {count} ejecuciones")
        print(f"Índice:   {args.output / 'index.json'}")
        print(f"Ranking:  {args.output / 'leaderboard.md'}")
        return 0

    if not args.prompts.exists():
        print(f"Dataset no encontrado: {args.prompts}", file=sys.stderr)
        return 1

    think_mode = {"auto": None, "true": True, "false": False}[args.think]

    compressor = PromptCompressor(
        config=CompressorConfig(
            model=args.model,
            strategy=args.strategy,
            think=think_mode,
        )
    )

    print(f"Ejecutando benchmark ({args.prompts.name})...")
    print(f"  Modelo: {args.model}")
    print(f"  Estrategia: {args.strategy}")
    print(f"  Thinking: {args.think}")
    print(f"  Similitud semántica: {'sí' if args.semantic else 'no'}")
    if args.limit:
        print(f"  Límite: {args.limit} prompts")

    report = run_benchmark(
        compressor,
        args.prompts,
        include_semantic=args.semantic,
        limit=args.limit,
    )

    json_path, md_path = save_report(report, args.output)
    summary = report.summary

    print()
    print("=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    print(f"Prompts procesados:     {summary['total_prompts']}")
    print(f"Ratio medio:            {summary['avg_compression_ratio']:.2%}")
    print(f"Ratio min / max:        {summary['min_compression_ratio']:.2%} / {summary['max_compression_ratio']:.2%}")
    print(f"Tokens ahorrados:       {summary['total_tokens_saved']}")
    print(f"Similitud formato:      {summary['avg_format_similarity']:.2%}")
    if summary.get("avg_semantic_similarity") is not None:
        print(f"Similitud semántica:    {summary['avg_semantic_similarity']:.2%}")
    print(f"Tiempo medio:           {summary['avg_processing_time_ms']:.0f} ms")
    print()
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")
    print(f"Latest: {json_path.parent / (model_slug(report.model) + '_latest.json')}")
    print(f"Índice: {args.output / 'index.json'}")
    print(f"Ranking: {args.output / 'leaderboard.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
