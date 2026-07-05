#!/usr/bin/env python3
"""
Benchmark E2E: prompt natural → compresión PCM → Mistral → comparación de respuestas.

Uso:
    python scripts/e2e_benchmark.py --limit 1 -q
    python scripts/e2e_benchmark.py --compressor-model pcm-granite --reasoning-effort none -q
    python scripts/e2e_benchmark.py --prompts data/e2e_prompts_extensive.json -q
"""

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from pcm.compressor import CompressorConfig, PromptCompressor
from pcm.e2e_benchmark import MistralClient, run_e2e_benchmark, save_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark E2E PCM → Mistral (API directa)"
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=ROOT / "data" / "e2e_prompts.json",
        help="Dataset de prompts (por defecto e2e_prompts.json con payload)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "e2e",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--compressor-model",
        default=os.getenv("OLLAMA_MODEL", "granite4.1:3b"),
        help="Modelo Ollama para compresión (default: OLLAMA_MODEL o granite4.1:3b)",
    )
    parser.add_argument(
        "--glossary-only",
        action="store_true",
        default=None,
        help="PCM_SYSTEM_GLOSSARY sin few-shots (auto si modelo pcm-*)",
    )
    parser.add_argument(
        "--target-model",
        default="mistral-medium-3.5",
        help="Modelo Mistral API",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=["high", "none"],
        help="Reasoning de mistral-medium-3.5 (solo high o none; medium/low no soportados)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Máximo de tokens de salida Mistral (reasoning consume presupuesto)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limitar prompts")
    parser.add_argument(
        "--check-api",
        action="store_true",
        help="Solo verificar MISTRAL_API_KEY con una llamada mínima",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Menos logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not args.prompts.exists():
        print(f"Dataset no encontrado: {args.prompts}", file=sys.stderr)
        return 1

    try:
        mistral = MistralClient(
            model=args.target_model,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_tokens,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.check_api:
        result = mistral.complete(
            "Responde solo: OK",
            system_prompt="Eres un asistente breve.",
        )
        print(f"API OK — modelo={args.target_model}, respuesta={result.content[:80]!r}")
        return 0

    glossary_only = args.glossary_only
    if glossary_only is None:
        glossary_only = args.compressor_model.startswith("pcm-")

    compressor = PromptCompressor(
        config=CompressorConfig(
            model=args.compressor_model,
            glossary_only=glossary_only,
        )
    )

    print(f"Ejecutando benchmark E2E ({args.prompts.name})...")
    print(f"  Compresor: {args.compressor_model} (glossary_only={glossary_only})")
    print(f"  Mistral:   {args.target_model} (reasoning={args.reasoning_effort})")
    if args.limit:
        print(f"  Límite:    {args.limit} prompts")

    report = run_e2e_benchmark(
        compressor,
        mistral,
        args.prompts,
        limit=args.limit,
    )
    json_path, md_path = save_report(report, args.output)
    summary = report.summary

    print()
    print("=" * 60)
    print("RESULTADOS E2E")
    print("=" * 60)
    print(f"Prompts procesados:        {summary['total_prompts']}")
    print(f"Ratio compresión (media):  {summary['avg_compression_ratio']:.2%}")
    print(f"Tiempo compresión (media): {summary['avg_compression_time_ms']:.0f} ms")
    print(f"Tiempo Mistral original:   {summary['avg_original_llm_time_ms']:.0f} ms")
    print(f"Tiempo Mistral comprimido: {summary['avg_compressed_llm_time_ms']:.0f} ms")
    print(f"Similitud respuestas:      {summary['avg_response_similarity']:.2%}")
    print(f"Tokens input ahorrados:    {summary['input_tokens_saved']}")
    print(f"Coste original (total):    ${summary['total_cost_original_usd']:.4f}")
    print(f"Coste comprimido (total):  ${summary['total_cost_compressed_usd']:.4f}")
    if "avg_concise_output_tokens" in summary:
        concise_total = sum(
            e.concise_llm.estimated_cost_usd
            for e in report.entries
            if e.concise_llm is not None
        )
        print(f"Tokens output concise (media): {summary['avg_concise_output_tokens']:.0f}")
        print(
            f"Ahorro output concise vs PCM:  "
            f"{summary.get('avg_output_token_savings_pct', 0):.1f}%"
        )
        print(f"Coste concise (total):       ${concise_total:.4f}")
        print(
            f"Δ coste concise vs baseline: "
            f"${summary.get('avg_cost_delta_concise_vs_baseline', 0):.6f} / prompt"
        )
    print()
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
