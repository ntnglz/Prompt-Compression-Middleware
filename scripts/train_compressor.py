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
