#!/usr/bin/env python3
"""Entrena LoRA del compresor PCM con mlx-lm."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
DEFAULT_DATA = ROOT / "data" / "training"
DEFAULT_ADAPTER = ROOT / "data" / "training" / "checkpoints" / "pcm-lora"
DEFAULT_FUSED = ROOT / "data" / "training" / "checkpoints" / "pcm-fused"
DEFAULT_LORA_CONFIG = DEFAULT_DATA / "lora_config.yaml"


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
    except AttributeError as exc:
        if "__module__" in str(exc):
            raise SystemExit(
                "Incompatibilidad mlx-lm / transformers detectada.\n"
                "Ejecuta: pip install 'transformers>=4.43,<5.0' -r requirements-training.txt"
            ) from exc
        raise


def write_run_config(
    base_config: Path,
    output: Path,
    *,
    batch_size: int,
    iters: int,
    learning_rate: float,
    lora_rank: int,
) -> Path:
    """Genera config YAML con overrides de CLI para mlx_lm lora."""
    with base_config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["batch_size"] = batch_size
    cfg["iters"] = iters
    cfg["learning_rate"] = learning_rate
    cfg.setdefault("lora_parameters", {})["rank"] = lora_rank
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune PCM con MLX LoRA")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--fused", type=Path, default=DEFAULT_FUSED)
    parser.add_argument("--config", type=Path, default=DEFAULT_LORA_CONFIG)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--fuse-only", action="store_true")
    args = parser.parse_args()

    require_mlx_lm()

    train_path = args.data / "train.jsonl"
    if not args.fuse_only:
        if not train_path.exists():
            raise SystemExit(f"Falta {train_path}. Ejecuta scripts/generate_dataset.py")
        if not args.config.exists():
            raise SystemExit(f"Falta {args.config}")

        if args.adapter.exists():
            shutil.rmtree(args.adapter)
        args.adapter.mkdir(parents=True, exist_ok=True)

        run_config = write_run_config(
            args.config,
            args.adapter / "run_config.yaml",
            batch_size=args.batch_size,
            iters=args.epochs * 100,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
        )

        # mlx_lm 0.31+: valid.jsonl en --data se carga automáticamente;
        # lora rank y mask_prompt van en el YAML (-c).
        lora_cmd = [
            sys.executable,
            "-m",
            "mlx_lm",
            "lora",
            "-c",
            str(run_config),
            "--model",
            args.model,
            "--train",
            "--data",
            str(args.data),
            "--adapter-path",
            str(args.adapter),
            "--batch-size",
            str(args.batch_size),
            "--iters",
            str(args.epochs * 100),
            "--learning-rate",
            str(args.learning_rate),
        ]
        run(lora_cmd)

    if not args.adapter.exists():
        raise SystemExit(f"No existe adapter en {args.adapter}. Entrena primero sin --fuse-only.")

    if args.fused.exists():
        shutil.rmtree(args.fused)
    args.fused.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "-m",
            "mlx_lm",
            "fuse",
            "--model",
            args.model,
            "--adapter-path",
            str(args.adapter),
            "--save-path",
            str(args.fused),
            "--dequantize",
        ]
    )

    print(f"Adapter: {args.adapter}")
    print(f"Fused:   {args.fused}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
