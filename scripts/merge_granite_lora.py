#!/usr/bin/env python3
"""Fusiona adapter LoRA granite con el modelo base para exportar a Ollama."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTER = ROOT / "data" / "training" / "v2" / "checkpoints" / "granite-lora"
DEFAULT_OUT = ROOT / "data" / "training" / "v2" / "checkpoints" / "granite-merged"
DEFAULT_BASE = "ibm-granite/granite-3.3-2b-instruct"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fusionar LoRA granite")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base", default=DEFAULT_BASE)
    args = parser.parse_args()

    if not (args.adapter / "adapter_model.safetensors").exists():
        print(f"Adapter no encontrado: {args.adapter}", file=sys.stderr)
        return 1

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Base:    {args.base}")
    print(f"Adapter: {args.adapter}")
    print(f"Salida:  {args.output}")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    dtype = torch.float16
    if torch.backends.mps.is_available():
        device_map = "auto"
        print("Dispositivo: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device_map = "auto"
        print("Dispositivo: CUDA")
    else:
        device_map = "cpu"
        print("Dispositivo: CPU (lento)")

    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    merged = model.merge_and_unload()

    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)

    size_mb = sum(f.stat().st_size for f in args.output.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"OK: modelo fusionado en {args.output} ({size_mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
