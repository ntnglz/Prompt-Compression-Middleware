#!/usr/bin/env python3
"""Detecta overlap entre train.jsonl y conjuntos de evaluación congelados."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pcm.training.dataset import (
    check_leakage,
    extract_user_text,
    load_excluded_hashes,
    read_jsonl,
    user_text_hash,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verificar leakage train vs eval")
    parser.add_argument(
        "--train",
        type=Path,
        default=ROOT / "data" / "training" / "v2" / "train.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "eval" / "excluded_manifest.json",
    )
    args = parser.parse_args()

    if not args.train.exists():
        print(f"Train no encontrado: {args.train}", file=sys.stderr)
        return 1

    excluded = load_excluded_hashes(ROOT)
    train_items = read_jsonl(args.train)
    leaks = check_leakage(train_items, excluded)

    if leaks:
        print(f"LEAKAGE DETECTADO: {len(leaks)} ejemplos", file=sys.stderr)
        for leak in leaks[:10]:
            print(f"  hash={leak['user_hash'][:12]}… text={leak['user_text']!r}", file=sys.stderr)
        if len(leaks) > 10:
            print(f"  … y {len(leaks) - 10} más", file=sys.stderr)
        return 1

    print(f"OK: 0 leakage en {len(train_items)} ejemplos ({len(excluded)} hashes excluidos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
