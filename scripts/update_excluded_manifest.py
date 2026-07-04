#!/usr/bin/env python3
"""Regenera data/eval/excluded_manifest.json con hashes prohibidos en train."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pcm.training.dataset import USER_PREFIX, read_jsonl, user_text_hash


def collect_hashes(root: Path) -> dict[str, str]:
    """Devuelve hash → source label."""
    mapping: dict[str, str] = {}

    def add(text: str, source: str) -> None:
        if text.strip():
            mapping[user_text_hash(text)] = source

    for path, field in (
        (root / "data" / "example_prompts.json", "text"),
        (root / "data" / "eval" / "holdout_curated.json", "text"),
        (root / "data" / "eval" / "holdout_synthetic.json", "text"),
    ):
        if not path.exists():
            continue
        for item in json.loads(path.read_text(encoding="utf-8")):
            add(item.get(field, ""), path.name)

    e2e = root / "data" / "e2e_prompts.json"
    if e2e.exists():
        for item in json.loads(e2e.read_text(encoding="utf-8")):
            add(item.get("instruction", ""), "e2e_prompts.json")

    valid_v1 = root / "data" / "training" / "valid.jsonl"
    if valid_v1.exists():
        for ex in read_jsonl(valid_v1):
            for msg in ex["messages"]:
                if msg["role"] == "user":
                    text = msg["content"].removeprefix(USER_PREFIX).strip()
                    add(text, "valid.jsonl")

    return mapping


def main() -> int:
    mapping = collect_hashes(ROOT)
    manifest = {
        "version": "v2a",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(mapping),
        "hashes": sorted(mapping.keys()),
        "sources": mapping,
    }
    out = ROOT / "data" / "eval" / "excluded_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Manifest: {out} ({manifest['count']} hashes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
