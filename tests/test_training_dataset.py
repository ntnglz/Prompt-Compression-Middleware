import json
from pathlib import Path

import pytest

from pcm.training.dataset import (
    VALID_PCM_KEYS,
    build_chat_example,
    load_gold_prompts,
    split_dataset,
    validate_pcm_output,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "example_prompts.json"


def test_valid_pcm_keys_includes_task():
    assert "TASK" in VALID_PCM_KEYS


def test_validate_pcm_output_accepts_gold_example():
    ok, errors = validate_pcm_output(
        "TASK=review INPUT=python CHECK=race,leak,perf FORMAT=markdown ORDER=severity"
    )
    assert ok is True
    assert errors == []


def test_validate_pcm_output_rejects_missing_task():
    ok, errors = validate_pcm_output("INPUT=python")
    assert ok is False
    assert any("TASK" in e for e in errors)


def test_validate_pcm_output_rejects_unknown_key():
    ok, errors = validate_pcm_output("TASK=review TARGET=client")
    assert ok is False
    assert any("TARGET" in e for e in errors)


def test_build_chat_example_structure():
    example = build_chat_example(
        "Analiza este código Python",
        "TASK=review INPUT=python",
    )
    assert len(example["messages"]) == 3
    assert example["messages"][0]["role"] == "system"
    assert example["messages"][1]["role"] == "user"
    assert "Prompt a comprimir:" in example["messages"][1]["content"]
    assert example["messages"][2]["content"].startswith("TASK=")


def test_load_gold_prompts_returns_ten():
    prompts = load_gold_prompts(GOLD)
    assert len(prompts) == 10
    assert all("text" in p and "expected_compression" in p for p in prompts)


def test_split_dataset_stratified():
    items = [{"_meta": {"category": "a"}, "messages": []} for _ in range(10)]
    items += [{"_meta": {"category": "b"}, "messages": []} for _ in range(10)]
    train, valid = split_dataset(items, valid_ratio=0.2, seed=42)
    assert len(train) + len(valid) == 20
    assert len(valid) >= 2


def test_build_chat_example_glossary_system():
    from pcm.compression_prompts import PCM_SYSTEM_GLOSSARY

    example = build_chat_example(
        "test",
        "TASK=explain TOPIC=test",
        system_prompt=PCM_SYSTEM_GLOSSARY,
    )
    system = example["messages"][0]["content"]
    assert "EJEMPLOS:" not in system
    assert "GLOSARIO PCM" in system


def test_write_jsonl_roundtrip(tmp_path):
    examples = [build_chat_example("hola", "TASK=explain TOPIC=test")]
    path = tmp_path / "out.jsonl"
    write_jsonl(examples, path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["messages"][2]["content"].startswith("TASK=")
