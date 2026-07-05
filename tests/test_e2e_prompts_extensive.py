"""Validación del dataset E2E extenso (prompts largos anonimizados)."""

import json
from pathlib import Path

import pytest

from pcm.canonical import count_tokens

DATA = Path("data/e2e_prompts_extensive.json")

FORBIDDEN = [
    "RegistroVisitas",
    "AquiEstuve",
    "/Users/",
    "/Volumes/DevSSD/",
    "antonio",
]


@pytest.fixture
def cases():
    return json.loads(DATA.read_text(encoding="utf-8"))


def test_dataset_exists_and_has_entries(cases):
    assert len(cases) >= 6


def test_instruction_minimum_length(cases):
    for case in cases:
        assert len(case["instruction"]) >= 120, case["id"]


def test_no_pii_or_project_names(cases):
    blob = json.dumps(cases, ensure_ascii=False).lower()
    for token in FORBIDDEN:
        assert token.lower() not in blob, f"found {token!r}"


def test_payload_cases_have_tool_like_content(cases):
    with_payload = [c for c in cases if c.get("payload")]
    assert len(with_payload) >= 5
    for case in with_payload:
        assert len(case["payload"]) >= 40, case["id"]


def test_token_budget_extensive(cases):
    """Prompts extensos: instrucción+payload debe superar el corpus corto."""
    totals = []
    for case in cases:
        instruction = case["instruction"]
        payload = case.get("payload", "")
        totals.append(count_tokens(instruction) + count_tokens(payload))
    assert max(totals) >= 80
    assert sum(totals) / len(totals) >= 100
