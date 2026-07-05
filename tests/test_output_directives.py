import json
from pathlib import Path

from pcm.output_directives import build_output_directives, has_response_block

CASES = json.loads(
    Path("data/examples/output_directives_cases.json").read_text(encoding="utf-8")
)


def test_concise_contains_seven_rules():
    text = build_output_directives(response_lang="en", output_style="concise")
    assert text.startswith("RESPONSE:")
    assert "Language: en" in text
    assert "Answer only what was asked" in text
    assert "No greeting, politeness" in text
    assert "Do not restate or recap" in text
    assert "Do not describe your process" in text
    assert "shortest form" in text
    assert "one line stating what is missing" in text


def test_normal_matches_gold():
    text = build_output_directives(response_lang="es", output_style="normal")
    gold = next(c for c in CASES if c["id"] == "normal_es")
    assert text == gold["expected"]


def test_has_response_block_detects_existing():
    system = "TASK=review INPUT=python\n\nRESPONSE:\n- Language: en"
    assert has_response_block(system) is True
    assert has_response_block("TASK=review INPUT=python") is False
