"""Tests de la política de umbral de compresión."""

from pcm.compression_policy import CompressionPolicy


def test_skips_instruction_below_min_tokens():
    policy = CompressionPolicy(min_instruction_tokens=12)
    count = lambda text: 8

    ok, reason = policy.should_compress_instruction(
        "Explica qué es un middleware en una frase.",
        count,
    )

    assert ok is False
    assert reason.startswith("below_min_tokens:")


def test_allows_instruction_at_min_tokens():
    policy = CompressionPolicy(min_instruction_tokens=12)
    count = lambda text: 12

    ok, reason = policy.should_compress_instruction("instrucción larga", count)

    assert ok is True
    assert reason == ""


def test_rejects_compression_without_token_savings():
    policy = CompressionPolicy()
    count = lambda text: 20 if "TASK=" not in text else 25

    ok, reason = policy.should_apply_compression(
        "Explica qué es un middleware en una frase.",
        "TASK=explain TOPIC=middleware FORMAT=one_sentence EXTRA=verbose",
        count,
    )

    assert ok is False
    assert reason.startswith("no_token_savings:")


def test_accepts_compression_with_token_savings():
    policy = CompressionPolicy()
    count = lambda text: 30 if "TASK=" not in text else 10

    ok, reason = policy.should_apply_compression(
        "Analiza cuidadosamente este código Python buscando race conditions.",
        "TASK=review INPUT=python CHECK=race",
        count,
    )

    assert ok is True
    assert reason == ""
