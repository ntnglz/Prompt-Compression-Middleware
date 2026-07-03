"""Tests del módulo de benchmark (sin Ollama)."""

from pcm.benchmark import (
    format_similarity,
    model_slug,
    parse_pcm_fields,
    rebuild_index,
    render_leaderboard,
    render_markdown_report,
    BenchmarkReport,
    BenchmarkEntry,
    save_index,
)


def test_parse_pcm_fields():
    text = "TASK=review INPUT=python CHECK=race,leak,perf FORMAT=markdown"
    fields = parse_pcm_fields(text)

    assert fields["TASK"] == "review"
    assert fields["INPUT"] == "python"
    assert fields["CHECK"] == "race,leak,perf"
    assert fields["FORMAT"] == "markdown"


def test_format_similarity_exact_match():
    expected = "TASK=translate FROM=es TO=en STYLE=formal"
    actual = "TASK=translate FROM=es TO=en STYLE=formal"
    result = format_similarity(expected, actual)

    assert result["score"] == 1.0
    assert result["missing_keys"] == []


def test_format_similarity_partial_match():
    expected = "TASK=review INPUT=python CHECK=race,leak,perf FORMAT=markdown ORDER=severity"
    actual = "TASK=analyze INPUT=python CHECK=security,performance"
    result = format_similarity(expected, actual)

    assert 0 < result["score"] < 1
    assert "FORMAT" in result["missing_keys"]
    assert "ORDER" in result["missing_keys"]


def test_model_slug():
    assert model_slug("qwen3:4b") == "qwen3_4b"
    assert model_slug("gemma3:4b") == "gemma3_4b"


def test_render_leaderboard():
    markdown = render_leaderboard(
        [
            {
                "model": "qwen3:4b",
                "strategy": "balanced",
                "generated_at": "2026-07-03T15:00:00+00:00",
                "summary": {
                    "avg_semantic_similarity": 0.98,
                    "avg_format_similarity": 1.0,
                    "avg_compression_ratio": 0.4,
                    "avg_processing_time_ms": 15000,
                },
                "json_path": "data/benchmarks/runs/qwen3_4b/qwen3_4b_latest.json",
            }
        ]
    )

    assert "qwen3:4b" in markdown
    assert "98%" in markdown


def test_render_markdown_report():
    report = BenchmarkReport(
        generated_at="2026-07-03T12:00:00+00:00",
        model="qwen3:4b",
        strategy="balanced",
        include_semantic=False,
        entries=[
            BenchmarkEntry(
                id="prompt_001",
                category="code_review",
                language="es",
                original_prompt="Analiza este código",
                expected_compression="TASK=review INPUT=python",
                compressed_prompt="TASK=analyze INPUT=python",
                original_tokens=10,
                compressed_tokens=6,
                compression_ratio=0.4,
                processing_time_ms=100.0,
                format_similarity=format_similarity(
                    "TASK=review INPUT=python",
                    "TASK=analyze INPUT=python",
                ),
            )
        ],
    )

    markdown = render_markdown_report(report)

    assert "# Benchmark PCM" in markdown
    assert "prompt_001" in markdown
    assert "TASK=analyze INPUT=python" in markdown
