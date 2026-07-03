"""Tests del benchmark E2E (sin llamadas reales a Mistral)."""

from pcm.e2e_benchmark import (
    E2EEntry,
    E2EReport,
    LLMCallResult,
    MISTRAL_PCM_SYSTEM_PROMPT,
    _extract_message_content,
    build_compressed_user_prompt,
    render_markdown_report,
    resolve_e2e_prompt_parts,
)


def test_extract_message_content_string():
    assert _extract_message_content("  hola  ") == "hola"


def test_extract_message_content_reasoning_chunks():
    chunks = [
        {"type": "thinking", "text": "razonando..."},
        {"type": "text", "text": "OK"},
    ]
    assert _extract_message_content(chunks) == "OK"

def test_extract_message_content_only_thinking():
    chunks = [{"type": "thinking", "text": "solo razonamiento interno"}]
    assert _extract_message_content(chunks) == ""


def test_mistral_pcm_system_prompt_mentions_pcm():
    assert "PCM" in MISTRAL_PCM_SYSTEM_PROMPT
    assert "CLAVE=valor" in MISTRAL_PCM_SYSTEM_PROMPT
    assert "bloque de código" in MISTRAL_PCM_SYSTEM_PROMPT


def test_resolve_e2e_prompt_parts_with_payload():
    item = {
        "instruction": "Analiza este código Python.",
        "payload": "x = 1",
        "payload_type": "code",
        "payload_lang": "python",
    }
    compress_input, full = resolve_e2e_prompt_parts(item)
    assert compress_input == "Analiza este código Python."
    assert "```python" in full
    assert "x = 1" in full


def test_resolve_e2e_prompt_parts_legacy_text():
    item = {"text": "Prompt completo sin payload."}
    compress_input, full = resolve_e2e_prompt_parts(item)
    assert compress_input == full == "Prompt completo sin payload."


def test_build_compressed_user_prompt_appends_code():
    item = {
        "payload": "print(1)",
        "payload_type": "code",
        "payload_lang": "python",
    }
    result = build_compressed_user_prompt(
        "TASK=review INPUT=python",
        item,
    )
    assert result.startswith("TASK=review INPUT=python")
    assert "```python" in result
    assert "print(1)" in result


def test_e2e_report_summary():
    entry = E2EEntry(
        id="prompt_001",
        category="code_review",
        language="es",
        original_prompt="Analiza este código",
        compressed_prompt="TASK=review INPUT=python",
        compression_time_ms=500.0,
        compression_ratio=0.4,
        original_llm=LLMCallResult(
            content="Informe original",
            processing_time_ms=2000.0,
            input_tokens=40,
            output_tokens=100,
            total_tokens=140,
            estimated_cost_usd=0.001,
        ),
        compressed_llm=LLMCallResult(
            content="Informe comprimido",
            processing_time_ms=1800.0,
            input_tokens=20,
            output_tokens=95,
            total_tokens=115,
            estimated_cost_usd=0.0008,
        ),
        response_similarity=0.9,
        response_evaluation="excellent",
    )
    report = E2EReport(
        generated_at="2026-07-03T00:00:00+00:00",
        compressor_model="granite4.1:3b",
        target_model="mistral-medium-3.5",
        reasoning_effort="high",
        entries=[entry],
    )

    summary = report.summary
    assert summary["total_prompts"] == 1
    assert summary["avg_response_similarity"] == 0.9
    assert summary["input_tokens_saved"] == 20
    assert summary["total_cost_original_usd"] == 0.001

    markdown = render_markdown_report(report)
    assert "Benchmark E2E PCM → Mistral" in markdown
    assert "mistral-medium-3.5" in markdown
