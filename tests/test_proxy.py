"""Tests del proxy HTTP (sin llamadas reales a upstream)."""

from unittest.mock import MagicMock

import pytest

from pcm.models import CompressionResult
from pcm.prompt_utils import join_instruction_and_payload, split_instruction_and_payload
from pcm.proxy import ChatProxy, ProxyConfig
from pcm.upstream import UpstreamTarget


@pytest.fixture
def mistral_upstream():
    return UpstreamTarget(
        provider="mistral",
        base_url="https://api.mistral.ai/v1",
        api_key="mistral-key",
        model="mistral-medium-3.5",
        supports_reasoning_effort=True,
        reasoning_effort="none",
    )


@pytest.fixture
def openai_upstream():
    return UpstreamTarget(
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_key="openai-key",
        model="gpt-4o-mini",
        supports_reasoning_effort=False,
        reasoning_effort=None,
    )


@pytest.fixture
def mock_compressor():
    compressor = MagicMock()
    compressor.config.model = "granite4.1:3b"
    compressor._count_tokens = MagicMock(
        side_effect=lambda text, model=None: (
            14
            if "TASK=" in text and "```" in text
            else 20
            if "```" in text
            else 6
            if "TASK=" in text
            else max(1, len(text.split()))
        )
    )
    compressor.compress.return_value = CompressionResult(
        original_prompt="Analiza este código Python.",
        compressed_prompt="TASK=review INPUT=python CHECK=race,leak,perf",
        original_tokens=12,
        compressed_tokens=6,
        compression_ratio=0.5,
        processing_time_ms=120.0,
    )
    return compressor


def test_split_instruction_and_payload_with_code_fence():
    text = "Analiza este código.\n\n```python\nx = 1\n```"
    instruction, payload = split_instruction_and_payload(text)
    assert instruction == "Analiza este código."
    assert payload.startswith("```python")


def test_join_instruction_and_payload():
    result = join_instruction_and_payload("TASK=review", "```python\nx=1\n```")
    assert "TASK=review" in result
    assert "```python" in result


@pytest.mark.asyncio
async def test_transform_request_compresses_user_and_keeps_code(
    mock_compressor, mistral_upstream
):
    proxy = ChatProxy(
        mock_compressor,
        ProxyConfig(
            inject_pcm_system=True,
            compress_roles=frozenset({"user"}),
            min_instruction_tokens=3,
        ),
    )
    body = {
        "messages": [
            {
                "role": "user",
                "content": "Analiza este código Python.\n\n```python\nx = 1\n```",
            }
        ]
    }

    transformed, stats = await proxy.transform_request(
        body, upstream=mistral_upstream
    )

    user_msg = next(m for m in transformed["messages"] if m["role"] == "user")
    assert user_msg["content"].startswith("TASK=review INPUT=python")
    assert "```python" in user_msg["content"]
    assert "x = 1" in user_msg["content"]
    assert stats.messages_compressed == 1
    assert stats.tokens_saved == 6
    assert stats.upstream_provider == "mistral"
    mock_compressor.compress.assert_called_once_with("Analiza este código Python.")


@pytest.mark.asyncio
async def test_transform_request_injects_pcm_system_prompt(
    mock_compressor, mistral_upstream
):
    proxy = ChatProxy(mock_compressor, ProxyConfig())
    body = {"messages": [{"role": "user", "content": "Hola"}]}

    transformed, _ = await proxy.transform_request(body, upstream=mistral_upstream)

    system_msgs = [m for m in transformed["messages"] if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert "PCM" in system_msgs[0]["content"]


@pytest.mark.asyncio
async def test_transform_request_skips_compression_when_disabled(
    mock_compressor, mistral_upstream
):
    proxy = ChatProxy(mock_compressor, ProxyConfig())
    body = {"messages": [{"role": "user", "content": "No comprimir esto"}]}

    transformed, stats = await proxy.transform_request(
        body, upstream=mistral_upstream, compress=False
    )

    user_msg = next(m for m in transformed["messages"] if m["role"] == "user")
    assert user_msg["content"] == "No comprimir esto"
    assert stats.messages_compressed == 0
    mock_compressor.compress.assert_not_called()


@pytest.mark.asyncio
async def test_transform_request_sets_mistral_reasoning(
    mock_compressor, mistral_upstream
):
    proxy = ChatProxy(
        mock_compressor,
        ProxyConfig(reasoning_effort="none"),
    )
    body = {"messages": [{"role": "user", "content": "Hola"}]}

    transformed, _ = await proxy.transform_request(body, upstream=mistral_upstream)

    assert transformed["model"] == "mistral-medium-3.5"
    assert transformed["reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_transform_request_strips_reasoning_for_openai(
    mock_compressor, openai_upstream
):
    proxy = ChatProxy(mock_compressor, ProxyConfig(reasoning_effort="none"))
    body = {
        "messages": [{"role": "user", "content": "Hola"}],
        "reasoning_effort": "high",
    }

    transformed, _ = await proxy.transform_request(body, upstream=openai_upstream)

    assert transformed["model"] == "gpt-4o-mini"
    assert "reasoning_effort" not in transformed


@pytest.mark.asyncio
async def test_transform_skips_short_instruction(mock_compressor, mistral_upstream):
    proxy = ChatProxy(
        mock_compressor,
        ProxyConfig(min_instruction_tokens=12),
    )
    original = "Explica qué es un middleware en una frase."
    body = {"messages": [{"role": "user", "content": original}]}

    transformed, stats = await proxy.transform_request(body, upstream=mistral_upstream)

    user_msg = next(m for m in transformed["messages"] if m["role"] == "user")
    assert user_msg["content"] == original
    assert stats.messages_compressed == 0
    mock_compressor.compress.assert_not_called()


@pytest.mark.asyncio
async def test_transform_skips_when_pcm_is_longer(mock_compressor, mistral_upstream):
    mock_compressor.compress.return_value = CompressionResult(
        original_prompt="Explica qué es un middleware en una frase larga para revisar.",
        compressed_prompt="TASK=explain TOPIC=middleware FORMAT=one_sentence STYLE=verbose",
        original_tokens=12,
        compressed_tokens=20,
        compression_ratio=-0.66,
        processing_time_ms=50.0,
        metadata={"skipped": True, "skip_reason": "no_token_savings"},
    )
    proxy = ChatProxy(
        mock_compressor,
        ProxyConfig(min_instruction_tokens=3),
    )
    original = "Explica qué es un middleware en una frase larga para revisar."
    body = {"messages": [{"role": "user", "content": original}]}

    transformed, stats = await proxy.transform_request(body, upstream=mistral_upstream)

    user_msg = next(m for m in transformed["messages"] if m["role"] == "user")
    assert user_msg["content"] == original
    assert stats.messages_compressed == 0
