"""Tests del proxy HTTP (sin llamadas reales a upstream)."""

from unittest.mock import MagicMock

import pytest

from pcm.models import CompressionResult
from pcm.prompt_utils import join_instruction_and_payload, split_instruction_and_payload
from pcm.proxy import ChatProxy, ProxyConfig


@pytest.fixture
def mock_compressor():
    compressor = MagicMock()
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
async def test_transform_request_compresses_user_and_keeps_code(mock_compressor):
    proxy = ChatProxy(
        mock_compressor,
        ProxyConfig(
            upstream_api_key="test-key",
            inject_pcm_system=True,
            compress_roles=frozenset({"user"}),
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

    transformed, stats = await proxy.transform_request(body)

    user_msg = next(m for m in transformed["messages"] if m["role"] == "user")
    assert user_msg["content"].startswith("TASK=review INPUT=python")
    assert "```python" in user_msg["content"]
    assert "x = 1" in user_msg["content"]
    assert stats.messages_compressed == 1
    assert stats.tokens_saved == 6
    mock_compressor.compress.assert_called_once_with("Analiza este código Python.")


@pytest.mark.asyncio
async def test_transform_request_injects_pcm_system_prompt(mock_compressor):
    proxy = ChatProxy(
        mock_compressor,
        ProxyConfig(upstream_api_key="test-key"),
    )
    body = {"messages": [{"role": "user", "content": "Hola"}]}

    transformed, _ = await proxy.transform_request(body)

    system_msgs = [m for m in transformed["messages"] if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert "PCM" in system_msgs[0]["content"]


@pytest.mark.asyncio
async def test_transform_request_skips_compression_when_disabled(mock_compressor):
    proxy = ChatProxy(
        mock_compressor,
        ProxyConfig(upstream_api_key="test-key"),
    )
    body = {"messages": [{"role": "user", "content": "No comprimir esto"}]}

    transformed, stats = await proxy.transform_request(body, compress=False)

    user_msg = next(m for m in transformed["messages"] if m["role"] == "user")
    assert user_msg["content"] == "No comprimir esto"
    assert stats.messages_compressed == 0
    mock_compressor.compress.assert_not_called()


@pytest.mark.asyncio
async def test_transform_request_sets_default_model_and_reasoning(mock_compressor):
    proxy = ChatProxy(
        mock_compressor,
        ProxyConfig(
            upstream_api_key="test-key",
            default_model="mistral-medium-3.5",
            reasoning_effort="none",
        ),
    )
    body = {"messages": [{"role": "user", "content": "Hola"}]}

    transformed, _ = await proxy.transform_request(body)

    assert transformed["model"] == "mistral-medium-3.5"
    assert transformed["reasoning_effort"] == "none"
