"""Tests de resolución multi-upstream."""

import os
from unittest.mock import patch

import pytest

from pcm.upstream import (
    UpstreamTarget,
    infer_provider_from_model,
    list_configured_providers,
    resolve_upstream,
)


def test_infer_provider_from_model():
    assert infer_provider_from_model("gpt-4o-mini") == "openai"
    assert infer_provider_from_model("o1-preview") == "openai"
    assert infer_provider_from_model("mistral-medium-3.5") == "mistral"
    assert infer_provider_from_model("openai/gpt-4o-mini") == "openrouter"
    assert infer_provider_from_model("llama-3") is None


@patch.dict(
    os.environ,
    {
        "MISTRAL_API_KEY": "mistral-key",
        "OPENAI_API_KEY": "openai-key",
    },
    clear=False,
)
def test_list_configured_providers():
    providers = list_configured_providers()
    assert "mistral" in providers
    assert "openai" in providers


@patch.dict(
    os.environ,
    {
        "MISTRAL_API_KEY": "mistral-key",
        "PCM_UPSTREAM_PROVIDER": "mistral",
        "PCM_REASONING_EFFORT": "none",
    },
    clear=False,
)
def test_resolve_mistral_default():
    target = resolve_upstream()
    assert target.provider == "mistral"
    assert target.base_url == "https://api.mistral.ai/v1"
    assert target.api_key == "mistral-key"
    assert target.supports_reasoning_effort is True
    assert target.reasoning_effort == "none"


@patch.dict(
    os.environ,
    {
        "OPENAI_API_KEY": "openai-key",
        "PCM_UPSTREAM_PROVIDER": "openai",
    },
    clear=False,
)
def test_resolve_openai_by_hint():
    target = resolve_upstream(provider_hint="openai", model="gpt-4o-mini")
    assert target.provider == "openai"
    assert target.base_url == "https://api.openai.com/v1"
    assert target.model == "gpt-4o-mini"
    assert target.supports_reasoning_effort is False
    assert target.reasoning_effort is None


@patch.dict(
    os.environ,
    {
        "MISTRAL_API_KEY": "mistral-key",
        "OPENAI_API_KEY": "openai-key",
        "PCM_UPSTREAM_PROVIDER": "mistral",
    },
    clear=False,
)
def test_resolve_openai_from_model_auto_route():
    target = resolve_upstream(model="gpt-4o")
    assert target.provider == "openai"
    assert target.api_key == "openai-key"


@patch.dict(
    os.environ,
    {
        "PCM_UPSTREAM_URL": "https://llm.example.com/v1",
        "PCM_UPSTREAM_API_KEY": "custom-key",
        "PCM_UPSTREAM_MODEL": "my-model",
    },
    clear=False,
)
def test_resolve_custom_provider():
    target = resolve_upstream(provider_hint="custom")
    assert target == UpstreamTarget(
        provider="custom",
        base_url="https://llm.example.com/v1",
        api_key="custom-key",
        model="my-model",
        supports_reasoning_effort=False,
        reasoning_effort=None,
    )


@patch.dict(os.environ, {}, clear=True)
def test_resolve_missing_api_key_raises():
    with pytest.raises(RuntimeError, match="API key no configurada"):
        resolve_upstream(provider_hint="openai")
