"""Registro y resolución de proveedores upstream OpenAI-compatible."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class UpstreamProviderSpec:
    """Definición estática de un proveedor."""

    id: str
    base_url: str
    api_key_env: str
    default_model: str
    supports_reasoning_effort: bool = False


@dataclass(frozen=True)
class UpstreamTarget:
    """Destino resuelto para una petición."""

    provider: str
    base_url: str
    api_key: str
    model: str
    supports_reasoning_effort: bool = False
    reasoning_effort: Optional[str] = None


def _provider_specs() -> dict[str, UpstreamProviderSpec]:
    return {
        "mistral": UpstreamProviderSpec(
            id="mistral",
            base_url=os.getenv(
                "MISTRAL_BASE_URL", "https://api.mistral.ai/v1"
            ).rstrip("/"),
            api_key_env="MISTRAL_API_KEY",
            default_model="mistral-medium-3.5",
            supports_reasoning_effort=True,
        ),
        "openai": UpstreamProviderSpec(
            id="openai",
            base_url=os.getenv(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ).rstrip("/"),
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-4o-mini",
            supports_reasoning_effort=False,
        ),
        "openrouter": UpstreamProviderSpec(
            id="openrouter",
            base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ).rstrip("/"),
            api_key_env="OPENROUTER_API_KEY",
            default_model="openai/gpt-4o-mini",
            supports_reasoning_effort=False,
        ),
    }


def infer_provider_from_model(model: str) -> Optional[str]:
    """Infiere el proveedor a partir del nombre del modelo."""
    name = model.lower()
    if name.startswith(("gpt-", "o1", "o3", "chatgpt-")):
        return "openai"
    if name.startswith(("mistral", "ministral", "codestral", "pixtral")):
        return "mistral"
    if "/" in name:
        return "openrouter"
    return None


def list_configured_providers() -> list[str]:
    """Proveedores con API key configurada en el entorno."""
    configured = []
    for provider_id, spec in _provider_specs().items():
        if os.getenv(spec.api_key_env, "").strip():
            configured.append(provider_id)
    custom_key = os.getenv("PCM_UPSTREAM_API_KEY", "").strip()
    if custom_key and "custom" not in configured:
        configured.append("custom")
    return configured


def resolve_upstream(
    *,
    provider_hint: Optional[str] = None,
    model: Optional[str] = None,
    default_provider: Optional[str] = None,
    default_model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_api_key: Optional[str] = None,
) -> UpstreamTarget:
    """
    Resuelve el destino upstream.

    Prioridad del proveedor:
    1. provider_hint (header x-pcm-provider o PCM_UPSTREAM_PROVIDER)
    2. inferencia por nombre de modelo
    3. default_provider del entorno
    """
    specs = _provider_specs()
    default_provider = (
        provider_hint
        or infer_provider_from_model(model or "")
        or default_provider
        or os.getenv("PCM_UPSTREAM_PROVIDER", "mistral")
    ).strip().lower()

    if default_provider == "custom":
        base_url = (custom_base_url or os.getenv("PCM_UPSTREAM_URL", "")).rstrip("/")
        api_key = custom_api_key or os.getenv("PCM_UPSTREAM_API_KEY", "")
        if not base_url or not api_key:
            raise RuntimeError(
                "Proveedor custom requiere PCM_UPSTREAM_URL y PCM_UPSTREAM_API_KEY."
            )
        resolved_model = model or default_model or os.getenv(
            "PCM_UPSTREAM_MODEL", "mistral-medium-3.5"
        )
        return UpstreamTarget(
            provider="custom",
            base_url=base_url,
            api_key=api_key,
            model=resolved_model,
            supports_reasoning_effort=False,
            reasoning_effort=None,
        )

    spec = specs.get(default_provider)
    if spec is None:
        available = ", ".join(sorted(specs.keys()) + ["custom"])
        raise RuntimeError(
            f"Proveedor upstream desconocido: {default_provider}. "
            f"Disponibles: {available}"
        )

    api_key = os.getenv(spec.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"API key no configurada para proveedor '{spec.id}'. "
            f"Define {spec.api_key_env} en .env"
        )

    resolved_model = (
        model
        or default_model
        or os.getenv("PCM_UPSTREAM_MODEL")
        or spec.default_model
    )
    effort = reasoning_effort or os.getenv("PCM_REASONING_EFFORT", "none")
    return UpstreamTarget(
        provider=spec.id,
        base_url=spec.base_url,
        api_key=api_key,
        model=resolved_model,
        supports_reasoning_effort=spec.supports_reasoning_effort,
        reasoning_effort=effort if spec.supports_reasoning_effort else None,
    )
