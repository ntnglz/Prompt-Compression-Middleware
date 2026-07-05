"""
Proxy HTTP compatible con OpenAI: comprime prompts y reenvía al LLM destino.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from .compressor import PromptCompressor
from .compression_policy import CompressionPolicy
from .e2e_benchmark import (
    MISTRAL_INPUT_PRICE_PER_M,
    MISTRAL_OUTPUT_PRICE_PER_M,
    MISTRAL_PCM_SYSTEM_PROMPT,
)
from .message_assembly import build_proxy_system_prompt
from .output_directives import has_response_block
from .prompt_utils import join_instruction_and_payload, split_instruction_and_payload
from .turn_cost import TurnCostMetrics, compute_turn_cost
from .upstream import UpstreamTarget, list_configured_providers, resolve_upstream

logger = logging.getLogger(__name__)


@dataclass
class ProxyConfig:
    default_provider: str = "mistral"
    default_model: str = ""
    reasoning_effort: str = "none"
    compress_roles: frozenset[str] = frozenset({"user"})
    inject_pcm_system: bool = True
    pcm_system_prompt: str = MISTRAL_PCM_SYSTEM_PROMPT
    output_style: str = "normal"
    response_lang: str = "en"
    timeout: float = 120.0
    min_instruction_tokens: int = 12

    @classmethod
    def from_env(cls) -> "ProxyConfig":
        roles = os.getenv("PCM_COMPRESS_ROLES", "user")
        policy = CompressionPolicy.from_env()
        return cls(
            default_provider=os.getenv("PCM_UPSTREAM_PROVIDER", "mistral"),
            default_model=os.getenv("PCM_UPSTREAM_MODEL", ""),
            reasoning_effort=os.getenv("PCM_REASONING_EFFORT", "none"),
            compress_roles=frozenset(r.strip() for r in roles.split(",") if r.strip()),
            inject_pcm_system=os.getenv("PCM_INJECT_SYSTEM", "true").lower()
            in ("1", "true", "yes"),
            output_style=os.getenv("PCM_OUTPUT_STYLE", "normal"),
            response_lang=os.getenv("PCM_RESPONSE_LANG", "en"),
            timeout=float(os.getenv("PCM_PROXY_TIMEOUT", "120")),
            min_instruction_tokens=policy.min_instruction_tokens,
        )


@dataclass
class ProxyCompressionStats:
    messages_compressed: int = 0
    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_time_ms: float = 0.0
    upstream_provider: str = ""
    upstream_model: str = ""
    per_message: list[dict[str, Any]] = field(default_factory=list)
    turn_cost: TurnCostMetrics | None = None

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens <= 0:
            return 0.0
        return 1 - (self.compressed_tokens / self.original_tokens)


class ChatProxy:
    """Comprime mensajes de chat y reenvía la petición al proveedor upstream."""

    def __init__(
        self,
        compressor: PromptCompressor,
        config: Optional[ProxyConfig] = None,
    ) -> None:
        self.compressor = compressor
        self.config = config or ProxyConfig.from_env()
        if not list_configured_providers():
            logger.warning(
                "Ningún proveedor upstream configurado. "
                "Define MISTRAL_API_KEY, OPENAI_API_KEY u OPENROUTER_API_KEY."
            )

    def _normalize_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
            return "".join(parts)
        return str(content)

    async def _compress_text(self, text: str) -> tuple[str, dict[str, Any]]:
        instruction, payload = split_instruction_and_payload(text)
        if not instruction:
            return text, {"skipped": True, "reason": "empty_instruction"}

        policy = CompressionPolicy(
            min_instruction_tokens=self.config.min_instruction_tokens
        )
        count_tokens = lambda value: self.compressor._count_tokens(
            value, self.compressor.config.model
        )

        should_compress, skip_reason = policy.should_compress_instruction(
            instruction, count_tokens
        )
        if not should_compress:
            return text, {"skipped": True, "reason": skip_reason}

        result = await asyncio.to_thread(self.compressor.compress, instruction)
        if result.metadata.get("skipped"):
            return text, {
                "skipped": True,
                "reason": result.metadata.get("skip_reason", "no_savings"),
            }

        compressed = join_instruction_and_payload(
            result.compressed_prompt,
            payload,
        )

        should_apply, revert_reason = policy.should_apply_compression(
            text,
            compressed,
            count_tokens,
        )
        if not should_apply:
            return text, {"skipped": True, "reason": revert_reason}

        instruction_tokens = count_tokens(instruction)
        compressed_instruction_tokens = count_tokens(result.compressed_prompt)
        full_original_tokens = count_tokens(text)
        full_compressed_tokens = count_tokens(compressed)
        return compressed, {
            "original_tokens": full_original_tokens,
            "compressed_tokens": full_compressed_tokens,
            "compression_ratio": (
                1 - (full_compressed_tokens / full_original_tokens)
                if full_original_tokens > 0
                else 0.0
            ),
            "processing_time_ms": result.processing_time_ms,
            "had_payload": bool(payload),
            "instruction_tokens": instruction_tokens,
            "compressed_instruction_tokens": compressed_instruction_tokens,
        }

    def _inject_system_blocks(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.config.inject_pcm_system:
            return messages

        out = [dict(m) for m in messages]
        system_idx = next(
            (i for i, m in enumerate(out) if m.get("role") == "system"),
            None,
        )

        base = build_proxy_system_prompt(
            response_lang=self.config.response_lang,
            output_style=self.config.output_style,  # type: ignore[arg-type]
            pcm_interpretation_hint=self.config.pcm_system_prompt,
        )

        if system_idx is None:
            out.insert(0, {"role": "system", "content": base})
            return out

        existing = self._normalize_content(out[system_idx].get("content"))
        if has_response_block(existing):
            hint = self.config.pcm_system_prompt.strip()
            if hint and hint not in existing:
                out[system_idx] = {
                    **out[system_idx],
                    "content": f"{existing}\n\n{hint}".strip(),
                }
            return out

        out[system_idx] = {
            **out[system_idx],
            "content": f"{existing}\n\n{base}".strip(),
        }
        return out

    def _prepare_upstream_body(
        self,
        body: dict[str, Any],
        upstream: UpstreamTarget,
    ) -> dict[str, Any]:
        prepared = dict(body)
        prepared["model"] = body.get("model") or upstream.model

        if upstream.supports_reasoning_effort and upstream.reasoning_effort:
            if "reasoning_effort" not in prepared:
                prepared["reasoning_effort"] = upstream.reasoning_effort
        else:
            prepared.pop("reasoning_effort", None)

        return prepared

    async def transform_request(
        self,
        body: dict[str, Any],
        *,
        upstream: UpstreamTarget,
        compress: bool = True,
    ) -> tuple[dict[str, Any], ProxyCompressionStats]:
        """Comprime roles configurados y prepara el body para upstream."""
        stats = ProxyCompressionStats(
            upstream_provider=upstream.provider,
            upstream_model=upstream.model,
        )
        transformed = self._prepare_upstream_body(body, upstream)
        messages = [dict(m) for m in body.get("messages", [])]

        if compress:
            for i, message in enumerate(messages):
                role = message.get("role", "")
                if role not in self.config.compress_roles:
                    continue

                content = self._normalize_content(message.get("content"))
                if not content.strip():
                    continue

                compressed_content, meta = await self._compress_text(content)
                if meta.get("skipped"):
                    continue

                messages[i] = {**message, "content": compressed_content}
                stats.messages_compressed += 1
                stats.original_tokens += int(meta["original_tokens"])
                stats.compressed_tokens += int(meta["compressed_tokens"])
                stats.compression_time_ms += float(meta["processing_time_ms"])
                stats.per_message.append(
                    {
                        "index": i,
                        "role": role,
                        **meta,
                    }
                )

        messages = self._inject_system_blocks(messages)
        transformed["messages"] = messages

        return transformed, stats

    async def forward_chat_completion(
        self,
        body: dict[str, Any],
        *,
        compress: bool = True,
        provider_hint: Optional[str] = None,
    ) -> tuple[dict[str, Any], ProxyCompressionStats]:
        """Comprime, reenvía a upstream y devuelve la respuesta."""
        upstream = resolve_upstream(
            provider_hint=provider_hint,
            model=body.get("model"),
            default_provider=self.config.default_provider,
            default_model=self.config.default_model or None,
            reasoning_effort=self.config.reasoning_effort,
        )

        transformed, stats = await self.transform_request(
            body,
            upstream=upstream,
            compress=compress,
        )
        url = f"{upstream.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {upstream.api_key}",
            "Content-Type": "application/json",
        }

        start = time.time()
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(url, headers=headers, json=transformed)
            response.raise_for_status()
            payload = response.json()

        usage = payload.get("usage") or {}
        output_tokens = int(usage.get("completion_tokens") or 0)
        assistant_content = ""
        choices = payload.get("choices") or []
        if choices:
            assistant_content = self._normalize_content(
                choices[0].get("message", {}).get("content")
            )

        stats.turn_cost = compute_turn_cost(
            messages=transformed.get("messages", []),
            output_text=assistant_content,
            output_tokens=output_tokens or None,
            input_price_per_m=MISTRAL_INPUT_PRICE_PER_M,
            output_price_per_m=MISTRAL_OUTPUT_PRICE_PER_M,
        )

        stats.compression_time_ms = round(stats.compression_time_ms, 2)
        logger.info(
            "Proxy OK provider=%s model=%s compressed=%s ratio=%.1f%% "
            "saved=%s tokens upstream=%.0fms",
            upstream.provider,
            transformed.get("model"),
            stats.messages_compressed,
            stats.compression_ratio * 100,
            stats.tokens_saved,
            (time.time() - start) * 1000,
        )
        return payload, stats

    def stats_as_headers(self, stats: ProxyCompressionStats) -> dict[str, str]:
        headers = {
            "X-PCM-Messages-Compressed": str(stats.messages_compressed),
            "X-PCM-Compression-Ratio": f"{stats.compression_ratio:.4f}",
            "X-PCM-Tokens-Saved": str(stats.tokens_saved),
            "X-PCM-Compression-Time-Ms": f"{stats.compression_time_ms:.2f}",
        }
        if stats.upstream_provider:
            headers["X-PCM-Upstream-Provider"] = stats.upstream_provider
        if stats.upstream_model:
            headers["X-PCM-Upstream-Model"] = stats.upstream_model
        if stats.turn_cost:
            headers["X-PCM-Input-Tokens"] = str(stats.turn_cost.input_tokens)
            headers["X-PCM-Output-Tokens"] = str(stats.turn_cost.output_tokens)
            headers["X-PCM-Cost-Total-USD"] = f"{stats.turn_cost.cost_total:.6f}"
        return headers
