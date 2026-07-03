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
from .e2e_benchmark import MISTRAL_PCM_SYSTEM_PROMPT
from .prompt_utils import join_instruction_and_payload, split_instruction_and_payload

logger = logging.getLogger(__name__)


@dataclass
class ProxyConfig:
    upstream_base_url: str = "https://api.mistral.ai/v1"
    upstream_api_key: str = ""
    default_model: str = "mistral-medium-3.5"
    reasoning_effort: str = "none"
    compress_roles: frozenset[str] = frozenset({"user"})
    inject_pcm_system: bool = True
    pcm_system_prompt: str = MISTRAL_PCM_SYSTEM_PROMPT
    timeout: float = 120.0
    min_instruction_tokens: int = 12

    @classmethod
    def from_env(cls) -> "ProxyConfig":
        roles = os.getenv("PCM_COMPRESS_ROLES", "user")
        policy = CompressionPolicy.from_env()
        return cls(
            upstream_base_url=os.getenv(
                "PCM_UPSTREAM_URL", "https://api.mistral.ai/v1"
            ).rstrip("/"),
            upstream_api_key=os.getenv("MISTRAL_API_KEY", ""),
            default_model=os.getenv("PCM_UPSTREAM_MODEL", "mistral-medium-3.5"),
            reasoning_effort=os.getenv("PCM_REASONING_EFFORT", "none"),
            compress_roles=frozenset(r.strip() for r in roles.split(",") if r.strip()),
            inject_pcm_system=os.getenv("PCM_INJECT_SYSTEM", "true").lower()
            in ("1", "true", "yes"),
            timeout=float(os.getenv("PCM_PROXY_TIMEOUT", "120")),
            min_instruction_tokens=policy.min_instruction_tokens,
        )


@dataclass
class ProxyCompressionStats:
    messages_compressed: int = 0
    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_time_ms: float = 0.0
    per_message: list[dict[str, Any]] = field(default_factory=list)

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
        if not self.config.upstream_api_key:
            logger.warning(
                "MISTRAL_API_KEY no configurada; el proxy no podrá reenviar peticiones."
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

    def _inject_pcm_system(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.config.inject_pcm_system:
            return messages

        pcm_hint = self.config.pcm_system_prompt.strip()
        out = [dict(m) for m in messages]
        system_idx = next(
            (i for i, m in enumerate(out) if m.get("role") == "system"),
            None,
        )

        if system_idx is None:
            out.insert(0, {"role": "system", "content": pcm_hint})
            return out

        existing = self._normalize_content(out[system_idx].get("content"))
        if pcm_hint in existing:
            return out

        out[system_idx] = {
            **out[system_idx],
            "content": f"{existing}\n\n{pcm_hint}".strip(),
        }
        return out

    async def transform_request(
        self,
        body: dict[str, Any],
        *,
        compress: bool = True,
    ) -> tuple[dict[str, Any], ProxyCompressionStats]:
        """Comprime roles configurados y prepara el body para upstream."""
        stats = ProxyCompressionStats()
        transformed = dict(body)
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

        messages = self._inject_pcm_system(messages)
        transformed["messages"] = messages

        if "model" not in transformed or not transformed["model"]:
            transformed["model"] = self.config.default_model

        if (
            "reasoning_effort" not in transformed
            and self.config.reasoning_effort
        ):
            transformed["reasoning_effort"] = self.config.reasoning_effort

        return transformed, stats

    async def forward_chat_completion(
        self,
        body: dict[str, Any],
        *,
        compress: bool = True,
    ) -> tuple[dict[str, Any], ProxyCompressionStats]:
        """Comprime, reenvía a upstream y devuelve la respuesta."""
        if not self.config.upstream_api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY no configurada. Añádela a .env para usar el proxy."
            )

        transformed, stats = await self.transform_request(body, compress=compress)
        url = f"{self.config.upstream_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.upstream_api_key}",
            "Content-Type": "application/json",
        }

        start = time.time()
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(url, headers=headers, json=transformed)
            response.raise_for_status()
            payload = response.json()

        stats.compression_time_ms = round(stats.compression_time_ms, 2)
        logger.info(
            "Proxy OK model=%s compressed=%s ratio=%.1f%% saved=%s tokens upstream=%.0fms",
            transformed.get("model"),
            stats.messages_compressed,
            stats.compression_ratio * 100,
            stats.tokens_saved,
            (time.time() - start) * 1000,
        )
        return payload, stats

    def stats_as_headers(self, stats: ProxyCompressionStats) -> dict[str, str]:
        return {
            "X-PCM-Messages-Compressed": str(stats.messages_compressed),
            "X-PCM-Compression-Ratio": f"{stats.compression_ratio:.4f}",
            "X-PCM-Tokens-Saved": str(stats.tokens_saved),
            "X-PCM-Compression-Time-Ms": f"{stats.compression_time_ms:.2f}",
        }
