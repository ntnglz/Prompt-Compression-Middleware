"""
Benchmark de ciclo completo: prompt natural → PCM → Mistral → comparación de respuestas.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .benchmark import load_example_prompts
from .compressor import CompressorConfig, PromptCompressor
from .message_assembly import build_system_prompt

MISTRAL_PCM_SYSTEM_PROMPT = """Eres un asistente preciso y conciso.

Si el mensaje del usuario está en formato PCM (pares CLAVE=valor en una línea, por ejemplo
TASK=review INPUT=python CHECK=race,leak,perf FORMAT=markdown), interprétalo como
instrucciones equivalentes en lenguaje natural y responde según lo pedido.

Si tras la línea PCM hay un bloque de código o texto adicional, trátalo como el contenido
a procesar (código a revisar, documento a traducir, informe a resumir, etc.)."""


MISTRAL_INPUT_PRICE_PER_M = 1.5
MISTRAL_OUTPUT_PRICE_PER_M = 7.5

logger = logging.getLogger(__name__)


def _extract_message_content(content: Any) -> str:
    """Normaliza content del SDK Mistral: str plano o lista de chunks (reasoning)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            chunk_type = getattr(chunk, "type", None)
            if chunk_type is None and isinstance(chunk, dict):
                chunk_type = chunk.get("type")
            if chunk_type == "thinking":
                continue
            text = getattr(chunk, "text", None)
            if text is None and isinstance(chunk, dict):
                text = chunk.get("text")
            if text:
                parts.append(str(text))
        return "".join(parts).strip()
    return str(content).strip()


def _format_payload(item: dict[str, Any]) -> str:
    """Formatea el payload adjunto (código o texto) para el prompt al LLM."""
    payload = item.get("payload") or item.get("code")
    if not payload:
        return ""

    payload_type = item.get("payload_type")
    if payload_type is None:
        payload_type = "code" if item.get("payload_lang") or item.get("code_lang") else "text"

    if payload_type == "code":
        lang = item.get("payload_lang") or item.get("code_lang") or ""
        return f"\n\n```{lang}\n{str(payload).strip()}\n```"
    return f"\n\n{str(payload).strip()}"


def resolve_e2e_prompt_parts(item: dict[str, Any]) -> tuple[str, str]:
    """
    Devuelve (texto_a_comprimir, prompt_completo_para_llm).

    Si el item tiene `instruction` + `payload`, solo comprime la instrucción
    y adjunta el payload intacto al prompt completo.
    """
    if "instruction" in item:
        instruction = str(item["instruction"]).strip()
        payload_suffix = _format_payload(item)
        return instruction, instruction + payload_suffix

    text = str(item["text"]).strip()
    return text, text


def build_compressed_user_prompt(compressed_instruction: str, item: dict[str, Any]) -> str:
    """Combina la línea PCM con el payload sin comprimir."""
    payload_suffix = _format_payload(item)
    if payload_suffix:
        return compressed_instruction.strip() + payload_suffix
    return compressed_instruction.strip()


@dataclass
class LLMCallResult:
    content: str
    processing_time_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "truncated": self.truncated,
        }


@dataclass
class E2EEntry:
    id: str
    category: str
    language: str
    original_prompt: str
    compressed_prompt: str
    compression_time_ms: float
    compression_ratio: float
    original_llm: LLMCallResult
    compressed_llm: LLMCallResult
    response_similarity: float
    response_evaluation: str
    concise_llm: LLMCallResult | None = None
    output_style_normal: str = "normal"
    compressed_user_prompt: str = ""
    payload_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "category": self.category,
            "language": self.language,
            "original_prompt": self.original_prompt,
            "compressed_prompt": self.compressed_prompt,
            "compression_time_ms": round(self.compression_time_ms, 2),
            "compression_ratio": round(self.compression_ratio, 4),
            "original_llm": self.original_llm.to_dict(),
            "compressed_llm": self.compressed_llm.to_dict(),
            "response_similarity": round(self.response_similarity, 4),
            "response_evaluation": self.response_evaluation,
        }
        if self.compressed_user_prompt:
            data["compressed_user_prompt"] = self.compressed_user_prompt
        if self.payload_chars:
            data["payload_chars"] = self.payload_chars
        if self.concise_llm is not None:
            data["concise_llm"] = self.concise_llm.to_dict()
        return data


@dataclass
class E2EReport:
    generated_at: str
    compressor_model: str
    target_model: str
    reasoning_effort: str
    entries: list[E2EEntry] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, Any]:
        if not self.entries:
            return {}

        concise_entries = [e for e in self.entries if e.concise_llm is not None]
        result: dict[str, Any] = {
            "total_prompts": len(self.entries),
            "avg_compression_ratio": round(
                sum(e.compression_ratio for e in self.entries) / len(self.entries), 4
            ),
            "avg_compression_time_ms": round(
                sum(e.compression_time_ms for e in self.entries) / len(self.entries), 2
            ),
            "avg_original_llm_time_ms": round(
                sum(e.original_llm.processing_time_ms for e in self.entries)
                / len(self.entries),
                2,
            ),
            "avg_compressed_llm_time_ms": round(
                sum(e.compressed_llm.processing_time_ms for e in self.entries)
                / len(self.entries),
                2,
            ),
            "avg_response_similarity": round(
                sum(e.response_similarity for e in self.entries) / len(self.entries), 4
            ),
            "total_original_input_tokens": sum(
                e.original_llm.input_tokens for e in self.entries
            ),
            "total_compressed_input_tokens": sum(
                e.compressed_llm.input_tokens for e in self.entries
            ),
            "input_tokens_saved": sum(
                e.original_llm.input_tokens - e.compressed_llm.input_tokens
                for e in self.entries
            ),
            "total_cost_original_usd": round(
                sum(e.original_llm.estimated_cost_usd for e in self.entries), 4
            ),
            "total_cost_compressed_usd": round(
                sum(e.compressed_llm.estimated_cost_usd for e in self.entries), 4
            ),
            "total_pipeline_time_ms": round(
                sum(
                    e.compression_time_ms
                    + e.original_llm.processing_time_ms
                    + e.compressed_llm.processing_time_ms
                    for e in self.entries
                ),
                2,
            ),
        }

        if concise_entries:
            result["avg_concise_output_tokens"] = round(
                sum(e.concise_llm.output_tokens for e in concise_entries)  # type: ignore[union-attr]
                / len(concise_entries),
                2,
            )
            savings = [
                (
                    (e.compressed_llm.output_tokens - e.concise_llm.output_tokens)  # type: ignore[union-attr]
                    / e.compressed_llm.output_tokens
                    * 100
                )
                for e in concise_entries
                if e.compressed_llm.output_tokens > 0
            ]
            if savings:
                result["avg_output_token_savings_pct"] = round(
                    sum(savings) / len(savings), 2
                )
            result["avg_cost_delta_concise_vs_baseline"] = round(
                sum(
                    e.concise_llm.estimated_cost_usd - e.original_llm.estimated_cost_usd  # type: ignore[union-attr]
                    for e in concise_entries
                )
                / len(concise_entries),
                6,
            )

        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "compressor_model": self.compressor_model,
            "target_model": self.target_model,
            "reasoning_effort": self.reasoning_effort,
            "summary": self.summary,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class MistralClient:
    """Cliente fino sobre la API oficial de Mistral."""

    def __init__(
        self,
        *,
        model: str = "mistral-medium-3.5",
        reasoning_effort: str = "high",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY no configurada. Obtén la clave en "
                "https://console.mistral.ai/ y añádela a .env"
            )
        try:
            from mistralai.client import Mistral
        except ImportError as exc:
            raise RuntimeError(
                "Instala el SDK: pip install mistralai"
            ) from exc
        self._client = Mistral(api_key=self.api_key)

    def complete(self, user_prompt: str, *, system_prompt: str) -> LLMCallResult:
        start = time.time()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        response = self._client.chat.complete(**kwargs)
        elapsed_ms = (time.time() - start) * 1000

        choice = response.choices[0]
        finish_reason = str(getattr(choice, "finish_reason", "") or "")
        content = _extract_message_content(choice.message.content)
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0) or (
            input_tokens + output_tokens
        )
        truncated = finish_reason == "length" or (
            not content and output_tokens >= int(self.max_tokens * 0.9)
        )
        if truncated and not content:
            logger.warning(
                "Mistral truncó la respuesta sin texto visible "
                "(reasoning consumió max_tokens=%s). Usa --max-tokens mayor.",
                self.max_tokens,
            )
        cost = (
            input_tokens * MISTRAL_INPUT_PRICE_PER_M / 1_000_000
            + output_tokens * MISTRAL_OUTPUT_PRICE_PER_M / 1_000_000
        )

        return LLMCallResult(
            content=content,
            processing_time_ms=elapsed_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            truncated=truncated,
        )


def _evaluate_response_similarity(
    compressor: PromptCompressor,
    response_a: str,
    response_b: str,
) -> tuple[float, str]:
    try:
        eval_response = compressor._ollama_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un evaluador de equivalencia de respuestas. "
                        "Responde solo con un número entre 0 y 1."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Dos modelos respondieron a la misma tarea. "
                        "¿Qué tan equivalentes son en contenido, calidad e información?\n"
                        f"Respuesta A:\n{response_a}\n\n"
                        f"Respuesta B:\n{response_b}\n\n"
                        "Responde solo con un número entre 0 y 1 "
                        "(1 = equivalentes, 0 = sin relación):"
                    ),
                },
            ],
            model=compressor.config.evaluator_model,
            temperature=0.0,
            think=False,
            num_predict=16,
        )
        score = compressor._parse_semantic_score(
            eval_response["message"]["content"].strip()
        )
    except Exception:
        return 0.0, "poor"

    if score >= 0.95:
        evaluation = "excellent"
    elif score >= 0.85:
        evaluation = "good"
    elif score >= 0.70:
        evaluation = "fair"
    else:
        evaluation = "poor"
    return score, evaluation


def run_e2e_benchmark(
    compressor: PromptCompressor,
    mistral: MistralClient,
    prompts_path: Path,
    *,
    limit: Optional[int] = None,
) -> E2EReport:
    prompts = load_example_prompts(prompts_path)
    if limit is not None:
        prompts = prompts[:limit]

    report = E2EReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        compressor_model=compressor.config.model,
        target_model=mistral.model,
        reasoning_effort=mistral.reasoning_effort,
    )

    for item in prompts:
        compress_input, original_full = resolve_e2e_prompt_parts(item)
        payload = item.get("payload") or item.get("code") or ""
        compression = compressor.compress(compress_input)
        compressed_user = build_compressed_user_prompt(
            compression.compressed_prompt,
            item,
        )
        original_llm = mistral.complete(
            original_full,
            system_prompt=MISTRAL_PCM_SYSTEM_PROMPT,
        )
        compressed_llm = mistral.complete(
            compressed_user,
            system_prompt=MISTRAL_PCM_SYSTEM_PROMPT,
        )
        concise_system = build_system_prompt(
            compressed_instruction=compression.compressed_prompt,
            response_lang=item.get("language", "en"),
            output_style="concise",
            pcm_interpretation_hint=MISTRAL_PCM_SYSTEM_PROMPT,
        )
        concise_llm = mistral.complete(
            compressed_user,
            system_prompt=concise_system,
        )
        similarity, evaluation = _evaluate_response_similarity(
            compressor,
            original_llm.content,
            compressed_llm.content,
        )

        report.entries.append(
            E2EEntry(
                id=item["id"],
                category=item["category"],
                language=item["language"],
                original_prompt=original_full,
                compressed_prompt=compression.compressed_prompt,
                compressed_user_prompt=compressed_user,
                payload_chars=len(str(payload)),
                compression_time_ms=compression.processing_time_ms,
                compression_ratio=compression.compression_ratio,
                original_llm=original_llm,
                compressed_llm=compressed_llm,
                concise_llm=concise_llm,
                response_similarity=similarity,
                response_evaluation=evaluation,
            )
        )

    return report


def render_markdown_report(report: E2EReport) -> str:
    summary = report.summary
    lines = [
        "# Benchmark E2E PCM → Mistral",
        "",
        f"- **Generado:** {report.generated_at}",
        f"- **Compresor:** `{report.compressor_model}`",
        f"- **Modelo destino:** `{report.target_model}`",
        f"- **Reasoning effort:** `{report.reasoning_effort}`",
        "",
        "## Resumen",
        "",
        "| Métrica | Valor |",
        "|---------|-------|",
        f"| Prompts | {summary.get('total_prompts', 0)} |",
        f"| Ratio compresión (media) | {summary.get('avg_compression_ratio', 0):.2%} |",
        f"| Tiempo compresión (media) | {summary.get('avg_compression_time_ms', 0):.0f} ms |",
        f"| Tiempo Mistral original (media) | {summary.get('avg_original_llm_time_ms', 0):.0f} ms |",
        f"| Tiempo Mistral comprimido (media) | {summary.get('avg_compressed_llm_time_ms', 0):.0f} ms |",
        f"| Similitud respuestas (media) | {summary.get('avg_response_similarity', 0):.2%} |",
        f"| Tokens input original (total) | {summary.get('total_original_input_tokens', 0)} |",
        f"| Tokens input comprimido (total) | {summary.get('total_compressed_input_tokens', 0)} |",
        f"| Tokens input ahorrados | {summary.get('input_tokens_saved', 0)} |",
        f"| Coste original (total) | ${summary.get('total_cost_original_usd', 0):.4f} |",
        f"| Coste comprimido (total) | ${summary.get('total_cost_compressed_usd', 0):.4f} |",
    ]
    if "avg_concise_output_tokens" in summary:
        lines.extend(
            [
                f"| Tokens output concise (media) | {summary['avg_concise_output_tokens']:.0f} |",
                f"| Ahorro tokens output concise vs PCM (media) | {summary.get('avg_output_token_savings_pct', 0):.1f}% |",
                f"| Δ coste concise vs baseline (media) | ${summary.get('avg_cost_delta_concise_vs_baseline', 0):.6f} |",
            ]
        )
    lines.extend(
        [
        "",
        "## Detalle por prompt",
        "",
        "| ID | Categoría | Payload | Compresión | Similitud | Truncado | t_comp | t_llm_orig | t_llm_pcm |",
        "|----|-----------|---------|------------|-----------|----------|--------|------------|-----------|",
        ]
    )

    for entry in report.entries:
        truncated = "sí" if (
            entry.original_llm.truncated or entry.compressed_llm.truncated
        ) else "no"
        payload = f"{entry.payload_chars} ch" if entry.payload_chars else "—"
        lines.append(
            f"| {entry.id} | {entry.category} | {payload} | {entry.compression_ratio:.0%} | "
            f"{entry.response_similarity:.0%} | {truncated} | "
            f"{entry.compression_time_ms:.0f} ms | "
            f"{entry.original_llm.processing_time_ms:.0f} ms | "
            f"{entry.compressed_llm.processing_time_ms:.0f} ms |"
        )

    return "\n".join(lines) + "\n"


def save_report(report: E2EReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = report.target_model.replace(":", "_").replace(".", "_").replace("-", "_")
    json_path = runs_dir / f"e2e_{slug}_{stamp}.json"
    md_path = runs_dir / f"e2e_{slug}_{stamp}.md"
    latest_json = runs_dir / f"e2e_{slug}_latest.json"
    latest_md = runs_dir / f"e2e_{slug}_latest.md"

    payload = report.to_json()
    for path in (json_path, latest_json):
        path.write_text(payload, encoding="utf-8")

    markdown = render_markdown_report(report)
    for path in (md_path, latest_md):
        path.write_text(markdown, encoding="utf-8")

    return json_path, md_path
