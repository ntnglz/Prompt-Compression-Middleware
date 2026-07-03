"""
Benchmark de compresión PCM sobre el dataset de prompts de ejemplo.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .compressor import PromptCompressor


FIELD_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^=\s]+)")


def parse_pcm_fields(text: str) -> dict[str, str]:
    """Extrae pares CLAVE=valor del formato PCM compacto."""
    normalized = " ".join(text.strip().split())
    return {
        key.upper(): value.lower()
        for key, value in FIELD_PATTERN.findall(normalized)
    }


def format_similarity(expected: str, actual: str) -> dict[str, Any]:
    """
    Compara la salida comprimida con la esperada por campos PCM.

    Devuelve key recall, value match y score combinado (0-1).
    """
    expected_fields = parse_pcm_fields(expected)
    actual_fields = parse_pcm_fields(actual)

    if not expected_fields:
        return {
            "key_recall": 0.0,
            "value_match": 0.0,
            "score": 0.0,
            "missing_keys": [],
            "extra_keys": sorted(actual_fields.keys()),
        }

    matched_values = sum(
        1 for key, value in expected_fields.items() if actual_fields.get(key) == value
    )
    present_keys = sum(1 for key in expected_fields if key in actual_fields)
    key_recall = present_keys / len(expected_fields)
    value_match = matched_values / len(expected_fields)
    score = (key_recall + value_match) / 2

    return {
        "key_recall": round(key_recall, 4),
        "value_match": round(value_match, 4),
        "score": round(score, 4),
        "missing_keys": sorted(set(expected_fields) - set(actual_fields)),
        "extra_keys": sorted(set(actual_fields) - set(expected_fields)),
        "expected_fields": expected_fields,
        "actual_fields": actual_fields,
    }


@dataclass
class BenchmarkEntry:
    id: str
    category: str
    language: str
    original_prompt: str
    expected_compression: str
    compressed_prompt: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    processing_time_ms: float
    format_similarity: dict[str, Any]
    semantic_similarity: Optional[float] = None
    semantic_evaluation: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "language": self.language,
            "original_prompt": self.original_prompt,
            "expected_compression": self.expected_compression,
            "compressed_prompt": self.compressed_prompt,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": round(self.compression_ratio, 4),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "format_similarity": self.format_similarity,
            "semantic_similarity": self.semantic_similarity,
            "semantic_evaluation": self.semantic_evaluation,
        }


@dataclass
class BenchmarkReport:
    generated_at: str
    model: str
    strategy: str
    include_semantic: bool
    think: Optional[bool] = None
    entries: list[BenchmarkEntry] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, Any]:
        if not self.entries:
            return {}

        ratios = [e.compression_ratio for e in self.entries]
        format_scores = [e.format_similarity["score"] for e in self.entries]
        semantic_scores = [
            e.semantic_similarity for e in self.entries if e.semantic_similarity is not None
        ]

        return {
            "total_prompts": len(self.entries),
            "avg_compression_ratio": round(sum(ratios) / len(ratios), 4),
            "min_compression_ratio": round(min(ratios), 4),
            "max_compression_ratio": round(max(ratios), 4),
            "total_tokens_saved": sum(e.original_tokens - e.compressed_tokens for e in self.entries),
            "avg_format_similarity": round(sum(format_scores) / len(format_scores), 4),
            "avg_semantic_similarity": (
                round(sum(semantic_scores) / len(semantic_scores), 4) if semantic_scores else None
            ),
            "avg_processing_time_ms": round(
                sum(e.processing_time_ms for e in self.entries) / len(self.entries), 2
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "model": self.model,
            "strategy": self.strategy,
            "include_semantic": self.include_semantic,
            "think": self.think,
            "summary": self.summary,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def load_example_prompts(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run_benchmark(
    compressor: PromptCompressor,
    prompts_path: Path,
    *,
    include_semantic: bool = False,
    limit: Optional[int] = None,
) -> BenchmarkReport:
    """Ejecuta el benchmark sobre el dataset de prompts."""
    prompts = load_example_prompts(prompts_path)
    if limit is not None:
        prompts = prompts[:limit]

    report = BenchmarkReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        model=compressor.config.model,
        strategy=compressor.config.strategy,
        include_semantic=include_semantic,
        think=compressor._resolve_think(),
    )

    for item in prompts:
        result = compressor.compress(item["text"])
        similarity = format_similarity(item["expected_compression"], result.compressed_prompt)

        semantic_similarity = None
        semantic_evaluation = None
        if include_semantic:
            comparison = compressor.compare_prompts(
                item["text"],
                result.compressed_prompt,
            )
            semantic_similarity = comparison.semantic_similarity
            semantic_evaluation = comparison.evaluation

        report.entries.append(
            BenchmarkEntry(
                id=item["id"],
                category=item["category"],
                language=item["language"],
                original_prompt=item["text"],
                expected_compression=item["expected_compression"],
                compressed_prompt=result.compressed_prompt,
                original_tokens=result.original_tokens,
                compressed_tokens=result.compressed_tokens,
                compression_ratio=result.compression_ratio,
                processing_time_ms=result.processing_time_ms,
                format_similarity=similarity,
                semantic_similarity=semantic_similarity,
                semantic_evaluation=semantic_evaluation,
            )
        )

    return report


def render_markdown_report(report: BenchmarkReport) -> str:
    """Genera un informe legible en Markdown."""
    summary = report.summary
    lines = [
        "# Benchmark PCM — example_prompts.json",
        "",
        f"- **Generado:** {report.generated_at}",
        f"- **Modelo:** `{report.model}`",
        f"- **Estrategia:** `{report.strategy}`",
        f"- **Thinking:** `{report.think}`" if report.think is not None else "- **Thinking:** `auto/off`",
        f"- **Similitud semántica (LLM):** {'sí' if report.include_semantic else 'no'}",
        "",
        "## Resumen",
        "",
        "| Métrica | Valor |",
        "|---------|-------|",
        f"| Prompts | {summary.get('total_prompts', 0)} |",
        f"| Ratio medio | {summary.get('avg_compression_ratio', 0):.2%} |",
        f"| Ratio min / max | {summary.get('min_compression_ratio', 0):.2%} / {summary.get('max_compression_ratio', 0):.2%} |",
        f"| Tokens ahorrados | {summary.get('total_tokens_saved', 0)} |",
        f"| Similitud formato (media) | {summary.get('avg_format_similarity', 0):.2%} |",
    ]

    if summary.get("avg_semantic_similarity") is not None:
        lines.append(
            f"| Similitud semántica (media) | {summary['avg_semantic_similarity']:.2%} |"
        )

    lines.extend(
        [
            f"| Tiempo medio | {summary.get('avg_processing_time_ms', 0):.0f} ms |",
            "",
            "## Detalle por prompt",
            "",
            "| ID | Categoría | Ratio | Formato | Semántica | Original → Comprimido |",
            "|----|-----------|-------|---------|-----------|------------------------|",
        ]
    )

    for entry in report.entries:
        semantic = (
            f"{entry.semantic_similarity:.0%}"
            if entry.semantic_similarity is not None
            else "—"
        )
        original = entry.original_prompt[:40].replace("|", "\\|") + "..."
        compressed = entry.compressed_prompt.replace("|", "\\|")
        lines.append(
            f"| {entry.id} | {entry.category} | {entry.compression_ratio:.0%} | "
            f"{entry.format_similarity['score']:.0%} | {semantic} | "
            f"{original} → `{compressed}` |"
        )

    lines.extend(["", "## Esperado vs obtenido", ""])

    for entry in report.entries:
        lines.extend(
            [
                f"### {entry.id} ({entry.category})",
                "",
                f"**Esperado:** `{entry.expected_compression}`",
                "",
                f"**Obtenido:** `{entry.compressed_prompt}`",
                "",
            ]
        )
        if entry.format_similarity["missing_keys"]:
            lines.append(
                f"- Claves faltantes: `{', '.join(entry.format_similarity['missing_keys'])}`"
            )
        if entry.format_similarity["extra_keys"]:
            lines.append(
                f"- Claves extra: `{', '.join(entry.format_similarity['extra_keys'])}`"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def model_slug(model: str) -> str:
    """Convierte nombre de modelo Ollama en slug seguro para archivos."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()


def render_leaderboard(runs: list[dict[str, Any]]) -> str:
    """Genera tabla comparativa de ejecuciones (mejor semántica, luego velocidad)."""
    if not runs:
        return "# Leaderboard PCM\n\nSin ejecuciones registradas.\n"

    def sort_key(run: dict[str, Any]) -> tuple:
        summary = run.get("summary", {})
        semantic = summary.get("avg_semantic_similarity")
        semantic_score = semantic if semantic is not None else -1
        time_ms = summary.get("avg_processing_time_ms", float("inf"))
        return (-semantic_score, time_ms)

    ranked = sorted(runs, key=sort_key)

    lines = [
        "# Leaderboard PCM",
        "",
        "Comparativa de ejecuciones del benchmark. Se prioriza similitud semántica y luego velocidad.",
        "",
        "| # | Modelo | Estrategia | Semántica | Formato | Ratio | Tiempo | Fecha | Archivo |",
        "|---|--------|------------|-----------|---------|-------|--------|-------|---------|",
    ]

    for index, run in enumerate(ranked, start=1):
        summary = run.get("summary", {})
        semantic = summary.get("avg_semantic_similarity")
        semantic_text = f"{semantic:.0%}" if semantic is not None else "—"
        generated = run.get("generated_at", "")[:19].replace("T", " ")
        file_name = Path(run.get("json_path", "")).name

        lines.append(
            f"| {index} | `{run.get('model', '?')}` | {run.get('strategy', '?')} | "
            f"{semantic_text} | {summary.get('avg_format_similarity', 0):.0%} | "
            f"{summary.get('avg_compression_ratio', 0):.0%} | "
            f"{summary.get('avg_processing_time_ms', 0):.0f} ms | {generated} | `{file_name}` |"
        )

    lines.append("")
    return "\n".join(lines)


def load_index(index_path: Path) -> list[dict[str, Any]]:
    if not index_path.exists():
        return []
    with index_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("runs", [])


def save_index(index_path: Path, runs: list[dict[str, Any]]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_runs": len(runs),
        "runs": runs,
    }
    index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_to_index(
    report: BenchmarkReport,
    json_path: Path,
    md_path: Path,
    index_path: Path,
) -> None:
    """Añade una ejecución al índice y regenera el leaderboard."""
    runs = load_index(index_path)

    entry = {
        "generated_at": report.generated_at,
        "model": report.model,
        "strategy": report.strategy,
        "include_semantic": report.include_semantic,
        "summary": report.summary,
        "json_path": str(json_path),
        "md_path": str(md_path),
    }

    runs = [run for run in runs if run.get("json_path") != str(json_path)]
    runs.append(entry)
    runs.sort(key=lambda run: run.get("generated_at", ""), reverse=True)

    save_index(index_path, runs)

    leaderboard_path = index_path.parent / "leaderboard.md"
    leaderboard_path.write_text(render_leaderboard(runs), encoding="utf-8")


def rebuild_index(output_dir: Path) -> int:
    """Reconstruye el índice escaneando JSON de ejecuciones guardadas."""
    runs: list[dict[str, Any]] = []
    candidates: list[Path] = []

    runs_dir = output_dir / "runs"
    if runs_dir.exists():
        candidates.extend(runs_dir.glob("*/*.json"))
        candidates.extend(runs_dir.glob("*_latest.json"))

    candidates.extend(output_dir.glob("benchmark_*.json"))

    seen: set[str] = set()
    for json_path in sorted(candidates):
        if json_path.name == "index.json" or json_path in seen:
            continue
        seen.add(str(json_path))

        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        if "summary" not in payload or "model" not in payload:
            continue

        md_path = json_path.with_suffix(".md")
        runs.append(
            {
                "generated_at": payload.get("generated_at", ""),
                "model": payload.get("model", ""),
                "strategy": payload.get("strategy", ""),
                "include_semantic": payload.get("include_semantic", False),
                "summary": payload.get("summary", {}),
                "json_path": str(json_path),
                "md_path": str(md_path) if md_path.exists() else "",
            }
        )

    deduped: dict[str, dict[str, Any]] = {}
    for run in runs:
        key = (
            f"{run.get('model')}|{run.get('generated_at')}|"
            f"{run.get('strategy')}|{run.get('include_semantic')}"
        )
        current = deduped.get(key)
        if current is None or "/runs/" in run.get("json_path", ""):
            deduped[key] = run

    runs = list(deduped.values())
    index_path = output_dir / "index.json"
    save_index(index_path, runs)

    leaderboard_path = output_dir / "leaderboard.md"
    leaderboard_path.write_text(render_leaderboard(runs), encoding="utf-8")
    return len(runs)


def save_report(
    report: BenchmarkReport,
    output_dir: Path,
    *,
    prefix: str = "benchmark",
) -> tuple[Path, Path]:
    """Guarda JSON/Markdown, copia latest por modelo y actualiza índice."""
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = model_slug(report.model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = output_dir / "runs" / slug
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{slug}_{timestamp}"
    json_path = runs_dir / f"{base_name}.json"
    md_path = runs_dir / f"{base_name}.md"

    json_path.write_text(report.to_json(), encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")

    latest_json = runs_dir / f"{slug}_latest.json"
    latest_md = runs_dir / f"{slug}_latest.md"
    latest_json.write_text(report.to_json(), encoding="utf-8")
    latest_md.write_text(render_markdown_report(report), encoding="utf-8")

    append_to_index(report, json_path, md_path, output_dir / "index.json")
    return json_path, md_path
