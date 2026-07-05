#!/usr/bin/env python3
"""
Extrae y anonimiza mensajes de usuario desde exports Cursor (.md) del corpus COE.

Origen (local, no versionado):
  ../Context-Optimization-Engine/data/benchmarks/corpus/transcripts/

Salida:
  data/e2e_prompts_extensive.json  — dataset E2E PCM (instruction + payload)

Uso:
  python scripts/extract_e2e_from_transcripts.py --list
  python scripts/extract_e2e_from_transcripts.py --write-curated
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COE_TRANSCRIPTS = (
    ROOT.parent
    / "Context-Optimization-Engine"
    / "data"
    / "benchmarks"
    / "corpus"
    / "transcripts"
)
OUTPUT = ROOT / "data" / "e2e_prompts_extensive.json"

# Reglas alineadas con COE benchmark-harness.md §13
_ANONYMIZE_PAIRS: list[tuple[str, str]] = [
    (r"RegistroVisitas", "ExampleApp"),
    (r"AquiEstuve", "ExampleApp"),
    (r"Registro Visitas", "ExampleApp"),
    (r"\bVisitPhoto\b", "RecordPhoto"),
    (r"\bVisitDetailView\b", "RecordDetailView"),
    (r"\bVisitType\b", "RecordType"),
    (r"\bVisit\b", "RecordItem"),
    (r"\bTrip\b", "RecordType"),
    (r"iOSPhotoService", "PlatformPhotoService"),
    (r"macOSPhotoService", "DesktopPhotoService"),
    (r"iOSLocationService", "PlatformLocationService"),
    (r"HistoricalImportService", "HistoricalImportService"),  # generic enough
    (r"/Users/[^\s\)`\"']+", "/workspace/app"),
    (r"/Volumes/DevSSD/[^\s\)`\"']+", "/workspace/build"),
    (r"antonio[^\s@]*", "developer"),
    (r"Antonio J\. González", "Developer"),
    (r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "[UUID]"),
]


def anonymize(text: str) -> str:
    out = text
    for pattern, repl in _ANONYMIZE_PAIRS:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


def parse_user_messages(markdown: str) -> list[str]:
    """Extrae bloques **User** de un export Cursor."""
    messages: list[str] = []
    for part in re.split(r"\n---\n", markdown):
        if "**User**" not in part:
            continue
        match = re.search(r"\*\*User\*\*\s*\n\n(.*?)(?=\n---|\Z)", part, re.S)
        if match:
            msg = match.group(1).strip()
            if msg:
                messages.append(msg)
    return messages


def list_transcripts(transcripts_dir: Path) -> None:
    if not transcripts_dir.exists():
        print(f"No existe: {transcripts_dir}", file=sys.stderr)
        return 1
    md_files = list(transcripts_dir.rglob("*.md"))
    if not md_files:
        zip_path = transcripts_dir / "Chats_Cursor.zip"
        if zip_path.exists():
            print(
                f"Nota: no hay .md en {transcripts_dir}; extrae Chats_Cursor.zip localmente "
                f"o usa --transcripts-dir con la carpeta descomprimida.",
                file=sys.stderr,
            )
        else:
            print(f"No hay transcripts .md en {transcripts_dir}", file=sys.stderr)
        return 1
    rows: list[tuple[int, str, int]] = []
    for path in sorted(transcripts_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        users = parse_user_messages(text)
        if not users:
            continue
        longest = max(len(u) for u in users)
        rows.append((longest, path.name, len(users)))
    rows.sort(reverse=True)
    print(f"{'chars':>8}  {'users':>5}  file")
    for length, name, count in rows[:25]:
        print(f"{length:8d}  {count:5d}  {name}")
    return 0


def curated_cases() -> list[dict]:
    """Casos seleccionados y anonimizados a mano desde el corpus COE."""
    raw = [
        {
            "id": "ext_001_ci_pytest_triage",
            "category": "dev_agent",
            "language": "en",
            "source_transcript": "synthetic/dev_pytest_failure_session_v1 + cursor_dev pattern",
            "instruction": (
                "Our CI pipeline for ExampleService just failed on the main branch and I need "
                "help triaging before we merge. Read the pytest output below carefully, count "
                "how many tests failed and in which areas, group failures by root cause rather "
                "than file order, and tell me what to fix first based on user impact and how "
                "isolated each fix is. Prioritize authentication problems (401 responses or "
                "missing token fields in JSON) before billing or rounding errors, and only then "
                "consider flaky or environmental failures. For each failure, give the likely "
                "component, a one-line hypothesis, and whether it looks like a quick fix or a "
                "deeper refactor. Respond with a numbered priority list and keep code change "
                "suggestions minimal until I confirm the ordering. Do not assume you can rerun "
                "the suite — analyze only what appears in the log."
            ),
            "payload": (
                "[pytest] ExampleService CI — main @ build 4821\n"
                "FAILED tests/test_auth.py::test_login_returns_token - AssertionError: expected 200, got 401\n"
                "FAILED tests/test_auth.py::test_refresh_rotates_token - KeyError: 'refresh_token'\n"
                "FAILED tests/test_billing.py::test_invoice_total - assert 49.99 == 50.00\n"
                "FAILED tests/test_notifications.py::test_push_payload - TimeoutError: broker not reachable\n"
                "======================== 4 failed, 42 passed in 4.2s ========================"
            ),
            "payload_type": "text",
        },
        {
            "id": "ext_002_gallery_scale_block",
            "category": "dev_agent",
            "language": "es",
            "source_transcript": "RegistroVisitas_Sprint_6.md",
            "instruction": (
                "Cuando intento probar la app en dispositivo físico y en Mac, la galería es "
                "demasiado grande (unas 10.000 fotos en el móvil y 70.000 en el Mac) y la "
                "aplicación se bloquea al intentar crear thumbnails para tantas fotos: se queda "
                "colgada y no vuelve. Necesito un plan concreto para que el explorador de galería "
                "funcione con bibliotecas grandes: paginación real, carga lazy de thumbnails y "
                "límites de memoria. Prioriza cambios de arquitectura antes de micro-optimizar UI. "
                "No propongas cargar toda la galería en memoria."
            ),
            "payload": (
                "[runtime] ExampleApp PhotoExplorer\n"
                "Gallery load started — totalCount=70432 (macOS)\n"
                "Thumbnail batch 1/?? — allocated 512MB, still loading...\n"
                "(process unresponsive > 30s, watchdog terminated UI thread)"
            ),
            "payload_type": "text",
        },
        {
            "id": "ext_003_continuation_misuse",
            "category": "dev_agent",
            "language": "es",
            "source_transcript": "RegistroVisitas_Sprint_6.md",
            "instruction": (
                "En iOS la app crashea al abrir la galería con este error: "
                "'SWIFT TASK CONTINUATION MISUSE: requestImageWithLimitedAccessOptions tried to "
                "resume its continuation more than once'. Ocurre en PlatformPhotoService.swift "
                "alrededor de la línea 220. Diagnose la causa raíz y propón un fix mínimo que "
                "garantice que la continuación solo se resume una vez. Incluye snippet corregido."
            ),
            "payload": (
                "Thread 1: Fatal error: SWIFT TASK CONTINUATION MISUSE\n"
                "requestImageWithLimitedAccessOptions(phAsset:size:localIdentifier:)\n"
                "  at PlatformPhotoService.swift:220\n"
                "PHImageManager callback invoked twice (degraded + final image)"
            ),
            "payload_type": "text",
        },
        {
            "id": "ext_004_global_styles_refactor",
            "category": "refactor",
            "language": "es",
            "source_transcript": "cursor_crear_un_sistema_de_estilos_glob.md",
            "instruction": (
                "Analiza el proyecto. Vamos a hacer una intervención transversal. Tenemos "
                "repartidos por toda la app `#if os(` para seleccionar colores según plataforma. "
                "Partimos de NativeMacOSStyles.swift y creamos un sistema de estilos global: "
                "dependencias de plataforma concentradas en un fichero, con `backgroundColor` y "
                "el resto de estilos declarados con compilación condicional iOS/macOS, de modo que "
                "en el resto de la app usemos `AppColors.background` sin `#if os`. Para inventariar "
                "estilos, busca `#if os(iOS)` y `#if os(macOS)` en el repo. Entrega plan por fases "
                "y lista de ficheros a tocar primero."
            ),
            "payload_type": "text",
            "payload": "",
        },
        {
            "id": "ext_005_gps_import_wizard",
            "category": "debug",
            "language": "es",
            "source_transcript": "Resolucion de issues 2025-08-26.md",
            "instruction": (
                "Estoy probando el wizard de importación de visitas desde la galería, pero siempre "
                "dice «No se encontraron fotos con datos de ubicación GPS» aunque las fotos sí "
                "tienen GPS en Fotos. Revisa el flujo HistoricalImport → PhotoMetadataService → "
                "TripClustering: sospecho que estamos convirtiendo PhotoAsset a PHAsset y perdemos "
                "location en el camino. Dame hipótesis ordenadas y qué logs añadir para confirmar."
            ),
            "payload": (
                "[wizard] scanPhotoLibrary dateRange=2024-01-01..2024-12-31\n"
                "photosInRange=1842\n"
                "extractLocationData → 0 items with GPS\n"
                "throw HistoricalImportError.noPhotosWithLocation"
            ),
            "payload_type": "text",
        },
        {
            "id": "ext_006_macos_photo_permission",
            "category": "ux",
            "language": "es",
            "source_transcript": "cursor_mejoras_de_ui_en_el_proyecto.md",
            "instruction": (
                "En macOS no está pidiendo permiso para acceder a fotos y fallan el wizard y "
                "añadir fotos a un RecordItem. En iOS, sin permiso, mostramos un botón en el "
                "detalle. ¿Deberíamos pedir permiso al abrir la app y bloquear si el usuario "
                "deniega? Evalúa pros/contras y recomienda un flujo unificado iOS/macOS con "
                "mínimos cambios de arquitectura."
            ),
            "payload_type": "text",
            "payload": "",
        },
        {
            "id": "ext_007_xcode_warnings",
            "category": "dev_agent",
            "language": "es",
            "source_transcript": "cursor_analizar_los_warnings_de_la_apli.md",
            "instruction": (
                "Analicemos los warnings de la aplicación ExampleApp. Clasifícalos por severidad "
                "y prioridad de corrección (Swift 6 concurrency primero, deprecaciones iOS 17, "
                "luego lógica). Cuenta cuántos warnings principales hay y resume en una tabla "
                "severity | file | one-line fix. No ejecutes cambios todavía."
            ),
            "payload": (
                "[xcodebuild] ExampleApp (iOS Simulator)\n"
                "warning: non-sendable result type '[RecordItem]' cannot be sent from main actor "
                "(MapScreenViewModel.swift:60)\n"
                "warning: capture of 'itemsWithLocation' with non-sendable type '[RecordItem]' "
                "(MapScreenViewModel.swift:64)\n"
                "warning: 'init(coordinateRegion:...)' was deprecated in iOS 17.0 (DetailView.swift:297)\n"
                "warning: 'onChange(of:perform:)' was deprecated in iOS 17.0 (GalleryView.swift:96)\n"
                "warning: string interpolation produces a debug description for an optional "
                "(SyncViewModel.swift:201)\n"
                "warning: initialization of immutable value 'totalItems' was never used "
                "(SyncViewModel.swift:206)\n"
                "warning: 'catch' block is unreachable (ImportViewModel.swift:105)\n"
                "** BUILD SUCCEEDED ** 15 warnings emitted"
            ),
            "payload_type": "text",
        },
        {
            "id": "ext_008_selection_lost",
            "category": "bugfix",
            "language": "es",
            "source_transcript": "RegistroVisitas_Sprint_6.md",
            "instruction": (
                "En iOS selecciono varias fotos en el explorador de galería pero al cerrar la "
                "hoja modal no se asocian al RecordItem — la selección se pierde. Revisa "
                "PhotoExplorerView y el callback `onPhotosSelected` en PhotoSelectionView. "
                "El fix debe propagar la selección al servicio de datos sin duplicar estado."
            ),
            "payload": (
                "[ui] PhotoExplorerView dismissed — selectedPhotos=3\n"
                "PhotoSelectionView.onPhotosSelected: nil (callback not wired)\n"
                "DataService: getPhotoReferences(visitId=[UUID]) → 0 photos"
            ),
            "payload_type": "text",
        },
    ]
    cases = []
    for item in raw:
        entry = {k: anonymize(v) if isinstance(v, str) else v for k, v in item.items()}
        if not entry.get("payload"):
            entry.pop("payload", None)
            entry.pop("payload_type", None)
        cases.append(entry)
    return cases


def write_curated(output: Path) -> int:
    cases = curated_cases()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(cases, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(cases)} cases → {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract/anonymize E2E prompts from COE transcripts")
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=COE_TRANSCRIPTS,
        help="COE corpus transcripts directory",
    )
    parser.add_argument("--list", action="store_true", help="List longest user messages per file")
    parser.add_argument(
        "--write-curated",
        action="store_true",
        help="Write data/e2e_prompts_extensive.json from curated selection",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if args.list:
        return list_transcripts(args.transcripts_dir)
    if args.write_curated:
        return write_curated(args.output)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
