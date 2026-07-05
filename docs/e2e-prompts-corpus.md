# E2E prompts — corpus extenso (transcripts COE)

Dataset para benchmark PCM con **instrucciones largas** derivadas de sesiones reales de agente Cursor, anonimizadas según [COE benchmark-harness §13](https://github.com/ntnglz/Context-Optimization-Engine/blob/main/docs/benchmark-harness.md).

## Origen

| Pieza | Ubicación |
|-------|-----------|
| Transcripts crudos (local, gitignored) | `../Context-Optimization-Engine/data/benchmarks/corpus/transcripts/` |
| Casos benchmark COE (referencia) | `../Context-Optimization-Engine/data/benchmarks/cases/dev_agent/` |
| Dataset E2E PCM (versionado) | `data/e2e_prompts_extensive.json` |

Los exports provienen del zip `Chats_Cursor.zip` en el corpus COE. **No se commitean** los `.md` brutos.

## Anonimización

| Original | Sustituto |
|----------|-----------|
| RegistroVisitas / AquiEstuve | `ExampleApp` |
| Visit / Trip | `RecordItem` / `RecordType` |
| Rutas `/Users/…`, `/Volumes/DevSSD/…` | `/workspace/app`, `/workspace/build` |
| UUIDs de visita | `[UUID]` |
| Nombres de autor | `developer` |

Script: `scripts/extract_e2e_from_transcripts.py`

```bash
# Inventariar mensajes User más largos en transcripts locales
python scripts/extract_e2e_from_transcripts.py --list

# Regenerar JSON curado (8 casos)
python scripts/extract_e2e_from_transcripts.py --write-curated
```

## Casos incluidos (`e2e_prompts_extensive.json`)

| ID | Categoría | Idioma | Origen transcript |
|----|-----------|--------|-------------------|
| `ext_001_ci_pytest_triage` | dev_agent | en | pytest CI + log |
| `ext_002_gallery_scale_block` | dev_agent | es | Sprint_6 — galería 10k/70k fotos |
| `ext_003_continuation_misuse` | dev_agent | es | Sprint_6 — continuation misuse |
| `ext_004_global_styles_refactor` | refactor | es | estilos globales `#if os` |
| `ext_005_gps_import_wizard` | debug | es | wizard GPS |
| `ext_006_macos_photo_permission` | ux | es | permisos fotos macOS |
| `ext_007_xcode_warnings` | dev_agent | es | xcodebuild warnings |
| `ext_008_selection_lost` | bugfix | es | selección galería perdida |

## Benchmark

Compresor de referencia: **`pcm-granite`** (`glossary_only` automático en el script E2E).

```bash
# Corpus corto (4 prompts sintéticos)
PYTHONPATH=src .venv/bin/python scripts/e2e_benchmark.py \
  --compressor-model pcm-granite --reasoning-effort none -q

# Corpus extenso (8 prompts desde transcripts)
PYTHONPATH=src .venv/bin/python scripts/e2e_benchmark.py \
  --compressor-model pcm-granite \
  --prompts data/e2e_prompts_extensive.json \
  --reasoning-effort none -q
```

### Resultados `pcm-granite` (2026-07-05, `reasoning=none`)

| Métrica | Corpus corto (4) | Corpus extenso (8) |
|---------|------------------|---------------------|
| Similitud PCM vs baseline | 93.25% | **90.62%** |
| Coste PCM vs baseline | −55% | −21% |
| Coste concise vs baseline | −64% | **−70%** |
| Ahorro output concise vs PCM | 26.4% | **56.7%** |

Run JSON extenso: [`data/e2e/runs/e2e_mistral_medium_3_5_20260705_205454.json`](../data/e2e/runs/e2e_mistral_medium_3_5_20260705_205454.json)

Métricas completas: [output_directives_e2e.md](../data/benchmarks/output_directives_e2e.md).

## Añadir casos

1. Exportar chat → COE `corpus/transcripts/`
2. Recortar mensaje User + payload (log, tool output, código)
3. Anonimizar con la tabla anterior
4. Añadir entrada en `curated_cases()` del script o editar JSON directamente
5. `pytest tests/test_e2e_prompts_extensive.py -v`
