# PCM visitor adoption (v1.0.0)

Reorganize PCM for GitHub visitors: understand, try, and integrate in under 15 minutes without fine-tuning or internal phase jargon.

Pattern: [Context Optimization Engine](https://github.com/ntnglz/Context-Optimization-Engine) visitor adoption (Fase 20 + v1.0.2 mitigation), adapted for PCM (instruction compression, Ollama, proxy-first).

## Canonical example

| | |
|---|---|
| **Input (instruction)** | Review this Python code carefully for race conditions, memory leaks, and optimization opportunities. Return a Markdown report ordered by severity. |
| **Output (PCM)** | `TASK=review INPUT=python CHECK=race,leak,perf FORMAT=markdown ORDER=severity` |
| **Payload** | Fenced Python block (threading + cache) — preserved verbatim |

## Tasks

### A — Canonical example
- [x] `data/examples/canonical_compress.json`
- [x] `data/examples/proxy_chat.json`
- [x] `data/examples/README.md`
- [x] `src/pcm/canonical.py` (single source of truth)
- [x] README before/after matches `python run.py --demo` / `--demo-stub`

### B — Packaging
- [x] `pyproject.toml` with `[dev]`, `[proxy]`, `[mcp]`, `[training]`
- [x] `pip install -e ".[dev]"` → `from pcm import …`
- [x] Reorganize `requirements.txt` (core + Docker)

### C — DX in `run.py`
- [x] `--demo` (Ollama if available)
- [x] `--demo-stub` / automatic fallback without Ollama
- [x] `--quickstart` (demo + OpenAI SDK snippet)
- [x] Visitor `--help` vs `--help-all` (maintainer commands)
- [x] `scripts/mcp/print_cursor_config.py`

### D — User docs (English)
- [x] `README.md`
- [x] `docs/getting-started.md`
- [x] `docs/FAQ.md`
- [x] Archive Spanish pre-migration in `docs/es/` with legacy banner

### E — Adoption content
- [x] When not to use PCM
- [x] Decision guide (mermaid)
- [x] Mode table (`--proxy`, `--http`, `--mcp-http`, `--stdio`)
- [x] PCM + COE link to COE repo getting-started
- [x] README badges
- [x] `CHANGELOG.md`
- [x] `docs/STATUS.md`

### F — Link hygiene
- [x] Maintainer (ES) labels on `docs/experimento-pcm-conclusiones.md`, `fase3*.md`
- [x] README does not send visitors to fine-tuning first
- [x] COE references point to canonical GitHub repo

### G — Verification gate
- [x] `pip install -e ".[dev]"` (or `PYTHONPATH=src` fallback)
- [x] `./scripts/ci-local.sh` / `pytest -m "not integration"`
- [x] `python run.py --demo-stub`
- [x] `pytest tests/test_run_demo.py -v`
- [x] `python run.py --ci`
- [ ] Manual walkthrough README → demo-stub → curl → print_cursor_config.py

## Out of scope

- Retrain models, change compressor/proxy logic
- PyPI publish, new fine-tune
- Rewrite `docs/fase3*.md` or notebooks
- GitHub Actions (document local CI honestly)
