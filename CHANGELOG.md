# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2026-07-05

### Added

- Visitor adoption layer: English README, `docs/getting-started.md`, `docs/FAQ.md`
- Canonical example in `data/examples/` and `src/pcm/canonical.py`
- `pyproject.toml` with extras `[dev]`, `[proxy]`, `[mcp]`, `[training]`
- `pip install -e ".[dev]"` — import `pcm` without `PYTHONPATH`
- `python run.py --demo`, `--demo-stub`, `--quickstart`, `--ci`, `--help-all`
- `scripts/mcp/print_cursor_config.py` for Cursor MCP setup
- `tests/test_run_demo.py` for demo and packaging smoke tests
- `docs/STATUS.md`, `docs/es/` legacy Spanish docs, adoption plan in `docs/plans/`

### Changed

- README restructured for GitHub visitors (Docker-first, no phase jargon on landing)
- `run.py` default with no args shows visitor help instead of starting HTTP
- COE references point to https://github.com/ntnglz/Context-Optimization-Engine
- Version unified to **1.0.0** (`pcm`, REST API, proxy)

### Maintainers

- Fine-tuning and experiment docs retained under `docs/` (Spanish), labeled for maintainers
- No compressor, proxy, or training pipeline logic changes

[1.0.0]: https://github.com/ntnglz/Prompt-Compression-Middleware/compare/master...feat/visitor-adoption-1.0.0
