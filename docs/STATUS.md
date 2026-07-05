# PCM status (maintainers)

> Documento de mantenimiento — visitantes: ver [README](../README.md) y [getting-started](getting-started.md).

## Experimento PCM (julio 2026)

| Hito | Estado |
|------|--------|
| Proxy + compresor Ollama | ✅ |
| Fine-tune MLX (`pcm-compressor`) | ✅ Fase 3a |
| Fine-tune cloud (`pcm-granite`) | ✅ Fase 3b |
| Validación E2E Mistral | ✅ 94.50% similitud semántica |
| Adopción visitante (v1.0.0) | ✅ docs EN, packaging, demo stub |

Experimento **cerrado con éxito**. Detalle: [experimento-pcm-conclusiones.md](experimento-pcm-conclusiones.md).

## Modelos compresor

| Modelo | Origen | Uso |
|--------|--------|-----|
| `granite4.1:3b` | Ollama Hub | Baseline default |
| `pcm-granite` | RunPod Fase 3b | Recomendado producción |
| `pcm-compressor` | MLX Mac Fase 3a | Experimental |

## CI

| Comando | Alcance |
|---------|---------|
| `./scripts/ci-local.sh` | pytest `-m "not integration"`, leakage check si hay train v2 |
| `./scripts/ci-local-full.sh` | + integración Ollama |
| `python run.py --ci` | alias de ci-local |

No hay GitHub Actions en este repo; CI local documentado en [getting-started](getting-started.md).

## Roadmap

| Fase | Estado |
|------|--------|
| 1 Prototipo | ✅ |
| 2 System prompt | ✅ |
| 3a MLX | ✅ |
| 3b Cloud | ✅ |
| Adopción visitante v1.0.0 | ✅ |
| COE integración | Planificado — [repo COE](https://github.com/ntnglz/Context-Optimization-Engine) |
| 3 RL / 4 LLM IR | Investigación |

## Enlaces internos

- [Fase 3b — granite cloud](fase3b-granite-cloud.md)
- [Fase 3 — MLX](fase3-finetuning.md)
- [Benchmarks](../data/benchmarks/README.md)
- [Plan adopción visitante](plans/2026-07-05-pcm-visitor-adoption.md)
