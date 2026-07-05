# Instrucciones de salida en el prompt de ida (PCM)

> **Estado:** Aprobado (revisión 2)  
> **Fecha:** 2026-07-05  
> **Repo:** Prompt Compression Middleware  
> **Alcance:** módulo interno de PCM · no es un producto ni middleware aparte

---

## 1. Problema

Una petición al LLM destino tiene esta forma:

```
messages[]  →  LLM  →  respuesta
```

Hoy PCM solo actúa sobre la **ida**:

| Pieza del prompt de ida | Quién la prepara hoy | Qué hace |
|-------------------------|----------------------|----------|
| Instrucción del usuario | **PCM** | Comprime la intención (`TASK=…`) sin perder semántica |
| Contexto (logs, RAG, historial) | **COE** (cliente) | Optimiza bloques adjuntos |
| Idioma de respuesta | **PCM / system** | `Answer in {response_lang}` |
| Forma de la respuesta | — | **No definido** |

Falta la segunda mitad del contrato con el modelo: **cómo debe redactar la respuesta** para gastar pocos tokens de salida sin perder lo necesario.

Las métricas actuales de PCM miden solo tokens de **entrada** (instrucción comprimida). No miden tokens de **salida** ni el coste conjunto. Eso impide optimizar lo que importa en producción: **input_price × tokens_ida + output_price × tokens_respuesta**.

---

## 2. Qué es (y qué no es)

Es un **módulo más dentro de PCM** que ensambla el bloque de instrucciones de salida e lo añade al prompt de ida.

| Es | No es |
|----|-------|
| Texto fijo + parametrizable en el system prompt | Un middleware entre LLM y cliente |
| Parte del ensamblaje `messages[]` junto a la instrucción PCM | Una segunda pasada LLM sobre la respuesta |
| Complemento de `response_lang` | Un nivel N6 de COE |
| Medible con precios distintos input/output | Un filtro post-generación (fuera de alcance v0) |

**Principio:** la única palanca para reducir tokens de salida *en el momento de generar* es decírselo al modelo **antes** de inferir. Comprimir la respuesta después con otro LLM no tiene sentido: esos tokens ya están pagados y la pasada extra cuesta más.

---

## 3. Composición del prompt de ida

Orden de ensamblaje en el `system` (proxy, MCP, `build_pcm_messages` de COE):

```
1. Instrucción comprimida PCM     ← ya existe (TASK=… o NL comprimido)
2. Instrucciones de salida        ← NUEVO (este módulo)
3. Hint de interpretación PCM     ← opcional, ya existe en proxy (MISTRAL_PCM_SYSTEM_PROMPT)
```

El `user` no cambia de estructura:

```
Context:
{optimized_context}          ← COE

Question:
{user_question}              ← puede coincidir con la instrucción ya comprimida en system
```

COE sigue entregando solo el contexto. PCM sigue comprimiendo solo la instrucción. Este módulo solo añade el bloque §2 del system.

---

## 4. Contenido de las instrucciones de salida

### 4.1 Parámetros

| Parámetro | Default | Función |
|-----------|---------|---------|
| `response_lang` | `en` | Idioma de la respuesta (ya existe) |
| `output_style` | `concise` | Perfil de densidad de salida |

Perfiles `output_style` v0:

| Valor | Uso | Instrucciones añadidas (resumen) |
|-------|-----|----------------------------------|
| `concise` | Agente, tool chains, re-ingesta | Sin cortesía, sin repetir la pregunta, sin meta-comentarios, solo hechos relevantes, formato mínimo |
| `normal` | Usuario final | Responder claro en `response_lang`; sin reglas agresivas de compresión |

### 4.2 Texto `output_style=concise` (v0)

Bloque a inyectar tras la instrucción PCM:

```
RESPONSE:
- Language: {response_lang}
- Answer only what was asked
- No greeting, politeness, hedging, or filler
- Do not restate or recap the question
- Do not describe your process unless asked
- Use the shortest form that preserves required facts (lists over paragraphs)
- If uncertain, one line stating what is missing; no speculation
```

### 4.3 Texto `output_style=normal` (v0)

```
RESPONSE:
- Language: {response_lang}
- Answer clearly using only the provided context
```

Equivalente al `system_addendum` actual de COE `build_pcm_messages`. Es el comportamiento por defecto para no regresar UX humana.

---

## 5. API

Módulo nuevo: `src/pcm/output_directives.py`

```python
def build_output_directives(
    *,
    response_lang: str = "en",
    output_style: Literal["concise", "normal"] = "normal",
) -> str:
    """Bloque RESPONSE: … para el system prompt."""
```

Ensamblaje en `src/pcm/message_assembly.py` (nuevo, o refactor de lógica dispersa en proxy/COE):

```python
def build_system_prompt(
    *,
    compressed_instruction: str,
    response_lang: str = "en",
    output_style: Literal["concise", "normal"] = "normal",
    pcm_interpretation_hint: str = "",
) -> str:
    parts = [
        compressed_instruction.strip(),
        build_output_directives(response_lang=response_lang, output_style=output_style),
    ]
    if pcm_interpretation_hint.strip():
        parts.append(pcm_interpretation_hint.strip())
    return "\n\n".join(parts)
```

**Integración COE** (`src/coe/pcm/compose.py`): sustituir el `system_addendum` hardcodeado por import de `pcm.message_assembly.build_system_prompt` (o `build_output_directives` si COE monta el system por su cuenta).

**Integración proxy** (`src/pcm/proxy.py`): tras comprimir, si el system no lleva ya bloque `RESPONSE:`, inyectarlo según config.

---

## 6. Métricas nuevas

### 6.1 Por petición

Extender `ProxyCompressionStats` y el benchmark E2E con un registro de coste por turno:

```python
@dataclass
class TurnCostMetrics:
    # Ida (lo que enviamos al LLM)
    input_tokens: int              # system + user completos, post-PCM/COE
    input_tokens_instruction: int  # solo instrucción PCM (diagnóstico)
    input_tokens_context: int      # solo contexto (diagnóstico)

    # Vuelta (lo que devuelve el LLM)
    output_tokens: int

    # Coste (precios por millón, configurables)
    input_price_per_m: float
    output_price_per_m: float
    cost_input: float              # input_tokens × input_price_per_m / 1e6
    cost_output: float
    cost_total: float              # cost_input + cost_output  ← métrica a minimizar

    # Comparativa baseline (misma pregunta, instrucción sin comprimir, output_style=normal)
    cost_total_baseline: float | None
    cost_delta: float | None       # cost_total - cost_total_baseline (negativo = ahorro)
```

Conteo de tokens: `tiktoken` encoding `gpt-4` (ya usado en `canonical.count_tokens` y `compressor._count_tokens`). Para `output_tokens`, leer `usage.completion_tokens` de la respuesta upstream cuando exista; si no, estimar con tiktoken sobre el contenido.

### 6.2 Qué reportar

| Métrica | Dónde | Para qué |
|---------|-------|----------|
| `input_tokens` | Proxy headers, E2E JSON | Ver efecto PCM+COE en ida |
| `output_tokens` | idem | Ver efecto `output_style=concise` |
| `cost_total` | idem | KPI principal |
| `cost_delta` vs baseline | E2E benchmark | Demostrar ahorro neto |

Headers proxy nuevos (ejemplo):

```
X-PCM-Input-Tokens: 126
X-PCM-Output-Tokens: 48
X-PCM-Cost-Total-USD: 0.00054
```

### 6.3 Benchmark E2E

Extender `e2e_benchmark.py` con matriz de brazos:

| Brazo | Instrucción | output_style | Mide |
|-------|-------------|--------------|------|
| **baseline** | Natural, sin PCM | `normal` | `cost_total_baseline` |
| **pcm-only** | PCM | `normal` | Ahorro en ida |
| **pcm+concise** | PCM | `concise` | Ahorro en ida + vuelta |
| **pcm+concise+coe** | PCM + contexto COE | `concise` | Stack completo |

Gate v0 (valores orientativos, ajustar con primer run):

| KPI | Umbral | Resultado 2026-07-05 (`reasoning=none`, 4 prompts) |
|-----|--------|-----------------------------------------------------|
| `cost_delta` (pcm+concise vs baseline) | ≤ **−15%** coste total medio | **−56%** ($0.0157 → $0.0069) |
| Comprensión E2E (similitud respuestas) | ≥ **0,90** (no regresar calidad) | **0,945** |
| `output_tokens` (concise vs normal, misma instrucción PCM) | ≤ **−20%** medio | **−42,6%** |

Informe completo: [data/benchmarks/output_directives_e2e.md](../../../data/benchmarks/output_directives_e2e.md)

**Nota:** con `reasoning=high`, Mistral reporta miles de `completion_tokens` aunque el texto visible sea breve; el benchmark de output directives debe usar `--reasoning-effort none`.

---

## 7. Archivos a tocar

```
src/pcm/
├── output_directives.py      # build_output_directives()
├── message_assembly.py       # build_system_prompt()
├── models.py                 # TurnCostMetrics
├── proxy.py                  # inyectar directives; headers de coste
├── canonical.py              # reutilizar count_tokens
└── e2e_benchmark.py          # brazos + cost_total

tests/
├── test_output_directives.py
├── test_message_assembly.py
└── test_turn_cost_metrics.py

data/examples/
└── output_directives_cases.json   # gold strings por output_style × lang
```

COE (cambio mínimo):

```
src/coe/pcm/compose.py   # response_lang + output_style → build_system_prompt
```

Docs:

```
docs/pcm-and-coe.md      # añadir output_style al diagrama de ida
```

---

## 8. Criterios de aceptación

- [ ] `build_output_directives("concise")` produce bloque estable con las 7 reglas de §4.2.
- [ ] `build_system_prompt` concatena instrucción PCM + directives + hint opcional.
- [ ] Proxy inyecta directives cuando faltan; no duplica si ya hay bloque `RESPONSE:`.
- [ ] Cada petición proxy registra `input_tokens`, `output_tokens`, `cost_total`.
- [ ] E2E benchmark reporta `cost_delta` por brazo.
- [ ] Default `output_style=normal` — sin cambio de comportamiento si el cliente no pasa el parámetro.
- [ ] COE `build_pcm_messages` acepta `output_style` sin romper API existente.

---

## 9. Fuera de alcance v0

- Filtro determinista post-respuesta (la palanca es el prompt, no saneo posterior).
- Segunda pasada LLM sobre la respuesta.
- Fine-tuning para aprender estilo conciso.
- `output_style` con más de dos perfiles (p. ej. `json`, `kv`) — v0.1 si hace falta.

---

## Referencias

- `src/pcm/compression_prompts.py` — compresión de instrucción (ida, parte 1)
- `src/coe/pcm/compose.py` — `build_pcm_messages`, `response_lang`
- `src/pcm/e2e_benchmark.py` — precios Mistral input/output ya definidos
- `docs/pcm-and-coe.md` — pipeline actual
