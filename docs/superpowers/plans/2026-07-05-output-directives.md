# Instrucciones de salida (output directives) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Añadir al prompt de ida un bloque `RESPONSE:` configurable (`output_style`) y medir coste por turno con tokens de entrada y salida por separado.

**Architecture:** Dos módulos nuevos en PCM (`output_directives`, `message_assembly`) ensamblan el system prompt. `turn_cost.py` calcula `cost_total` desde tokens ida/vuelta. El proxy inyecta directives en system; COE `build_pcm_messages` usa `build_system_prompt`. E2E benchmark añade brazos `normal` vs `concise`.

**Tech Stack:** Python 3.11+, pytest, tiktoken, httpx (proxy), Mistral SDK (E2E opcional)

**Spec:** `docs/superpowers/specs/2026-07-05-rcm-v0-design.md`

---

## File map

| File | Responsibility |
|------|----------------|
| `src/pcm/output_directives.py` | `build_output_directives()` — bloques `RESPONSE:` concise/normal |
| `src/pcm/message_assembly.py` | `build_system_prompt()`, `build_proxy_system_prompt()` |
| `src/pcm/turn_cost.py` | `TurnCostMetrics`, `compute_turn_cost()`, conteo messages[] |
| `src/pcm/proxy.py` | Inyectar directives; headers `X-PCM-*-Tokens`, `X-PCM-Cost-Total-USD` |
| `src/pcm/e2e_benchmark.py` | Brazos E2E con `output_style`; `cost_delta` en summary |
| `src/pcm/__init__.py` | Exportar API pública |
| `data/examples/output_directives_cases.json` | Gold strings |
| `tests/test_output_directives.py` | Tests directives |
| `tests/test_message_assembly.py` | Tests ensamblaje |
| `tests/test_turn_cost.py` | Tests coste |
| `tests/test_proxy_output_directives.py` | Tests inyección proxy |
| `docs/pcm-and-coe.md` | Diagrama ida + `output_style` |
| `../Context-Optimization-Engine/src/coe/pcm/compose.py` | `output_style` en `build_pcm_messages` |

---

### Task 1: `output_directives.py`

**Files:**
- Create: `src/pcm/output_directives.py`
- Create: `data/examples/output_directives_cases.json`
- Create: `tests/test_output_directives.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_output_directives.py
import json
from pathlib import Path

from pcm.output_directives import build_output_directives, has_response_block

CASES = json.loads(
    Path("data/examples/output_directives_cases.json").read_text(encoding="utf-8")
)


def test_concise_contains_seven_rules():
    text = build_output_directives(response_lang="en", output_style="concise")
    assert text.startswith("RESPONSE:")
    assert "Language: en" in text
    assert "Answer only what was asked" in text
    assert "No greeting, politeness" in text
    assert "Do not restate or recap" in text
    assert "Do not describe your process" in text
    assert "shortest form" in text
    assert "one line stating what is missing" in text


def test_normal_matches_gold():
    text = build_output_directives(response_lang="es", output_style="normal")
    gold = next(c for c in CASES if c["id"] == "normal_es")
    assert text == gold["expected"]


def test_has_response_block_detects_existing():
    system = "TASK=review INPUT=python\n\nRESPONSE:\n- Language: en"
    assert has_response_block(system) is True
    assert has_response_block("TASK=review INPUT=python") is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Volumes/DevSSD/PythonProjects/Prompt Compression Middleware"
pytest tests/test_output_directives.py -v
```

Expected: FAIL — `ModuleNotFoundError: pcm.output_directives`

- [ ] **Step 3: Create gold cases JSON**

```json
[
  {
    "id": "normal_en",
    "output_style": "normal",
    "response_lang": "en",
    "expected": "RESPONSE:\n- Language: en\n- Answer clearly using only the provided context"
  },
  {
    "id": "normal_es",
    "output_style": "normal",
    "response_lang": "es",
    "expected": "RESPONSE:\n- Language: es\n- Answer clearly using only the provided context"
  },
  {
    "id": "concise_en",
    "output_style": "concise",
    "response_lang": "en",
    "expected": "RESPONSE:\n- Language: en\n- Answer only what was asked\n- No greeting, politeness, hedging, or filler\n- Do not restate or recap the question\n- Do not describe your process unless asked\n- Use the shortest form that preserves required facts (lists over paragraphs)\n- If uncertain, one line stating what is missing; no speculation"
  }
]
```

Save to `data/examples/output_directives_cases.json`.

- [ ] **Step 4: Implement `output_directives.py`**

```python
# src/pcm/output_directives.py
from __future__ import annotations

from typing import Literal

OutputStyle = Literal["concise", "normal"]

_RESPONSE_MARKER = "RESPONSE:"

_NORMAL_TEMPLATE = """RESPONSE:
- Language: {response_lang}
- Answer clearly using only the provided context"""

_CONCISE_TEMPLATE = """RESPONSE:
- Language: {response_lang}
- Answer only what was asked
- No greeting, politeness, hedging, or filler
- Do not restate or recap the question
- Do not describe your process unless asked
- Use the shortest form that preserves required facts (lists over paragraphs)
- If uncertain, one line stating what is missing; no speculation"""


def has_response_block(text: str) -> bool:
    return _RESPONSE_MARKER in text


def build_output_directives(
    *,
    response_lang: str = "en",
    output_style: OutputStyle = "normal",
) -> str:
    if output_style == "concise":
        return _CONCISE_TEMPLATE.format(response_lang=response_lang)
    return _NORMAL_TEMPLATE.format(response_lang=response_lang)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_output_directives.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add src/pcm/output_directives.py data/examples/output_directives_cases.json tests/test_output_directives.py
git commit -m "feat: add output directives for outbound prompt RESPONSE block"
```

---

### Task 2: `message_assembly.py`

**Files:**
- Create: `src/pcm/message_assembly.py`
- Create: `tests/test_message_assembly.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_message_assembly.py
from pcm.message_assembly import build_proxy_system_prompt, build_system_prompt


def test_build_system_prompt_order():
    system = build_system_prompt(
        compressed_instruction="TASK=review INPUT=python CHECK=race",
        response_lang="en",
        output_style="concise",
        pcm_interpretation_hint="Interpret PCM key=value lines.",
    )
    parts = system.split("\n\n")
    assert parts[0] == "TASK=review INPUT=python CHECK=race"
    assert parts[1].startswith("RESPONSE:")
    assert parts[2] == "Interpret PCM key=value lines."


def test_build_system_prompt_without_hint():
    system = build_system_prompt(
        compressed_instruction="TASK=review INPUT=python",
        response_lang="en",
        output_style="normal",
    )
    assert "TASK=review" in system
    assert "RESPONSE:" in system
    assert system.count("\n\n") == 1


def test_build_proxy_system_prompt_no_instruction():
    """Proxy: instrucción comprimida va en user, no en system."""
    system = build_proxy_system_prompt(
        response_lang="en",
        output_style="concise",
        pcm_interpretation_hint="PCM hint here.",
    )
    assert not system.startswith("TASK=")
    assert system.startswith("RESPONSE:")
    assert system.endswith("PCM hint here.")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_message_assembly.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `message_assembly.py`**

```python
# src/pcm/message_assembly.py
from __future__ import annotations

from typing import Literal

from .output_directives import OutputStyle, build_output_directives

def build_system_prompt(
    *,
    compressed_instruction: str,
    response_lang: str = "en",
    output_style: OutputStyle = "normal",
    pcm_interpretation_hint: str = "",
) -> str:
    """COE / compose: instrucción PCM + RESPONSE + hint opcional."""
    parts = [
        compressed_instruction.strip(),
        build_output_directives(
            response_lang=response_lang,
            output_style=output_style,
        ),
    ]
    if pcm_interpretation_hint.strip():
        parts.append(pcm_interpretation_hint.strip())
    return "\n\n".join(parts)


def build_proxy_system_prompt(
    *,
    response_lang: str = "en",
    output_style: OutputStyle = "normal",
    pcm_interpretation_hint: str = "",
) -> str:
    """Proxy: solo RESPONSE + hint (instrucción comprimida en role=user)."""
    parts = [
        build_output_directives(
            response_lang=response_lang,
            output_style=output_style,
        ),
    ]
    if pcm_interpretation_hint.strip():
        parts.append(pcm_interpretation_hint.strip())
    return "\n\n".join(parts)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_message_assembly.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pcm/message_assembly.py tests/test_message_assembly.py
git commit -m "feat: assemble system prompt with output directives"
```

---

### Task 3: `turn_cost.py`

**Files:**
- Create: `src/pcm/turn_cost.py`
- Create: `tests/test_turn_cost.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_turn_cost.py
from pcm.turn_cost import TurnCostMetrics, compute_turn_cost, count_message_tokens


def test_count_message_tokens_sums_roles():
    messages = [
        {"role": "system", "content": "a b c"},
        {"role": "user", "content": "d e"},
    ]
    # tiktoken gpt-4: exact count checked against canonical helper
    from pcm.canonical import count_tokens
    expected = count_tokens("a b c") + count_tokens("d e")
    assert count_message_tokens(messages) == expected


def test_compute_turn_cost_splits_prices():
    metrics = compute_turn_cost(
        messages=[{"role": "user", "content": "hello world"}],
        output_text="short answer",
        output_tokens=3,
        input_price_per_m=1.5,
        output_price_per_m=7.5,
    )
    assert metrics.input_tokens > 0
    assert metrics.output_tokens == 3
    assert metrics.cost_input == round(metrics.input_tokens * 1.5 / 1_000_000, 6)
    assert metrics.cost_output == round(3 * 7.5 / 1_000_000, 6)
    assert metrics.cost_total == round(metrics.cost_input + metrics.cost_output, 6)


def test_turn_cost_metrics_to_dict():
    m = TurnCostMetrics(
        input_tokens=100,
        input_tokens_instruction=40,
        input_tokens_context=60,
        output_tokens=20,
        input_price_per_m=1.5,
        output_price_per_m=7.5,
        cost_input=0.00015,
        cost_output=0.00015,
        cost_total=0.0003,
    )
    d = m.to_dict()
    assert d["cost_total"] == 0.0003
    assert d["output_tokens"] == 20
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_turn_cost.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement `turn_cost.py`**

```python
# src/pcm/turn_cost.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import count_tokens


@dataclass
class TurnCostMetrics:
    input_tokens: int
    input_tokens_instruction: int
    input_tokens_context: int
    output_tokens: int
    input_price_per_m: float
    output_price_per_m: float
    cost_input: float
    cost_output: float
    cost_total: float
    cost_total_baseline: float | None = None
    cost_delta: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "input_tokens_instruction": self.input_tokens_instruction,
            "input_tokens_context": self.input_tokens_context,
            "output_tokens": self.output_tokens,
            "input_price_per_m": self.input_price_per_m,
            "output_price_per_m": self.output_price_per_m,
            "cost_input": round(self.cost_input, 6),
            "cost_output": round(self.cost_output, 6),
            "cost_total": round(self.cost_total, 6),
            "cost_total_baseline": self.cost_total_baseline,
            "cost_delta": self.cost_delta,
        }


def count_message_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content") or ""
        if isinstance(content, str):
            total += count_tokens(content)
    return total


def compute_turn_cost(
    *,
    messages: list[dict[str, Any]],
    output_text: str,
    output_tokens: int | None = None,
    input_price_per_m: float = 1.5,
    output_price_per_m: float = 7.5,
    input_tokens_instruction: int = 0,
    input_tokens_context: int = 0,
    cost_total_baseline: float | None = None,
) -> TurnCostMetrics:
    input_tokens = count_message_tokens(messages)
    if output_tokens is None:
        output_tokens = count_tokens(output_text)
    cost_input = input_tokens * input_price_per_m / 1_000_000
    cost_output = output_tokens * output_price_per_m / 1_000_000
    cost_total = cost_input + cost_output
    cost_delta = None
    if cost_total_baseline is not None:
        cost_delta = cost_total - cost_total_baseline
    return TurnCostMetrics(
        input_tokens=input_tokens,
        input_tokens_instruction=input_tokens_instruction,
        input_tokens_context=input_tokens_context,
        output_tokens=output_tokens,
        input_price_per_m=input_price_per_m,
        output_price_per_m=output_price_per_m,
        cost_input=cost_input,
        cost_output=cost_output,
        cost_total=cost_total,
        cost_total_baseline=cost_total_baseline,
        cost_delta=cost_delta,
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_turn_cost.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pcm/turn_cost.py tests/test_turn_cost.py
git commit -m "feat: add per-turn input/output cost metrics"
```

---

### Task 4: Integración proxy

**Files:**
- Modify: `src/pcm/proxy.py`
- Create: `tests/test_proxy_output_directives.py`

- [ ] **Step 1: Extend `ProxyConfig`**

En `src/pcm/proxy.py`, añadir a `ProxyConfig`:

```python
output_style: str = "normal"
response_lang: str = "en"
```

Y en `from_env()`:

```python
output_style=os.getenv("PCM_OUTPUT_STYLE", "normal"),
response_lang=os.getenv("PCM_RESPONSE_LANG", "en"),
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_proxy_output_directives.py
import pytest

from pcm.proxy import ChatProxy, ProxyConfig
from pcm.compressor import PromptCompressor


@pytest.mark.asyncio
async def test_transform_injects_response_block(monkeypatch):
    config = ProxyConfig(
        inject_pcm_system=True,
        output_style="concise",
        response_lang="en",
        compress_roles=frozenset(),
    )
    proxy = ChatProxy(PromptCompressor(), config=config)

    class FakeUpstream:
        provider = "mistral"
        model = "mistral-small"
        base_url = "https://example.com"
        api_key = "x"
        supports_reasoning_effort = False
        reasoning_effort = ""

    body = {"messages": [{"role": "user", "content": "Hello"}]}
    transformed, _ = await proxy.transform_request(
        body, upstream=FakeUpstream(), compress=False
    )
    system = next(m["content"] for m in transformed["messages"] if m["role"] == "system")
    assert "RESPONSE:" in system
    assert "Answer only what was asked" in system
```

- [ ] **Step 3: Refactor `_inject_pcm_system` → `_inject_system_blocks`**

Reemplazar lógica de inyección por:

```python
from .message_assembly import build_proxy_system_prompt
from .output_directives import has_response_block

def _inject_system_blocks(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not self.config.inject_pcm_system:
        return messages

    out = [dict(m) for m in messages]
    system_idx = next((i for i, m in enumerate(out) if m.get("role") == "system"), None)

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

    out[system_idx] = {**out[system_idx], "content": f"{existing}\n\n{base}".strip()}
    return out
```

Actualizar llamada en `transform_request`: `messages = self._inject_system_blocks(messages)`.

- [ ] **Step 4: Añadir coste en `forward_chat_completion`**

Tras `payload = response.json()`:

```python
from .e2e_benchmark import MISTRAL_INPUT_PRICE_PER_M, MISTRAL_OUTPUT_PRICE_PER_M
from .turn_cost import compute_turn_cost

usage = payload.get("usage") or {}
output_tokens = int(usage.get("completion_tokens") or 0)
assistant_content = ""
choices = payload.get("choices") or []
if choices:
    assistant_content = self._normalize_content(choices[0].get("message", {}).get("content"))

turn_cost = compute_turn_cost(
    messages=transformed.get("messages", []),
    output_text=assistant_content,
    output_tokens=output_tokens or None,
    input_price_per_m=MISTRAL_INPUT_PRICE_PER_M,
    output_price_per_m=MISTRAL_OUTPUT_PRICE_PER_M,
)
stats.turn_cost = turn_cost  # añadir campo opcional a ProxyCompressionStats
```

Extender `ProxyCompressionStats`:

```python
turn_cost: TurnCostMetrics | None = None
```

Y `stats_as_headers`:

```python
if stats.turn_cost:
    headers["X-PCM-Input-Tokens"] = str(stats.turn_cost.input_tokens)
    headers["X-PCM-Output-Tokens"] = str(stats.turn_cost.output_tokens)
    headers["X-PCM-Cost-Total-USD"] = f"{stats.turn_cost.cost_total:.6f}"
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_proxy_output_directives.py tests/test_proxy.py -v
```

Expected: PASS (ajustar `test_proxy.py` si asumía texto exacto del system antiguo)

- [ ] **Step 6: Commit**

```bash
git add src/pcm/proxy.py tests/test_proxy_output_directives.py
git commit -m "feat: proxy injects output directives and reports turn cost"
```

---

### Task 5: Export público

**Files:**
- Modify: `src/pcm/__init__.py`

- [ ] **Step 1: Añadir exports**

```python
from .output_directives import build_output_directives, has_response_block
from .message_assembly import build_system_prompt, build_proxy_system_prompt
from .turn_cost import TurnCostMetrics, compute_turn_cost

__all__ = [
    # ... existentes ...
    "build_output_directives",
    "has_response_block",
    "build_system_prompt",
    "build_proxy_system_prompt",
    "TurnCostMetrics",
    "compute_turn_cost",
]
```

- [ ] **Step 2: Commit**

```bash
git add src/pcm/__init__.py
git commit -m "feat: export output directives and turn cost API"
```

---

### Task 6: E2E benchmark — brazos `output_style`

**Files:**
- Modify: `src/pcm/e2e_benchmark.py`
- Modify: `tests/test_e2e_benchmark.py` (si existe lógica de summary)

- [ ] **Step 1: Añadir campo `output_style` a `E2EEntry`**

```python
output_style: str = "normal"
concise_llm: LLMCallResult | None = None  # misma instrucción PCM, style=concise
```

- [ ] **Step 2: En el runner principal**, tras la llamada `compressed_llm`, añadir tercera llamada:

```python
from .message_assembly import build_system_prompt

concise_system = build_system_prompt(
    compressed_instruction=compressed_instruction,
    response_lang=item.get("language", "en"),
    output_style="concise",
    pcm_interpretation_hint=MISTRAL_PCM_SYSTEM_PROMPT,
)
concise_llm = client.complete(user_prompt, system_prompt=concise_system)
```

- [ ] **Step 3: Extender `E2EReport.summary`**

```python
"avg_concise_output_tokens": ...,
"avg_output_token_savings_pct": ...,  # vs compressed_llm normal
"avg_cost_delta_concise_vs_baseline": ...,
```

`cost_delta` = `concise_llm.estimated_cost_usd` vs `original_llm.estimated_cost_usd` por entry.

- [ ] **Step 4: Test unitario del summary** (sin llamar Mistral)

Mock `E2EEntry` con `LLMCallResult` ficticios y verificar que `summary` incluye las claves nuevas.

- [ ] **Step 5: Commit**

```bash
git add src/pcm/e2e_benchmark.py tests/
git commit -m "feat: E2E benchmark compares normal vs concise output_style"
```

---

### Task 7: Integración COE (repo hermano)

**Files:**
- Modify: `../Context-Optimization-Engine/src/coe/pcm/compose.py`

- [ ] **Step 1: Añadir parámetros a `build_pcm_messages`**

```python
def build_pcm_messages(
    *,
    compressed_instruction: str,
    optimized_context: str,
    user_question: str,
    response_lang: str,
    output_style: str = "normal",
    system_addendum: str = "",
    pcm_interpretation_hint: str = "",
) -> list[dict[str, str]]:
    try:
        from pcm.message_assembly import build_system_prompt
        system = build_system_prompt(
            compressed_instruction=compressed_instruction,
            response_lang=response_lang,
            output_style=output_style,  # type: ignore[arg-type]
            pcm_interpretation_hint=pcm_interpretation_hint or system_addendum,
        )
    except ImportError:
        addendum = system_addendum.strip() or (
            f"Answer in {response_lang}. Answer clearly using only the provided context."
        )
        system = f"{compressed_instruction.strip()}\n\n{addendum}"
    # user sin cambios...
```

- [ ] **Step 2: Propagar en `optimize_with_pcm`**

Añadir `output_style: str = "normal"` y pasarlo a `build_pcm_messages`.

- [ ] **Step 3: Test en COE**

```python
# tests/test_pcm_compose.py
def test_build_pcm_messages_includes_response_block():
    from coe.pcm.compose import build_pcm_messages
    messages = build_pcm_messages(
        compressed_instruction="TASK=review INPUT=python",
        optimized_context="ctx",
        user_question="Review this",
        response_lang="en",
        output_style="concise",
    )
    assert "RESPONSE:" in messages[0]["content"]
    assert "Answer only what was asked" in messages[0]["content"]
```

- [ ] **Step 4: Commit en COE**

```bash
cd "../Context-Optimization-Engine"
git add src/coe/pcm/compose.py tests/test_pcm_compose.py
git commit -m "feat: wire output_style into PCM message assembly"
```

---

### Task 8: Documentación

**Files:**
- Modify: `docs/pcm-and-coe.md`

- [ ] **Step 1: Añadir fila a la tabla de ida**

```markdown
| **Output style** | **PCM** (`output_style`) | `normal` vs `concise` — reglas RESPONSE en system |
```

- [ ] **Step 2: Añadir sección métricas**

Documentar headers `X-PCM-Input-Tokens`, `X-PCM-Output-Tokens`, `X-PCM-Cost-Total-USD` y env vars `PCM_OUTPUT_STYLE`, `PCM_RESPONSE_LANG`.

- [ ] **Step 3: Commit**

```bash
git add docs/pcm-and-coe.md
git commit -m "docs: output directives and turn cost metrics"
```

---

### Task 9: Verificación final

- [ ] **Step 1: Run full test suite**

```bash
cd "/Volumes/DevSSD/PythonProjects/Prompt Compression Middleware"
pytest tests/ -v --ignore=tests/test_e2e_benchmark.py
```

Expected: all PASS

- [ ] **Step 2: Smoke manual proxy (opcional)**

```bash
PCM_OUTPUT_STYLE=concise python run.py --proxy
# curl chat completion → comprobar headers X-PCM-Cost-Total-USD
```

- [ ] **Step 3: Checklist spec §8**

Marcar criterios de aceptación en la spec o en PR description.

---

## Spec coverage (self-review)

| Spec § | Task |
|--------|------|
| §4 output directives concise/normal | Task 1 |
| §5 API build_system_prompt | Task 2 |
| §5 integración proxy | Task 4 |
| §5 integración COE | Task 7 |
| §6 TurnCostMetrics | Task 3 |
| §6 headers proxy | Task 4 |
| §6.3 E2E brazos | Task 6 |
| §8 criterios aceptación | Task 9 |
| §7 docs | Task 8 |
