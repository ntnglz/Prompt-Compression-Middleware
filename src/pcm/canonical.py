"""Canonical visitor example — single source of truth for README, demo, and data/examples/."""

from __future__ import annotations

from dataclasses import dataclass

from .prompt_utils import join_instruction_and_payload

COE_REPO = "https://github.com/ntnglz/Context-Optimization-Engine"
COE_GETTING_STARTED = f"{COE_REPO}/blob/master/docs/getting-started.md"

CANONICAL_INSTRUCTION = (
    "Review this Python code carefully for race conditions, memory leaks, "
    "and optimization opportunities. Return a Markdown report ordered by severity."
)

CANONICAL_PAYLOAD = """```python
import threading

CACHE = {}

def get_user(user_id):
    if user_id not in CACHE:
        CACHE[user_id] = expensive_fetch(user_id)
    return CACHE[user_id]

def expensive_fetch(user_id):
    users = []
    for i in range(100000):
        users.append({"id": i, "data": "x" * 100})
    return users[user_id % len(users)]

counter = 0

def increment():
    global counter
    val = counter
    counter = val + 1

threads = [threading.Thread(target=increment) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```"""

CANONICAL_PCM = (
    "TASK=review INPUT=python CHECK=race,leak,perf FORMAT=markdown ORDER=severity"
)

# Long instruction inspired by anonymized Cursor dev-session transcripts in the
# COE benchmark corpus (ExampleService / ExampleApp — see COE dev_agent cases).
CURSOR_DEV_INSTRUCTION = (
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
)

CURSOR_DEV_PAYLOAD = """```text
[pytest] ExampleService CI — main @ build 4821
FAILED tests/test_auth.py::test_login_returns_token - AssertionError: expected 200, got 401
FAILED tests/test_auth.py::test_refresh_rotates_token - KeyError: 'refresh_token'
FAILED tests/test_billing.py::test_invoice_total - assert 49.99 == 50.00
======================== 3 failed, 42 passed in 4.2s ========================
```"""

CURSOR_DEV_PCM = (
    "TASK=triage INPUT=pytest_ci CHECK=auth_401,missing_token,billing_rounding "
    "FORMAT=priority_list ORDER=user_impact START=auth"
)

CANONICAL_USER_MESSAGE = join_instruction_and_payload(
    CANONICAL_INSTRUCTION, CANONICAL_PAYLOAD
)

CANONICAL_COMPRESSED_MESSAGE = join_instruction_and_payload(
    CANONICAL_PCM, CANONICAL_PAYLOAD
)

CURSOR_DEV_USER_MESSAGE = join_instruction_and_payload(
    CURSOR_DEV_INSTRUCTION, CURSOR_DEV_PAYLOAD
)

CURSOR_DEV_COMPRESSED_MESSAGE = join_instruction_and_payload(
    CURSOR_DEV_PCM, CURSOR_DEV_PAYLOAD
)

SHORT_INSTRUCTION = "Hi, reply in one line."


@dataclass(frozen=True)
class TokenMetrics:
    instruction_before: int
    instruction_after: int
    payload_tokens: int

    @property
    def instruction_saved(self) -> int:
        return self.instruction_before - self.instruction_after

    @property
    def instruction_ratio(self) -> float:
        if self.instruction_before == 0:
            return 0.0
        return self.instruction_saved / self.instruction_before

    @property
    def message_before(self) -> int:
        return self.instruction_before + self.payload_tokens

    @property
    def message_after(self) -> int:
        return self.instruction_after + self.payload_tokens

    @property
    def message_saved(self) -> int:
        return self.message_before - self.message_after

    @property
    def message_ratio(self) -> float:
        if self.message_before == 0:
            return 0.0
        return self.message_saved / self.message_before


def count_tokens(text: str) -> int:
    """Token count aligned with PromptCompressor (tiktoken gpt-4 encoding)."""
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def token_metrics(
    instruction: str,
    compressed_instruction: str,
    payload: str,
) -> TokenMetrics:
    return TokenMetrics(
        instruction_before=count_tokens(instruction),
        instruction_after=count_tokens(compressed_instruction),
        payload_tokens=count_tokens(payload),
    )


def format_token_metrics(metrics: TokenMetrics, *, label: str) -> str:
    inst_pct = metrics.instruction_ratio * 100
    msg_pct = metrics.message_ratio * 100
    lines = [
        f"{label}",
        f"  Instruction: {metrics.instruction_before} → {metrics.instruction_after} tokens "
        f"({metrics.instruction_saved} saved, {inst_pct:.0f}% on instruction)",
        f"  Payload:     {metrics.payload_tokens} tokens (unchanged — use COE for context)",
        f"  Full message:{metrics.message_before} → {metrics.message_after} tokens "
        f"({metrics.message_saved} saved, {msg_pct:.0f}% on total input)",
    ]
    return "\n".join(lines)


def canonical_metrics() -> TokenMetrics:
    return token_metrics(CANONICAL_INSTRUCTION, CANONICAL_PCM, CANONICAL_PAYLOAD)


def cursor_dev_metrics() -> TokenMetrics:
    return token_metrics(CURSOR_DEV_INSTRUCTION, CURSOR_DEV_PCM, CURSOR_DEV_PAYLOAD)


def canonical_compress_request() -> dict:
    """Body for POST /compress (REST API)."""
    return {"prompt": CANONICAL_INSTRUCTION}


def canonical_proxy_chat_request() -> dict:
    """Body for POST /v1/chat/completions (proxy)."""
    return {
        "model": "mistral-medium-3.5",
        "messages": [{"role": "user", "content": CANONICAL_USER_MESSAGE}],
    }


def cursor_dev_compress_request() -> dict:
    return {"prompt": CURSOR_DEV_INSTRUCTION}


def cursor_dev_proxy_chat_request() -> dict:
    return {
        "model": "mistral-medium-3.5",
        "messages": [{"role": "user", "content": CURSOR_DEV_USER_MESSAGE}],
    }
