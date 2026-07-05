"""Canonical visitor example — single source of truth for README, demo, and data/examples/."""

from __future__ import annotations

from .prompt_utils import join_instruction_and_payload

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

CANONICAL_USER_MESSAGE = join_instruction_and_payload(
    CANONICAL_INSTRUCTION, CANONICAL_PAYLOAD
)

CANONICAL_COMPRESSED_MESSAGE = join_instruction_and_payload(
    CANONICAL_PCM, CANONICAL_PAYLOAD
)

SHORT_INSTRUCTION = "Hi, reply in one line."


def canonical_compress_request() -> dict:
    """Body for POST /compress (REST API)."""
    return {"prompt": CANONICAL_INSTRUCTION}


def canonical_proxy_chat_request() -> dict:
    """Body for POST /v1/chat/completions (proxy)."""
    return {
        "model": "mistral-medium-3.5",
        "messages": [{"role": "user", "content": CANONICAL_USER_MESSAGE}],
    }
