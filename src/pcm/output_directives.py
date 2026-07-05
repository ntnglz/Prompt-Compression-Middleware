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
