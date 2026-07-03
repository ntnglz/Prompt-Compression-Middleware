"""Utilidades para separar instrucción comprimible y payload intacto."""

from __future__ import annotations

import re

_CODE_FENCE_RE = re.compile(r"\n*```", re.MULTILINE)


def split_instruction_and_payload(text: str) -> tuple[str, str]:
    """Divide texto en instrucción (comprimible) y payload (código/texto adjunto)."""
    text = str(text).strip()
    match = _CODE_FENCE_RE.search(text)
    if not match:
        return text, ""
    instruction = text[: match.start()].strip()
    payload = text[match.start() :].strip()
    return instruction, payload


def join_instruction_and_payload(instruction: str, payload: str) -> str:
    """Recombina instrucción PCM con payload sin modificar."""
    instruction = instruction.strip()
    payload = payload.strip()
    if payload:
        return f"{instruction}\n\n{payload}"
    return instruction
