from __future__ import annotations

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
