"""Política de cuándo aplicar compresión PCM (umbral mínimo y ahorro neto)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class CompressionPolicy:
    """Evita comprimir instrucciones demasiado cortas o sin ahorro neto de tokens."""

    min_instruction_tokens: int = 12

    def should_compress_instruction(
        self,
        instruction: str,
        count_tokens: Callable[[str], int],
    ) -> tuple[bool, str]:
        instruction = instruction.strip()
        if not instruction:
            return False, "empty_instruction"

        tokens = count_tokens(instruction)
        if tokens < self.min_instruction_tokens:
            return False, f"below_min_tokens:{tokens}<{self.min_instruction_tokens}"
        return True, ""

    def should_apply_compression(
        self,
        original_text: str,
        compressed_text: str,
        count_tokens: Callable[[str], int],
    ) -> tuple[bool, str]:
        original_tokens = count_tokens(original_text)
        compressed_tokens = count_tokens(compressed_text)
        if compressed_tokens >= original_tokens:
            return False, f"no_token_savings:{compressed_tokens}>={original_tokens}"
        return True, ""

    @classmethod
    def from_env(cls) -> "CompressionPolicy":
        import os

        return cls(
            min_instruction_tokens=int(
                os.getenv("PCM_MIN_INSTRUCTION_TOKENS", "12")
            ),
        )
