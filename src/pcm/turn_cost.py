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
    cost_input = round(input_tokens * input_price_per_m / 1_000_000, 6)
    cost_output = round(output_tokens * output_price_per_m / 1_000_000, 6)
    cost_total = round(cost_input + cost_output, 6)
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
