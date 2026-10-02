"""Cost and latency accounting for the support system, from TRACES (Day 4) and a price card.

Everything here is arithmetic on measured token counts:
  * tokens come from the spans (real counts from the local model's tokenizer),
  * prices come from a ``PriceCard`` (SIMULATED: the local model costs nothing, so we price its token counts as if a
    hosted model had produced them; token counts differ between tokenizers, so treat dollars as relative, not absolute),
  * latency comes from a ``LatencyModel`` whose parameters are ASSUMPTIONS, not measurements.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common import llm, tracing  # noqa: E402
from common.tracing import SpanRecord  # noqa: E402


@dataclass(frozen=True)
class PriceCard:
    name: str
    input: float  # USD per million tokens
    output: float
    cached_input: float | None = None  # USD per million cached-prefix tokens (read)

    @classmethod
    def from_llm(cls, model: str) -> PriceCard:
        inp, cached, out = llm.PRICES[model]
        return cls(model, inp, out, cached)

    def cost(self, input_tokens: float, output_tokens: float, cached_tokens: float = 0.0) -> float:
        """``input_tokens`` EXCLUDES the cached part (billing convention used throughout common/llm.py)."""
        cached_rate = self.cached_input if self.cached_input is not None else self.input
        return (
            input_tokens * self.input + cached_tokens * cached_rate + output_tokens * self.output
        ) / 1e6


CHEAP = PriceCard.from_llm("claude-haiku-4-5")
STRONG = PriceCard.from_llm("claude-sonnet-5")


@dataclass(frozen=True)
class LatencyModel:
    """ASSUMED, not measured: time to first token grows with the prompt (prefill), then tokens stream out."""

    base_s: float = 0.25  # network + queueing
    prefill_tok_s: float = 6000.0  # prompt tokens processed per second
    decode_tok_s: float = 70.0  # output tokens per second

    def call(self, input_tokens: float, output_tokens: float) -> float:
        return self.base_s + input_tokens / self.prefill_tok_s + output_tokens / self.decode_tok_s


def chat_calls(spans: Sequence[SpanRecord]) -> list[tuple[int, int]]:
    """(input tokens, output tokens) of every model call, in start order."""
    return [
        (s.get(tracing.INPUT_TOKENS, 0), s.get(tracing.OUTPUT_TOKENS, 0))
        for s in sorted(spans, key=lambda s: s.start_ns)
        if s.get(tracing.OPERATION) == "chat"
    ]


def trace_cost(spans: Sequence[SpanRecord], card: PriceCard) -> float:
    return sum(card.cost(i, o) for i, o in chat_calls(spans))


DEFAULT_LATENCY = LatencyModel()


def trace_latency(
    spans: Sequence[SpanRecord], model: LatencyModel = DEFAULT_LATENCY, tool_s: float = 0.05
) -> float:
    """Sequential estimate: model calls and tool calls of one conversation happen one after another."""
    n_tools = sum(1 for s in spans if s.get(tracing.OPERATION) == "execute_tool")
    return sum(model.call(i, o) for i, o in chat_calls(spans)) + n_tools * tool_s
