"""Week 1 Day 1 - Solution: a mini provider-agnostic LLM wrapper.

Approach
--------
* One public function, ``complete()``, with a per-provider branch that translates
  our neutral arguments into each SDK's request shape and back.
* Usage is normalised so ``input_tokens`` always means *uncached* input
  (Anthropic already reports it that way; OpenAI includes cached tokens, so we subtract).
* Cost comes from a price table; unknown models (local) cost 0.
* ``__main__`` sends one prompt to every provider that is configured and prints a table.

Run:  uv run python weeks/week01_how-llms-work/solutions/day1_solution.py

Expected output (numbers vary):
    provider   model              in  out   cost($)  latency  stop        answer
    anthropic  claude-opus-5      27   38  0.001085    2.10s  end_turn    Sunlight scatters off...
    openai     gpt-6.1-sol        25   31  0.000360    1.40s  completed   Shorter blue wavelengths...
    truncation check -> stop_reason='max_tokens'
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = {"anthropic": "claude-opus-5", "openai": "gpt-6.1-sol", "ollama": "llama3.2:3b"}
# USD per 1M tokens: (input, output). Snapshot - check the pricing pages.
PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-luna": (0.10, 0.50),
}

TOTAL_COST_USD = 0.0  # stretch: running total across calls


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    stop_reason: str | None


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICES:
        return 0.0
    inp, out = PRICES[model]
    return (input_tokens * inp + output_tokens * out) / 1_000_000


def complete(
    prompt: str,
    system: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
) -> LLMResponse:
    global TOTAL_COST_USD
    provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
    model = model or DEFAULT_MODELS[provider]
    messages = [{"role": "user", "content": prompt}]
    t0 = time.perf_counter()

    if provider == "anthropic":
        import anthropic

        kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        msg = anthropic.Anthropic().messages.create(**kwargs)
        text = "".join(b.text for b in msg.content if b.type == "text")
        in_tok, out_tok, stop = msg.usage.input_tokens, msg.usage.output_tokens, msg.stop_reason

    elif provider == "openai":
        from openai import OpenAI

        kwargs = {"model": model, "input": messages, "max_output_tokens": max_tokens}
        if system:
            kwargs["instructions"] = system
        r = OpenAI().responses.create(**kwargs)
        cached = r.usage.input_tokens_details.cached_tokens if r.usage.input_tokens_details else 0
        text = r.output_text
        in_tok, out_tok = r.usage.input_tokens - cached, r.usage.output_tokens
        stop = r.incomplete_details.reason if r.incomplete_details else r.status

    elif provider == "ollama":
        from openai import OpenAI

        client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), api_key="ollama"
        )
        if system:
            messages = [{"role": "system", "content": system}, *messages]
        r = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
        text = r.choices[0].message.content or ""
        in_tok, out_tok = r.usage.prompt_tokens, r.usage.completion_tokens
        stop = r.choices[0].finish_reason

    else:
        raise ValueError(f"unknown provider {provider!r}")

    cost = _cost(model, in_tok, out_tok)
    TOTAL_COST_USD += cost
    return LLMResponse(text, provider, model, in_tok, out_tok, cost, time.perf_counter() - t0, stop)


def configured_providers() -> list[str]:
    providers = []
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append("anthropic")
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    if os.getenv("USE_OLLAMA") == "1":
        providers.append("ollama")
    return providers


if __name__ == "__main__":
    providers = configured_providers()
    if not providers:
        raise SystemExit("No provider configured: add a key to .env (or set USE_OLLAMA=1).")

    prompt = "Why is the sky blue?"
    system = "You are a concise assistant. Answer in one sentence."
    print(
        f"{'provider':10} {'model':18} {'in':>4} {'out':>4} {'cost($)':>9} {'latency':>8}  "
        f"{'stop':10}  answer"
    )
    for p in providers:
        try:
            r = complete(prompt, system=system, provider=p)
        except Exception as e:  # keep going so one broken provider doesn't hide the others
            print(f"{p:10} ERROR: {type(e).__name__}: {e}")
            continue
        answer = r.text.replace("\n", " ")[:60]
        print(
            f"{r.provider:10} {r.model:18} {r.input_tokens:>4} {r.output_tokens:>4} "
            f"{r.cost_usd:>9.6f} {r.latency_s:>7.2f}s  {r.stop_reason!s:10}  {answer}"
        )

    # Acceptance check: truncation must be visible in stop_reason.
    short = complete("Write a 200-word essay about rivers.", provider=providers[0], max_tokens=5)
    print(f"\ntruncation check -> stop_reason={short.stop_reason!r}  text={short.text!r}")
    print(f"total spent this run: ${TOTAL_COST_USD:.6f}")
