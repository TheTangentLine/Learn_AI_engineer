"""Week 1 Day 2 - Solution: token-cost estimator + low-confidence answer flagger.

Part A - Token-cost estimator
    Counts tokens for the same texts with three tokenizers:
      * ``o200k_base`` (tiktoken)     - the encoding family used by recent OpenAI models
      * Qwen2.5 (Hugging Face)        - a typical open-weights tokenizer
      * Claude ``count_tokens`` API   - only if ANTHROPIC_API_KEY is set (never use tiktoken for Claude)
    and prices 1M requests at a few model prices.

Part B - Confidence flagger
    Generates answers greedily with a small local model, reads the log-probability of every
    generated token, and flags answers whose weakest token falls under a threshold.

Run:  uv run python weeks/week01_how-llms-work/solutions/day2_solution.py
(first run downloads Qwen2.5-0.5B-Instruct, ~1 GB, from Hugging Face)
"""

from __future__ import annotations

import math
import os

import tiktoken
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
torch.manual_seed(0)

LOCAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SAMPLES = {
    "english": "The quick brown fox jumps over the lazy dog while the farmer watches.",
    "vietnamese": "Con cáo nâu nhanh nhẹn nhảy qua con chó lười trong khi người nông dân quan sát.",
    "japanese": "素早い茶色の狐が、農夫が見ている間に怠惰な犬を飛び越える。",
    "python": "def fib(n: int) -> int:\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "json_pretty": '{\n    "user_id": 12345,\n    "name": "Ada",\n    "roles": [\n        "admin"\n    ]\n}',
    "json_minified": '{"user_id":12345,"name":"Ada","roles":["admin"]}',
}

# USD per 1M input tokens - what 1M requests of this text would cost as *input*.
INPUT_PRICES = {"claude-opus-5": 5.00, "gpt-6.1-sol": 2.00, "gpt-6-luna": 0.10}


# ----------------------------------------------------------------- Part A


def claude_counter():
    """Return a function counting Claude tokens, or None without a key."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    client = anthropic.Anthropic()

    def count(text: str) -> int:
        resp = client.messages.count_tokens(
            model="claude-opus-5", messages=[{"role": "user", "content": text}]
        )
        # count_tokens includes a few tokens of message framing; subtract an empty message.
        return resp.input_tokens

    empty = count(".")  # framing overhead + 1 token for "."
    return lambda text: count(text) - empty + 1


def part_a(hf_tok) -> None:
    enc = tiktoken.get_encoding("o200k_base")
    claude = claude_counter()

    print("=" * 88)
    print("PART A - Token cost estimator")
    print("=" * 88)
    header = f"{'sample':14} {'chars':>5} {'o200k':>6} {'qwen':>5} {'claude':>6} {'chars/tok':>9}"
    print(header + "  $ per 1M requests (o200k count) " + " / ".join(INPUT_PRICES))
    for name, text in SAMPLES.items():
        n_o200k = len(enc.encode(text))
        n_qwen = len(hf_tok.encode(text))
        n_claude = str(claude(text)) if claude else "-"
        costs = " / ".join(f"{n_o200k * p:,.2f}" for p in INPUT_PRICES.values())
        print(f"{name:14} {len(text):>5} {n_o200k:>6} {n_qwen:>5} {n_claude:>6} "
              f"{len(text) / n_o200k:>9.2f}  {costs}")

    print("\nToken boundaries (o200k_base):")
    for text in ["Strawberry", " strawberry", "9.11 is greater than 9.9"]:
        pieces = [enc.decode([i]) for i in enc.encode(text)]
        print(f"  {text!r:28} -> {pieces}")

    print(
        "\nTakeaways: non-English text and pretty-printed JSON cost noticeably more tokens per\n"
        "character; minify JSON you send to a model, and budget multilingual traffic separately.\n"
        "Tokenizers differ per vendor, so always count with the tokenizer of the model you call."
    )


# ----------------------------------------------------------------- Part B


CONFIDENCE_SYSTEM_PROMPT = (
    "Reply with only the answer itself (a name, number or short phrase). No full sentences."
)


@torch.no_grad()
def answer_with_logprobs(model, tok, question: str, max_new_tokens: int = 24):
    """Greedy-decode an answer and return [(token_str, prob, top_alternatives)]."""
    messages = [
        {"role": "system", "content": CONFIDENCE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    input_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )["input_ids"]

    steps = []
    past = None
    next_input = input_ids
    for _ in range(max_new_tokens):
        out = model(input_ids=next_input, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]                      # scores for every vocab entry
        logprobs = torch.log_softmax(logits, dim=-1)    # normalise -> log P(token | context)
        token_id = int(torch.argmax(logprobs))          # greedy choice
        if token_id == tok.eos_token_id or token_id in tok.all_special_ids:
            break
        top = torch.topk(logprobs, 3)
        alts = [(tok.decode([int(i)]), math.exp(float(lp))) for lp, i in zip(*top, strict=True)]
        steps.append((tok.decode([token_id]), math.exp(float(logprobs[token_id])), alts))
        next_input = torch.tensor([[token_id]])
    return steps


def confidence_report(steps, threshold: float):
    """Score an answer by its weakest *content* token (skip leading punctuation/whitespace).

    With a "short answer only" system prompt, the first content token is usually the whole
    signal: a confident model puts ~100% of its mass there, while a guessing model splits its
    bets across several candidates. We still scan every token and keep the weakest, in case the
    answer has a shaky second word (e.g. a fabricated surname).
    """
    content = [s for s in steps if any(c.isalnum() for c in s[0])]
    if not content:
        return {"answer": "", "min_prob": 0.0, "weakest_token": "", "alternatives": [],
                "flagged": True}
    weakest = min(range(len(content)), key=lambda i: content[i][1])
    return {
        "answer": "".join(t for t, _, _ in steps).strip(),
        "min_prob": content[weakest][1],
        "weakest_token": content[weakest][0],
        "alternatives": content[weakest][2],
        "flagged": content[weakest][1] < threshold,
    }


def part_b(model, tok, threshold: float = 0.5) -> None:
    questions = [
        "What is the capital of France?",
        "What is 12 times 12?",
        "Who wrote Romeo and Juliet?",
        "In which year did World War II end?",
        "Who was the first King of Mars?",
        "What was the population of the village of Zrtovnik in 1823?",
        "What is the 4th word of the 2nd chapter of the novel 'The Glass Orchard' by Mira Toll?",
    ]
    print("\n" + "=" * 88)
    print(f"PART B - Confidence flagger (local {LOCAL_MODEL}, flag if weakest token p < {threshold})")
    print("=" * 88)
    for q in questions:
        rep = confidence_report(answer_with_logprobs(model, tok, q), threshold)
        flag = "LOW CONFIDENCE" if rep["flagged"] else "ok"
        alts = ", ".join(f"{t!r} {p:.0%}" for t, p in rep["alternatives"])
        print(f"\nQ: {q}\nA: {rep['answer']}")
        print(f"   [{flag}] min p={rep['min_prob']:.2f} at {rep['weakest_token']!r} "
              f"(alternatives: {alts})")

    print(
        "\nReading the result: real facts ('capital of France', 'WWII end year') score near\n"
        "p=1.0. Fabricated or unanswerable questions ('King of Mars', the invented novel) show\n"
        "the model splitting its bets across several candidates at 5-30% each - a cheap\n"
        "hallucination signal. It is not proof: models can be confidently wrong (see '12 times\n"
        "12' and 'Shakespeare' above, both correct but only 0.5-0.8 confident). In production,\n"
        "combine this with retrieval (Week 3) and evaluation (Week 7), never use it alone."
    )


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    part_a(tok)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL, dtype=torch.float32).eval()
    part_b(model, tok)
