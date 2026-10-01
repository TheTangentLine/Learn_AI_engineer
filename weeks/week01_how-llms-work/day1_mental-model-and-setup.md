# Week 1, Day 1: The Mental Model + Your First Calls

**Time:** ~2.5h · **Needs:** one API key (Anthropic or OpenAI)

## Learning objectives
- Place LLMs correctly in the AI → ML → DL → LLM hierarchy, and explain why that matters for engineering.
- Explain the training loop, and the difference between **training** and **inference**.
- Describe what actually happens during one API call.
- Call Claude and GPT directly with their official SDKs, and read the `usage` block on every response.
- Build a tiny provider-agnostic wrapper that records tokens, cost and latency.

---

## 1. The hierarchy: Russian dolls

**AI → ML → DL → LLMs**

- **Artificial Intelligence**: any system that mimics smart behaviour. A video-game NPC built from `if/else` rules is technically AI.
- **Machine Learning**: you don't write the rules. You write a generic algorithm that *finds* the rules from data.
  - *CS analogy:* traditional programming is writing the body of `def f(x): return x * 2`. ML is giving the computer pairs of `(x, y)` and asking it to write the body.
- **Deep Learning**: ML using **neural networks** with many layers. Classical ML hits a ceiling, while deep nets keep improving with more data and compute.
- **Large Language Models**: a deep-learning architecture (the **Transformer**) trained to **predict the next token** over trillions of tokens of text.

## 2. A neural network, the engineer's view

A neural network is a **directed acyclic graph of matrix multiplications**.

| Component | Engineering equivalent |
|---|---|
| Input layer | Your data converted to numbers (tokens → integer IDs → vectors) |
| Weights | The program's variables, but set by the machine. Frontier models have hundreds of billions to trillions of them. |
| Hidden layers | Pipeline stages: multiply by weights, apply a non-linearity, pass on |
| Activation function | An `if`-like non-linearity (e.g. ReLU: "if negative, output 0"). Without it the whole network collapses into one linear equation. |
| Output layer | For an LLM: one score (a **logit**) for every token in the vocabulary |

## 3. How it learns: TDD on steroids

The training loop runs billions of times:

1. **Forward pass (execution):** feed in text and predict the next token.
2. **Loss (the failing test):** compare the prediction with the real next token. Wrong and confident gives a big loss.
3. **Backward pass (the debugger):** calculus (backpropagation) works out how much each weight contributed to the error.
4. **Optimizer step (the patch):** nudge every weight slightly in the direction that reduces the loss.

An LLM is built in stages:

| Stage | What happens | Result |
|---|---|---|
| **Pre-training** | Next-token prediction over a large share of the public internet, books and code | A "base model" that autocompletes documents |
| **Post-training (SFT)** | Fine-tune on curated *(instruction, ideal answer)* pairs | Follows instructions and chats |
| **Preference/RL training** | RLHF / RL from verifiable rewards (math, code, tool use) | Helpful, safer behaviour. **Reasoning models** learn to "think" before answering |

You will run a small version of all three in Weeks 9–10.

## 4. Inference: what happens when you call the API

```
"What is the capital of France?"
   │ 1. tokenize         →  [3923, 374, 279, 6864, 315, 9822, 30]
   │ 2. prefill          →  one forward pass over ALL prompt tokens (builds the KV cache; Day 4)
   │ 3. decode loop      →  forward pass → logits → sample 1 token → append → repeat (Day 3)
   │ 4. stop             →  end-of-turn token, stop sequence, or max_tokens
   ▼ 5. detokenize       →  "The capital of France is Paris."
```

Four consequences you will feel all course long:

1. **The API is stateless.** The model remembers nothing between calls. "Chat history" is just you re-sending every previous message, and **you pay for those tokens again every turn**.
2. **Output is generated one token at a time.** Long outputs are slow, which is why streaming exists (Day 5). Output tokens cost about 5x input tokens.
3. **Output is sampled.** The same prompt can give different answers (Day 3). Design for that: validate, retry, and evaluate.
4. **The model only knows its training data and your prompt.** Anything else (today's date, your database, your docs) must be put into the context (RAG, Week 3) or fetched with tools (Week 5).

---

## 5. Setup

```bash
brew install uv                 # Python package/project manager
uv sync --extra local           # install deps into .venv
cp .env.example .env            # add ANTHROPIC_API_KEY and/or OPENAI_API_KEY
```

Get keys at <https://console.anthropic.com> and <https://platform.openai.com>. Set a **monthly spend limit** in both consoles before you start: this course makes thousands of calls.

> **Never commit `.env`.** It is already in `.gitignore`. If a key leaks, revoke it immediately.

## 6. First call: Claude (Messages API)

```python
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY from the environment

msg = client.messages.create(
    model="claude-opus-5",
    max_tokens=1024,                     # REQUIRED on Anthropic: hard cap on output tokens
    system="You are a concise assistant. Answer in one sentence.",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)

for block in msg.content:                # content is a LIST of typed blocks
    if block.type == "text":             # (thinking, text, tool_use, ...)
        print(block.text)

print(msg.stop_reason)                   # end_turn | max_tokens | tool_use | refusal ...
print(msg.usage)                         # input_tokens, output_tokens, cache_* tokens
```

## 7. First call: GPT (Responses API)

```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()                         # reads OPENAI_API_KEY

r = client.responses.create(
    model="gpt-6.1-sol",
    instructions="You are a concise assistant. Answer in one sentence.",
    input=[{"role": "user", "content": "Why is the sky blue?"}],
    max_output_tokens=1024,
)

print(r.output_text)                      # convenience: concatenated text output
print(r.usage)                            # input_tokens, output_tokens, input_tokens_details.cached_tokens
```

> The old `client.chat.completions.create(...)` API still works (and Ollama, vLLM and most open-source servers speak it). For OpenAI models, the **Responses API** is the current recommended interface.

## 8. Same idea, different shapes

| Concept | Anthropic | OpenAI |
|---|---|---|
| System prompt | `system="..."` (top-level) | `instructions="..."` |
| Conversation | `messages=[{role, content}]` | `input=[{role, content}]` or a plain string |
| Output cap | `max_tokens` (required) | `max_output_tokens` (optional) |
| Text out | `[b.text for b in msg.content if b.type == "text"]` | `r.output_text` |
| Why it stopped | `msg.stop_reason` | `r.status` / `r.incomplete_details` |
| Input tokens | `usage.input_tokens` (**excludes** cached) | `usage.input_tokens` (**includes** cached) |
| Cached input | `usage.cache_read_input_tokens` | `usage.input_tokens_details.cached_tokens` |

That last row is a trap. The two providers count cached tokens differently, so a naive cost calculator will be wrong for one of them. Today's challenge normalises this.

## 9. What a call costs

Prices are quoted **per million tokens (MTok)**, separately for input and output:

```
cost = input_tokens  × input_price  / 1_000_000
     + output_tokens × output_price / 1_000_000
```

| Model | Input $/MTok | Output $/MTok |
|---|---|---|
| `claude-opus-5` | 5.00 | 25.00 |
| `claude-sonnet-5` | 2.00 | 10.00 |
| `claude-haiku-4-5` | 1.00 | 5.00 |
| `gpt-6.1-sol` | 2.00 | 10.00 |
| `gpt-6-luna` | 0.10 | 0.50 |

*(Snapshot Oct 2026. Always check the providers' pricing pages.)*

## Pitfalls & production notes
- **`max_tokens` is a hard stop, not a target.** If you hit it, the answer is cut mid-sentence (`stop_reason == "max_tokens"`). Always check the stop reason.
- **Reasoning models spend hidden tokens.** Thinking/reasoning tokens are billed as output even when you don't see them. Budget for them.
- **Handle `refusal`.** Current Claude models can stop with `stop_reason == "refusal"` (HTTP 200). Check before you use the content.
- **Don't hardcode model names across your codebase.** Keep them in one config (we use `common/llm.py` + `.env`). Models are deprecated every few months.

---

## Daily Challenge: build a mini `llm.py`

Build `my_llm.py`, a single file that hides provider differences behind one function.

**Requirements**
1. `complete(prompt: str, system: str | None = None, provider: str | None = None) -> LLMResponse`.
2. `LLMResponse` is a dataclass with `text`, `provider`, `model`, `input_tokens`, `output_tokens`, `cost_usd`, `latency_s` and `stop_reason`.
3. The provider defaults to the `LLM_PROVIDER` env var, and the model defaults to a per-provider default dict.
4. Normalise usage so `input_tokens` means *uncached* input on both providers.
5. Compute `cost_usd` from a price table. An unknown model costs `0.0` (so local models work).
6. `__main__`: send the same prompt to every provider you have a key for, then print a comparison table (provider, model, tokens, cost, latency, first 60 chars of the answer).

**Acceptance criteria**
- Running `python my_llm.py` prints one row per available provider, with no crash when a key is missing (skip that provider).
- The costs match a hand calculation from the token counts.
- A truncated answer (`max_tokens=5`) is detectable from `stop_reason`.

**Stretch**
- Add `provider="ollama"`: install [Ollama](https://ollama.com), `ollama pull llama3.2:3b`, and call it with the OpenAI SDK pointed at `base_url="http://localhost:11434/v1"` using `chat.completions.create`.
- Add a module-level running total of cost across calls.

**Solution:** [solutions/day1_solution.py](solutions/day1_solution.py). From tomorrow onwards we use the fuller version in [`common/llm.py`](../../common/llm.py), which adds streaming, async, structured output and prompt caching on top of exactly this design.

## Further reading
- Anthropic: [Messages API](https://docs.claude.com/en/api/messages) · [Models overview](https://docs.claude.com/en/docs/about-claude/models/overview)
- OpenAI: [Responses API](https://developers.openai.com/api/docs/guides/text) · [Models](https://developers.openai.com/api/docs/models)
- 3Blue1Brown, *Neural Networks* series (chapters 1–3): the best visual intuition for weights and backprop.
