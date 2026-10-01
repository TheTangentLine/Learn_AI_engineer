# Week 1: How LLMs Work (for Engineers)

**Phase 1: Foundations** · ~2–3 hours/day · Prerequisites: Python, HTTP/JSON, basic terminal

Most LLM bugs in production trace back to one of five facts covered this week:
- the model sees **tokens**, not characters
- it **samples** from a probability distribution
- it has a **finite context**, and you pay for all of it on every call
- the API is **stateless**
- the network is **unreliable**

This week turns those facts into engineering instincts and leaves you with a reusable `common/llm.py` that every later week builds on.

## Learning goals
By Sunday you can:
- Explain training vs. inference, and what "next-token prediction" means for your code.
- Call Claude and GPT (and a local model) through one interface, with token and cost accounting.
- Predict token counts and cost for any input, and read logprobs as a confidence signal.
- Explain temperature, top-k and top-p from the math, and implement them yourself.
- Reason about context-window cost, the KV cache and prompt caching, and measure the savings.
- Write concurrent, rate-limit-safe, streaming LLM code.
- Pick a model for a task with data instead of vibes.

## Schedule
| Day | Lesson | Challenge | Needs API key? |
|---|---|---|---|
| 1 | [Mental model + setup + first calls](day1_mental-model-and-setup.md) | Build a mini provider-agnostic `llm.py` with cost tracking | Yes (one provider) |
| 2 | [Tokenization & logprobs](day2_tokenization-and-logprobs.md) | Token-cost estimator + low-confidence answer flagger | No (local model) |
| 3 | [Sampling & decoding](day3_sampling-and-decoding.md) | Measure output diversity vs. temperature and plot it | No (local model) |
| 4 | [Context window, KV cache & prompt caching](day4_context-kv-cache-prompt-caching.md) | KV-cache calculator + prompt-caching benchmark | Part B only |
| 5 | [Streaming, async, retries & rate limits](day5_streaming-async-retries.md) | Concurrent batch processor with backoff | No (fake client), Yes for the real run |
| 6 | [Model landscape & selection](day6_model-landscape-and-selection.md) | Model-selection benchmark table | Yes |
| 7 | [Weekly challenge: `llm-cli`](day7_weekly-challenge.md) | Terminal chat app with streaming, provider switching and a cost meter | Yes |

Solutions are in [solutions/](solutions/). Try each challenge for at least 45 minutes before opening them.

## Setup (once)
```bash
# from the repo root
brew install uv            # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra local      # creates .venv with everything for Week 1
cp .env.example .env       # then paste at least one API key
uv run python -m common.llm "Say hi"   # smoke test
```
Without `uv`: `python3 -m venv .venv && .venv/bin/pip install -e ".[local]"`.

## Running solutions
```bash
uv run python weeks/week01_how-llms-work/solutions/day2_solution.py
```
