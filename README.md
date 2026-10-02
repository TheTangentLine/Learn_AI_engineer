# AI Engineer Roadmap: 12 Weeks, Day by Day

A hands-on path from "I can call an API" to "I can design, evaluate, secure, fine-tune, deploy and operate LLM products". 

- **12 weeks · 7 days each**: 6 lesson days + 1 weekly-challenge day (~2.5–3 h/day).
- **Every day ends with a challenge** and has a **reference solution** in `solutions/`. Try it for 45+ minutes before peeking.
- **Provider-agnostic:** Claude, GPT and local open models (Ollama) behind one small wrapper, [`common/llm.py`](common/llm.py).
- **Verified code:** solutions are executed while the course is written; anything that needs an API key and has not been run is labelled as such in its lesson.

## The map

| Phase | Week | Theme | Weekly challenge |
|---|---|---|---|
| **1 · Foundations** | [1](weeks/week01_how-llms-work/) | How LLMs work: tokens, sampling, context, streaming, model choice | `llm-cli` terminal chat |
| | [2](weeks/week02_prompting-and-structured-outputs/) | Prompting & structured outputs, workflows, memory, DSPy | Document-extraction pipeline |
| **2 · Retrieval** | [3](weeks/week03_embeddings-and-rag/) | Embeddings, vector search, chunking, RAG, hybrid search | Docs Q&A bot |
| | [4](weeks/week04_advanced-rag-and-evaluation/) | Advanced RAG & retrieval evaluation | Before/after eval report |
| **3 · Agents** | [5](weeks/week05_tool-use-and-agents/) | Tool use, agent loop, tool design, MCP, context engineering, sandboxes | Research agent with verified citations |
| | [6](weeks/week06_frameworks-and-multi-agent/) | Frameworks & multi-agent systems, human-in-the-loop, agent evaluation | Customer-support multi-agent system |
| **4 · Production quality** | [7](weeks/week07_evals-observability-llmops/) | Evals, LLM-as-judge, CI gates, tracing, cost & latency, A/B tests & drift | Production harness for Week 6 |
| | [8](weeks/week08_security-and-guardrails/) | Security: threat models, prompt injection, guardrails, agent permissions, privacy, grounding, red-teaming | Red-team & harden three systems; report + regression suite |
| **5 · Models** | [9](weeks/week09_transformers-from-scratch/) | Transformers from scratch: PyTorch, BPE, attention, the Qwen2 decoder, training a mini-GPT, MoE, FlashAttention | KV cache + top-p sampling, benchmarked |
| | 10 | Fine-tuning: SFT, LoRA/QLoRA, DPO, merge & export | Small model beats prompted base |
| | 11 | Inference & deployment: quantisation, vLLM, FastAPI, Docker | Deploy + load test |
| **6 · Capstone** | 12 | A production AI product, end to end | Demo + portfolio README |

> **Status:** Weeks 1–9 are written (lessons, solutions, tests); Weeks 10–12 are being added one at a time, each building on the previous weeks' artifacts. Run `./scripts/check.sh` to lint and run every offline test.

## Setup

```bash
brew install uv                      # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra local                # core deps + torch/transformers for local-model lessons
cp .env.example .env                 # add ANTHROPIC_API_KEY and/or OPENAI_API_KEY (one is enough)
uv run python -m common.llm "Say hi" # smoke test
uv run pytest                        # offline tests for the shared wrapper (no keys needed)
```
Without `uv`: `python3 -m venv .venv && .venv/bin/pip install -e ".[local]"`.

Set a **monthly spend limit** in each provider console before you start. Most lessons cost cents; benchmarks and agent runs can cost dollars.

## Repository layout

```
common/llm.py                   provider-agnostic wrapper: complete / stream / async / structured, usage + cost
tests/                          offline tests for the shared modules in common/
scripts/check.sh                lint + every offline test; scripts/mutate.py = a tiny mutation tester
weeks/weekNN_<slug>/
  README.md                     goals, schedule, prerequisites
  dayN_<slug>.md                lesson + daily challenge
  day7_weekly-challenge.md      the weekly project: brief, requirements, rubric, stretch goals
  solutions/dayN_solution.py    reference solutions (read the module docstring first)
  solutions/weekly/             reference implementation of the weekly challenge
archive/                        the previous, shorter tracks (kept for reference)
```

## How to use each day
1. Read the lesson (~45 min). Run the code snippets; change things and watch what happens.
2. Do the **daily challenge** without looking at the solution. Use the acceptance criteria as your definition of done.
3. Compare with `solutions/dayN_solution.py`. Note *differences in approach*, not just correctness.
4. Do at least one **stretch goal** on days where you have energy.
5. On day 7, build the weekly project, **write its tests**, and tick the checklist at the bottom of the page.

## Conventions
- Model IDs and prices live in `common/llm.py` / `.env`. They go stale every few months; update them in one place.
- Sampling parameters (`temperature`, `top_p`) are not available on every current model. Lessons say so explicitly where it matters.
- Costs are shown in USD per million tokens (MTok) from the providers' pricing pages as of Oct 2026.
