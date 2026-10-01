# Week 2: Prompting & Structured Outputs

**Phase 1: Foundations** · ~2.5–3 hours/day · Prerequisites: Week 1 (`common/llm.py`, async, retries)

Week 1 gave you a raw text pipe. This week turns it into something you can put inside software: prompts you can **version and measure**, outputs that are **typed and validated**, multi-step **workflows** you control, **memory** that doesn't blow your budget, and prompts that are **optimised by data** instead of by hunch.

## Learning goals
By Sunday you can:
- Structure a prompt (role, context, task, constraints, format, examples) and show with data that it improved.
- Use reasoning controls correctly (CoT, extended thinking/effort, self-consistency) and know when they pay off.
- Get **schema-valid** output from any provider, with validate-and-retry as a safety net.
- Build the five workflow patterns (chain, route, parallelise, orchestrate, evaluate-and-optimise) as plain code.
- Keep a long conversation inside a token budget without forgetting what matters.
- Treat prompts as artifacts: versioned, evaluated and optimised against a metric.

## Schedule
| Day | Lesson | Challenge | Key needed? |
|---|---|---|---|
| 1 | [Prompt anatomy](day1_prompt-anatomy.md) | Rewrite 3 bad prompts; measure before/after | Real run: yes. `--offline` self-test: no |
| 2 | [Reasoning: CoT, thinking, self-consistency](day2_reasoning-and-self-consistency.md) | Self-consistency voter vs. a single call | No (local model) or yes |
| 3 | [Structured outputs](day3_structured-outputs.md) | Typed extraction from messy emails, zero schema failures | `--offline` available |
| 4 | [Workflow patterns](day4_workflow-patterns.md) | Ticket router + parallel summariser | `--offline` available |
| 5 | [Conversation memory](day5_conversation-memory.md) | Bounded-budget chatbot that still recalls turn 1 | `--offline` available |
| 6 | [Prompt optimisation (DSPy ideas)](day6_prompt-optimization.md) | Optimise a classifier vs. a hand-written prompt | `--offline` available |
| 7 | [Weekly challenge](day7_weekly-challenge.md) | Document-extraction pipeline with an accuracy report | `--offline` available |

## How this week is tested
You (probably) have API keys; the author did not. So every solution separates **logic** from the **LLM call**, and the logic is tested with `common/fake.py`, a scripted fake LLM that replaces `llm.complete/structured/stream` inside a `with fake_llm([...])` block. `--offline` demos are labelled *harness self-test*: they check plumbing, never model quality. Real-model numbers come from **your** run.

```bash
uv run python weeks/week02_prompting-and-structured-outputs/solutions/day1_solution.py --offline
uv run pytest weeks/week02_prompting-and-structured-outputs/solutions/weekly
```

## What has been verified
| Item | How |
|---|---|
| Day 1 harness | `--offline` self-test; checkers unit-tested inside the script |
| Day 2 | Real run on a local Qwen2.5-0.5B (numbers in the lesson are real, and honest: voting did **not** help that tiny model); `--offline` tests the extractor and voter |
| Day 3, 4, 5 | `--offline` with scripted models and assertions on control flow |
| Day 6 | Optimiser executed offline; **real DSPy 3.4 code** executed against a scripted OpenAI-compatible server |
| Day 7 `doc_extract` | 9 offline tests on real generated PDFs + real pypdf, with fault injection (mutation-checked) |
| `common/fake.py` | 4 unit tests |

**Not run by the author (no API keys):** any real hosted-model result. Day 1's real prompt scores, Day 3's field accuracy, Day 4's routing quality, Day 5's real recall and Day 7's real accuracy are yours to measure; the harnesses are built for exactly that.
