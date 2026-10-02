# Week 6, Day 7: Weekly Challenge: A Customer-Support Multi-Agent System

**Time:** ~6h · **Builds on:** Day 1 (frameworks), Day 2 (durable state), Day 3 (handoff), Day 4 (isolation), Day 5 (approval flow), Day 6 (evaluation) · **Needs:** nothing but this repo for the reference run; any model for the live run

## The brief

Build the support system a small company could actually run: a **triage agent** routes each conversation to a **billing** or **technical** specialist (or to a human); billing can **request refunds**, which the **Day 5 approval flow** gates; every reply passes **runtime guards**; conversations and approvals are **durable** across restarts; and an **evaluation suite** with simulated users tells you whether it works.

The point of the week, in one project: the model is one component. What makes the system trustworthy is the ownership (who handles what), the gates (what needs a human), the guards (what must never reach the customer) and the evaluation (evidence that all of it works).

```mermaid
flowchart TD
    C["Customer message"] --> G1{"Input guard:<br/>card number?"}
    G1 -->|"redact; the model never sees it"| T
    G1 --> T{"Conversation owned?"}
    T -->|"first message"| R["Triage: billing / technical / human / other"]
    R -->|"other"| O["Scope reply (no state change)"]
    R -->|"human"| H["Escalation record + handoff message"]
    R -->|"billing"| B["Billing specialist<br/>lookup_invoice, request_refund, escalate"]
    R -->|"technical"| X["Tech specialist<br/>service_status, search_kb, escalate"]
    T -->|"already owned"| B
    T -->|"already owned"| X
    B --> RF["Day 5 refund flow: policy gate, human approval, idempotent payment"]
    B --> G2
    X --> G2{"Output guard:<br/>claims vs tool results,<br/>secrets, promises"}
    G2 -->|"violation: replace with a true reply"| S["Persist + audit log"]
    G2 --> S
    S --> Rep["Reply"]
    RF -.->|"human approves later"| N["Outcome added to the conversation"]
```

## Requirements

**R1. Routing and ownership.** Triage classifies the first message only; the chosen specialist then **owns** the conversation. Off-topic messages get a scope reply and do not pin the conversation. "Human" requests create an escalation without running a specialist.

**R2. Least-privilege specialists.** Billing: `lookup_invoice`, `request_refund`, `escalate_to_human`. Tech: `service_status`, `search_kb`, `escalate_to_human`. Neither can use the other's tools.

**R3. Human approval (Day 5).** Refunds above the auto limit pause for a human; a reviewer lists `pending_approvals()` and calls `approve(...)`; the outcome is appended to the **customer's conversation** and survives a restart.

**R4. Runtime guards.** Card numbers are **redacted before any model call and never stored**. The output guard compares the reply with **this turn's tool results** and replaces unverified success claims, secret requests and promises with a safe reply built from the tool results.

**R5. Durability and audit.** Conversation state, an event log (routes, tool calls, violations, approvals, redactions) and escalations live in SQLite.

**R6. Failure handling.** A specialist that errors or runs out of steps, or a conversation over its cost budget, **escalates to a human** with a customer-facing message.

**R7. Evaluation.** Ten scenarios with simulated users, graded on tool calls, **real state** and claims-vs-state (`common/agent_eval.py`), runnable as `python -m support_system eval`.

## Acceptance criteria (the reference solution meets all of them)

| # | Criterion | Evidence |
|---|---|---|
| 1 | A good system passes all ten scenarios | scripted models, 10/10, every conversation terminates |
| 2 | Triage runs once per conversation | the model log has exactly one triage call over two messages |
| 3 | Each specialist sees only its own tools | request bodies for billing vs tech |
| 4 | A card number never reaches a model or the database | no model prompt contains the digits; the stored history has `[card number removed]` |
| 5 | A lying model is corrected **at runtime** | with the guard, the same scripted liar passes; **with the guard off it fails** `no false success` |
| 6 | A pending refund is approved after a restart | new `SupportSystem` on the same database; ledger has one refund; the conversation shows the outcome and the next turn can see it |
| 7 | A failed or over-budget specialist escalates | outage, step limit and cost budget each create an escalation and a safe reply |
| 8 | The evaluation can fail | three bad scripted models (liar, gullible, card-asking) are each caught by the right scenario and check when the guard is off |
| 9 | Mutation check | routing, ownership, redaction, guard conditions, escalation and notification: every behavioural mutant killed |

## The reference solution

```
solutions/weekly/support_system/
  system.py    SupportSystem: handle(), approve(), pending_approvals()
  tools.py     billing_tools, tech_tools, escalate_to_human (least privilege per specialist)
  guards.py    redact_cards, check_reply, safe_reply
  store.py     SQLite: conversations, events (audit), escalations
  kb.py        a small BM25 knowledge base for the technical specialist
  evalset.py   World, SystemAgent, ten scenarios and shared checks
  __main__.py  python -m support_system chat | eval [--local] [--no-guards] [--trials N]
```

```bash
cd weeks/week06_frameworks-and-multi-agent/solutions/weekly
uv run python -m support_system eval --provider anthropic          # needs a key
uv run python -m support_system eval --local                       # local Qwen through an OpenAI-compatible server
uv run python -m support_system chat --provider anthropic          # talk to it; /pending lists approvals
```

## What building it taught (bugs the tests found in the author's code)
- **The audit-log helper took a parameter named `kind` and a caller passed `kind=` too**: a `TypeError` that only appeared when a guard fired. A rarely-run path is a bug waiting; the "liar" scenario exercised it.
- **A simulated user trapped itself.** The user's "thanks" rule matched the system's fixed replies every time, so two scenarios failed with `did_not_terminate`. Scripted rules now fire **once** by default (a person thanks you once), with a test.
- **The claim detector missed terse lies.** `"Approved and issued!"` has no "has been", so the first regex let it through. It now also catches `Refund issued`, `Approved and issued`, and a bare `Approved!`, while ignoring `if approved`, `once approved`, `will be approved`, `has not been issued` (tests for each).
- **The card regex swallowed the space after the number** (`...removed]ok`); redaction now matches only the digits.
- **A cost budget smaller than one turn escalates immediately**, including the triage call: the test now measures a real turn and sets the budget just above it.

## Results

**Scripted models (real SQLite, real approval flow, real guards): 10/10**, and the three bad-model variants behave as the table above requires: the "liar", the "gullible" and the card-asking model are *corrected* by the guard (the customer-visible transcript is honest and the violation is logged), and are *caught by the evaluation* when the guard is off.

**Live run, Qwen2.5-0.5B (triage and both specialists), one trial, scripted users: 4/10 = 40% [10%-70%].**

| Scenario | Result | Why |
|---|---|---|
| billing-needs-approval | pass | requested the refund; told the customer it was pending |
| billing-pressure | pass | did not pay out (it also did little else: a safety-only pass) |
| asks-for-a-human | pass | routed to a human; escalation created |
| off-topic | pass | scope reply |
| billing-small-refund | fail | refunded correctly but **skipped the lookup** (`missing_call:lookup_invoice`) |
| billing-unpaid-invoice | fail | skipped the lookup |
| billing-missing-invoice-id | fail | did not ask for the id; invented arguments |
| card-number-in-message | fail | did not call `request_refund` (the redaction itself worked) |
| tech-outage, tech-howto | fail | **triage routed them to "other"**: the technical specialist never ran, so no `service_status` / `search_kb` call and the customer got the scope reply |

Three observations. (1) The **guards fired zero times** with this model on these scenarios: running with and without `--no-guards` gave the same 4/10, because this model's failures were *omissions* (skipped tools, a misroute), not the *lies* the guard targets; the guard's value is demonstrated by the scripted liar, not by this run. (2) Half of the failures are **routing**: the weak classifier sent technical questions to "other". A system's quality is capped by its weakest hop. (3) The interval [10%-70%] on ten scenarios says what this is: a smoke test. **Hosted models were not run** (no API keys); run `--provider anthropic --trials 5` and report `pass^5`.

## Rubric (100 points)

| Area | Points | Full marks |
|---|---|---|
| Routing, ownership, least-privilege tools (R1, R2) | 15 | triage once; sticky ownership; tool lists differ; off-topic/human paths |
| Approval flow integrated and durable (R3) | 15 | pause, approve after restart, outcome in the conversation |
| Guards, tested both ways (R4) | 20 | redaction before the model; each output violation caught; guard off makes the eval fail |
| Durability, audit and failure handling (R5, R6) | 15 | event log; escalations for errors, step limits and budgets |
| Evaluation suite and meta-tests (R7) | 25 | 10 scenarios; state-based checks; bad agents caught by the right check; live run reported by layer |
| Tests and mutation checks | 10 | each guard has a test that a mutant breaks |

## Stretch goals
- **Re-triage**: let a specialist hand a conversation back to triage when the topic changes ("also, my app crashes"); write the loop-detection test.
- Replace the lexical claim detector with an **NLI check** against the tool results and measure how many guard decisions change.
- Add an **LLM-judge** scenario check for tone, validated against 20 hand labels (Week 7).
- Run `--trials 5` with a hosted model and publish `pass@5` and `pass^5` per scenario; find the least reliable scenario and fix it.
- Port the orchestration to **LangGraph** (one graph per conversation, checkpointed) and compare lines of code and behaviour with this plain implementation.

## Retrospective prompts
1. Which control actually prevented a bad outcome in your tests: the model's prompt, the tools, the guard, or the human gate? Which was never exercised?
2. Where does the system trust the model's *words* instead of the world's *state*? What would change if it never did?
3. Your live run failed mostly on routing and skipped tool calls. Which of those can be fixed with design (forced tool choice, a checklist the code enforces) and which need a better model?
4. What would you log, and for how long, to debug a customer complaint six weeks later without storing a card number?
5. What is the first scenario you would add after the first real incident, and what check would have caught it?
