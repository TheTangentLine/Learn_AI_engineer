"""Week 5 Day 5 - Solution: an agent that survives a 50-step task inside a fixed context budget.

The task: read 50 pages one at a time (``get_page``); 5 of them hold an access code buried at the END of the page;
then report the codes for pages 3, 17, 29, 44 and 48. Pages are ~2,500 characters, so an untrimmed history keeps
growing and, by step 50, every early page is still being re-sent on every call.

How this is measured (read this before trusting any number):
  * TOKENS are real: counted with the Qwen2.5 tokenizer over the exact messages each model call receives.
  * The MODEL is a scripted policy (``ScriptedReader``), not an LLM: it reads pages in order and, at the end,
    can only report codes that are VISIBLE in its context (page results, notes, or the summary text).
    So "codes found" measures *what each strategy preserves*, which is the question; it says nothing about
    how well a real model would use it. One strategy uses a REAL summariser (Qwen2.5-0.5B, cached).
  * Every strategy's context is checked against the call/result invariant on every single call.

Strategies: naive | truncate | clear-old | clear-old + drop-summary | compaction (Qwen summary) | clear-old + notes

  uv run python weeks/week05_tool-use-and-agents/solutions/day5_solution.py
"""

from __future__ import annotations

import random
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common import context as cx  # noqa: E402
from common.agent import AgentRun, run_agent  # noqa: E402
from common.fake import fake_llm, tool_calls  # noqa: E402
from common.tools import ToolRegistry, tool  # noqa: E402

N_PAGES = 50
NEEDLES = {3: "TANGO-417", 17: "OSCAR-902", 29: "LIMA-058", 44: "ZULU-731", 48: "ECHO-266"}
QUESTION = (
    "Read pages 1 to 50 one at a time with get_page. When you have read them all, report the access code "
    "found on each of pages 3, 17, 29, 44 and 48."
)
CODE_RE = re.compile(r"\b[A-Z]{3,5}-\d{3}\b")
VOCAB = (
    "the quarterly report covers shipping volumes warehouse staffing regional demand supplier contracts "
    "maintenance windows budget variance customer complaints delivery windows route planning inventory "
    "audit findings safety training equipment upgrades vendor negotiations seasonal forecasts pricing"
).split()

# ----------------------------------------------------------------------------- the task


def page_text(n: int) -> str:
    rng = random.Random(n)
    sentences = []
    while sum(len(s) for s in sentences) < 2300:
        sentences.append(
            " ".join(rng.choice(VOCAB) for _ in range(rng.randint(9, 16))).capitalize() + "."
        )
    body = " ".join(sentences)
    body += f"\nTicket reference {rng.choice(['QWE', 'RTY', 'UIO'])}-{rng.randint(100, 999)} (not an access code)."
    if n in NEEDLES:
        body += f"\nFACT: the access code for page {n} is {NEEDLES[n]}."  # at the END: head-truncation loses it
    return f"=== PAGE {n} ===\n{body}"


@tool
def get_page(n: int) -> str:
    """Return the text of page n (1-50).

    Args:
        n: Page number, 1 to 50.
    """
    if not 1 <= n <= N_PAGES:
        raise ValueError(f"page must be between 1 and {N_PAGES}")
    return page_text(n)


# ----------------------------------------------------------------------------- the scripted "model"

PAIR_RES = [
    re.compile(r"access code for page (\d+) is ([A-Z]{3,5}-\d{3})"),
    re.compile(r"code_(\d+): ([A-Z]{3,5}-\d{3})"),
    re.compile(r"page (\d+)[^.\n]{0,60}?\b([A-Z]{3,5}-\d{3})\b", re.I),  # free-text summaries
]


def visible_pairs(messages: list[dict]) -> dict[int, str]:
    """page -> code for everything the context still shows (results, notes, summary text)."""
    pairs: dict[int, str] = {}
    for m in messages:
        text = m.get("content") or ""
        for rx in PAIR_RES:
            for page, code in rx.findall(text):
                pairs.setdefault(int(page), code)
    return pairs


def last_page_called(messages: list[dict]) -> int:
    for m in reversed(messages):
        for c in reversed(m.get("tool_calls", [])):
            if c["name"] == "get_page":
                return c["args"]["n"]
    return 0


class ScriptedReader:
    """Reads pages in order. With ``use_notes`` it saves each code it sees with ``remember``."""

    def __init__(self, use_notes: bool = False):
        self.use_notes = use_notes
        self.context_tokens: list[int] = []  # tokens of what EACH call received
        self.invariant_ok = True
        self.counter = cx.approx_tokens

    def __call__(self, prompt: str, call):
        msgs = call.messages
        self.context_tokens.append(cx.count_tokens(msgs, self.counter))
        self.invariant_ok &= cx.check_invariants(msgs) == []
        last = msgs[-1]
        if self.use_notes and last["role"] == "tool" and last["name"] == "get_page":
            m = re.search(r"access code for page (\d+) is ([A-Z]{3,5}-\d{3})", last["content"])
            if m:
                return tool_calls(("remember", {"key": f"code_{m.group(1)}", "value": m.group(2)}))
        n = last_page_called(msgs)
        if n < N_PAGES:
            return tool_calls(("get_page", {"n": n + 1}))
        found = visible_pairs(msgs)
        reported = {p: found[p] for p in NEEDLES if p in found}
        if not reported:
            return "I could not find any access codes."
        return "Codes: " + ", ".join(f"page {p}: {c}" for p, c in sorted(reported.items()))


# ----------------------------------------------------------------------------- strategies


@dataclass
class Strategy:
    name: str
    hook_factory: Callable  # (Env) -> (hook or None, scratchpad or None)
    use_notes: bool = False


def _clear(keep_last=3):
    return cx.clear_old_tool_results(keep_last=keep_last)


COMPACT_AT = (
    6000  # tokens; ~9 pages. Compaction must see results BEFORE they are cleared (order matters)
)

STRATEGIES = [
    Strategy("naive (no trimming)", lambda s: (None, None)),
    Strategy("truncate results to 800 chars", lambda s: (cx.truncate_tool_results(800), None)),
    Strategy("clear old results (keep 3)", lambda s: (_clear(3), None)),
    Strategy(
        "compaction, no summary",
        lambda s: (
            cx.compact_history(
                cx.drop_summarizer, trigger_tokens=COMPACT_AT, keep_last=3, counter=s.counter
            ),
            None,
        ),
    ),
    Strategy(
        "compaction, Qwen summary",
        lambda s: (
            cx.compact_history(
                s.summarizer, trigger_tokens=COMPACT_AT, keep_last=3, counter=s.counter
            ),
            None,
        ),
    ),
    Strategy(
        "clear old + scratchpad notes",
        lambda s: (lambda pad: (cx.chain(_clear(3), cx.inject_notes(pad)), pad))(cx.Scratchpad()),
        use_notes=True,
    ),
]


@dataclass
class Outcome:
    name: str
    run: AgentRun
    context_tokens: list[int]
    found: dict[int, str] = field(default_factory=dict)
    invariant_ok: bool = True

    @property
    def peak(self) -> int:
        return max(self.context_tokens)

    @property
    def total_input(self) -> int:
        return sum(self.context_tokens)

    @property
    def codes_right(self) -> int:
        return sum(self.found.get(p) == c for p, c in NEEDLES.items())

    @property
    def wrong_codes(self) -> int:
        return sum(1 for p, c in self.found.items() if NEEDLES.get(p) != c)


@dataclass
class Env:
    counter: Callable[[str], int] = cx.approx_tokens
    summarizer: Callable[[str], str] = cx.drop_summarizer


def run_strategy(strategy: Strategy, env: Env, max_steps: int = 80) -> Outcome:
    hook, pad = strategy.hook_factory(env)
    policy = ScriptedReader(use_notes=strategy.use_notes)
    policy.counter = env.counter
    tools = [get_page, *(pad.tools() if pad else [])]
    with fake_llm([(r"(?s).*", policy)]):
        run = run_agent(
            QUESTION,
            ToolRegistry(tools),
            provider="anthropic",
            max_steps=max_steps,
            context_hook=hook,
        )
    found = {int(p): c for p, c in re.findall(r"page (\d+): ([A-Z]{3,5}-\d{3})", run.answer)}
    return Outcome(strategy.name, run, policy.context_tokens, found, policy.invariant_ok)


def main() -> None:
    from common.local_llm import LocalChat

    qwen = LocalChat()
    env = Env(counter=qwen.text_tokens)
    # call the local model DIRECTLY: llm.complete would be intercepted by the scripted policy's fake_llm patch
    system = (
        "You compress an agent's working history. Write a short factual summary that preserves EVERY concrete "
        "fact the agent will need later: numbers, codes, ids, page numbers, decisions. Add nothing new."
    )
    env.summarizer = lambda transcript: qwen(system, transcript, max_new_tokens=250)
    results = [run_strategy(s, env) for s in STRATEGIES]
    base = results[0]
    print(
        f"{N_PAGES} pages, {len(NEEDLES)} buried codes; tokens counted with the Qwen2.5 tokenizer\n"
    )
    print(
        f"{'strategy':40s} {'steps':>5} {'peak ctx':>9} {'total input':>12} {'vs naive':>9} {'codes right':>12} {'wrong':>6}  valid"
    )
    for r in results:
        print(
            f"{r.name:40s} {len(r.run.steps):5d} {r.peak:9d} {r.total_input:12d} {r.total_input / base.total_input:8.0%} "
            f"{r.codes_right:>9d}/{len(NEEDLES)} {r.wrong_codes:6d}  {'yes' if r.invariant_ok else 'NO'}"
        )
    print("\ncontext tokens at selected steps (what that call received):")
    marks = [1, 10, 20, 30, 40, 50]
    print(f"{'strategy':40s} " + " ".join(f"{m:>7d}" for m in marks))
    for r in results:
        print(
            f"{r.name:40s} "
            + " ".join(f"{r.context_tokens[min(m, len(r.context_tokens)) - 1]:7d}" for m in marks)
        )
    print("\nfinal answers:")
    for r in results:
        print(f"  {r.name:40s} {r.run.answer[:110]}")


if __name__ == "__main__":
    main()
