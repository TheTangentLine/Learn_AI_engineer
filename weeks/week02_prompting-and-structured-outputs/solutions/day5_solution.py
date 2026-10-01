"""Week 2 Day 5 - Solution: tiered conversation memory under a hard token budget.

Tier 1  sliding window   the last K messages, verbatim (exact wording for the current thread)
Tier 2  rolling summary  evicted messages are compressed (in batches) into a short summary
Tier 3  durable facts    things that must survive forever (names, allergies, decisions)

Each turn the prompt is assembled as   system + <memory>facts + summary</memory> + window,
then squeezed to fit ``budget_tokens``. Compare with a sliding-window-only baseline, and with
sending the whole history ("naive").

  uv run python .../day5_solution.py --offline    # scripted model that answers FROM THE PROMPT
  uv run python .../day5_solution.py              # your provider

Why the offline demo is meaningful: a model can only use what is in its prompt. A fake that
answers recall questions strictly from prompt contents makes forgetting observable and testable.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common import llm  # noqa: E402
from common.fake import fake_llm  # noqa: E402


def default_counter(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        return len(text) // 4 + 1


class Memo(BaseModel):
    summary: str = Field(description="Running summary of the conversation so far, at most 60 words")
    facts: list[str] = Field(
        description="Durable facts about the user: names, preferences, constraints, "
        "decisions. Keep every old fact unless contradicted."
    )


COMPRESS_SYSTEM = "You maintain the memory of a chat assistant. Be faithful: never invent facts."
BASE_SYSTEM = "You are a helpful assistant. Use the <memory> block for things said earlier."


@dataclass
class TieredMemory:
    budget_tokens: int = 500
    keep_last: int = 6  # messages kept verbatim
    evict_batch: int = 4  # evict this many at once -> one compress call per batch
    use_summary: bool = True  # False = sliding-window baseline
    counter: callable = default_counter
    window: list[dict] = field(default_factory=list)
    summary: str = ""
    facts: list[str] = field(default_factory=list)
    compress_calls: int = 0

    # ------------------------------------------------------------ writing
    def add(self, role: str, content: str) -> None:
        self.window.append({"role": role, "content": content})
        if len(self.window) > self.keep_last + self.evict_batch:
            evicted, self.window = self.window[: self.evict_batch], self.window[self.evict_batch :]
            while self.window and self.window[0]["role"] != "user":  # keep the window valid
                evicted.append(self.window.pop(0))
            if self.use_summary:
                self._compress(evicted)

    def _compress(self, evicted: list[dict]) -> None:
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in evicted)
        prompt = (
            f"<current_summary>\n{self.summary or '(none)'}\n</current_summary>\n"
            f"<facts>\n" + "\n".join(f"- {f}" for f in self.facts) + "\n</facts>\n"
            f"<new_messages>\n{transcript}\n</new_messages>\n\n"
            "Update the running summary and the list of durable facts."
        )
        memo, _ = llm.structured(prompt, Memo, system=COMPRESS_SYSTEM, max_tokens=600)
        self.summary, self.facts = memo.summary, list(dict.fromkeys(memo.facts))
        self.compress_calls += 1

    # ------------------------------------------------------------ reading
    def _memory_block(self, summary: str) -> str:
        if not (summary or self.facts):
            return ""
        facts = "\n".join(f"- {f}" for f in self.facts)
        return (
            f"\n<memory>\n<facts>\n{facts}\n</facts>\n<summary>\n{summary}\n</summary>\n</memory>"
        )

    def build(self) -> tuple[str, list[dict]]:
        """Return (system_prompt, messages) that fit the budget. Order of sacrifice:
        oldest window messages -> summary text -> (never) the newest message or the facts."""
        window, summary = list(self.window), self.summary

        def size() -> int:
            sys_text = BASE_SYSTEM + self._memory_block(summary)
            return self.counter(sys_text) + sum(self.counter(m["content"]) + 4 for m in window)

        while size() > self.budget_tokens and len(window) > 1:
            window.pop(0)
            while len(window) > 1 and window[0]["role"] != "user":
                window.pop(0)
        words = summary.split()
        while size() > self.budget_tokens and words:
            words = words[: int(len(words) * 0.8)]
            summary = " ".join(words)
        return BASE_SYSTEM + self._memory_block(summary), window

    def prompt_tokens(self) -> int:
        system, messages = self.build()
        return self.counter(system) + sum(self.counter(m["content"]) + 4 for m in messages)


def chat_turn(memory: TieredMemory, user_text: str) -> str:
    memory.add("user", user_text)
    system, messages = memory.build()
    reply = llm.complete(messages, system=system, max_tokens=300).text.strip()
    memory.add("assistant", reply)
    return reply


# ----------------------------------------------------------------- the scenario

OPENING = "Hi! My name is Priya and I'm allergic to peanuts. I'm planning a trip to Lisbon."
FILLER = [
    "What's a good way to spend a rainy afternoon?",
    "Can you recommend a podcast about history?",
    "How do I get better at sketching?",
    "Tell me something interesting about octopuses.",
    "What should I cook tonight, something simple?",
    "Any tips for sleeping better on planes?",
    "How do tides work, in a few sentences?",
    "What's a fun word game to play with friends?",
    "Explain compound interest like I'm twelve.",
    "Which houseplants are hard to kill?",
]
QUESTIONS = [
    ("What is my name?", "priya"),
    ("What am I allergic to?", "peanut"),
    ("Which city am I planning to visit?", "lisbon"),
]


def run_scenario(memory: TieredMemory, turns: int = 30, label: str = "") -> dict:
    chat_turn(memory, OPENING)
    sizes = []
    for i in range(turns - 1):
        chat_turn(memory, FILLER[i % len(FILLER)] + (f" (#{i})" if i >= len(FILLER) else ""))
        sizes.append(memory.prompt_tokens())
    answers = {q: chat_turn(memory, q) for q, _ in QUESTIONS}
    hits = sum(key in answers[q].lower() for q, key in QUESTIONS)
    print(
        f"{label:<22} recall {hits}/{len(QUESTIONS)} | max prompt tokens {max(sizes):>4} "
        f"(budget {memory.budget_tokens}) | compress calls {memory.compress_calls}"
    )
    for q, a in answers.items():
        print(f"    {q:<36} -> {a[:70]}")
    return {"hits": hits, "max_tokens": max(sizes), "sizes": sizes}


# ----------------------------------------------------------------- offline model


def offline_rules():
    def compress(prompt: str) -> str:
        facts = re.findall(r"^- (.+)$", prompt.split("<facts>")[1].split("</facts>")[0], re.M)
        new = prompt.split("<new_messages>")[1]
        if m := re.search(r"my name is (\w+)", new, re.I):
            facts.append(f"The user's name is {m.group(1)}")
        if m := re.search(r"allergic to (\w+)", new, re.I):
            facts.append(f"The user is allergic to {m.group(1)}")
        if m := re.search(r"trip to (\w+)", new, re.I):
            facts.append(f"The user is planning a trip to {m.group(1)}")
        return json.dumps(
            {
                "summary": "Earlier the user chatted about everyday topics.",
                "facts": list(dict.fromkeys(facts)),
            }
        )

    def chat(prompt: str) -> str:
        """Answers recall questions using ONLY the text of the prompt (like a real model must)."""
        last = prompt.strip().splitlines()[-1].lower()
        if "my name" in last:
            m = re.search(r"name is (\w+)", prompt, re.I)
            return f"Your name is {m.group(1)}." if m else "Sorry, I don't know your name."
        if "allergic" in last:
            m = re.search(r"allergic to (\w+)", prompt, re.I)
            return f"You're allergic to {m.group(1)}." if m else "I don't know."
        if "city" in last:
            m = re.search(r"trip to (\w+)", prompt, re.I)
            return f"You're going to {m.group(1)}." if m else "I'm not sure where you're going."
        return "Sure, here is a short answer about that. " * 3

    return [(r"You maintain the memory of a chat assistant", compress), (r"(?s).*", chat)]


def run_offline() -> None:
    print("*** OFFLINE: scripted model that answers strictly from what is in its prompt ***\n")
    with fake_llm(offline_rules()):
        naive = TieredMemory(budget_tokens=10**9, keep_last=10**6, use_summary=False)
        window_only = TieredMemory(budget_tokens=250, use_summary=False)
        tiered = TieredMemory(budget_tokens=250)
        r_naive = run_scenario(naive, label="naive (full history)")
        r_window = run_scenario(window_only, label="sliding window only")
        r_tiered = run_scenario(tiered, label="tiered memory")
    assert r_naive["hits"] == 3, "full history remembers everything (at a growing price)"
    assert r_window["hits"] == 0, "a plain window must have forgotten the opening message"
    assert r_tiered["hits"] == 3, "facts tier must keep name/allergy/destination"
    assert r_tiered["max_tokens"] <= 250, "tiered prompt must respect the budget on every turn"
    assert (
        r_naive["sizes"][-1] > 4 * r_tiered["max_tokens"] // 3
        and r_naive["sizes"][-1] > r_naive["sizes"][0]
    )
    assert tiered.compress_calls >= 5
    print(
        f"\nnaive prompt grew {r_naive['sizes'][0]} -> {r_naive['sizes'][-1]} tokens; "
        f"tiered stayed <= {r_tiered['max_tokens']}"
    )
    print("self-test passed")


def main() -> None:
    if "--offline" in sys.argv:
        return run_offline()
    print(f"provider: {llm.resolve()}")
    run_scenario(TieredMemory(budget_tokens=250, use_summary=False), label="sliding window only")
    run_scenario(TieredMemory(budget_tokens=250), label="tiered memory")


if __name__ == "__main__":
    main()
