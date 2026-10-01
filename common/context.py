"""Context engineering for long-running agents: keep the window small without losing what matters.

Every strategy is a *context hook*: ``hook(messages, run) -> messages`` (see ``common/agent.py``), applied just
before each model call. A hook returns a NEW list (it never edits messages in place) and the agent keeps that
list as its working history, so a compaction is paid for once, not on every step. ``run.transcript`` keeps every
message the run ever produced, untrimmed, for debugging.

    from common.context import chain, clear_old_tool_results, truncate_tool_results, compact_history
    hook = chain(truncate_tool_results(2000), clear_old_tool_results(keep_last=4),
                 compact_history(summarizer, trigger_tokens=6000))
    run = run_agent(task, registry, context_hook=hook)

Strategies, cheapest first:
  truncate_tool_results   cap each result (the head is kept, with a visible note)
  clear_old_tool_results  replace the content of OLD tool results with a placeholder (the call stays visible)
  inject_notes            a scratchpad kept OUTSIDE the history and shown at the top of every request
  compact_history         summarise the oldest exchanges into the first message when over a token trigger

The invariant every hook must keep (tested; violating it makes providers reject the request):
every tool result follows the assistant message that made the call, and every call has a result.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

Message = dict[str, Any]
Hook = Callable[[list[Message], Any], list[Message]]
CLEARED = "[result cleared to save context; call the tool again if you still need it]"
NOTES_HEADER = "[Your notes so far]"
SUMMARY_HEADER = "[Summary of earlier work]"


# ----------------------------------------------------------------------------- measuring


def message_text(m: Message) -> str:
    text = m.get("content") or ""
    for c in m.get("tool_calls", []):
        text += f"\n{c['name']}({json.dumps(c['args'], ensure_ascii=False)})"
    return text


def approx_tokens(text: str) -> int:
    """~4 characters per token for English prose. Fine for budgeting; use a real tokenizer to measure."""
    return -(-len(text) // 4)


def count_tokens(messages: list[Message], counter: Callable[[str], int] = approx_tokens) -> int:
    return sum(counter(message_text(m)) + 4 for m in messages)  # +4: per-message framing overhead


# ----------------------------------------------------------------------------- structure helpers


def exchanges(messages: list[Message]) -> list[list[Message]]:
    """Split a history into units that must never be separated: [user] or [assistant + its tool results]."""
    units: list[list[Message]] = []
    for m in messages:
        if m["role"] == "tool" and units and units[-1][0]["role"] == "assistant":
            units[-1].append(m)
        else:
            units.append([m])
    return units


def check_invariants(messages: list[Message]) -> list[str]:
    """Problems that make a provider reject the conversation (empty list = fine)."""
    problems = []
    pending: set[str] = set()
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            if pending:
                problems.append(f"message {i}: tool calls {sorted(pending)} never got results")
            pending = {c["id"] for c in m.get("tool_calls", [])}
        elif m["role"] == "tool":
            if m["tool_call_id"] not in pending:
                problems.append(
                    f"message {i}: tool result {m['tool_call_id']!r} has no matching call before it"
                )
            pending.discard(m["tool_call_id"])
        elif pending:
            problems.append(
                f"message {i}: a {m['role']} message arrived while calls {sorted(pending)} were unanswered"
            )
            pending = set()
    if pending:
        problems.append(f"tool calls {sorted(pending)} never got results")
    return problems


# ----------------------------------------------------------------------------- the first message
# The first user message carries the task and, optionally, a summary block and a notes block, in that order:
#   <task>\n\n[Summary of earlier work]\n<summary>\n\n[Your notes so far]\n- k: v ...


def split_first(body: str) -> tuple[str, str, str]:
    """(task, summary, notes) of the first message's text; missing parts are ''."""
    notes = summary = ""
    if "\n\n" + NOTES_HEADER in body:
        body, notes = body.split("\n\n" + NOTES_HEADER, 1)
        notes = NOTES_HEADER + notes
    if "\n\n" + SUMMARY_HEADER in body:
        body, summary = body.split("\n\n" + SUMMARY_HEADER, 1)
    return body, summary.strip(), notes.strip()


def join_first(task: str, summary: str = "", notes: str = "") -> str:
    out = task
    if summary:
        out += f"\n\n{SUMMARY_HEADER}\n{summary}"
    if notes:
        out += f"\n\n{notes}"
    return out


# ----------------------------------------------------------------------------- strategies


def truncate_tool_results(max_chars: int = 2000) -> Hook:
    def hook(messages: list[Message], run: Any = None) -> list[Message]:
        out = []
        for m in messages:
            if m["role"] == "tool" and len(m["content"]) > max_chars:
                extra = len(m["content"]) - max_chars
                m = {
                    **m,
                    "content": m["content"][:max_chars] + f"\n[truncated: {extra} more characters]",
                }
            out.append(m)
        return out

    return hook


def clear_old_tool_results(keep_last: int = 3, placeholder: str = CLEARED) -> Hook:
    """Keep the last ``keep_last`` tool results verbatim; replace the content of earlier ones.
    The assistant's calls stay, so the model still sees WHAT it did, just not the bulky output."""

    def hook(messages: list[Message], run: Any = None) -> list[Message]:
        idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
        old = set(idx[: max(0, len(idx) - keep_last)])
        return [
            {**m, "content": placeholder} if i in old and m["content"] != placeholder else m
            for i, m in enumerate(messages)
        ]

    return hook


class Scratchpad:
    """Notes the agent writes with a tool and that are re-shown at the top of every request."""

    def __init__(self, max_chars: int = 4000):
        self.notes: dict[str, str] = {}
        self.max_chars = max_chars

    def remember(self, key: str, value: str) -> str:
        previous = self.notes.get(key)
        trial = {**self.notes, key: value}
        if sum(len(k) + len(v) for k, v in trial.items()) > self.max_chars:
            raise ValueError(
                f"notes are full ({self.max_chars} characters). Shorten or delete notes with forget(key): "
                f"keys now: {', '.join(self.notes)}"
            )
        self.notes[key] = value
        return (
            f"Saved note {key!r}."
            if previous is None
            else f"Updated note {key!r} (was: {previous[:60]!r})."
        )

    def forget(self, key: str) -> str:
        if key not in self.notes:
            raise ValueError(f"no note {key!r}. Existing keys: {', '.join(self.notes) or 'none'}")
        del self.notes[key]
        return f"Deleted note {key!r}."

    def render(self) -> str:
        if not self.notes:
            return ""
        return NOTES_HEADER + "\n" + "\n".join(f"- {k}: {v}" for k, v in self.notes.items())

    def tools(self) -> list:
        from .tools import tool

        pad = self

        @tool
        def remember(key: str, value: str) -> str:
            """Save a short note you will need later (a number, an id, a decision). Notes survive context trimming;
            tool results do not. Saving the same key again replaces it.

            Args:
                key: Short label, e.g. "code_page_3".
                value: The fact itself, as short as possible.
            """
            return pad.remember(key, value)

        @tool
        def forget(key: str) -> str:
            """Delete a note you no longer need, to free space.

            Args:
                key: The label of the note to delete.
            """
            return pad.forget(key)

        return [remember, forget]


def inject_notes(pad: Scratchpad) -> Hook:
    """Show the notes at the top of every request (merged into the first user message, never a new message)."""

    def hook(messages: list[Message], run: Any = None) -> list[Message]:
        if not messages or messages[0]["role"] != "user":
            return messages
        task, summary, _old_notes = split_first(
            messages[0]["content"]
        )  # replace the old block, never stack
        first = {**messages[0], "content": join_first(task, summary, pad.render())}
        return [first, *messages[1:]]

    return hook


def compact_history(
    summarizer: Callable[[str], str],
    *,
    trigger_tokens: int,
    keep_last: int = 4,
    counter: Callable[[str], int] = approx_tokens,
    max_summary_chars: int = 3000,
) -> Hook:
    """When the history exceeds ``trigger_tokens``, summarise everything except the first message (the task)
    and the last ``keep_last`` exchanges into the first message. Repeated compactions fold the previous
    summary into the next one. The summarizer sees plain text and returns plain text."""

    def hook(messages: list[Message], run: Any = None) -> list[Message]:
        if count_tokens(messages, counter) <= trigger_tokens or not messages:
            return messages
        units = exchanges(messages[1:])
        if len(units) <= keep_last:
            return messages
        old, recent = units[: len(units) - keep_last], units[len(units) - keep_last :]
        transcript = "\n".join(
            f"{m['role'].upper()}: {message_text(m)}" for unit in old for m in unit
        )
        task, previous, _notes = split_first(messages[0]["content"])
        if previous:
            transcript = "PREVIOUS SUMMARY:\n" + previous + "\n\nNEW STEPS:\n" + transcript
        summary = summarizer(transcript).strip()[:max_summary_chars]
        merged = {**messages[0], "content": join_first(task, summary)}
        return [merged, *[m for unit in recent for m in unit]]

    return hook


def chain(*hooks: Hook) -> Hook:
    def hook(messages: list[Message], run: Any = None) -> list[Message]:
        for h in hooks:
            messages = h(messages, run)
        return messages

    return hook


# ----------------------------------------------------------------------------- summarizers


def llm_summarizer(
    provider: str | None = None, model: str | None = None, max_tokens: int = 400
) -> Callable[[str], str]:
    """Summarise with a model. The prompt names what must survive: facts, numbers, ids, decisions, open items."""
    from . import llm

    system = (
        "You compress an agent's working history. Write a short factual summary that preserves EVERY concrete "
        "fact the agent will need later: numbers, codes, ids, names, file paths, decisions taken, and what is "
        "still unfinished. Do not add anything that is not in the history."
    )

    def summarize(transcript: str) -> str:
        return llm.complete(
            transcript, system=system, provider=provider, model=model, max_tokens=max_tokens
        ).text

    return summarize


def drop_summarizer(transcript: str) -> str:
    """The honest baseline: no summary at all, only a note that work happened."""
    n = transcript.count("\nASSISTANT:") + transcript.startswith("ASSISTANT:")
    return (
        f"({n} earlier steps were removed from the context; their details are no longer available.)"
    )
