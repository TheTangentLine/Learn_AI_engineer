"""Slash commands. ``handle`` returns text for the UI to print; it never touches the terminal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chat import ChatSession

HELP = """\
/help                    show this help
/provider <name> [model] switch provider (anthropic | openai | ollama), history is kept
/model <name>            change model for the current provider
/system [text]           show or set the system prompt
/budget <tokens>         set the context budget (older messages are trimmed to fit)
/history                 list the conversation
/undo                    remove the last exchange
/retry                   regenerate the last answer
/clear                   forget the conversation
/cost                    session cost meter
/save [file]  /load file save or load a transcript (JSON)
/quit                    exit (Ctrl-D also works)"""

DEFAULT_SAVE = "outputs/llm_cli_chat.json"


@dataclass
class CommandResult:
    output: str = ""
    quit: bool = False
    resend: str | None = None  # text to send again (for /retry)


def handle(session: ChatSession, line: str) -> CommandResult:
    name, _, arg = line.strip().partition(" ")
    arg = arg.strip()
    name = name.lower()

    if name in ("/quit", "/exit", "/q"):
        return CommandResult(quit=True)
    if name == "/help":
        return CommandResult(HELP)
    if name == "/provider":
        if not arg:
            return CommandResult(f"current: {session.provider}/{session.model}")
        provider, *rest = arg.split()
        try:
            session.set_provider(provider, rest[0] if rest else None)
        except ValueError as e:
            return CommandResult(str(e))
        return CommandResult(
            f"now using {session.provider}/{session.model} "
            f"({len(session.messages)} messages carried over)"
        )
    if name == "/model":
        if not arg:
            return CommandResult(f"current: {session.provider}/{session.model}")
        session.model = arg
        return CommandResult(f"model set to {session.provider}/{session.model}")
    if name == "/system":
        if arg:
            session.system = arg
            return CommandResult("system prompt set")
        return CommandResult(f"system prompt: {session.system!r}")
    if name == "/budget":
        if not arg.isdigit() or int(arg) < 100:
            return CommandResult("usage: /budget <tokens>  (at least 100)")
        session.max_context_tokens = int(arg)
        return CommandResult(
            f"context budget = {arg} tokens (now using {session.context_tokens()})"
        )
    if name == "/history":
        if not session.messages:
            return CommandResult("(empty)")
        lines = []
        for i, m in enumerate(session.messages):
            text = m["content"].replace("\n", " ")
            flag = " [partial]" if m.get("partial") else ""
            lines.append(
                f"{i:>2} {m['role']:<9} {text[:90]}{'...' if len(text) > 90 else ''}{flag}"
            )
        return CommandResult("\n".join(lines))
    if name == "/undo":
        text = session.pop_last_exchange()
        return CommandResult("removed last exchange" if text else "nothing to undo")
    if name == "/retry":
        text = session.pop_last_exchange()
        return CommandResult("regenerating..." if text else "nothing to retry", resend=text)
    if name == "/clear":
        session.clear()
        return CommandResult("conversation cleared")
    if name == "/cost":
        m = session.meter
        u = m.usage
        rows = [
            f"calls: {m.calls}",
            f"tokens: in={u.input_tokens:,} cached_read={u.cache_read_tokens:,} out={u.output_tokens:,}",
            f"total cost: ${m.cost_usd:.5f}",
            f"context now: ~{session.context_tokens():,} / {session.max_context_tokens:,} tokens",
        ]
        rows += [f"  {k}: ${v:.5f}" for k, v in sorted(m.by_model.items())]
        return CommandResult("\n".join(rows))
    if name == "/save":
        path = session.save(arg or DEFAULT_SAVE)
        return CommandResult(f"saved to {path}")
    if name == "/load":
        if not arg or not Path(arg).exists():
            return CommandResult("usage: /load <file>  (file not found)")
        session.load(arg)
        return CommandResult(
            f"loaded {len(session.messages)} messages; now {session.provider}/{session.model}"
        )
    return CommandResult(f"unknown command {name!r}; try /help")
