"""Terminal front-end:  python -m llm_cli [--provider anthropic] [--model ...] [--system ...]

Run from this directory (solutions/weekly) or add it to PYTHONPATH.
"""

from __future__ import annotations

import argparse
import os
import sys

# The CLI owns retries so it can show them to the user, so turn off the SDK's hidden ones.
# (Must be set before the first client is created.)
os.environ.setdefault("LLM_MAX_RETRIES", "0")

try:  # arrow keys / history in input() on macOS and Linux
    import readline  # noqa: F401
except ImportError:
    pass

from . import commands
from .chat import ChatSession, SendResult


def footer(session: ChatSession, r: SendResult) -> str:
    bits = [f"{session.provider}/{session.model}"]
    if r.response:
        u = r.response.usage
        bits.append(
            f"in {u.input_tokens:,}"
            + (f" (+{u.cache_read_tokens:,} cached)" if u.cache_read_tokens else "")
        )
        bits.append(f"out {u.output_tokens:,}")
        bits.append(f"${r.response.cost_usd:.5f}")
    if r.ttft_s is not None:
        bits.append(f"first token {r.ttft_s:.2f}s")
    bits.append(f"session ${session.meter.cost_usd:.4f}")
    return "[" + " · ".join(bits) + "]"


def run_turn(session: ChatSession, text: str) -> None:
    def on_retry(n: int, delay: float, exc: Exception) -> None:
        print(f"\n  ! {type(exc).__name__}; retry {n} in {delay:.1f}s ...", file=sys.stderr)

    r = session.send(text, on_chunk=lambda c: print(c, end="", flush=True), on_retry=on_retry)
    print()
    if r.dropped_messages:
        print(f"  (context budget: dropped {r.dropped_messages} oldest message(s))")
    if r.over_budget:
        print("  ! your newest message alone exceeds the context budget")
    if r.interrupted:
        print("  ^C - generation stopped (partial answer kept; its cost isn't tracked)")
    elif r.error:
        print(f"  ! {type(r.error).__name__}: {r.error}\n  (use /retry to try again)")
    else:
        print(footer(session, r))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="llm-cli", description="Terminal chat with any LLM provider")
    ap.add_argument("--provider", choices=["anthropic", "openai", "ollama"])
    ap.add_argument("--model")
    ap.add_argument("--system", default="You are a concise, helpful assistant.")
    ap.add_argument("--budget", type=int, default=8000, help="context token budget")
    ap.add_argument("--load", help="resume from a saved transcript")
    args = ap.parse_args(argv)

    session = ChatSession(args.provider, args.model, args.system, args.budget)
    if args.load:
        session.load(args.load)
    print(f"llm-cli · {session.provider}/{session.model} · /help for commands · Ctrl-D to exit")

    while True:
        try:
            line = input("\nyou> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\n(use /quit or Ctrl-D to exit)")
            continue
        if not line:
            continue
        if line.startswith("/"):
            res = commands.handle(session, line)
            if res.output:
                print(res.output)
            if res.quit:
                return 0
            if res.resend:
                run_turn(session, res.resend)
            continue
        run_turn(session, line)


if __name__ == "__main__":
    raise SystemExit(main())
