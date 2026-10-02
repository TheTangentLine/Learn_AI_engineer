"""CLI:  python -m support_system chat [--provider local|anthropic|openai] [--db support.sqlite]
        python -m support_system eval [--provider ...] [--trials N]
Run from weeks/week06_frameworks-and-multi-agent/solutions/weekly."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from .evalset import agent_factory, build_scenarios, make_world_factory
from .system import SupportSystem


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="support_system")
    ap.add_argument("mode", choices=["chat", "eval"])
    ap.add_argument(
        "--provider",
        default=None,
        help="anthropic | openai | ollama (an OpenAI-compatible endpoint) ",
    )
    ap.add_argument("--model", default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--no-guards", action="store_true")
    ap.add_argument(
        "--local",
        action="store_true",
        help="serve the local Qwen model on an OpenAI-compatible port and use it",
    )
    args = ap.parse_args(argv)
    server = None
    if args.local:
        from common import llm
        from common.local_server import LocalOpenAIServer

        server = LocalOpenAIServer().__enter__()
        os.environ["OLLAMA_BASE_URL"] = server.url
        llm._ollama.cache_clear()
        args.provider, args.model = "ollama", "local-qwen"
    try:
        if args.mode == "eval":
            from common.agent_eval import run_eval, summarize

            results = run_eval(
                build_scenarios(
                    make_world_factory(args.provider, args.model, guards=not args.no_guards)
                ),
                agent_factory,
                trials=args.trials,
                max_turns=6,
            )
            for r in results:
                print(
                    f"{'PASS' if r.passed else 'FAIL'}  {r.scenario_id:28s} {', '.join(r.failures)}"
                )
            print()
            print(summarize(results, ks=(args.trials,) if args.trials > 1 else ()))
            return 0
        db = args.db or str(Path(tempfile.mkdtemp()) / "support.sqlite")
        system = SupportSystem(
            db, provider=args.provider, model=args.model, guards=not args.no_guards
        )
        print(f"Support chat (db: {db}). Type 'quit' to exit, '/pending' to list approvals.")
        while True:
            try:
                line = input("you> ").strip()
            except EOFError:
                break
            if line in ("quit", "exit"):
                break
            if line == "/pending":
                print(system.pending_approvals())
                continue
            r = system.handle("cli", line)
            print(
                f"[{r.agent}] {r.text}"
                + (f"   (guard: {', '.join(r.violations)})" if r.violations else "")
            )
        return 0
    finally:
        if server:
            server.__exit__(None, None, None)


if __name__ == "__main__":
    sys.exit(main())
