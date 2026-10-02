"""CLI:  python -m research_agent "your question" [--out report.md] [--provider local|anthropic|openai]
        python -m research_agent --eval                      # the 6-question evaluation

Run from weeks/week05_tool-use-and-agents/solutions/weekly (so the package is importable)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from common.corpus import load_course_docs  # noqa: E402

from .agent import research  # noqa: E402
from .evalset import QUESTIONS, score  # noqa: E402
from .web import LocalWeb  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research_agent")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--out", default=None, help="write the report here")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--no-notes", action="store_true", help="skip the MCP notes server")
    args = ap.parse_args(argv)
    if not args.eval and not args.question:
        ap.error("give a question or --eval")
    with LocalWeb(load_course_docs()) as web:
        if args.eval:
            passed = 0
            for q in QUESTIONS:
                r = research(
                    q.question,
                    web,
                    provider=args.provider,
                    max_steps=args.max_steps,
                    use_notes=not args.no_notes,
                )
                s = score(q, r)
                passed += s.passed
                print(
                    f"{'PASS' if s.passed else 'FAIL'}  {q.id:13s} steps={len(r.run.steps):2d} sources={len(r.sources)} status={r.run.status:9s} {s.notes}"
                )
            print(f"\npassed {passed}/{len(QUESTIONS)}")
            return 0
        r = research(
            args.question,
            web,
            provider=args.provider,
            max_steps=args.max_steps,
            use_notes=not args.no_notes,
        )
        print(r.run.trace(), file=sys.stderr)
        print(f"verification: {r.verification.summary()}", file=sys.stderr)
        if args.out:
            Path(args.out).write_text(r.report)
            print(f"wrote {args.out}", file=sys.stderr)
        else:
            print(r.report)
    return 0 if r.run.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
