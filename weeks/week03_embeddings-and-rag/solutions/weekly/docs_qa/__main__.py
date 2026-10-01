"""python -m docs_qa {index,ask,chat,eval} [--offline] [--rerank]    (run from solutions/weekly)"""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

from common import llm
from common.corpus import ROOT
from common.embed import get_embedder
from common.fake import fake_llm

from . import golden
from .app import build_bot, build_index
from .evaluate import evaluate
from .offline import make_rules


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="docs_qa")
    ap.add_argument("command", choices=["index", "ask", "chat", "eval"])
    ap.add_argument("question", nargs="*", help="for `ask`")
    ap.add_argument(
        "--offline", action="store_true", help="scripted extractive reader, no API keys"
    )
    ap.add_argument("--rerank", action="store_true", help="cross-encoder re-ranking (slower)")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "docs_qa_eval.md")
    args = ap.parse_args(argv)

    emb = get_embedder()
    index, report = build_index(emb)
    print(f"index: {report}")
    if args.command == "index":
        return 0
    reranker = None
    if args.rerank:
        from common.rerank import get_reranker

        reranker = get_reranker()
    answerable = [q for q, _, _ in golden.ANSWERABLE]
    bot = build_bot(index, answerable, golden.UNANSWERABLE, reranker=reranker, k=args.k)
    print(f"gate calibrated: tau = {bot.tau:.3f}")

    ctx = fake_llm(make_rules(emb)) if args.offline else nullcontext()
    if args.offline:
        print("*** OFFLINE: scripted extractive reader; retrieval and gating are real ***")
    else:
        print(f"provider: {llm.resolve()}")
    with ctx:
        if args.command == "ask":
            a = bot.ask(" ".join(args.question))
            print(f"\n{a.text}")
            for i, h in enumerate(a.cited, 1):
                print(f"  [{i}] {h.metadata['doc']} › {h.metadata['heading']}")
            if a.abstained:
                print(f"  ({a.reason})")
        elif args.command == "chat":
            history: list[tuple[str, str]] = []
            print("Ask about the course. Ctrl-D to exit.")
            while True:
                try:
                    q = input("\nyou> ").strip()
                except EOFError:
                    print()
                    return 0
                if q:
                    a = bot.ask(q, history)
                    history.append((q, a.text))
                    print(a.text)
                    if a.standalone != q:
                        print(f"  (searched for: {a.standalone})")
        else:
            text, m = evaluate(bot, golden.ANSWERABLE, golden.UNANSWERABLE, golden.FOLLOWUPS)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
            print("\n" + text)
            print(f"\nreport written to {args.out}")
            if m["bad_gold"]:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
