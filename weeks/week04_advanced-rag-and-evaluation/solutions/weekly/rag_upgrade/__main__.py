"""python -m rag_upgrade [--llm] [--out report.md]    (run from solutions/weekly; needs the Day 1 golden set)"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.embed import get_embedder
from common.evalkit import load_golden
from common.rag import RagIndex
from common.rerank import get_reranker

from .configs import build_configs
from .harness import evaluate_config, render_report, select_best, split_golden

ROOT = Path(__file__).resolve().parents[5]
PINNED_WEEKS = (1, 2, 3)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rag_upgrade")
    ap.add_argument("--golden", type=Path, default=ROOT / "outputs" / "golden_week4.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "week4_report.md")
    ap.add_argument(
        "--llm",
        action="store_true",
        help="include LLM-generated variants (uses the local model cache)",
    )
    ap.add_argument("--max-context-ratio", type=float, default=2.5)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(ROOT / "weeks/week03_embeddings-and-rag/solutions/weekly"))
    from docs_qa.app import course_sources

    emb = get_embedder()
    index = RagIndex(emb, ROOT / "outputs" / "week4_index")
    index.sync([d for d in course_sources() if d.meta["week"] in PINNED_WEEKS])
    golden = load_golden(args.golden)
    dev_q, test_q = split_golden(golden)
    print(f"{len(index.chunks)} chunks | {len(dev_q)} dev + {len(test_q)} test questions")

    chat = None
    if args.llm:
        from common.local_llm import LocalChat

        chat = LocalChat()
    configs = build_configs(index, emb, get_reranker(), chat)
    baseline = configs[0].name
    dev = {c.name: evaluate_config(c, dev_q) for c in configs}
    winner = select_best(dev, baseline, args.max_context_ratio)
    test = {c.name: evaluate_config(c, test_q) for c in configs}
    text, summary = render_report(
        dev, test, baseline, winner, len(dev_q), len(test_q), args.max_context_ratio
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    print(f"\nreport written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
