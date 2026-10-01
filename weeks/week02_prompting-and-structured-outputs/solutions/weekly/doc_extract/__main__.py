"""python -m doc_extract [--pdf-dir DIR] [--out report.md] [--offline]   (run from solutions/weekly)"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from common import llm
from common.fake import fake_llm

from . import samples
from .offline_model import make_rules
from .pipeline import process_folder
from .report import build_report

OUT = Path(__file__).resolve().parents[5] / "outputs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", type=Path, default=OUT / "week2_docs")
    ap.add_argument("--out", type=Path, default=OUT / "week2_report.md")
    ap.add_argument(
        "--offline", action="store_true", help="scripted fake model (tests the pipeline)"
    )
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    gold = samples.generate(args.pdf_dir)
    print(f"generated {len(gold)} sample PDFs in {args.pdf_dir}")
    if args.offline:
        print(
            "*** OFFLINE: scripted model with injected faults - tests the pipeline, not a real model ***"
        )
        with fake_llm(make_rules(gold)):
            records = asyncio.run(process_folder(args.pdf_dir, args.concurrency))
    else:
        print(f"provider: {llm.resolve()}")
        records = asyncio.run(process_folder(args.pdf_dir, args.concurrency))
    text, metrics = build_report(records, gold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    print(f"\nreport written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
