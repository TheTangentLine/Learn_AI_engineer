"""Assemble the Docker build context: the API package, a markdown corpus to retrieve from, the requirements and the Dockerfile, in one directory.

python build_context.py OUT_DIR [--docs DIR]       # default docs: this repository's own lessons (weeks/*/day*.md)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def build(out: Path, docs: Path | None = None, limit_weeks: int | None = None) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(
        HERE.parent / "llmapi",
        out / "llmapi",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    n = 0
    if docs is None:
        for p in sorted((ROOT / "weeks").glob("week*/day*.md")):
            if limit_weeks is None or int(p.parent.name[4:6]) <= limit_weeks:
                dst = out / "docs" / p.parent.name / p.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(p, dst)
                n += 1
    else:
        shutil.copytree(docs, out / "docs")
        n = sum(1 for _ in (out / "docs").rglob("*.md"))
    for name in ("Dockerfile", "requirements-api.txt", ".dockerignore"):
        shutil.copy(HERE / name, out / name)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    return {"out": str(out), "docs": n, "bytes": size}


if __name__ == "__main__":
    args = sys.argv[1:]
    docs = Path(args[args.index("--docs") + 1]) if "--docs" in args else None
    print(build(Path(args[0]), docs))
