"""Assemble the Docker build context for the Copilot service: the product package, the shared ``common/`` modules, the Week 11 gateway, the requirements and the Dockerfile.

python build_context.py OUT_DIR
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLUTIONS = HERE.parent
ROOT = SOLUTIONS.parents[2]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def build(out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(SOLUTIONS / "copilot", out / "copilot", ignore=IGNORE)
    shutil.copytree(ROOT / "common", out / "common", ignore=IGNORE)
    shutil.copytree(
        ROOT / "weeks/week11_inference-serving-deployment/solutions/llmapi",
        out / "llmapi",
        ignore=IGNORE,
    )
    for name in ("Dockerfile", "requirements-copilot.txt", ".dockerignore"):
        shutil.copy(HERE / name, out / name)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    return {"out": str(out), "bytes": size, "files": sum(1 for f in out.rglob("*") if f.is_file())}


if __name__ == "__main__":
    print(build(Path(sys.argv[1])))
