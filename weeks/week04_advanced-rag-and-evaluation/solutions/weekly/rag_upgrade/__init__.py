"""rag_upgrade: decide which RAG upgrades are worth shipping, with evidence (Week 4 weekly challenge)."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[5]
for p in (_ROOT, _ROOT / "weeks/week04_advanced-rag-and-evaluation/solutions"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
