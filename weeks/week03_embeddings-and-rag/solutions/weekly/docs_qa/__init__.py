"""docs_qa: a Q&A bot over this course's lessons (Week 3 weekly challenge, reference solution)."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
