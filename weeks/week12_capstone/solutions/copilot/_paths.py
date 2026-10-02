"""Where the shared code lives. In the repository the package sits at ``weeks/week12_capstone/solutions/copilot`` and ``common/`` is at the repository root; in the container both are siblings under
``/app``. ``ROOT`` is the nearest ancestor that contains ``common/``, so the same imports work in both places."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p / "common").is_dir()), HERE.parent)
WEEK11 = ROOT / "weeks" / "week11_inference-serving-deployment" / "solutions"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# APPENDED, not inserted: the Week 11 solutions directory has modules named like this week's (day4_solution.py, ...) and must never shadow them; only ``llmapi`` is wanted from it
if WEEK11.is_dir() and str(WEEK11) not in sys.path:
    sys.path.append(str(WEEK11))
