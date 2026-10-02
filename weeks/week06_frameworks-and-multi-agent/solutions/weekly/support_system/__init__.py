"""support_system: triage -> billing/tech specialists, a human-approval step for refunds, runtime guards, and an eval suite."""

import sys
from pathlib import Path

_ROOT = str(
    Path(__file__).resolve().parents[5]
)  # the repo root, so `import common` works from anywhere
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
_SOL = str(Path(__file__).resolve().parents[2])  # this week's solutions dir (for _weeks.load)
if _SOL not in sys.path:
    sys.path.append(
        _SOL
    )  # APPEND: inserting at the front would shadow the caller's own day1_solution.py etc.
