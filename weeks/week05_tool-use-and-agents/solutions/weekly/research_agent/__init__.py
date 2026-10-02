"""research_agent: search + fetch + notes (MCP) -> a cited markdown report whose citations are machine-checked."""

import sys
from pathlib import Path

_ROOT = str(
    Path(__file__).resolve().parents[5]
)  # the repo root, so `import common` works from anywhere
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
