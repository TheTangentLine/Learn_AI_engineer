"""llm-cli: a terminal chat client (Week 1 weekly challenge, reference solution)."""

import sys
from pathlib import Path

# Make `common` importable even if the repo was not installed with `pip install -e .`
_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
