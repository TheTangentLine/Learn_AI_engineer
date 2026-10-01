"""doc_extract: PDF -> classify -> typed JSON -> validate -> accuracy report (Week 2 weekly challenge)."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
