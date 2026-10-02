"""Load solution modules from OTHER weeks under unique names (every week has day1_solution.py ... day7).

    from _weeks import load
    w2 = load("week02_prompting-and-structured-outputs", "day4_solution")     # -> module "w2_day4_solution"

While a module loads, this week's same-named modules are hidden from ``sys.modules`` (the other week's files import
each other by plain name), then everything is restored.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

WEEKS = Path(__file__).resolve().parents[2]
_PLAIN = re.compile(r"(test_)?day\d_solution|test_day\d")


def load(week_dir: str, name: str):
    folder = WEEKS / week_dir / "solutions"
    key = f"w{week_dir[4:6].lstrip('0') or '0'}_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, folder / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    hidden = {k: sys.modules.pop(k) for k in list(sys.modules) if _PLAIN.fullmatch(k)}
    saved_path = list(sys.path)
    sys.path.insert(0, str(folder))
    try:
        spec.loader.exec_module(mod)
    finally:
        # the loaded module may insert ITS OWN directory at sys.path[0] (many solution files do); left in place it would
        # shadow the caller's modules of the same name (day1_solution!), so restore the path exactly
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if _PLAIN.fullmatch(k)]:
            del sys.modules[k]
        sys.modules.update(hidden)
    return mod
