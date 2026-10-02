"""Week 5 shortcut: ``load('day2_solution')`` -> the module registered as ``w5_day2_solution``."""

from __future__ import annotations

from _weeks import load as _load


def load(name: str):
    return _load("week05_tool-use-and-agents", name)
