"""Tests for the capstone report: targets are computed from the results (never typed in), the documents render every section, and the portfolio README is the template filled in."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import capstone_report as CR

HERE = Path(__file__).resolve().parent


def rep(p=0.7):
    r = (p, p - 0.2, min(1.0, p + 0.15))
    return {
        "overall": r,
        "single": r,
        "multi": (0.75, 0.3, 0.95),
        "out_of_scope": (1.0, 0.57, 1.0),
        "adversarial": (1.0, 0.34, 1.0),
    }


def results():
    return {
        "chunks": 1219,
        "dev": {"overall": (0.82, 0.66, 0.92), "n": 34},
        "test": {"overall": (0.76, 0.59, 0.87), "n": 33, "answerable": (0.69, 0.50, 0.83)},
        "all": {"out_of_scope": (1.0, 0.72, 1.0), "retrieval_hit5": (0.98, 0.90, 1.0)},
        "systems": {
            "extractive (shipped)": rep(0.76),
            "model + verification + fallback": rep(0.64),
        },
        "floors": {"abstain": 0.21, "dump": 0.55},
        "retrieval": {
            "BM25 only": {"hit5": (0.96, 0.87, 0.99), "both": 4, "multi": 8},
            "dense only": {"hit5": (0.96, 0.87, 0.99), "both": 4, "multi": 8},
            "hybrid": {"hit5": (0.98, 0.90, 1.0), "both": 7, "multi": 8},
            "hybrid + week scope": {"hit5": (0.98, 0.90, 1.0), "both": 8, "multi": 8},
        },
        "gate": {"auc_cosine": 0.962, "auc_rerank": 0.996, "threshold": -1.94},
        "demo": [
            {
                "label": "a plain fact",
                "question": "What is HNSW?",
                "answer": "HNSW is a graph index [1].",
                "mode": "extractive",
                "abstained": False,
                "blocked": False,
                "sources": ["day2_vector-databases.md > 2. HNSW"],
            },
            {
                "label": "a refusal",
                "question": "How do I file a tax return?",
                "answer": "I don't know based on the provided sources.",
                "mode": "gate",
                "abstained": True,
                "blocked": False,
                "sources": [],
            },
        ],
        "security": {
            "direct_none": 13,
            "direct_input": 5,
            "direct_leaks": 0,
            "overblocked": 0,
            "legit": 58,
            "poison_through": 4,
            "poison_table": {
                "extractive answerer": {"no defences": 5, "quarantine + output guard": 4}
            },
        },
        "gate_prs": {
            "refactor: no behaviour change": {
                "passed": True,
                "why": "0 newly pass, 0 newly fail",
                "refactor": True,
            },
            "tune: a": {"passed": False, "why": "overall below floor", "refactor": False},
            "tune: b": {"passed": False, "why": "net regression", "refactor": False},
            "tune: c": {"passed": False, "why": "retrieval", "refactor": False},
            "cleanup: d": {"passed": False, "why": "attacks", "refactor": False},
        },
        "gate_refactor_passes": True,
        "latency": {
            "p50": 0.388,
            "p95": 0.478,
            "stages_ms": {
                "retrieve": {"p50": 17.0, "p95": 23.0},
                "gate": {"p50": 370.0, "p95": 458.0},
                "TOTAL": {"p50": 388.0, "p95": 478.0},
            },
        },
        "cost": {"seconds": 0.415, "per_1000": 0.0231, "llm_prompt_tokens": 1152},
        "tests": 140,
        "load_closed": [{"users": 1, "rps": 2.57, "lat50": 0.39, "lat95": 0.48, "errors": 0}],
    }


def test_each_target_is_computed_from_the_results_and_the_unmet_one_is_marked():
    t = {x["id"]: x for x in CR.targets(results())}
    assert [k for k, v in t.items() if not v["met"]] == [
        "R2"
    ]  # 69% < 80%: the one target this run missed
    assert (
        t["R1"]["outcome"].startswith("98%")
        and t["R5"]["outcome"] == "478 ms"
        and t["R6"]["outcome"] == "$0.023"
    )
    assert "4 of 10" in t["R4"]["outcome"] and t["R8"]["outcome"].startswith("fails on 4 of 4")


def test_changing_a_result_changes_the_verdict():
    r = copy.deepcopy(results())
    r["test"]["answerable"] = (0.85, 0.7, 0.93)
    assert all(x["met"] for x in CR.targets(r))
    r["security"]["direct_leaks"] = 1
    assert [x["id"] for x in CR.targets(r) if not x["met"]] == ["R4"]
    r = copy.deepcopy(results())
    r["latency"]["p95"] = 3.0
    r["gate_prs"]["tune: a"]["passed"] = True  # the gate no longer fails on 4 bad changes
    assert {x["id"] for x in CR.targets(r) if not x["met"]} == {"R2", "R5", "R8"}
    r["tests"] = 0
    assert "R7" in {x["id"] for x in CR.targets(r) if not x["met"]}


def test_the_report_has_every_section_the_numbers_and_the_not_run_list():
    text = CR.render_report(results())
    for heading in (
        "## 1. Targets against outcomes (7 of 8 met)",
        "## 2. Quality",
        "## 3. Retrieval and the gate",
        "## 4. Security",
        "## 5. The CI gate",
        "## 6. Latency, cost, load",
        "## Not run",
        "## Limits",
    ):
        assert heading in text, heading
    assert "| R2 |" in text and "**NO**" in text and "1,219 chunks" in text
    assert "always abstain **21%**" in text and "dump the top BM25 chunk **55%**" in text
    assert "| refactor: no behaviour change | PASS |" in text and "| tune: a | FAIL |" in text
    assert "| 1 | 2.57 | 0.39 s | 0.48 s | 0 |" in text and "$0.023" in text
    for item in CR.NOT_RUN:
        assert item in text
    assert "{" not in text and not re.search(r"\bnan\b", text.lower())


def test_the_report_says_so_when_no_load_test_ran():
    r = results()
    del r["load_closed"]
    assert "The load test was not run in this invocation" in CR.render_report(r)


def test_the_portfolio_readme_is_the_template_filled_in_without_inventing_a_demo():
    template = (HERE.parent / "templates" / "portfolio_readme_template.md").read_text()
    text = CR.render_portfolio(results(), template)
    assert (
        text.startswith("# Course Copilot") and "<Project name>" not in text and "<URL" not in text
    )
    assert "not deployed (no cloud account was used)" in text
    assert (
        "69% [50%, 83%] of answerable questions; overall 76% [59%, 87%]" in text
        and "478 ms" in text
        and "$0.023 machine time" in text
    )
    assert (
        "was **not met**" in text and "69% [50%, 83%]" in text
    )  # the unmet target is stated where an interviewer will read it
    for section in (
        "## What it does",
        "## Architecture",
        "## Decisions you can argue with",
        "## Run it",
        "## Honest limits",
    ):
        assert section in text  # the human-written parts of the template survive untouched
