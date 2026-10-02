"""Tests for Week 7 Day 1: the dataset is solvable and leak-free, the diagnosis finds each root cause, the comparison is honest."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import day1_solution as d1  # noqa: E402
import evalcases as ec  # noqa: E402
import scripted  # noqa: E402
from diagnose import diagnose, pareto  # noqa: E402

from common.fake import fake_llm  # noqa: E402


@pytest.fixture(scope="module")
def cases():
    return ec.build_cases(lambda: None)


def run(cases_subset, **faults):
    with fake_llm(scripted.rules(**faults)):
        return d1.run_variant(cases_subset, "t", provider="anthropic", **{})


# ----------------------------------------------------------------------------- the dataset


def test_dataset_shape_split_and_determinism(cases):
    assert len(cases) == 50 and len({c.id for c in cases}) == 50
    assert Counter(c.split for c in cases) == {"dev": 30, "test": 20}
    kinds = {c.kind for c in cases}
    assert len(kinds) == 9
    for kind in kinds:
        splits = {c.split for c in cases if c.kind == kind}
        assert splits == {"dev", "test"}, f"{kind} must appear in BOTH splits (stratified)"
    again = ec.build_cases(lambda: None)
    assert [(c.id, c.split) for c in again] == [(c.id, c.split) for c in cases], (
        "the split is a function of the seed"
    )
    other = ec.build_cases(lambda: None, seed=2)
    assert [c.split for c in other] != [c.split for c in cases]
    assert Counter(c.route for c in cases) == {
        "billing": 28,
        "technical": 14,
        "human": 4,
        "other": 4,
    }


def test_every_invoice_a_case_mentions_exists_and_the_extra_invoices_are_registered(cases):
    for inv in ec.EXTRA_INVOICES:
        assert ec.d5.INVOICES[inv] == ec.EXTRA_INVOICES[inv]
    import re

    for c in cases:
        text = (
            c.scenario.user_factory().first_message()
            + " "
            + " ".join(r for _, r in c.scenario.user_factory().rules)
        )
        for inv in re.findall(r"INV-\d+", text):
            assert inv in ec.d5.INVOICES, (c.id, inv)
    assert ec.d5.INVOICES["INV-3005"]["amount"] == ec.d5.AUTO_LIMIT, (
        "one case sits exactly on the auto-approval boundary"
    )


def test_no_triage_example_appears_in_any_evaluation_case(cases):
    """Leakage guard: few-shot examples that overlap the evaluation set would inflate the 'improvement'."""
    import re

    from support_system.system import TRIAGE_EXAMPLES

    examples = re.findall(r'"([^"]+)"', TRIAGE_EXAMPLES)
    assert len(examples) == 5
    openings = [c.scenario.user_factory().first_message().lower() for c in cases]
    for ex in examples:
        assert not any(ex.lower() in o or o in ex.lower() for o in openings), ex
        words = set(re.findall(r"[a-z0-9']+", ex.lower())) - {
            "my",
            "the",
            "a",
            "to",
            "i",
            "is",
            "me",
            "you",
            "can",
            "since",
            "this",
            "after",
        }
        for o in openings:
            overlap = words & set(re.findall(r"[a-z0-9']+", o))
            assert len(overlap) <= 3, f"example {ex!r} shares too many words with case {o!r}"


def test_the_good_scripted_system_solves_all_fifty_cases(cases):
    """The dataset is SOLVABLE: a failing case then means the SYSTEM is wrong, not the case."""
    runs = run(cases)
    bad = [(r.case_id, r.failures) for r in runs if not r.passed]
    assert bad == [], bad


def test_the_knowledge_base_can_answer_every_how_to_case():
    from support_system.kb import KnowledgeBase

    kb = KnowledgeBase()
    for question, words in ec.HOWTO:
        top = kb.search(question)
        assert top, question
        text = " ".join(t + " " + b for _, t, b, _ in top[:1]).lower()
        assert any(w in text for w in words), (question, top[0][0])


# ----------------------------------------------------------------------------- each fault shows up as its own root cause


def only(cases, kinds):
    return [c for c in cases if c.kind in kinds]


@pytest.mark.parametrize(
    "fault,kinds,expected",
    [
        ("misroute_tech", ["tech-outage", "tech-howto"], "misroute"),
        ("skip_tools", ["billing-small", "tech-howto"], "skipped_tools"),
        ("narrate", ["billing-small", "tech-outage"], "narrated_instead_of_acting"),
        ("skip_lookup", ["billing-small"], "skipped_verification"),
        ("wrong_invoice", ["billing-small"], "invented_arguments"),
        ("refund_unpaid", ["billing-unpaid"], "acted_on_ineligible_input"),
        ("guess_invoice", ["billing-missing-id"], "did_not_ask_for_missing_info"),
        ("liar", ["billing-approval"], "hallucinated_success"),
        ("ignore_kb", ["tech-howto"], "ignored_tool_result"),
        ("loop", ["billing-small"], "specialist_stuck_or_failed"),
    ],
)
def test_each_injected_fault_is_diagnosed_as_its_root_cause(cases, fault, kinds, expected):
    subset = only(cases, kinds)
    kw = {"guards": False} if fault == "liar" else {}
    with fake_llm(scripted.rules(**{fault: True})):
        runs = d1.run_variant(subset, fault, provider="anthropic", **kw)
    failing = [r for r in runs if not r.passed]
    assert failing, f"{fault} should break something"
    assert Counter(r.cause for r in failing).most_common(1)[0][0] == expected, Counter(
        r.cause for r in failing
    )


def test_diagnose_prefers_the_earliest_failure_point():
    kw = dict(
        kind="tech-outage",
        expected_route="technical",
        calls=[],
        agent_text="I can help with billing and technical questions.",
    )
    assert (
        diagnose(["missing_call:service_status"], actual_route="other", **kw).cause == "misroute"
    ), "no calls because it never reached the specialist"
    assert (
        diagnose(["missing_call:service_status"], actual_route="technical", **kw).cause
        == "skipped_tools"
    )
    assert (
        diagnose(["missing_call:service_status"], actual_route="tech", **kw).cause
        == "skipped_tools"
    ), "'tech' and 'technical' are the same route"
    assert (
        diagnose(["agent_error"], actual_route="other", error="boom", **{**kw, "calls": []}).cause
        == "crash"
    )
    assert (
        diagnose(["did_not_terminate"], actual_route="technical", **kw).cause == "conversation_loop"
    )
    for text in (
        "Let me check that for you.",
        "I'll look that up now.",
        "First, I will search the knowledge base.",
    ):
        assert (
            diagnose(
                ["missing_call:service_status"],
                actual_route="technical",
                **{**kw, "agent_text": text},
            ).cause
            == "narrated_instead_of_acting"
        ), text
    assert (
        diagnose(
            ["missing_call:service_status"],
            actual_route="technical",
            **{**kw, "agent_text": "I can help with billing."},
        ).cause
        == "skipped_tools"
    ), "capability is not narration"
    narr = {**kw, "agent_text": "Let me check that for you."}
    assert (
        diagnose(["missing_call:service_status"], actual_route="technical", **narr).cause
        == "narrated_instead_of_acting"
    )
    with_calls = {**kw, "calls": ["lookup_invoice"]}
    assert (
        diagnose(["missing_call:lookup_invoice"], actual_route="technical", **with_calls).cause
        == "skipped_verification"
    )
    assert (
        diagnose(["check_failed:something new"], actual_route="technical", **with_calls).cause
        == "other"
    )
    assert (
        diagnose(["check_failed:something new"], actual_route="technical", **with_calls).evidence
        == "check_failed:something new"
    )
    assert (
        diagnose(
            ["forbidden_call:request_refund", "missing_call:lookup_invoice"],
            actual_route="billing",
            **{**with_calls, "expected_route": "billing"},
        ).cause
        == "acted_on_ineligible_input"
    )
    assert diagnose([], actual_route=None, **kw).cause == "other", (
        "a case with no route event cannot be called a misroute"
    )


def test_pareto_sorts_by_frequency_and_attributes_kinds():
    p = pareto(
        [("a", "x"), ("b", "x"), ("a", "y"), ("c", "x"), ("a", "z"), ("b", "y")], total_cases=20
    )
    assert (
        [(c, n) for c, n, *_ in p.rows] == [("x", 3), ("y", 2), ("z", 1)]
        and p.total_failures == 6
        and p.total_cases == 20
    )
    assert (
        p.rows[0][2] == 0.5
        and sum(r[2] for r in p.rows) == pytest.approx(1.0)
        and p.rows[0][3] == {"a": 1, "b": 1, "c": 1}
    )
    text = str(p)
    assert "6 failures in 20 cases" in text and "x " in text
    assert pareto([], 5).rows == [] and "0 failures" in str(pareto([], 5))


# ----------------------------------------------------------------------------- comparing variants honestly


def cr(case_id, passed, split="dev", kind="k", cause=""):
    return d1.CaseRun(
        case_id,
        kind,
        split,
        "v",
        passed,
        [] if passed else ["f"],
        "billing",
        "billing",
        [],
        1,
        cause=cause,
    )


def test_compare_is_paired_by_case_id_and_lists_gains_and_regressions():
    before = [cr("a", False), cr("b", False), cr("c", True), cr("d", True), cr("e", False)]
    after = [
        cr("e", True),
        cr("d", False),
        cr("c", True),
        cr("b", True),
        cr("a", False),
    ]  # a different ORDER: matching is by id
    r = d1.compare(before, after)
    assert (
        r["gained"] == ["b", "e"]
        and r["lost"] == ["d"]
        and (r["wins"], r["losses"], r["ties"]) == (2, 1, 2)
    )
    assert r["diff"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        d1.compare([cr("a", True)], [cr("z", True)])


def test_a_clear_improvement_has_an_interval_above_zero_and_noise_does_not():
    before = [cr(f"c{i}", False) for i in range(30)]
    after = [cr(f"c{i}", True) for i in range(30)]
    assert d1.compare(before, after)["ci_low"] > 0.9
    flip = [cr(f"c{i}", i % 2 == 0) for i in range(30)]
    flop = [cr(f"c{i}", i % 2 == 1) for i in range(30)]
    c = d1.compare(flip, flop)
    assert c["ci_low"] < 0 < c["ci_high"] and c["diff"] == 0


def test_save_load_round_trip_and_the_report_reads_only_dev_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(d1, "OUT", tmp_path)
    base = [
        cr("a", False, "dev", "k1", "skipped_tools"),
        cr("b", False, "test", "k2", "misroute"),
        cr("c", True, "dev"),
        cr("d", True, "test"),
    ]
    better = [
        cr("a", True, "dev"),
        cr("b", False, "test", "k2", "misroute"),
        cr("c", True, "dev"),
        cr("d", False, "test"),
    ]
    d1.save(base, tmp_path / "w7d1_base.jsonl")
    d1.save(better, tmp_path / "w7d1_better.jsonl")
    assert d1.load(tmp_path / "w7d1_base.jsonl") == base
    text = d1.report(["base", "better"])
    assert "skipped_tools" in text and "misroute" not in text.split("== better")[0], (
        "the TEST split's failures are never analysed"
    )
    assert (
        "== better vs base on DEV" in text
        and "== better vs base on TEST" in text
        and "REGRESSIONS: ['d']" in text
    )


def test_by_kind_and_pass_rate_helpers():
    runs = [cr("a", True, kind="x"), cr("b", False, kind="x"), cr("c", True, kind="y")]
    assert d1.by_kind(runs) == {"x": (1, 2), "y": (1, 1)}
    assert d1.pass_rate(runs).startswith("67% [")
    assert d1.split_of([cr("a", True, "dev"), cr("b", True, "test")], "test")[0].case_id == "b"


def test_variants_change_exactly_one_thing_each():
    assert d1.VARIANTS["baseline"] == {}
    assert d1.VARIANTS["smart"] == {"force_first_tool": "smart"} and d1.VARIANTS["fewshot"] == {
        "triage_few_shot": True
    }
    assert d1.VARIANTS["smart+fewshot"] == {**d1.VARIANTS["smart"], **d1.VARIANTS["fewshot"]}
    assert d1.VARIANTS["rules"] == {"triage_mode": "rules"} and d1.VARIANTS["rules+smart"] == {
        **d1.VARIANTS["rules"],
        **d1.VARIANTS["smart"],
    }


def test_pareto_orders_by_frequency_not_alphabetically():
    p = pareto([("k", "zebra")] * 3 + [("k", "apple")], total_cases=10)
    assert [c for c, *_ in p.rows] == ["zebra", "apple"]
