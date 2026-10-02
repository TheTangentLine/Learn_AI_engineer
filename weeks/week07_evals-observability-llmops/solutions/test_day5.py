"""Tests for Week 7 Day 5: the cost and latency arithmetic is right, the variants differ by exactly one lever, the choice
is made on dev only, the cache replay counts wrong answers, and the paraphrase labels are consistent."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("opentelemetry.sdk")
sys.path.insert(0, str(Path(__file__).parent))

import cachelab as cb  # noqa: E402
import costlab as cl  # noqa: E402
import day1_solution as d1  # noqa: E402
import day5_solution as d5  # noqa: E402
import evalcases as ec  # noqa: E402
import scripted  # noqa: E402
from support_system.kb import KnowledgeBase  # noqa: E402

from common import llm, tracing  # noqa: E402
from common.cache import ResponseCache, normalize  # noqa: E402
from common.fake import fake_llm  # noqa: E402
from common.tracing import SpanRecord  # noqa: E402


def chat_span(id_, start, i, o, parent=None):
    return SpanRecord(
        "t",
        id_,
        parent,
        "chat m",
        start,
        start + 1,
        "unset",
        "",
        {tracing.OPERATION: "chat", tracing.INPUT_TOKENS: i, tracing.OUTPUT_TOKENS: o},
    )


def tool_span(id_, start):
    return SpanRecord(
        "t",
        id_,
        None,
        "execute_tool x",
        start,
        start + 1,
        "unset",
        "",
        {tracing.OPERATION: "execute_tool"},
    )


# ----------------------------------------------------------------------------- price cards and latency


def test_price_card_arithmetic_is_hand_checked():
    haiku = cl.PriceCard("h", input=1.0, output=5.0, cached_input=0.1)
    assert haiku.cost(1000, 500) == pytest.approx((1000 * 1 + 500 * 5) / 1e6)
    assert haiku.cost(1000, 500, cached_tokens=10_000) == pytest.approx(
        (1000 + 10_000 * 0.1 + 2500) / 1e6
    )
    no_cache_price = cl.PriceCard("x", 2.0, 8.0)
    assert no_cache_price.cost(0, 0, cached_tokens=1_000_000) == pytest.approx(2.0), (
        "unknown cached rate = full price"
    )


def test_price_cards_come_from_the_repo_price_table():
    assert (
        cl.CHEAP.input == llm.PRICES["claude-haiku-4-5"][0]
        and cl.CHEAP.output == llm.PRICES["claude-haiku-4-5"][2]
    )
    assert (
        cl.STRONG.input > cl.CHEAP.input
        and cl.STRONG.cached_input == llm.PRICES["claude-sonnet-5"][1]
    )


def test_latency_model_is_prefill_plus_decode_plus_base():
    m = cl.LatencyModel(base_s=0.25, prefill_tok_s=1000, decode_tok_s=50)
    assert m.call(2000, 100) == pytest.approx(0.25 + 2.0 + 2.0)
    assert m.call(0, 0) == 0.25


def test_trace_cost_and_latency_use_chat_spans_in_order_and_count_tools():
    spans = [chat_span("b", 20, 300, 20), chat_span("a", 10, 100, 10), tool_span("t", 15)]
    assert cl.chat_calls(spans) == [(100, 10), (300, 20)], "start order"
    card = cl.PriceCard("c", 1.0, 5.0)
    assert cl.trace_cost(spans, card) == pytest.approx((400 * 1 + 30 * 5) / 1e6)
    m = cl.LatencyModel(0.1, 1000, 100)
    assert cl.trace_latency(spans, m, tool_s=0.5) == pytest.approx(
        0.1 + 0.1 + 0.1 + 0.1 + 0.3 + 0.2 + 0.5
    )


# ----------------------------------------------------------------------------- variants and the dev-only decision


def test_every_variant_differs_from_the_baseline_by_exactly_its_named_levers():
    assert d5.VARIANTS["base"] == d5.BASE_OPTS
    for name, opts in d5.VARIANTS.items():
        if name == "base":
            continue
        extra = {k: v for k, v in opts.items() if d5.BASE_OPTS.get(k) != v}
        expected = {k: v for part in name.split("+") for k, v in d5.LEVERS[part].items()}
        assert extra == expected and all(opts[k] == v for k, v in d5.BASE_OPTS.items()), name
    assert (
        len(d5.VARIANTS) == 8 and len({str(sorted(v.items())) for v in d5.VARIANTS.values()}) == 8
    )


def test_choose_on_dev_picks_the_cheapest_variant_that_does_not_lose_dev_accuracy():
    rows = {
        "base": {"dev_rate": 0.5, "cost": 1.0},
        "cheap_but_worse": {"dev_rate": 0.4, "cost": 0.3},
        "good": {"dev_rate": 0.5, "cost": 0.8},
        "best": {"dev_rate": 0.7, "cost": 0.9},
        "cheapest_ok": {"dev_rate": 0.6, "cost": 0.6},
    }
    assert d5.choose_on_dev(rows) == "cheapest_ok"
    assert d5.choose_on_dev({"base": rows["base"], "x": rows["cheap_but_worse"]}) == "base", (
        "base itself qualifies"
    )
    tie = {
        "base": {"dev_rate": 0.5, "cost": 1.0},
        "b": {"dev_rate": 0.5, "cost": 0.7},
        "a": {"dev_rate": 0.5, "cost": 0.7},
    }
    assert d5.choose_on_dev(tie) == "a", "ties break by name, deterministically"


def test_the_choice_never_reads_a_test_field():
    row = {"dev_rate": 0.5, "cost": 1.0}
    rows = {"base": row, "x": {"dev_rate": 0.6, "cost": 0.5}}  # no test_rate key at all
    assert d5.choose_on_dev(rows) == "x"


def make_runs(passed: set[str], ids: list[tuple[str, str]]):
    return [
        d1.CaseRun(
            cid, cid.rsplit("-", 1)[0], split, "v", cid in passed, [], "billing", "billing", [], 1
        )
        for cid, split in ids
    ]


IDS = [(f"billing-small-{i:02d}", "dev" if i < 6 else "test") for i in range(10)] + [
    (f"human-{i:02d}", "dev") for i in range(4)
]


def test_decide_restricts_the_gate_to_a_split():
    base = make_runs({f"billing-small-{i:02d}" for i in range(10)}, IDS)
    cand = make_runs(
        {f"billing-small-{i:02d}" for i in range(6)}, IDS
    )  # loses 4, all of them TEST cases
    assert d5.decide(base, cand, "dev").regressions == []
    assert d5.decide(base, cand, "test").regressions == [
        f"billing-small-0{i}" for i in range(6, 10)
    ]
    assert len(d5.decide(base, cand).regressions) == 4


def test_critical_cases_in_the_gate_use_the_day3_patterns():
    base = make_runs({"human-00"}, IDS)
    cand = make_runs(set(), IDS)
    d = d5.decide(base, cand)
    assert d.critical_regressions == ["human-00"] and d.status == "fail"


def test_usage_and_cost_per_1k_average_over_cases():
    by_case = {
        "a": [chat_span("1", 1, 100, 10), chat_span("2", 2, 200, 20)],
        "b": [chat_span("3", 1, 300, 30)],
    }
    u = d5.usage(by_case)
    assert u == {"cases": 2, "model_calls": 1.5, "input_tokens": 300.0, "output_tokens": 30.0}
    assert d5.usage(by_case, ["b"])["input_tokens"] == 300
    card = cl.PriceCard("c", 1.0, 5.0)
    assert d5.cost_per_1k(by_case, card) == pytest.approx(
        ((300 + 150) + (300 + 150)) / 2 / 1e6 * 1000
    )
    assert d5.cost_per_1k({}, card) == 0.0


def test_per_case_groups_spans_by_the_eval_case_root():
    root = SpanRecord(
        "t1", "r", None, "eval.case", 0, 9, "unset", "", {"app.eval.case_id": "human-00"}
    )
    child = SpanRecord("t1", "c", "r", "chat m", 1, 2, "unset", "", {})
    stray = SpanRecord("t2", "s", None, "other", 0, 1, "unset", "", {})
    got = d5.per_case([child, root, stray])
    assert list(got) == ["human-00"] and len(got["human-00"]) == 2


# ----------------------------------------------------------------------------- the cache lab


def test_intents_share_an_id_across_a_same_pair_and_are_unique_otherwise():
    ids = d5.intents()
    for p in cb.SAME:
        assert ids[p.a] == ids[p.b]
    for p in cb.NEAR:
        assert ids[p.a] != ids[p.b], p
    assert len(set(ids.values())) < len(ids)


def test_surface_noise_never_changes_the_question():
    """The traffic generator adds case, spacing and punctuation only: after normalisation the question is unchanged."""
    rng = random.Random(0)
    for text in sorted(d5.intents()):
        for _ in range(20):
            assert normalize(d5.surface_variant(text, rng)) == normalize(text)


def test_surface_noise_really_varies_the_text():
    rng = random.Random(1)
    variants = {d5.surface_variant("how do I reset my password", rng) for _ in range(300)}
    assert "how do I reset my password" in variants
    assert any(v.isupper() for v in variants) and any(
        v[0].isupper() and not v.isupper() for v in variants
    )
    assert any(v.endswith(("?", "!", ".")) for v in variants) and any("  " in v for v in variants)
    assert len(variants) >= 8


def test_traffic_is_deterministic_skewed_and_labelled():
    a, b = d5.traffic(200, seed=3), d5.traffic(200, seed=3)
    assert a == b and a != d5.traffic(200, seed=4)
    ids = d5.intents()
    assert all(i in set(ids.values()) for _, i in a)
    counts = sorted((sum(1 for _, i in a if i == k) for k in set(i for _, i in a)), reverse=True)
    assert counts[0] > 3 * counts[-1], "a few intents dominate"
    big = d5.traffic(1000, seed=0)
    top_share = max(sum(1 for _, i in big if i == k) for k in set(i for _, i in big)) / 1000
    assert top_share > 0.10, (
        f"popularity skew: the top intent should take a large share, got {top_share:.1%}"
    )
    quiet = d5.traffic(100, seed=0, surface_noise=0.0)
    assert all(t in ids for t, _ in quiet)


def test_cache_replay_counts_hits_savings_and_wrong_answers_by_hand():
    stream = [("a b c", 1), ("A B C!", 1), ("x y z", 2), ("a b c d", 3), ("x y z", 2)]
    r = d5.cache_replay(stream, ResponseCache(), miss_cost=0.01)
    assert (r["requests"], r["hit_rate"], r["wrong_answers"]) == (5, 0.4, 0)  # exact: 2nd and 5th
    assert (
        r["cost"] == pytest.approx(0.03)
        and r["cost_no_cache"] == pytest.approx(0.05)
        and r["saved"] == pytest.approx(0.4)
    )
    fuzzy = d5.cache_replay(stream, ResponseCache(fuzzy=True, threshold=0.6), miss_cost=0.01)
    # "a b c d" (intent 3) vs "a b c" (intent 1): "a" is a stopword, so the word sets are {b,c,d} and {b,c}: 2/3 >= 0.6
    assert (
        fuzzy["hit_rate"] == 0.6
        and fuzzy["wrong_answers"] == 1
        and fuzzy["wrong_rate"] == pytest.approx(0.2)
    )


def test_exact_matching_serves_no_wrong_answers_on_the_simulated_traffic_and_fuzzy_does():
    stream = d5.traffic(400)
    exact = d5.cache_replay(stream, ResponseCache(max_entries=10_000), 1.0)
    fuzzy = d5.cache_replay(
        stream, ResponseCache(max_entries=10_000, fuzzy=True, threshold=0.6), 1.0
    )
    assert exact["wrong_answers"] == 0 and exact["hit_rate"] > 0.5
    assert fuzzy["wrong_answers"] > 0 and fuzzy["hit_rate"] >= exact["hit_rate"]


# ----------------------------------------------------------------------------- the labelled pairs


def test_pairs_are_well_formed():
    assert (
        len(cb.SAME) == 20
        and len(cb.NEAR) == 20
        and all(p.same for p in cb.SAME)
        and not any(p.same for p in cb.NEAR)
    )
    keys = {(p.a, p.b) for p in cb.PAIRS}
    assert len(keys) == 40 and all(
        p.a != p.b and normalize(p.a) != normalize(p.b) for p in cb.PAIRS
    )


def test_every_paraphrase_pair_is_answered_by_the_same_help_article():
    """The label-consistency check: a SAME pair is only 'same' if the real knowledge base retrieves one article for both."""
    kb = KnowledgeBase()
    top = lambda q: kb.search(q, 1)[0][0]  # noqa: E731
    bad = [(p.a, p.b, top(p.a), top(p.b)) for p in cb.SAME if top(p.a) != top(p.b)]
    assert bad == []


def test_some_near_pairs_are_corroborated_by_different_articles_and_the_rest_are_judgement():
    kb = KnowledgeBase()
    top = lambda q: kb.search(q, 1)[0][0]  # noqa: E731
    different = sum(top(p.a) != top(p.b) for p in cb.NEAR)
    assert different >= 5, (
        "at least a quarter of the near-miss labels are backed by retrieval, not only by my judgement"
    )


def test_guards_ok_checks_entities_and_negation():
    assert cb.guards_ok("refund INV-1", "please refund inv-1")
    assert not cb.guards_ok("refund INV-1", "refund INV-2")
    assert not cb.guards_ok("export works", "export does not work")
    assert cb.guards_ok(
        "export fails", "export is failing"
    )  # both negative-ish words: 'fails' / 'failing' are negation markers


def test_curve_is_hand_checked_with_a_stub_similarity():
    pairs = [
        cb.Pair("a1", "a2", True),
        cb.Pair("b1", "b2", True),
        cb.Pair("c1", "c2", False),
        cb.Pair("d1", "d2", False),
    ]
    sims = {"a1": 0.9, "b1": 0.6, "c1": 0.8, "d1": 0.3}
    rows = cb.curve(pairs, lambda a, b: sims[a], [0.5, 0.7, 0.95], guards=None)
    assert [(r["hit_rate_same"], r["wrong_rate_near"]) for r in rows] == [
        (1.0, 0.5),
        (0.5, 0.5),
        (0.0, 0.0),
    ]
    assert rows[0]["precision"] == pytest.approx(2 / 3) and rows[2]["precision"] == 1.0, (
        "no hits: nothing wrong was served"
    )
    assert rows[1]["wrong_pairs"] == ["c1 ~ c2"]
    only_same = cb.curve(pairs, lambda a, b: sims[a], [0.5], guards=lambda a, b: a != "c1")
    assert only_same[0]["wrong_rate_near"] == 0.0


# ----------------------------------------------------------------------------- cascade and batch


def test_cascade_projection_is_hand_checked():
    ids = [(f"billing-small-0{i}", "dev") for i in range(5)]
    # 00 pass, clean | 01 pass but the trace shows a defect | 02 fail + defect | 03 fail, trace looks clean | 04 pass + only an EVENT
    runs = make_runs({"billing-small-00", "billing-small-01", "billing-small-04"}, ids)

    def case(cid, *extra):
        return [
            SpanRecord(
                cid, f"r{cid}", None, "eval.case", 0, 9, "unset", "", {"app.eval.case_id": cid}
            ),
            SpanRecord(
                cid,
                f"c{cid}",
                f"r{cid}",
                "chat m",
                1,
                2,
                "unset",
                "",
                {tracing.OPERATION: "chat", tracing.INPUT_TOKENS: 1000, tracing.OUTPUT_TOKENS: 100},
            ),
            *extra,
        ]

    def no_tool_agent(cid):  # a DEFECT: the specialist answered without any tool call
        return SpanRecord(
            cid,
            f"a{cid}",
            f"r{cid}",
            "invoke_agent billing",
            1,
            2,
            "unset",
            "",
            {
                tracing.OPERATION: "invoke_agent",
                tracing.AGENT_NAME: "billing",
                tracing.AGENT_STATUS: "done",
            },
        )

    def escalation_event(cid):  # an EVENT: not a defect, must not trigger the strong tier
        return SpanRecord(
            cid,
            f"w{cid}",
            f"r{cid}",
            "invoke_workflow support",
            1,
            2,
            "unset",
            "",
            {
                tracing.OPERATION: "invoke_workflow",
                "app.reply.status": "escalated",
                "app.reply.agent": "human",
            },
        )

    by_case = {
        "billing-small-00": case("billing-small-00"),
        "billing-small-01": case("billing-small-01", no_tool_agent("billing-small-01")),
        "billing-small-02": case("billing-small-02", no_tool_agent("billing-small-02")),
        "billing-small-03": case("billing-small-03"),
        "billing-small-04": case("billing-small-04", escalation_event("billing-small-04")),
    }
    cheap, strong = cl.PriceCard("c", 1.0, 5.0), cl.PriceCard("s", 3.0, 15.0)
    r = d5.cascade_projection(runs, by_case, cheap, strong)
    conv = (1000 * 1 + 100 * 5) / 1e6  # one conversation on the cheap card
    assert (
        r["escalation_rate"] == 0.4
        and r["escalated_that_failed"] == 1
        and r["escalated_that_passed"] == 1
    )
    assert r["failed_not_escalated"] == 1, "the failure the trace cannot see"
    assert r["cheap_only_cost"] == pytest.approx(conv * 1000)
    assert r["cascade_cost"] == pytest.approx(conv * 1000 + 3 * conv * 2 / 5 * 1000)
    assert r["strong_only_cost"] == pytest.approx(3 * conv * 1000)
    assert r["cheap_pass"] == 0.6 and r["cascade_pass_upper_bound"] == 0.8, (
        "only escalated FAILURES can be fixed"
    )


def test_batch_cost_applies_the_assumed_discount():
    assert (
        d5.batch_cost(10.0) == 5.0
        and d5.batch_cost(10.0, 0.3) == pytest.approx(7.0)
        and d5.batch_cost(0.0) == 0.0
    )


# ----------------------------------------------------------------------------- the mechanism, end to end with scripted models


def run_scripted(name, **faults):
    with fake_llm(scripted.rules(**faults)):
        return d5.run_variant_traced(name, provider="anthropic")


@pytest.fixture(scope="module")
def scripted_runs():
    return {n: run_scripted(n) for n in ("base", "direct")}


def test_direct_replies_remove_a_model_call_from_refund_conversations_and_nothing_else(
    scripted_runs,
):
    (base_runs, base_spans), (direct_runs, direct_spans) = (
        scripted_runs["base"],
        scripted_runs["direct"],
    )
    base_calls, direct_calls = d5.per_case(base_spans), d5.per_case(direct_spans)
    saved = {
        cid: len(cl.chat_calls(base_calls[cid])) - len(cl.chat_calls(direct_calls[cid]))
        for cid in base_calls
    }
    assert all(v >= 0 for v in saved.values()), "never MORE calls"
    refunds = {
        cid: sum(1 for t in tracing.tool_sequence(spans) if t == "request_refund")
        for cid, spans in base_calls.items()
    }
    assert saved == {cid: min(n, 1) if n else 0 for cid, n in refunds.items()} or all(
        (v > 0) == (refunds[cid] > 0) for cid, v in saved.items()
    ), "exactly the conversations in which a refund was requested lose their last model call"
    assert sum(1 for v in saved.values() if v > 0) >= 10
    passed = lambda runs: {r.case_id for r in runs if r.passed}  # noqa: E731
    # the keyword router misroutes one how-to ("...email invoices are sent to" matches "invoices"): it does so in
    # BOTH variants. What matters is that building the reply in code changes no verdict.
    assert passed(direct_runs) == passed(base_runs) and len(passed(base_runs)) == 49
    assert d5.cost_per_1k(direct_calls, cl.CHEAP) < d5.cost_per_1k(base_calls, cl.CHEAP)


def test_save_load_and_report_work_from_files_only(scripted_runs, tmp_path):
    for name, (runs, spans) in scripted_runs.items():
        d5.save_variant(name, runs, spans, tmp_path)
    runs, spans = d5.load_variant("direct", tmp_path)
    assert len(runs) == 50 and len(d5.per_case(spans)) == 50
    text = d5.report(["base", "direct"], tmp_path)
    assert "direct" in text and "Chosen on DEV only" in text
    rows = d5.summarize_all(["base", "direct"], tmp_path)
    assert (
        rows["base"]["cost"] > rows["direct"]["cost"]
        and rows["direct"]["model_calls"] < rows["base"]["model_calls"]
    )
    assert (
        rows["direct"]["dev_rate"] == rows["base"]["dev_rate"] == 1.0
        and rows["direct"]["gate_all"] == "pass"
    )
    assert d5.choose_on_dev(rows) == "direct"


def test_the_case_set_matches_the_day1_dataset(scripted_runs):
    runs, _ = scripted_runs["base"]
    assert [r.case_id for r in runs] == [c.id for c in ec.build_cases(lambda: None)]


# ----------------------------------------------------------------------------- the headline, on the saved real-model runs

OUT = d5.OUT
needs_runs = pytest.mark.skipif(
    not all((OUT / f"w7d5_{n}.jsonl").exists() for n in d5.VARIANTS),
    reason="Day 5 real-model runs not generated",
)


@needs_runs
def test_real_runs_the_chosen_variant_cuts_cost_without_losing_score_and_without_critical_regressions():
    rows = d5.summarize_all(list(d5.VARIANTS))
    pick = d5.choose_on_dev(rows)
    assert pick == "lean+hide+direct"
    r, base = rows[pick], rows["base"]
    assert r["cost"] / base["cost"] - 1 <= -0.40
    assert r["dev_rate"] >= base["dev_rate"] and r["test_rate"] >= base["test_rate"]
    assert r["critical"] == [] and r["gate_all"] in ("pass", "improved")
    assert r["model_calls"] < base["model_calls"] and r["input_tokens"] < 0.6 * base["input_tokens"]


@needs_runs
def test_real_runs_each_lever_alone_is_not_enough_for_forty_percent():
    rows = d5.summarize_all(list(d5.VARIANTS))
    for name in ("lean", "hide", "direct"):
        assert rows[name]["cost"] / rows["base"]["cost"] - 1 > -0.40, name


def test_prefix_tokens_lean_is_shorter_for_both_specialists():
    try:
        full, lean = d5.prefix_tokens(False), d5.prefix_tokens(True)
    except Exception as exc:  # the tokenizer is not in the local Hugging Face cache
        pytest.skip(f"local tokenizer unavailable: {type(exc).__name__}")
    assert set(full) == set(lean) == {"billing", "tech"}
    for agent in full:
        assert 0 < lean[agent] < full[agent], agent
    assert full["billing"] > full["tech"], (
        "three tools with long descriptions vs three shorter ones"
    )
    assert full["billing"] < 1024, (
        "below the minimum prompt length that provider prompt caches typically require"
    )
