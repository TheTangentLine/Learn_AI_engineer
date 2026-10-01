"""Tests for Week 5 Day 3: both tool designs can answer every question (a fair comparison), the scoring is
strict, and every design lever (errors, defaults, enums, pagination) behaves as the lesson claims."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))

import day3_solution as d3  # noqa: E402

from common.chat import ToolCall  # noqa: E402
from common.fake import fake_llm, tool_calls  # noqa: E402

QS = d3.build_questions()
BAD = d3.make_bad_registry()
GOOD = d3.make_good_registry()


def run(reg, name, **args):
    return reg.execute(ToolCall("t", name, args))


# ----------------------------------------------------------------------------- data and questions


def test_db_is_deterministic_and_statuses_are_not_tied_to_customers():
    assert d3.build_db() == d3.DB and len(d3.DB) == 40 and len({o["id"] for o in d3.DB}) == 40
    assert d3.DB[0]["id"] == "A1001" and d3.DB[-1]["id"] == "A1040"
    combos = {(o["customer"], o["status"]) for o in d3.DB}
    assert len(combos) >= 12, "a modular status cycle once left most customer+status pairs empty"
    assert {o["status"] for o in d3.DB} == set(d3.STATUSES)


def test_question_set_shape_and_answers_are_usable():
    assert len(QS) == 40 and len({q.id for q in QS}) == 40
    kinds = {k: sum(q.kind == k for q in QS) for k in {q.kind for q in QS}}
    assert kinds == {"customer": 8, "customer+status": 8, "recent": 8, "by-id": 8, "after-date": 8}
    for q in QS:
        assert 1 <= len(q.expected) <= 10, (
            f"{q.id}: a question with no answer, or more than one default page"
        )


def test_expected_answers_match_an_independent_computation():
    by_id = {o["id"]: o for o in d3.DB}
    for q in QS:
        rows = [by_id[i] for i in q.expected]
        if q.kind == "customer":
            assert all(o["customer"] == q.customer for o in rows)
            assert len(rows) == sum(o["customer"] == q.customer for o in d3.DB)
        elif q.kind == "customer+status":
            assert all(o["customer"] == q.customer and o["status"] == q.status for o in rows)
        elif q.kind == "recent":
            same = [o for o in d3.DB if o["status"] == q.status]
            newest = sorted(o["created"] for o in same)[-len(rows) :]
            assert sorted(o["created"] for o in rows) == newest
        elif q.kind == "after-date":
            date = re.search(r"\d{4}-\d{2}-\d{2}", q.text).group()
            assert all(o["created"] > date and o["status"] == q.status for o in rows)


# ----------------------------------------------------------------------------- fairness: both designs can express every question


def perfect_good_call(q):
    if q.kind == "by-id":
        return ToolCall("c", "get_order", {"order_id": q.expected[0]})
    args = {
        "customer": {"customer_email": q.customer},
        "customer+status": {"customer_email": q.customer, "status": q.status},
    }.get(q.kind)
    if q.kind == "recent":
        args = {"status": q.status, "limit": len(q.expected), "newest_first": True}
    if q.kind == "after-date":
        date = re.search(r"\d{4}-\d{2}-\d{2}", q.text).group()
        args = {"status": q.status, "created_after": date}
    return ToolCall("c", "find_orders", args)


def perfect_bad_call(q):
    if q.kind == "by-id":
        return ToolCall("c", "get", {"id": q.expected[0]})
    f = {
        "customer": "",
        "customer+status": f"status={q.status}",
        "recent": f"status={q.status};sort=desc",
    }.get(q.kind)
    if q.kind == "after-date":
        date = re.search(r"\d{4}-\d{2}-\d{2}", q.text).group()
        f = f"status={q.status};after={date}"
    return ToolCall("c", "search", {"q": q.customer, "f": f, "n": len(q.expected)})


@pytest.mark.parametrize("q", QS, ids=lambda q: q.id)
def test_every_question_is_answerable_by_both_designs(q):
    assert d3.score_calls(q, GOOD, [perfect_good_call(q)]).ok, "GOOD cannot answer it"
    assert d3.score_calls(q, BAD, [perfect_bad_call(q)]).ok, (
        "BAD cannot answer it: the comparison would be unfair"
    )
    v2 = d3.make_good_registry(newest_default=True)
    call = perfect_good_call(q)
    call.args.pop("newest_first", None)  # v2 must not need the flag
    assert d3.score_calls(q, v2, [call]).ok


# ----------------------------------------------------------------------------- scoring strictness


def test_score_requires_exactly_the_expected_orders():
    q = next(x for x in QS if x.id == "cs-0")
    assert d3.score_calls(q, GOOD, [perfect_good_call(q)]).ok
    assert not d3.score_calls(
        q, GOOD, [ToolCall("c", "find_orders", {"customer_email": q.customer})]
    ).ok, "superset"
    assert not d3.score_calls(
        q,
        GOOD,
        [
            ToolCall(
                "c", "find_orders", {"customer_email": q.customer, "status": q.status, "limit": 1}
            )
        ],
    ).ok, "subset"
    assert not d3.score_calls(q, BAD, [ToolCall("c", "list_all", {})]).ok, (
        "dumping everything is not an answer"
    )
    none = d3.score_calls(q, GOOD, [])
    assert not none.ok and not none.called
    wrong_arg = d3.score_calls(q, GOOD, [ToolCall("c", "find_orders", {"status": "shipping"})])
    assert not wrong_arg.ok and wrong_arg.n_errors == 1


def test_score_ignores_order_and_counts_errors_from_both_error_styles():
    q = next(x for x in QS if x.id == "recent-1")
    out = d3.score_calls(q, GOOD, [perfect_good_call(q)])
    assert out.ok and out.n_errors == 0
    opaque = d3.score_calls(q, BAD, [ToolCall("c", "search", {"q": "", "f": "status:shipped"})])
    assert opaque.n_errors == 1 and not opaque.ok, (
        "'Error: ...' strings count even though they are not raised"
    )


# ----------------------------------------------------------------------------- the BAD tools behave as specified


def test_bad_search_filter_language():
    assert (
        run(BAD, "search", q="alice@example.com", f="status=cancelled").content.count("'id'") == 4
    )
    rows = run(BAD, "search", q="", f="sort=desc", n=3).content.splitlines()
    assert [eval(r)["id"] for r in rows] == ["A1040", "A1039", "A1038"]  # noqa: S307 - our own repr output
    assert run(BAD, "search", q="", f="after=2026-04-27").content.count("'id'") == 2
    assert run(BAD, "search", q="zzz").content == "[]"
    assert (
        run(BAD, "get", id="a1007").content.startswith("{'id': 'A1007'")
        and run(BAD, "get", id="A9999").content == "None"
    )
    assert run(BAD, "list_all").content.count("\n") == 39


def test_bad_errors_are_opaque_and_the_helpful_variant_teaches_the_syntax():
    opaque = run(BAD, "search", q="x", f="status:shipped").content
    assert opaque == "Error: bad filter"
    helpful = run(
        d3.make_bad_registry(helpful_errors=True), "search", q="x", f="status:shipped"
    ).content
    for needle in (
        "'status:shipped'",
        "no '='",
        "key=value",
        "status=shipped;after=",
        "pending|shipped|delivered|cancelled",
        "sort",
    ):
        assert needle in helpful, needle
    unknown = run(d3.make_bad_registry(helpful_errors=True), "search", q="x", f="color=red").content
    assert "unknown filter" in unknown and "Allowed keys" in unknown
    assert run(BAD, "search", q="x", f="status=shipping").content == "Error: bad filter", (
        "invalid value, not just bad syntax"
    )


# ----------------------------------------------------------------------------- the GOOD tools behave as specified


def test_good_find_orders_formats_paginates_and_hints():
    out = run(GOOD, "find_orders", customer_email="ALICE@example.com", limit=3).content
    lines = out.splitlines()
    assert (
        lines[0] == "3 of 10 matching orders (raise limit to see the other 7):" and len(lines) == 4
    )
    assert lines[1].startswith("A1001 | alice@example.com | ") and lines[1].endswith(
        "| created 2026-01-04"
    )
    full = run(GOOD, "find_orders", customer_email="alice@example.com").content
    assert full.splitlines()[0] == "10 of 10 matching orders:"
    empty = run(GOOD, "find_orders", customer_email="nobody@example.com").content
    assert empty == "0 of 0 matching orders. Nothing matched; try fewer filters."


def test_good_find_orders_validates_and_the_errors_say_what_is_allowed():
    bad_status = run(GOOD, "find_orders", status="shipping")
    assert (
        bad_status.is_error
        and "'pending', 'shipped', 'delivered' or 'cancelled'" in bad_status.content
    )
    assert (
        "between 1 and 50" in run(GOOD, "find_orders", limit=0).content
        and "between 1 and 50" in run(GOOD, "find_orders", limit=51).content
    )
    date = run(GOOD, "find_orders", created_after="March 1")
    assert (
        date.is_error and "ISO date like 2026-03-01" in date.content and "'March 1'" in date.content
    )
    assert "Invalid arguments" in run(GOOD, "find_orders", email="a@b.c").content, (
        "misspelt parameter name"
    )
    missing = run(GOOD, "get_order", order_id="Z1")
    assert (
        missing.is_error and "A1001..A1040" in missing.content and "find_orders" in missing.content
    )
    assert run(GOOD, "get_order", order_id=" a1007 ").content.startswith("A1007 | ")


def test_good_sort_default_and_documentation_change_together_in_v2():
    v1 = GOOD.specs()[0]
    v2 = d3.make_good_registry(newest_default=True).specs()[0]
    assert v1["parameters"]["properties"]["newest_first"]["default"] is False
    assert v2["parameters"]["properties"]["newest_first"]["default"] is True
    assert (
        "oldest first unless newest_first is true" in v1["description"]
        and "newest first unless newest_first is false" in v2["description"]
    )
    assert "{ORDER_NOTE}" not in v1["description"] + v2[
        "description"
    ] and "{NEWEST_ARG}" not in str(v1) + str(v2)
    default_call = ToolCall("c", "find_orders", {"status": "pending", "limit": 2})
    newest = (
        d3.make_good_registry(newest_default=True).execute(default_call).content.splitlines()[1:]
    )
    oldest = GOOD.execute(default_call).content.splitlines()[1:]
    assert newest[0].split(" | ")[-1] > oldest[0].split(" | ")[-1]


def test_status_is_an_enum_in_the_good_schema_and_a_free_string_in_the_bad_one():
    good = GOOD.specs()[0]["parameters"]["properties"]["status"]
    assert "enum" in str(good) and "cancelled" in str(good)
    assert "enum" not in str(BAD.specs()[0]["parameters"])


# ----------------------------------------------------------------------------- the linter


def test_lint_flags_the_bad_designs_and_passes_the_good_ones():
    bad_issues = {s["name"]: d3.lint_spec(s) for s in BAD.specs()}
    assert len(bad_issues["search"]) >= 6 and any("no description" in i for i in bad_issues["get"])
    assert any("cryptic" in i for i in bad_issues["search"]) and not any(
        "cryptic" in i for i in bad_issues["get"]
    ), "'id' is fine"
    assert any("characters" in i for i in bad_issues["list_all"])
    for reg in (GOOD, d3.make_good_registry(newest_default=True)):
        assert all(d3.lint_spec(s) == [] for s in reg.specs())
    spec = {
        "name": "t",
        "description": "x" * 50,
        "parameters": {"properties": {"status": {"type": "string", "description": "d"}}},
    }
    assert d3.lint_spec(spec) == ["parameter 'status' looks like a closed set but has no enum"]


# ----------------------------------------------------------------------------- evaluation plumbing


def test_first_call_eval_scores_what_the_model_actually_called():
    q = next(x for x in QS if x.id == "id-0")
    with fake_llm([(r"(?s).*", tool_calls(("get_order", {"order_id": "A1007"})))]) as f:
        out = d3.first_call_eval(GOOD, [q], "anthropic")
    assert out[0].ok and out[0].called and len(f.calls) == 1
    with fake_llm([(r"(?s).*", "I cannot look that up.")]):
        out = d3.first_call_eval(GOOD, [q], "anthropic")
    assert not out[0].ok and not out[0].called


def test_error_recovery_eval_shows_the_model_its_failure_and_scores_the_second_call():
    q = next(x for x in QS if x.id == "cs-0")
    first = d3.wrong_first_call(q)
    assert (
        first.args["f"] == f"status:{q.status}"
        and run(BAD, first.name, **first.args).is_error is False
    )  # returns text
    assert d3.score_calls(q, BAD, [first]).n_errors == 1
    fixed = tool_calls(("search", {"q": q.customer, "f": f"status={q.status}"}))
    with fake_llm([(r"Error: bad filter", fixed), (r"(?s).*", "give up")]) as f:
        out = d3.error_recovery_eval(BAD, [q], "anthropic")
    assert out[0].ok
    assert "search(" in f.calls[0].prompt and "Error: bad filter" in f.calls[0].prompt, (
        "the model sees its call AND the error"
    )
    with fake_llm([(r"(?s).*", "give up")]):
        assert not d3.error_recovery_eval(BAD, [q], "anthropic")[0].ok


def test_rate_and_by_kind_summaries():
    outs = [
        d3.Outcome(QS[0], [], "", 0, True),
        d3.Outcome(QS[1], [], "", 0, False),
        d3.Outcome(QS[8], [], "", 0, True),
    ]
    assert d3.rate(outs) == [1.0, 0.0, 1.0] and d3.rate(outs, lambda o: o.called) == [0.0, 0.0, 0.0]
    assert d3.by_kind(outs) == {"customer": (1, 2), "customer+status": (1, 1)}


def test_context_cost_ordering():
    c = d3.context_cost()
    assert (
        c["list_all (BAD)"]
        > c["search n=10 (BAD)"]
        > c["find_orders limit=10 (GOOD)"]
        > c["find_orders limit=3 (GOOD)"]
    )
    assert c["list_all (BAD)"] > 3 * c["find_orders limit=10 (GOOD)"]


def test_uses_fix_detects_a_reply_that_states_the_syntax_but_not_a_shrug():
    q = QS[0]
    mk = lambda said: d3.Outcome(q, [], "", 0, False, said)  # noqa: E731
    assert d3.uses_fix(mk("The format is key=value, e.g. status=shipped"))
    assert d3.uses_fix(mk("Try status=cancelled"))
    assert not d3.uses_fix(mk("The filter you used is incorrect. It should be `status:shipped`."))
    assert not d3.uses_fix(mk("Could you provide an order ID?"))


def test_error_recovery_eval_records_what_the_model_said_when_it_does_not_retry():
    q = next(x for x in QS if x.id == "cs-0")
    with fake_llm([(r"(?s).*", "Please use key=value filters.")]):
        out = d3.error_recovery_eval(d3.make_bad_registry(helpful_errors=True), [q], "anthropic")[0]
    assert (
        not out.called
        and not out.ok
        and out.said == "Please use key=value filters."
        and d3.uses_fix(out)
    )


def test_after_filters_are_strictly_after_the_boundary_date_in_both_designs():
    boundary = next(o for o in d3.DB if o["id"] == "A1005")["created"]
    good = run(GOOD, "find_orders", created_after=boundary, limit=50).content
    bad = run(BAD, "search", q="", f=f"after={boundary}", n=50).content
    for out in (good, bad):
        assert "A1005" not in out and "A1006" in out, (
            "an order created ON the date is not 'after' it"
        )


def test_a_run_with_no_tool_call_is_never_a_success_even_when_nothing_was_expected():
    q = d3.Question("x", "k", "t", [])
    assert d3.score_calls(q, GOOD, []).ok is False


def test_holdout_questions_are_new_answerable_and_use_the_same_scoring():
    hold = d3.build_holdout()
    assert len(hold) == 8 and not {q.text for q in hold} & {q.text for q in QS}
    for q in hold:
        assert 2 <= len(q.expected) <= 5
        call = perfect_good_call(q)
        assert d3.score_calls(q, GOOD, [call]).ok
        assert d3.score_calls(
            q,
            d3.make_good_registry(newest_default=True),
            [ToolCall("c", "find_orders", {"status": q.status, "limit": len(q.expected)})],
        ).ok
        assert not d3.score_calls(
            q, GOOD, [ToolCall("c", "find_orders", {"status": q.status, "limit": len(q.expected)})]
        ).ok, "v1 default is oldest-first"
