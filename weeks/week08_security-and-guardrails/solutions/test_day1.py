"""Tests for Week 8 Day 1: the threat model is internally consistent, matches the code, and the validator catches each way it can rot."""

from __future__ import annotations

import dataclasses
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import threatmodel as tm  # noqa: E402

M = tm.RESEARCH_AGENT


def replace_threat(model: tm.Model, tid: str, **changes) -> tm.Model:
    threats = tuple(dataclasses.replace(t, **changes) if t.id == tid else t for t in model.threats)
    return dataclasses.replace(model, threats=threats)


def messages(model: tm.Model) -> str:
    return "\n".join(tm.validate(model))


# ----------------------------------------------------------------------------- the real model


def test_the_research_agent_threat_model_validates_against_the_real_code():
    assert tm.validate(M) == []


def test_every_tool_the_agent_exposes_is_in_the_model_and_in_a_threat():
    actual: set[str] = set()
    for src in M.tool_sources:
        actual |= tm.tools_in_source(tm.ROOT / src)
    assert (
        actual
        == set(M.tools)
        == {
            "search_web",
            "fetch_page",
            "create_note",
            "append_to_note",
            "read_note",
            "search_notes",
            "list_notes",
        }
    )
    covered = {s for t in M.threats for s in t.surface}
    assert set(M.tools) <= covered and set(M.entry_points) <= covered


def test_the_model_has_honest_gaps_and_the_expected_summary():
    s = tm.summary(M)
    assert s == {
        "mitigated": 5,
        "partial": 3,
        "gap": 1,
        "accepted": 1,
        "total": 10,
        "open_risk": 49,
    }
    gap = next(t for t in M.threats if t.status == "gap")
    assert gap.id == "T03" and gap.owasp == "LLM05" and "image" in gap.title


def test_ranking_is_by_risk_then_gaps_first():
    ranked = tm.ranked(M)
    assert [t.risk for t in ranked] == sorted((t.risk for t in M.threats), reverse=True)
    equal = [t for t in ranked if t.risk == 16]
    assert [t.status for t in equal] == sorted(
        (t.status for t in equal), key=["gap", "partial", "mitigated", "accepted"].index
    )
    assert ranked[0].id in {"T01", "T02", "T05"} or ranked[0].risk == 20


def test_risk_is_likelihood_times_impact():
    t = tm.Threat("X", "x", "LLM01", "S", (), (), 3, 5, "accepted", reason="r")
    assert t.risk == 15


def test_markdown_lists_every_threat_control_and_the_summary():
    md = tm.render_markdown(M)
    for t in M.threats:
        assert f"| {t.id} |" in md
    for c in M.controls:
        assert f"**{c.id}**" in md and c.evidence[0] in md
    assert "10 threats: 5 mitigated, 3 partial, 1 gaps, 1 accepted; open risk score 49." in md
    assert md.index("| T02 |") < md.index("| T10 |"), "highest risk first"


# ----------------------------------------------------------------------------- the validator catches rot


def test_a_renamed_or_deleted_test_breaks_the_model():
    controls = tuple(
        dataclasses.replace(
            c,
            evidence=tuple(
                e.replace("test_fetcher_refuses_everything_off_the_allowlist", "test_renamed_away")
                for e in c.evidence
            ),
        )
        for c in M.controls
    )
    out = messages(dataclasses.replace(M, controls=controls))
    assert "test_renamed_away" in out and "is not defined" in out


def test_a_missing_file_and_a_missing_symbol_and_a_non_test_are_reported(tmp_path):
    assert "no such file" in tm.resolve("code:nowhere/at/all.py#f")
    assert "missing a symbol" in tm.resolve("code:common/agent.py")
    assert "is not defined" in tm.resolve("code:common/agent.py#NoSuchThing")
    assert "must start with" in tm.resolve("common/agent.py#run_agent")
    assert "not a test function" in tm.resolve("test:common/agent.py::run_agent")
    assert (
        tm.resolve("code:common/agent.py#run_agent") is None
        and tm.resolve("code:common/tools.py#ToolRegistry.compact") is None
    )
    assert (
        tm.resolve(
            "test:tests/test_agent.py::test_stop_when_ends_the_run_without_another_model_call"
        )
        is None
    )


def test_resolve_uses_the_given_root_and_nested_symbols(tmp_path):
    (tmp_path / "m.py").write_text(
        textwrap.dedent("""
        class A:
            def method(self): ...
            class Inner:
                def deep(self): ...
        def top(): ...
        async def aio(): ...
        def test_x(): ...
    """)
    )
    for ref in (
        "code:m.py#A",
        "code:m.py#A.method",
        "code:m.py#A.Inner.deep",
        "code:m.py#top",
        "code:m.py#aio",
        "test:m.py::test_x",
    ):
        assert tm.resolve(ref, tmp_path) is None, ref
    assert tm.resolve("code:m.py#method", tmp_path) is not None, (
        "methods are addressed as Class.method"
    )


def test_a_control_without_a_test_or_without_evidence_is_reported():
    no_test = dataclasses.replace(M.controls[0], evidence=(M.controls[0].evidence[0],))
    assert "has no test among its evidence" in messages(
        dataclasses.replace(M, controls=(no_test, *M.controls[1:]))
    )
    empty = dataclasses.replace(M.controls[0], evidence=())
    assert "cites no evidence" in messages(
        dataclasses.replace(M, controls=(empty, *M.controls[1:]))
    )


def test_a_tool_added_to_the_code_without_a_threat_fails_the_model():
    without = dataclasses.replace(M, tools=tuple(t for t in M.tools if t != "fetch_page"))
    assert "the code defines tools the model does not list: ['fetch_page']" in messages(without)
    extra = dataclasses.replace(M, tools=(*M.tools, "delete_everything"))
    out = messages(extra)
    assert (
        "no longer defines: ['delete_everything']" in out
        and "'delete_everything' appears in no threat" in out
    )


def test_a_tool_that_no_threat_mentions_is_reported():
    threats = tuple(
        dataclasses.replace(t, surface=tuple(s for s in t.surface if s != "read_note"))
        for t in M.threats
    )
    assert "tool 'read_note' appears in no threat" in messages(
        dataclasses.replace(M, threats=threats)
    )
    threats = tuple(
        dataclasses.replace(t, surface=tuple(s for s in t.surface if s != "page_content"))
        for t in M.threats
    )
    assert "entry point 'page_content' appears in no threat" in messages(
        dataclasses.replace(M, threats=threats)
    )


@pytest.mark.parametrize(
    "tid,changes,expected",
    [
        ("T02", {"controls": ()}, "'mitigated' but cites no control"),
        ("T03", {"controls": ("C1",)}, "a 'gap' but cites controls"),
        ("T03", {"remaining": " "}, "says nothing about what remains"),
        ("T01", {"remaining": ""}, "says nothing about what remains"),
        ("T01", {"controls": ()}, "'partial' but cites no control"),
        ("T10", {"reason": ""}, "'accepted' without a reason"),
        ("T01", {"owasp": "LLM99"}, "unknown OWASP id"),
        ("T01", {"stride": "X"}, "unknown STRIDE letter"),
        ("T01", {"status": "fixed"}, "status must be one of"),
        ("T01", {"likelihood": 6}, "likelihood and impact are 1-5"),
        ("T01", {"impact": 0}, "likelihood and impact are 1-5"),
        ("T01", {"assets": ("A9",)}, "unknown asset"),
        ("T01", {"controls": ("C99",)}, "unknown control"),
    ],
)
def test_each_consistency_rule_fires(tid, changes, expected):
    assert expected in messages(replace_threat(M, tid, **changes))


def test_duplicates_and_orphan_controls_are_reported():
    dup = dataclasses.replace(M, threats=(*M.threats, M.threats[0]))
    assert "duplicate threat ids" in messages(dup)
    dupc = dataclasses.replace(M, controls=(*M.controls, M.controls[0]))
    assert "duplicate control ids" in messages(dupc)
    orphan = tm.Control("C99", "unused", (M.controls[0].evidence[0], M.controls[0].evidence[1]))
    assert "control C99 mitigates no threat" in messages(
        dataclasses.replace(M, controls=(*M.controls, orphan))
    )


def test_tools_in_source_reads_decorators_and_tool_name_sets(tmp_path):
    f = tmp_path / "x.py"
    f.write_text(
        textwrap.dedent("""
        from common.tools import tool
        NOTE_TOOLS = {"a_note", "b_note"}
        OTHER = {"not_a_tool"}
        DYNAMIC_TOOLS = set(range(3))
        @tool
        def plain(): ...
        @tool(name="x")
        def called(): ...
        def undecorated(): ...
    """)
    )
    assert tm.tools_in_source(f) == {"plain", "called", "a_note", "b_note"}


def test_the_owasp_list_and_stride_are_complete():
    assert sorted(tm.OWASP_LLM_2025) == [f"LLM{i:02d}" for i in range(1, 11)] and set(
        tm.STRIDE
    ) == set("STRIDE")
    assert {t.owasp for t in M.threats} >= {"LLM01", "LLM05", "LLM06", "LLM09", "LLM10"}


def test_equal_risks_list_gaps_before_mitigated_ones_even_when_the_id_sorts_later():
    a = tm.Threat("T1", "fine", "LLM01", "S", (), (), 2, 2, "mitigated", ("C",))
    b = tm.Threat("T9", "open", "LLM01", "S", (), (), 2, 2, "gap", (), remaining="all of it")
    c = tm.Threat("T5", "partly", "LLM01", "S", (), (), 2, 2, "partial", ("C",), remaining="some")
    d = tm.Threat("T2", "bigger", "LLM01", "S", (), (), 5, 5, "accepted", reason="r")
    model = tm.Model("m", "d", (), (), (), (), (), (a, b, c, d))
    assert [t.id for t in tm.ranked(model)] == ["T2", "T9", "T5", "T1"]
