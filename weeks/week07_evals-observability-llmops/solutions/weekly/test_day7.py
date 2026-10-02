"""Tests for the Week 7 weekly challenge: the runner, the combined gate, the dashboard, the monitor and its incident drills."""

from __future__ import annotations

import copy
import html.parser
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("opentelemetry.sdk")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))  # the llmops package
sys.path.insert(0, str(HERE.parent))  # the Week 7 solutions: evalcases, scripted, day5_solution ...

import day5_solution as d5  # noqa: E402
import evalcases as ec  # noqa: E402
import scripted  # noqa: E402
import traceeval as te  # noqa: E402
from llmops import __main__ as cli  # noqa: E402
from llmops import dashboard, drill, gate, legacy, monitor, runner  # noqa: E402

from common import tracing  # noqa: E402
from common.fake import fake_llm  # noqa: E402

BASE_OPTS = d5.VARIANTS["base"]
CRITICAL = ["billing-pressure-*", "human-*", "off-topic-*"]


def scripted_run(name="base", options=None, *, cases=None, trials=1, budget=None, **faults):
    with fake_llm(scripted.rules(**faults)):
        return runner.run_suite(
            name,
            options or BASE_OPTS,
            provider="anthropic",
            provider_label="scripted",
            real_money=False,
            cases=cases,
            trials=trials,
            budget_usd=budget,
            clock=lambda: 1_700_000_000.0,
        )


@pytest.fixture(scope="module")
def base_run():
    return scripted_run("base")


@pytest.fixture(scope="module")
def broken_run():
    return scripted_run("broken", skip_lookup=True)


def clone(run: runner.Run, **manifest_updates) -> runner.Run:
    return runner.Run(
        {**copy.deepcopy(run.manifest), **manifest_updates},
        copy.deepcopy(run.results),
        copy.deepcopy(run.cases),
        run.spans,
    )


def scale(run: runner.Run, key: str, factor=None, ids=None, add=None) -> runner.Run:
    out = clone(run)
    for row in out.cases:
        if ids is None or row["id"] in ids:
            row[key] = row[key] * factor if factor is not None else row[key] + add
    return out


# ----------------------------------------------------------------------------- the runner


def test_a_scripted_run_has_a_complete_manifest_rows_and_spans(base_run):
    m = base_run.manifest
    assert (
        base_run.complete
        and m["n_cases"] == m["n_done"] == 50
        and m["trials"] == 1
        and m["provider"] == "scripted"
    )
    assert (
        m["created_at"] == 1_700_000_000.0
        and m["spend_usd"] > 0
        and m["aborted_reason"] == ""
        and m["version"] == 1
    )
    assert m["price_card"]["name"] == "claude-haiku-4-5" and m["latency_model"]["base_s"] == 0.25
    assert len(base_run.cases) == 50 and set(base_run.results.cases) == {
        r["id"] for r in base_run.cases
    }
    assert all(v in (0.0, 1.0) for v in base_run.results.cases.values())
    roots = [s for s in base_run.spans if s.name == "eval.case"]
    assert len(roots) == 50 and {s.get("app.eval.trial") for s in roots} == {0}
    row = next(r for r in base_run.cases if r["id"] == "billing-small-00")
    assert row["kind"] == "billing-small" and row["split"] in ("dev", "test") and row["trials"] == 1
    assert (
        row["model_calls"] >= 1
        and row["input_tokens"] > 0
        and row["cost_usd"] > 0
        and row["latency_s"] > 0
    )


def test_case_rows_agree_with_the_spans_they_summarise(base_run):
    by_case = d5.per_case(base_run.spans)
    for row in base_run.cases:
        spans = by_case[row["id"]]
        assert row["input_tokens"] == sum(
            s.get(tracing.INPUT_TOKENS, 0) for s in spans if s.get(tracing.OPERATION) == "chat"
        )
        assert row["output_tokens"] == sum(
            s.get(tracing.OUTPUT_TOKENS, 0) for s in spans if s.get(tracing.OPERATION) == "chat"
        )
    assert base_run.manifest["spend_usd"] == pytest.approx(
        sum(r["cost_usd"] for r in base_run.cases)
    )


def test_the_dataset_hash_is_stable_and_changes_when_a_case_is_reworded():
    cases = ec.build_cases(lambda: None)
    h = runner.dataset_hash(cases)
    assert h == runner.dataset_hash(ec.build_cases(lambda: None)) and len(h) == 16
    assert runner.dataset_hash(cases[:-1]) != h, "a missing case"
    reworded = ec.build_cases(lambda: None)
    sc = reworded[0].scenario
    original = sc.user_factory
    user = original()
    user.opening = user.opening + " please"
    sc.user_factory = lambda: user
    assert runner.dataset_hash(reworded) != h, (
        "same ids, different words: the id-set check of Day 3 would not notice"
    )


def test_trials_aggregate_into_fractions_failures_and_per_trial_spans(monkeypatch):
    cases = ec.build_cases(lambda: None)[:3]
    sequence = {c.id: [True, False, True] for c in cases[:1]}
    sequence[cases[1].id] = [False, False, False]
    sequence[cases[2].id] = [True, True, True]
    seen = {c.id: 0 for c in cases}

    def fake_run(case_list, name, **kw):
        (case,) = case_list
        i = seen[case.id]
        seen[case.id] += 1
        ok = sequence[case.id][i]
        return [
            runner.d1.CaseRun(
                case.id,
                case.kind,
                case.split,
                name,
                ok,
                [] if ok else [f"missing_call:x{i}"],
                case.route,
                case.route,
                [],
                1,
            )
        ]

    monkeypatch.setattr(runner.d1, "run_variant", fake_run)
    run = runner.run_suite(
        "t", {}, provider="x", real_money=False, cases=cases, trials=3, clock=lambda: 0.0
    )
    assert [round(run.results.cases[c.id], 3) for c in cases] == [0.667, 0.0, 1.0]
    row = run.cases[1]
    assert row["pass_fraction"] == 0.0 and row["failures"] == [
        "missing_call:x0",
        "missing_call:x1",
        "missing_call:x2",
    ]
    assert run.cases[0]["failures"] == ["missing_call:x1"] and run.cases[2]["failures"] == []
    roots = [s for s in run.spans if s.name == "eval.case"]
    assert len(roots) == 9 and sorted({s.get("app.eval.trial") for s in roots}) == [0, 1, 2]
    assert run.complete and run.manifest["trials"] == 3


def test_a_trials_of_zero_is_an_error():
    with pytest.raises(ValueError, match="trials"):
        runner.run_suite("t", {}, provider="x", real_money=False, trials=0)


def test_the_spend_cap_stops_the_run_and_marks_it_incomplete():
    run = scripted_run("capped", budget=0.002)
    m = run.manifest
    assert not run.complete and "budget" in m["aborted_reason"] and m["n_done"] < m["n_cases"]
    assert len(run.cases) == m["n_done"] == len(run.results.cases)
    assert m["spend_usd"] > 0.002 and m["budget_usd"] == 0.002
    generous = scripted_run("ok", budget=100.0)
    assert generous.complete and generous.manifest["spend_usd"] < 100.0


def test_a_hosted_provider_must_have_a_budget_and_the_guess_can_be_overridden():
    for provider in ("anthropic", "openai"):
        with pytest.raises(ValueError, match="budget"):
            runner.run_suite("t", {}, provider=provider, cases=[])
    with pytest.raises(ValueError, match="budget"):
        runner.run_suite("t", {}, provider="ollama", real_money=True, cases=[])
    run = runner.run_suite("t", {}, provider="anthropic", real_money=False, cases=[])
    assert (
        run.complete is False or run.manifest["n_cases"] == 0
    )  # an empty case list: nothing to do, nothing spent
    assert (
        runner.run_suite("t", {}, provider="anthropic", budget_usd=1.0, cases=[]).manifest[
            "budget_usd"
        ]
        == 1.0
    )


def test_run_directories_round_trip(base_run, tmp_path):
    d = runner.save_run(base_run, tmp_path / "nested" / "base")
    assert {p.name for p in d.iterdir()} == {
        "manifest.json",
        "results.json",
        "cases.jsonl",
        "spans.jsonl",
    }
    loaded = runner.load_run(d)
    assert (
        loaded.manifest == base_run.manifest
        and loaded.results.cases == base_run.results.cases
        and loaded.cases == base_run.cases
    )
    assert len(loaded.spans) == len(base_run.spans) and loaded.spans[0] == base_run.spans[0]
    assert runner.load_run(d, spans=False).spans == []
    assert (d / "manifest.json").read_text().endswith("\n") and list(
        json.loads((d / "manifest.json").read_text())
    ) == sorted(base_run.manifest)


def test_loading_something_that_is_not_a_run_or_an_unknown_version_fails_clearly(
    base_run, tmp_path
):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        runner.load_run(tmp_path)
    d = runner.save_run(base_run, tmp_path / "r")
    manifest = json.loads((d / "manifest.json").read_text())
    manifest["version"] = 99
    (d / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="version"):
        runner.load_run(d)


def test_promote_creates_a_baseline_and_replaces_one_only_deliberately(
    base_run, broken_run, tmp_path
):
    target = tmp_path / "evals" / "baseline"
    runner.save_run(base_run, tmp_path / "a")
    runner.save_run(broken_run, tmp_path / "b")
    assert "baseline base written" in runner.promote(tmp_path / "a", target)
    with pytest.raises(FileExistsError, match="force"):
        runner.promote(tmp_path / "b", target)
    with pytest.raises(ValueError, match="reason"):
        runner.promote(tmp_path / "b", target, force=True, reason="   ")
    assert runner.load_run(target).name == "base", "the refusals changed nothing"
    runner.promote(tmp_path / "b", target, force=True, reason="accepted the new refund policy")
    got = runner.load_run(target)
    assert (
        got.name == "broken"
        and got.manifest["accepted_because"] == "accepted the new refund policy"
    )


def test_an_incomplete_run_can_never_become_the_baseline(tmp_path):
    run = scripted_run("capped", budget=0.002)
    runner.save_run(run, tmp_path / "x")
    with pytest.raises(ValueError, match="incomplete"):
        runner.promote(tmp_path / "x", tmp_path / "baseline")


# ----------------------------------------------------------------------------- the gate


def test_identical_runs_pass_and_the_report_is_readable(base_run):
    same = scripted_run("same")
    report = gate.evaluate(base_run, same, gate.Policy(critical=CRITICAL))
    assert report.status == "pass" and report.exit_code == 0
    assert [c.name for c in report.checks] == [
        "completeness",
        "dataset",
        "price card",
        "quality",
        "cost",
        "latency",
    ]
    md = report.markdown()
    assert (
        "LLMOps gate: **PASS**" in md
        and "`base` (baseline) vs `same` (candidate)" in md
        and "cost per 1k conversations" in md
    )


def test_a_broken_variant_fails_on_quality_even_though_it_is_cheaper(base_run, broken_run):
    report = gate.evaluate(base_run, broken_run, gate.Policy(critical=CRITICAL))
    by = {c.name: c for c in report.checks}
    assert report.status == "fail" and report.exit_code == 1
    assert by["quality"].status == "fail" and by["cost"].status == "improved", (
        "a cost cut does not excuse a quality loss"
    )
    assert (
        report.quality.segment_regressions
        and "billing-unpaid" in report.quality.segment_regressions
    )
    assert "billing-unpaid" in report.markdown()


def test_an_incomplete_candidate_is_refused_before_anything_else(base_run):
    capped = scripted_run("capped", budget=0.002)
    report = gate.evaluate(base_run, capped)
    assert report.status == "fail" and [c.name for c in report.checks] == ["completeness"]
    assert "incomplete" in report.checks[0].detail and "budget" in report.checks[0].detail
    assert (
        gate.evaluate(capped, base_run)
        .checks[0]
        .detail.startswith("the baseline run is incomplete")
    )


def test_a_changed_dataset_fails_unless_explicitly_allowed(base_run):
    smaller = scripted_run("small", cases=ec.build_cases(lambda: None)[:40])
    report = gate.evaluate(base_run, smaller)
    assert (
        report.status == "fail"
        and report.checks[-1].name == "dataset"
        and "cases changed" in report.checks[-1].detail
    )
    allowed = gate.evaluate(base_run, smaller, gate.Policy(allow_dataset_change=True))
    assert any(c.name == "quality" for c in allowed.checks)


def test_different_price_cards_make_costs_incomparable(base_run):
    other = clone(base_run, price_card={**base_run.manifest["price_card"], "input": 9.0})
    report = gate.evaluate(base_run, other)
    assert (
        report.status == "fail"
        and report.checks[-1].name == "price card"
        and "not comparable" in report.checks[-1].detail
    )


def test_a_different_model_is_a_warning_not_a_failure(base_run):
    other = clone(base_run, model="another-model")
    report = gate.evaluate(base_run, other)
    pc = next(c for c in report.checks if c.name == "price card")
    assert pc.status == "warn" and "model" in pc.detail and report.status == "warn"


@pytest.mark.parametrize(
    "factor,expected",
    [(0.7, "improved"), (1.0, "pass"), (1.05, "pass"), (1.10, "pass"), (1.30, "fail")],
)
def test_cost_regression_thresholds_with_a_clear_signal(base_run, factor, expected):
    cand = scale(base_run, "cost_usd", factor)
    check = next(c for c in gate.evaluate(base_run, cand).checks if c.name == "cost")
    assert check.status == expected, check.detail


def test_a_large_but_noisy_cost_increase_warns_and_a_clear_one_fails(base_run):
    only_one = scale(
        base_run,
        "cost_usd",
        ids={"billing-small-00"},
        add=sum(r["cost_usd"] for r in base_run.cases) * 0.30,
    )
    check = next(c for c in gate.evaluate(base_run, only_one).checks if c.name == "cost")
    assert check.status == "warn" and "includes zero" in check.detail, (
        "+30% of the total from ONE case: not proven"
    )
    uniform = scale(base_run, "cost_usd", 1.30)
    assert (
        next(c for c in gate.evaluate(base_run, uniform).checks if c.name == "cost").status
        == "fail"
    )
    custom = gate.evaluate(base_run, uniform, gate.Policy(max_cost_increase=0.5))
    assert next(c for c in custom.checks if c.name == "cost").status == "pass"


def test_latency_regression_thresholds(base_run):
    slow = scale(base_run, "latency_s", 1.4)
    check = next(c for c in gate.evaluate(base_run, slow).checks if c.name == "latency")
    assert check.status == "fail" and "ASSUMED" in check.detail
    fast = next(
        c
        for c in gate.evaluate(base_run, scale(base_run, "latency_s", 0.8)).checks
        if c.name == "latency"
    )
    assert fast.status == "improved"
    ok = next(
        c
        for c in gate.evaluate(base_run, scale(base_run, "latency_s", 1.2)).checks
        if c.name == "latency"
    )
    assert ok.status == "pass"
    tolerant = gate.evaluate(base_run, slow, gate.Policy(max_latency_increase=0.5))
    assert next(c for c in tolerant.checks if c.name == "latency").status == "pass"


def test_a_zero_cost_baseline_does_not_divide_by_zero(base_run):
    free = scale(base_run, "cost_usd", 0.0)
    check = next(c for c in gate.evaluate(free, base_run).checks if c.name == "cost")
    assert check.status == "pass" and "zero" in check.detail


def test_overall_status_is_the_worst_check_and_improved_only_when_nothing_is_worse():

    def mk(*statuses):
        return gate.GateReport(
            [gate.Check(f"c{i}", x, "") for i, x in enumerate(statuses)], None, "a", "b"
        )

    assert mk("pass", "pass").status == "pass" and mk("pass", "improved").status == "improved"
    assert mk("improved", "warn").status == "warn" and mk("pass", "warn", "fail").status == "fail"
    assert mk().status == "pass" and mk("fail").exit_code == 1 and mk("warn").exit_code == 0


def test_the_critical_policy_is_passed_through_to_the_quality_check(base_run):
    cand = clone(base_run)
    cand.results.cases["human-00"] = 0.0
    with_critical = gate.evaluate(base_run, cand, gate.Policy(critical=["human-*"]))
    assert next(c for c in with_critical.checks if c.name == "quality").status == "fail"
    assert with_critical.quality.critical_regressions == ["human-00"]
    assert (
        next(c for c in gate.evaluate(base_run, cand).checks if c.name == "quality").status
        == "pass"
    )


def test_markdown_table_cells_cannot_break_the_table(base_run):
    report = gate.GateReport([gate.Check("x", "warn", "a | b")], None, "a", "b")
    assert "a / b" in report.markdown()


# ----------------------------------------------------------------------------- the dashboard


class Tags(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags: list[str] = []
        self.attrs: list[tuple[str, dict]] = []
        self.stack: list[str] = []
        self.errors: list[str] = []

    VOID = {"meta", "br", "line", "circle", "polyline", "input", "img"}

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attrs.append((tag, dict(attrs)))
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.tags.append(tag)
        self.attrs.append((tag, dict(attrs)))

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> with stack {self.stack[-3:]}")
        else:
            self.stack.pop()


def parse(page: str) -> Tags:
    p = Tags()
    p.feed(page)
    p.close()
    return p


def test_the_dashboard_is_well_formed_offline_and_script_free(base_run, broken_run):
    page = dashboard.render([base_run, broken_run])
    p = parse(page)
    assert p.errors == [] and p.stack == [], (p.errors, p.stack)
    assert (
        "script" not in p.tags
        and "iframe" not in p.tags
        and "link" not in p.tags
        and "img" not in p.tags
    )
    assert (
        "http://" not in page
        and "https://" not in page
        and "src=" not in page
        and "@import" not in page
    )
    for tag, attrs in p.attrs:
        if tag == "svg":
            assert attrs.get("role") == "img" and attrs.get("aria-label")
    assert page.startswith("<!doctype html>") and "<title>" in page and 'lang="en"' in page
    assert "simulated" in page and "assumed model" in page


def test_every_number_on_the_page_comes_from_the_artifacts(base_run, broken_run):
    page = dashboard.render([base_run, broken_run])
    for run in (base_run, broken_run):
        s = dashboard.run_stats(run)
        assert (
            f"{s['pass_rate']:.1%}" in page
            and f"${s['cost_per_1k']:.3f}" in page
            and f"{s['calls']:.2f}" in page
        )
    base, broken = dashboard.run_stats(base_run), dashboard.run_stats(broken_run)
    assert f"({broken['cost_per_1k'] / base['cost_per_1k'] - 1:+.0%})" in page, (
        "cost change against the baseline"
    )
    assert base["pass_rate"] == pytest.approx(sum(r["pass_fraction"] for r in base_run.cases) / 50)
    assert base["cost_per_1k"] == pytest.approx(
        sum(r["cost_usd"] for r in base_run.cases) / 50 * 1000
    )
    assert "billing-unpaid" in page and ">0%<" in page, "the segment the broken run lost"


def test_run_names_notes_and_kinds_are_escaped_everywhere(base_run):
    nasty = '<script>alert("x")</script>'
    run = clone(
        base_run, name=nasty, variant='"><img src=x onerror=alert(1)>', note="<b>n</b> & more"
    )
    run.cases[0]["kind"] = nasty
    page = dashboard.render([base_run, run], title=nasty)
    assert nasty not in page and "<img src=x" not in page and "<b>n</b>" not in page
    assert (
        "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in page
        and "&lt;b&gt;n&lt;/b&gt; &amp; more" in page
    )
    assert parse(page).errors == []


def test_an_incomplete_run_is_flagged_on_the_page():
    capped = scripted_run("capped", budget=0.002)
    page = dashboard.render([capped])
    assert "NO: incomplete" in page and "capped" in page


def test_the_baseline_defaults_to_the_first_run_and_an_unknown_name_falls_back(
    base_run, broken_run
):
    first = dashboard.render([base_run, broken_run])
    assert "Baseline: <b>base</b>" in first
    assert "Baseline: <b>broken</b>" in dashboard.render([base_run, broken_run], baseline="broken")
    assert "Baseline: <b>base</b>" in dashboard.render([base_run, broken_run], baseline="nope")
    with pytest.raises(ValueError):
        dashboard.render([])


def test_a_single_run_and_runs_without_spans_still_render(base_run):
    only = dashboard.render([base_run])
    assert parse(only).errors == [] and "baseline 52%" not in only
    bare = runner.Run(copy.deepcopy(base_run.manifest), base_run.results, base_run.cases, [])
    page = dashboard.render([bare])
    assert "No spans in this run." in page and "no spans" in page and parse(page).errors == []


def test_pareto_frontier_by_hand():
    pts = [
        (1.0, 0.5, "a"),
        (0.6, 0.5, "b"),
        (0.6, 0.7, "c"),
        (0.9, 0.9, "d"),
        (0.5, 0.4, "e"),
        (0.9, 0.9, "dup"),
    ]
    assert [n for _, _, n in dashboard.pareto(pts)] == ["e", "c", "d", "dup"], (
        "a and b are dominated by c; identical points both stay"
    )
    assert (
        dashboard.pareto([(1.0, 0.5, "only")]) == [(1.0, 0.5, "only")]
        and dashboard.pareto([]) == []
    )


def test_the_heatmap_colour_follows_the_rate():
    assert dashboard._cell(None) == '<td class="na">–</td>'
    assert (
        "hsl(0 " in dashboard._cell(0.0)
        and "hsl(120 " in dashboard._cell(1.0)
        and "hsl(60 " in dashboard._cell(0.5)
    )


def test_the_scatter_labels_the_frontier_and_the_baseline(base_run, broken_run):
    svg = dashboard._scatter(
        [dashboard.run_stats(base_run), dashboard.run_stats(broken_run)], "base"
    )
    assert (
        'class="pt base"' in svg
        and "base: $" in svg
        and "<title>" in svg
        and "polyline" in svg
        or "front" in svg
    )


# ----------------------------------------------------------------------------- the monitor and its drills


SCENARIOS = {
    "reference": ([drill.Segment(300)], 1),
    "healthy": ([drill.Segment(500)], 2),
    "skip_lookup": ([drill.Segment(300), drill.Segment(300, faults={"skip_lookup": True})], 2),
    "mix_shift": ([drill.Segment(300), drill.Segment(400, mix=drill.MIX_TECH_HEAVY)], 2),
    "loop": ([drill.Segment(300), drill.Segment(300, faults={"loop": True})], 2),
}
_CACHE: dict[str, list] = {}


def spans_of(name: str):
    """Simulating traffic is the slow part: each scenario is simulated once per test session."""
    if name not in _CACHE:
        segments, seed = SCENARIOS[name]
        _CACHE[name] = drill.simulate(segments, seed=seed)
    return _CACHE[name]


@pytest.fixture(scope="module")
def reference():
    return monitor.turns_from(spans_of("reference"))


def alerts_for(name, reference, *, arl0=1000):
    prod = monitor.turns_from(spans_of(name))
    return monitor.monitor(prod, reference, monitor.MonitorConfig(arl0=arl0, n_sims=300)), prod


def test_turns_describe_each_trace_without_reading_content(reference):
    assert len(reference) == 300 and [t.start_ns for t in reference] == sorted(
        t.start_ns for t in reference
    )
    assert {t.route for t in reference} <= {"billing", "tech", "human", "triage"}
    assert (
        all(t.cost >= 0 for t in reference)
        and any(t.escalated for t in reference)
        and any(t.cost > 0 for t in reference)
    )
    assert all(isinstance(t.defect_codes, tuple) for t in reference)


def test_turns_from_ignores_traces_without_a_workflow_span():
    stray = tracing.SpanRecord("t", "s", None, "eval.case", 0, 1, "unset", "", {})
    assert monitor.turns_from([stray]) == []


def test_a_healthy_period_raises_no_alert(reference):
    report, prod = alerts_for("healthy", reference)
    assert report.alerts == [] and report.turns == len(prod) == 500


def test_a_model_fault_is_noticed_within_a_few_dozen_turns(reference):
    report, _ = alerts_for("skip_lookup", reference)
    alert = report.first("defect_rate")
    assert alert is not None and 301 <= alert.at_turn <= 345, [
        (a.kind, a.at_turn) for a in report.alerts
    ]
    assert (
        "refund_without_lookup" in alert.evidence
        and "CUSUM" in alert.evidence
        and len(alert.trace_id) == 32
    )
    assert report.first("route_drift") is None


def test_a_change_in_what_customers_ask_is_a_route_drift_and_nothing_else(reference):
    report, _ = alerts_for("mix_shift", reference)
    kinds = [a.kind for a in report.alerts]
    assert kinds == ["route_drift"], kinds
    assert (
        300 < report.alerts[0].at_turn <= 500
        and "chi-square" in report.alerts[0].evidence
        and "tech" in report.alerts[0].evidence
    )


def test_a_looping_model_trips_the_defect_escalation_and_cost_alarms(reference):
    report, _ = alerts_for("loop", reference)
    kinds = {a.kind for a in report.alerts}
    assert {"defect_rate", "escalation_rate", "cost_spike"} <= kinds
    cost = report.first("cost_spike")
    assert "+" in cost.evidence and "tolerance +25%" in cost.evidence
    assert [a.at_turn for a in report.alerts] == sorted(a.at_turn for a in report.alerts)


def test_the_alert_threshold_follows_the_requested_false_alarm_interval(reference):
    prod = monitor.turns_from(spans_of("skip_lookup"))

    def at(arl0):
        return monitor.monitor(prod, reference, monitor.MonitorConfig(arl0=arl0, n_sims=300)).first(
            "defect_rate"
        )

    tight, loose, quiet = at(100), at(1000), at(4_000)

    def threshold(alert):
        return float(alert.evidence.split("crossed ")[1].split(" ")[0])

    assert threshold(tight) < threshold(loose) < threshold(quiet)
    assert tight.at_turn <= loose.at_turn <= quiet.at_turn


def test_a_reference_with_no_defects_still_gives_a_usable_baseline():
    clean_ref = monitor.turns_from(
        drill.simulate([drill.Segment(200, mix={"off-topic": 1, "human": 1})], seed=3)
    )
    assert monitor._rate(clean_ref, "defect") == 0.0
    prod = monitor.turns_from(
        drill.simulate(
            [
                drill.Segment(100, mix={"off-topic": 1, "human": 1}),
                drill.Segment(200, mix={"billing-small": 1}, faults={"skip_lookup": True}),
            ],
            seed=4,
        )
    )
    report = monitor.monitor(prod, clean_ref, monitor.MonitorConfig(arl0=500, n_sims=300))
    assert report.first("defect_rate") is not None and report.reference_rates["defect"] == 0.0


def test_a_tiny_reference_is_refused(reference):
    with pytest.raises(ValueError, match="at least 50"):
        monitor.monitor(reference, reference[:10])


def test_the_drill_is_deterministic_and_respects_the_mix():
    a = drill.simulate([drill.Segment(40)], seed=9)
    b = drill.simulate([drill.Segment(40)], seed=9)
    assert [(s.name, s.attributes.get("app.reply.status")) for s in a] == [
        (s.name, s.attributes.get("app.reply.status")) for s in b
    ]
    only_human = monitor.turns_from(drill.simulate([drill.Segment(30, mix={"human": 1})], seed=1))
    assert {t.route for t in only_human} == {"human"} and all(t.escalated for t in only_human)
    pool = drill.openings_by_kind()
    assert set(pool) >= set(drill.MIX_NORMAL) and all(pool[k] for k in drill.MIX_NORMAL)


# ----------------------------------------------------------------------------- the command line


def test_cli_run_gate_dashboard_baseline_and_monitor_end_to_end(tmp_path, capsys):
    base, broken = tmp_path / "base", tmp_path / "broken"
    assert cli.main(["run", "--provider", "scripted", "--variant", "base", "--out", str(base)]) == 0
    assert (
        cli.main(
            [
                "run",
                "--provider",
                "scripted",
                "--variant",
                "base",
                "--fault",
                "skip_lookup",
                "--name",
                "broken",
                "--out",
                str(broken),
            ]
        )
        == 0
    )
    assert "complete" in capsys.readouterr().out
    summary = tmp_path / "summary.md"
    assert (
        cli.main(
            [
                "gate",
                "--baseline",
                str(base),
                "--candidate",
                str(base),
                "--summary-file",
                str(summary),
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "gate",
                "--baseline",
                str(base),
                "--candidate",
                str(broken),
                "--summary-file",
                str(summary),
            ]
        )
        == 1
    )
    text = summary.read_text()
    assert text.count("LLMOps gate") == 2 and "PASS" in text and "FAIL" in text
    page = tmp_path / "dash.html"
    assert (
        cli.main(
            [
                "dashboard",
                "--runs",
                str(base),
                str(broken),
                "--baseline",
                "base",
                "--out",
                str(page),
            ]
        )
        == 0
    )
    assert parse(page.read_text()).errors == []
    assert cli.main(["baseline", "--from", str(broken), "--to", str(base)]) == 1, (
        "refuses to replace without --force"
    )
    assert "refused" in capsys.readouterr().err
    assert (
        cli.main(
            [
                "baseline",
                "--from",
                str(broken),
                "--to",
                str(base),
                "--force",
                "--reason",
                "new policy",
            ]
        )
        == 0
    )
    assert runner.load_run(base).manifest["accepted_because"] == "new policy"


def test_cli_monitor_exit_codes(tmp_path, capsys):
    for name, key in (("ref", "reference"), ("ok", "healthy"), ("bad", "skip_lookup")):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(tracing.asdict(s)) + "\n" for s in spans_of(key))
        )
    args = ["--reference", str(tmp_path / "ref.jsonl"), "--arl0", "1000", "--sims", "300"]
    assert cli.main(["monitor", "--spans", str(tmp_path / "ok.jsonl"), *args]) == 0
    assert "no alerts" in capsys.readouterr().out
    assert cli.main(["monitor", "--spans", str(tmp_path / "bad.jsonl"), *args]) == 1
    assert "ALERT defect_rate" in capsys.readouterr().out


def test_cli_refuses_unsafe_combinations(tmp_path):
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "--provider", "anthropic", "--out", str(tmp_path / "x")])
    assert e.value.code == 2, "a hosted provider without a budget"
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "--provider", "local", "--fault", "liar", "--out", str(tmp_path / "x")])
    assert e.value.code == 2, "--fault is for the scripted provider only"


def test_cli_run_that_hits_its_budget_exits_nonzero_and_the_gate_then_refuses(tmp_path, capsys):
    base, capped = tmp_path / "base", tmp_path / "capped"
    assert cli.main(["run", "--provider", "scripted", "--variant", "base", "--out", str(base)]) == 0
    assert (
        cli.main(
            [
                "run",
                "--provider",
                "scripted",
                "--variant",
                "base",
                "--budget",
                "0.002",
                "--name",
                "capped",
                "--out",
                str(capped),
            ]
        )
        == 2
    )
    assert "INCOMPLETE" in capsys.readouterr().out
    assert cli.main(["gate", "--baseline", str(base), "--candidate", str(capped)]) == 1


# ----------------------------------------------------------------------------- the CI workflow


def test_the_llmops_workflow_is_safe_and_calls_commands_that_exist():
    path = HERE.parent / "ci" / "llmops.yml.example"
    wf = yaml.safe_load(path.read_text())
    on = wf.get("on") or wf.get(True)
    assert "pull_request" in on and "pull_request_target" not in on and "schedule" in on
    assert (
        wf["permissions"] == {"contents": "read"}
        and wf["concurrency"]["cancel-in-progress"] is True
    )
    assert "env" not in wf, "no workflow-level secrets"
    for name, job in wf["jobs"].items():
        assert 0 < job["timeout-minutes"] <= 30, name
    assert "secrets." not in json.dumps(wf["jobs"]["unit"])
    gate_job = wf["jobs"]["gate"]
    assert "head.repo.full_name == github.repository" in gate_job["if"]
    steps = "\n".join(s.get("run", "") for s in gate_job["steps"])
    assert "--budget" in steps, "the live run has a spend cap"
    for sub in ("run", "gate", "dashboard"):
        assert f"python -m llmops {sub}" in steps
    assert (HERE / "llmops" / "__main__.py").exists()


# ----------------------------------------------------------------------------- the real Day 5 runs

REAL = d5.OUT / "w7d5_base.jsonl"
needs_real = pytest.mark.skipif(
    not all((d5.OUT / f"w7d5_{n}.jsonl").exists() for n in d5.VARIANTS),
    reason="Day 5 real-model runs not generated",
)


@needs_real
def test_the_day5_runs_import_into_run_directories_and_pass_the_gate_with_the_known_numbers():
    base, final = legacy.import_day5("base"), legacy.import_day5("lean+hide+direct")
    assert (
        base.complete
        and base.manifest["dataset_hash"] == final.manifest["dataset_hash"]
        and base.manifest["provider"] == "local"
    )
    report = gate.evaluate(base, final, gate.Policy(critical=CRITICAL))
    by = {c.name: c for c in report.checks}
    assert (
        report.status == "improved"
        and by["quality"].status == "improved"
        and by["cost"].status == "improved"
    )
    cost_check = by["cost"].detail
    assert "-43" in cost_check or "-44" in cost_check, cost_check
    assert report.quality.critical_regressions == []


@needs_real
def test_the_real_dashboard_shows_all_eight_variants_and_is_well_formed():
    runs = [legacy.import_day5(v) for v in d5.VARIANTS]
    page = dashboard.render(runs, baseline="base")
    assert parse(page).errors == [] and all(
        f"<th>{v}</th>" in page.replace("+", "+") for v in d5.VARIANTS
    )
    assert "lean+hide+direct" in page and "$0.619" in page and "$1.097" in page
    front = [
        n
        for _, _, n in dashboard.pareto(
            [(s["cost_per_1k"], s["pass_rate"], s["name"]) for s in map(dashboard.run_stats, runs)]
        )
    ]
    assert "lean+hide+direct" in front and "lean+direct" in front and "base" not in front


def test_the_quality_thresholds_of_the_policy_reach_the_quality_check(base_run):
    """-6 points over 50 cases: a large drop whose interval includes zero. warn by default; the policy decides the rest."""
    cand = clone(base_run)
    for cid in [c for c, v in cand.results.cases.items() if v == 1.0][:3]:
        cand.results.cases[cid] = 0.0

    def quality(**kw):
        return next(
            c
            for c in gate.evaluate(base_run, cand, gate.Policy(**kw)).checks
            if c.name == "quality"
        )

    assert quality().status == "warn" and "not proven" in quality().detail
    assert quality(strict=True).status == "fail"
    assert quality(max_drop=0.2).status == "pass"


def test_allowing_a_dataset_change_lets_the_quality_check_compare_the_common_cases(base_run):
    smaller = scripted_run("small", cases=ec.build_cases(lambda: None)[:40])
    refused = next(
        c
        for c in gate.evaluate(base_run, smaller, gate.Policy(allow_dataset_change=False)).checks
        if c.name == "dataset"
    )
    assert refused.status == "fail"
    quality = next(
        c
        for c in gate.evaluate(base_run, smaller, gate.Policy(allow_dataset_change=True)).checks
        if c.name == "quality"
    )
    assert "suite changed" not in quality.detail and quality.status in ("pass", "improved", "warn")


@pytest.mark.parametrize(
    "factor,tolerance", [(1.25, 0.25), (1.1, 0.1), (1.2, 0.2), (1.3, 0.3), (1.7, 0.7), (1.15, 0.15)]
)
def test_a_latency_exactly_at_the_tolerance_passes(base_run, factor, tolerance):
    """x * 1.1 / x - 1 is 0.10000000000000009 in floating point: a tolerance is inclusive, so the check must absorb that."""
    at_limit = scale(base_run, "latency_s", factor)
    check = next(
        c
        for c in gate.evaluate(
            base_run, at_limit, gate.Policy(max_latency_increase=tolerance)
        ).checks
        if c.name == "latency"
    )
    assert check.status == "pass", check.detail


def test_a_latency_just_over_the_default_tolerance_fails(base_run):
    over = next(
        c
        for c in gate.evaluate(base_run, scale(base_run, "latency_s", 1.26)).checks
        if c.name == "latency"
    )
    assert over.status == "fail"


def test_the_cards_describe_the_latest_run_and_its_percentiles(base_run, broken_run):
    page = dashboard.render([base_run, broken_run])
    latest = dashboard.run_stats(broken_run)
    assert (
        f"pass rate of broken (baseline {dashboard.run_stats(base_run)['pass_rate']:.0%})" in page
    )
    assert f"<b>{latest['p95']:.2f}s</b>" in page
    s = dashboard.run_stats(base_run)
    lat = [r["latency_s"] for r in base_run.cases]
    assert (
        s["p95"] == te.percentile(lat, 95)
        and s["p50"] == te.percentile(lat, 50)
        and s["p95"] > s["p50"]
    )


def test_a_tiny_bar_is_still_visible_and_the_tool_table_lists_the_busiest_tool_first(base_run):
    assert "width:0.5%" in dashboard._bars([("big", 1000.0), ("tiny", 0.1)], " tok", "t")
    page = dashboard.render([base_run])
    names = [
        m for m in re.findall(r"<tr><th scope='row'>([a-z_]+)</th><td>(\d+)</td><td>\d+</td>", page)
    ]
    calls = [int(c) for _, c in names]
    assert len(names) >= 3 and calls == sorted(calls, reverse=True)


# ----------------------------------------------------------------------------- the monitor on hand-built turns


def turn(i, *, route="billing", defect=False, escalated=False, cost=1.0):
    return monitor.Turn(
        f"{i:032x}", i, route, defect, ("tool_loop",) if defect else (), escalated, cost
    )


def reference_turns(n=100, **kw):
    return [turn(i, **kw) for i in range(n)]


def test_a_cost_increase_is_reported_only_when_it_is_large_AND_clearly_real():
    ref = reference_turns(100, cost=1.0)
    cfg = monitor.MonitorConfig(window=200, n_sims=200, cost_increase=0.25)
    steady_small = [turn(i, cost=1.1) for i in range(200)]  # +10%, certain: below the tolerance
    assert monitor.monitor(steady_small, ref, cfg).first("cost_spike") is None
    steady_big = [turn(i, cost=1.5) for i in range(200)]
    alert = monitor.monitor(steady_big, ref, cfg).first("cost_spike")
    assert alert is not None and "+50%" in alert.evidence
    noisy = monitor.MonitorConfig(window=10, n_sims=200, cost_increase=0.25)
    one_big = [turn(i, cost=0.0) for i in range(9)] + [
        turn(9, cost=20.0)
    ]  # mean +100%, but resamples often miss the outlier
    assert monitor.monitor(one_big, ref, noisy).first("cost_spike") is None, "large but not proven"


def test_alerts_are_reported_in_the_order_they_fired_not_the_order_they_were_checked():
    ref = reference_turns(100, route="billing")
    prod = [turn(i, route="tech") for i in range(200)] + [
        turn(200 + i, route="tech", defect=True) for i in range(150)
    ]
    report = monitor.monitor(prod, ref, monitor.MonitorConfig(window=200, n_sims=300, arl0=300))
    kinds = [(a.kind, a.at_turn) for a in report.alerts]
    assert {k for k, _ in kinds} >= {"route_drift", "defect_rate"} and [
        t for _, t in kinds
    ] == sorted(t for _, t in kinds), kinds
    assert report.alerts[0].kind == "route_drift" and report.alerts[0].at_turn == 200


def test_the_evidence_quotes_the_rate_over_the_last_hundred_turns():
    ref = reference_turns(100)
    prod = reference_turns(300) + [turn(300 + i, defect=(i % 2 == 0)) for i in range(60)]
    report = monitor.monitor(prod, ref, monitor.MonitorConfig(n_sims=300, arl0=300, window=10_000))
    alert = report.first("defect_rate")
    assert alert is not None
    last = prod[max(0, alert.at_turn - 100) : alert.at_turn]
    expected = sum(t.defect for t in last) / len(last)
    assert f"last 100 turns {expected:.1%}" in alert.evidence and "tool_loop" in alert.evidence


def test_the_defect_alert_depends_on_how_many_times_the_reference_rate_counts_as_a_change():
    ref = [turn(i, defect=(i % 20 == 0)) for i in range(200)]  # 5% defects
    prod = [turn(i, defect=(i % 6 == 0)) for i in range(400)]  # ~17%
    low = monitor.monitor(
        prod, ref, monitor.MonitorConfig(rate_shift=1.5, n_sims=300, arl0=300, window=10_000)
    ).first("defect_rate")
    high = monitor.monitor(
        prod, ref, monitor.MonitorConfig(rate_shift=4.0, n_sims=300, arl0=300, window=10_000)
    ).first("defect_rate")
    assert low is not None and high is not None and low.evidence != high.evidence
    with pytest.raises(ValueError, match="differ"):
        monitor.monitor(prod, ref, monitor.MonitorConfig(rate_shift=1.0, n_sims=100, window=10_000))


def test_turns_are_sorted_by_time_and_use_the_first_workflow_span_of_a_trace():
    def workflow(trace, span, start, agent):
        return tracing.SpanRecord(
            trace,
            span,
            "root" + trace,
            "invoke_workflow support",
            start,
            start + 5,
            "unset",
            "",
            {
                tracing.OPERATION: "invoke_workflow",
                "app.reply.agent": agent,
                "app.reply.status": "ok",
            },
        )

    spans = [
        workflow("b" * 32, "s2", 200, "tech"),
        workflow("a" * 32, "s1", 50, "billing"),
        workflow("a" * 32, "s3", 90, "human"),
    ]
    turns = monitor.turns_from(spans)
    assert [t.trace_id[0] for t in turns] == ["a", "b"], "time order"
    assert turns[0].route == "billing", "a trace with two turns is described by its first"


def test_changing_only_the_split_of_a_case_changes_the_dataset_hash():
    cases = ec.build_cases(lambda: None)
    h = runner.dataset_hash(cases)
    moved = ec.build_cases(lambda: None)
    moved[0] = copy.copy(moved[0])
    moved[0].split = "test" if moved[0].split == "dev" else "dev"
    assert runner.dataset_hash(moved) != h


def test_traffic_that_all_goes_to_one_route_in_both_periods_is_not_a_route_drift_and_does_not_crash():
    ref = reference_turns(100, route="billing")
    prod = reference_turns(400, route="billing")
    report = monitor.monitor(prod, ref, monitor.MonitorConfig(window=200, n_sims=200))
    assert report.first("route_drift") is None
    shifted = [turn(i, route="tech") for i in range(400)]
    assert (
        monitor.monitor(shifted, ref, monitor.MonitorConfig(window=200, n_sims=200)).first(
            "route_drift"
        )
        is not None
    )


def test_the_scatter_axis_labels_match_the_plotted_range(base_run, broken_run):
    stats = [dashboard.run_stats(base_run), dashboard.run_stats(broken_run)]
    svg = dashboard._scatter(stats, "base")
    labels = [float(x) / 100 for x in re.findall(r'text-anchor="end">(\d+)%', svg)]
    ys = [s["pass_rate"] for s in stats]
    assert len(labels) == 3 and labels == sorted(labels)
    assert labels[0] <= min(ys) + 0.005 and labels[-1] >= max(ys) - 0.005, (
        "the axis spans the data (it is zoomed, not 0 to 100%)"
    )
