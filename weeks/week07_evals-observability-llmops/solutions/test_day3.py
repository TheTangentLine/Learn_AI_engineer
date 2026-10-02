"""Tests for Week 7 Day 3: the gate decides correctly in each situation, and the CI files are safe and consistent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

import day3_solution as d3  # noqa: E402
import evalcases as ec  # noqa: E402
import evalgate as eg  # noqa: E402
import scripted  # noqa: E402

from common.fake import fake_llm  # noqa: E402

HERE = Path(__file__).parent


def suite(passes: dict[str, float] | None = None, n=40, fail=(), name="x") -> eg.SuiteResult:
    """n cases in 4 segments of n/4 (a-01.., b-01.., ...), all passing except those in ``fail``."""
    segs = "abcd"
    ids = [f"{segs[i % 4]}-{i // 4 + 1:02d}" for i in range(n)]
    cases = {i: 0.0 if i in fail else 1.0 for i in ids}
    cases.update(passes or {})
    return eg.SuiteResult(name, cases)


# ----------------------------------------------------------------------------- the decision


def test_identical_runs_pass_with_zero_diff():
    d = eg.compare(suite(), suite())
    assert d.status == "pass" and d.diff == 0 and d.regressions == [] and d.exit_code == 0
    assert d.reasons == ["no significant change"]


def test_noise_within_the_interval_passes():
    # 1 case lost out of 40: -2.5%, interval includes zero, below the 5% warn threshold
    d = eg.compare(suite(), suite(fail={"a-01"}))
    assert d.status == "pass" and d.regressions == ["a-01"]
    assert d.ci[0] < 0 <= d.ci[1]


def test_significant_regression_fails_with_the_interval_in_the_reason():
    d = eg.compare(suite(), suite(fail={f"{s}-0{i}" for s in "ab" for i in range(1, 8)}))
    assert d.status == "fail" and d.exit_code == 1 and d.ci[1] < 0
    assert any("significant drop" in r for r in d.reasons)
    assert d.losses == 14 and d.wins == 0


def test_significant_improvement_is_reported_and_does_not_fail():
    base = suite(fail={f"{s}-0{i}" for s in "ab" for i in range(1, 8)})
    d = eg.compare(base, suite())
    assert d.status == "improved" and d.exit_code == 0 and d.ci[0] > 0
    assert len(d.gains) == 14 and d.regressions == []


def test_a_critical_regression_blocks_even_when_the_average_improves():
    """The reason averages are not enough: 8 gains hide one safety failure."""
    base = suite(fail={f"b-0{i}" for i in range(1, 9)})  # b was broken, a-01 was fine
    cur = suite(fail={"a-01"})  # fixes all of b, but breaks a-01
    d = eg.compare(base, cur, critical=["a-*"])
    assert d.diff > 0 and d.ci[0] > 0  # overall: a significant improvement
    assert d.status == "fail" and d.critical_regressions == ["a-01"]
    assert any("critical" in r for r in d.reasons)
    # without declaring it critical the same change is just an improvement
    assert eg.compare(base, cur).status == "improved"


def test_a_critical_case_that_was_already_failing_does_not_block():
    base = suite(fail={"a-01"})
    cur = suite(fail={"a-01"})
    assert eg.compare(base, cur, critical=["a-*"]).status == "pass"


def test_a_flaky_critical_case_blocks_only_when_it_was_perfect_before():
    base, cur = suite(), suite()
    base.cases["a-01"], cur.cases["a-01"] = 1.0, 0.8  # 5 trials, one failed now
    d = eg.compare(base, cur, critical=["a-*"])
    assert d.critical_regressions == ["a-01"] and d.status == "fail"
    base.cases["a-01"], cur.cases["a-01"] = (
        0.8,
        0.4,
    )  # it was already flaky and got worse: bad, but not a hard-floor violation
    assert eg.compare(base, cur, critical=["a-*"]).critical_regressions == []


def test_a_changed_case_set_fails_instead_of_comparing_apples_to_oranges():
    base = suite()
    cur = suite()
    cur.cases["z-99"] = 1.0
    del cur.cases["a-01"]
    d = eg.compare(base, cur)
    assert d.status == "fail" and "suite changed" in d.reasons[0]
    assert "z-99" in d.reasons[0] and "a-01" in d.reasons[0]
    ok = eg.compare(base, cur, allow_case_changes=True)
    assert ok.status in ("pass", "improved") and ok.n == 39  # only the common cases are compared


def test_large_point_drop_that_is_not_significant_warns_and_strict_fails():
    base = suite(n=20)
    cur = suite(n=20, fail={"a-01", "b-01"})  # -10%, 2 losses / 0 wins: interval touches zero
    warn = eg.compare(base, cur)
    assert warn.ci[1] >= 0 and warn.status == "warn" and warn.exit_code == 0
    assert any("not proven" in r for r in warn.reasons)
    assert eg.compare(base, cur, strict=True).status == "fail"
    assert eg.compare(base, cur, max_drop=0.2).status == "pass"  # threshold is configurable


def test_the_drop_threshold_is_inclusive():
    base = suite(n=20)
    cur = suite(n=20, fail={"a-01"})  # exactly -5%
    assert eg.compare(base, cur).status == "warn"


def test_a_segment_collapse_warns_even_when_the_average_barely_moves():
    """6 cases lost in one kind, 5 gained in another: net -2.5% overall, one segment destroyed."""
    base = suite(n=40, fail={f"b-{i:02d}" for i in range(1, 6)})  # b: 5/10 fail already
    cur_fail = {f"a-{i:02d}" for i in range(1, 8)}  # a: 7/10 lost
    cur = suite(n=40, fail=cur_fail)
    d = eg.compare(base, cur)
    assert "a" in d.segment_regressions and d.segment_regressions["a"] == (1.0, 0.3)
    assert d.status in ("warn", "fail")
    assert any("segment regression" in r or "significant" in r for r in d.reasons)


def test_segment_regression_isolated_from_the_overall_decision():
    base = suite(n=80, fail={f"b-{i:02d}" for i in range(1, 21)})  # b entirely failing before
    cur = suite(n=80, fail={f"a-{i:02d}" for i in range(1, 11)})  # a: 10 of 20 lost; b fully fixed
    d = eg.compare(base, cur)
    assert d.wins == 20 and d.losses == 10 and d.diff > 0  # net improvement, yet one segment halved
    assert d.segment_regressions == {"a": (1.0, 0.5)}
    assert d.status == "warn"
    assert eg.compare(base, cur, strict=True).status == "fail"
    assert eg.compare(base, cur, segment_drop=0.6).segment_regressions == {}


def test_tiny_segments_are_not_alarming():
    base = suite(n=8)  # segments of 2 cases: below min_segment_cases
    cur = suite(n=8, fail={"a-01", "a-02"})
    assert eg.compare(base, cur).segment_regressions == {}


def test_segment_of():
    assert eg.segment_of("billing-small-03") == "billing-small"
    assert eg.segment_of("human-01") == "human"
    assert eg.segment_of("standalone") == "standalone"
    assert eg.segment_of("x-y") == "x-y"  # trailing part not numeric


def test_is_critical_uses_glob_patterns():
    assert eg.is_critical("human-01", ["human-*"])
    assert not eg.is_critical("humane-01", ["human-*"])
    assert eg.is_critical("billing-pressure-02", ["off-*", "billing-pressure-*"])
    assert not eg.is_critical("anything", [])


def test_different_seeds_do_not_flip_a_clear_decision():
    base = suite(fail={f"{s}-0{i}" for s in "ab" for i in range(1, 8)})
    for seed in range(5):
        assert eg.compare(suite(), base, seed=seed).status == "fail"
        assert eg.compare(base, suite(), seed=seed).status == "improved"


# ----------------------------------------------------------------------------- reporting and persistence


def test_markdown_names_the_cases_and_the_numbers():
    base, cur = suite(), suite(fail={"a-01", "a-02", "a-03", "b-01"})
    cur.cases["c-01"] = 1.0
    md = eg.render_markdown(eg.compare(base, cur, critical=["a-0[12]"]), base, cur)
    assert "Eval gate" in md and "FAIL" in md
    assert "`a-01`" in md and "`a-03`" in md
    assert "Regressed (4)" in md and "CRITICAL" in md
    assert (
        "| pass rate | 100.0% | 90.0% |" in md
    )  # 36/40, but with c-01 already passing the rate is exact


def test_markdown_truncates_long_lists():
    base, cur = suite(n=80), suite(n=80, fail={f"{s}-{i:02d}" for s in "ab" for i in range(1, 11)})
    md = eg.render_markdown(eg.compare(base, cur), base, cur)
    assert "Regressed (20)" in md and md.count("`") < 100
    cur2 = suite(n=80, fail={f"{s}-{i:02d}" for s in "abc" for i in range(1, 11)})
    md2 = eg.render_markdown(eg.compare(base, cur2), base, cur2)
    assert "..." in md2


def test_save_load_round_trip(tmp_path):
    r = suite()
    r.meta = {"model": "m", "trials": 3}
    eg.save(r, tmp_path / "deep" / "baseline.json")
    got = eg.load(tmp_path / "deep" / "baseline.json")
    assert got.cases == r.cases and got.meta == r.meta and got.variant == r.variant
    text = (tmp_path / "deep" / "baseline.json").read_text()
    assert text.endswith("\n") and list(json.loads(text)["cases"]) == sorted(
        r.cases
    )  # stable diffs in git


# ----------------------------------------------------------------------------- baselines move only on purpose


def test_baseline_is_created_freely_but_replaced_only_deliberately(tmp_path):
    src, tgt = tmp_path / "cur.json", tmp_path / "evals" / "baseline.json"
    eg.save(suite(name="v1"), src)
    msg = eg.accept_baseline(src, tgt, force=False)
    assert "baseline written" in msg and tgt.exists()
    eg.save(suite(fail={"a-01"}, name="v2"), src)
    with pytest.raises(SystemExit, match="--force"):
        eg.accept_baseline(src, tgt, force=False)
    with pytest.raises(SystemExit, match="--reason"):
        eg.accept_baseline(src, tgt, force=True, reason="  ")
    assert eg.load(tgt).variant == "v1"  # untouched by the refusals
    eg.accept_baseline(src, tgt, force=True, reason="dropped the legacy refund case on purpose")
    got = eg.load(tgt)
    assert got.variant == "v2" and got.meta["accepted_because"].startswith("dropped")


def test_cli_exit_codes_and_summary_file(tmp_path):
    base, ok, bad = tmp_path / "b.json", tmp_path / "ok.json", tmp_path / "bad.json"
    eg.save(suite(), base)
    eg.save(suite(), ok)
    eg.save(suite(fail={f"{s}-0{i}" for s in "ab" for i in range(1, 8)}), bad)
    summary = tmp_path / "summary.md"
    args = ["compare", "--baseline", str(base), "--summary-file", str(summary)]
    assert eg.main([*args, "--current", str(ok)]) == 0
    assert eg.main([*args, "--current", str(bad)]) == 1
    text = summary.read_text()
    assert (
        text.count("Eval gate") == 2 and "PASS" in text and "FAIL" in text
    )  # appended, not overwritten
    crit = ["compare", "--baseline", str(base), "--current", str(bad), "--critical", "a-*, b-*"]
    assert eg.main(crit) == 1


def test_cli_baseline_command(tmp_path, capsys):
    src, tgt = tmp_path / "s.json", tmp_path / "t.json"
    eg.save(suite(), src)
    assert eg.main(["baseline", "--from", str(src), "--to", str(tgt)]) == 0
    assert "40 cases" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        eg.main(["baseline", "--from", str(src), "--to", str(tgt)])


# ----------------------------------------------------------------------------- a critical suite in plain pytest


@pytest.fixture(scope="module")
def critical_cases():
    cases = ec.build_cases(lambda: None)
    chosen = [c for c in cases if eg.is_critical(c.id, d3.CRITICAL)]
    assert len(chosen) >= 8
    return chosen


def run_cases(cases, *, system=None, **faults):
    """Run cases against a scripted model with the given faults; ``system`` holds SupportSystem options (guards=False ...)."""
    with fake_llm(scripted.rules(**faults)):
        return d3.d1.run_variant(cases, "t", provider="anthropic", **(system or {}))


def test_the_critical_cases_pass_for_the_good_system_as_a_pytest_assertion(critical_cases):
    """The pattern: the must-pass set is an ordinary test, so a failure blocks the PR with the case id and the reason."""
    runs = run_cases(critical_cases)
    failing = {r.case_id: r.failures for r in runs if not r.passed}
    assert not failing, failing


def test_the_gate_blocks_a_model_that_lies_once_the_runtime_guard_is_removed():
    cases = ec.build_cases(lambda: None)
    base = d3.suite_from_runs(run_cases(cases), "good")
    # the guard corrects the liar, so the end-to-end behaviour is unchanged: the gate must NOT fire (no false alarm)
    guarded = d3.suite_from_runs(run_cases(cases, liar=True), "liar+guard")
    assert eg.compare(base, guarded, critical=d3.CRITICAL).status == "pass"
    # a PR that disables the guard exposes the lie, and the pressure cases are critical
    unguarded = d3.suite_from_runs(run_cases(cases, liar=True, system={"guards": False}), "liar")
    d = eg.compare(base, unguarded, critical=d3.CRITICAL)
    assert d.status == "fail" and d.exit_code == 1
    assert d.critical_regressions and all(
        i.startswith("billing-pressure") for i in d.critical_regressions
    )


def test_a_bad_tool_policy_is_a_significant_regression_even_with_no_critical_cases_hit():
    cases = ec.build_cases(lambda: None)
    base = d3.suite_from_runs(run_cases(cases), "good")
    bad = d3.suite_from_runs(run_cases(cases, skip_lookup=True), "skip_lookup")
    d = eg.compare(base, bad, critical=d3.CRITICAL)
    assert d.status == "fail" and d.critical_regressions == [] and d.ci[1] < 0
    assert set(d.segment_regressions) == {
        "billing-small",
        "billing-unpaid",
    }  # both require the lookup
    assert eg.compare(base, d3.suite_from_runs(run_cases(cases), "again")).status == "pass"


def test_suite_from_runs_maps_pass_to_one_and_carries_metadata():
    runs = run_cases(ec.build_cases(lambda: None)[:3])
    s = d3.suite_from_runs(runs, "v", provider="p", trials=1)
    assert set(s.cases.values()) == {1.0} and len(s.cases) == 3
    assert s.meta == {"provider": "p", "trials": 1} and s.variant == "v"


# ----------------------------------------------------------------------------- the CI files


def workflow() -> dict:
    return yaml.safe_load((HERE / "ci" / "evals.yml.example").read_text())


def triggers(wf: dict) -> dict:
    return wf.get("on") or wf.get(True)  # YAML 1.1 parses the bare key `on` as True


def test_workflow_never_uses_pull_request_target():
    wf = workflow()
    assert "pull_request" in triggers(wf)
    assert "pull_request_target" not in triggers(wf), "fork code would run with this repo's secrets"


def test_workflow_has_least_privilege_and_timeouts():
    wf = workflow()
    assert wf["permissions"] == {"contents": "read"}
    assert wf["concurrency"]["cancel-in-progress"] is True
    for name, job in wf["jobs"].items():
        assert 0 < job["timeout-minutes"] <= 30, name


def test_secrets_are_only_exposed_to_the_job_that_needs_them_and_never_to_forks():
    wf = workflow()
    assert "env" not in wf, "no workflow-level secrets"
    assert "secrets." not in json.dumps(wf["jobs"]["unit"])
    ev = wf["jobs"]["eval"]
    assert "secrets.ANTHROPIC_API_KEY" in json.dumps(ev["env"])
    assert "head.repo.full_name == github.repository" in ev["if"]
    assert ev["needs"] == "unit"


def test_workflow_commands_exist_in_this_repo():
    """The example must not call a command nobody wrote."""
    text = (HERE / "ci" / "evals.yml.example").read_text()
    for needle in ("solutions/evalgate.py compare", "solutions/day3_solution.py run"):
        assert needle in text
    root = HERE.parents[2]
    assert (root / "weeks/week07_evals-observability-llmops/solutions/evalgate.py").exists()
    assert "--summary-file" in text and "GITHUB_STEP_SUMMARY" in text
    assert "evalgate run" not in text


def test_run_subcommand_arguments_are_accepted_by_the_real_cli_parser():
    with pytest.raises(SystemExit) as e:
        d3.main(["run", "--provider", "x", "--variant", "nonsense", "--out", "o"])
    assert e.value.code == 2


def test_promptfoo_config_is_valid_yaml_and_labelled_not_run():
    text = (HERE / "ci" / "promptfooconfig.yaml.example").read_text()
    assert "NOT RUN" in text.upper()
    cfg = yaml.safe_load(text)
    assert cfg["tests"] and cfg["providers"]


# ----------------------------------------------------------------------------- the decisions on the REAL Day 1 runs

OUT = HERE.parents[2] / "outputs"
needs_runs = pytest.mark.skipif(
    not (OUT / "w7d1_rules+prefetch.jsonl").exists(), reason="Day 1 run outputs not generated"
)


@needs_runs
@pytest.mark.parametrize(
    "variant,status",
    [
        ("rules+prefetch", "pass"),
        ("rules", "warn"),
        ("baseline", "fail"),
        ("fewshot", "fail"),
    ],
)
def test_real_runs_decisions(variant, status):
    base = d3.suite_from_day1("rules+prefetch")
    d = eg.compare(base, d3.suite_from_day1(variant), critical=d3.CRITICAL)
    assert d.status == status, d.reasons


@needs_runs
def test_real_baseline_run_trips_the_critical_case_and_the_significance_test():
    d = eg.compare(
        d3.suite_from_day1("rules+prefetch"), d3.suite_from_day1("baseline"), critical=d3.CRITICAL
    )
    assert "human-01" in d.critical_regressions and d.ci[1] < 0


# ----------------------------------------------------------------------------- interactions between the rules


def test_a_critical_failure_is_never_downgraded_by_a_later_softer_rule():
    base = suite(n=20)
    cur = suite(n=20, fail={"a-01", "b-01"})  # -10%: would be only a WARN on its own
    assert eg.compare(base, cur).status == "warn"
    d = eg.compare(base, cur, critical=["a-*"])
    assert d.status == "fail" and d.critical_regressions == ["a-01"]
    assert not any("not proven" in r for r in d.reasons), (
        "the soft rule must not run after a hard failure"
    )


def test_a_segment_of_exactly_the_minimum_size_is_monitored():
    base = suite(n=12)  # four segments of exactly 3 cases
    cur = suite(n=12, fail={"a-01", "a-02", "a-03"})
    assert eg.compare(base, cur).segment_regressions == {"a": (1.0, 0.0)}
    assert eg.compare(base, cur, min_segment_cases=4).segment_regressions == {}


def test_every_cli_threshold_reaches_the_gate(tmp_path):
    base, cur = tmp_path / "b.json", tmp_path / "c.json"
    eg.save(suite(n=20), base)
    eg.save(suite(n=20, fail={"a-01", "a-02", "b-01"}), cur)  # -15%, segment a: 2 of 5 lost (40%)

    def run(*extra):
        return eg.main(["compare", "--baseline", str(base), "--current", str(cur), *extra])

    assert run() == 0  # warn only
    assert run("--strict") == 1
    assert run("--max-drop", "0.5") == 0
    assert (
        run("--max-drop", "0.5", "--segment-drop", "0.3", "--strict") == 1
    )  # the segment rule now fires
    assert run("--max-drop", "0.5", "--segment-drop", "0.5", "--strict") == 0
    cur2 = tmp_path / "c2.json"
    r = suite(n=20)
    r.cases["z-01"] = 1.0
    eg.save(r, cur2)
    assert eg.main(["compare", "--baseline", str(base), "--current", str(cur2)]) == 1
    assert (
        eg.main(
            ["compare", "--baseline", str(base), "--current", str(cur2), "--allow-case-changes"]
        )
        == 0
    )
