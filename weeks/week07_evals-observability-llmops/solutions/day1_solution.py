"""Week 7 Day 1 - Solution: eval-driven development on the Week 6 support system, with a real (small) model.

The loop (each step is a function here, each output is a file in ``outputs/``):

  1. RUN       the baseline on the 50-case dataset (30 dev + 20 test). Results go to JSONL: nothing is thrown away.
  2. DIAGNOSE  every failure gets a root cause (``diagnose.py``); a Pareto table says what to fix first.
  3. FIX       the top causes, ONE change at a time (forced first tool call; triage examples).
  4. MEASURE   on DEV, with paired bootstrap intervals (same cases, so difficulty cancels out).
  5. CONFIRM   the winner on the held-out TEST split. Never tune on it.

  uv run python weeks/week07_evals-observability-llmops/solutions/day1_solution.py run baseline
  uv run python weeks/week07_evals-observability-llmops/solutions/day1_solution.py run smart
  uv run python weeks/week07_evals-observability-llmops/solutions/day1_solution.py report baseline smart
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import evalcases as ec  # noqa: E402
from diagnose import Pareto, diagnose, pareto  # noqa: E402
from support_system import evalset as w6  # noqa: E402

from common import agent_eval as ae  # noqa: E402
from common.evalkit import bootstrap_ci, fmt_ci, paired_bootstrap  # noqa: E402

OUT = ROOT / "outputs"

# the variants compared in the lesson: ONE change each, so a difference has one explanation
VARIANTS: dict[str, dict] = {
    "baseline": {},
    "always": {"force_first_tool": "always"},
    "smart": {"force_first_tool": "smart"},
    "fewshot": {"triage_few_shot": True},
    "smart+fewshot": {"force_first_tool": "smart", "triage_few_shot": True},
    "rules": {"triage_mode": "rules"},
    "rules+llm": {"triage_mode": "rules+llm"},
    "rules+smart": {"triage_mode": "rules", "force_first_tool": "smart"},
    "rules+prefetch": {"triage_mode": "rules", "prefetch_invoice": True},
}


@dataclass
class CaseRun:
    case_id: str
    kind: str
    split: str
    variant: str
    passed: bool
    failures: list[str]
    expected_route: str
    route: str | None
    calls: list[str]
    turns: int
    cause: str = ""
    evidence: str = ""
    transcript: list[tuple[str, str]] = field(default_factory=list)


def run_variant(
    cases,
    variant: str,
    *,
    provider: str | None,
    model: str | None = None,
    max_turns: int = 4,
    **system_kwargs,
) -> list[CaseRun]:
    """Run every case with a fresh world; keep the world long enough to read the route triage chose."""
    worlds: dict[str, w6.World] = {}
    make = w6.make_world_factory(provider, model, **system_kwargs)
    runs = []
    for case in cases:

        def capture(case=case):
            worlds[case.id] = make()
            return worlds[case.id]

        scenario = case.scenario
        scenario.state_factory = capture
        r = ae.run_scenario(scenario, w6.agent_factory, 0, max_turns)
        world = worlds[case.id]
        route_events = world.system.store.events(world.conversation_id, "route")
        route = route_events[0]["category"] if route_events else None
        calls = [c.name for c in r.calls]
        agent_text = " ".join(t for who, t in r.transcript if who == "agent")
        run = CaseRun(
            case.id,
            case.kind,
            case.split,
            variant,
            r.passed,
            r.failures,
            case.route,
            route,
            calls,
            r.turns,
            transcript=r.transcript,
        )
        if not r.passed:
            failed_events = world.system.store.events(world.conversation_id, "specialist_failed")
            d = diagnose(
                r.failures,
                kind=case.kind,
                expected_route=case.route,
                actual_route=route,
                calls=calls,
                agent_text=agent_text,
                error=r.error,
                specialist_failed=(
                    failed_events[0].get("status", "failed") if failed_events else ""
                ),
            )
            run.cause, run.evidence = d.cause, d.evidence
        runs.append(run)
        world._tmp.cleanup() if world._tmp else None
    return runs


def save(runs: list[CaseRun], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(asdict(r)) for r in runs) + "\n")


def load(path: Path) -> list[CaseRun]:
    return [
        CaseRun(
            **{**json.loads(line), "transcript": [tuple(t) for t in json.loads(line)["transcript"]]}
        )
        for line in path.read_text().splitlines()
        if line.strip()
    ]


# ----------------------------------------------------------------------------- analysis


def split_of(runs: list[CaseRun], split: str) -> list[CaseRun]:
    return [r for r in runs if r.split == split]


def pass_rate(runs: list[CaseRun]) -> str:
    return fmt_ci(bootstrap_ci([1.0 if r.passed else 0.0 for r in runs]))


def failure_pareto(runs: list[CaseRun]) -> Pareto:
    return pareto([(r.kind, r.cause) for r in runs if not r.passed], len(runs))


def by_kind(runs: list[CaseRun]) -> dict[str, tuple[int, int]]:
    out: dict[str, list[bool]] = {}
    for r in runs:
        out.setdefault(r.kind, []).append(r.passed)
    return {k: (sum(v), len(v)) for k, v in out.items()}


def compare(before: list[CaseRun], after: list[CaseRun]) -> dict:
    """Paired comparison on the SAME cases (matched by id). Positive diff = after is better."""
    a = {r.case_id: r for r in after}
    pairs = [(float(b.passed), float(a[b.case_id].passed)) for b in before if b.case_id in a]
    if not pairs:
        raise ValueError("no cases in common")
    res = paired_bootstrap([y for _, y in pairs], [x for x, _ in pairs])
    gained = sorted(
        b.case_id for b in before if b.case_id in a and not b.passed and a[b.case_id].passed
    )
    lost = sorted(
        b.case_id for b in before if b.case_id in a and b.passed and not a[b.case_id].passed
    )
    return {**res, "gained": gained, "lost": lost}


def report(names: list[str]) -> str:
    runs = {n: load(OUT / f"w7d1_{n}.jsonl") for n in names}
    lines = []
    base = runs[names[0]]
    lines.append(
        f"== {names[0]}: pass rate dev {pass_rate(split_of(base, 'dev'))}  test {pass_rate(split_of(base, 'test'))}"
    )
    lines.append("failure causes (DEV only: this is what we are allowed to look at):")
    lines.append(str(failure_pareto(split_of(base, "dev"))))
    for n in names[1:]:
        for split in ("dev", "test"):
            cmp = compare(split_of(base, split), split_of(runs[n], split))
            lines.append(
                f"== {n} vs {names[0]} on {split.upper()}: {pass_rate(split_of(runs[n], split))}  diff {cmp['diff']:+.0%} [{cmp['ci_low']:+.0%}, {cmp['ci_high']:+.0%}] wins/losses {cmp['wins']}/{cmp['losses']}"
            )
            if cmp["lost"]:
                lines.append(f"   REGRESSIONS: {cmp['lost']}")
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    if argv[:1] == ["report"]:
        print(report(argv[1:]))
        return
    if argv[:1] == ["run"] and len(argv) == 2 and argv[1] in VARIANTS:
        from common import llm
        from common.local_server import LocalOpenAIServer

        cases = ec.build_cases(lambda: None)
        with LocalOpenAIServer() as srv:
            os.environ["OLLAMA_BASE_URL"] = srv.url
            llm._ollama.cache_clear()
            runs = run_variant(
                cases, argv[1], provider="ollama", model="local-qwen", **VARIANTS[argv[1]]
            )
        save(runs, OUT / f"w7d1_{argv[1]}.jsonl")
        print(
            f"{argv[1]}: dev {pass_rate(split_of(runs, 'dev'))}  test {pass_rate(split_of(runs, 'test'))}"
        )
        return
    print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
