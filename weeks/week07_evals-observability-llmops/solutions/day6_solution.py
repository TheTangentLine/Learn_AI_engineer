"""Week 7 Day 6 - Solution: feedback loops, an A/B harness, and drift detection.

There are no real users here, so the experiment is a REPLAY: users are simulated, but each user's outcome is looked up in the
REAL per-case results of the Day 5 runs (the local model, greedy: a variant either passes a case or it does not). That makes
the true effect of the change KNOWN (the difference of the two pass rates over the case population), so the harness itself can
be validated: does a 95% interval cover the truth 95% of the time? Does peeking really inflate false positives? How many
users until a decision?

  uv run python weeks/week07_evals-observability-llmops/solutions/day6_solution.py report
"""

from __future__ import annotations

import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import costlab as cl  # noqa: E402
import day5_solution as d5  # noqa: E402

from common import abtest as ab  # noqa: E402
from common import drift, tracing  # noqa: E402
from common.cache import jaccard, normalize  # noqa: E402

OUT = ROOT / "outputs"
CONTROL, TREATMENT = "base", "lean+hide+direct"  # Day 5's baseline and the variant it chose


# ----------------------------------------------------------------------------- real per-case outcomes


@dataclass(frozen=True)
class Outcome:
    passed: bool
    cost_usd: float  # simulated price card, real token counts
    latency_s: float  # assumed latency model, real token counts
    escalated: bool


def load_outcomes(variant: str, out: Path = OUT) -> dict[str, Outcome]:
    """case id -> what this variant did on that case (from the saved Day 5 runs and traces)."""
    runs, spans = d5.load_variant(variant, out)
    by_case = d5.per_case(spans)
    result = {}
    for r in runs:
        group = by_case[r.case_id]
        escalated = any(
            s.get(tracing.OPERATION) == "invoke_workflow"
            and s.get("app.reply.status") == "escalated"
            for s in group
        )
        result[r.case_id] = Outcome(
            r.passed, cl.trace_cost(group, cl.CHEAP), cl.trace_latency(group), escalated
        )
    return result


def true_effect(control: dict[str, Outcome], treat: dict[str, Outcome]) -> dict[str, float]:
    """The population-level truth the replay is sampling from: the mean over the 50 cases (every case equally likely)."""
    n = len(control)
    return {
        "pass": sum(treat[c].passed for c in control) / n
        - sum(control[c].passed for c in control) / n,
        "cost": sum(treat[c].cost_usd for c in control) / n
        - sum(control[c].cost_usd for c in control) / n,
        "latency": sum(treat[c].latency_s for c in control) / n
        - sum(control[c].latency_s for c in control) / n,
        "escalated": sum(treat[c].escalated for c in control) / n
        - sum(control[c].escalated for c in control) / n,
    }


# ----------------------------------------------------------------------------- a replayed experiment


def replay(
    outcomes: dict[str, dict[str, Outcome]],
    n_users: int,
    *,
    experiment: str = "exp",
    weights: dict[str, float] | None = None,
    seed: int = 0,
    drop_escalated_from: str | None = None,
) -> list[dict]:
    """Simulate ``n_users`` users: each asks one question (a case drawn uniformly) and is assigned an arm by hash.
    ``drop_escalated_from`` injects a LOGGING BUG: escalated conversations of that arm never reach the analysis."""
    weights = weights or {arm: 1.0 for arm in outcomes}
    case_ids = sorted(next(iter(outcomes.values())))
    rng = random.Random(seed)
    rows = []
    for i in range(n_users):
        case = rng.choice(case_ids)
        arm = ab.assign(f"{experiment}-{seed}", f"user{i}", weights)
        o = outcomes[arm][case]
        if drop_escalated_from == arm and o.escalated:
            continue
        rows.append(
            {
                "user": i,
                "arm": arm,
                "case": case,
                "passed": o.passed,
                "cost": o.cost_usd,
                "latency": o.latency_s,
                "escalated": o.escalated,
            }
        )
    return rows


def analyse(rows: list[dict], control: str = CONTROL, treat: str = TREATMENT) -> dict:
    a = [r for r in rows if r["arm"] == control]
    b = [r for r in rows if r["arm"] == treat]
    primary = ab.compare_rates(
        sum(r["passed"] for r in a), len(a), sum(r["passed"] for r in b), len(b)
    )
    guard = {
        "escalated": ab.compare_rates(
            sum(r["escalated"] for r in a), len(a), sum(r["escalated"] for r in b), len(b)
        ),
        "cost": ab.bootstrap_diff([r["cost"] for r in a], [r["cost"] for r in b], n_boot=1000),
        "latency": ab.bootstrap_diff(
            [r["latency"] for r in a], [r["latency"] for r in b], n_boot=1000
        ),
    }
    return {
        "primary": primary,
        "guardrails": guard,
        "decision": ab.decide(primary, guard, tolerance={"escalated": 0.02}),
    }


def coverage_and_power(outcomes, n_users: int, n_experiments: int = 400) -> dict:
    """Repeat the experiment many times with different users: how often does the interval contain the truth, and how often
    does the test detect the (known) effect?"""
    control, treat = outcomes[CONTROL], outcomes[TREATMENT]
    truth = true_effect(control, treat)["pass"]
    below = above = detected = 0
    for k in range(n_experiments):
        rows = replay(outcomes, n_users, experiment="cov", seed=k)
        a = [r["passed"] for r in rows if r["arm"] == CONTROL]
        b = [r["passed"] for r in rows if r["arm"] == TREATMENT]
        r = ab.compare_rates(sum(a), len(a), sum(b), len(b))
        below += truth < r["ci"][0]  # the interval sits entirely above the truth
        above += truth > r["ci"][1]  # the interval sits entirely below the truth
        detected += r["p"] < 0.05
    return {
        "truth": truth,
        "coverage": 1 - (below + above) / n_experiments,
        "missed_low": below / n_experiments,
        "missed_high": above / n_experiments,
        "power": detected / n_experiments,
        "n_per_arm": n_users // 2,
    }


# ----------------------------------------------------------------------------- peeking, on the same outcomes


def stopping_rules(
    outcomes, n_per_look: int, looks: int, n_experiments: int, *, null: bool, seed0: int = 0
) -> dict:
    """Run many replayed experiments with ``looks`` interim analyses and report, per stopping rule, how often it declares a
    winner and how many users it used on average. ``null=True`` makes it an A/A test: both arms get the CONTROL outcomes."""
    arms = {
        CONTROL: outcomes[CONTROL],
        TREATMENT: outcomes[CONTROL] if null else outcomes[TREATMENT],
    }
    pocock, obf = ab.GroupSequential(looks, kind="pocock"), ab.GroupSequential(looks, kind="obf")
    crit = 1.959964
    wins = {"fixed horizon": 0, "peek, p<0.05 at any look": 0, "pocock": 0, "obf": 0}
    used = {"pocock": 0, "obf": 0}
    for k in range(n_experiments):
        rows = replay(arms, n_per_look * looks * 2, experiment="stop", seed=seed0 + k)
        zs = []
        for look in range(1, looks + 1):
            seen = rows[: n_per_look * look * 2]
            a = [r["passed"] for r in seen if r["arm"] == CONTROL]
            b = [r["passed"] for r in seen if r["arm"] == TREATMENT]
            zs.append(ab.compare_rates(sum(a), len(a), sum(b), len(b))["z"])
        wins["fixed horizon"] += abs(zs[-1]) > crit
        wins["peek, p<0.05 at any look"] += any(abs(z) > crit for z in zs)
        for name, gs in (("pocock", pocock), ("obf", obf)):
            stop = next((i for i, z in enumerate(zs, 1) if gs.check(i, z).startswith("stop")), None)
            wins[name] += stop is not None
            used[name] += (stop or looks) * n_per_look * 2
    out = {k: v / n_experiments for k, v in wins.items()}
    out["avg_users_pocock"] = used["pocock"] / n_experiments
    out["avg_users_obf"] = used["obf"] / n_experiments
    out["max_users"] = n_per_look * looks * 2
    return out


# ----------------------------------------------------------------------------- drift


def route_mix(messages_by_route: dict[str, int]) -> dict[str, int]:
    return dict(messages_by_route)


def embedding_windows(window: int, share_new: float, *, seed: int):
    """Two windows of real question embeddings (bge-small): the OLD window is mostly how-to questions; the NEW one has
    ``share_new`` billing messages mixed in. The mix is simulated; the embeddings are real."""
    import cachelab as cb
    import evalcases as ec

    from common.embed import get_embedder

    emb = get_embedder("local")
    howto = sorted({t for p in cb.SAME for t in (p.a, p.b)})
    billing = sorted(
        {
            c.scenario.user_factory().opening
            for c in ec.build_cases(lambda: None)
            if c.kind.startswith("billing")
        }
    )
    rng = random.Random(seed)

    def draw(share: float):
        texts = [rng.choice(billing if rng.random() < share else howto) for _ in range(window)]
        return emb.embed_documents(texts)

    return draw(0.1), draw(share_new)


def cusum_stream(
    p_before: float, p_after: float, change_at: int, length: int, seed: int
) -> list[int]:
    rng = np.random.default_rng(seed)
    p = np.where(np.arange(length) < change_at, p_before, p_after)
    return (rng.random(length) < p).astype(int).tolist()


def first_alarm(stream: list[int], cusum: drift.BernoulliCusum) -> int | None:
    for i, x in enumerate(stream, 1):
        if cusum.update(x):
            return i
    return None


# ----------------------------------------------------------------------------- feedback into evaluation


def implicit_signals(history: list[dict], *, reask_similarity: float = 0.7) -> dict:
    """Signals a conversation emits without anyone clicking: the user repeating themselves (the answer did not land), saying
    thanks (it did), and how many turns it took. Heuristics from the stored history; validate against explicit feedback."""
    users = [
        m["content"]
        for m in history
        if m.get("role") == "user" and isinstance(m.get("content"), str)
    ]
    reasks = sum(
        1
        for prev, cur in zip(users, users[1:], strict=False)
        if jaccard(prev, cur) >= reask_similarity or normalize(prev) == normalize(cur)
    )
    thanks = bool(users) and bool(
        re.search(r"\b(thanks|thank you|thx|perfect|great)\b", users[-1], re.I)
    )
    return {"turns": len(users), "reasks": reasks, "thanks": thanks}


def cases_from_feedback(system, *, redact=lambda t: t) -> list[dict]:
    """Thumbs-down conversations as CANDIDATE eval cases: the opening message, the reply, the reason and the trace link.
    They need a human label before they join the dataset (the customer said it was bad, not what good would be). Duplicate
    openings are merged; text goes through ``redact`` (the comment too)."""
    seen: dict[str, dict] = {}
    for ev in system.store.events(kind="feedback"):
        if ev["rating"] != -1:
            continue
        _, history = system.store.get(ev["conversation_id"])
        users = [m["content"] for m in history if m.get("role") == "user"]
        assistant = [
            m["content"]
            for m in history
            if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"]
        ]
        if not users:
            continue
        key = normalize(users[0])
        row = seen.setdefault(
            key,
            {
                "opening": redact(users[0]),
                "last_reply": redact(assistant[-1]) if assistant else "",
                "reasons": [],
                "comments": [],
                "trace_ids": [],
                "conversations": [],
                "needs_label": True,
            },
        )
        row["reasons"].append(ev["reason"])
        if ev.get("comment"):
            row["comments"].append(redact(ev["comment"]))
        if ev.get("trace_id"):
            row["trace_ids"].append(ev["trace_id"])
        row["conversations"].append(ev["conversation_id"])
    return sorted(seen.values(), key=lambda r: (-len(r["conversations"]), r["opening"]))


# ----------------------------------------------------------------------------- the report


def report() -> str:
    outcomes = {CONTROL: load_outcomes(CONTROL), TREATMENT: load_outcomes(TREATMENT)}
    truth = true_effect(outcomes[CONTROL], outcomes[TREATMENT])
    lines = [
        f"Replay of Day 5's {CONTROL} vs {TREATMENT}: simulated users, real per-case outcomes (50-case population)",
        f"True effect in this population: pass {truth['pass']:+.1%}, cost ${truth['cost'] * 1000:+.3f} per 1k conversations, "
        f"latency {truth['latency']:+.2f}s, escalated {truth['escalated']:+.1%}",
        "",
    ]
    need = ab.sample_size(sum(o.passed for o in outcomes[CONTROL].values()) / 50, truth["pass"])
    lines.append(
        f"Planned sample size for 80% power at the true effect: {need} users per arm ({2 * need} users in total)"
    )
    for n in (200, 2 * need, 1200):
        cp = coverage_and_power(outcomes, n)
        lines.append(
            f"  {n:>5} users ({cp['n_per_arm']}/arm): the 95% interval contained the truth {cp['coverage']:.1%} of 400 replays; the test detected the effect {cp['power']:.1%}"
        )
    lines.append("")
    rows = replay(outcomes, 2 * need, experiment="live", seed=123)
    res = analyse(rows)
    counts = {a: sum(1 for r in rows if r["arm"] == a) for a in outcomes}
    srm = ab.srm_check(counts, {a: 1 for a in outcomes})
    p = res["primary"]
    lines.append(
        f"One experiment of {len(rows)} users: arms {counts}, SRM p={srm.p:.3f} ({'ok' if srm.ok else 'BROKEN'})"
    )
    lines.append(
        f"  pass {p['control_rate']:.1%} -> {p['treat_rate']:.1%}, diff {p['diff']:+.1%} [{p['ci'][0]:+.1%}, {p['ci'][1]:+.1%}], p={p['p']:.4f}"
    )
    for name, g in res["guardrails"].items():
        scale = 1000 if name == "cost" else 1
        lines.append(
            f"  guardrail {name}: diff {g['diff'] * scale:+.3f} [{g['ci'][0] * scale:+.3f}, {g['ci'][1] * scale:+.3f}]"
            + ("  (USD per 1k)" if name == "cost" else "")
        )
    lines.append(f"  decision: {res['decision']}")
    for n_bug in (2 * need, 10_000):
        broken = replay(outcomes, n_bug, experiment="live", seed=123, drop_escalated_from=TREATMENT)
        bc = {a: sum(1 for r in broken if r["arm"] == a) for a in outcomes}
        bs = ab.srm_check(bc, {a: 1 for a in outcomes})
        bp = analyse(broken)["primary"]
        lines.append(
            f"Logging bug (treatment's escalated conversations never logged), {n_bug} users: arms {bc}, SRM p={bs.p:.2g} "
            f"({'ok' if bs.ok else 'BROKEN'}); the pass-rate diff reads {bp['diff']:+.1%} (true {truth['pass']:+.1%})"
        )
    lines.append("")
    lines.append("Peeking (10 looks of 100 users per arm each):")
    for null, label in ((True, "A/A (no real difference)"), (False, "real effect")):
        r = stopping_rules(outcomes, 100, 10, 1000, null=null)
        lines.append(
            f"  {label:<26} "
            + ", ".join(
                f"{k} {v:.1%}"
                for k, v in r.items()
                if k in ("fixed horizon", "peek, p<0.05 at any look", "pocock", "obf")
            )
            + f";  avg users used: pocock {r['avg_users_pocock']:.0f}, obf {r['avg_users_obf']:.0f} of {r['max_users']}"
        )
    return "\n".join(lines)


def detection_rate(
    p_old: dict[str, float],
    p_new: dict[str, float],
    n: int,
    *,
    trials: int = 300,
    alpha: float = 0.01,
    seed: int = 0,
) -> float:
    """How often a chi-square test between two simulated windows of ``n`` messages flags the change in the route mix."""
    rng = np.random.default_rng(seed)
    keys = sorted(p_old)
    flagged = 0
    for _ in range(trials):
        a = dict(zip(keys, rng.multinomial(n, [p_old[k] for k in keys]), strict=True))
        b = dict(zip(keys, rng.multinomial(n, [p_new[k] for k in keys]), strict=True))
        flagged += drift.chi2_homogeneity(a, b)["p"] < alpha
    return flagged / trials


def drift_report() -> str:
    lines = []
    old = {"billing": 28, "technical": 14, "human": 4, "other": 4}  # the Day 1 dataset's route mix
    new = {
        "billing": 28,
        "technical": 30,
        "human": 4,
        "other": 12,
    }  # SIMULATED post-launch mix: more how-to and off-topic
    value = drift.psi(old, new)
    lines.append(
        f"Route mix, old vs a SIMULATED post-launch mix: PSI {value:.3f} ({drift.psi_band(value)})"
    )
    po = {k: v / sum(old.values()) for k, v in old.items()}
    pn = {k: v / sum(new.values()) for k, v in new.items()}
    lines.append(
        "  how often a chi-square test (alpha 0.01) between two windows of n messages notices it: "
        + ", ".join(f"n={n}: {detection_rate(po, pn, n):.0%}" for n in (50, 200, 1000))
    )
    lines.append(
        "  and with NO change (false alarms, same test): "
        + ", ".join(f"n={n}: {detection_rate(po, po, n, seed=1):.1%}" for n in (50, 200, 1000))
    )
    lines.append("")
    for share, label in (
        (0.1, "same mix (10% billing)"),
        (0.25, "25% billing"),
        (0.4, "40% billing"),
    ):
        ps = []
        for seed in range(5):
            a, b = embedding_windows(80, share, seed=seed)
            ps.append(drift.centroid_shift_test(a, b, n_perm=500, seed=seed)["p"])
        lines.append(
            f"Embedding centroid test (bge-small, windows of 80; old window 10% billing), new window {label}: p-values over 5 draws {[round(float(p), 3) for p in ps]}"
        )
    lines.append("")
    p0, n_streams = 0.10, 300
    for arl0 in (500, 2000):
        for p1 in (0.13, 0.16, 0.20):
            h = drift.calibrate_threshold(p0, p1, arl0, n_sims=1500, seed=1)
            delay = float(
                np.mean(
                    [
                        first_alarm(
                            cusum_stream(p0, p1, 0, 4000, s), drift.BernoulliCusum(p0, p1, h)
                        )
                        or 4000
                        for s in range(n_streams)
                    ]
                )
            )
            false = float(
                np.mean(
                    [
                        first_alarm(
                            cusum_stream(p0, p0, 0, 6 * arl0, 1000 + s),
                            drift.BernoulliCusum(p0, p1, h),
                        )
                        or 6 * arl0
                        for s in range(n_streams)
                    ]
                )
            )
            lines.append(
                f"CUSUM thumbs-down rate {p0:.0%} -> {p1:.0%}, threshold {h:.2f} for a false alarm about every {arl0}: average false-alarm run {false:.0f} (censored at {6 * arl0}), average delay after a real change {delay:.0f} observations"
            )
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    if argv[:1] == ["drift"]:
        print(drift_report())
    elif argv[:1] == ["report"]:
        print(report())
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
