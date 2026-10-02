"""Watch production traces and raise alerts. The online counterpart of the gate: the gate stops a bad CHANGE, the monitor
notices a bad WEEK (a provider changed the model, traffic shifted, a dependency broke).

    reference = traces_from(spans_of_last_good_period)
    alerts = monitor(production_spans, reference, MonitorConfig(arl0=500))

One conversation turn = one trace (``invoke_workflow support``). Per trace we derive, WITHOUT reading any content:
  defect      a Day 4 trace defect (refund without lookup, tool loop, specialist without a tool, agent failure ...)
  escalated   the turn ended in a hand-off to a human
  route       which agent handled it (billing / tech / human / none), for input drift
  cost        simulated cost from the token counts
Alerts: a CUSUM per rate (calibrated to a false-alarm interval by simulation), a chi-square on the route mix per window,
and a cost-per-turn comparison per window. Every alert says where it fired and shows its evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import costlab as cl
import traceeval as te

from common import abtest, drift, tracing
from common.tracing import SpanRecord


@dataclass(frozen=True)
class Turn:
    trace_id: str
    start_ns: int
    route: str
    defect: bool
    defect_codes: tuple[str, ...]
    escalated: bool
    cost: float


@dataclass
class MonitorConfig:
    arl0: float = 500.0  # tolerated false-alarm interval, in turns
    rate_shift: float = 2.0  # alert when a rate becomes this many times the reference rate (the CUSUM's p1 = p0 x shift)
    window: int = 200  # turns per window for the route-mix and cost checks
    route_alpha: float = 0.001
    cost_increase: float = 0.25  # relative increase in mean cost per turn that counts
    seed: int = 0
    n_sims: int = 1500


@dataclass
class Alert:
    kind: str  # defect_rate | escalation_rate | route_drift | cost_spike
    at_turn: int  # 1-based index of the turn (in time order) at which it fired
    trace_id: str
    evidence: str


@dataclass
class MonitorReport:
    alerts: list[Alert]
    turns: int
    reference_rates: dict[str, float] = field(default_factory=dict)

    def first(self, kind: str) -> Alert | None:
        return next((a for a in self.alerts if a.kind == kind), None)


def turns_from(spans: list[SpanRecord], price: cl.PriceCard = cl.CHEAP) -> list[Turn]:
    """Group spans by trace and describe every ``invoke_workflow`` turn, in time order. Traces without a workflow span
    (eval wrappers, stray spans) are ignored."""
    by_trace: dict[str, list[SpanRecord]] = {}
    for s in spans:
        by_trace.setdefault(s.trace_id, []).append(s)
    turns = []
    for tid, group in by_trace.items():
        workflows = [s for s in group if s.get(tracing.OPERATION) == "invoke_workflow"]
        if not workflows:
            continue
        w = min(workflows, key=lambda s: s.start_ns)
        defects = tuple(sorted({v.code for v in te.check_trace(group) if v.code in te.DEFECTS}))
        route = w.get("app.reply.agent") or "none"
        turns.append(
            Turn(
                tid,
                w.start_ns,
                route,
                bool(defects),
                defects,
                w.get("app.reply.status") == "escalated",
                cl.trace_cost(group, price),
            )
        )
    return sorted(turns, key=lambda t: t.start_ns)


def _rate(turns: list[Turn], attr: str) -> float:
    return sum(getattr(t, attr) for t in turns) / len(turns) if turns else 0.0


def monitor(
    production: list[Turn], reference: list[Turn], config: MonitorConfig | None = None
) -> MonitorReport:
    """Compare production traffic (in time order) with a reference period believed healthy."""
    cfg = config or MonitorConfig()
    if len(reference) < 50:
        raise ValueError("the reference period needs at least 50 turns to set the baseline rates")
    report = MonitorReport(
        [],
        len(production),
        {
            "defect": _rate(reference, "defect"),
            "escalated": _rate(reference, "escalated"),
            "cost": sum(t.cost for t in reference) / len(reference),
        },
    )
    for attr, kind in (("defect", "defect_rate"), ("escalated", "escalation_rate")):
        p0 = min(
            max(_rate(reference, attr), 0.01), 0.45
        )  # a floor: a rate that was exactly 0 still needs a baseline to compare with
        p1 = min(p0 * cfg.rate_shift, 0.9)
        h = drift.calibrate_threshold(p0, p1, cfg.arl0, n_sims=cfg.n_sims, seed=cfg.seed)
        cusum = drift.BernoulliCusum(p0, p1, h)
        for i, t in enumerate(production, 1):
            if cusum.update(getattr(t, attr)):
                window = production[max(0, i - 100) : i]
                codes = sorted({c for x in window for c in x.defect_codes})
                extra = (
                    f"; defect codes in the last 100 turns: {codes}"
                    if attr == "defect" and codes
                    else ""
                )
                report.alerts.append(
                    Alert(
                        kind,
                        i,
                        t.trace_id,
                        f"{attr} rate CUSUM crossed {h:.2f} (reference {p0:.1%}, alarm tuned for a false alarm about every {cfg.arl0:.0f} turns); last 100 turns {_rate(window, attr):.1%}{extra}",
                    )
                )
                break
    ref_routes = _counts(reference)
    ref_cost = [t.cost for t in reference]
    for start in range(0, len(production) - cfg.window + 1, cfg.window):
        window = production[start : start + cfg.window]
        end = start + cfg.window
        win_routes = _counts(window)
        if not report.first("route_drift") and len(set(ref_routes) | set(win_routes)) >= 2:
            # (one route in both samples means nothing to compare: a chi-square needs at least two categories)
            test = drift.chi2_homogeneity(ref_routes, win_routes)
            if test["p"] < cfg.route_alpha:
                report.alerts.append(
                    Alert(
                        "route_drift",
                        end,
                        window[-1].trace_id,
                        f"route mix differs from the reference (chi-square {test['chi2']:.1f}, df {test['df']}, p={test['p']:.2g}); window {_mix(window)} vs reference {_mix(reference)}",
                    )
                )
        if not report.first("cost_spike"):
            diff = abtest.bootstrap_diff(
                ref_cost, [t.cost for t in window], n_boot=1000, seed=cfg.seed
            )
            base = diff["control_mean"]
            if base > 0 and diff["diff"] / base > cfg.cost_increase and diff["ci"][0] > 0:
                report.alerts.append(
                    Alert(
                        "cost_spike",
                        end,
                        window[-1].trace_id,
                        f"cost per turn {diff['diff'] / base:+.0%} against the reference (95% interval [{diff['ci'][0] / base:+.0%}, {diff['ci'][1] / base:+.0%}], tolerance {cfg.cost_increase:+.0%})",
                    )
                )
    report.alerts.sort(key=lambda a: a.at_turn)
    return report


def _counts(turns: list[Turn]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in turns:
        out[t.route] = out.get(t.route, 0) + 1
    return out


def _mix(turns: list[Turn]) -> str:
    c = _counts(turns)
    return "{" + ", ".join(f"{k} {v / len(turns):.0%}" for k, v in sorted(c.items())) + "}"
