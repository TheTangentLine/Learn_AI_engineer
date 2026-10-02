"""A single self-contained HTML page for cost and quality across runs.

    html = render([baseline_run, candidate_run], baseline="baseline")

Properties (each is tested): no scripts, no external requests (works offline and cannot phone home), every dynamic value
is HTML-escaped (a case id or a note containing ``<script>`` is text), charts are inline SVG with text alternatives, and
every number on the page is computed from the run artifacts: nothing is typed in by hand. Prices and latencies are labelled
as simulated / assumed, because they are.
"""

from __future__ import annotations

import html

import traceeval as te

from common import tracing
from common.evalkit import bootstrap_ci

from .runner import Run

KIND_ORDER = [
    "billing-small",
    "billing-approval",
    "billing-unpaid",
    "billing-missing-id",
    "billing-pressure",
    "tech-outage",
    "tech-howto",
    "human",
    "off-topic",
]


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def run_stats(run: Run) -> dict:
    rows = run.cases
    n = max(len(rows), 1)
    passed = [r["pass_fraction"] for r in rows]
    mean, lo, hi = bootstrap_ci(passed) if rows else (0.0, 0.0, 0.0)
    by_kind: dict[str, list[float]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r["pass_fraction"])
    summary = tracing.summarize(run.spans) if run.spans else None
    return {
        "name": run.name,
        "variant": run.manifest["variant"],
        "model": run.manifest["model"],
        "provider": run.manifest["provider"],
        "trials": run.manifest["trials"],
        "complete": run.complete,
        "pass_rate": mean,
        "pass_ci": (lo, hi),
        "cost_per_1k": sum(r["cost_usd"] for r in rows) / n * 1000,
        "in_tokens": sum(r["input_tokens"] for r in rows) / n,
        "out_tokens": sum(r["output_tokens"] for r in rows) / n,
        "calls": sum(r["model_calls"] for r in rows) / n,
        "p50": te.percentile([r["latency_s"] for r in rows], 50),
        "p95": te.percentile([r["latency_s"] for r in rows], 95),
        "by_kind": {k: sum(v) / len(v) for k, v in by_kind.items()},
        "defect_cases": sum(1 for r in rows if r["defects"]),
        "by_agent": summary["by_agent"] if summary else {},
        "by_tool": summary["by_tool"] if summary else {},
        "spend": run.manifest["spend_usd"],
    }


def pareto(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """The cost/quality frontier: runs that no other run beats on BOTH (cheaper and at least as good, or better and no dearer)."""
    front = []
    for c, q, n in points:
        if not any((c2 <= c and q2 >= q) and (c2 < c or q2 > q) for c2, q2, _ in points):
            front.append((c, q, n))
    return sorted(front)


def _scatter(stats: list[dict], baseline: str) -> str:
    w, h, pad = 520, 300, 44
    pts = [(s["cost_per_1k"], s["pass_rate"], s["name"]) for s in stats]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1 = 0.0, max(xs) * 1.15 or 1.0
    spread = max(max(ys) - min(ys), 0.1)
    y0, y1 = (
        max(0.0, min(ys) - 0.15 * spread - 0.02),
        min(1.0, max(ys) + 0.15 * spread + 0.02),
    )  # zoomed: the range of the data, not 0 to 100%

    def px(x):
        return pad + (x - x0) / (x1 - x0) * (w - 2 * pad)

    def py(y):
        return h - pad - (y - y0) / (y1 - y0) * (h - 2 * pad)

    front = pareto(pts)
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(_scatter_alt(stats))}">']
    out.append(
        f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" class="axis"/><line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" class="axis"/>'
    )
    out.append(
        f'<text x="{w / 2}" y="{h - 8}" class="lbl" text-anchor="middle">cost per 1,000 conversations (simulated USD)</text>'
    )
    out.append(
        f'<text x="9" y="{h / 2}" class="lbl" text-anchor="middle" transform="rotate(-90 9 {h / 2})">pass rate</text>'
    )
    for frac in (0, 0.5, 1.0):
        yv = y0 + (y1 - y0) * frac
        out.append(
            f'<text x="{pad - 6}" y="{py(yv) + 4:.1f}" class="lbl" text-anchor="end">{yv:.0%}</text>'
        )
        out.append(
            f'<text x="{px(x1 * frac):.1f}" y="{h - pad + 16}" class="lbl" text-anchor="middle">{x1 * frac:.2f}</text>'
        )
    if len(front) > 1:
        out.append(
            '<polyline class="front" fill="none" points="'
            + " ".join(f"{px(c):.1f},{py(q):.1f}" for c, q, _ in front)
            + '"/>'
        )
    on_front = {n for _, _, n in front}
    for c, q, n in pts:
        cls = "pt base" if n == baseline else ("pt front" if n in on_front else "pt")
        out.append(
            f'<circle cx="{px(c):.1f}" cy="{py(q):.1f}" r="6" class="{cls}"><title>{esc(n)}: ${c:.3f} per 1k, {q:.0%} pass</title></circle>'
        )
        out.append(f'<text x="{px(c) + 9:.1f}" y="{py(q) + 4:.1f}" class="lbl">{esc(n)}</text>')
    out.append("</svg>")
    return "".join(out)


def _scatter_alt(stats: list[dict]) -> str:
    return "Cost against pass rate: " + "; ".join(
        f"{s['name']} costs ${s['cost_per_1k']:.3f} per 1k and passes {s['pass_rate']:.0%}"
        for s in stats
    )


def _bars(items: list[tuple[str, float]], unit: str, title: str) -> str:
    total = sum(v for _, v in items) or 1.0
    top = max((v for _, v in items), default=1.0) or 1.0
    rows = []
    for name, v in items:
        rows.append(
            f'<div class="barrow"><span class="barname">{esc(name)}</span>'
            f'<span class="bar" style="width:{max(v / top * 100, 0.5):.1f}%"></span>'
            f'<span class="barval">{v:,.0f}{esc(unit)} ({v / total:.0%})</span></div>'
        )
    return f'<div class="bars" role="group" aria-label="{esc(title)}">' + "".join(rows) + "</div>"


def _cell(rate: float | None) -> str:
    if rate is None:
        return '<td class="na">–</td>'
    hue = int(120 * rate)  # 0 red .. 120 green
    return f'<td style="background:hsl({hue} 55% 38% / 0.55)">{rate:.0%}</td>'


def render(
    runs: list[Run], *, baseline: str | None = None, title: str = "Support system: quality and cost"
) -> str:
    if not runs:
        raise ValueError("no runs to show")
    stats = [run_stats(r) for r in runs]
    baseline = baseline or stats[0]["name"]
    base = next((s for s in stats if s["name"] == baseline), stats[0])
    latest = stats[-1]
    kinds = [k for k in KIND_ORDER if any(k in s["by_kind"] for s in stats)] + sorted(
        {k for s in stats for k in s["by_kind"]} - set(KIND_ORDER)
    )

    def rel(a, b):
        return "" if not b else f" ({a / b - 1:+.0%})"

    head = "".join(f"<th>{esc(s['name'])}</th>" for s in stats)
    table_rows = [
        ("variant", [esc(s["variant"]) for s in stats]),
        ("model", [esc(f"{s['provider']} / {s['model']}") for s in stats]),
        ("trials per case", [str(s["trials"]) for s in stats]),
        ("finished", ["yes" if s["complete"] else "<b>NO: incomplete</b>" for s in stats]),
        (
            "pass rate (95% interval)",
            [f"{s['pass_rate']:.1%} [{s['pass_ci'][0]:.0%}-{s['pass_ci'][1]:.0%}]" for s in stats],
        ),
        ("cases with a trace defect", [str(s["defect_cases"]) for s in stats]),
        (
            "cost per 1k conversations (simulated)",
            [
                f"${s['cost_per_1k']:.3f}{rel(s['cost_per_1k'], base['cost_per_1k']) if s is not base else ''}"
                for s in stats
            ],
        ),
        ("input tokens per conversation", [f"{s['in_tokens']:.0f}" for s in stats]),
        ("output tokens per conversation", [f"{s['out_tokens']:.0f}" for s in stats]),
        ("model calls per conversation", [f"{s['calls']:.2f}" for s in stats]),
        (
            "estimated latency p50 / p95 (assumed model)",
            [f"{s['p50']:.2f}s / {s['p95']:.2f}s" for s in stats],
        ),
        ("spend of this run (simulated)", [f"${s['spend']:.4f}" for s in stats]),
    ]
    body = "".join(
        f"<tr><th scope='row'>{esc(label)}</th>"
        + "".join(f"<td>{v}</td>" for v in values)
        + "</tr>"
        for label, values in table_rows
    )
    kind_rows = "".join(
        f"<tr><th scope='row'>{esc(k)}</th>"
        + "".join(_cell(s["by_kind"].get(k)) for s in stats)
        + "</tr>"
        for k in kinds
    )
    agent_items = sorted(
        ((a, v["input_tokens"] + 5 * v["output_tokens"]) for a, v in latest["by_agent"].items()),
        key=lambda kv: -kv[1],
    )
    tool_rows = "".join(
        f"<tr><th scope='row'>{esc(t)}</th><td>{v['calls']}</td><td>{v['errors']}</td><td>{v['ms'] / max(v['calls'], 1):.1f}</td></tr>"
        for t, v in sorted(latest["by_tool"].items(), key=lambda kv: -kv[1]["calls"])
    )
    notes = "".join(
        f"<li>{esc(r.name)}: {esc(r.manifest['note'])}</li>" for r in runs if r.manifest.get("note")
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
:root {{ --bg:#fff; --fg:#1d2330; --muted:#5b6475; --line:#d8dce5; --card:#f5f6fa; --accent:#2e5bff; --good:#1a8f4a; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#14171f; --fg:#e8ebf2; --muted:#9aa3b5; --line:#2c3242; --card:#1c2029; --accent:#7e9bff; --good:#4ac27a; }} }}
body {{ font:15px/1.5 system-ui,sans-serif; background:var(--bg); color:var(--fg); margin:0; padding:24px 16px; }}
main {{ max-width:980px; margin:0 auto; }} h1 {{ font-size:22px; }} h2 {{ font-size:17px; margin-top:32px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }}
.card b {{ display:block; font-size:24px; }} .card span {{ color:var(--muted); font-size:13px; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }} th,td {{ border:1px solid var(--line); padding:6px 8px; text-align:right; }}
th[scope=row] {{ text-align:left; font-weight:500; }} thead th {{ background:var(--card); }} .na {{ color:var(--muted); text-align:center; }}
svg {{ width:100%; max-width:560px; }} .axis {{ stroke:var(--muted); }} .lbl {{ fill:var(--muted); font-size:11px; }}
.pt {{ fill:var(--muted); }} .pt.front {{ fill:var(--good); }} .pt.base {{ fill:var(--accent); }} .front {{ stroke:var(--good); stroke-width:1.5; stroke-dasharray:4 3; }}
.barrow {{ display:grid; grid-template-columns:110px 1fr 190px; align-items:center; gap:8px; margin:4px 0; }} .bar {{ height:14px; background:var(--accent); border-radius:3px; }}
.barval {{ color:var(--muted); font-size:13px; }} .scroll {{ overflow-x:auto; }} footer {{ margin-top:32px; color:var(--muted); font-size:13px; }}
</style></head><body><main>
<h1>{esc(title)}</h1>
<p>Baseline: <b>{esc(base["name"])}</b>. Latest run: <b>{esc(latest["name"])}</b>. All dollar figures are <b>simulated</b> (real token counts priced with a price card) and latencies are <b>estimated</b> from an assumed model.</p>
<div class="cards">
<div class="card"><b>{latest["pass_rate"]:.0%}</b><span>pass rate of {esc(latest["name"])} (baseline {base["pass_rate"]:.0%})</span></div>
<div class="card"><b>${latest["cost_per_1k"]:.3f}</b><span>per 1,000 conversations{rel(latest["cost_per_1k"], base["cost_per_1k"]) if latest is not base else ""}</span></div>
<div class="card"><b>{latest["calls"]:.2f}</b><span>model calls per conversation ({latest["in_tokens"]:.0f} input tokens)</span></div>
<div class="card"><b>{latest["p95"]:.2f}s</b><span>estimated p95 latency</span></div>
</div>
<h2>Runs</h2><div class="scroll"><table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></div>
<h2>Cost against quality</h2>{_scatter(stats, base["name"])}
<p style="color:var(--muted);font-size:13px">Green dashed line: runs no other run beats on both cost and pass rate. Blue: the baseline.</p>
<h2>Where the tokens go in {esc(latest["name"])}</h2>{_bars(agent_items, " tok-eq", "weighted tokens by agent") if agent_items else "<p>No spans in this run.</p>"}
<h2>Pass rate by kind of request</h2><div class="scroll"><table><thead><tr><th></th>{head}</tr></thead><tbody>{kind_rows}</tbody></table></div>
<h2>Tool calls in {esc(latest["name"])}</h2><table><thead><tr><th>tool</th><th>calls</th><th>errors</th><th>avg ms</th></tr></thead><tbody>{tool_rows or '<tr><td colspan=4 class="na">no spans</td></tr>'}</tbody></table>
{("<h2>Notes</h2><ul>" + notes + "</ul>") if notes else ""}
<footer>Generated from run artifacts (manifest, cases, spans). Token weights count one output token as five input tokens, the price ratio of the card used.</footer>
</main></body></html>
"""
