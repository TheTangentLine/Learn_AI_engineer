"""A one-file HTML dashboard of how the product is doing: quality by kind, where the time goes, load behaviour and cost. Static, no JavaScript, no external requests, safe to open from a file or attach to a report.

    html = render(metrics)           # ``metrics``: see the keys used below; any missing section is simply left out
    Path("outputs/w12_dashboard.html").write_text(html)

Every value is escaped; the page contains no data from users, only aggregates.
"""

from __future__ import annotations

import html as H

CSS = """
:root{--bg:#fafaf7;--fg:#1c1c1a;--mut:#6b6b66;--bar:#3b6ea5;--bar2:#c28a2c;--ok:#2e7d4f;--bad:#b3392f;--card:#fff;--line:#e3e1d8}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--mut:#9a9990;--bar:#6aa0d6;--bar2:#d9a548;--ok:#5bbf86;--bad:#e5766c;--card:#1f1f1c;--line:#34332e}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:860px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}p.sub{color:var(--mut);margin:0 0 16px}
section{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:12px 0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:4px 8px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}
.bar{display:inline-block;height:10px;background:var(--bar);border-radius:2px;vertical-align:middle}.bar.p95{background:var(--bar2)}.ok{color:var(--ok)}.bad{color:var(--bad)}.note{color:var(--mut);font-size:13px}
"""


def e(x) -> str:
    return H.escape(str(x))


def pct(rate) -> str:
    p, lo, hi = rate
    return "n/a" if p != p else f"{p:.0%}"


def bars(rows: list[tuple[str, float, float]], unit: str = "ms") -> str:
    """Horizontal bars for (label, p50, p95); widths are relative to the largest p95."""
    top = max((r[2] for r in rows), default=1.0) or 1.0
    out = ["<table><tr><th>stage</th><th>p50</th><th>p95</th><th></th></tr>"]
    for label, p50, p95 in rows:
        out.append(
            f"<tr><td>{e(label)}</td><td>{p50:,.1f} {unit}</td><td>{p95:,.1f} {unit}</td>"
            f'<td style="width:45%"><span class="bar" style="width:{100 * p50 / top:.1f}%"></span><br><span class="bar p95" style="width:{100 * p95 / top:.1f}%"></span></td></tr>'
        )
    out.append("</table>")
    return "".join(out)


def render(m: dict) -> str:
    parts = [
        f"<h1>{e(m.get('title', 'Course Copilot'))}</h1><p class='sub'>{e(m.get('subtitle', ''))}</p>"
    ]
    if "pass_by_kind" in m:
        rows = "".join(
            f"<tr><td>{e(k)}</td><td>{v['passed']}/{v['n']}</td><td class='{'ok' if v['passed'] == v['n'] else ''}'>{pct(v['rate'])}</td><td>{e(v.get('interval', ''))}</td></tr>"
            for k, v in m["pass_by_kind"].items()
        )
        parts.append(
            f"<section><h2>Golden set ({e(m.get('split', 'test'))} split)</h2><table><tr><th>kind</th><th>passed</th><th>rate</th><th>95% interval</th></tr>{rows}</table></section>"
        )
    if "stages_ms" in m:
        rows = [(k, v["p50"], v["p95"]) for k, v in m["stages_ms"].items()]
        parts.append(
            f"<section><h2>Where the time goes (cold questions)</h2>{bars(rows)}<p class='note'>Bars: p50 above, p95 below.</p></section>"
        )
    if "load_closed" in m:
        rows = "".join(
            f"<tr><td>{r['users']}</td><td>{r['rps']:.2f}</td><td>{r['lat50']:.2f} s</td><td>{r['lat95']:.2f} s</td><td class='{'bad' if r['errors'] else ''}'>{r['errors']}</td></tr>"
            for r in m["load_closed"]
        )
        parts.append(
            f"<section><h2>Load: concurrent users</h2><table><tr><th>users</th><th>req/s</th><th>latency p50</th><th>latency p95</th><th>errors</th></tr>{rows}</table></section>"
        )
    if "load_open" in m:
        rows = "".join(
            f"<tr><td>{r['rate']}</td><td>{r['n']}</td><td>{r['lat50']:.2f} s</td><td>{r['lat95']:.2f} s</td><td class='{'bad' if r['errors'] else ''}'>{r['errors']}</td></tr>"
            for r in m["load_open"]
        )
        parts.append(
            f"<section><h2>Load: arrival rate</h2><table><tr><th>per second</th><th>sent</th><th>latency p50</th><th>latency p95</th><th>errors</th></tr>{rows}</table></section>"
        )
    if "cost" in m:
        rows = "".join(
            f"<tr><td>{e(r['name'])}</td><td>{r['seconds']:.3f}</td><td>{r['tokens']:,}</td><td>${r['per_1000']:.3f}</td></tr>"
            for r in m["cost"]
        )
        parts.append(
            f"<section><h2>Cost per 1,000 questions</h2><table><tr><th>configuration</th><th>seconds each</th><th>tokens each</th><th>per 1,000</th></tr>{rows}</table>"
            f"<p class='note'>{e(m.get('cost_note', 'Prices are assumptions supplied as inputs.'))}</p></section>"
        )
    if "notes" in m:
        parts.append(
            "<section><h2>Notes</h2>"
            + "".join(f"<p class='note'>{e(n)}</p>" for n in m["notes"])
            + "</section>"
        )
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{e(m.get('title', 'Course Copilot'))}</title><style>{CSS}</style></head><body><main>{''.join(parts)}</main></body></html>"
