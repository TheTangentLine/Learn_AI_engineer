"""Compare pipeline output to gold labels and render a Markdown accuracy report."""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict

from .pipeline import Record


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def values_match(got, gold) -> bool:
    if isinstance(gold, bool) or got is None or gold is None:
        return got == gold
    if isinstance(gold, (int, float)):
        return isinstance(got, (int, float)) and abs(float(got) - float(gold)) < 0.011
    if isinstance(gold, list):  # list of dicts (line items): order-insensitive
        if not isinstance(got, list) or len(got) != len(gold):
            return False

        def key(d):
            return (_norm(d["description"]), float(d["quantity"]), round(float(d["unit_price"]), 2))

        try:
            return sorted(map(key, got)) == sorted(map(key, gold))
        except (KeyError, TypeError, ValueError):
            return False
    return _norm(got) == _norm(gold)


def field_scores(record: Record, gold: dict) -> dict[str, bool]:
    got = record.data or {}
    return {f: values_match(got.get(f), v) for f, v in gold["data"].items()}


def build_report(records: list[Record], gold: dict[str, dict]) -> tuple[str, dict]:
    n = len(records)
    status = Counter(r.status for r in records)
    cls_ok = sum(r.doc_type == gold[r.file]["doc_type"] for r in records)
    confusion = Counter((gold[r.file]["doc_type"], r.doc_type or "-") for r in records)

    per_type: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    exact_docs = 0
    for r in records:
        g = gold[r.file]
        if r.status != "ok":
            continue
        scores = field_scores(r, g)
        for f, ok in scores.items():
            per_type[g["doc_type"]][f].append(ok)
        exact_docs += all(scores.values()) and r.doc_type == g["doc_type"]

    all_fields = [ok for t in per_type.values() for oks in t.values() for ok in oks]
    field_acc = sum(all_fields) / len(all_fields) if all_fields else 0.0
    retried = sum(r.attempts > 1 for r in records)
    lat = sorted(r.seconds for r in records)
    p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0
    cost = sum(r.cost_usd for r in records)

    out = [
        "# Document extraction report",
        "",
        f"- documents: **{n}** | ok: **{status['ok']}** | needs_review: **{status['needs_review']}** "
        f"| failed: **{status['failed']}**",
        f"- classification accuracy: **{cls_ok}/{n}**",
        f"- fully-correct documents (every field + type): **{exact_docs}/{n}**",
        f"- field accuracy (documents with status ok): **{field_acc:.1%}**",
        f"- needed a retry: **{retried}** | total cost: **${cost:.4f}** "
        f"(${cost / max(n, 1):.4f}/doc) | latency p50 {statistics.median(lat) if lat else 0:.2f}s, p95 {p95:.2f}s",
        "",
        "## Classification confusion (gold -> predicted)",
        "",
    ]
    out += [f"- {g} -> {p}: {c}" for (g, p), c in sorted(confusion.items())]
    out += ["", "## Per-field accuracy", "", "| type | field | correct |", "|---|---|---|"]
    for t, fields in sorted(per_type.items()):
        for f, oks in fields.items():
            out.append(f"| {t} | {f} | {sum(oks)}/{len(oks)} |")
    problems = [r for r in records if r.status != "ok"]
    out += ["", "## Not ok", ""] + (
        [f"- `{r.file}` **{r.status}**: {r.reason}" for r in problems] or ["(none)"]
    )
    wrong = [
        (r, f)
        for r in records
        if r.status == "ok"
        for f, ok in field_scores(r, gold[r.file]).items()
        if not ok
    ]
    out += ["", "## Wrong fields", ""] + ([f"- `{r.file}`: {f}" for r, f in wrong] or ["(none)"])
    metrics = {
        "status": dict(status),
        "classification": cls_ok / max(n, 1),
        "field_accuracy": field_acc,
        "exact_docs": exact_docs,
        "retried": retried,
        "cost_usd": cost,
    }
    return "\n".join(out), metrics
