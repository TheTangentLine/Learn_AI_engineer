"""A checker for design documents: does it have the sections a reviewer needs, and are its requirements and risks the kind a test can hold you to?

    problems = check(Path("design/DESIGN.md").read_text())      # [] when the document is complete

It checks STRUCTURE and MEASURABILITY, not wisdom: a document can pass and still be wrong. What it catches is the common ways a design doc is vague (a requirement with no target, a risk with no
mitigation, no non-goals, no evaluation split) so a reviewer's time goes to the content.
"""

from __future__ import annotations

import re

SECTIONS = [
    "Problem and users",
    "Goals and non-goals",
    "Requirements",
    "Architecture",
    "Data",
    "Evaluation plan",
    "Decisions and alternatives",
    "Risks",
    "Security and privacy",
    "Cost and latency budget",
    "Rollout and operations",
    "Open questions",
]
NUMBER = re.compile(r"\d")


def sections(text: str) -> dict[str, str]:
    """{section title without its number: body}, for every ``## n. Title`` heading."""
    out: dict[str, str] = {}
    parts = re.split(r"^##\s+\d+\.\s+(.+?)\s*$", text, flags=re.M)
    for title, body in zip(parts[1::2], parts[2::2], strict=False):
        out[title.strip()] = body
    return out


def table_rows(body: str) -> list[list[str]]:
    """The data rows of the first Markdown table in ``body`` (header and separator removed), each as a list of cells."""
    rows = []
    for line in body.splitlines():
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            rows.append(cells)
        elif rows:
            break
    return rows[1:]  # drop the header


def check(
    text: str, *, min_requirements: int = 5, min_risks: int = 5, min_decisions: int = 3
) -> list[str]:
    problems: list[str] = []
    secs = sections(text)
    for want in SECTIONS:
        if want not in secs:
            problems.append(f"missing section: {want}")
    if "```mermaid" not in secs.get("Architecture", ""):
        problems.append("Architecture has no diagram (a ```mermaid block)")
    goals = secs.get("Goals and non-goals", "")
    if (
        not re.search(r"non-goals", goals, re.I)
        or len(re.findall(r"non-goals?:?\**\s*\S", goals, re.I)) < 1
    ):
        problems.append("no non-goals: say what you will not do")
    reqs = table_rows(secs.get("Requirements", ""))
    if len(reqs) < min_requirements:
        problems.append(f"only {len(reqs)} requirements (need {min_requirements})")
    for r in reqs:
        if len(r) < 4 or not all(c for c in r[1:4]):
            problems.append(
                f"requirement {r[0] if r else '?'} has an empty cell (statement, measurement and target are all required)"
            )
        elif not NUMBER.search(r[3]) and not re.search(
            r"\b(present|tested|reported|demonstrably)\b", r[3], re.I
        ):
            problems.append(f"requirement {r[0]} has no checkable target: {r[3]!r}")
    ev = secs.get("Evaluation plan", "")
    if not re.search(r"\bdev\b", ev, re.I) or not re.search(r"\btest\b", ev, re.I):
        problems.append("the evaluation plan does not name a dev/test split")
    if not re.search(r"baseline", ev, re.I):
        problems.append("the evaluation plan names no baseline to beat")
    dec = table_rows(secs.get("Decisions and alternatives", ""))
    if len(dec) < min_decisions:
        problems.append(f"only {len(dec)} decisions with alternatives (need {min_decisions})")
    for d in dec:
        if len(d) < 4 or not all(d[:4]):
            problems.append(
                f"decision {d[0][:30] if d else '?'!r} lacks alternatives, a reason, or a way to find out it was wrong"
            )
    risks = table_rows(secs.get("Risks", ""))
    if len(risks) < min_risks:
        problems.append(f"only {len(risks)} risks (need {min_risks})")
    for r in risks:
        if len(r) < 4 or not r[3]:
            problems.append(f"risk {r[0][:30]!r} has no mitigation")
    if not re.search(r"assum", secs.get("Cost and latency budget", ""), re.I):
        problems.append("the cost section does not say which numbers are assumptions")
    return problems
