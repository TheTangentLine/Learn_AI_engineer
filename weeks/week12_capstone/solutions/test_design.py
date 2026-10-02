"""Tests for the design-document checker and the shipped templates and reference document."""

from __future__ import annotations

from pathlib import Path

from copilot import design as D

HERE = Path(__file__).resolve().parent
REFERENCE = (HERE / "design" / "DESIGN.md").read_text()


def test_the_reference_design_document_passes_its_own_checker():
    assert D.check(REFERENCE) == []
    assert list(D.sections(REFERENCE)) == D.SECTIONS


def test_the_blank_template_fails_it_in_the_expected_places():
    problems = D.check((HERE.parent / "templates" / "design_doc_template.md").read_text())
    for expect in (
        "has no diagram",
        "only 1 requirements",
        "requirement R1 has an empty cell",
        "only 0 decisions",
        "only 0 risks",
    ):
        assert any(expect in p for p in problems), (expect, problems)
    assert not any(
        p.startswith("missing section") for p in problems
    )  # the template has every section, it is the content that is missing


def replace(old: str, new: str) -> str:
    assert old in REFERENCE
    return REFERENCE.replace(old, new, 1)


def test_each_planted_defect_is_reported():
    cases = {
        "missing section: Risks": REFERENCE.replace("## 8. Risks", "## 8. Hazards"),
        "has no diagram": REFERENCE.replace("```mermaid", "```text"),
        "no non-goals": replace("- **Non-goals:**", "- **Ideas:**").replace(
            "Goals and non-goals", "Goals and ideas"
        ),
        "no checkable target": replace("| ≥ 90% |", "| high |"),
        "empty cell": replace("| hit@5 over the 52 answerable golden questions |", "|  |"),
        "does not name a dev/test split": REFERENCE.replace("**dev**", "**d**")
        .replace("**test**", "**t**")
        .replace("dev and test", "d and t")
        .replace("dev/test", "d/t")
        .replace("dev", "d")
        .replace("test", "t"),
        "no baseline": REFERENCE.replace("Baselines to beat", "Things").replace(
            "baseline", "reference"
        ),
        "has no mitigation": replace(
            "| high | the quality numbers flatter the product | paraphrase questions; report intervals; say so in the report |",
            "| high | the quality numbers flatter the product |  |",
        ),
        "which numbers are assumptions": REFERENCE.replace("assumptions", "facts").replace(
            "assumption", "fact"
        ),
    }
    for expect, text in cases.items():
        assert any(expect in p for p in D.check(text)), (expect, D.check(text))


def test_too_few_requirements_risks_or_decisions_are_reported():
    only_two = "\n".join(
        line
        for line in REFERENCE.splitlines()
        if not line.startswith(("| R3", "| R4", "| R5", "| R6", "| R7", "| R8"))
    )
    assert any("only 2 requirements" in p for p in D.check(only_two))
    assert D.check(REFERENCE, min_risks=20) and any(
        "only 8 risks" in p for p in D.check(REFERENCE, min_risks=20)
    )
    assert any("decisions with alternatives" in p for p in D.check(REFERENCE, min_decisions=9))


def test_table_rows_drops_the_header_and_separator_and_stops_at_the_end_of_the_table():
    body = "intro\n| a | b |\n|---|:---:|\n| 1 | 2 |\n| 3 | 4 |\n\nafter\n| x | y |\n"
    assert D.table_rows(body) == [["1", "2"], ["3", "4"]] and D.table_rows("no table") == []


def test_the_other_templates_exist_and_name_their_required_parts():
    t = HERE.parent / "templates"
    assert "How well it works (measured)" in (t / "portfolio_readme_template.md").read_text()
    retro = (t / "retrospective_template.md").read_text()
    assert "targets vs outcome" in retro and "What surprised me" in retro


def test_the_minimums_are_inclusive():
    assert D.check(REFERENCE, min_requirements=8, min_risks=8, min_decisions=5) == []
    assert any("only 8 requirements" in p for p in D.check(REFERENCE, min_requirements=9))
    assert any("only 8 risks" in p for p in D.check(REFERENCE, min_risks=9))
