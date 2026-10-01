"""Tests for Day 6: the generated report, table extraction, chunk strategies, answerability, images."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from day6_solution import (  # noqa: E402
    CHART_VALUES,
    TABLES,
    answers,
    approx_image_tokens,
    cell_questions,
    extract_images,
    extract_tables,
    image_chunks,
    image_message,
    make_report,
    markdown_table_chunks,
    naive_chunks,
    row_sentence_chunks,
    scripted_describer,
)


@pytest.fixture(scope="module")
def pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("rep") / "r.pdf"
    make_report(path)
    return path


def test_pdfplumber_recovers_every_table_exactly(pdf):
    tables = extract_tables(pdf)
    assert len(tables) == len(TABLES) == 8
    for got, want in zip(tables, TABLES, strict=True):
        assert got["header"] == want["header"] and got["rows"] == want["rows"]
        assert got["caption"] == want["title"], "caption is the text directly above the table"


def test_naive_text_loses_the_row_structure(pdf):
    """The reason this lesson exists: pypdf puts each cell on its own line, detached from its header."""
    from pypdf import PdfReader

    lines = [
        ln.strip()
        for p in PdfReader(str(pdf)).pages
        for ln in (p.extract_text() or "").splitlines()
    ]
    row = TABLES[1]["rows"][0]  # Engineering row of the headcount table
    assert row[0] in lines and row[1] in lines, "label and value are separate lines"
    assert not any(row[0] in ln and row[1] in ln for ln in lines), "...never on the same line"


def test_markdown_chunks_keep_header_rows_and_caption_together(pdf):
    chunks = markdown_table_chunks(extract_tables(pdf))
    assert len(chunks) == 8
    for chunk, t in zip(chunks, TABLES, strict=True):
        lines = chunk.splitlines()
        assert (
            lines[0] == t["title"]
            and lines[1].startswith("| Item |")
            and lines[2].startswith("|---")
        )
        assert len(lines) == 3 + len(t["rows"])
        assert all(ln.count("|") == lines[1].count("|") for ln in lines[1:] if ln.startswith("|"))


def test_row_sentences_make_every_cell_self_describing(pdf):
    chunks = row_sentence_chunks(extract_tables(pdf))
    assert len(chunks) == sum(len(t["rows"]) for t in TABLES)
    first = chunks[0]
    assert TABLES[0]["title"] in first and "Item: North America" in first and "Q1 2024 = " in first


def test_every_question_is_answerable_by_table_aware_chunks_but_not_always_by_naive(pdf):
    qs = cell_questions()
    assert len(qs) == 40 and qs == cell_questions(), "deterministic"
    for q in qs:  # the (row, col, value) triple must really be in the source table
        t = next(t for t in TABLES if t["title"] == q["table"])
        r = next(r for r in t["rows"] if r[0] == q["row"])
        assert r[t["header"].index(q["col"])] == q["value"]
    tables = extract_tables(pdf)
    for chunks in (markdown_table_chunks(tables), row_sentence_chunks(tables)):
        assert all(any(answers(c, q) for c in chunks) for q in qs)
    naive = naive_chunks(pdf, 300)
    assert sum(any(answers(c, q) for c in naive) for q in qs) < len(qs), (
        "small naive chunks lose evidence"
    )


def test_answers_requires_row_column_and_value_together():
    q = {"row": "EMEA", "col": "Q3 2024", "value": "4,210"}
    assert answers("EMEA | Q3 2024 | 4,210", q)
    assert (
        not answers("EMEA | 4,210", q)
        and not answers("Q3 2024 | 4,210", q)
        and not answers("EMEA | Q3 2024", q)
    )


def test_chart_is_extracted_and_becomes_a_citable_text_chunk(pdf):
    imgs = extract_images(pdf)
    assert len(imgs) == 1 and imgs[0][0] == 6
    assert imgs[0][1].startswith(b"\x89PNG"), "the extracted bytes are a real PNG"
    (chunk,) = image_chunks(pdf, scripted_describer)
    assert chunk.startswith("[Figure on page") and all(f"{q} = " in chunk for q in CHART_VALUES)


def test_image_messages_use_each_providers_shape_and_token_estimate_scales():
    anth = image_message("anthropic", b"\x89PNG", "what?")[0]["content"]
    assert (
        anth[0]["type"] == "image"
        and anth[0]["source"]["media_type"] == "image/png"
        and anth[1]["type"] == "text"
    )
    oai = image_message("openai", b"\x89PNG", "what?")[0]["content"]
    assert oai[0]["type"] == "input_text" and oai[1]["type"] == "input_image"
    assert oai[1]["image_url"].startswith("data:image/png;base64,")
    assert approx_image_tokens(800, 800) > approx_image_tokens(400, 260) > 0


def test_questions_are_spread_across_rows_and_columns_not_just_the_first_cells():
    qs = cell_questions()
    for t in TABLES:
        mine = [q for q in qs if q["table"] == t["title"]]
        assert len(mine) == 5
        assert len({q["row"] for q in mine}) >= 2, "questions must probe different rows"
        assert len({q["col"] for q in mine}) >= 2, "...and different columns"


def test_question_set_covers_most_rows_overall():
    """A first-cells-only generator would touch ~2 rows per table; random sampling touches most of them."""
    qs = cell_questions()
    assert len({(q["table"], q["row"]) for q in qs}) >= 25
