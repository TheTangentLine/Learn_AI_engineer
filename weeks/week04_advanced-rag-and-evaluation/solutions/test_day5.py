"""Tests for Day 5: the shop DB, SQL extraction, the NL->SQL pipeline (scripted writers), term graph."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from day5_solution import (  # noqa: E402
    QUESTIONS,
    answer_question,
    build_term_graph,
    communities,
    evaluate,
    extract_sql,
    make_shop_db,
    results_equal,
)


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("shop") / "shop.db")
    make_shop_db(path)
    return path


def test_shop_db_is_deterministic_and_has_an_unreachable_secrets_table(tmp_path):
    a, b = str(tmp_path / "a.db"), str(tmp_path / "b.db")
    make_shop_db(a)
    make_shop_db(b)
    qa = sqlite3.connect(a).execute("SELECT SUM(quantity), COUNT(*) FROM order_items").fetchone()
    qb = sqlite3.connect(b).execute("SELECT SUM(quantity), COUNT(*) FROM order_items").fetchone()
    assert qa == qb
    assert (
        sqlite3.connect(a).execute("SELECT token FROM secrets").fetchone()[0].startswith("sk-live")
    )


def test_every_gold_query_is_valid_safe_and_nonempty(db):
    from common.sqlsafe import run_readonly, validate_sql

    for q in QUESTIONS:
        v = validate_sql(q.gold, {"customers", "products", "orders", "order_items"}, 1000)
        _, rows = run_readonly(db, v.sql, 1000)
        assert rows, f"gold query returns nothing: {q.question}"


@pytest.mark.parametrize(
    "a,b,ordered,expected",
    [
        ([(1, 2.001)], [(1, 2.0)], False, True),
        ([(1,), (2,)], [(2,), (1,)], False, True),
        ([(1,), (2,)], [(2,), (1,)], True, False),
        ([(1,)], [(1,), (1,)], False, False),
        ([("a", 1)], [("a", 2)], False, False),
    ],
)
def test_results_equal(a, b, ordered, expected):
    assert results_equal(a, b, ordered) is expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("SELECT COUNT(*) FROM customers", "SELECT COUNT(*) FROM customers"),
        ("```sql\nSELECT 1;\n```", "SELECT 1"),
        ("SQL: SELECT 2", "SELECT 2"),
        ("SELECT 3;\n\nThis query counts things.", "SELECT 3"),
        ("", ""),
    ],
)
def test_extract_sql(raw, expected):
    assert extract_sql(raw) == expected


def writer(*sqls):
    queue = list(sqls)
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return queue.pop(0) if queue else "SELECT 1"

    generate.prompts = prompts
    return generate


def test_pipeline_runs_valid_sql(db):
    r = answer_question(db, "How many customers?", writer("SELECT COUNT(*) FROM customers"))
    assert r.rows == [(60,)] and r.attempts == 1 and r.error is None and not r.blocked


def test_sql_error_triggers_one_repair_with_the_error_in_the_prompt(db):
    gen = writer("SELECT COUNT(*) FROM customerz", "SELECT COUNT(*) FROM customers")
    r = answer_question(db, "How many customers?", gen)
    assert r.rows == [(60,)] and r.attempts == 2
    assert "customerz" in gen.prompts[1] and "failed" in gen.prompts[1]


def test_unsafe_sql_is_blocked_and_the_model_is_told_why(db):
    gen = writer("DELETE FROM customers", "SELECT COUNT(*) FROM customers")
    r = answer_question(db, "Delete everyone", gen)
    assert r.rows == [(60,)] and r.attempts == 2
    assert "only SELECT" in gen.prompts[1]
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 60


def test_persistent_unsafe_output_is_surfaced_not_executed(db):
    for bad in (
        "SELECT token FROM secrets",
        "SELECT * FROM sqlite_master",
        "DROP TABLE orders; SELECT 1",
    ):
        r = answer_question(db, "attack", writer(bad, bad))
        assert r.rows is None and r.blocked and r.attempts == 2, bad
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 300


def test_evaluate_scores_oracle_100_and_wrong_writer_low(db, capsys):
    gold = {q.question: q.gold for q in QUESTIONS}
    import re

    oracle = lambda p: gold[re.findall(r"Q: (.*)\nSQL:", p)[-1]]  # noqa: E731
    assert all(evaluate(db, oracle, "oracle"))
    wrong = evaluate(db, lambda p: "SELECT COUNT(*) FROM customers", "always-count")
    assert 0 < sum(wrong) < len(wrong) // 2, "a constant query can only luck into a few answers"


TEXTS = [
    "BM25 and RRF are used in hybrid search.",
    "BM25 with RRF fusion.",
    "retry with backoff and jitter",
    "backoff and jitter prevent herds",
    "BM25 only here.",
    "no known terms in this chunk",
]


def test_term_graph_counts_cooccurrence_and_ignores_unknown_chunks():
    g = build_term_graph(TEXTS, ["BM25", "RRF", "hybrid search", "retry", "backoff", "jitter"])
    assert g["BM25"]["RRF"]["weight"] == 2 and g["backoff"]["jitter"]["weight"] == 2
    assert not g.has_edge("BM25", "backoff") and g.number_of_nodes() == 6


def test_communities_partition_the_graph_and_separate_topics():
    g = build_term_graph(TEXTS, ["BM25", "RRF", "hybrid search", "retry", "backoff", "jitter"])
    comms = communities(g, top=10)
    flat = [t for c in comms for t in c]
    assert sorted(flat) == sorted(g.nodes) and len(flat) == len(set(flat))
    assert any({"BM25", "RRF"} <= set(c) for c in comms) and any(
        {"backoff", "jitter"} <= set(c) for c in comms
    )
    assert not any({"BM25", "jitter"} <= set(c) for c in comms)


def test_pipeline_is_still_safe_if_the_validator_has_a_bug(db, monkeypatch):
    """Defence in depth: pretend validation is bypassed; the read-only executor must still protect the data."""
    from types import SimpleNamespace

    import day5_solution as d5

    monkeypatch.setattr(d5, "validate_sql", lambda sql, tables, rows: SimpleNamespace(sql=sql))
    for attack in (
        "DELETE FROM customers",
        "DROP TABLE orders",
        "UPDATE customers SET name = 'pwned'",
    ):
        r = d5.answer_question(db, "attack", writer(attack, attack))
        assert r.rows is None and r.error and "SQL error" in r.error, attack
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 60
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 300
    assert conn.execute("SELECT COUNT(*) FROM customers WHERE name = 'pwned'").fetchone()[0] == 0
    conn.close()
