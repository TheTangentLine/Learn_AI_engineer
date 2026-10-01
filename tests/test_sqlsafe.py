"""Adversarial tests for common/sqlsafe.py: every layer must hold even if the one before it fails."""

from __future__ import annotations

import sqlite3

import pytest

from common.sqlsafe import UnsafeSQL, run_readonly, validate_sql

ALLOWED = {"customers", "orders"}


@pytest.fixture()
def db(tmp_path):
    path = str(tmp_path / "shop.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, country TEXT);"
        "CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INT, total REAL);"
        "CREATE TABLE secrets(id INTEGER PRIMARY KEY, token TEXT);"
        "INSERT INTO customers VALUES (1,'Ada','UK'),(2,'Lin','CN'),(3,'Sam','US');"
        "INSERT INTO orders VALUES (1,1,10.5),(2,1,20),(3,2,5);"
        "INSERT INTO secrets VALUES (1,'s3cr3t');"
    )
    conn.commit()
    conn.close()
    return path


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM customers",
        "SELECT c.name, SUM(o.total) FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.name",
        "WITH t AS (SELECT * FROM customers) SELECT name FROM t",
        "SELECT name FROM customers UNION SELECT country FROM customers",
        "SELECT * FROM main.customers",
        "SELECT name FROM customers WHERE id IN (SELECT customer_id FROM orders)",
        "SELECT name, ROW_NUMBER() OVER (ORDER BY id) FROM customers",
        "select COUNT(*) from CUSTOMERS",
    ],
)
def test_legitimate_queries_pass(sql):
    v = validate_sql(sql, ALLOWED)
    assert v.sql.upper().startswith(("SELECT", "WITH")) and "LIMIT" in v.sql.upper()


@pytest.mark.parametrize(
    "sql,fragment",
    [
        ("SELECT 1; DROP TABLE customers", "exactly one"),
        ("DROP TABLE customers", "only SELECT"),
        ("UPDATE customers SET name='x'", "only SELECT"),
        ("DELETE FROM orders", "only SELECT"),
        ("INSERT INTO orders VALUES (9,9,9)", "only SELECT"),
        ("REPLACE INTO customers VALUES (1,'x','y')", "only SELECT"),
        ("CREATE TABLE x(a)", "only SELECT"),
        ("PRAGMA table_info(customers)", "only SELECT"),
        ("ATTACH DATABASE 'other.db' AS o", "only SELECT"),
        ("VACUUM", "only SELECT"),
        ("BEGIN", "only SELECT"),
        ("", "empty"),
        ("   ", "empty"),
        ("SELECT * FROM secrets", "not available"),
        ("SELECT * FROM sqlite_master", "not available"),
        ('SELECT * FROM "sqlite_master"', "not available"),
        ("SELECT * FROM [sqlite_master]", "not available"),
        ("SELECT * FROM pragma_table_info('customers')", "table-valued"),
        ("SELECT load_extension('evil.so')", "not allowed"),
        ("SELECT readfile('/etc/passwd')", "not allowed"),
        ("SELECT * FROM temp.customers", "schema"),
        ("SELECT * FROM other.customers", "schema"),
        ("SELECT name FROM customers WHERE id IN (SELECT id FROM secrets)", "not available"),
        (
            "SELECT * FROM customers WHERE name = 'x' UNION SELECT token, 1, 2 FROM secrets",
            "not available",
        ),
        ("SELEC * FROM customers", None),  # unparseable: must be rejected one way or another
        ("SELECT " + "a, " * 3000 + "b FROM customers", "longer"),
    ],
)
def test_dangerous_or_invalid_sql_is_rejected(sql, fragment):
    with pytest.raises(UnsafeSQL) as e:
        validate_sql(sql, ALLOWED)
    if fragment:
        assert fragment in str(e.value)


def test_comments_cannot_smuggle_a_second_statement_and_never_reach_the_database():
    v = validate_sql(
        "SELECT name /* ; DROP TABLE customers */ FROM customers -- ; DROP TABLE orders", ALLOWED
    )
    assert "DROP" not in v.sql.upper() and "--" not in v.sql and "/*" not in v.sql
    assert v.tables == {"customers"}


def test_limit_is_added_and_clamped():
    assert (
        validate_sql("SELECT * FROM customers", ALLOWED, max_rows=7).sql.upper().endswith("LIMIT 7")
    )
    assert (
        validate_sql("SELECT * FROM customers LIMIT 3", ALLOWED, max_rows=7)
        .sql.upper()
        .endswith("LIMIT 3")
    )
    assert (
        validate_sql("SELECT * FROM customers LIMIT 9999", ALLOWED, max_rows=7)
        .sql.upper()
        .endswith("LIMIT 7")
    )


def test_validated_queries_run_and_return_data(db):
    v = validate_sql(
        "SELECT country, COUNT(*) AS n FROM customers GROUP BY country ORDER BY country", ALLOWED
    )
    cols, rows = run_readonly(db, v.sql)
    assert cols == ["country", "n"] and rows == [("CN", 1), ("UK", 1), ("US", 1)]


# ---- layers 4-6 must hold even if the validator is bypassed entirely ----


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM customers",
        "UPDATE customers SET name = 'pwned'",
        "DROP TABLE customers",
        "INSERT INTO customers VALUES (9, 'x', 'y')",
        "CREATE TABLE evil(a)",
        "ATTACH DATABASE ':memory:' AS m",
        "PRAGMA writable_schema = 1",
    ],
)
def test_readonly_connection_and_authorizer_block_writes_without_the_validator(db, sql):
    with pytest.raises(sqlite3.Error):
        run_readonly(db, sql)
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='evil'").fetchone()[0] == 0
    conn.close()


def test_runaway_queries_are_aborted_by_the_step_budget(db):
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    assert validate_sql(sql, ALLOWED).sql  # syntactically a SELECT: only layer 6 can stop it
    with pytest.raises(sqlite3.OperationalError):
        run_readonly(db, sql, max_steps=50_000)


def test_row_cap_is_enforced_even_without_limit(db):
    cols, rows = run_readonly(db, "SELECT * FROM customers", max_rows=2)
    assert len(rows) == 2


def test_error_messages_are_model_safe_and_helpful():
    with pytest.raises(UnsafeSQL) as e:
        validate_sql("SELECT * FROM secrets", ALLOWED)
    msg = str(e.value)
    assert "customers" in msg and "orders" in msg and "Traceback" not in msg


def test_each_layer_holds_on_its_own(db, monkeypatch):
    """Layers mask each other, so test them independently: disable one and prove the next still holds."""
    import sqlite3 as sq

    import common.sqlsafe as ss

    # (a) authorizer disabled -> the read-only connection must still refuse writes
    monkeypatch.setattr(ss, "_authorizer", lambda *a: sq.SQLITE_OK)
    with pytest.raises(sq.OperationalError, match="readonly|read-only"):
        run_readonly(db, "DELETE FROM customers")

    # (b) read-only mode disabled (writable connection) -> the authorizer must still deny writes
    monkeypatch.undo()
    real_connect = sq.connect
    monkeypatch.setattr(
        ss.sqlite3,
        "connect",
        lambda target, **kw: real_connect(target.replace("mode=ro", "mode=rw"), **kw),
    )
    with pytest.raises(sq.DatabaseError):
        run_readonly(db, "DELETE FROM customers")
    monkeypatch.undo()
    conn = real_connect(db)
    assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 3
    conn.close()
