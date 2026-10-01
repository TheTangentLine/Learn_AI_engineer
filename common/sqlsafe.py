"""Safe execution of LLM-generated SQL (SQLite). Treat the model's SQL as UNTRUSTED INPUT.

    from common.sqlsafe import validate_sql, run_readonly, UnsafeSQL
    v = validate_sql(model_sql, allowed_tables={"customers", "orders"})   # raises UnsafeSQL
    cols, rows = run_readonly("shop.db", v.sql)                           # read-only, row- and step-limited

Defence in depth: each layer assumes the previous one failed.
  1. PARSE      sqlglot builds an AST; exactly one statement; only SELECT / set operations (UNION...).
  2. ALLOW-LIST every referenced table must be in `allowed_tables` (so sqlite_master, pragma_* and
                attached databases are out); dangerous functions are blocked.
  3. REWRITE    we execute the SQL *re-generated from the AST* (comments and odd formatting never reach
                the database) with a LIMIT added or clamped.
  4. READ-ONLY  the connection is opened with mode=ro, so even a validator bug cannot write.
  5. AUTHORIZER SQLite's authorizer callback denies every action except SELECT/READ/FUNCTION.
  6. BUDGET     a progress handler aborts runaway queries (e.g. infinite recursive CTEs).

None of this makes it safe to give the model access to *sensitive rows*: row-level security and
least-privilege database roles are still your job (Week 8).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

MAX_SQL_CHARS = 4000
BLOCKED_FUNCTIONS = {
    "load_extension",
    "readfile",
    "writefile",
    "edit",
    "fts3_tokenizer",
    "zipfile",
    "sqlar_compress",
    "sqlite_compileoption_get",
    "sqlite_source_id",
    "sqlite_version",
    "randomblob",
    "zeroblob",
}
BLOCKED_PREFIXES = ("pragma_", "sqlite_")
FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Command,
    exp.Pragma,
    exp.Attach,
    exp.Detach,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Merge,
    exp.Set,
)


class UnsafeSQL(Exception):
    """The SQL was rejected; the message is safe to show to the model so it can try again."""


@dataclass
class Validated:
    sql: str  # the SQL to execute: regenerated from the AST, with a LIMIT
    tables: set[str]


def validate_sql(sql: str, allowed_tables: set[str], max_rows: int = 100) -> Validated:
    allowed = {t.lower() for t in allowed_tables}
    if not sql or not sql.strip():
        raise UnsafeSQL("empty query")
    if len(sql) > MAX_SQL_CHARS:
        raise UnsafeSQL(f"query longer than {MAX_SQL_CHARS} characters")
    try:
        statements = sqlglot.parse(sql, dialect="sqlite")
    except SqlglotError as exc:
        raise UnsafeSQL(f"could not parse the SQL: {str(exc)[:120]}") from exc
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise UnsafeSQL("exactly one SQL statement is allowed")
    tree = statements[0]
    if not isinstance(tree, exp.Select | exp.SetOperation):
        raise UnsafeSQL(f"only SELECT queries are allowed (got {type(tree).__name__.upper()})")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise UnsafeSQL(f"forbidden operation: {type(node).__name__.upper()}")

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables: set[str] = set()
    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        if not isinstance(t.this, exp.Identifier) or not name:
            raise UnsafeSQL("table-valued functions are not allowed")
        if t.args.get("db") and t.args["db"].name.lower() not in ("main",):
            raise UnsafeSQL(f"schema '{t.args['db'].name}' is not allowed")
        if name in cte_names:
            continue
        if name.startswith(BLOCKED_PREFIXES) or name not in allowed:
            raise UnsafeSQL(
                f"table '{t.name}' is not available (allowed: {', '.join(sorted(allowed))})"
            )
        tables.add(name)

    for f in tree.find_all(exp.Func):
        fname = (f.name if isinstance(f, exp.Anonymous) else f.sql_name()).lower()
        if fname in BLOCKED_FUNCTIONS or fname.startswith(BLOCKED_PREFIXES):
            raise UnsafeSQL(f"function '{fname}' is not allowed")

    limit = tree.args.get("limit")
    if limit is None:
        tree = tree.limit(max_rows)
    else:
        try:
            if int(limit.expression.name) > max_rows:
                tree = tree.limit(max_rows)
        except (ValueError, AttributeError):
            tree = tree.limit(max_rows)
    return Validated(tree.sql(dialect="sqlite", comments=False), tables)


_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


def _authorizer(action, arg1, arg2, db_name, trigger):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def run_readonly(
    db_path: str, sql: str, max_rows: int = 100, max_steps: int = 2_000_000
) -> tuple[list[str], list[tuple]]:
    """Execute already-validated SQL on a read-only connection with an authorizer and a step budget.

    Raises sqlite3.OperationalError on denied actions, budget overrun or SQL errors."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.set_authorizer(_authorizer)
        steps = {"n": 0}

        def budget() -> int:
            steps["n"] += 1000  # called every 1000 VM instructions
            return 1 if steps["n"] > max_steps else 0  # non-zero aborts the query

        conn.set_progress_handler(budget, 1000)
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description or []]
        return cols, cur.fetchmany(max_rows)
    finally:
        conn.close()
