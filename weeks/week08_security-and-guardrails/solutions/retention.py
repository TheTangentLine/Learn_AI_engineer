"""Retention and erasure for the support system's SQLite store: personal data should live exactly as long as there is a reason to keep it.

  purge(db, max_age_s, now)        delete events and escalations older than the limit, and conversations with no recent activity
  erase_conversation(db, cid)      a person's request to be forgotten: remove that conversation everywhere in this database

Deliberately NOT done: deleting the refund LEDGER. Financial records have their own legal retention period, so the row stays (amount,
invoice, refund id) and only its key, which embeds the conversation id, is replaced by a one-way tombstone: a policy decision to confirm
with whoever owns the legal requirement, never a default. Nothing here can reach copies OUTSIDE the database (backups, model-provider logs,
trace backends): list each of those and give it its own retention and its own deletion path.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass


@dataclass
class PurgeReport:
    events: int
    escalations: int
    conversations: int
    workflow: int = 0  # rows of the refund workflow's own tables (approvals, checkpoints, writes)
    ledger_unlinked: int = 0  # ledger rows kept, with the link to the conversation removed

    @property
    def total(self) -> int:
        return (
            self.events
            + self.escalations
            + self.conversations
            + self.workflow
            + self.ledger_unlinked
        )


def _has(con: sqlite3.Connection, table: str) -> bool:
    return (
        con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def _tombstone(key: str) -> str:
    return "erased-" + hashlib.sha256(key.encode()).hexdigest()[:10]


INVOICE = r"[A-Z]+-\d+"


def _exact_keys(
    con: sqlite3.Connection, table: str, column: str, prefix: str, cid: str, suffix: str = ""
) -> list[str]:
    """Values of ``column`` of the form ``<prefix><cid>-<INVOICE>`` and nothing else. A prefix match alone is ambiguous: with conversations
    "a" and "a-b", the key "a-b-INV-3002" starts with "a-" but belongs to "a-b" (the workflow's key format cannot tell them apart by
    prefix, which is itself a finding about that format)."""
    like = (prefix + cid).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "-%"
    pattern = re.compile(re.escape(prefix + cid) + "-" + INVOICE + suffix)
    rows = con.execute(
        f"SELECT DISTINCT {column} FROM {table} WHERE {column} LIKE ? ESCAPE '\\'", (like,)
    ).fetchall()  # noqa: S608
    return [r[0] for r in rows if pattern.fullmatch(r[0])]


def _erase_workflow(con: sqlite3.Connection, conversation_ids: list[str]) -> tuple[int, int]:
    """The refund flow keys its rows by ``<conversation>-<invoice>``. Approvals and LangGraph checkpoints hold the conversation id and the
    model-written reason: delete them. The refund LEDGER row is a financial record: keep the amount and invoice, replace the key (which embeds
    the conversation id) with a one-way tombstone so the row can no longer be tied to the person's conversation."""
    deleted = unlinked = 0
    for cid in conversation_ids:
        if _has(con, "approvals"):
            for key in _exact_keys(con, "approvals", "ticket_id", "", cid):
                deleted += con.execute("DELETE FROM approvals WHERE ticket_id = ?", (key,)).rowcount
        for table in ("checkpoints", "writes"):
            if _has(con, table):
                for key in _exact_keys(con, table, "thread_id", "refund:", cid):
                    deleted += con.execute(
                        f"DELETE FROM {table} WHERE thread_id = ?", (key,)
                    ).rowcount  # noqa: S608
        if _has(con, "refunds"):
            for key in _exact_keys(con, "refunds", "key", "", cid, suffix=":" + INVOICE):
                # the ledger key is "<ticket>:<invoice>"; the ticket part is what embeds the conversation id
                con.execute("UPDATE refunds SET key = ? WHERE key = ?", (_tombstone(key), key))
                unlinked += 1
    return deleted, unlinked


def purge(db_path: str, max_age_s: float, *, now: float | None = None) -> PurgeReport:
    """Delete everything older than ``max_age_s`` seconds. A conversation is kept while it has ANY recent event; when a conversation goes,
    so do its workflow rows, and its ledger rows are unlinked (kept as financial records)."""
    now = time.time() if now is None else now
    cutoff = now - max_age_s
    con = sqlite3.connect(db_path, timeout=30)
    try:
        with con:
            ev = con.execute("DELETE FROM events WHERE at < ?", (cutoff,)).rowcount
            es = con.execute("DELETE FROM escalations WHERE at < ?", (cutoff,)).rowcount
            # a conversation with no remaining events is stale (the table has no timestamp of its own)
            stale = [
                r[0]
                for r in con.execute(
                    "SELECT id FROM conversations WHERE id NOT IN (SELECT DISTINCT conversation_id FROM events WHERE conversation_id IS NOT NULL)"
                )
            ]
            for cid in stale:
                con.execute("DELETE FROM conversations WHERE id = ?", (cid,))
            wf, unlinked = _erase_workflow(con, stale)
        return PurgeReport(ev, es, len(stale), wf, unlinked)
    finally:
        con.close()


def erase_conversation(db_path: str, conversation_id: str) -> PurgeReport:
    """A person's request to be forgotten, for everything this database holds about one conversation."""
    con = sqlite3.connect(db_path, timeout=30)
    try:
        with con:
            ev = con.execute(
                "DELETE FROM events WHERE conversation_id = ?", (conversation_id,)
            ).rowcount
            es = con.execute(
                "DELETE FROM escalations WHERE conversation_id = ?", (conversation_id,)
            ).rowcount
            cv = con.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,)).rowcount
            wf, unlinked = _erase_workflow(con, [conversation_id])
        return PurgeReport(ev, es, cv, wf, unlinked)
    finally:
        con.close()


def dump(db_path: str) -> str:
    """Every row of every table as text: for tests and audits that ask 'is this value anywhere in the database?'"""
    con = sqlite3.connect(db_path)
    try:
        out = []
        for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            for row in con.execute(f"SELECT * FROM {name}"):  # noqa: S608 - table names come from sqlite_master
                out.append(f"{name}: {row}")
        return "\n".join(out)
    finally:
        con.close()
