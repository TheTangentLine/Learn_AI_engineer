"""Durable conversation state (SQLite): which agent owns a conversation, its history, an event log, escalations."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


class Store:
    def __init__(self, path: str | Path, clock=time.time):
        self.path = str(path)
        self.clock = clock
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, agent TEXT, history TEXT NOT NULL DEFAULT '[]');
                CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT, at REAL, kind TEXT, detail TEXT);
                CREATE TABLE IF NOT EXISTS escalations (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT, at REAL, reason TEXT, resolved INTEGER DEFAULT 0);
                """
            )

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ---- conversations
    def get(self, conversation_id: str) -> tuple[str | None, list[dict]]:
        with self._conn() as c:
            row = c.execute(
                "SELECT agent, history FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return (row[0], json.loads(row[1])) if row else (None, [])

    def save(self, conversation_id: str, agent: str | None, history: list[dict]) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO conversations (id, agent, history) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET agent = excluded.agent, history = excluded.history",
                (conversation_id, agent, json.dumps(history)),
            )

    def conversations(self) -> list[str]:
        with self._conn() as c:
            return [r[0] for r in c.execute("SELECT id FROM conversations ORDER BY rowid")]

    # ---- events (the audit log)
    def log(self, conversation_id: str, kind: str, **detail) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO events (conversation_id, at, kind, detail) VALUES (?, ?, ?, ?)",
                (conversation_id, self.clock(), kind, json.dumps(detail, default=str)),
            )

    def events(self, conversation_id: str | None = None, kind: str | None = None) -> list[dict]:
        q, args = "SELECT conversation_id, at, kind, detail FROM events WHERE 1=1", []
        if conversation_id:
            q, args = q + " AND conversation_id = ?", args + [conversation_id]
        if kind:
            q, args = q + " AND kind = ?", args + [kind]
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY id", args).fetchall()
        return [
            {"conversation_id": r[0], "at": r[1], "kind": r[2], **json.loads(r[3])} for r in rows
        ]

    # ---- human escalations
    def escalate(self, conversation_id: str, reason: str) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO escalations (conversation_id, at, reason) VALUES (?, ?, ?)",
                (conversation_id, self.clock(), reason),
            )
            return cur.lastrowid

    def escalations(self, open_only: bool = True) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, conversation_id, at, reason, resolved FROM escalations"
                + (" WHERE resolved = 0" if open_only else "")
                + " ORDER BY id"
            ).fetchall()
        return [
            dict(zip(("id", "conversation_id", "at", "reason", "resolved"), r, strict=True))
            for r in rows
        ]
