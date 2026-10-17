from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from config import settings
from index.base import SessionIndex, SessionRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    created_at   REAL NOT NULL,
    challenge_id TEXT NOT NULL,
    instruction  TEXT NOT NULL,
    verdict      TEXT NOT NULL,
    score        REAL NOT NULL,
    reasoning    TEXT NOT NULL,
    label        TEXT,
    evidence     TEXT NOT NULL,
    signals      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_created ON sessions (created_at);
"""

_INSERT = """
INSERT OR REPLACE INTO sessions
    (session_id, created_at, challenge_id, instruction, verdict, score, reasoning, label,
     evidence, signals)
VALUES
    (:session_id, :created_at, :challenge_id, :instruction, :verdict, :score, :reasoning, :label,
     :evidence, :signals)
"""


class SqliteIndex(SessionIndex):
    """One SQLite file, one row per session. The source of truth for the verdict log.

    `evidence` (list) and `signals` (dict) are stored as JSON text. Search (base class) is Cosmos
    Embed1 cosine when vectors exist, keyword overlap otherwise.
    """

    backend = "SQLite"

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else settings.data_dir / "index" / "sessions.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def add(self, record: SessionRecord) -> None:
        row = record.to_dict()
        row["evidence"] = json.dumps(row["evidence"])
        row["signals"] = json.dumps(row["signals"])
        with self.conn:
            self.conn.execute(_INSERT, row)

    def all(self) -> list[SessionRecord]:
        rows = self.conn.execute("SELECT * FROM sessions ORDER BY created_at").fetchall()
        return [self._to_record(r) for r in rows]

    def get(self, session_id: str) -> SessionRecord | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return self._to_record(row) if row else None

    def where(
        self, challenge_id: str | None = None, label: str | None = None
    ) -> list[SessionRecord]:
        """Plain filter, newest first. The kind of question SQLite answers in one line."""
        clauses, params = [], []
        if challenge_id:
            clauses.append("challenge_id = ?")
            params.append(challenge_id)
        if label:
            clauses.append("label = ?")
            params.append(label)
        sql = "SELECT * FROM sessions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        return [self._to_record(r) for r in self.conn.execute(sql, params).fetchall()]

    @staticmethod
    def _to_record(row: sqlite3.Row) -> SessionRecord:
        d = dict(zip(row.keys(), row, strict=True))
        d["evidence"] = json.loads(d["evidence"])
        d["signals"] = json.loads(d["signals"])
        return SessionRecord.from_dict(d)
