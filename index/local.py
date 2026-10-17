from __future__ import annotations

import json
from pathlib import Path

from config import settings
from index.base import SessionIndex, SessionRecord, tokens


class LocalIndex(SessionIndex):
    """JSONL on disk, keyword-overlap search. The original backend; SqliteIndex is the default."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else settings.data_dir / "index" / "sessions.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: SessionRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def all(self) -> list[SessionRecord]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [SessionRecord.from_dict(json.loads(line)) for line in f if line.strip()]

    def search(self, query: str, k: int = 5) -> list[SessionRecord]:
        q = tokens(query)
        scored = [(len(q & tokens(r.text())), r.created_at, r) for r in self.all()]
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [r for s, _, r in scored[:k] if s > 0]
