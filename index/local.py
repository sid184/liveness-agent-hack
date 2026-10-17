from __future__ import annotations

import json
from pathlib import Path

from config import settings
from index.base import SessionIndex, SessionRecord


class LocalIndex(SessionIndex):
    """JSONL on disk. The original backend; SqliteIndex is the default."""

    backend = "JSONL"

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
