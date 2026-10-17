from __future__ import annotations

import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

_WORD = re.compile(r"[a-z0-9]+")


def tokens(s: str) -> set[str]:
    """Lower-case word set, the unit of the keyword-overlap search both backends use for now."""
    return set(_WORD.findall(s.lower()))


@dataclass
class SessionRecord:
    challenge_id: str
    instruction: str
    verdict: str  # PASS | FAIL | ERROR
    score: float  # 0..1, higher = more confident it is a live human
    reasoning: str  # the agent's written justification
    evidence: list[str] = field(default_factory=list)  # paths to clips / frames
    signals: dict[str, Any] = field(default_factory=dict)  # raw YOLO + VLM outputs
    label: str | None = None  # ground truth when known: live | attack
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SessionRecord:
        return cls(**d)

    def text(self) -> str:
        """What gets embedded / searched."""
        return f"{self.challenge_id} {self.verdict} {self.instruction} {self.reasoning}"


def rank(
    query: str,
    records: list[SessionRecord],
    k: int = 5,
    query_vec: list[float] | None = None,
) -> list[SessionRecord]:
    """Most relevant records for a query. Cosmos Embed1 cosine when the query and the records
    carry vectors (`signals["embedding"]`), keyword overlap otherwise. Newest first on ties."""
    vectors = [r.signals.get("embedding") for r in records]
    if query_vec and any(vectors):
        from index.embed import cosine

        scored = [
            (cosine(query_vec, v) if v else -1.0, r.created_at, r)
            for v, r in zip(vectors, records, strict=True)
        ]
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [r for s, _, r in scored[:k] if s > 0]
    q = tokens(query)
    scored_kw = [(len(q & tokens(r.text())), r.created_at, r) for r in records]
    scored_kw.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [r for s, _, r in scored_kw[:k] if s > 0]


class SessionIndex(ABC):
    backend = "base"  # shown on the session page next to "Verdict log"

    @abstractmethod
    def add(self, record: SessionRecord) -> None: ...

    @abstractmethod
    def all(self) -> list[SessionRecord]: ...

    def search(
        self, query: str, k: int = 5, query_vec: list[float] | None = None
    ) -> list[SessionRecord]:
        """Most relevant past sessions for a natural-language query."""
        if query_vec is None:
            from index.embed import embed

            query_vec = embed(query)
        return rank(query, self.all(), k, query_vec)

    def get(self, session_id: str) -> SessionRecord | None:
        return next((r for r in self.all() if r.session_id == session_id), None)

    def where(
        self, challenge_id: str | None = None, label: str | None = None
    ) -> list[SessionRecord]:
        """Plain filter, newest first."""
        rows = [
            r
            for r in self.all()
            if (challenge_id is None or r.challenge_id == challenge_id)
            and (label is None or r.label == label)
        ]
        return sorted(rows, key=lambda r: r.created_at, reverse=True)

    def similar(self, record: SessionRecord, k: int = 3) -> list[SessionRecord]:
        """Past sessions most like this one, excluding itself."""
        hits = self.search(record.text(), k + 1, query_vec=record.signals.get("embedding"))
        return [r for r in hits if r.session_id != record.session_id][:k]
