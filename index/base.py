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


class SessionIndex(ABC):
    @abstractmethod
    def add(self, record: SessionRecord) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[SessionRecord]:
        """Most relevant past sessions for a natural-language query."""

    @abstractmethod
    def all(self) -> list[SessionRecord]: ...
