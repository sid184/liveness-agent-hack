"""Session index: the verdict log the agent writes to and reasons over across sessions.

Swappable. `sqlite` (default) is one database file; `local` is the original JSONL file; `vast`
is filled in on the day. Both local backends use keyword-overlap search until embeddings land.
"""

from index.base import SessionIndex, SessionRecord


def get_index(backend: str | None = None) -> SessionIndex:
    from config import settings

    backend = backend or settings.index_backend
    if backend == "sqlite":
        from index.sqlite import SqliteIndex

        return SqliteIndex()
    if backend == "local":
        from index.local import LocalIndex

        return LocalIndex()
    if backend == "vast":
        raise NotImplementedError("VAST AI OS backend is wired on hackathon day")
    raise ValueError(f"unknown INDEX_BACKEND {backend!r}")


__all__ = ["SessionIndex", "SessionRecord", "get_index"]
