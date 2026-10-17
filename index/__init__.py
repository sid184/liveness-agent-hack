"""Session index: the verdict log the agent writes to and reasons over across sessions.

Swappable. `sqlite` (default) is one database file; `local` is the original JSONL file; `vast` is
our own table in the team's VastDB bucket. All three search with Cosmos Embed1 vectors when the
endpoint is configured and keyword overlap otherwise (`index.base.rank`).
"""

from index.base import SessionIndex, SessionRecord

_vast = None  # one VastDB connection per process


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
        global _vast
        if _vast is None:
            from index.vast import VastIndex

            _vast = VastIndex()
        return _vast
    raise ValueError(f"unknown INDEX_BACKEND {backend!r}")


__all__ = ["SessionIndex", "SessionRecord", "get_index"]
