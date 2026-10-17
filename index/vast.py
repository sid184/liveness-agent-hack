"""Verdict log on VAST: one row per session in our own table in the team's VastDB bucket.

Table `<VAST_SCHEMA>.<VAST_TABLE>` (default `hackathon.liveness_sessions`), created on first use.
Never `vss-schema` / `vss-collection`: that is the organisers' search index. Connects to the data
VIP (`S3_ENDPOINT`) with the team's ACCESS_KEY / SECRET_KEY, as the organisers' vastdb skills do.

The flat columns (geo, cosmos verdict and confidence, model, prompt version, order, clip) are
there so the table answers questions in VAST directly; `evidence` and `signals` keep the full
record as JSON text so it round-trips. VastDB has no upsert, so a re-added session is a new row
and reads keep the newest row per session_id.
"""

from __future__ import annotations

import json
from typing import Any

from config import settings
from index.base import SessionIndex, SessionRecord

PROTECTED = frozenset({"vss-collection", "vss-prompts-events"})


def _columns():
    import pyarrow as pa

    return pa.schema(
        [
            ("session_id", pa.utf8()),
            ("created_at", pa.float64()),
            ("challenge_id", pa.utf8()),
            ("instruction", pa.utf8()),
            ("verdict", pa.utf8()),
            ("score", pa.float64()),
            ("reasoning", pa.utf8()),
            ("label", pa.utf8()),
            ("geo", pa.float64()),
            ("cosmos_verdict", pa.utf8()),
            ("cosmos_confidence", pa.float64()),
            ("vlm_model", pa.utf8()),
            ("prompt_version", pa.utf8()),
            ("challenge_order", pa.utf8()),
            ("clip", pa.utf8()),
            ("vss_object_key", pa.utf8()),
            ("evidence", pa.utf8()),
            ("signals", pa.utf8()),
        ]
    )


def to_row(r: SessionRecord) -> dict[str, Any]:
    s = r.signals
    vlm = s.get("vlm") or {}
    return {
        "session_id": r.session_id,
        "created_at": float(r.created_at),
        "challenge_id": r.challenge_id,
        "instruction": r.instruction,
        "verdict": r.verdict,
        "score": float(r.score),
        "reasoning": r.reasoning,
        "label": r.label or "",
        "geo": float((s.get("geometric") or {}).get("geo") or 0.0),
        "cosmos_verdict": vlm.get("verdict") or "",
        "cosmos_confidence": float(vlm.get("confidence") or 0.0),
        "vlm_model": s.get("vlm_model") or "",
        "prompt_version": s.get("prompt_version") or "",
        "challenge_order": ",".join(s.get("order") or []),
        "clip": r.evidence[0] if r.evidence else "",
        "vss_object_key": (s.get("vss") or {}).get("object_key") or "",
        "evidence": json.dumps(r.evidence),
        "signals": json.dumps(s),
    }


def from_row(row: dict[str, Any]) -> SessionRecord:
    return SessionRecord(
        challenge_id=row["challenge_id"],
        instruction=row["instruction"],
        verdict=row["verdict"],
        score=float(row["score"]),
        reasoning=row["reasoning"],
        evidence=json.loads(row["evidence"] or "[]"),
        signals=json.loads(row["signals"] or "{}"),
        label=row["label"] or None,
        session_id=row["session_id"],
        created_at=float(row["created_at"]),
    )


def _endpoint(url: str) -> str:
    return url if url.startswith(("http://", "https://")) else f"http://{url}"


class VastIndex(SessionIndex):
    backend = "VastDB"

    def __init__(
        self,
        bucket: str | None = None,
        schema: str | None = None,
        table: str | None = None,
    ):
        import logging

        import urllib3
        import vastdb

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # ssl_verify=False
        # The SDK logs a 400 while probing sorted columns on every select; reads still succeed.
        logging.getLogger("vastdb.errors").setLevel(logging.ERROR)
        self.bucket = bucket or settings.vastdb_bucket
        self.schema = schema or settings.vast_schema
        self.table = table or settings.vast_table
        missing = [
            n
            for n, v in (
                ("S3_ENDPOINT", settings.vdb_endpoint),
                ("ACCESS_KEY", settings.access_key),
                ("SECRET_KEY", settings.secret_key),
                ("VASTDB_BUCKET", self.bucket),
            )
            if not v
        ]
        if missing:
            raise RuntimeError(f"INDEX_BACKEND=vast needs {', '.join(missing)} (.env.team)")
        if self.table in PROTECTED or self.schema == settings.vdb_schema:
            raise RuntimeError("refusing to write into the organisers' VSS schema or tables")
        self.session = vastdb.connect(
            endpoint=_endpoint(settings.vdb_endpoint),
            access=settings.access_key,
            secret=settings.secret_key,
            ssl_verify=False,
        )
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self.session.transaction() as tx:
            bkt = tx.bucket(self.bucket)
            schema = bkt.schema(self.schema, fail_if_missing=False)
            if schema is None:
                schema = bkt.create_schema(self.schema, fail_if_exists=False)
            if schema.table(self.table, fail_if_missing=False) is None:
                schema.create_table(self.table, columns=_columns(), fail_if_exists=False)

    def add(self, record: SessionRecord) -> None:
        import pyarrow as pa

        cols = _columns()
        arrow = pa.Table.from_pylist([to_row(record)], schema=cols)
        with self.session.transaction() as tx:
            tx.bucket(self.bucket).schema(self.schema).table(self.table).insert(arrow)

    def all(self) -> list[SessionRecord]:
        names = [f.name for f in _columns()]
        with self.session.transaction() as tx:
            t = tx.bucket(self.bucket).schema(self.schema).table(self.table)
            rows = t.select(columns=names).read_all().to_pylist()
        newest: dict[str, dict[str, Any]] = {}
        for row in rows:
            cur = newest.get(row["session_id"])
            if cur is None or row["created_at"] >= cur["created_at"]:
                newest[row["session_id"]] = row
        return sorted((from_row(r) for r in newest.values()), key=lambda r: r.created_at)
