"""Thin W&B wrapper: no-op when WANDB_API_KEY is missing so local runs never block on login."""

from __future__ import annotations

from typing import Any

from config import settings
from index.base import SessionRecord

_run = None


def start_run(
    name: str | None = None,
    config: dict[str, Any] | None = None,
    job_type: str | None = None,
    group: str | None = None,
):
    global _run
    if not settings.wandb_api_key:
        return None
    import wandb

    _run = wandb.init(
        project=settings.wandb_project,
        entity=settings.wandb_entity or None,
        name=name,
        config=config or {},
        job_type=job_type,
        group=group,
        reinit="finish_previous",
    )
    return _run


def active() -> bool:
    return _run is not None


def url() -> str | None:
    return getattr(_run, "url", None) if _run is not None else None


def log(data: dict[str, Any]) -> None:
    if _run is not None:
        _run.log(data)


def log_session(record: SessionRecord) -> None:
    """Mirror one session record to W&B, flattened. The index is the source of truth; this is
    the same record, so the two can never disagree."""
    if _run is None:
        return
    s = record.signals
    row: dict[str, Any] = {
        "session_id": record.session_id,
        "challenge": record.challenge_id,
        "verdict": record.verdict,
        "score": record.score,
        "label": record.label or "",
        "vlm_model": s.get("vlm_model"),
        "vlm_latency_s": s.get("vlm_latency_s"),
        "vlm_raw": s.get("vlm_raw"),
        "prompt_version": s.get("prompt_version"),
        "challenge_by": s.get("challenge_by"),
        "order": ",".join(s.get("order") or []),
        "agent_note": (s.get("agent") or {}).get("text"),
        "vss_object_key": (s.get("vss") or {}).get("object_key"),
        "reasoning": record.reasoning,
    }
    row |= {
        f"latency.{st['step']}": st["latency_s"]
        for st in s.get("stack") or []
        if st.get("latency_s") is not None
    }
    row |= {f"geo.{k}": v for k, v in (s.get("geometric") or {}).items()}
    row |= {f"vlm.{k}": v for k, v in (s.get("vlm") or {}).items() if v is not None}
    _run.log(row)


def finish() -> None:
    global _run
    if _run is not None:
        _run.finish()
        _run = None
