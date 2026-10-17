"""The agent: a text model on W&B Inference that writes the call's decision note.

It never decides the verdict (that is `agent.loop.combine`, a fixed formula). Every call has a
fallback, so a W&B outage changes the wording of a session, never its outcome.
"""

from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from config import settings
from eval.tracing import op
from index import SessionRecord
from reason.prompts import VERDICT_WRITER


class AgentLLM:
    """Text model on W&B Inference, OpenAI-compatible."""

    def __init__(
        self, base_url: str | None = None, api_key: str | None = None, model: str | None = None
    ):
        headers = {}
        if settings.wandb_entity:
            headers["OpenAI-Project"] = f"{settings.wandb_entity}/{settings.wandb_project}"
        self.model = model or settings.agent_model
        self.client = OpenAI(
            base_url=base_url or settings.agent_base_url,
            api_key=api_key or settings.agent_api_key,
            default_headers=headers or None,
            timeout=20.0,
            max_retries=1,
        )

    def chat(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return resp.choices[0].message.content or ""


def enabled() -> bool:
    return settings.agent_enabled and bool(settings.agent_api_key)


def _window_line(i: int, r: SessionRecord) -> str:
    s = r.signals
    vlm = s.get("vlm") or {}
    gates = (s.get("geometric") or {}).get("gates")
    return (
        f"{i}. {s.get('window_verdict') or r.verdict} score {r.score:.2f}; YOLO gates {gates}; "
        f"Cosmos {vlm.get('verdict')} at {vlm.get('confidence')}: {vlm.get('observed')}; "
        f"artifacts: {vlm.get('artifacts') or 'none'}"
    )


def _similar_line(r: SessionRecord) -> str:
    label = f", ground truth {r.label}" if r.label else ""
    return f"[{r.verdict} {r.score:.2f}{label}] {r.reasoning[:160]}"


@op("agent.write_verdict")
def write_verdict(
    windows: list[SessionRecord],
    decision: str,
    similar: list[SessionRecord],
    threshold: float,
    llm: AgentLLM | None = None,
) -> dict[str, Any] | None:
    """The decision note. None when the agent is off; {"error"} when it fails. Callers keep
    Cosmos's own text in either case."""
    if not enabled():
        return None
    prompt = VERDICT_WRITER.format(
        decision=decision,
        n=len(windows),
        threshold=threshold,
        windows="\n".join(_window_line(i, w) for i, w in enumerate(windows, 1)),
        similar="; ".join(_similar_line(r) for r in similar) or "none yet",
    )
    t0 = time.perf_counter()
    try:
        llm = llm or AgentLLM()
        text = llm.chat([{"role": "user", "content": prompt}], max_tokens=220).strip()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}
    if not text:
        return None
    return {
        "text": text,
        "model": llm.model,
        "latency_s": round(time.perf_counter() - t0, 2),
        "similar_ids": [r.session_id for r in similar],
    }
