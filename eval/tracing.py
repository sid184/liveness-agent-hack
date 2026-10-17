"""Weave tracing: one trace per session (YOLO -> Cosmos -> agent), no-op without a W&B key.

`op` decorates a function as a Weave op once `init()` has run; before that, or without a key,
it is a plain call. Weave also patches the OpenAI client, so the Cosmos and agent LLM requests
show up as children of the op that made them.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from config import settings

_client = None
_tried = False


def init() -> bool:
    """Start Weave for this process. Safe to call many times; returns whether tracing is on."""
    global _client, _tried
    if _tried:
        return _client is not None
    _tried = True
    if not (settings.weave_enabled and settings.wandb_api_key):
        return False
    try:
        import weave

        project = settings.wandb_project
        if settings.wandb_entity:
            project = f"{settings.wandb_entity}/{project}"
        _client = weave.init(project)
    except Exception as e:  # never let tracing break a verdict
        print(f"weave disabled: {type(e).__name__}: {e}")
        _client = None
    return _client is not None


def enabled() -> bool:
    return _client is not None


def _drop(keys: tuple[str, ...]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Inputs too big or unserialisable to trace (raw frames) become a short description."""

    def post(inputs: dict[str, Any]) -> dict[str, Any]:
        out = dict(inputs)
        for k in keys:
            v = out.get(k)
            if v is not None:
                out[k] = f"<{len(v)} items>" if hasattr(v, "__len__") else f"<{type(v).__name__}>"
        out.pop("self", None)
        return out

    return post


def op(name: str, drop: tuple[str, ...] = ()) -> Callable:
    """Trace the decorated function as Weave op `name` when tracing is on."""

    def wrap(fn: Callable) -> Callable:
        traced: Callable | None = None

        @functools.wraps(fn)
        def call(*args, **kwargs):
            nonlocal traced
            if _client is None:
                return fn(*args, **kwargs)
            if traced is None:
                import weave

                traced = weave.op(fn, name=name, postprocess_inputs=_drop(drop))
            return traced(*args, **kwargs)

        return call

    return wrap
