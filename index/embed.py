"""Cosmos Embed1 text embeddings (256-dim) on the organisers' GPU, for semantic session search.

`embed()` returns None when the endpoint is not configured or fails, and callers fall back to
keyword overlap, so nothing breaks away from the event network.
"""

from __future__ import annotations

import math

import httpx

from config import settings

DIM = 256


def available() -> bool:
    return bool(settings.cosmos_embed1_url)


def embed(text: str, timeout_s: float = 10.0) -> list[float] | None:
    if not available() or not text.strip():
        return None
    headers = {}
    if settings.gpu_bearer_token:
        headers["Authorization"] = f"Bearer {settings.gpu_bearer_token}"
    url = settings.cosmos_embed1_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    try:
        r = httpx.post(
            f"{url}/embeddings",
            headers=headers,
            json={
                "input": text[:2000],
                "model": settings.embed_model,
                "request_type": "query",
                "encoding_format": "float",
            },
            timeout=timeout_s,
        )
        r.raise_for_status()
        vec = r.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"embed failed, keyword search instead: {type(e).__name__}: {e}")
        return None
    return [float(x) for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
