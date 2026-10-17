"""The team's VSS archive on VAST: upload session clips, search them, pull uploaded clips back.

Every session clip is uploaded with tags (verdict, label, session id) and a liveness-specific
analysis prompt, so the VSS pipeline's own captions describe the head motion and any artefacts.
Then the organisers' search UI, or `search()`, answers "every check that failed at profile".

JWT from POST /api/v1/auth/login with the team USERNAME / PASSWORD (.env.team). Everything here
is a no-op returning None when INGRESS_URL is not set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from config import settings

# Every session clip we upload carries this tag, so searches can stay inside judged sessions.
# Not plain "liveness": the team tags its raw footage with that.
TAG = "liveness-session"

# Max 800 characters (ingest-config custom_prompt_max_length).
ANALYSIS_PROMPT = (
    "This clip is a liveness check from a video call: one person turns their head up, down, "
    "left and right on request. Describe the head motion in order with timestamps. Say whether "
    "the head rotates like one rigid 3D object: ears, nose and jawline appearing and "
    "disappearing consistently, face texture staying attached at profile, hair, neck and "
    "background moving with the head. Name any deepfake or replay artefact you see and when: "
    "texture sliding or flicker at the face edge, warping at profile, a cut or jump, a phone "
    "or monitor bezel, moire, screen glare, a flat photo."
)

_token: str | None = None


def available() -> bool:
    return bool(settings.ingress_url and settings.team_username and settings.team_password)


def _base() -> str:
    return settings.ingress_url.rstrip("/")


def _login() -> str:
    global _token
    r = httpx.post(
        f"{_base()}/api/v1/auth/login",
        json={"username": settings.team_username, "password": settings.team_password},
        timeout=20,
        verify=False,
    )
    r.raise_for_status()
    _token = r.json()["access_token"]
    return _token


def _request(method: str, path: str, **kw) -> httpx.Response:
    """Authenticated call; logs in on first use and once more on a 401."""
    for attempt in (0, 1):
        token = _token or _login()
        r = httpx.request(
            method,
            f"{_base()}{path}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=kw.pop("timeout", 60),
            verify=False,
            **kw,
        )
        if r.status_code == 401 and attempt == 0:
            _login()
            continue
        r.raise_for_status()
        return r
    raise RuntimeError("unreachable")


def tags_for(verdict: str, label: str | None, session_id: str) -> list[str]:
    tags = [TAG, f"verdict-{verdict.lower()}", f"session-{session_id}"]
    if label:
        tags.append(f"label-{label}")
    return tags


def upload(clip: str | Path, verdict: str, label: str | None, session_id: str) -> dict | None:
    """Upload one session clip. Returns {"object_key", "tags"} or None when VSS is not set up."""
    if not available():
        return None
    clip = Path(clip)
    tags = tags_for(verdict, label, session_id)
    with clip.open("rb") as f:
        r = _request(
            "POST",
            "/api/v1/videos/upload",
            files={"file": (f"{session_id}_{clip.name}", f, "video/mp4")},
            data={
                "is_public": "false",
                "tags": ",".join(tags),
                "custom_prompt": ANALYSIS_PROMPT,
                "camera_id": settings.vss_camera_id,
                "location": "video call",
            },
            timeout=120,
        )
    body = r.json()
    return {"object_key": body.get("object_key"), "tags": tags}


def search(
    query: str, top_k: int = 10, tags: list[str] | None = None, min_similarity: float = 0.2
) -> dict[str, Any] | None:
    if not available():
        return None
    body: dict[str, Any] = {
        "query": query,
        "top_k": top_k,
        "llm_top_n": 3,
        "min_similarity": min_similarity,
        "include_public": False,
    }
    if tags:
        body["tags"] = tags
    return _request("POST", "/api/v1/search", json=body).json()


def ask(question: str, top_k: int = 10) -> dict[str, Any] | None:
    """Grounded answer over the archive (VSS agent), e.g. 'which checks failed at profile?'."""
    if not available():
        return None
    return _request("POST", "/api/v1/agent/ask", json={"question": question, "top_k": top_k}).json()


def explore(scope: str = "mine", page: int = 48) -> list[dict[str, Any]]:
    """Every fully indexed upload (the backend calls them chunks), all pages."""
    if not available():
        return []
    out: list[dict[str, Any]] = []
    while True:
        body = _request(
            "GET",
            "/api/v1/videos/explore",
            params={"scope": scope, "limit": page, "offset": len(out)},
        ).json()
        chunks = body.get("chunks") or []
        out += chunks
        if not chunks or len(out) >= int(body.get("total") or 0):
            return out


def download(source: str, dest: Path) -> Path:
    """Fetch one uploaded video (an s3:// original_video or source URI) to a local file."""
    _token or _login()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream(
        "GET",
        f"{_base()}/api/v1/videos/stream",
        params={"source": source, "token": _token},
        timeout=120,
        verify=False,
    ) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return dest
