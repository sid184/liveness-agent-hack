"""Verifier overlay backend. Serves the setup tab, the floating overlay and the details page.

Passive: the caller is never asked to do anything. While a call is being watched, the browser
records the caller's tile (screen share of the call window, or a webcam) in back-to-back windows
and posts each one here. Every window runs the session pipeline; after each, `decide_call` says
whether the call is decided (PASS / FAIL / INCONCLUSIVE) or the browser should keep watching.
Run with `make ui`, then open http://localhost:8765 in Chrome.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from agent.loop import MAX_WINDOWS, WINDOW_S, decide_call, finish_call, run_session
from config import settings
from eval import tracing, vss_archive, wandb_log
from index import SessionRecord, get_index

STATIC = Path(__file__).resolve().parent / "static"
SESSIONS = (settings.data_dir / "sessions").resolve()
SESSIONS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="liveness-agent")
if settings.ui_origins:  # the pages are hosted on Vercel and call this API through a tunnel
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.ui_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/media", StaticFiles(directory=SESSIONS), name="media")

_calls: dict[str, list[SessionRecord]] = {}  # call id -> its windows so far, in order
_locks: dict[str, asyncio.Lock] = {}  # one window at a time per call, so decisions stay ordered
_subscribers: set[asyncio.Queue] = set()  # open /api/events streams
_models: dict[str, Any] = {}  # detector and reasoner, built on first verify


def _pipeline() -> dict[str, Any]:
    if not _models:
        from perception import Detector
        from reason import VisionReasoner

        _models["detector"] = Detector()
        _models["objects"] = Detector(settings.yolo_detect_model)
        _models["reasoner"] = VisionReasoner()
        tracing.init()
        # One W&B run for the whole server process; every verdict is a row in it.
        wandb_log.start_run(name="live-ui", job_type="live", config={"source": "ui"})
    return _models


def _transcode(src: Path, dst: Path) -> None:
    """Browser webm has no frame count and a variable rate; give OpenCV a constant-rate mp4.

    Also shrinks to 640 px wide: Mac cameras record 3024 px frames, and YOLO at that size spent
    about 1.5 s per frame in preprocessing (measured 17 Oct). The model runs at 640 anyway.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an", "-r", "15",
        "-vf", "scale=640:-2", "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", str(dst),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(422, f"could not decode the recording: {proc.stderr.strip()[:300]}")


def _media_url(path: str) -> str | None:
    try:
        return "/media/" + Path(path).resolve().relative_to(SESSIONS).as_posix()
    except ValueError:
        return None


def _public(record: SessionRecord, with_signals: bool = False) -> dict[str, Any]:
    d = record.to_dict()
    urls = [u for u in map(_media_url, record.evidence) if u]
    d["frames"] = [u for u in urls if u.endswith(".jpg")]
    d["clip"] = next((u for u in urls if u.endswith(".mp4")), None)
    d["cited"] = (record.signals.get("vlm") or {}).get("cited_frames") or []
    d["assurance"] = (record.signals.get("call") or {}).get("assurance") or record.signals.get(
        "assurance"
    )
    d["stack"] = record.signals.get("stack") or []
    if with_signals:
        d["signals"].pop("embedding", None)  # 256 floats nobody reads
    else:
        d.pop("signals")
    return d


def _summary(record: SessionRecord) -> dict[str, Any]:
    call = record.signals.get("call") or {}
    return {
        k: getattr(record, k)
        for k in ("session_id", "challenge_id", "verdict", "score", "instruction", "reasoning")
    } | {
        "created_at": record.created_at,
        "label": record.label,
        "windows": len(call.get("windows") or []) or None,  # set on a call's deciding window
        "assurance": call.get("assurance") or record.signals.get("assurance"),
        "window_no": record.signals.get("window_no"),
    }


@app.get("/")
def setup_page() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/session/{session_id}")
def session_page(session_id: str) -> FileResponse:
    return FileResponse(STATIC / "session.html")


@app.post("/api/watch/start")
def watch_start() -> dict[str, Any]:
    """Open a call. The browser then records windows of `window_s` and posts each one."""
    call_id = uuid.uuid4().hex[:10]
    _calls[call_id] = []
    _locks[call_id] = asyncio.Lock()
    return {"call_id": call_id, "window_s": WINDOW_S, "max_windows": MAX_WINDOWS}


@app.post("/api/watch/window")
async def watch_window(
    call_id: Annotated[str, Form()],
    clip: Annotated[UploadFile, File()],
    label: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    """Judge one window of the call; return the call's decision so far."""
    if call_id not in _calls:
        raise HTTPException(404, "unknown call; POST /api/watch/start first")
    data = await clip.read()
    lock = _locks.get(call_id)
    if lock is None:  # decided and cleaned up while this window was uploading
        raise HTTPException(409, "this call is already decided")
    async with lock:
        windows = _calls.get(call_id)
        if windows is None or decide_call(
            [w.signals.get("window_verdict") or w.verdict for w in windows]
        ) not in ("WATCHING",):
            raise HTTPException(409, "this call is already decided")
        folder = SESSIONS / uuid.uuid4().hex[:12]
        folder.mkdir(parents=True)
        raw = folder / "clip.upload"
        raw.write_bytes(data)
        mp4 = folder / "clip.mp4"
        await run_in_threadpool(_transcode, raw, mp4)
        raw.unlink()

        models = _pipeline()
        try:
            record = await run_in_threadpool(
                run_session,
                mp4,
                label=label if label in ("live", "attack") else None,
                detector=models["detector"],
                reasoner=models["reasoner"],
                objects=models["objects"],
                evidence_dir=folder,
                call_id=call_id,
                window_no=len(windows) + 1,
            )
        except Exception as e:  # VLM endpoint down, unreadable clip: show it in the overlay
            raise HTTPException(502, f"{type(e).__name__}: {e}") from e
        windows.append(record)
        decision = decide_call([w.verdict for w in windows])
        if decision != "WATCHING":
            record = await run_in_threadpool(finish_call, windows, decision)
        wandb_log.log_session(record)

    out = _public(record) | {
        "call_id": call_id,
        "decision": decision,
        "window_no": len(windows),
        "max_windows": MAX_WINDOWS,
    }
    if decision != "WATCHING":
        out["similar"] = [_summary(r) for r in get_index().similar(record, k=3)]
        _calls.pop(call_id, None)
        _locks.pop(call_id, None)
    return out


@app.get("/api/sessions")
def sessions(limit: int = 20) -> list[dict[str, Any]]:
    """Decided calls (and one-off clips), newest first; a call's undecided windows are left out."""
    records = sorted(get_index().all(), key=lambda r: r.created_at, reverse=True)
    shown = [r for r in records if not r.signals.get("call_id") or r.signals.get("call")]
    return [_summary(r) for r in shown[:limit]]


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: str) -> dict[str, Any]:
    index = get_index()
    record = index.get(session_id)
    if record is None:
        raise HTTPException(404, "no such session")
    similar = [_summary(r) for r in index.similar(record, k=3)]
    return _public(record, with_signals=True) | {"similar": similar}


@app.get("/api/archive/search")
def archive_search(q: str, top_k: int = 10) -> dict[str, Any]:
    """Search the team's VSS archive of session clips, e.g. q='face texture slides at profile'."""
    if not vss_archive.available():
        raise HTTPException(503, "VSS archive not configured (.env.team)")
    res = vss_archive.search(q, top_k=top_k, tags=[vss_archive.TAG]) or {}
    return {
        "query": q,
        "answer": (res.get("llm_synthesis") or {}).get("response"),
        "hits": [
            {
                "video": c.get("original_video"),
                "similarity": c.get("similarity_score"),
                "start_s": c.get("best_match_start_sec"),
                "end_s": c.get("best_match_end_sec"),
            }
            for c in res.get("chunk_results") or []
        ],
    }


class TriggerRequest(BaseModel):
    reason: str | None = None  # e.g. "new payee, £2m"; shown in the overlay


@app.post("/api/trigger")
async def trigger(req: TriggerRequest | None = None) -> dict[str, Any]:
    """Start watching from outside: a payment approval step, a new-payee event, or curl."""
    event = {"type": "trigger"} | (req.model_dump() if req else {})
    for q in list(_subscribers):
        q.put_nowait(event)
    return {"delivered_to": len(_subscribers)}


@app.get("/api/events")
async def events() -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue()
    _subscribers.add(queue)

    async def stream():
        try:
            yield ": connected\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _subscribers.discard(queue)

    return StreamingResponse(stream(), media_type="text/event-stream")
