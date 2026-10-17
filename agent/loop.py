"""Passive liveness during a call: the caller is never asked to do anything.

The call is recorded in short windows. Each window is one session: YOLO pose measures the head
(one face, no jumps, enough natural movement), Cosmos judges whether that movement is physically
coherent, and `combine` gives the window a verdict. `decide_call` turns the windows of one call
into a decision: FAIL on the first failed window, PASS after enough clean ones, otherwise keep
watching until the window budget runs out (INCONCLUSIVE). `finish_call` has the agent write the
decision note.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from agent import llm as agent_llm
from capture.frames import read_frames, sample_frames
from config import settings
from eval import tracing, vss_archive
from eval.tracing import op
from index import SessionRecord, get_index
from index.embed import embed
from perception import Detection, Detector, HeadPose, max_jump, motion_range
from perception.cues import FrameCue, add_objects, cues_from_detections, frame_diff, summarise
from reason import VisionReasoner
from reason.prompts import COHERENCE_VERSIONS

_FIELD = re.compile(r"^(OBSERVED|ARTIFACTS|COHERENT|VERDICT|CONFIDENCE):\s*(.*)$", re.M)
_FRAME_NUM = re.compile(r"\bframes?\s*(\d+(?:\s*(?:,|and|-|to)\s*\d+)*)", re.I)

MODE = "passive_watch"  # SessionRecord.challenge_id: there is no challenge any more
INSTRUCTION = "None. The caller was not asked to do anything; this is natural conversation."

# Scoring constants. Tune on the labelled clips, not by feel.
THRESHOLD = 0.5
JUMP_PENALTY = 0.5  # a cut or glitch between frames halves the Cosmos score
MAX_JUMP = 0.8  # (yaw, pitch) jump between geometry frames above this is a cut, replay or glitch
MIN_FACE_FRACTION = (
    0.6  # frames with exactly one measurable face; a real turn loses a few at profile
)
# A visible head turn: a natural glance of about 30 degrees (Sid, measured 17 Oct on live windows).
# Informational; the hard frames, not this, decide the assurance level.
MIN_YAW_RANGE = 0.35
MIN_PITCH_RANGE = 0.4
N_GEO_FRAMES = 40  # YOLO is cheap: about 4 fps over a 10 s clip, enough to see jumps
OBJECT_EVERY = 4  # the detection pass (devices, cups) runs on every 4th pose frame

# Call decision.
WINDOW_S = 8  # seconds per recorded window
PASSES_NEEDED = 2  # clean windows before the caller is cleared
MAX_WINDOWS = 8  # then stop watching: INCONCLUSIVE


def parse_verdict(text: str) -> dict[str, Any]:
    """Pull the structured fields out of a COHERENCE response. Missing fields are None.

    With the think/answer prompt (v3) only the <answer> block is parsed."""
    answer = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.S)
    if answer:
        text = answer.group(1)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    # Models sometimes copy the template's angle brackets: "VERDICT: <PASS>", "ARTIFACTS: <none>".
    fields = {k.lower(): v.strip().strip("<>").strip() for k, v in _FIELD.findall(text)}
    verdict = (fields.get("verdict") or "").upper()
    conf_match = re.search(r"\d+", fields.get("confidence") or "")
    artifacts = fields.get("artifacts")
    cited: list[int] = []
    for m in _FRAME_NUM.finditer(artifacts or ""):
        cited += [int(n) for n in re.findall(r"\d+", m.group(1))]
    return {
        "observed": fields.get("observed"),
        "artifacts": None
        if artifacts is None
        else (None if artifacts.lower() == "none" else artifacts),
        "cited_frames": sorted(set(cited)),
        "verdict": verdict if verdict in ("PASS", "FAIL") else None,
        "confidence": int(conf_match.group()) / 100 if conf_match else None,
    }


def poses_from_detections(per_frame: list[list[Detection]]) -> list[HeadPose | None]:
    """One head pose per frame from the single person in it, None when not measurable."""
    return [c.pose for c in cues_from_detections(per_frame)]


def geometric_signal(cues: list[FrameCue], frame_diff: float | None = None) -> dict[str, Any]:
    """What YOLO alone can say: five gates, the cues behind them, and a 0..1 grade.

    one_face   exactly one person measurable in most frames
    smooth     no jump between frames (a cut, splice or glitch)
    alive      natural micro-movement: rules out a held photo
    not_frozen consecutive frames differ: rules out a frozen feed
    no_device  the face is not inside a phone, monitor or laptop: rules out a screen replay
    """
    poses = [c.pose for c in cues]
    yaw_range, pitch_range = motion_range(poses)
    moved = yaw_range >= MIN_YAW_RANGE or pitch_range >= MIN_PITCH_RANGE
    for c in cues:
        c.turn_counts = moved  # a turned head that never moved is a photo of one, not a turn
    summary = summarise(cues, [])
    if frame_diff is not None:
        summary.frame_diff = frame_diff
    n = len(poses)
    measurable = sum(p is not None for p in poses)
    jump = max_jump(poses)
    gates = {
        "one_face": n > 0 and measurable / n >= MIN_FACE_FRACTION,
        "smooth": measurable >= 2 and jump <= MAX_JUMP,
        "alive": summary.alive,
        "not_frozen": frame_diff is None or not summary.frozen,
        "no_device": summary.device is None,
    }
    return {
        "frames": n,
        "measurable_frames": measurable,
        "sequence": [p.facing if p else "?" for p in poses],
        "yaw_range": round(yaw_range, 3),
        "pitch_range": round(pitch_range, 3),
        "moved": moved,
        "max_jump": round(jump, 3),
        "cues": summary.micro
        | {
            "frame_diff": summary.frame_diff,
            "frozen": not gates["not_frozen"],
            "device": summary.device,
            "stress_frames": summary.stress_frames,
            "stress_kinds": summary.stress_kinds,
        },
        "stress_track": [c.stress for c in cues],
        "gates": gates,
        "geo": round(sum(gates.values()) / len(gates), 3),
    }


def combine(geo: dict[str, Any], vlm: dict[str, Any]) -> tuple[str, float]:
    """One window's verdict: PASS | FAIL | INCONCLUSIVE | ERROR, and a 0..1 score.

    FAIL on hard evidence: Cosmos names an artefact, the face is inside a device, the feed is
    frozen, or the score is under the threshold (a jump between frames halves it). PASS needs one
    face showing natural micro-movement; without it the window is INCONCLUSIVE and the agent
    keeps watching. A still head is fine: `assurance` says how strong a PASS is.
    """
    if vlm.get("verdict") is None:
        return "ERROR", 0.0
    conf = vlm.get("confidence") if vlm.get("confidence") is not None else 0.5
    cosmos = conf if vlm["verdict"] == "PASS" else 1 - conf
    gates = geo.get("gates") or {}
    score = round(cosmos * (1.0 if gates.get("smooth", True) else JUMP_PENALTY), 3)
    hard_fail = not gates.get("no_device", True) or not gates.get("not_frozen", True)
    if vlm["verdict"] == "FAIL" or score < THRESHOLD or hard_fail:
        return "FAIL", score
    if not (gates.get("one_face", True) and gates.get("alive", True)):
        return "INCONCLUSIVE", score
    return "PASS", score


def assurance(geo: dict[str, Any], verdict: str) -> str | None:
    """How strong a PASS is. strong: Cosmos judged at least one stress frame (head turned, hand
    or object at the face) and found it clean, which is where a face swap breaks. basic: clean
    and alive, but the caller never gave a hard frame, so a still face swap is not ruled out."""
    if verdict != "PASS":
        return None
    return "strong" if (geo.get("cues") or {}).get("stress_frames") else "basic"


def decide_call(verdicts: list[str]) -> str:
    """The call so far, from its window verdicts in order: PASS | FAIL | INCONCLUSIVE | WATCHING."""
    if "FAIL" in verdicts:
        return "FAIL"
    if verdicts.count("PASS") >= PASSES_NEEDED:
        return "PASS"
    if len(verdicts) >= MAX_WINDOWS:
        return "INCONCLUSIVE"
    return "WATCHING"


def _where(url: str) -> str:
    """Where a model ran, for the stack labels: the organisers' GPUs are CoreWeave."""
    if not url:
        return "local"
    organisers = [settings.cosmos3_reason_url, settings.cosmos_embed1_url, settings.yolo_url]
    if any(u and url.startswith(u.rstrip("/").removesuffix("/v1")) for u in organisers):
        return "CoreWeave GPU"
    if "runpod" in url:
        return "RunPod GPU"
    if "nvidia.com" in url:
        return "NVIDIA hosted"
    return urlparse(url).netloc or url


def _cue_line(geo: dict[str, Any], passed: int) -> str:
    c = geo.get("cues") or {}
    bits = [f"{passed}/{len(geo['gates'])} gates over {geo['frames']} frames"]
    bits.append("alive" if geo["gates"].get("alive") else "no natural movement")
    if c.get("frozen"):
        bits.append("frozen feed")
    if c.get("device"):
        bits.append(f"face inside a {c['device']}")
    if c.get("stress_frames"):
        bits.append(f"{c['stress_frames']} hard frames ({', '.join(c.get('stress_kinds') or [])})")
    return "; ".join(bits)


def _stage(step: str, component: str, t0: float, **detail: Any) -> dict[str, Any]:
    return {
        "step": step,
        "component": component,
        "latency_s": round(time.perf_counter() - t0, 2),
    } | {k: v for k, v in detail.items() if v is not None}


def head_crop(
    per_frame: list[list[Detection]], shape: tuple[int, ...], margin: float = 1.2
) -> list[int] | None:
    """One box around the head across the whole window, from the face keypoints of the most
    confident person in each frame, grown by `margin` of its size and clamped to the image.

    A call recording is usually a whole meeting window; the face is a small patch of it. Cosmos
    judges a 640 px face far better than a 640 px window with a face in it (17 Oct).
    """
    pts = []
    for dets in per_frame:
        people = [d for d in dets if d.keypoints is not None and d.kp_conf is not None]
        if not people:
            continue
        d = max(people, key=lambda d: d.area)  # the caller, not someone in the background
        ok = d.kp_conf[:5] >= 0.3
        if ok.any():
            pts.append(d.keypoints[:5][ok])
    if not pts:
        return None
    allp = np.concatenate(pts)
    x1, y1 = allp.min(axis=0)
    x2, y2 = allp.max(axis=0)
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    side = max(w, h) * (1 + 2 * margin)  # square-ish, hair to collar
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + h * 0.4  # the face sits below the eye line
    H, W = shape[:2]
    bx1, by1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    bx2, by2 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    if bx2 - bx1 < 32 or by2 - by1 < 32:
        return None
    return [bx1, by1, bx2, by2]


@op("yolo.pose_gates", drop=("detector", "objects"))
def observe(video_path: str, detector: Detector, objects: Detector | None = None) -> dict[str, Any]:
    """Two YOLO passes over a dense sample: pose on every frame (head, shoulders, wrists), object
    detection on every OBJECT_EVERY-th (phones, screens, cups). Gates, cues and per-frame
    stress flags come out; Cosmos is told which frames are the hard ones."""
    geo_frames = sample_frames(video_path, N_GEO_FRAMES)
    per_frame = [detector.detect(f.image) for f in geo_frames]
    cues = cues_from_detections(per_frame)
    if objects is not None:
        for i in range(0, len(geo_frames), OBJECT_EVERY):
            add_objects(cues[i], objects.detect(geo_frames[i].image))
    # Frozen check on 8 consecutive video frames (the pose sample is spread out in time).
    start = geo_frames[len(geo_frames) // 2].index if geo_frames else 0
    consecutive = read_frames(video_path, [start + k for k in range(8)])
    grey = [cv2.cvtColor(cv2.resize(f.image, (320, 180)), cv2.COLOR_BGR2GRAY) for f in consecutive]
    geo = geometric_signal(cues, frame_diff(grey))
    geo["frame_indices"] = [f.index for f in geo_frames]
    geo["crop"] = head_crop(per_frame, geo_frames[0].image.shape) if geo_frames else None
    geo["pose_track"] = [None if c.pose is None else [c.pose.yaw, c.pose.pitch] for c in cues]
    return {"geo": geo, "poses": [c.pose for c in cues]}


def pick_frames(geo: dict[str, Any], n: int) -> list[int]:
    """Which video frames Cosmos sees, in time order: stress frames first (a hand or object at
    the face, then the furthest turns left, right, up and down), where a face swap breaks;
    then evenly spread fill. Falls back to an even spread when YOLO measured nothing."""
    idx = geo.get("frame_indices") or []
    track = geo.get("pose_track") or []
    stress = geo.get("stress_track") or [[] for _ in idx]
    if not idx:
        return []
    if n >= len(idx):
        return list(idx)
    chosen: list[int] = []

    def take(k: int) -> None:
        if k not in chosen and len(chosen) < n and all(abs(k - c) > 1 for c in chosen):
            chosen.append(k)

    for k, why in enumerate(stress):  # occlusions first: the hardest thing for a swap
        if any(" at face" in w for w in why):
            take(k)
    valid = [k for k, t in enumerate(track) if t]
    if valid:
        for axis, pick in ((0, max), (0, min), (1, max), (1, min)):
            take(pick(valid, key=lambda k, a=axis: track[k][a]))
    for k in np.linspace(0, len(idx) - 1, n).round().astype(int):
        take(int(k))
    for k in range(len(idx)):  # still short (tiny clips): anything not yet taken
        if len(chosen) >= n:
            break
        if k not in chosen:
            chosen.append(k)
    return [idx[k] for k in sorted(chosen)]


@op("cosmos.judge", drop=("reasoner", "geo"))
def judge(
    video_path: str,
    geo: dict[str, Any],
    reasoner: VisionReasoner,
    n_frames: int = 12,
    prompt_version: str | None = None,
) -> dict[str, Any]:
    """Cosmos on a few frames (it is the expensive call), chosen by YOLO at the most-turned
    moments, and told what the detector measured at each."""
    version = prompt_version or settings.prompt_version
    template = COHERENCE_VERSIONS[version]
    if reasoner.frame_format == "images" and reasoner.max_images:
        n_frames = min(n_frames, reasoner.max_images)  # keep the prompt's frame numbers true
    picked = pick_frames(geo, n_frames)
    frames = read_frames(video_path, picked) if picked else sample_frames(video_path, n_frames)
    crop = geo.get("crop")
    if crop:
        x1, y1, x2, y2 = crop
        for f in frames:
            f.image = f.image[y1:y2, x1:x2]
    geo_idx = geo.get("frame_indices") or []
    stress = geo.get("stress_track") or []
    facing_at, notes = [], []
    for f in frames:
        if not geo_idx:
            facing_at.append("?")
            notes.append("?")
            continue
        j = min(range(len(geo_idx)), key=lambda k: abs(geo_idx[k] - f.index))
        facing_at.append(geo["sequence"][j])
        extra = [w for w in (stress[j] if j < len(stress) else []) if " at face" in w]
        notes.append(", ".join([f"head {geo['sequence'][j]}"] + extra))
    duration = frames[-1].t_seconds - frames[0].t_seconds if len(frames) > 1 else 0.0
    prompt = template.format(
        instruction=INSTRUCTION,
        criteria="The head moves as one rigid 3D object, with nothing sliding or flickering.",
        observed_sequence="; ".join(f"{i + 1}: {s}" for i, s in enumerate(notes)),
        n=len(frames),
        duration=duration,
    )
    result = reasoner.describe([f.image for f in frames], prompt, video_path=video_path)
    return {
        "frames": frames,
        "facing_at": facing_at,
        "frame_notes": notes,
        "result": result,
        "vlm": parse_verdict(result.text),
        "prompt_version": version,
    }


@op("liveness_window", drop=("detector", "reasoner", "objects"))
def run_session(
    video_path: str | Path,
    n_frames: int = 12,
    label: str | None = None,
    detector: Detector | None = None,
    reasoner: VisionReasoner | None = None,
    objects: Detector | None = None,
    evidence_dir: str | Path | None = None,
    prompt_version: str | None = None,
    archive: bool | None = None,
    persist: bool = True,
    call_id: str | None = None,
    window_no: int = 1,
) -> SessionRecord:
    """One window of a call: YOLO pose gates -> Cosmos -> verdict -> VSS archive -> verdict log.

    `evidence_dir` saves the frames Cosmos saw as JPEGs (the UI shows them). `archive` uploads the
    clip to the team's VSS (default VSS_ARCHIVE). `persist=False` skips the index (eval sweeps).
    """
    tracing.init()
    video_path = str(video_path)
    detector = detector or Detector()
    objects = objects or Detector(settings.yolo_detect_model)
    reasoner = reasoner or VisionReasoner()
    stack: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    seen = observe(video_path, detector, objects)
    geo, poses = seen["geo"], seen["poses"]
    passed = sum(geo["gates"].values())
    stack.append(
        _stage(
            "Head-pose gates",
            f"YOLO11 pose · {Path(settings.yolo_model).stem} · local",
            t0,
            detail=_cue_line(geo, passed),
        )
    )

    t0 = time.perf_counter()
    judged = judge(video_path, geo, reasoner, n_frames, prompt_version)
    frames, vlm, result = judged["frames"], judged["vlm"], judged["result"]
    stack.append(
        _stage(
            "Physical coherence",
            f"{result.model} · {_where(str(reasoner.client.base_url))}",
            t0,
            detail=f"{vlm.get('verdict') or 'no verdict'}"
            + (f", cites frames {vlm['cited_frames']}" if vlm.get("cited_frames") else ""),
        )
    )
    verdict, score = combine(geo, vlm)
    level = assurance(geo, verdict)

    evidence = [video_path]
    if evidence_dir is not None:
        folder = Path(evidence_dir)
        folder.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames, 1):
            path = folder / f"frame_{i:02d}.jpg"
            cv2.imwrite(str(path), f.image)
            evidence.append(str(path))

    record = SessionRecord(
        challenge_id=MODE,
        instruction=INSTRUCTION,
        verdict=verdict,
        score=score,
        reasoning=vlm.get("observed") or result.text[:500],
        evidence=evidence,
        signals={
            "call_id": call_id,
            "window_no": window_no,
            "assurance": level,
            "geometric": geo,
            "vlm": vlm,
            "vlm_raw": result.text,
            "vlm_reasoning": result.reasoning,
            "vlm_model": result.model,
            "vlm_latency_s": round(result.latency_s, 2),
            "prompt_version": judged["prompt_version"],
            "threshold": THRESHOLD,
            "frame_indices": [f.index for f in frames],
            "facing_at_frames": judged["facing_at"],
            "frame_notes": judged["frame_notes"],
            "poses": [
                None if p is None else {"yaw": p.yaw, "pitch": p.pitch, "facing": p.facing}
                for p in poses
            ],
        },
        label=label,
    )

    t0 = time.perf_counter()
    vec = embed(record.text())
    if vec:
        record.signals["embedding"] = vec
        stack.append(_stage("Window embedding", "Cosmos Embed1 · CoreWeave GPU", t0))

    if archive if archive is not None else settings.vss_archive:
        t0 = time.perf_counter()
        try:
            vss = vss_archive.upload(video_path, verdict, label, record.session_id)
        except Exception as e:
            vss = {"error": f"{type(e).__name__}: {e}"[:200]}
        if vss:
            record.signals["vss"] = vss
            stack.append(
                _stage(
                    "Clip archive",
                    "VAST VSS · team archive",
                    t0,
                    detail=vss.get("error") or "tags " + ", ".join(vss.get("tags", [])),
                )
            )

    index = get_index() if persist else None
    if tracing.enabled():
        stack.append({"step": "Trace", "component": "W&B Weave", "latency_s": None})
    if index is not None:
        stack.append({"step": "Verdict log", "component": index.backend, "latency_s": None})
    record.signals["stack"] = stack
    if index is not None:
        index.add(record)
    return record


@op("liveness_call_decision")
def finish_call(windows: list[SessionRecord], decision: str, persist: bool = True) -> SessionRecord:
    """Close a call: the agent writes the decision note over all its windows, with similar past
    windows from the verdict log in context. The note and the decision are stored on the last
    window, which is the one the overlay links to. The decision itself is `decide_call`'s."""
    last = windows[-1]
    index = get_index() if persist else None
    levels = [w.signals.get("assurance") for w in windows]
    level = None
    if decision == "PASS":
        level = "strong" if "strong" in levels else "basic"
    call = {
        "decision": decision,
        "assurance": level,
        "windows": [
            {
                "session_id": w.session_id,
                "verdict": w.verdict,
                "score": w.score,
                "assurance": w.signals.get("assurance"),
            }
            for w in windows
        ],
    }
    if agent_llm.enabled():
        t0 = time.perf_counter()
        similar = index.similar(last, k=3) if index is not None else []
        similar = [r for r in similar if r.signals.get("call_id") != last.signals.get("call_id")]
        note = agent_llm.write_verdict(windows, decision, similar, THRESHOLD, level)
        if note:
            call["agent"] = note
            if note.get("text"):
                last.reasoning = note["text"]
            source = index.backend if index is not None else "-"
            last.signals.setdefault("stack", []).insert(
                -1 if index is not None else len(last.signals.get("stack", [])),
                _stage(
                    "Call decision",
                    f"Agent · {note.get('model') or settings.agent_model} · W&B Inference",
                    t0,
                    detail=note.get("error")
                    or f"{decision}{f' ({level})' if level else ''} over {len(windows)} windows; "
                    f"{len(similar)} similar past windows from {source}",
                ),
            )
    last.signals["call"] = call
    last.signals["window_verdict"] = last.verdict
    last.verdict = decision  # what the overlay and the history show for this call
    if index is not None:
        index.add(last)  # SQLite upserts; VastDB keeps the newest row per session
    return last
