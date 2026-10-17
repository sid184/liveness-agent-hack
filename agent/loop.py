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
from perception import Detection, Detector, HeadPose, head_pose, max_jump, motion_range
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
MIN_FACE_FRACTION = 0.9  # frames in which exactly one face must be measurable
# Natural head movement needed before a PASS counts. A still talking head shows Cosmos no 3D
# structure to judge. Measured 17 Oct: deliberate turns span yaw 1.0 to 1.2.
MIN_YAW_RANGE = 0.5
MIN_PITCH_RANGE = 0.4
N_GEO_FRAMES = 40  # YOLO is cheap: about 4 fps over a 10 s clip, enough to see jumps

# Call decision.
WINDOW_S = 8  # seconds per recorded window
PASSES_NEEDED = 2  # clean windows before the caller is cleared
MAX_WINDOWS = 6  # then stop watching: INCONCLUSIVE


def parse_verdict(text: str) -> dict[str, Any]:
    """Pull the structured fields out of a COHERENCE response. Missing fields are None.

    With the think/answer prompt (v3) only the <answer> block is parsed."""
    answer = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.S)
    if answer:
        text = answer.group(1)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    fields = {k.lower(): v.strip() for k, v in _FIELD.findall(text)}
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
    """One head pose per frame from the most confident person, None when not measurable."""
    out: list[HeadPose | None] = []
    for dets in per_frame:
        people = sorted((d for d in dets if d.keypoints is not None), key=lambda d: -d.conf)
        if len(people) != 1:
            out.append(None)  # nobody, or more than one person: not a clean measurement
            continue
        p = people[0]
        out.append(head_pose(p.keypoints, p.kp_conf))
    return out


def geometric_signal(poses: list[HeadPose | None]) -> dict[str, Any]:
    """What the keypoint detector alone can say. Three gates, each a boolean, plus a 0..1 grade."""
    n = len(poses)
    measurable = sum(p is not None for p in poses)
    jump = max_jump(poses)
    yaw_range, pitch_range = motion_range(poses)
    gates = {
        "one_face": n > 0 and measurable / n >= MIN_FACE_FRACTION,
        "smooth": measurable >= 2 and jump <= MAX_JUMP,
        "moved": yaw_range >= MIN_YAW_RANGE or pitch_range >= MIN_PITCH_RANGE,
    }
    return {
        "frames": n,
        "measurable_frames": measurable,
        "sequence": [p.facing if p else "?" for p in poses],
        "yaw_range": round(yaw_range, 3),
        "pitch_range": round(pitch_range, 3),
        "max_jump": round(jump, 3),
        "gates": gates,
        "geo": round(sum(gates.values()) / len(gates), 3),
    }


def combine(geo: dict[str, Any], vlm: dict[str, Any]) -> tuple[str, float]:
    """One window's verdict: PASS | FAIL | INCONCLUSIVE | ERROR, and a 0..1 score.

    Cosmos decides realism; the score is its confidence that the caller is real, halved when
    YOLO saw a jump. A Cosmos FAIL always stands: it saw an artefact. A PASS only counts when
    YOLO saw exactly one face that moved enough to judge; otherwise the window is INCONCLUSIVE
    and the agent keeps watching.
    """
    if vlm.get("verdict") is None:
        return "ERROR", 0.0
    conf = vlm.get("confidence") if vlm.get("confidence") is not None else 0.5
    cosmos = conf if vlm["verdict"] == "PASS" else 1 - conf
    gates = geo.get("gates") or {}
    score = round(cosmos * (1.0 if gates.get("smooth", True) else JUMP_PENALTY), 3)
    if vlm["verdict"] == "FAIL" or score < THRESHOLD:
        return "FAIL", score
    if not (gates.get("one_face", True) and gates.get("moved", True)):
        return "INCONCLUSIVE", score
    return "PASS", score


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


def _stage(step: str, component: str, t0: float, **detail: Any) -> dict[str, Any]:
    return {
        "step": step,
        "component": component,
        "latency_s": round(time.perf_counter() - t0, 2),
    } | {k: v for k, v in detail.items() if v is not None}


@op("yolo.pose_gates", drop=("detector",))
def observe(video_path: str, detector: Detector) -> dict[str, Any]:
    """YOLO pose on a dense sample: per-frame head pose and the three gates."""
    geo_frames = sample_frames(video_path, N_GEO_FRAMES)
    per_frame = [detector.detect(f.image) for f in geo_frames]
    poses = poses_from_detections(per_frame)
    geo = geometric_signal(poses)
    geo["frame_indices"] = [f.index for f in geo_frames]
    geo["pose_track"] = [None if p is None else [p.yaw, p.pitch] for p in poses]
    return {"geo": geo, "poses": poses}


def pick_frames(geo: dict[str, Any], n: int) -> list[int]:
    """Which video frames Cosmos sees: the most-turned moments first (furthest left, right, up and
    down, where a face swap breaks), then evenly spread fill, in time order. Falls back to an even
    spread when YOLO measured nothing."""
    idx = geo.get("frame_indices") or []
    track = geo.get("pose_track") or []
    if not idx:
        return []
    if n >= len(idx):
        return list(idx)
    valid = [k for k, t in enumerate(track) if t]
    chosen: list[int] = []
    if valid:
        for axis, pick in ((0, max), (0, min), (1, max), (1, min)):
            k = pick(valid, key=lambda k, a=axis: track[k][a])
            if k not in chosen and len(chosen) < n:
                chosen.append(k)
    for k in np.linspace(0, len(idx) - 1, n).round().astype(int):
        if len(chosen) >= n:
            break
        if all(abs(int(k) - c) > 1 for c in chosen):
            chosen.append(int(k))
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
    geo_idx = geo.get("frame_indices") or []
    facing_at = []
    for f in frames:
        if not geo_idx:
            facing_at.append("?")
            continue
        j = min(range(len(geo_idx)), key=lambda k: abs(geo_idx[k] - f.index))
        facing_at.append(geo["sequence"][j])
    duration = frames[-1].t_seconds - frames[0].t_seconds if len(frames) > 1 else 0.0
    prompt = template.format(
        instruction=INSTRUCTION,
        criteria="The head moves as one rigid 3D object, with nothing sliding or flickering.",
        observed_sequence=", ".join(f"{i + 1}:{s}" for i, s in enumerate(facing_at)),
        n=len(frames),
        duration=duration,
    )
    result = reasoner.describe([f.image for f in frames], prompt, video_path=video_path)
    return {
        "frames": frames,
        "facing_at": facing_at,
        "result": result,
        "vlm": parse_verdict(result.text),
        "prompt_version": version,
    }


@op("liveness_window", drop=("detector", "reasoner"))
def run_session(
    video_path: str | Path,
    n_frames: int = 12,
    label: str | None = None,
    detector: Detector | None = None,
    reasoner: VisionReasoner | None = None,
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
    reasoner = reasoner or VisionReasoner()
    stack: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    seen = observe(video_path, detector)
    geo, poses = seen["geo"], seen["poses"]
    passed = sum(geo["gates"].values())
    stack.append(
        _stage(
            "Head-pose gates",
            f"YOLO11 pose · {Path(settings.yolo_model).stem} · local",
            t0,
            detail=f"{passed}/{len(geo['gates'])} gates over {geo['frames']} frames, "
            f"head turned yaw {geo['yaw_range']:.2f} pitch {geo['pitch_range']:.2f}",
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
    call = {
        "decision": decision,
        "windows": [
            {"session_id": w.session_id, "verdict": w.verdict, "score": w.score} for w in windows
        ],
    }
    if agent_llm.enabled():
        t0 = time.perf_counter()
        similar = index.similar(last, k=3) if index is not None else []
        similar = [r for r in similar if r.signals.get("call_id") != last.signals.get("call_id")]
        note = agent_llm.write_verdict(windows, decision, similar, THRESHOLD)
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
                    or f"{decision} over {len(windows)} windows; "
                    f"{len(similar)} similar past windows from {source}",
                ),
            )
    last.signals["call"] = call
    last.signals["window_verdict"] = last.verdict
    last.verdict = decision  # what the overlay and the history show for this call
    if index is not None:
        index.add(last)  # SQLite upserts; VastDB keeps the newest row per session
    return last
