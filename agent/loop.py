"""One liveness session, end to end, against a recorded clip.

Live webcam capture plugs in by recording a window to a file first (capture.record_window).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from capture.frames import sample_frames
from challenge import IssuedChallenge, issue
from index import SessionRecord, get_index
from perception import (
    Detection,
    Detector,
    HeadPose,
    directions_reached,
    head_pose,
    in_order,
    max_jump,
)
from reason import VisionReasoner
from reason.prompts import COHERENCE

_FIELD = re.compile(r"^(OBSERVED|ARTIFACTS|COHERENT|VERDICT|CONFIDENCE):\s*(.*)$", re.M)
_FRAME_NUM = re.compile(r"\bframes?\s*(\d+(?:\s*(?:,|and|-|to)\s*\d+)*)", re.I)

# Scoring constants. Tune on the labelled clips, not by feel.
GEO_FLOOR = 0.3  # how much of the Cosmos score survives when geometry sees nothing
GEO_WEIGHT = 0.7  # how much geometry can add back
THRESHOLD = 0.5
MAX_JUMP = 0.8  # (yaw, pitch) jump between geometry frames above this is a cut, replay or glitch
MIN_FACE_FRACTION = 0.9  # frames in which exactly one face must be measurable
N_GEO_FRAMES = 40  # YOLO is cheap: about 4 fps over a 10 s clip, enough to see jumps


def parse_verdict(text: str) -> dict[str, Any]:
    """Pull the structured fields out of a COHERENCE response. Missing fields are None."""
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


def geometric_signal(poses: list[HeadPose | None], ch: IssuedChallenge) -> dict[str, Any]:
    """What the keypoint detector alone can say. Four gates, each a boolean, plus a 0..1 grade."""
    n = len(poses)
    measurable = sum(p is not None for p in poses)
    first = directions_reached(poses)
    reached = [d for d in ch.order if first.get(d) is not None]
    jump = max_jump(poses)
    gates = {
        "one_face": n > 0 and measurable / n >= MIN_FACE_FRACTION,
        "all_directions": len(reached) == len(ch.order),
        "in_order": in_order(first, ch.order),
        "smooth": measurable >= 2 and jump <= MAX_JUMP,
    }
    return {
        "frames": n,
        "measurable_frames": measurable,
        "sequence": [p.facing if p else "?" for p in poses],
        "first_seen": first,
        "reached": reached,
        "max_jump": round(jump, 3),
        "gates": gates,
        "geo": round(sum(gates.values()) / len(gates), 3),
    }


def combine(geo: dict[str, Any], vlm: dict[str, Any]) -> tuple[str, float]:
    """score = cosmos * (floor + weight * geo). Cosmos decides realism; geometry scales it.

    Geometry seeing nothing drags a confident PASS below the line; it can never lift a FAIL.
    """
    if vlm.get("verdict") is None:
        return "ERROR", 0.0
    conf = vlm.get("confidence") if vlm.get("confidence") is not None else 0.5
    cosmos = conf if vlm["verdict"] == "PASS" else 1 - conf
    g = geo.get("geo")
    factor = 1.0 if g is None else GEO_FLOOR + GEO_WEIGHT * g
    score = round(cosmos * factor, 3)
    return ("PASS" if score >= THRESHOLD else "FAIL"), score


def run_session(
    video_path: str | Path,
    challenge: IssuedChallenge | None = None,
    n_frames: int = 12,
    label: str | None = None,
    detector: Detector | None = None,
    reasoner: VisionReasoner | None = None,
) -> SessionRecord:
    ch = challenge or issue()
    detector = detector or Detector()
    reasoner = reasoner or VisionReasoner()

    # Geometry on a dense sample (cheap), Cosmos on a sparse one (expensive, max 16 images).
    geo_frames = sample_frames(video_path, N_GEO_FRAMES)
    per_frame = [detector.detect(f.image) for f in geo_frames]
    poses = poses_from_detections(per_frame)
    geo = geometric_signal(poses, ch)

    frames = sample_frames(video_path, n_frames)
    # Tell Cosmos what the detector saw at each of its frames: nearest geometry frame by index.
    geo_idx = [f.index for f in geo_frames]
    facing_at = []
    for f in frames:
        j = min(range(len(geo_idx)), key=lambda k: abs(geo_idx[k] - f.index))
        facing_at.append(geo["sequence"][j])
    duration = frames[-1].t_seconds - frames[0].t_seconds if len(frames) > 1 else 0.0
    prompt = COHERENCE.format(
        instruction=ch.instruction,
        observed_sequence=", ".join(f"{i + 1}:{s}" for i, s in enumerate(facing_at)),
        n=len(frames),
        duration=duration,
    )
    result = reasoner.describe([f.image for f in frames], prompt, video_path=video_path)
    vlm = parse_verdict(result.text)
    verdict, score = combine(geo, vlm)

    record = SessionRecord(
        challenge_id=ch.id,
        instruction=ch.instruction,
        verdict=verdict,
        score=score,
        reasoning=vlm.get("observed") or result.text[:500],
        evidence=[str(video_path)],
        signals={
            "order": ch.order,
            "geometric": geo,
            "vlm": vlm,
            "vlm_raw": result.text,
            "vlm_model": result.model,
            "vlm_latency_s": round(result.latency_s, 2),
            "frame_indices": [f.index for f in frames],
            "facing_at_frames": facing_at,
            "poses": [
                None if p is None else {"yaw": p.yaw, "pitch": p.pitch, "facing": p.facing}
                for p in poses
            ],
        },
        label=label,
    )
    get_index().add(record)
    return record
