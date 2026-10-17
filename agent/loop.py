"""One liveness session, end to end, against a recorded clip.

Live webcam capture plugs in later by swapping sample_frames() for a buffered iter_frames().
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from capture.frames import sample_frames
from challenge import IssuedChallenge, issue
from index import SessionRecord, get_index
from perception import Detection, Detector
from reason import VisionReasoner
from reason.prompts import COHERENCE

_FIELD = re.compile(r"^(OBSERVED|COHERENT|VERDICT|CONFIDENCE):\s*(.*)$", re.M)


def parse_verdict(text: str) -> dict[str, Any]:
    """Pull the structured fields out of a COHERENCE response. Missing fields are None."""
    fields = {k.lower(): v.strip() for k, v in _FIELD.findall(text)}
    verdict = (fields.get("verdict") or "").upper()
    conf_match = re.search(r"\d+", fields.get("confidence") or "")
    return {
        "observed": fields.get("observed"),
        "coherent": (fields.get("coherent") or "").lower().startswith("yes") or None
        if "coherent" in fields
        else None,
        "verdict": verdict if verdict in ("PASS", "FAIL") else None,
        "confidence": int(conf_match.group()) / 100 if conf_match else None,
    }


def geometric_signal(per_frame: list[list[Detection]], ch: IssuedChallenge) -> dict[str, Any]:
    """What YOLO alone can say about the challenge. Only object_pass has a geometric test so far."""
    n = len(per_frame)
    person_frames = sum(any(d.label == "person" for d in dets) for dets in per_frame)
    out: dict[str, Any] = {"frames": n, "person_frames": person_frames}
    if ch.challenge.objects:
        obj_frames = 0
        over_frames = 0
        for dets in per_frame:
            persons = [d for d in dets if d.label == "person"]
            objs = [d for d in dets if d.label in ch.challenge.objects]
            if objs:
                obj_frames += 1
            if objs and persons and any(o.overlap_fraction(p) > 0.5 for o in objs for p in persons):
                over_frames += 1
        out |= {"object_frames": obj_frames, "object_over_person_frames": over_frames}
        out["geometric_pass"] = over_frames >= 1
    return out


def combine(geo: dict[str, Any], vlm: dict[str, Any]) -> tuple[str, float]:
    """Naive first cut: the VLM decides, YOLO can only veto. Karim replaces this."""
    conf = vlm.get("confidence") if vlm.get("confidence") is not None else 0.5
    if vlm.get("verdict") is None:
        return "ERROR", 0.0
    score = conf if vlm["verdict"] == "PASS" else 1 - conf
    if geo.get("geometric_pass") is False:
        score *= 0.5
    return ("PASS" if score >= 0.5 else "FAIL"), round(score, 3)


def run_session(
    video_path: str | Path,
    challenge: IssuedChallenge | None = None,
    n_frames: int = 8,
    label: str | None = None,
    detector: Detector | None = None,
    reasoner: VisionReasoner | None = None,
) -> SessionRecord:
    ch = challenge or issue()
    frames = sample_frames(video_path, n_frames)
    detector = detector or Detector()
    reasoner = reasoner or VisionReasoner()

    per_frame = [detector.detect(f.image) for f in frames]
    geo = geometric_signal(per_frame, ch)

    duration = frames[-1].t_seconds - frames[0].t_seconds if len(frames) > 1 else 0.0
    prompt = COHERENCE.format(
        instruction=ch.instruction, criteria=ch.criteria, n=len(frames), duration=duration
    )
    result = reasoner.describe([f.image for f in frames], prompt)
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
            "geometric": geo,
            "vlm": vlm,
            "vlm_raw": result.text,
            "vlm_model": result.model,
            "vlm_latency_s": round(result.latency_s, 2),
            "detections": [
                [{"label": d.label, "conf": round(d.conf, 3), "box": d.box} for d in dets]
                for dets in per_frame
            ],
        },
        label=label,
    )
    get_index().add(record)
    return record
