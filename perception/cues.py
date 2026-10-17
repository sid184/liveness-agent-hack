"""What YOLO alone can say about whether a caller is live. Pure functions, no model calls.

Built from two YOLO passes over the same frames: pose (17 COCO keypoints) and detection (COCO
classes). Each cue answers one attack:

- micro-motion  -> a photo or a frozen frame: a live head shifts against the shoulders and its
                   pose ratios wobble even when the person sits still; a flat photo moves rigidly
- frozen        -> a frozen feed: consecutive frames are pixel-identical (a webcam never is)
- device        -> a replay on a phone, monitor or laptop: the face sits inside that device's box
- stress frames -> a live face swap: it breaks when the head turns or something crosses the face
                   (a hand, a cup), so those frames are the ones Cosmos should judge

Keypoints (COCO): 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 9/10 wrists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from perception.detector import Detection
from perception.pose import HeadPose, head_pose

NOSE, L_SH, R_SH, L_WR, R_WR = 0, 5, 6, 9, 10
MIN_CONF = 0.3

DEVICES = frozenset({"cell phone", "tv", "laptop"})  # a face inside one of these is a replay
HELD = frozenset({"cup", "bottle", "wine glass", "cell phone"})  # natural occluders near a face

# Thresholds, from scripts/calibrate_cues.py on 17 Oct (3 team clips, plus a held photo and a
# frozen frame made from each). Measured: live 0.37-0.49 / 0.24-0.41, live with the head held
# still 0.27 / 0.18 (one 3 s slice), held photo 0.019-0.065 / 0.015-0.032, frozen <= 0.007.
# Frame difference: live 1.1-9.3, frozen 0.00. Re-run on a truly still caller before trusting.
MIN_POSE_JITTER = 0.10  # spread of the nose in the eyes' own frame, summed over 2 axes
MIN_HEAD_BODY = 0.06  # spread of the nose against the shoulders, summed over 3 measures
# Re-measured 17 Oct on the 640 px recordings the UI makes: a real caller holding still reads
# 0.32-0.49, a frozen feed 0.00-0.007. 0.35 failed a still human; 0.05 sits between the two.
FROZEN_DIFF = 0.05  # median mean-abs difference of consecutive grey frames (0-255) below this
HAND_RADIUS = 2.0  # a wrist within this many eye distances of the nose = hand at the face
HELD_RADIUS = 2.5  # an object box edge within this many eye distances of the nose
DEVICE_MIN_FRAMES = 2  # sampled frames with the face inside a device before it counts


@dataclass
class FrameCue:
    pose: HeadPose | None = None
    nose: tuple[float, float] | None = None
    eye_dist: float | None = None
    # Shape measures that a rigid move of the whole picture (shift, turn, zoom) leaves unchanged,
    # so a photo held in a hand keeps them constant while a live head does not.
    face_shape: tuple[float, float] | None = None  # nose in the eyes' own frame, in eye distances
    head_body: tuple[float, float, float] | None = None  # nose vs shoulders, in shoulder widths
    hand_at_face: bool = False
    object_at_face: str | None = None  # filled from the detection pass, sampled frames only
    device_at_face: str | None = None
    objects_checked: bool = False

    @property
    def turned(self) -> bool:
        return self.pose is not None and self.pose.facing != "centre"

    turn_counts: bool = True  # set False when the head never moved: a photo of a turned head

    @property
    def stress(self) -> list[str]:
        """Why this frame is a hard one for a face swap; empty when it is not."""
        out = []
        if self.hand_at_face:
            out.append("hand at face")
        if self.object_at_face:
            out.append(f"{self.object_at_face} at face")
        if self.turned and self.turn_counts:
            out.append(f"head turned {self.pose.facing}")
        return out


def _pt(xy: np.ndarray, conf: np.ndarray, j: int) -> np.ndarray | None:
    return xy[j] if conf[j] >= MIN_CONF else None


def frame_cue(person: Detection | None) -> FrameCue:
    """Cues from one person's keypoints. An empty cue when nobody (or several people) is there."""
    if person is None or person.keypoints is None or person.kp_conf is None:
        return FrameCue()
    xy, cf = person.keypoints, person.kp_conf
    cue = FrameCue(pose=head_pose(xy, cf))
    nose = _pt(xy, cf, NOSE)
    le, re_ = _pt(xy, cf, 1), _pt(xy, cf, 2)
    if nose is None or le is None or re_ is None:
        return cue
    eye_dist = float(np.linalg.norm(le - re_))
    if eye_dist < 1e-6:
        return cue
    cue.nose, cue.eye_dist = (float(nose[0]), float(nose[1])), eye_dist
    axis = (le - re_) / eye_dist  # along the eyes
    perp = np.array([-axis[1], axis[0]])  # across them
    off = nose - (le + re_) / 2
    cue.face_shape = (float(off @ axis / eye_dist), float(off @ perp / eye_dist))
    ls, rs = _pt(xy, cf, L_SH), _pt(xy, cf, R_SH)
    if ls is not None and rs is not None:
        width = float(np.linalg.norm(ls - rs))
        if width > 1e-6:
            sh_axis = (ls - rs) / width
            sh_perp = np.array([-sh_axis[1], sh_axis[0]])
            rel = nose - (ls + rs) / 2
            cue.head_body = (
                float(rel @ sh_axis / width),
                float(rel @ sh_perp / width),
                eye_dist / width,
            )
    for j in (L_WR, R_WR):
        w = _pt(xy, cf, j)
        if w is not None and np.linalg.norm(w - nose) <= HAND_RADIUS * eye_dist:
            cue.hand_at_face = True
    return cue


def cues_from_detections(per_frame: list[list[Detection]]) -> list[FrameCue]:
    """One cue per frame from the single person in it; an empty cue when there are 0 or 2+."""
    out = []
    for dets in per_frame:
        people = [d for d in dets if d.keypoints is not None]
        # The caller is the largest person in frame. People in an office background are
        # smaller; requiring exactly one person made every open-plan call unmeasurable (17 Oct).
        out.append(frame_cue(max(people, key=lambda d: d.area) if people else None))
    return out


def _box_near(box: tuple[int, int, int, int], x: float, y: float, margin: float) -> bool:
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return (dx * dx + dy * dy) ** 0.5 <= margin


def add_objects(cue: FrameCue, objects: list[Detection]) -> None:
    """Mark a device around the face, or a held object near it, from one frame's detections."""
    cue.objects_checked = True
    if cue.nose is None or cue.eye_dist is None:
        return
    x, y = cue.nose
    for d in sorted(objects, key=lambda d: -d.conf):
        if d.label in DEVICES and _box_near(d.box, x, y, 0):  # the nose is inside the device
            cue.device_at_face = d.label
            return
    for d in sorted(objects, key=lambda d: -d.conf):
        if d.label in HELD and _box_near(d.box, x, y, HELD_RADIUS * cue.eye_dist):
            cue.object_at_face = d.label
            return


def micro_motion(cues: list[FrameCue]) -> dict[str, float]:
    """How much a head moves even when 'still', measured so that moving the whole picture does
    not count: a photo held in a hand shifts, turns and zooms as one, and keeps both numbers at
    detector noise; a live head shifts against its own eyes and against its shoulders."""
    face = np.array([c.face_shape for c in cues if c.face_shape is not None])
    body = np.array([c.head_body for c in cues if c.head_body is not None])
    jitter = float(face.std(axis=0).sum()) if len(face) >= 4 else 0.0
    head_body = float(body.std(axis=0).sum()) if len(body) >= 4 else 0.0
    return {"pose_jitter": round(jitter, 4), "head_body": round(head_body, 4)}


def frame_diff(grey_frames: list[np.ndarray]) -> float:
    """Median mean-absolute difference between consecutive frames (0-255). ~0 = frozen."""
    if len(grey_frames) < 2:
        return 0.0
    diffs = [
        float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))
        for a, b in zip(grey_frames, grey_frames[1:], strict=False)
    ]
    return round(float(np.median(diffs)), 3)


@dataclass
class CueSummary:
    micro: dict[str, float]
    frame_diff: float
    devices: list[str] = field(default_factory=list)  # device label per sampled frame that had one
    stress_frames: int = 0
    stress_kinds: list[str] = field(default_factory=list)

    @property
    def alive(self) -> bool:
        return (
            self.micro["pose_jitter"] >= MIN_POSE_JITTER or self.micro["head_body"] >= MIN_HEAD_BODY
        )

    @property
    def frozen(self) -> bool:
        return self.frame_diff < FROZEN_DIFF

    @property
    def device(self) -> str | None:
        return self.devices[0] if len(self.devices) >= DEVICE_MIN_FRAMES else None


def summarise(cues: list[FrameCue], grey_frames: list[np.ndarray]) -> CueSummary:
    kinds: list[str] = []
    for c in cues:
        for k in c.stress:
            kind = k.split(" at ")[0] if " at " in k else "head turned"
            if kind not in kinds:
                kinds.append(kind)
    return CueSummary(
        micro=micro_motion(cues),
        frame_diff=frame_diff(grey_frames),
        devices=[c.device_at_face for c in cues if c.device_at_face],
        stress_frames=sum(bool(c.stress) for c in cues),
        stress_kinds=kinds,
    )
