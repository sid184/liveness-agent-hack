"""Head pose from YOLO pose keypoints. No angles in degrees, just robust proxies.

COCO keypoint order: 0 nose, 1 left eye, 2 right eye, 3 left ear, 4 right ear. "Left" is the
person's left, which appears on the right of the image for a camera-facing subject.

yaw  > 0: head turned to the person's left  (nose moves toward the left ear)
pitch > 0: head tilted up                    (nose rises above the eye line)
Both are dimensionless ratios normalised by the inter-eye distance, so they are independent of
how close the person sits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4

# Absolute thresholds, used only by HeadPose.facing (single frame, no baseline).
YAW_TURNED = 0.55  # |yaw| above this counts as facing left or right
PITCH_TILTED = 0.45  # |pitch| above this counts as facing up or down

# Baseline-relative thresholds, used by classify() over a whole clip. Tuned on live_01 (17 Oct):
# a clear 45-degree turn reads about 0.45 from baseline; full profile saturates at 1.0; looking up
# reads about +0.28 from baseline; a lazy "down" reads about -0.18.
TURN_DEV = 0.35
UP_DEV = 0.2
DOWN_DEV = -0.15


@dataclass(frozen=True)
class HeadPose:
    yaw: float
    pitch: float
    left_ear_visible: bool
    right_ear_visible: bool
    confidence: float  # min keypoint confidence over nose and visible eyes
    profile: bool = False  # only one eye visible: yaw is saturated to +-1, pitch is unknown (0)

    @property
    def facing(self) -> str:
        """Dominant direction: up | down | left | right | centre."""
        if abs(self.yaw) >= YAW_TURNED and abs(self.yaw) >= abs(self.pitch):
            return "left" if self.yaw > 0 else "right"
        if abs(self.pitch) >= PITCH_TILTED:
            return "up" if self.pitch > 0 else "down"
        return "centre"


def head_pose(xy: np.ndarray, conf: np.ndarray, min_conf: float = 0.3) -> HeadPose | None:
    """Pose from one person's keypoints: xy is (17, 2) pixels, conf is (17,).

    Returns None when the nose or both eyes are missing (face out of frame or facing away).
    With exactly one eye visible the head is in full profile: the yaw is saturated to +-1 by
    which eye we can see, and the pitch is unknown.
    """
    if conf[NOSE] < min_conf:
        return None
    l_ok, r_ok = conf[L_EYE] >= min_conf, conf[R_EYE] >= min_conf
    if not (l_ok or r_ok):
        return None
    if l_ok != r_ok:
        # Only the person's left eye visible: the camera sees their left side, so the head is
        # turned to their right (yaw negative). And the reverse.
        return HeadPose(
            yaw=-1.0 if l_ok else 1.0,
            pitch=0.0,
            left_ear_visible=bool(conf[L_EAR] >= min_conf),
            right_ear_visible=bool(conf[R_EAR] >= min_conf),
            confidence=float(min(conf[NOSE], conf[L_EYE] if l_ok else conf[R_EYE])),
            profile=True,
        )
    nose, le, re = xy[NOSE], xy[L_EYE], xy[R_EYE]
    eye_mid = (le + re) / 2
    eye_dist = float(np.linalg.norm(le - re))
    if eye_dist < 1e-6:
        return None
    # Horizontal: nose offset from the eye midpoint, in the direction of the person's left eye.
    # Image x grows to the right; the person's left eye is on the image right when facing us.
    left_dir = (le - re) / eye_dist
    yaw = float(np.dot(nose - eye_mid, left_dir)) / eye_dist
    # Vertical: nose below the eye line is normal (about -0.6 to -1.0 inter-eye distances);
    # tilting up brings the nose toward the eyes. Centre that so 0 is a level head.
    raw_pitch = float(nose[1] - eye_mid[1]) / eye_dist  # positive = nose below eyes (image y down)
    pitch = -(raw_pitch - 0.8)  # 0.8 is a typical level-head ratio; positive = up
    return HeadPose(
        yaw=round(yaw, 3),
        pitch=round(pitch, 3),
        left_ear_visible=bool(conf[L_EAR] >= min_conf),
        right_ear_visible=bool(conf[R_EAR] >= min_conf),
        confidence=float(min(conf[NOSE], conf[L_EYE], conf[R_EYE])),
    )


def classify(poses: list[HeadPose | None]) -> list[str]:
    """Per-frame direction label relative to the clip's own resting pose.

    The baseline is the median yaw and pitch over frontal frames, which removes the camera
    angle (a laptop camera below the face makes a level head read as slightly 'up'). Profile
    frames are left or right outright. Unmeasurable frames are '?'.
    """
    frontal = [p for p in poses if p is not None and not p.profile]
    if frontal:
        base_yaw = float(np.median([p.yaw for p in frontal]))
        base_pitch = float(np.median([p.pitch for p in frontal]))
    else:
        base_yaw = base_pitch = 0.0
    out = []
    for p in poses:
        if p is None:
            out.append("?")
        elif p.profile:
            out.append("left" if p.yaw > 0 else "right")
        else:
            dy, dp = p.yaw - base_yaw, p.pitch - base_pitch
            if abs(dy) >= TURN_DEV and abs(dy) >= abs(dp):
                out.append("left" if dy > 0 else "right")
            elif dp >= UP_DEV:
                out.append("up")
            elif dp <= DOWN_DEV:
                out.append("down")
            else:
                out.append("centre")
    return out


def directions_reached(poses: list[HeadPose | None]) -> dict[str, int | None]:
    """First frame index at which each direction was seen, or None."""
    first: dict[str, int | None] = {"up": None, "down": None, "left": None, "right": None}
    for i, f in enumerate(classify(poses)):
        if f in first and first[f] is None:
            first[f] = i
    return first


def in_order(first: dict[str, int | None], order: list[str]) -> bool:
    """True when every direction in `order` was reached, and in that sequence."""
    idx = [first.get(d) for d in order]
    if any(i is None for i in idx):
        return False
    return all(a < b for a, b in zip(idx, idx[1:], strict=False))


def max_jump(poses: list[HeadPose | None]) -> float:
    """Largest change in (yaw, pitch) between consecutive frontal frames.

    Replays and swap glitches jump. Profile frames are skipped: their yaw is saturated and their
    pitch unknown, so a real turn into profile would look like a cut.
    """
    pts = [(p.yaw, p.pitch) for p in poses if p is not None and not p.profile]
    if len(pts) < 2:
        return 0.0
    arr = np.array(pts)
    return float(np.max(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
