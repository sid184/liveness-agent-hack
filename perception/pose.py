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

# Thresholds on the ratios below. Tuned on the first real clips; keep them here, not inline.
YAW_TURNED = 0.55  # |yaw| above this counts as facing left or right
PITCH_TILTED = 0.45  # |pitch| above this counts as facing up or down


@dataclass(frozen=True)
class HeadPose:
    yaw: float
    pitch: float
    left_ear_visible: bool
    right_ear_visible: bool
    confidence: float  # min keypoint confidence over nose and eyes

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

    Returns None when the nose or both eyes are missing, which means we cannot measure anything
    (face out of frame, or a full profile where one eye is hidden and the proxy breaks down).
    """
    if conf[NOSE] < min_conf or conf[L_EYE] < min_conf or conf[R_EYE] < min_conf:
        return None
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


def motion_range(poses: list[HeadPose | None]) -> tuple[float, float]:
    """(yaw range, pitch range) over the measurable frames: how much the head actually turned."""
    pts = [(p.yaw, p.pitch) for p in poses if p is not None]
    if len(pts) < 2:
        return 0.0, 0.0
    arr = np.array(pts)
    span = arr.max(axis=0) - arr.min(axis=0)
    return float(span[0]), float(span[1])


def max_jump(poses: list[HeadPose | None]) -> float:
    """Largest frame-to-frame change in (yaw, pitch). Replays and swap glitches jump."""
    pts = [(p.yaw, p.pitch) for p in poses if p is not None]
    if len(pts) < 2:
        return 0.0
    arr = np.array(pts)
    return float(np.max(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
