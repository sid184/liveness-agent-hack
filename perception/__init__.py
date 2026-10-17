"""YOLO pose: person boxes plus COCO keypoints (nose, eyes, ears, ...).

`yolo11n-pose.pt` is the default. Head pose (yaw, pitch, facing) is derived in `pose.py`.
"""

from perception.detector import Detection, Detector
from perception.pose import HeadPose, directions_reached, head_pose, in_order, max_jump

__all__ = [
    "Detection",
    "Detector",
    "HeadPose",
    "directions_reached",
    "head_pose",
    "in_order",
    "max_jump",
]
