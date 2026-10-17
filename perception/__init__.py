"""YOLO detection and tracking. Stock yolo11n weights are COCO-trained.

COCO has 'person', 'cell phone', 'cup', 'bottle', 'book' and friends, but no 'face' or 'hand'.
Face and hand localisation is a decision for Mon/Tue (see CLAUDE.md).
"""

from perception.detector import Detection, Detector

__all__ = ["Detection", "Detector"]
