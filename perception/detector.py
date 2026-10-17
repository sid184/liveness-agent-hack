from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import settings


@dataclass
class Detection:
    label: str
    conf: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    track_id: int | None = None
    keypoints: np.ndarray | None = None  # (17, 2) pixel xy, pose models only
    kp_conf: np.ndarray | None = None  # (17,)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def overlap_fraction(self, other: Detection) -> float:
        """Fraction of this box's area that lies inside `other`."""
        ax1, ay1, ax2, ay2 = self.box
        bx1, by1, bx2, by2 = other.box
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        return (iw * ih) / self.area if self.area else 0.0


class Detector:
    """Thin wrapper so the rest of the code never imports ultralytics directly."""

    def __init__(self, model_path: str | None = None):
        from ultralytics import YOLO  # slow import, keep it local

        self.model = YOLO(model_path or settings.yolo_model)
        self.names: dict[int, str] = self.model.names

    def detect(self, image: np.ndarray, conf: float = 0.25) -> list[Detection]:
        res = self.model.predict(image, conf=conf, verbose=False)[0]
        return self._to_detections(res)

    def track(self, image: np.ndarray, conf: float = 0.25) -> list[Detection]:
        """Like detect, but assigns persistent track ids across successive calls."""
        res = self.model.track(image, conf=conf, persist=True, verbose=False)[0]
        return self._to_detections(res)

    def _to_detections(self, res) -> list[Detection]:
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        kps = res.keypoints
        kp_xy = kps.xy.cpu().numpy() if kps is not None and kps.xy is not None else None
        kp_cf = kps.conf.cpu().numpy() if kps is not None and kps.conf is not None else None
        out = []
        for i, (xyxy, c, k, tid) in enumerate(
            zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), ids, strict=True)
        ):
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            out.append(
                Detection(
                    self.names[int(k)],
                    float(c),
                    (x1, y1, x2, y2),
                    tid,
                    keypoints=kp_xy[i] if kp_xy is not None else None,
                    kp_conf=kp_cf[i] if kp_cf is not None else None,
                )
            )
        return out

    @staticmethod
    def annotate(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
        out = image.copy()
        for d in detections:
            x1, y1, x2, y2 = d.box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
            if d.keypoints is not None and d.kp_conf is not None:
                for j in range(5):  # nose, eyes, ears
                    if d.kp_conf[j] >= 0.3:
                        cv2.circle(out, tuple(int(v) for v in d.keypoints[j]), 4, (0, 0, 255), -1)
            tag = f"{d.label} {d.conf:.2f}" + (f" #{d.track_id}" if d.track_id is not None else "")
            cv2.putText(
                out,
                tag,
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                1,
                cv2.LINE_AA,
            )
        return out
