"""Run YOLO on one image and print detections; writes an annotated copy to data/out/.

uv run python -m scripts.smoke_yolo                  # synthetic drawn image
uv run python -m scripts.smoke_yolo --image path.jpg
"""

from __future__ import annotations

import argparse
import time

import cv2

import scripts._paths  # noqa: F401
from capture.frames import read_image, synthetic_test_image
from config import settings
from perception import Detector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    if args.image:
        image = read_image(args.image)
    else:
        image = synthetic_test_image(settings.data_dir / "samples" / "synthetic.jpg")
        print("no input given; using synthetic image")

    t0 = time.perf_counter()
    det = Detector()
    print(f"model={settings.yolo_model} loaded in {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    dets = det.detect(image, conf=args.conf)
    print(f"{len(dets)} detections in {time.perf_counter() - t0:.2f}s")
    for d in dets:
        print(f"  {d.label:<12} {d.conf:.2f}  box={d.box}")
    out = settings.data_dir / "out" / "smoke_yolo.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), Detector.annotate(image, dets))
    print(f"annotated -> {out}")


if __name__ == "__main__":
    main()
