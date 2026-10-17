"""Measure the YOLO liveness cues on real clips and on fakes made from them, to set thresholds.

For every clip: the real thing (cut into WINDOW_S windows), a "held photo" (one frame of it,
jittered as if held in a hand, with camera noise) and a "frozen frame" (one frame repeated), both
encoded with x264 like the live path. Prints micro-motion and frame difference per window, so
the thresholds in perception/cues.py can sit between live and fake.

    uv run python -m scripts.calibrate_cues                    # data/clips/**/*.mp4
    uv run python -m scripts.calibrate_cues --clips data/clips/vss
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

import scripts._paths  # noqa: F401
from agent.loop import WINDOW_S, observe
from capture.frames import split_clip
from config import settings
from perception import Detector
from perception.cues import FROZEN_DIFF, MIN_HEAD_BODY, MIN_POSE_JITTER


def _encode(frames: list[np.ndarray], out: Path, fps: float = 15.0) -> Path:
    h, w = frames[0].shape[:2]
    raw = out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-preset",
         "veryfast", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )  # fmt: skip
    raw.unlink()
    return out


def fakes_from(clip: Path, out_dir: Path, seconds: float = WINDOW_S) -> dict[str, Path]:
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, still = cap.read()
    cap.release()
    if not ok:
        return {}
    fps, count = 15.0, int(seconds * 15)
    rng = np.random.default_rng(0)
    h, w = still.shape[:2]
    held, pos = [], np.zeros(4)  # dx, dy, angle, scale: a slow random walk, like a hand
    for _ in range(count):
        pos = 0.92 * pos + rng.normal(0, [1.2, 1.2, 0.25, 0.002])
        m = cv2.getRotationMatrix2D((w / 2, h / 2), pos[2], 1 + pos[3])
        m[:, 2] += pos[:2]
        img = cv2.warpAffine(still, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        noise = rng.normal(0, 2.0, img.shape)
        held.append(np.clip(img + noise, 0, 255).astype(np.uint8))
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "held photo": _encode(held, out_dir / f"{clip.stem}_photo.mp4", fps),
        "frozen frame": _encode([still] * count, out_dir / f"{clip.stem}_frozen.mp4", fps),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=str(settings.data_dir / "clips"))
    args = ap.parse_args()
    clips = sorted(p for p in Path(args.clips).rglob("*.mp4") if "calib" not in p.parts)
    work = settings.data_dir / "calib"
    detector = Detector()
    print(
        f"thresholds: pose_jitter >= {MIN_POSE_JITTER}  head_body >= {MIN_HEAD_BODY}  "
        f"frame_diff >= {FROZEN_DIFF} (else frozen)\n"
    )
    print(f"{'kind':<13} {'clip':<34} {'pose_jitter':>11} {'head_body':>9} {'diff':>6} "
          f"{'stress':>6}  alive frozen")  # fmt: skip
    for clip in clips:
        windows = split_clip(clip, WINDOW_S, work / clip.stem)
        # Short slices where the head stayed centred stand in for a caller sitting still.
        slices = split_clip(clip, 3, work / f"{clip.stem}_3s")
        still = [w for w in slices if not observe(str(w), detector)["geo"]["moved"]]
        cases = (
            [("live", w) for w in windows]
            + [("live still", w) for w in still]
            + [(k, v) for k, v in fakes_from(clip, work / "fakes").items()]
        )
        for kind, path in cases:
            g = observe(str(path), detector)["geo"]
            c = g["cues"]
            print(
                f"{kind:<13} {path.name[:34]:<34} {c['pose_jitter']:>11.4f} {c['head_body']:>9.4f} "
                f"{c['frame_diff']:>6.2f} {c['stress_frames']:>6}  {g['gates']['alive']!s:<5} "
                f"{c['frozen']}"
            )


if __name__ == "__main__":
    main()
