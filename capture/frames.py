from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray  # BGR, as OpenCV gives it
    index: int
    t_seconds: float


def open_source(source: str | int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video source {source!r}")
    return cap


def iter_frames(source: str | int, max_frames: int | None = None) -> Iterator[Frame]:
    """Yield frames from a webcam index (0) or a video path until exhausted or max_frames."""
    cap = open_source(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    try:
        while max_frames is None or i < max_frames:
            ok, image = cap.read()
            if not ok:
                break
            yield Frame(image=image, index=i, t_seconds=i / fps)
            i += 1
    finally:
        cap.release()


def sample_frames(path: str | Path, n: int = 8) -> list[Frame]:
    """Evenly spaced frames from a video file. This is what the vision reasoner sees."""
    cap = open_source(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        cap.release()
        raise RuntimeError(f"no frames reported for {path}")
    picks = np.linspace(0, total - 1, num=min(n, total), dtype=int)
    frames: list[Frame] = []
    try:
        for idx in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, image = cap.read()
            if ok:
                frames.append(Frame(image=image, index=int(idx), t_seconds=int(idx) / fps))
    finally:
        cap.release()
    return frames


def read_frames(path: str | Path, indices: list[int]) -> list[Frame]:
    """Specific frames by index, in the order given."""
    cap = open_source(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[Frame] = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, image = cap.read()
            if ok:
                frames.append(Frame(image=image, index=int(idx), t_seconds=int(idx) / fps))
    finally:
        cap.release()
    return frames


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def to_jpeg_bytes(image: np.ndarray, max_edge: int = 768, quality: int = 85) -> bytes:
    """Downscale so the longest edge is max_edge, then JPEG-encode.

    Hosted VLM endpoints cap inline base64 payloads, so keep these small.
    """
    h, w = image.shape[:2]
    scale = max_edge / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def montage(images: list[np.ndarray], cols: int = 4, tile_w: int = 320) -> np.ndarray:
    """Tile frames left-to-right, top-to-bottom into one image, numbered.

    Fallback for single-image VLMs: temporal order is preserved by reading order.
    """
    if not images:
        raise ValueError("no images to tile")
    tiles = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        tile = cv2.resize(img, (tile_w, int(h * tile_w / w)), interpolation=cv2.INTER_AREA)
        cv2.putText(
            tile, f"{i + 1}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA
        )
        tiles.append(tile)
    tile_h = max(t.shape[0] for t in tiles)
    tiles = [
        cv2.copyMakeBorder(t, 0, tile_h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles
    ]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r : r + cols]) for r in range(0, len(tiles), cols)]
    return np.vstack(rows)


def _draw_scene(img: np.ndarray, object_x: int | None) -> None:
    """A cartoon person, plus a dark 'phone' rectangle centred at object_x if given."""
    cv2.rectangle(img, (200, 330), (440, 480), (60, 60, 200), -1)  # body
    cv2.circle(img, (320, 220), 90, (190, 160, 130), -1)  # face
    cv2.circle(img, (290, 200), 10, (40, 40, 40), -1)  # eyes
    cv2.circle(img, (350, 200), 10, (40, 40, 40), -1)
    cv2.ellipse(img, (320, 255), (30, 12), 0, 0, 180, (40, 40, 40), 3)  # mouth
    if object_x is not None:
        cv2.rectangle(img, (object_x - 30, 150), (object_x + 30, 290), (30, 30, 30), -1)
        cv2.rectangle(img, (object_x - 24, 160), (object_x + 24, 280), (80, 80, 80), -1)


def synthetic_test_image(save_to: str | Path | None = None) -> np.ndarray:
    """A drawn stand-in for 'object passing in front of a face', for smoke tests with no camera."""
    img = np.full((480, 640, 3), 235, dtype=np.uint8)
    _draw_scene(img, object_x=300)
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_to), img)
    return img


def synthetic_test_clip(save_to: str | Path, seconds: float = 4.0, fps: int = 15) -> Path:
    """A short mp4 of the cartoon 'phone' sweeping left to right across the face."""
    save_to = Path(save_to)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * fps)
    writer = cv2.VideoWriter(str(save_to), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    if not writer.isOpened():
        raise RuntimeError("could not open VideoWriter (mp4v)")
    try:
        for i in range(n):
            img = np.full((480, 640, 3), 235, dtype=np.uint8)
            _draw_scene(img, object_x=int(80 + (560 - 80) * i / max(n - 1, 1)))
            writer.write(img)
    finally:
        writer.release()
    return save_to


def split_clip(path: str | Path, seconds: float, out_dir: str | Path) -> list[Path]:
    """Cut a recording into back-to-back windows of `seconds` (re-encoded so every window
    starts on a keyframe). Used to replay a recorded call as the windows a live call would send."""
    import subprocess

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("window_*.mp4"):
        old.unlink()
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-force_key_frames", f"expr:gte(t,n_forced*{seconds})",
        "-f", "segment", "-segment_time", str(seconds), "-reset_timestamps", "1",
        str(out / "window_%02d.mp4"),
    ]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True)
    windows = sorted(out.glob("window_*.mp4"))
    # A last sliver under a second has too few frames to judge.
    if len(windows) > 1:
        cap = open_source(str(windows[-1]))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        if n / fps < 1.0:
            windows[-1].unlink()
            windows = windows[:-1]
    return windows
