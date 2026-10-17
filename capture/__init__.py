"""Webcam or file -> frames. Everything downstream consumes numpy BGR images."""

from capture.frames import (
    Frame,
    iter_frames,
    montage,
    read_image,
    sample_frames,
    synthetic_test_clip,
    synthetic_test_image,
    to_jpeg_bytes,
)

__all__ = [
    "Frame",
    "iter_frames",
    "montage",
    "read_image",
    "sample_frames",
    "synthetic_test_clip",
    "synthetic_test_image",
    "to_jpeg_bytes",
]
