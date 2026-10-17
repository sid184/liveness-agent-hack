"""Send one image (or sampled frames from a clip) to the vision reasoner and print the reply.

uv run python -m scripts.smoke_vlm                       # synthetic drawn image
uv run python -m scripts.smoke_vlm --image path.jpg
uv run python -m scripts.smoke_vlm --video clip.mp4 --frames 6
"""

from __future__ import annotations

import argparse

import scripts._paths  # noqa: F401
from capture.frames import read_image, sample_frames, synthetic_test_image
from config import settings
from reason import VisionReasoner
from reason.prompts import DESCRIBE, DESCRIBE_SEQUENCE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--video")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--prompt")
    ap.add_argument("--model", help="override VLM_MODEL")
    ap.add_argument("--frame-format", choices=["images", "montage", "video_frames"])
    args = ap.parse_args()

    if args.video:
        frames = sample_frames(args.video, args.frames)
        images = [f.image for f in frames]
        duration = frames[-1].t_seconds - frames[0].t_seconds
        prompt = args.prompt or DESCRIBE_SEQUENCE.format(n=len(images), duration=duration)
    elif args.image:
        images = [read_image(args.image)]
        prompt = args.prompt or DESCRIBE
    else:
        path = settings.data_dir / "samples" / "synthetic.jpg"
        images = [synthetic_test_image(path)]
        prompt = args.prompt or DESCRIBE
        print(f"no input given; using synthetic image at {path}")

    vlm = VisionReasoner(model=args.model, frame_format=args.frame_format)
    print(f"model={vlm.model} base_url={settings.vlm_base_url}")
    print(f"frames={len(images)} format={vlm.frame_format}")
    res = vlm.describe(images, prompt)
    print(f"latency={res.latency_s:.1f}s usage={res.usage}")
    if res.reasoning:
        print("-" * 20, "reasoning", "-" * 20)
        print(res.reasoning.strip()[:1500])
    print("-" * 20, "answer", "-" * 20)
    print(res.text)


if __name__ == "__main__":
    main()
