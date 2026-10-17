"""One clip through the whole pipeline: challenge -> YOLO -> VLM -> verdict -> index.

uv run python -m scripts.smoke_pipeline                         # synthetic clip, object_pass
uv run python -m scripts.smoke_pipeline --video clip.mp4 --challenge head_turn --label live
"""

from __future__ import annotations

import argparse
import json

import scripts._paths  # noqa: F401
from agent.loop import run_session
from capture.frames import synthetic_test_clip
from challenge import issue
from config import settings
from eval import wandb_log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--challenge", default=None, help="object_pass | head_turn | fingers")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--label", choices=["live", "attack"])
    ap.add_argument("--wandb", action="store_true", help="log this run to W&B")
    args = ap.parse_args()

    video = args.video
    challenge_id = args.challenge
    if not video:
        video = synthetic_test_clip(settings.data_dir / "samples" / "synthetic_pass.mp4")
        challenge_id = challenge_id or "object_pass"
        print(f"no clip given; synthetic clip at {video}")

    ch = issue(challenge_id)
    print(f"challenge: [{ch.id}] {ch.instruction}")
    if args.wandb:
        wandb_log.start_run(
            name=f"smoke-{ch.id}", config={"video": str(video), "frames": args.frames}
        )

    rec = run_session(video, ch, n_frames=args.frames, label=args.label)

    print(f"verdict: {rec.verdict}  score={rec.score}")
    print(f"vlm ({rec.signals['vlm_model']}, {rec.signals['vlm_latency_s']}s):")
    print(rec.signals["vlm_raw"])
    print("geometric:", json.dumps(rec.signals["geometric"]))
    print(f"session {rec.session_id} written to index")
    if args.wandb:
        wandb_log.log_session(rec)
        wandb_log.finish()


if __name__ == "__main__":
    main()
