"""A recorded clip through the whole pipeline, as a live call would send it.

The clip is cut into windows of WINDOW_S seconds; each window is judged in order (YOLO -> Cosmos
-> verdict -> VSS -> index) until the call is decided, then the agent writes the decision note.

uv run python -m scripts.smoke_pipeline                         # synthetic clip
uv run python -m scripts.smoke_pipeline --video data/clips/live_01.mp4 --label live --wandb
uv run python -m scripts.smoke_pipeline --video clip.mp4 --window 4   # shorter windows
"""

from __future__ import annotations

import argparse
import json
import uuid

import scripts._paths  # noqa: F401
from agent.loop import WINDOW_S, decide_call, finish_call, run_session
from capture.frames import split_clip, synthetic_test_clip
from config import settings
from eval import tracing, wandb_log
from perception import Detector
from reason import VisionReasoner
from reason.prompts import COHERENCE_VERSIONS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--window", type=float, default=WINDOW_S, help="seconds per window")
    ap.add_argument("--label", choices=["live", "attack"])
    ap.add_argument("--wandb", action="store_true", help="log this run to W&B")
    ap.add_argument("--prompt", choices=list(COHERENCE_VERSIONS), help="judge prompt version")
    ap.add_argument("--archive", action="store_true", help="upload each window to the team VSS")
    args = ap.parse_args()

    video = args.video
    if not video:
        video = synthetic_test_clip(settings.data_dir / "samples" / "synthetic_pass.mp4")
        print(f"no clip given; synthetic clip at {video}")

    tracing.init()
    call_id = uuid.uuid4().hex[:10]
    folder = settings.data_dir / "sessions" / f"smoke_{call_id}"
    parts = split_clip(video, args.window, folder / "windows")
    print(f"call {call_id}: {len(parts)} windows of {args.window:g} s from {video}")
    if args.wandb:
        wandb_log.start_run(name=f"smoke-{call_id}", config={"video": str(video)})

    detector, objects = Detector(), Detector(settings.yolo_detect_model)
    reasoner = VisionReasoner()
    windows, decision = [], "WATCHING"
    for i, part in enumerate(parts, 1):
        rec = run_session(
            part,
            detector=detector,
            objects=objects,
            reasoner=reasoner,
            n_frames=args.frames,
            label=args.label,
            prompt_version=args.prompt,
            archive=args.archive,
            evidence_dir=folder / f"window_{i:02d}",
            call_id=call_id,
            window_no=i,
        )
        windows.append(rec)
        geo = rec.signals["geometric"]
        vlm = rec.signals["vlm"]
        print(
            f"\nwindow {i}: {rec.verdict} {rec.score} ({rec.signals.get('assurance')})  "
            f"gates {json.dumps(geo['gates'])}"
        )
        print(f"  yolo: {rec.signals['stack'][0].get('detail')}")
        print(f"  cosmos saw: {'; '.join(rec.signals.get('frame_notes') or [])}")
        print(f"  cosmos ({rec.signals['vlm_model']}, {rec.signals['vlm_latency_s']}s): "
              f"{vlm.get('verdict')} {vlm.get('confidence')} | {vlm.get('observed')}")  # fmt: skip
        if vlm.get("artifacts"):
            print(f"  artifacts: {vlm['artifacts']}")
        decision = decide_call([w.verdict for w in windows])
        if decision != "WATCHING":
            break
    if decision == "WATCHING":
        decision = "INCONCLUSIVE"  # the recording ran out before the call was decided
        print(f"\nrecording ended after {len(windows)} windows without a decision")
    rec = finish_call(windows, decision)
    if args.wandb:
        for w in windows:
            wandb_log.log_session(w)

    level = (rec.signals.get("call") or {}).get("assurance")
    print(f"\nDECISION: {decision}{f' ({level})' if level else ''}\nnote: {rec.reasoning}")
    print("\nstack (last window):")
    for st in rec.signals.get("stack", []):
        lat = "" if st.get("latency_s") is None else f"{st['latency_s']:.1f}s"
        print(f"  {st['step']:<20} {st['component']:<50} {lat:>6}  {st.get('detail') or ''}")
    print(f"\nsession {rec.session_id} written to {settings.index_backend}")
    if args.wandb:
        wandb_log.finish()


if __name__ == "__main__":
    main()
