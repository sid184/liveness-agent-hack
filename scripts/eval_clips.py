"""Every labelled clip under every judge prompt version, compared in W&B.

This is how the prompt gets fixed: v1 (the prep-week prompt) gave a cartoon confidence 100; each
later version has to separate `live` from `attack` clips better, and the W&B runs show it.

    uv run python -m scripts.eval_clips                         # data/clips, v1 vs v4 vs v5
    uv run python -m scripts.eval_clips --versions v4,v5 --frames 5
    uv run python -m scripts.eval_clips --from-vss              # first pull the team's VSS uploads
    uv run python -m scripts.eval_clips --index v4              # also write v4 verdicts to the log
    uv run python -m scripts.eval_clips --index v4 --archive    # ...and archive them to VSS, tagged

Labels come from the filename: `live` -> live; `attack`, `fake`, `swap`, `replay`, `screen`,
`photo` -> attack; anything else is skipped. Each clip is judged as one passive window (no
instruction to the caller). INCONCLUSIVE (too little head movement to clear) counts as neither
right nor wrong and is reported as its own rate.

One W&B run per prompt version (job_type eval, grouped per sweep), each with a table of every
clip (frames, gates, Cosmos's verdict and cited frames, score) and summary metrics.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import scripts._paths  # noqa: F401
from agent.loop import (
    MAX_WINDOWS,
    THRESHOLD,
    combine,
    decide_call,
    judge,
    observe,
    run_session,
)
from capture.frames import montage
from config import settings
from eval import tracing, vss_archive, wandb_log
from perception import Detector
from reason import VisionReasoner
from reason.prompts import COHERENCE_VERSIONS

VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
ATTACK_WORDS = ("attack", "fake", "swap", "replay", "screen", "photo", "deepfake")


def label_of(name: str) -> str | None:
    n = name.lower()
    if any(w in n for w in ATTACK_WORDS):
        return "attack"
    if "live" in n or "real" in n:
        return "live"
    return None


def label_from_tags(tags: list[str]) -> str | None:
    """Whole-tag match: `liveness` must not read as `live`."""
    t = {x.lower().strip() for x in tags or []}
    if t & {"attack", "fake", "deepfake", "swap", "faceswap", "replay", "screen", "photo"} or any(
        x.startswith(("attack", "label-attack")) for x in t
    ):
        return "attack"
    if t & {"live", "real", "genuine", "label-live"}:
        return "live"
    return None


def pull_from_vss(dest: Path, since: str) -> list[Path]:
    """Download the team's own uploads (on or after `since`, YYYY-MM-DD) that we don't have yet.

    VSS renames uploads to <timestamp>_<id>.mp4, so the label comes from the upload tags (`live` /
    `attack`) and goes into the local filename. Untagged clips are saved as unlabelled_*.mp4,
    which the eval skips: rename them by hand, or re-upload tagged.
    """
    if not vss_archive.available():
        print("VSS not configured (.env.team: INGRESS_URL, USERNAME, PASSWORD); skipping pull")
        return []
    got = []
    dest.mkdir(parents=True, exist_ok=True)
    for item in vss_archive.explore(scope="mine"):
        tags = item.get("tags") or []
        uploaded = str(item.get("upload_timestamp") or "")[:10]
        source = item.get("original_video")
        if not source or uploaded < since or vss_archive.TAG in tags:
            continue  # the pre-loaded corpus, or our own judged sessions
        stem = Path(item.get("filename") or source.rsplit("/", 1)[-1]).stem
        name = f"{label_from_tags(tags) or 'unlabelled'}_{stem}"
        if any(p.stem.endswith(stem) or stem in p.stem for p in dest.iterdir()):
            continue
        path = dest / f"{name}.mp4"
        try:
            vss_archive.download(source, path)
            got.append(path)
            print(f"pulled {path.name}  tags={tags}")
        except Exception as e:
            print(f"could not pull {stem}: {type(e).__name__}: {e}")
    return got


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    live = [r["score"] for r in rows if r["label"] == "live" and r["verdict"] != "ERROR"]
    attack = [r["score"] for r in rows if r["label"] == "attack" and r["verdict"] != "ERROR"]
    ok = [r for r in rows if r["verdict"] not in ("ERROR", "INCONCLUSIVE")]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    out = {
        "clips": len(rows),
        "errors": sum(r["verdict"] == "ERROR" for r in rows),
        "inconclusive_rate": mean([1.0 if r["verdict"] == "INCONCLUSIVE" else 0.0 for r in rows]),
        "accuracy": mean([1.0 if r["correct"] else 0.0 for r in ok]),
        "live_pass_rate": mean([1.0 if s >= THRESHOLD else 0.0 for s in live]),
        "attack_caught_rate": mean([1.0 if s < THRESHOLD else 0.0 for s in attack]),
        "call_live_pass_rate": mean(
            [1.0 if r.get("call") == "PASS" else 0.0 for r in rows if r["label"] == "live"]
        ),
        "call_attack_fail_rate": mean(
            [1.0 if r.get("call") == "FAIL" else 0.0 for r in rows if r["label"] == "attack"]
        ),
        "mean_score_live": mean(live),
        "mean_score_attack": mean(attack),
        "cosmos_mean_confidence": mean(
            [r["cosmos_confidence"] for r in ok if r["cosmos_confidence"] is not None]
        ),
        "mean_latency_s": mean([r["latency_s"] for r in rows]),
    }
    if live and attack:
        out["separation"] = round(out["mean_score_live"] - out["mean_score_attack"], 3)
        out["margin"] = round(min(live) - max(attack), 3)  # > 0: one threshold splits them all
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=str(settings.data_dir / "clips"))
    # v1 is the naive baseline; v2/v3 were written for the head-circle challenge.
    ap.add_argument("--versions", default="v1,v4,v6,v7")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--from-vss", action="store_true", help="pull the team's VSS uploads first")
    ap.add_argument(
        "--since", default=time.strftime("%Y-%m-%d"), help="with --from-vss: uploads from this date"
    )
    ap.add_argument("--index", metavar="VERSION", help="also run this version as full sessions")
    ap.add_argument(
        "--archive", action="store_true", help="with --index, upload each clip to VSS with tags"
    )
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    clips_dir = Path(args.clips)
    clips_dir.mkdir(parents=True, exist_ok=True)
    if args.from_vss:
        pull_from_vss(clips_dir / "vss", args.since)
    clips = sorted(
        p for p in clips_dir.rglob("*") if p.suffix.lower() in VIDEO_EXT and label_of(p.name)
    )
    if not clips:
        print(f"no labelled clips under {clips_dir} (name them live_*.mp4 / attack_*.mp4)")
        return
    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    unknown = [v for v in versions if v not in COHERENCE_VERSIONS]
    if unknown:
        raise SystemExit(f"unknown prompt versions {unknown}; have {list(COHERENCE_VERSIONS)}")
    print(f"{len(clips)} clips x {len(versions)} prompt versions, model {settings.vlm_model}")

    tracing.init()
    detector, reasoner = Detector(), VisionReasoner()

    # YOLO once per clip: the geometry does not depend on the prompt.
    seen = {clip: observe(str(clip), detector) for clip in clips}

    group = f"eval-{time.strftime('%m%d-%H%M')}"
    summary: dict[str, dict[str, Any]] = {}
    for version in versions:
        if not args.no_wandb:
            wandb_log.start_run(
                name=f"{group}-{version}",
                job_type="eval",
                group=group,
                config={
                    "prompt_version": version,
                    "prompt": COHERENCE_VERSIONS[version],
                    "vlm_model": settings.vlm_model,
                    "vlm_frame_format": settings.vlm_frame_format,
                    "frames": args.frames,
                    "threshold": THRESHOLD,
                    "clips": len(clips),
                },
            )
        rows: list[dict[str, Any]] = []
        images: list[Any] = []
        for clip in clips:
            s = seen[clip]
            t0 = time.perf_counter()
            try:
                j = judge(str(clip), s["geo"], reasoner, args.frames, version)
                vlm, raw = j["vlm"], j["result"].text
                images.append(montage([f.image for f in j["frames"]], cols=6, tile_w=200))
            except Exception as e:
                vlm, raw = {"verdict": None}, f"{type(e).__name__}: {e}"
                images.append(None)
            verdict, score = combine(s["geo"], vlm)
            label = label_of(clip.name)
            rows.append(
                {
                    "clip": clip.name,
                    "label": label,
                    "yaw_range": s["geo"]["yaw_range"],
                    "pitch_range": s["geo"]["pitch_range"],
                    "pose_jitter": s["geo"]["cues"]["pose_jitter"],
                    "head_body": s["geo"]["cues"]["head_body"],
                    "hard_frames": s["geo"]["cues"]["stress_frames"],
                    "geo": s["geo"]["geo"],
                    **{f"gate_{k}": v for k, v in s["geo"]["gates"].items()},
                    "cosmos_verdict": vlm.get("verdict"),
                    "cosmos_confidence": vlm.get("confidence"),
                    "cited_frames": ",".join(map(str, vlm.get("cited_frames") or [])),
                    "score": score,
                    "verdict": verdict,
                    # What a live call decides if every window looks like this one.
                    "call": decide_call([verdict] * MAX_WINDOWS),
                    "correct": verdict == ("PASS" if label == "live" else "FAIL"),
                    "latency_s": round(time.perf_counter() - t0, 2),
                    "observed": vlm.get("observed"),
                    "artifacts": vlm.get("artifacts"),
                    "raw": raw,
                }
            )
            r = rows[-1]
            print(
                f"  {version} {clip.name:<32} {label:<6} cosmos={r['cosmos_verdict']} "
                f"conf={r['cosmos_confidence']} geo={r['geo']} -> window {verdict} {score} "
                f"-> call {r['call']}"
            )
        m = metrics(rows)
        summary[version] = m
        if wandb_log.active():
            import wandb

            cols = list(rows[0]) + ["frames"]
            table = wandb.Table(columns=cols)
            for r, im in zip(rows, images, strict=True):
                rgb = None if im is None else np.ascontiguousarray(im[:, :, ::-1])
                img = None if rgb is None else wandb.Image(rgb, caption=r["clip"])
                table.add_data(*r.values(), img)
            wandb_log.log({"clips": table})
            for k, v in m.items():
                if v is not None:
                    wandb_log._run.summary[k] = v
            print(f"  W&B: {wandb_log.url()}")
            wandb_log.finish()

    print(f"\n{'metric':<24}" + "".join(f"{v:>10}" for v in summary))
    for k in next(iter(summary.values())):
        vals = [summary[v].get(k) for v in summary]
        print(f"{k:<24}" + "".join(f"{'-' if x is None else x!s:>10}" for x in vals))

    if args.index:
        print(f"\nwriting {args.index} sessions to the verdict log ({settings.index_backend})")
        for clip in clips:
            rec = run_session(
                clip,
                n_frames=args.frames,
                label=label_of(clip.name),
                detector=detector,
                reasoner=reasoner,
                prompt_version=args.index,
                archive=args.archive,
                evidence_dir=settings.data_dir / "sessions" / f"eval_{clip.stem}",
            )
            print(f"  {clip.name}: {rec.verdict} {rec.score} -> {rec.session_id}")


if __name__ == "__main__":
    main()
