"""Where do the real clips separate from the fakes? Every signal we have, per clip, ranked.

For each labelled clip (filename `live_*` / `attack_*`, as `eval_clips --from-vss` saves them):
YOLO's cues (micro-movement, frame difference, hard moments, turn range, jumps) and Cosmos's
probability that the caller is real under each prompt version. For each signal: the live and
attack ranges, AUC (the chance a random live clip scores higher than a random attack; 0.5 = no
signal, 1.0 or 0.0 = perfect, below 0.5 = the signal points the other way) and the threshold
with the best balanced accuracy. Then the full pipeline's verdict per clip.

    uv run python -m scripts.separation                       # data/clips, prompts v4 and v7
    uv run python -m scripts.separation --versions v4,v6,v7 --synthetic   # + photo/frozen fakes
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

import scripts._paths  # noqa: F401
from agent.loop import assurance, combine, judge, observe, parse_verdict
from capture.frames import montage
from config import settings
from eval import tracing, wandb_log
from perception import Detector
from perception.dynamics import measure
from reason import VisionReasoner
from reason.prompts import COHERENCE_VIDEO
from scripts.eval_clips import VIDEO_EXT, label_of

YOLO_FEATURES = ["pose_jitter", "head_body", "frame_diff", "stress_frames", "yaw_range", "max_jump"]
DYNAMICS = [
    "jerk_ratio",
    "head_movement",
    "jerk_rms",
    "camera_shake",
    "exposure_flicker",
    "bg_noise",
    "shoulder_motion",
    "head_shoulder_corr",
]


def auc(live: list[float], attack: list[float]) -> float:
    """P(live > attack), ties count half."""
    if not live or not attack:
        return float("nan")
    wins = sum((a < b) + 0.5 * (a == b) for b in live for a in attack)
    return wins / (len(live) * len(attack))


def best_threshold(live: list[float], attack: list[float]) -> tuple[float, str, float]:
    """(threshold, direction, balanced accuracy): live is `direction` ('>=' or '<') the line."""
    best = (float("nan"), ">=", 0.0)
    cuts = sorted(set(live + attack))
    cuts = [(a + b) / 2 for a, b in zip(cuts, cuts[1:], strict=False)] or cuts
    for t in cuts:
        for d in (">=", "<"):
            tp = np.mean([(v >= t) if d == ">=" else (v < t) for v in live])
            tn = np.mean([(v < t) if d == ">=" else (v >= t) for v in attack])
            bal = (tp + tn) / 2
            if bal > best[2]:
                best = (round(float(t), 4), d, round(float(bal), 3))
    return best


def synthetic_fakes(clips: list[Path]) -> list[tuple[Path, str]]:
    from scripts.calibrate_cues import fakes_from

    out = []
    for c in clips:
        for kind, path in fakes_from(c, settings.data_dir / "calib" / "fakes").items():
            out.append((path, kind))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=str(settings.data_dir / "clips"))
    ap.add_argument("--versions", default="v4,v7")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--synthetic", action="store_true", help="add held-photo / frozen fakes")
    ap.add_argument("--video-judge", action="store_true", help="also ask Cosmos about the clip")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    versions = [v.strip() for v in args.versions.split(",") if v.strip()]

    clips = sorted(
        p
        for p in Path(args.clips).rglob("*")
        if p.suffix.lower() in VIDEO_EXT and label_of(p.name) and "calib" not in p.parts
    )
    # The archive holds several uploads of the same file; count each video once.
    seen_hash: set[str] = set()
    unique = []
    for c in clips:
        h = hashlib.md5(c.read_bytes()).hexdigest()
        if h not in seen_hash:
            seen_hash.add(h)
            unique.append(c)
    if len(unique) < len(clips):
        print(f"{len(clips) - len(unique)} duplicate uploads dropped")
    cases: list[tuple[Path, str, str]] = [(c, label_of(c.name), "uploaded") for c in unique]
    if args.synthetic:
        live = [c for c, lab, _ in cases if lab == "live"]
        cases += [(p, "attack", kind) for p, kind in synthetic_fakes(live)]
    n_live = sum(lab == "live" for _, lab, _ in cases)
    print(f"{len(cases)} clips: {n_live} live, {len(cases) - n_live} attack; prompts {versions}")

    tracing.init()
    detector, objects = Detector(), Detector(settings.yolo_detect_model)
    reasoner = VisionReasoner()
    video_judge = VisionReasoner(frame_format="video") if args.video_judge else None
    rows: list[dict[str, Any]] = []
    images = []
    for path, label, source in cases:
        t0 = time.perf_counter()
        geo = observe(str(path), detector, objects)["geo"]
        c = geo["cues"]
        row: dict[str, Any] = {
            "clip": path.name,
            "label": label,
            "source": source,
            "pose_jitter": c["pose_jitter"],
            "head_body": c["head_body"],
            "frame_diff": c["frame_diff"],
            "stress_frames": c["stress_frames"],
            "yaw_range": geo["yaw_range"],
            "max_jump": geo["max_jump"],
            "measurable": geo["measurable_frames"] / max(geo["frames"], 1),
            "alive": geo["gates"]["alive"],
            "device": c["device"] or "",
        }
        img = None
        for v in versions:
            try:
                j = judge(str(path), geo, reasoner, args.frames, v)
                vlm = j["vlm"]
                if img is None:
                    img = montage([f.image for f in j["frames"]], cols=5, tile_w=200)
            except Exception as e:
                vlm = {"verdict": None, "observed": f"{type(e).__name__}: {e}"}
            conf = vlm.get("confidence")
            p_real = None
            if vlm.get("verdict") and conf is not None:
                p_real = conf if vlm["verdict"] == "PASS" else 1 - conf
            row[f"p_real_{v}"] = p_real
            row[f"artifacts_{v}"] = vlm.get("artifacts") or ""
            if v == versions[-1]:
                verdict, score = combine(geo, vlm)
                row["pipeline"] = verdict
                row["assurance"] = assurance(geo, verdict) or ""
                row["observed"] = (vlm.get("observed") or "")[:300]
        dyn = measure(str(path), detector)
        for k in DYNAMICS:
            row[k] = dyn.get(k)
        if video_judge is not None:
            try:
                dur = dyn.get("dyn_frames", 0) / 15
                res = video_judge.describe([], COHERENCE_VIDEO.format(duration=max(dur, 1)),
                                           video_path=str(path), max_tokens=400)  # fmt: skip
                vv = parse_verdict(res.text)
                conf = vv.get("confidence")
                row["p_real_video"] = (
                    None
                    if not vv.get("verdict") or conf is None
                    else (conf if vv["verdict"] == "PASS" else 1 - conf)
                )
                row["artifacts_video"] = (vv.get("artifacts") or "")[:300]
            except Exception as e:
                row["p_real_video"] = None
                row["artifacts_video"] = f"{type(e).__name__}: {str(e)[:200]}"
        row["latency_s"] = round(time.perf_counter() - t0, 1)
        rows.append(row)
        images.append(img)
        print(
            f"  {label:<6} {source:<12} {path.name[:40]:<40} jit {row['pose_jitter']:.3f} "
            f"hb {row['head_body']:.3f} diff {row['frame_diff']:.2f} "
            f"hard {row['stress_frames']:>2} "
            + " ".join(f"{v}={row[f'p_real_{v}']}" for v in versions)
            + f" -> {row['pipeline']} {row['assurance']}"
        )
        print(
            "         "
            + " ".join(f"{k}={row[k]}" for k in DYNAMICS)
            + (f" video={row.get('p_real_video')}" if video_judge else "")
        )

    # Separation per signal.
    features = (
        YOLO_FEATURES
        + DYNAMICS
        + ["measurable"]
        + [f"p_real_{v}" for v in versions]
        + (["p_real_video"] if video_judge else [])
    )
    stats = []
    for f in features:
        ok = lambda v: v is not None and not (isinstance(v, float) and np.isnan(v))  # noqa: E731
        live = [r.get(f) for r in rows if r["label"] == "live" and ok(r.get(f))]
        att = [r.get(f) for r in rows if r["label"] == "attack" and ok(r.get(f))]
        if not live or not att:
            continue
        t, d, bal = best_threshold(live, att)
        stats.append(
            {
                "signal": f,
                "auc": round(auc(live, att), 3),
                "live_range": f"{min(live):.3f}-{max(live):.3f}",
                "attack_range": f"{min(att):.3f}-{max(att):.3f}",
                "best_rule": f"live if {d} {t}",
                "balanced_acc": bal,
            }
        )
    stats.sort(key=lambda s: -abs(s["auc"] - 0.5))
    print(f"\n{'signal':<14} {'AUC':>6} {'live range':>16} {'attack range':>16}  best rule")
    for s in stats:
        print(
            f"{s['signal']:<14} {s['auc']:>6.3f} {s['live_range']:>16} {s['attack_range']:>16}  "
            f"{s['best_rule']} (balanced acc {s['balanced_acc']})"
        )
    by = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r["pipeline"])
    print("\npipeline verdicts by label:")
    for lab, vs in by.items():
        print(f"  {lab:<6} " + ", ".join(f"{v} {vs.count(v)}" for v in sorted(set(vs))))

    if not args.no_wandb:
        import wandb

        wandb_log.start_run(
            name=f"separation-{time.strftime('%m%d-%H%M')}",
            job_type="separation",
            config={"versions": versions, "clips": len(rows), "live": n_live},
        )
        if wandb_log.active():
            cols = list(dict.fromkeys(k for r in rows for k in r)) + ["frames"]
            table = wandb.Table(columns=cols)
            for r, im in zip(rows, images, strict=True):
                rgb = None if im is None else np.ascontiguousarray(im[:, :, ::-1])
                table.add_data(
                    *[r.get(k) for k in cols[:-1]], wandb.Image(rgb) if rgb is not None else None
                )
            st = wandb.Table(columns=list(stats[0]), data=[list(s.values()) for s in stats])
            wandb_log.log({"clips": table, "separation": st})
            for s in stats:
                wandb_log._run.summary[f"auc/{s['signal']}"] = s["auc"]
            print(f"\nW&B: {wandb_log.url()}")
            wandb_log.finish()


if __name__ == "__main__":
    main()
