"""How a clip moves, not how much: signals that tell a camera from a video generator.

Measured on a few seconds of consecutive frames resampled to a fixed rate, so 24 and 30 fps
sources compare fairly. Each signal targets something a webcam does and a generator tends not to:

- jerk_ratio      share of head-motion energy above 3 Hz, on an even time grid (see _uniform)
- camera_shake    tiny global shifts of the background: a laptop wobbles, a render is locked
- exposure_flicker background brightness changing over time: auto-exposure reacts, a render does not
- bg_noise        fresh grain on the background every frame: sensor noise vs a clean render
- shoulder_motion shoulders moving on their own (breathing, shifting), in shoulder widths
- head_shoulder_corr how tightly shoulder motion follows head motion (rigid avatar ~1, frozen ~0)

Experimental (17 Oct): ranked by scripts/separation.py on 3 real and 6 generated clips. Pixel
signals (bg_noise, camera_shake) may not survive a call's compression; motion ones should.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from perception.detector import Detection, Detector

RATE = 15.0  # Hz, frames per second after resampling
SECONDS = 6.0  # how much of the clip to measure
WIDTH = 640  # frames are resized to this width first
NOSE, L_EYE, R_EYE, L_SH, R_SH = 0, 1, 2, 5, 6


def read_resampled(path: str, rate: float = RATE, seconds: float = SECONDS) -> list[np.ndarray]:
    """Consecutive frames from the start of the clip, picked at `rate` Hz, resized to WIDTH."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out, i, next_t = [], 0, 0.0
    while len(out) < int(rate * seconds):
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        if t + 1e-6 < next_t:
            continue
        next_t += 1 / rate
        h, w = frame.shape[:2]
        out.append(cv2.resize(frame, (WIDTH, int(h * WIDTH / w)), interpolation=cv2.INTER_AREA))
    cap.release()
    return out


def _person(dets: list[Detection]) -> Detection | None:
    from perception.cues import main_person

    p = main_person(dets)
    return p if p is not None and p.kp_conf is not None else None


def _band_ratio(x: np.ndarray, rate: float, lo: float = 3.0, floor: float = 0.3) -> float:
    """Energy of the velocity above `lo` Hz over energy above `floor` Hz."""
    if len(x) < 8:
        return float("nan")
    v = np.diff(x)
    spec = np.abs(np.fft.rfft(v - v.mean())) ** 2
    f = np.fft.rfftfreq(len(v), 1 / rate)
    total = spec[f >= floor].sum()
    return float(spec[f >= lo].sum() / total) if total > 0 else float("nan")


def _background_mask(shape: tuple[int, int], box: tuple[int, int, int, int] | None) -> np.ndarray:
    h, w = shape
    mask = np.ones((h, w), np.float32)
    if box is not None:
        x1, y1, x2, y2 = box
        pad_x, pad_y = int(0.15 * (x2 - x1)), int(0.15 * (y2 - y1))
        mask[max(0, y1 - pad_y) : min(h, y2 + pad_y), max(0, x1 - pad_x) : min(w, x2 + pad_x)] = 0
    return mask


def head_track(path: str, detector: Detector, seconds: float = SECONDS) -> dict[str, np.ndarray]:
    """Nose and shoulder positions on every native frame, with real timestamps."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    t_, nose, sh, eye_d, sh_w, i = [], [], [], [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 or i / fps
        i += 1
        if t > seconds:
            break
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (WIDTH, int(h * WIDTH / w)), interpolation=cv2.INTER_AREA)
        p = _person(detector.detect(small))
        if p is None:
            continue
        xy, cf = p.keypoints, p.kp_conf
        if min(cf[NOSE], cf[L_EYE], cf[R_EYE]) < 0.3:
            continue
        t_.append(t)
        nose.append(xy[NOSE])
        eye_d.append(np.linalg.norm(xy[L_EYE] - xy[R_EYE]))
        has_sh = min(cf[L_SH], cf[R_SH]) >= 0.3
        sh.append((xy[L_SH] + xy[R_SH]) / 2 if has_sh else (np.nan, np.nan))
        sh_w.append(np.linalg.norm(xy[L_SH] - xy[R_SH]) if has_sh else np.nan)
    cap.release()
    return {
        "t": np.array(t_),
        "nose": np.array(nose).reshape(-1, 2),
        "shoulders": np.array(sh, dtype=float).reshape(-1, 2),
        "eye_dist": float(np.median(eye_d)) if eye_d else 1.0,
        "shoulder_width": float(np.nanmedian(sh_w)) if sh_w and not np.all(np.isnan(sh_w)) else 1.0,
    }


def _uniform(t: np.ndarray, x: np.ndarray, rate: float = RATE) -> np.ndarray:
    """Resample a track onto an even time grid by interpolation. Picking every k-th frame of a
    24 fps clip spaces frames unevenly and adds fake high-frequency jitter (it made 24 fps
    generated clips look 'jerky' on 17 Oct); interpolating on real timestamps does not."""
    g = np.arange(t[0], t[-1], 1 / rate)
    return np.column_stack([np.interp(g, t, x[:, k]) for k in range(x.shape[1])])


def measure(path: str, detector: Detector) -> dict[str, Any]:
    tr = head_track(path, detector)
    out: dict[str, Any] = {"dyn_frames": int(len(tr["t"]))}
    if len(tr["t"]) >= 10:
        n = _uniform(tr["t"], tr["nose"]) / tr["eye_dist"]
        out["jerk_ratio"] = round(
            float(np.nanmean([_band_ratio(n[:, 0], RATE), _band_ratio(n[:, 1], RATE)])), 4
        )
        out["jerk_rms"] = round(
            float(np.sqrt((np.diff(n, n=2, axis=0) ** 2).sum(axis=1).mean())), 4
        )
        out["head_movement"] = round(float(np.ptp(n, axis=0).max()), 3)
        ok = ~np.isnan(tr["shoulders"][:, 0])
        if ok.sum() >= 10:
            s = _uniform(tr["t"][ok], tr["shoulders"][ok]) / tr["shoulder_width"]
            n2 = _uniform(tr["t"][ok], tr["nose"][ok])
            out["shoulder_motion"] = round(
                float(np.linalg.norm(s - s.mean(axis=0), axis=1).std()), 4
            )
            hv = np.linalg.norm(np.diff(n2, axis=0), axis=1)
            sv = np.linalg.norm(np.diff(s, axis=0), axis=1)
            if hv.std() > 0 and sv.std() > 0:
                out["head_shoulder_corr"] = round(float(np.corrcoef(hv, sv)[0, 1]), 4)

    # Background signals on frames resampled by time. Unlike the head track these still depend
    # on the source frame rate a little (a shift between unevenly spaced frames); treat with care.
    frames = read_resampled(path)
    if len(frames) < 8:
        return out
    people = [_person(detector.detect(frames[0]))]
    # Background: everything outside the (padded) person box of the first frame with a person.
    box = next((p.box for p in people if p is not None), None)
    grey = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    mask = _background_mask(grey[0].shape, box)
    if mask.sum() > 0.05 * mask.size:
        win = cv2.createHanningWindow(grey[0].shape[::-1], cv2.CV_32F)
        shifts = []
        for a, b in zip(grey, grey[1:], strict=False):
            (dx, dy), _ = cv2.phaseCorrelate(a * mask, b * mask, win)
            shifts.append((dx, dy))
        sh = np.array(shifts)
        out["camera_shake"] = round(
            float(np.linalg.norm(sh - np.median(sh, axis=0), axis=1).mean()), 4
        )
        lum = np.array([(g * mask).sum() / mask.sum() for g in grey])
        out["exposure_flicker"] = round(float(np.std(np.diff(lum))), 4)
        resid = np.array([g - cv2.GaussianBlur(g, (0, 0), 1.5) for g in grey])
        tstd = resid.std(axis=0)
        out["bg_noise"] = round(float(np.median(tstd[mask > 0])), 4)
    return out
