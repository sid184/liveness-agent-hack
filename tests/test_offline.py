"""Fast tests with no network and no model weights."""

import random

import numpy as np

from agent.loop import combine, geometric_signal, parse_verdict
from capture.frames import montage, synthetic_test_image, to_jpeg_bytes
from challenge import issue
from index.base import SessionRecord
from index.local import LocalIndex
from index.sqlite import SqliteIndex
from perception import HeadPose, head_pose


def test_synthetic_image_and_jpeg():
    img = synthetic_test_image()
    assert img.shape == (480, 640, 3)
    assert len(to_jpeg_bytes(img, max_edge=320)) < 60_000


def test_montage_tiles_in_order():
    tiles = montage([synthetic_test_image()] * 5, cols=4, tile_w=160)
    assert tiles.shape[1] == 160 * 4
    assert tiles.shape[0] == 2 * 120


def test_issue_shuffles_order():
    a = issue("head_circle", rng=random.Random(0))
    b = issue("head_circle", rng=random.Random(1))
    assert sorted(a.order) == ["down", "left", "right", "up"]
    assert a.order != b.order
    assert "then" in a.instruction and "{" not in a.instruction
    assert a.deadline > a.issued_at


def test_parse_verdict():
    text = (
        "OBSERVED: head turns up then left\nARTIFACTS: boundary flicker in frames 4 and 5, "
        "ear missing frame 7\nVERDICT: FAIL\nCONFIDENCE: 85"
    )
    v = parse_verdict(text)
    assert v["verdict"] == "FAIL" and v["confidence"] == 0.85
    assert v["cited_frames"] == [4, 5, 7]
    assert (
        parse_verdict("OBSERVED: fine\nARTIFACTS: none\nVERDICT: PASS\nCONFIDENCE: 70")["artifacts"]
        is None
    )
    assert parse_verdict("garbage")["verdict"] is None


def _kp(nose, le, re_, l_ear=None, r_ear=None):
    """Synthetic keypoints: nose, left eye, right eye (image coords), optional ears."""
    xy = np.zeros((17, 2))
    cf = np.zeros(17)
    for j, pt in ((0, nose), (1, le), (2, re_), (3, l_ear), (4, r_ear)):
        if pt is not None:
            xy[j] = pt
            cf[j] = 0.9
    return xy, cf


def test_head_pose_directions():
    # Eyes 40 px apart; person's left eye is on the image right (x=120).
    level = head_pose(*_kp((100, 132), (120, 100), (80, 100)))
    assert level is not None and level.facing == "centre"
    left = head_pose(*_kp((124, 132), (120, 100), (80, 100)))  # nose toward left eye
    assert left.facing == "left" and left.yaw > 0
    right = head_pose(*_kp((76, 132), (120, 100), (80, 100)))
    assert right.facing == "right"
    up = head_pose(*_kp((100, 108), (120, 100), (80, 100)))  # nose near the eye line
    assert up.facing == "up"
    down = head_pose(*_kp((100, 160), (120, 100), (80, 100)))
    assert down.facing == "down"
    assert head_pose(*_kp((100, 132), None, (80, 100))) is None


def test_geometric_signal_and_combine():
    ch = issue("head_circle", rng=random.Random(0))
    seq = ["centre"]
    for d in ch.order:
        seq += [d, "centre"]
    # Build poses that face each direction in the asked order, smoothly.
    table = {
        "centre": (0.0, 0.0),
        "left": (0.7, 0.0),
        "right": (-0.7, 0.0),
        "up": (0.0, 0.6),
        "down": (0.0, -0.6),
    }
    poses = [HeadPose(*table[s], True, True, 0.9) for s in seq]
    geo = geometric_signal(poses, ch)
    assert geo["gates"] == {
        "one_face": True,
        "all_directions": True,
        "in_order": True,
        "smooth": True,
    }
    assert geo["geo"] == 1.0
    assert combine(geo, {"verdict": "PASS", "confidence": 1.0}) == ("PASS", 1.0)
    # Wrong order: same directions, reversed.
    rseq = ["centre"]
    for d in reversed(ch.order):
        rseq += [d, "centre"]
    rev = [HeadPose(*table[s], True, True, 0.9) for s in rseq]
    geo_rev = geometric_signal(rev, ch)
    assert geo_rev["gates"]["all_directions"] and not geo_rev["gates"]["in_order"]
    # Geometry seeing nothing drags a confident PASS to a FAIL; it never lifts a FAIL.
    nothing = geometric_signal([None] * 6, ch)
    assert nothing["geo"] == 0.0
    assert combine(nothing, {"verdict": "PASS", "confidence": 1.0}) == ("FAIL", 0.3)
    assert combine(geo, {"verdict": "FAIL", "confidence": 0.8}) == ("FAIL", 0.2)
    assert combine(geo, {"verdict": None}) == ("ERROR", 0.0)


def test_local_index_roundtrip(tmp_path):
    idx = LocalIndex(tmp_path / "s.jsonl")
    idx.add(SessionRecord("object_pass", "pass a phone", "PASS", 0.9, "phone crossed face"))
    idx.add(SessionRecord("head_turn", "turn head", "FAIL", 0.2, "no parallax, looks replayed"))
    assert len(idx.all()) == 2
    hits = idx.search("replayed video with no parallax")
    assert hits and hits[0].challenge_id == "head_turn"


def test_sqlite_index_roundtrip(tmp_path):
    idx = SqliteIndex(tmp_path / "s.db")
    a = SessionRecord(
        "object_pass", "pass a phone", "PASS", 0.9, "phone crossed face", label="live"
    )
    b = SessionRecord(
        "head_turn",
        "turn head",
        "FAIL",
        0.2,
        "no parallax, looks replayed",
        label="attack",
        signals={"geometric": {"frames": 6}},
    )
    idx.add(a)
    idx.add(b)
    assert [r.session_id for r in idx.all()] == [a.session_id, b.session_id]
    assert idx.all()[1].signals == {"geometric": {"frames": 6}}
    hits = idx.search("replayed video with no parallax")
    assert hits and hits[0].challenge_id == "head_turn"
    assert [r.session_id for r in idx.where(label="attack")] == [b.session_id]
    assert idx.where(challenge_id="object_pass", label="attack") == []
    idx.add(a)  # same session_id twice is an upsert, not a duplicate
    assert len(idx.all()) == 2
