"""Fast tests with no network and no model weights."""

import random

from agent.loop import combine, geometric_signal, parse_verdict
from capture.frames import montage, synthetic_test_image, to_jpeg_bytes
from challenge import issue
from index.base import SessionRecord
from index.local import LocalIndex
from index.sqlite import SqliteIndex
from perception.detector import Detection


def test_synthetic_image_and_jpeg():
    img = synthetic_test_image()
    assert img.shape == (480, 640, 3)
    assert len(to_jpeg_bytes(img, max_edge=320)) < 60_000


def test_montage_tiles_in_order():
    tiles = montage([synthetic_test_image()] * 5, cols=4, tile_w=160)
    assert tiles.shape[1] == 160 * 4
    assert tiles.shape[0] == 2 * 120


def test_issue_fills_params():
    ch = issue("fingers", rng=random.Random(0))
    assert "{n}" not in ch.instruction
    assert ch.params["n"] in (2, 3, 4)
    assert ch.deadline > ch.issued_at


def test_parse_verdict():
    text = "OBSERVED: a phone crosses the face\nCOHERENT: yes fine\nVERDICT: PASS\nCONFIDENCE: 85"
    v = parse_verdict(text)
    assert v == {
        "observed": "a phone crosses the face",
        "coherent": True,
        "verdict": "PASS",
        "confidence": 0.85,
    }
    assert parse_verdict("garbage")["verdict"] is None


def test_geometric_signal_and_combine():
    ch = issue("object_pass")
    person = Detection("person", 0.9, (100, 100, 400, 480))
    phone_over = Detection("cell phone", 0.8, (200, 150, 260, 300))
    phone_off = Detection("cell phone", 0.8, (500, 150, 560, 300))
    geo = geometric_signal([[person, phone_off], [person, phone_over], [person]], ch)
    assert geo["object_frames"] == 2
    assert geo["object_over_person_frames"] == 1
    assert geo["geometric_pass"] is True
    assert combine(geo, {"verdict": "PASS", "confidence": 0.9}) == ("PASS", 0.9)
    assert combine({"geometric_pass": False}, {"verdict": "PASS", "confidence": 0.9}) == (
        "FAIL",
        0.45,
    )
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
