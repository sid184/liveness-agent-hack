"""Fast tests with no network and no model weights."""

import numpy as np

from agent.loop import (
    MAX_WINDOWS,
    NO_RESPONSE_WINDOWS,
    combine,
    decide_call,
    geometric_signal,
    head_crop,
    parse_verdict,
)
from capture.frames import montage, synthetic_test_image, to_jpeg_bytes
from index.base import SessionRecord
from index.local import LocalIndex
from index.sqlite import SqliteIndex
from perception import HeadPose, head_pose
from perception.detector import Detection


def test_synthetic_image_and_jpeg():
    img = synthetic_test_image()
    assert img.shape == (480, 640, 3)
    assert len(to_jpeg_bytes(img, max_edge=320)) < 60_000


def test_montage_tiles_in_order():
    tiles = montage([synthetic_test_image()] * 5, cols=4, tile_w=160)
    assert tiles.shape[1] == 160 * 4
    assert tiles.shape[0] == 2 * 120


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
    prof = head_pose(*_kp((100, 132), None, (80, 100)))  # only the right eye visible
    assert prof is not None and prof.profile and prof.yaw == 1.0
    assert head_pose(*_kp((100, 132), None, None)) is None


def test_geometric_signal_and_combine():
    table = {
        "centre": (0.0, 0.0),
        "left": (0.7, 0.0),
        "right": (-0.7, 0.0),
        "up": (0.0, 0.6),
    }
    moving = [HeadPose(*table[s], True, True, 0.9) for s in
              ["centre", "left", "centre", "right", "centre", "up", "centre"]]  # fmt: skip
    geo = geometric_signal(moving)
    assert geo["gates"] == {"one_face": True, "smooth": True, "moved": True}
    assert geo["geo"] == 1.0 and geo["yaw_range"] == 1.4
    assert combine(geo, {"verdict": "PASS", "confidence": 0.9}) == ("PASS", 0.9)
    # Cosmos naming an artefact always fails the window, whatever YOLO saw.
    assert combine(geo, {"verdict": "FAIL", "confidence": 0.8}) == ("FAIL", 0.2)
    # A still head: Cosmos may say PASS, but there was nothing to judge.
    still = geometric_signal([HeadPose(0.0, 0.0, True, True, 0.9)] * 6)
    assert still["gates"]["moved"] is False
    assert combine(still, {"verdict": "PASS", "confidence": 1.0}) == ("INCONCLUSIVE", 1.0)
    # A jump between frames (cut, glitch) halves the score.
    jumpy = geometric_signal([HeadPose(*table[s], True, True, 0.9) for s in ["left", "right"]])
    assert jumpy["gates"]["smooth"] is False
    assert combine(jumpy, {"verdict": "PASS", "confidence": 0.9}) == ("FAIL", 0.45)
    # Nobody in frame.
    nothing = geometric_signal([None] * 6)
    assert nothing["geo"] == 0.0
    assert combine(nothing, {"verdict": None}) == ("ERROR", 0.0)


def test_still_window_is_a_strike_not_a_fail():
    still = {"gates": {"one_face": True, "smooth": True, "moved": False}}
    moved = {"gates": {"one_face": True, "smooth": True, "moved": True}}
    no_resp = {
        "verdict": "FAIL",
        "confidence": 1.0,
        "artifacts": "no response to instruction",
        "cited_frames": [],
    }
    assert combine(still, no_resp)[0] == "INCONCLUSIVE"
    # the same artefact plus a real one, or on a window that did move, is a FAIL
    warp = dict(
        no_resp,
        artifacts="no response to instruction; boundary warp frames 3-5",
        cited_frames=[3, 4, 5],
    )
    assert combine(still, warp)[0] == "FAIL"
    # YOLO saw the turn and Cosmos found nothing else wrong: the detector wins on movement
    assert combine(moved, no_resp) == ("PASS", 0.9)
    assert combine(moved, warp)[0] == "FAIL"


def test_head_crop_from_keypoints():
    xy, cf = _kp((100, 132), (120, 100), (80, 100))
    d = Detection("person", 0.9, (60, 80, 140, 300), keypoints=xy, kp_conf=cf)
    box = head_crop([[d], [d], []], (480, 640, 3))
    assert box is not None
    x1, y1, x2, y2 = box
    assert x1 < 80 and x2 > 120 and y1 < 100 and y2 > 132  # contains the face points with margin
    assert 0 <= x1 and 0 <= y1 and x2 <= 640 and y2 <= 480
    assert head_crop([[], []], (480, 640, 3)) is None


def test_decide_call():
    assert decide_call(["PASS"]) == "WATCHING"
    assert decide_call(["INCONCLUSIVE", "PASS", "PASS"]) == "PASS"
    assert decide_call(["PASS", "FAIL"]) == "FAIL"
    assert decide_call(["INCONCLUSIVE"] * (NO_RESPONSE_WINDOWS - 1)) == "WATCHING"
    assert decide_call(["INCONCLUSIVE"] * NO_RESPONSE_WINDOWS) == "FAIL"  # asked, never turned
    assert decide_call(["INCONCLUSIVE", "PASS", "INCONCLUSIVE", "INCONCLUSIVE"]) == "WATCHING"
    mixed = ["PASS", "INCONCLUSIVE", "ERROR", "INCONCLUSIVE"] + ["ERROR", "INCONCLUSIVE"] * 2
    assert len(mixed) == MAX_WINDOWS and decide_call(mixed) == "INCONCLUSIVE"
    assert decide_call(["ERROR", "PASS"]) == "WATCHING"


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


def test_rank_prefers_embeddings_then_keywords():
    from index.base import rank

    a = SessionRecord("head_circle", "turn", "PASS", 0.9, "smooth rigid turn")
    b = SessionRecord("head_circle", "turn", "FAIL", 0.1, "texture slides at profile")
    a.signals["embedding"] = [1.0, 0.0]
    b.signals["embedding"] = [0.0, 1.0]
    assert [r.session_id for r in rank("anything", [a, b], 1, query_vec=[0.1, 0.9])] == [
        b.session_id
    ]
    # No query vector: keyword overlap.
    assert rank("slides at profile", [a, b], 1)[0].session_id == b.session_id


def test_index_similar_excludes_self(tmp_path):
    idx = SqliteIndex(tmp_path / "s.db")
    a = SessionRecord("head_circle", "turn", "FAIL", 0.1, "texture slides at profile")
    b = SessionRecord("head_circle", "turn", "FAIL", 0.2, "texture slides at the jaw")
    a.signals["embedding"], b.signals["embedding"] = [1.0, 0.1], [0.9, 0.2]
    idx.add(a)
    idx.add(b)
    assert [r.session_id for r in idx.similar(a)] == [b.session_id]
    assert idx.get(b.session_id).reasoning == b.reasoning


def test_parse_verdict_think_answer():
    text = (
        "<think>frame 3 looks odd</think><answer>OBSERVED: turns left\n"
        "ARTIFACTS: texture slides in frame 3\nVERDICT: FAIL\nCONFIDENCE: 90</answer>"
    )
    v = parse_verdict(text)
    assert v["verdict"] == "FAIL" and v["cited_frames"] == [3] and v["confidence"] == 0.9


def test_eval_labels_metrics():
    from scripts.eval_clips import label_from_tags, label_of, metrics

    assert label_from_tags(["fraud", "team_10", "liveness"]) is None
    assert label_from_tags(["liveness", "live"]) == "live"
    assert label_from_tags(["liveness", "Attack"]) == "attack"

    assert label_of("live_01_udlr.mp4") == "live"
    assert label_of("attack_faceswap_02.mp4") == "attack"
    assert label_of("Screen_Replay.mov") == "attack"
    assert label_of("notes.mp4") is None
    rows = [
        {"label": "live", "score": 0.8, "verdict": "PASS", "correct": True,
         "cosmos_confidence": 0.8, "latency_s": 1.0},
        {"label": "attack", "score": 0.2, "verdict": "FAIL", "correct": True,
         "cosmos_confidence": 0.8, "latency_s": 1.0},
    ]  # fmt: skip
    m = metrics(rows)
    assert m["accuracy"] == 1.0 and m["separation"] == 0.6 and m["margin"] == 0.6


def test_vast_row_roundtrip_and_vss_tags():
    from eval.vss_archive import tags_for
    from index.vast import from_row, to_row

    r = SessionRecord(
        "head_circle", "turn", "FAIL", 0.12, "slides", evidence=["a.mp4"], label="attack",
        signals={"vlm": {"verdict": "FAIL"}, "geometric": {"geo": 0.75},
                 "vss": {"object_key": "k"}},
    )  # fmt: skip
    row = to_row(r)
    assert row["challenge_order"] == "" and row["geo"] == 0.75
    assert row["vss_object_key"] == "k" and row["clip"] == "a.mp4"
    assert from_row(row) == r
    assert tags_for("FAIL", "attack", "abc") == [
        "liveness-session",
        "verdict-fail",
        "session-abc",
        "label-attack",
    ]


def test_run_session_offline(tmp_path, monkeypatch):
    """Whole session with a fake detector and a fake Cosmos: stack labels, frames saved."""
    import agent.loop as loop
    from capture.frames import synthetic_test_clip
    from perception import Detection
    from reason.vlm import VlmResult

    monkeypatch.setattr(loop, "embed", lambda text: None)
    monkeypatch.setattr(loop.tracing, "init", lambda: False)
    monkeypatch.setattr(loop.agent_llm, "enabled", lambda: False)
    clip = synthetic_test_clip(tmp_path / "c.mp4")

    class FakeDetector:
        def detect(self, image):
            xy, cf = _kp((100, 132), (120, 100), (80, 100))
            return [Detection("person", 0.9, (0, 0, 10, 10), keypoints=xy, kp_conf=cf)]

    class FakeReasoner:
        frame_format, max_images = "images", 5

        class client:  # noqa: N801
            base_url = "http://localhost/v1"

        def describe(self, images, prompt, video_path=None):
            assert "5 frames" in prompt  # capped to max_images, prompt agrees
            text = "OBSERVED: still\nARTIFACTS: flat in frame 2\nVERDICT: FAIL\nCONFIDENCE: 80"
            return VlmResult(text, 0.1, "fake-cosmos", None)

    rec = loop.run_session(
        clip,
        detector=FakeDetector(),
        reasoner=FakeReasoner(),
        evidence_dir=tmp_path / "ev",
        archive=False,
        persist=False,
    )
    assert rec.verdict == "FAIL" and rec.signals["vlm"]["cited_frames"] == [2]
    steps = [s["step"] for s in rec.signals["stack"]]
    assert steps[:2] == ["Head-pose gates", "Physical coherence"]
    assert len([e for e in rec.evidence if e.endswith(".jpg")]) == 5
    done = loop.finish_call([rec], "FAIL", persist=False)
    assert done.verdict == "FAIL" and done.signals["call"]["decision"] == "FAIL"
    assert done.signals["window_verdict"] == "FAIL"


def test_pick_frames_takes_extremes_in_time_order():
    from agent.loop import pick_frames

    track = [[0.0, 0.0]] * 40
    track = [list(t) for t in track]
    track[7] = [0.9, 0.0]  # furthest left
    track[30] = [-0.8, 0.0]  # furthest right
    track[15] = [0.0, 0.6]  # furthest up
    track[22] = [0.0, -0.5]  # furthest down
    geo = {"frame_indices": [i * 3 for i in range(40)], "pose_track": track}
    picked = pick_frames(geo, 5)
    assert len(picked) == 5 and picked == sorted(picked)
    assert {21, 90, 45, 66} <= set(picked)
    assert pick_frames({"frame_indices": [0, 3], "pose_track": [None, None]}, 5) == [0, 3]
    assert len(pick_frames({"frame_indices": list(range(40)), "pose_track": [None] * 40}, 5)) == 5
