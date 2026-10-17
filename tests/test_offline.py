"""Fast tests with no network and no model weights."""

import numpy as np

from agent.loop import (
    MAX_WINDOWS,
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
    bracketed = parse_verdict(
        "OBSERVED: <still>\nARTIFACTS: <none>\nVERDICT: <PASS>\nCONFIDENCE: 95"
    )
    assert bracketed["verdict"] == "PASS" and bracketed["artifacts"] is None


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


def _cue(yaw=0.0, pitch=0.0, face=(0.0, 0.0), body=(0.0, -0.6, 0.4), **kw):
    from perception.cues import FrameCue

    return FrameCue(
        pose=HeadPose(yaw, pitch, True, True, 0.9),
        nose=(100.0, 100.0),
        eye_dist=40.0,
        face_shape=face,
        head_body=body,
        **kw,
    )


def test_geometric_signal_and_combine():
    from agent.loop import assurance

    rng = np.random.default_rng(0)
    # A still live head: no turns, but its shape wobbles against its eyes and shoulders.
    still = [
        _cue(face=tuple(rng.normal(0, 0.08, 2)), body=tuple(rng.normal([0, -0.6, 0.4], 0.04)))
        for _ in range(20)
    ]
    geo = geometric_signal(still, frame_diff=2.0)
    assert geo["gates"] == {
        "one_face": True,
        "smooth": True,
        "alive": True,
        "not_frozen": True,
        "no_device": True,
    }
    assert geo["moved"] is False and geo["cues"]["stress_frames"] == 0
    # Clean and alive but no hard frame: a generated talking head looks exactly like this, so
    # the window does not clear the caller (basic assurance, keep watching).
    assert combine(geo, {"verdict": "PASS", "confidence": 0.9}) == ("INCONCLUSIVE", 0.9)
    assert assurance(geo, "INCONCLUSIVE") == "basic"
    # Cosmos naming an artefact always fails the window, whatever YOLO saw.
    assert combine(geo, {"verdict": "FAIL", "confidence": 0.8}) == ("FAIL", 0.2)
    # A photo: identical shape every frame -> not alive -> nothing to clear the caller on.
    photo = geometric_signal([_cue() for _ in range(20)], frame_diff=2.0)
    assert photo["gates"]["alive"] is False
    assert combine(photo, {"verdict": "PASS", "confidence": 1.0}) == ("INCONCLUSIVE", 1.0)
    # A frozen feed or a face inside a phone fails outright, even if Cosmos says PASS.
    frozen = geometric_signal(still, frame_diff=0.0)
    assert combine(frozen, {"verdict": "PASS", "confidence": 1.0})[0] == "FAIL"
    on_phone = geometric_signal(
        [_cue(face=c.face_shape, body=c.head_body, device_at_face="cell phone") for c in still],
        frame_diff=2.0,
    )
    assert on_phone["gates"]["no_device"] is False
    assert combine(on_phone, {"verdict": "PASS", "confidence": 1.0})[0] == "FAIL"
    # A hand at the face is a hard frame: a clean PASS there is strong assurance.
    hand = [_cue(face=c.face_shape, body=c.head_body) for c in still]
    hand[5].hand_at_face = True
    geo_hand = geometric_signal(hand, frame_diff=2.0)
    assert geo_hand["cues"]["stress_frames"] == 1
    assert assurance(geo_hand, "PASS") == "strong"
    # A photo of a turned head is not a turn: the head never moved.
    turned_photo = geometric_signal([_cue(yaw=0.8) for _ in range(20)], frame_diff=2.0)
    assert turned_photo["cues"]["stress_frames"] == 0
    # A jump between frames (cut, glitch) halves the score.
    jumpy = geometric_signal([_cue(0.7), _cue(-0.7)] * 3, frame_diff=2.0)
    assert jumpy["gates"]["smooth"] is False
    assert combine(jumpy, {"verdict": "PASS", "confidence": 0.9})[0] == "FAIL"
    assert combine(geometric_signal([]), {"verdict": None}) == ("ERROR", 0.0)


def test_cues_from_keypoints_and_objects():
    from perception import Detection
    from perception.cues import add_objects, frame_cue, frame_diff, micro_motion

    xy, cf = _kp((100, 132), (120, 100), (80, 100))
    xy[5], xy[6], cf[5], cf[6] = (140, 200), (60, 200), 0.9, 0.9  # shoulders
    xy[9], cf[9] = (110, 150), 0.9  # a wrist near the nose
    cue = frame_cue(Detection("person", 0.9, (0, 0, 200, 300), keypoints=xy, kp_conf=cf))
    assert cue.hand_at_face and cue.face_shape is not None and cue.head_body is not None
    # Moving the whole picture rigidly (a held photo) leaves the shape measures unchanged.
    theta = 0.3
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    moved = (xy @ rot.T) * 1.2 + 15
    cue2 = frame_cue(Detection("person", 0.9, (0, 0, 200, 300), keypoints=moved, kp_conf=cf))
    assert np.allclose(cue.face_shape, cue2.face_shape)
    assert np.allclose(cue.head_body, cue2.head_body)
    assert micro_motion([cue, cue2] * 3)["pose_jitter"] < 1e-6
    # A face inside a phone's box is a device; a cup beside it is a held object.
    add_objects(cue, [Detection("cell phone", 0.8, (60, 60, 160, 180))])
    assert cue.device_at_face == "cell phone"
    cue3 = frame_cue(Detection("person", 0.9, (0, 0, 200, 300), keypoints=xy, kp_conf=cf))
    add_objects(cue3, [Detection("cup", 0.7, (150, 120, 190, 170))])
    assert cue3.object_at_face == "cup" and "cup at face" in cue3.stress
    still = np.full((10, 10), 128, np.uint8)
    assert frame_diff([still] * 4) == 0.0
    assert frame_diff([still, still + 3, still]) == 3.0


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
    # Passive: nobody was asked to move, so still windows keep the call watching, never fail it.
    assert decide_call(["INCONCLUSIVE"] * 3) == "WATCHING"
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
        objects=FakeDetector(),
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
