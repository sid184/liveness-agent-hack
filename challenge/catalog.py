from __future__ import annotations

import random
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Challenge:
    id: str
    instruction: str  # shown to the caller; may contain {n}
    duration_s: float
    criteria: str  # natural-language pass criteria, given to the vision reasoner
    objects: tuple[str, ...] = ()  # COCO class names YOLO should see crossing the face
    params: dict = field(default_factory=dict)


CHALLENGES: list[Challenge] = [
    Challenge(
        id="object_pass",
        instruction=(
            "Pick up your phone or a cup and pass it slowly in front of your face, left to right."
        ),
        duration_s=6,
        criteria=(
            "A handheld object crosses in front of the face and partially hides it for a few "
            "frames. "
            "The occlusion edge, lighting on the object and face, and the parallax between object "
            "and head must stay consistent frame to frame."
        ),
        objects=("cell phone", "cup", "bottle", "remote", "book"),
    ),
    Challenge(
        id="head_turn",
        instruction=(
            "Turn your head slowly to the left, then to the right, "
            "while moving the camera a little."
        ),
        duration_s=6,
        criteria=(
            "The head rotates left then right. Ears, nose and jawline change shape consistently "
            "with "
            "a real 3D head; the background shifts consistently with camera motion."
        ),
    ),
    Challenge(
        id="fingers",
        instruction="Hold up {n} fingers right next to your face, then lower your hand.",
        duration_s=5,
        criteria=(
            "A real hand appears next to the face showing exactly {n} extended fingers, with "
            "lighting "
            "and scale consistent with the face, then leaves the frame."
        ),
        params={"n": (2, 3, 4)},
    ),
]


@dataclass(frozen=True)
class IssuedChallenge:
    challenge: Challenge
    instruction: str  # params filled in
    criteria: str
    params: dict
    issued_at: float  # time.time()

    @property
    def id(self) -> str:
        return self.challenge.id

    @property
    def deadline(self) -> float:
        return self.issued_at + self.challenge.duration_s


def issue(challenge_id: str | None = None, rng: random.Random | None = None) -> IssuedChallenge:
    """Pick a challenge (random unless named), resolve its parameters and stamp it."""
    rng = rng or random.Random()
    if challenge_id is None:
        ch = rng.choice(CHALLENGES)
    else:
        ch = next((c for c in CHALLENGES if c.id == challenge_id), None)
        if ch is None:
            raise KeyError(f"unknown challenge {challenge_id!r}")
    params = {k: rng.choice(v) for k, v in ch.params.items()}
    return IssuedChallenge(
        challenge=ch,
        instruction=ch.instruction.format(**params),
        criteria=ch.criteria.format(**params),
        params=params,
        issued_at=time.time(),
    )
