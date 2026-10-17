from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass, field

DIRECTIONS = ("up", "down", "left", "right")
# Every order of the four directions. The caller gets one at random so a recording of a previous
# session, or a pre-rendered fake, cannot be replayed in the right sequence.
ORDERS: tuple[tuple[str, ...], ...] = tuple(itertools.permutations(DIRECTIONS))


@dataclass(frozen=True)
class Challenge:
    id: str
    instruction: str  # shown to the caller; may contain {order_text}
    duration_s: float
    criteria: str  # natural-language pass criteria, given to the vision reasoner
    params: dict = field(default_factory=dict)  # name -> tuple of allowed values


CHALLENGES: list[Challenge] = [
    Challenge(
        id="head_circle",
        instruction="Slowly turn your head {order_text}, pausing briefly at each.",
        duration_s=10,
        criteria=(
            "The head rotates as one rigid object through these directions in order: "
            "{order_text}. Ears, nose and jawline appear and disappear the way a real 3D head "
            "would. The face texture stays attached to the skull at every angle, with no "
            "sliding, warping, flicker or blurring at the face boundary, especially at profile. "
            "Hair, neck and background move consistently with the head and with each other."
        ),
        params={"order": ORDERS},
    ),
]


def order_text(order: tuple[str, ...] | list[str]) -> str:
    return ", then ".join(order)


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
    def order(self) -> list[str]:
        return list(self.params.get("order", DIRECTIONS))

    @property
    def deadline(self) -> float:
        return self.issued_at + self.challenge.duration_s


def issue(
    challenge_id: str | None = None,
    rng: random.Random | None = None,
    params: dict | None = None,
) -> IssuedChallenge:
    """Pick a challenge (random unless named), resolve its parameters and stamp it.

    `params` pins parameters, e.g. `{"order": ("up", "down", "left", "right")}` when judging a
    recorded clip in which the person performed a known order.
    """
    rng = rng or random.Random()
    if challenge_id is None:
        ch = rng.choice(CHALLENGES)
    else:
        ch = next((c for c in CHALLENGES if c.id == challenge_id), None)
        if ch is None:
            raise KeyError(f"unknown challenge {challenge_id!r}")
    params = {k: rng.choice(v) for k, v in ch.params.items()} | dict(params or {})
    if "order" in params:
        params["order"] = tuple(params["order"])
        if sorted(params["order"]) != sorted(DIRECTIONS):
            raise ValueError(f"order must be a permutation of {DIRECTIONS}, got {params['order']}")
    fmt = dict(params)
    if "order" in params:
        fmt["order_text"] = order_text(params["order"])
    return IssuedChallenge(
        challenge=ch,
        instruction=ch.instruction.format(**fmt),
        criteria=ch.criteria.format(**fmt),
        params=params,
        issued_at=time.time(),
    )
