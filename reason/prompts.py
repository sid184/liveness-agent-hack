"""Prompt text lives here, not inline, so clip experiments can iterate on it."""

DESCRIBE = (
    "Describe what the person in this image is doing, and which way their head is facing. "
    "Be concrete."
)

DESCRIBE_SEQUENCE = (
    "These are {n} frames sampled in order from a {duration:.0f} second video. Describe how the "
    "person's head moves across the frames: which directions it turns, in what order, and "
    "whether the motion looks like a real 3D head. Be concrete about frames."
)

# The judge prompt. It is told what the detector measured so it can confirm or contradict it,
# and it is told what would make it fail, so agreeing is not the easy answer.
COHERENCE = """You are the physical-plausibility judge in a liveness check on a video call. \
Your job is to find reasons this could be a deepfake. Agreeing is not the safe answer; \
missing a fake costs money.

The caller was asked: "{instruction}"
A keypoint detector measured the head direction per frame as: {observed_sequence}
You see {n} frames sampled in order over {duration:.0f} seconds. Frames are numbered from 1.

Judge only what you can see. Look for, and name the frame numbers where you see:
- the face texture sliding, warping, blurring or flickering at the boundary, especially at profile
- ears, nose or jawline that do not appear or disappear the way a rotating 3D head would
- hair, neck, glasses or background that do not move with the head
- a cut, jump or loop between consecutive frames
- a screen, bezel, moire, reflections or a flat photo instead of a person

Answer in exactly this format, nothing else:
OBSERVED: <the head motion you actually see, frame by frame, in one or two sentences>
ARTIFACTS: <none | list each problem with its frame numbers>
VERDICT: <PASS|FAIL>
CONFIDENCE: <integer 0-100, your confidence in the VERDICT>"""

# Cosmos Reason models expose their reasoning when asked for think/answer tags. Append this to a
# prompt when VLM_MODEL is a Cosmos model; harmless on others.
COSMOS_THINK_SUFFIX = (
    "\n\nThink step by step inside <think></think> tags, then give the final answer inside "
    "<answer></answer> tags."
)
