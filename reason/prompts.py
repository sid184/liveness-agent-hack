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

# The first judge prompt (prep week). Kept as the baseline for the W&B eval: it parroted the pass
# criteria back and gave a cartoon confidence 100.
COHERENCE_V1 = """You are judging a liveness check on a video call.
The caller was asked: "{instruction}"
Pass criteria: {criteria}
You see {n} frames sampled in order over {duration:.0f} seconds.

Answer in exactly this format:
OBSERVED: <what physically happens across the frames, concretely>
COHERENT: <yes|no> <one sentence on occlusion, lighting, parallax and timing>
VERDICT: <PASS|FAIL>
CONFIDENCE: <integer 0-100>"""

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

# Passive judge (17 Oct pivot): nobody asked the caller to do anything, so it judges whatever
# natural head movement happened, and is told plainly that a still head proves nothing either way.
COHERENCE_PASSIVE = """You are the physical-plausibility judge for a video call. The caller has \
not been asked to do anything: this is a short window of natural conversation. Your job is to \
find reasons this could be a deepfake, a face swap, or a replayed recording. Agreeing is not the \
safe answer; missing a fake costs money.

A keypoint detector measured the head direction per frame as: {observed_sequence}
You see {n} frames sampled in order over {duration:.0f} seconds. Frames are numbered from 1.

Judge only what you can see. Look for, and name the frame numbers where you see:
- the face texture sliding, warping, blurring or flickering at the boundary, especially when the \
head turns
- ears, nose or jawline that do not appear or disappear the way a rotating 3D head would
- hair, neck, glasses or background that do not move with the head
- a cut, jump or loop between consecutive frames
- a screen, bezel, moire, reflections or a flat photo instead of a person
A head that barely moves is not evidence of a fake; say so in OBSERVED and judge the rest.

Answer in exactly this format, nothing else:
OBSERVED: <the head and face motion you actually see, frame by frame, in one or two sentences>
ARTIFACTS: <none | list each problem with its frame numbers>
VERDICT: <PASS|FAIL>
CONFIDENCE: <integer 0-100, your confidence in the VERDICT>"""

# Judge prompts by version. PROMPT_VERSION picks the live one; scripts/eval_clips.py runs them all
# on the labelled clips and logs the comparison to W&B. Add a version, never edit one in place,
# so earlier eval runs stay comparable. v1-v3 were written for the head-circle challenge.
COHERENCE_VERSIONS = {
    "v1": COHERENCE_V1,
    "v2": COHERENCE,
    "v3": COHERENCE + COSMOS_THINK_SUFFIX,
    "v4": COHERENCE_PASSIVE,
    "v5": COHERENCE_PASSIVE + COSMOS_THINK_SUFFIX,
}

# Agent (text LLM on W&B Inference). It never decides; the decision is computed.
VERDICT_WRITER = """You write the decision note for a liveness check on a video call, for a \
finance approver who is not technical. The caller was never asked to do anything: the agent \
watched the call in short windows. The decision is already made; do not change it.

Decision: {decision} (after {n} windows; a window needs score {threshold} or more)
Windows, in order:
{windows}
Similar past windows from the verdict log: {similar}

Write three sentences at most. Say what decided it (cite the window and frame numbers when the \
judge gave them), and, if a similar past window is relevant, name how. If the decision is \
INCONCLUSIVE, say what was missing. No preamble."""
