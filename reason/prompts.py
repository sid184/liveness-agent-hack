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

# v7 (17 Oct, afternoon): YOLO now marks the hard frames (head turned, a hand or cup at the
# face) and they are the ones Cosmos sees first; a still head is normal, so the checklist covers
# what a frontal face gives away too, plus Sid's tells of generated video from v6.
COHERENCE_STRESS = """You are the physical-plausibility judge for a video call. The caller has not \
been asked to do anything: this is a short window of natural conversation. Your job is to find \
reasons this could be a deepfake, a face swap, a replayed recording or a photo. Agreeing is not \
the safe answer; missing a fake costs money.

You see {n} frames sampled over {duration:.0f} seconds, numbered from 1. A detector measured each:
{observed_sequence}
Frames marked with a hand, an object or a turned head are the hardest for a face swap: look \
hardest there.

Name the frame numbers where you see any of these:
- a hand, cup or object crossing the face that the face texture draws over, or that blurs, \
melts or shows the face through it
- face texture sliding, warping or flickering at its edge, at the hairline, the ears or the jaw
- a turned head whose ears, nose or jawline do not appear or hide like a real 3D head
- teeth or the inside of the mouth smeared or blurry while the rest is sharp
- eyes whose reflections or gaze do not match each other or the room
- light on the face that does not match the light on the neck, hair and room
- a phone, monitor or laptop edge, glare, moire, or a flat photo instead of a person
- a cut, jump or loop between consecutive frames
- signs of generated video: background and framing perfectly locked with no camera shake or \
sensor noise; skin or hair waxy or unnaturally smooth; eyes that never blink across the frames
A head that holds still is normal on a call and is not evidence of a fake; judge the rest.

Answer in exactly this format, nothing else:
OBSERVED: <what the head, face and hands do, frame by frame>
ARTIFACTS: <none | list each problem with its frame numbers>
VERDICT: <PASS|FAIL>
CONFIDENCE: <integer 0-100, your confidence in the VERDICT>"""

# v8 (17 Oct, experimental): Cosmos gets the whole clip (VLM_FRAME_FORMAT=video), not stills,
# and is asked about the physics of the motion, where generated video differs from a camera.
# Not a COHERENCE_VERSIONS entry: it takes no frame list. scripts/separation.py --video-judge.
COHERENCE_VIDEO = """You are watching a {duration:.0f} second clip of one person on a video call. \
It is either a real camera recording or a video made by a generator (an AI model). Judge the \
physics of the motion, which generated video usually gets subtly wrong:
- does the head have weight and inertia: small abrupt corrections, settling, uneven speed, or \
does it glide smoothly like an animation curve?
- does hair lag and settle after the head moves; do earrings, collars and necklaces move with it?
- do the shoulders and chest move on their own (breathing, shifting) or stay frozen?
- is there any camera shake, refocus, exposure change or sensor noise, or is the frame perfectly \
stable and clean?
- do skin texture, teeth, hairline and background details stay fixed, or shimmer and re-form?
- are blinks natural in timing and duration?

Answer in exactly this format, nothing else:
OBSERVED: <what the motion is like, citing seconds>
ARTIFACTS: <none | each sign of generation, with the second it appears>
VERDICT: <PASS if a real camera recording, FAIL if generated>
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
    "v7": COHERENCE_STRESS,
}

# Agent (text LLM on W&B Inference). It never decides; the decision is computed.
VERDICT_WRITER = """You write the decision note for a liveness check on a video call, for a \
finance approver who is not technical. The caller was never asked to do anything: the agent \
watched the call in short windows. The decision is already made; do not change it.

Decision: {decision}, assurance {assurance} (after {n} windows; a window needs score {threshold})
Assurance: strong = the judge saw the caller in at least one hard moment (head turned, a hand or \
cup at the face) and found it clean; basic = clean and alive but the caller stayed still, so a \
still face swap is not ruled out.
Windows, in order:
{windows}
Similar past windows from the verdict log: {similar}

Write three sentences at most. Say what decided it (cite the window and frame numbers when the \
judge gave them), and the assurance level and what would raise it. If a similar past window is \
relevant, name how. If the decision is INCONCLUSIVE, say what was missing. No preamble."""

# v6 (Sid, 17 Oct, after the first AI-generated test video passed v4 six times): the overlay
# then asked the caller to turn their head, so a head that never moved was evidence. Adds the
# tells of generated talking-head video, which v4 had nothing to say about. Retired with the
# head-turn request (Karim, 17 Oct: never ask the caller); kept so its eval runs stay comparable.
COHERENCE_VERSIONS["v6"] = """You are the physical-plausibility judge for a video call. The caller \
has been asked, on screen, to keep talking and to turn their head slowly to each side. Your job is \
to find reasons this could be a deepfake, a face swap, a generated talking-head video, or a \
replayed recording. Agreeing is not the safe answer; missing a fake costs money.

A keypoint detector measured the head direction per frame as: {observed_sequence}
You see {n} frames sampled in order over {duration:.0f} seconds. Frames are numbered from 1.

Judge only what you can see. Look for, and name the frame numbers where you see:
- the face texture sliding, warping, blurring or flickering at the boundary, especially as the \
head turns
- ears, nose or jawline that do not appear or disappear the way a rotating 3D head would
- hair, neck, glasses or background that do not move with the head
- a cut, jump or loop between consecutive frames
- a screen, bezel, moire, reflections or a flat photo instead of a person
- signs of generated video: background and framing perfectly locked with no camera shake or sensor \
noise; skin or hair waxy or unnaturally smooth; eyes that never blink across the frames; a mouth \
interior or teeth that blur or change shape; lighting on the face that does not match the neck \
and body
- no response to the instruction: the head stays within a few degrees for the whole window \
although \
the caller was asked to turn it. List this under ARTIFACTS as "no response to instruction".

Answer in exactly this format, nothing else:
OBSERVED: <the head and face motion you actually see, frame by frame, in one or two sentences>
ARTIFACTS: <none | list each problem with its frame numbers>
VERDICT: <PASS|FAIL>
CONFIDENCE: <integer 0-100, your confidence in the VERDICT>"""
