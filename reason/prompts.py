"""Prompt text lives here, not inline, so Sunday's clip experiments can iterate on it."""

DESCRIBE = (
    "Describe what the person in this image is doing, and whether any object is passing in "
    "front of their face. Be concrete."
)

DESCRIBE_SEQUENCE = (
    "These are {n} frames sampled in order from a {duration:.0f} second video. Describe what "
    "the person does across the frames, and whether any object passes in front of their face. "
    "Be concrete about timing, occlusion and motion."
)

COHERENCE = """You are judging a liveness check on a video call.
The caller was asked: "{instruction}"
Pass criteria: {criteria}
You see {n} frames sampled in order over {duration:.0f} seconds.

Answer in exactly this format:
OBSERVED: <what physically happens across the frames, concretely>
COHERENT: <yes|no> <one sentence on occlusion, lighting, parallax and timing>
VERDICT: <PASS|FAIL>
CONFIDENCE: <integer 0-100>"""

# Cosmos Reason models expose their reasoning when asked for think/answer tags. Append this to a
# prompt when VLM_MODEL is a Cosmos model; harmless on others.
COSMOS_THINK_SUFFIX = (
    "\n\nThink step by step inside <think></think> tags, then give the final answer inside "
    "<answer></answer> tags."
)
