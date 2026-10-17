"""Vision reasoner: 'did this physical interaction happen coherently?'

This is the Cosmos seam. The client speaks OpenAI chat completions with image content, which
both NVIDIA's hosted VLMs and a Cosmos NIM / vLLM server accept. Swap VLM_BASE_URL and
VLM_MODEL in .env; nothing here changes.
"""

from reason.vlm import VisionReasoner, VlmResult

__all__ = ["VisionReasoner", "VlmResult"]
