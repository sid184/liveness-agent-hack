from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from capture.frames import montage, to_jpeg_bytes
from config import settings


@dataclass
class VlmResult:
    text: str
    latency_s: float
    model: str
    usage: dict[str, Any] | None
    reasoning: str | None = None  # Cosmos/Qwen think-trace when the server splits it out


def _data_url(image: np.ndarray, max_edge: int) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(to_jpeg_bytes(image, max_edge)).decode()


class VisionReasoner:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        frame_format: str | None = None,
        timeout_s: float = 90.0,
    ):
        self.model = model or settings.vlm_model
        self.frame_format = frame_format or settings.vlm_frame_format
        # Hosted endpoints can queue for minutes; fail fast rather than hang the demo.
        self.client = OpenAI(
            base_url=base_url or settings.vlm_base_url,
            api_key=api_key or settings.vlm_api_key,
            timeout=timeout_s,
            max_retries=1,
        )

    def _content(
        self,
        images: list[np.ndarray],
        prompt: str,
        max_edge: int,
        video_path: str | Path | None = None,
    ) -> list[dict]:
        if self.frame_format == "video" and video_path:
            # Cosmos NIM / vLLM: the clip itself, base64 mp4. The server samples frames.
            b64 = base64.b64encode(Path(video_path).read_bytes()).decode()
            return [
                {"type": "text", "text": prompt},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + b64}},
            ]
        if self.frame_format == "montage" and len(images) > 1:
            images = [montage(images)]
            max_edge = max(max_edge, 1024)
        if self.frame_format == "video_frames":
            # Cosmos NIM / vLLM extension: one ordered list of frames, fewer tokens than N images.
            return [
                {"type": "text", "text": prompt},
                {
                    "type": "video_frames",
                    "video_frames": [_data_url(im, max_edge) for im in images],
                },
            ]
        return [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": _data_url(im, max_edge)}} for im in images
        ]

    def describe(
        self,
        images: list[np.ndarray],
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        max_edge: int = 768,
        video_path: str | Path | None = None,
    ) -> VlmResult:
        """Send frames (or, with frame_format=video, the clip itself) plus a prompt."""
        if not images and not (self.frame_format == "video" and video_path):
            raise ValueError("at least one image is required")
        t0 = time.perf_counter()
        content = self._content(images, prompt, max_edge, video_path)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        usage = resp.usage.model_dump() if resp.usage else None
        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        # A reasoning parser on the server (vLLM --reasoning-parser) can swallow the whole reply
        # into the reasoning field when the model never closes a think tag. Fall back to it.
        text = msg.content or reasoning or ""
        return VlmResult(
            text=text,
            latency_s=time.perf_counter() - t0,
            model=self.model,
            usage=usage,
            reasoning=reasoning,
        )
