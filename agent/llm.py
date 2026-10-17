from __future__ import annotations

from openai import OpenAI

from config import settings


class AgentLLM:
    """Text model on W&B Inference, OpenAI-compatible."""

    def __init__(
        self, base_url: str | None = None, api_key: str | None = None, model: str | None = None
    ):
        headers = {}
        if settings.wandb_entity:
            headers["OpenAI-Project"] = f"{settings.wandb_entity}/{settings.wandb_project}"
        self.model = model or settings.agent_model
        self.client = OpenAI(
            base_url=base_url or settings.agent_base_url,
            api_key=api_key or settings.agent_api_key,
            default_headers=headers or None,
        )

    def chat(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return resp.choices[0].message.content or ""
