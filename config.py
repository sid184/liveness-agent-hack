"""Single source of settings, read from .env. Everything that changes on the day lives here."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    # Keys
    nvidia_api_key: str
    wandb_api_key: str
    hf_token: str

    # Vision reasoner (the Cosmos seam)
    vlm_base_url: str
    vlm_model: str
    vlm_api_key: str
    vlm_frame_format: str  # images | montage | video_frames

    # Agent LLM (W&B Inference)
    agent_base_url: str
    agent_model: str
    agent_api_key: str

    # W&B logging
    wandb_project: str
    wandb_entity: str

    # Perception
    yolo_model: str

    # Index
    index_backend: str  # sqlite | local | vast

    # Paths
    root: Path
    data_dir: Path


def load_settings() -> Settings:
    return Settings(
        nvidia_api_key=_env("NVIDIA_API_KEY"),
        wandb_api_key=_env("WANDB_API_KEY"),
        hf_token=_env("HF_TOKEN"),
        # Code defaults are the no-GPU fallback (Llama on NVIDIA hosted). .env sets the real
        # default, the Cosmos pod. See .env.example.
        vlm_base_url=_env("VLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        vlm_model=_env("VLM_MODEL", "meta/llama-3.2-11b-vision-instruct"),
        vlm_api_key=_env("VLM_API_KEY") or _env("NVIDIA_API_KEY"),
        vlm_frame_format=_env("VLM_FRAME_FORMAT", "montage"),
        agent_base_url=_env("AGENT_BASE_URL", "https://api.inference.wandb.ai/v1"),
        agent_model=_env("AGENT_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        agent_api_key=_env("AGENT_API_KEY") or _env("WANDB_API_KEY"),
        wandb_project=_env("WANDB_PROJECT", "liveness-agent"),
        wandb_entity=_env("WANDB_ENTITY"),
        yolo_model=_env("YOLO_MODEL", "yolo11n.pt"),
        index_backend=_env("INDEX_BACKEND", "sqlite"),
        root=ROOT,
        data_dir=ROOT / "data",
    )


settings = load_settings()
