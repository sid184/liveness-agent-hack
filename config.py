"""Single source of settings, read from .env. Everything that changes on the day lives here."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

ROOT = Path(__file__).resolve().parent
# .env.team is a copy of the organisers' /config/<team>.config (gitignored). Loaded first so
# .env can reference its values (e.g. VLM_API_KEY=${GPU_BEARER_TOKEN}). .env then overrides both
# it and the shell: the lab VM exports the organisers' W&B key, and our own W&B project (set in
# .env) must win over it. Leave a variable out of .env to take the team's value.
load_dotenv(ROOT / ".env.team")
load_dotenv(ROOT / ".env", override=True)
# USERNAME is also the login name in most shells, and load_dotenv never overrides the
# environment, so the team login is read from the file itself when it is there.
_TEAM_FILE = {k: (v or "").strip() for k, v in dotenv_values(ROOT / ".env.team").items()}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _flag(name: str, default: bool) -> bool:
    v = _env(name).lower()
    return default if not v else v in ("1", "true", "yes", "on")


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
    vlm_frame_format: str  # images | montage | video_frames | video
    vlm_max_images: int  # cap on images per request in images mode; 0 = no cap

    # Agent LLM (W&B Inference)
    agent_base_url: str
    agent_model: str
    agent_api_key: str
    agent_enabled: bool  # agent writes the call's decision note

    # W&B logging
    wandb_project: str
    wandb_entity: str
    weave_enabled: bool  # trace every session to Weave (needs a W&B key)
    prompt_version: str  # which COHERENCE prompt in reason/prompts.py the pipeline uses

    # Perception
    yolo_model: str

    # Index
    index_backend: str  # sqlite | local | vast
    vast_schema: str  # our own schema in the team VastDB bucket, never vss-schema
    vast_table: str
    embed_model: str  # Cosmos Embed1 on the organisers' GPU; empty URL = keyword search
    vss_archive: bool  # upload each session clip to the team's VSS archive
    vss_camera_id: str

    # UI
    ui_origins: tuple[str, ...]

    # Organisers' stack (from .env.team; empty when not at the event)
    gpu_bearer_token: str
    cosmos3_reason_url: str
    yolo_url: str
    cosmos_embed1_url: str
    ingress_url: str
    team_username: str
    team_password: str
    s3_endpoint: str
    access_key: str
    secret_key: str
    vastdb_bucket: str
    vdb_schema: str
    vdb_endpoint: str

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
        vlm_max_images=int(_env("VLM_MAX_IMAGES", "0") or 0),
        agent_base_url=_env("AGENT_BASE_URL", "https://api.inference.wandb.ai/v1"),
        agent_model=_env("AGENT_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        agent_api_key=_env("AGENT_API_KEY") or _env("WANDB_API_KEY"),
        agent_enabled=_flag("AGENT_ENABLED", True),
        wandb_project=_env("WANDB_PROJECT", "liveness-agent"),
        # On the lab VM the team is WANDB_TEAM; WANDB_ENTITY overrides it for a personal run.
        wandb_entity=_env("WANDB_ENTITY") or _env("WANDB_TEAM"),
        weave_enabled=_flag("WEAVE_ENABLED", True),
        prompt_version=_env("PROMPT_VERSION", "v7"),
        yolo_model=_env("YOLO_MODEL", "yolo11n-pose.pt"),
        index_backend=_env("INDEX_BACKEND", "sqlite"),
        vast_schema=_env("VAST_SCHEMA", "hackathon"),
        vast_table=_env("VAST_TABLE", "liveness_sessions"),
        embed_model=_env("COSMOS_EMBED1_MODEL", "nvidia/cosmos-embed1"),
        vss_archive=_flag("VSS_ARCHIVE", True),
        vss_camera_id=_env("VSS_CAMERA_ID", "liveness"),
        ui_origins=tuple(o.strip() for o in _env("UI_ORIGINS").split(",") if o.strip()),
        gpu_bearer_token=_env("GPU_BEARER_TOKEN"),
        cosmos3_reason_url=_env("COSMOS3_REASON_URL"),
        yolo_url=_env("YOLO_URL"),
        cosmos_embed1_url=_env("COSMOS_EMBED1_URL"),
        ingress_url=_env("INGRESS_URL"),
        team_username=_TEAM_FILE.get("USERNAME") or _env("TEAM_USERNAME") or _env("USERNAME"),
        team_password=_TEAM_FILE.get("PASSWORD") or _env("TEAM_PASSWORD") or _env("PASSWORD"),
        s3_endpoint=_env("S3_ENDPOINT"),
        access_key=_env("ACCESS_KEY"),
        secret_key=_env("SECRET_KEY"),
        vastdb_bucket=_env("VASTDB_BUCKET"),
        vdb_schema=_env("VDB_SCHEMA"),
        vdb_endpoint=_env("VDB_ENDPOINT") or _env("S3_ENDPOINT"),
        root=ROOT,
        data_dir=ROOT / "data",
    )


settings = load_settings()
