# Repo skeleton (Sat 10 Oct 2026)

Goal: a runnable skeleton matching the layout in HACKATHON_CONTEXT.md, with the two swappable
seams (vision reasoner, session index) behind config, and two passing smoke tests.

- [x] uv project: pyproject.toml, Python 3.12 pinned, deps installed
- [x] ffmpeg installed via Homebrew
- [x] .env with standard var names (NVIDIA_API_KEY, WANDB_API_KEY, HF_TOKEN); .env.example committed
- [x] .gitignore extended (venv, caches, data/, wandb/, model weights)
- [x] config.py: single settings object read from .env
- [x] capture/: frames from webcam or file, frame sampling, synthetic test image
- [x] perception/: YOLO detect + track wrapper
- [x] reason/: OpenAI-compatible vision reasoner (Llama 3.2 Vision now, Cosmos on the day)
- [x] challenge/: challenge catalogue with pass criteria
- [x] index/: SessionIndex interface + local implementation
- [x] agent/, ui/, eval/: stubs with the intended interface documented
- [x] scripts/smoke_vlm.py runs and returns a description
- [x] scripts/smoke_yolo.py runs and returns detections
- [x] Makefile: setup, smoke-vlm, smoke-yolo, lint, test
- [x] CLAUDE.md: architecture, conventions, demo script, external-dependency status
- [x] README.md
- [x] HACKATHON_CONTEXT.md "Where we are now" and keys table updated
- [x] tasks/lessons.md started

## Review (Sat 10 Oct 2026)

Done: full skeleton per the context layout, uv + Python 3.12, ffmpeg, standard env names, lint
clean, 6 offline tests passing, three smoke scripts verified against live services:
- YOLO: 2 persons + tie on a real photo in 0.75 s (first inference).
- VLM: Llama 3.2 11B Vision, 7 s on a single image, 3.5 s on a 6-frame montage.
- Pipeline: synthetic clip -> verdict -> local index -> W&B run, 13 s wall clock.

Found along the way:
- Hosted Cosmos API is disabled for all accounts (NVIDIA staff, Feb 2026). Stand-in VLM in place.
- Llama 3.2 Vision hosted accepts one image per call; montage mode added and made the default.
- Llama 3.2 90B Vision hung for over three minutes; 11B is the default. Client timeout 90 s added.
- The W&B key is the employer account (instalily.ai). Flagged in the context file; needs replacing.
- The VLM parrots pass criteria with confidence 100 on a cartoon. Prompt work is Sunday's first job.

Not done (out of scope for the skeleton): live webcam window, agent LLM in the loop, embeddings
search, ui/, scoring beyond the placeholder, GitHub remote and first commit.

# Self-hosted Cosmos on RunPod (Sat 10 Oct 2026, late)

Goal: real Cosmos-Reason2-8B output to calibrate prompts against, for a few dollars of credit.

- [x] RunPod account, API key (`RUNPOD_API_KEY` in .env), SSH key `~/.ssh/runpod_cosmos` on the pod
- [x] Network volume `cosmos` 50GB (CA-MTL-1) at /workspace; pod `cosmos` on an A40 48GB, $0.59/hr
- [x] vLLM 0.31 serving `nvidia/Cosmos-Reason2-8B` with the qwen3 reasoning parser (`scripts/serve_cosmos.sh`)
- [x] Client falls back to `reasoning_content`; smoke_vlm prints the reasoning trace
- [x] Verified single image, 8-frame sequence, structured COHERENCE verdict, think/answer prompt
- [x] vLLM installed into a persistent venv on the volume so restarts skip the install
- [x] Pod stopped; billing confirmed down to the volume only
- [ ] First real webcam clip through the pipeline against Cosmos (Sid was on a call; next session)

## Review (Sat 10 Oct 2026, late)

Numbers on the A40: weights pull 14 s (16GB to the volume), cold start about 4 min, single image
3.1 s, 8 frames 3.2 s, structured verdict 4.5 s, think/answer prompt 10.4 s. It described a
left-to-right crossing correctly and reasons per frame. Balance $13.79 after tonight; the volume
costs about $0.12/day while the pod is stopped.

Found along the way: the proxy URL is fixed per pod id and survives stop/start, the SSH IP and port
do not. `--reasoning-parser qwen3` puts plain replies in `reasoning_content`. A venv made with
`--system-site-packages` silently skipped installing vllm. Pip onto the NFS volume is slow (about
15 min for 8GB), so the install was done once and persisted. The VLM is still sycophantic
(confidence 100 on a cartoon); that is Sunday's prompt work.
