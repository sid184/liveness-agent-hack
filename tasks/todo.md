# Hackathon day work split (Sat 17 Oct 2026)

Two tracks, one `main` branch, no PRs. `make lint && make test` before every push. Tests stay
offline; anything that hits the network is a `scripts/smoke_*` script. Keep the two seams clean:
`index.SessionIndex` (`index/base.py`) and `reason.VisionReasoner` (`reason/vlm.py`). If you
need a setting, add it to `config.py` and `.env.example`; never hardcode a URL or key.

## Track A: Sid (laptop, own keys, no VM)

Critical path: real footage through the pipeline, then the live demo.

- [ ] Record clips into `data/clips/`: 5 live (phone pass, head turn, fingers), 3 attacks (face on
      a phone screen, replayed video, still photo). Name them `live_01.mp4`, `attack_01.mp4`.
- [ ] Run each: `make smoke-pipeline ARGS="--video data/clips/live_01.mp4 --challenge object_pass --label live"`
- [ ] Rewrite `reason/prompts.COHERENCE` so Cosmos must cite frames and say what would make it FAIL
- [ ] Replace `combine()` in `agent/loop.py`: `score = cosmos * (0.3 + 0.7 * geo)`, `geo = min(1, over_frames/2)`,
      threshold 0.5; tune on the labelled clips; update `tests/test_offline.py`
- [ ] Live window: `capture.iter_frames(0)` buffered for the challenge duration -> same `run_session`
- [ ] Web UI (`ui/`): live feed, challenge text + countdown, verdict + reasoning + evidence frames.
      Web app only (organisers' rule). FastAPI + one HTML page is enough.
- [ ] Agent in the loop (`agent/llm.py`): pick the challenge, write the verdict text. Until the team
      W&B key arrives, point `AGENT_BASE_URL` at NVIDIA hosted (same OpenAI client).
- [ ] Pitch: `docs/USE_CASES.md` pitch lines, architecture slide from `docs/target_workflow.mmd`

### Sponsor visibility (plan in `HACKATHON_CONTEXT.md`, "Sponsor plan")

- [x] **W&B eval** `scripts/eval_clips.py`: every clip in `data/clips/`, one W&B run per prompt
      version (`config.prompt_version`), a `wandb.Table` of clip, label, score, verdict, cited
      frames, reasoning; summary metrics: mean score by label, separation, accuracy at threshold.
      Keep each prompt version in `reason/prompts.py` so runs are comparable.
- [x] **Weave tracing**: `weave.init` with the team project; `@weave.op` on `Detector.detect`,
      `VisionReasoner.describe`, `AgentLLM.chat`, `run_session`. No-op without a key.
- [x] **Agent on W&B Inference**: `AgentLLM` picks the direction order and writes the verdict
      paragraph, with `get_index().search()` results (similar past sessions) in the prompt.
- [x] **Cosmos visible**: verdict view shows Cosmos's reasoning with its cited frames beside it.
- [x] **Session page stack labels**: each step tagged with component and latency (YOLO11 pose,
      Cosmos3-Reason on CoreWeave, VastDB, VSS, W&B Inference + Weave). Review page only.
- [x] **Port the overlay UI** from the prep repo (`sid184/liveness-agent`, branch `dev`, `ui/`).
- [ ] **Demo script**: one sentence per sponsor at the moment it acts; rehearse it.

## Track B: whoever has a VM seat (`bc-<username>`, `/config/<team>.config`)

Everything that needs the organisers' credentials. On the VM the config is already in the
environment, so `config.settings` picks it up with no `.env.team`.

- [ ] **First, 2 minutes:** `cat /config/*.config` -> save as a file -> AirDrop to Sid, who stores
      it as `.env.team` (gitignored). Unblocks Cosmos3 and the team W&B key on his laptop.
- [ ] Clone the repo on the VM, `make setup` (or `pip install -e .` if uv is missing), then
      `make smoke-vlm ARGS="--video <any clip> --frame-format video"` against Cosmos3-Reason.
      Report latency and whether `video` or `images` format works better. (`VLM_*` in `.env`
      already point at `COSMOS3_REASON_URL`; set `VLM_API_KEY=$GPU_BEARER_TOKEN`.)
- [x] `index/vast.py`: `VastIndex(SessionIndex)` on the `vastdb` SDK (`pip install vastdb pyarrow`),
      data VIP = `S3_ENDPOINT`, bucket `VASTDB_BUCKET`, new schema `hackathon`, table
      `liveness_sessions`, columns = `SessionRecord` fields (`evidence`/`signals` as JSON text).
      `add`, `all`, `where`, `search`. Wire `INDEX_BACKEND=vast` in `index/__init__.py`.
      Never touch `vss-collection`. Prove it with the organisers' `vastdb-read/query.py`.
- [x] `index/embed.py`: Cosmos Embed1 client (`POST $COSMOS_EMBED1_URL/v1/embeddings`,
      `request_type: query`, 256-dim). Store the vector per session; `search` = cosine over them.
      Works for both `SqliteIndex` and `VastIndex`.
- [x] `eval/vss_archive.py`: after each session, upload the clip to the VSS backend
      (`POST $INGRESS_URL/api/v1/videos/upload`, JWT from `/api/v1/auth/login`, tags
      `session_id`, `verdict`, `label`, `camera_id=liveness`). Then the organisers' search UI can
      answer "every check where the phone never crossed the face". Demo moment.
- [ ] **VAST demo moments (two of four judges; do these before the optional items):** verdict
      log in VastDB as the default backend on the day; VSS archive with tags so a search like
      "every check that failed at profile" returns our attack clips in the organisers' UI; the
      agent's "similar past sessions" read comes from VastDB. Rehearse the search query.
- [ ] Optional: `perception/remote.py`, their YOLO11s (`POST $YOLO_URL/v1/infer`, `video_base64`),
      returning the same `Detection` list. Only if local YOLO is a problem.
- [ ] Optional, late: deploy the web UI with the organisers' `deploy-app-no-registry` skill.

### Built 17 Oct (written from the organisers' docs; verify with `make check` on `.env.team`)

- [x] `scripts/check_stack.py` (`make check`): one OK/FAIL line per sponsor component
- [x] `agent/loop.py`: stages traced, timed and labelled in `signals["stack"]`; frames saved
- [x] Run `make check` with the team config; fix whatever is red (11/11 on the VM)
- [x] Passive watch (no challenge): windows -> decision, UI and API, verified on team clips
- [ ] Record and upload **attack** clips tagged `attack` (face swap, phone replay, generated);
      live ones tagged `live`
- [ ] `make eval ARGS="--from-vss"` on the team's uploaded clips; read the W&B comparison
- [ ] `make eval ARGS="--index v4 --archive"` once a version separates: fills VastDB and VSS
- [ ] Rehearse the VSS search ("face texture slides at profile") in the organisers' UI

## Integration points

| Piece | Interface | Switch |
|---|---|---|
| Vision model | `VisionReasoner.describe(images, prompt, video_path)` | `VLM_*` in `.env` |
| Verdict log | `SessionIndex.add/all/where/search` | `INDEX_BACKEND=sqlite|vast` |
| Embeddings | `embed(text) -> list[float]` (new, `index/embed.py`) | `EMBED_URL` (new) |
| Archive | `vss_archive.upload(record, clip_path)` (new) | `INGRESS_URL` set or not |

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
