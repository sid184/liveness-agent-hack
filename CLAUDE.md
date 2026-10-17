# liveness-agent: working notes for Claude Code

Read `HACKATHON_CONTEXT.md` first (brief, plan, status). Keep both files current. Multi-step
work is tracked in `tasks/todo.md`; corrections go in `tasks/lessons.md`.

## What this is

Moment-of-action liveness for video calls. The agent asks the caller to turn their head slowly
through four directions (up, down, left, right) in a randomised order, watches, and judges whether
the head moved like a real 3D head: pose gates from YOLO keypoints, physical coherence from Cosmos.
Pass or fail with evidence frames and written reasoning, every verdict logged and indexed so later
verdicts can reason over earlier ones.

**Pivot on 17 Oct (build day):** the three-challenge catalogue (object pass, head turn, fingers) is
replaced by one challenge, the head circle, with the direction order shuffled per session so a
recording cannot be replayed. Face-swap deepfakes break at profile (NYU GOTCHA, Euro S&P 2024),
which is the artefact Cosmos is asked to find. Test set: our own head-circle clips as `live`, and
`attack` clips made by face-swapping those clips, replaying them on a phone screen, or generating
them with video models. Label every clip with how it was made.

## Architecture

```
capture/    webcam or file -> Frame(image, index, t_seconds); sample_frames(); montage()
challenge/  catalogue of challenges with pass criteria; issue() stamps one with a time window
perception/ Detector: YOLO detect/track -> Detection(label, conf, box, track_id)
reason/     VisionReasoner: OpenAI-compatible VLM call with frames  <- the Cosmos seam
agent/      loop.run_session(): challenge -> frames -> YOLO -> VLM -> combine -> SessionRecord
            llm.AgentLLM: text model on W&B Inference (not yet used in the loop)
index/      SessionIndex (add/search/all/where); SQLite now, VAST on the day  <- swappable
eval/       wandb_log: start_run/log_session/finish, mirrors the index record; no-op without a key
ui/         (empty) live feed + challenge + verdict page, Wed
scripts/    smoke_vlm, smoke_yolo, smoke_pipeline; network checks live here, not in tests/
config.py   settings dataclass read from .env; the only place os.environ is read
```

One session (`agent/loop.py`): `issue()` -> `sample_frames()` -> `Detector.detect()` per frame ->
`geometric_signal()` -> `VisionReasoner.describe(COHERENCE prompt)` -> `parse_verdict()` ->
`combine()` -> `SessionRecord` -> `get_index().add()`.

## Conventions

- Python 3.12 via uv. Run everything from the repo root: `uv run python -m scripts.<name>`.
- Model ids, URLs and backends live in `.env`, never in code or prose. Read them via
  `config.settings`. Changing the vision model on the day is two lines in `.env`.
- Two swappable seams, keep them clean: `reason.VisionReasoner` (`VLM_BASE_URL`, `VLM_MODEL`,
  `VLM_FRAME_FORMAT`) and `index.get_index()` (`INDEX_BACKEND`).
- `ultralytics` is imported only inside `perception/`. `openai` only inside `reason/` and `agent/`.
- Prompts live in `reason/prompts.py` so clip experiments can iterate on text alone.
- `tests/` is offline only and must pass in under a few seconds. Anything that hits the network or
  loads weights is a `scripts/smoke_*` script.
- `make lint` and `make test` before committing. Ruff, line length 100.
- `data/` is gitignored: samples, outputs, the verdict log (`data/index/sessions.db`), recorded clips.
- The index is the source of truth for a session; W&B only mirrors the same record (`log_session`).
- Secrets in `.env` only. `.env.example` documents every variable.

## Commands

```
make setup            # uv sync (ffmpeg via brew separately)
make test             # offline tests
make lint / make fmt
make smoke-yolo  ARGS="--image data/samples/zidane.jpg"
make smoke-vlm   ARGS="--video clip.mp4 --frames 6"
make smoke-pipeline ARGS="--video clip.mp4 --challenge object_pass --label live --wandb"
ffmpeg -f avfoundation -framerate 30 -i "0" -t 10 data/clips/live_01.mp4   # webcam clip (device 0 is
                                   # the MacBook camera; list with -list_devices true -i ""; untested)
```

## External dependency status (verified Sat 10 Oct 2026)

- **Hosted Cosmos is unavailable.** `nvidia/cosmos-reason2-8b` is listed by `/v1/models` on
  integrate.api.nvidia.com but every call returns 404 "Function not found for account". NVIDIA
  staff confirmed on the developer forum (7 Feb 2026) the hosted API was deliberately disabled.
  `cosmos-reason1-7b` was deprecated 24 Mar 2026. Options: the organisers' endpoint on the day;
  self-host `nvidia/Cosmos-Reason2-8B` with vLLM on a 24GB GPU (HF gated, token works, licence
  accepted for 8B); keep the stand-in below. Self-host recipe: `scripts/serve_cosmos.sh` on the
  GPU box, then point `VLM_BASE_URL` at it with `VLM_FRAME_FORMAT=images`. NVIDIA's stated minimum
  is 32GB for the 8B and 24GB for the 2B; a community L4 (24GB) run of the 8B worked at
  `--max-model-len 8192`, about 15 s per 8 s clip at 4 fps.
- **Self-hosted Cosmos verified (10 Oct, RunPod A40 48GB, vLLM 0.31):** weights 16GB pull in 14 s to
  the network volume, cold start about 4 min, then 3 s for 8 frames with a plain prompt and 10 s with
  the think/answer prompt (about 200 reasoning tokens). It reasons frame by frame ("covers the face
  from frame 3 to frame 6"). Quirk: vLLM's `--reasoning-parser qwen3` puts a plain reply entirely in
  `reasoning_content`, so the client falls back to that when `content` is empty. Pod proxy URL is
  `https://<pod-id>-8000.proxy.runpod.net/v1`, any API key string.
- **Pod ops (RunPod, pod `cosmos`, id `exfdzxdrcdk7st`):** $0.59/hr while running, about $0.12/day
  stopped (50GB network volume). The volume at `/workspace` keeps the weights (`/workspace/hf`) and
  the vLLM venv (`/workspace/venv`); the container disk is wiped on stop. Session:
  `runpodctl pod start exfdzxdrcdk7st` (or the console); read the SSH IP and port from the console
  (they change per start) and `ssh -p <port> -i ~/.ssh/runpod_cosmos root@<ip>`; copy the script
  with `scp` if `/workspace/serve_cosmos.sh` is missing; `nohup bash serve_cosmos.sh >
  /workspace/vllm.log 2>&1 &`; wait for "Application startup complete" (about 4 min). `.env` already points at the pod.
  With the pod off, comment out the three `VLM_*` lines in `.env` to fall back to Llama. **Stop
  when done:** `runpodctl pod stop exfdzxdrcdk7st`. A running pod bills whether or not it serves.
- **Default VLM is the Cosmos pod** (set in `.env`; code defaults are the fallback below).
- **Fallback VLM:** `meta/llama-3.2-11b-vision-instruct` on NVIDIA hosted. Accepts exactly one
  image per call, hence `VLM_FRAME_FORMAT=montage` (frames tiled and numbered in one image).
  3 to 7 s per call. The 90B variant hung for over three minutes; avoid.
- **Cosmos request format** (NIM / vLLM, for the day): OpenAI chat with `image_url` base64 (up to
  16), `video_url` base64 mp4, or `video_frames` list; `media_io_kwargs: {video: {fps}}` default 4;
  reasoning via think/answer tags in the prompt (`reason.prompts.COSMOS_THINK_SUFFIX`);
  `max_tokens` 4096; best under 16k multimodal tokens. Cosmos-Reason2 is a Qwen3-VL fine-tune.
- **W&B Inference** works with the key alone, no project header. Text models only (Llama 3.3 70B,
  gpt-oss-120b, DeepSeek V4, Qwen 3.8 27B, Nemotron 3.5). Logging verified end to end.
- **YOLO:** switching to `yolo11n-pose.pt` (COCO keypoints: nose, eyes, ears, shoulders) for
  head pose. Downloaded and loaded on 17 Oct. The detection model `yolo11n.pt` has no `face` or
  `hand` class, which is why pose keypoints are the route.

## Build plan, 17 Oct, in order (file by file)

Steps 1 to 3 need nothing external. Step 4 needs Cosmos (team config or our pod) and clips.

1. **Head pose from YOLO** (30 min)
   - `perception/detector.py` edit: load `yolo11n-pose.pt`, return COCO keypoints with each box
     (0 nose, 1 left eye, 2 right eye, 3 left ear, 4 right ear). Weights verified: 6 MB, 2 s load.
   - `perception/pose.py` new: keypoints -> yaw and pitch proxies per frame, "facing" label
     (up/down/left/right/centre), ear visibility.
   - `.env.example`, `.env` edit: `YOLO_MODEL=yolo11n-pose.pt`.
2. **Challenge and prompt** (15 min)
   - `challenge/catalog.py` edit: single `head_circle` challenge; param `order` = shuffled
     `["up", "down", "left", "right"]`; instruction and criteria templated on it.
   - `reason/prompts.py` edit: `COHERENCE` rewritten for rigid head rotation (ears, nose, jaw
     reveal and occlude consistently; texture stays attached at profile, no slide or flicker;
     hair, neck, background move with the head); must cite frames; must say what would fail it.
3. **Scoring** (15 min)
   - `agent/loop.py` edit: `geometric_signal` -> pose gates (exactly one face throughout; each
     direction reached; in the asked order; inside the window; smooth, no jumps between frames).
     `combine` -> `score = cosmos * (0.3 + 0.7 * geo)`, `geo` graded 0..1 from gates met,
     threshold 0.5, `ERROR` when no verdict.
   - `tests/test_offline.py` edit: pose maths, gates, formula.
4. **Real clips** (1 pm checkpoint)
   - `data/clips/live_*.mp4`, `data/clips/attack_*.mp4` (gitignored), label in the filename.
   - `scripts/smoke_pipeline.py` edit: drop `--challenge`; keep `--label`.
   - `scripts/eval_clips.py` new: run every clip, print a table of score by label; tune prompt
     and threshold until live and attack separate.
5. **Live path and web UI** (2 h)
   - `capture/frames.py` edit: `record_window(seconds, source=0)` webcam -> clip file.
   - `ui/app.py` new: FastAPI; start session, stream feed, get verdict.
   - `ui/static/index.html` new: feed, challenge line + countdown, verdict panel with cited frames.
   - `agent/loop.py` edit: `run_session` also returns the cited frame indices.
   - `Makefile` edit: `make ui`.
6. **Agent in the loop** (45 min, if time)
   - `agent/loop.py` edit: `AgentLLM` picks the direction order and writes the verdict paragraph.
   - `reason/prompts.py` edit: the two prompts.
7. **Track B, VM seat holder, in parallel** (see `tasks/todo.md`)
   - `index/vast.py` new, `index/embed.py` new, `eval/vss_archive.py` new,
     `index/__init__.py` edit (`vast` branch).
8. **From 3:30:** freeze, backup video of the demo, rehearse. Submit at tokensand.com/vastlondon.

Untouched: `config.py` beyond new settings, `reason/vlm.py`, `index/sqlite.py`,
`eval/wandb_log.py`, the docs.

## Known gaps, in priority order

1. The VLM is sycophantic: on a cartoon clip it parroted the pass criteria back and gave
   confidence 100. The COHERENCE prompt must demand concrete evidence and be tested against
   attack clips on Sunday before any scoring work.
2. `combine()` is a placeholder (VLM decides, YOLO can veto). Karim owns scoring.
3. Live webcam path: `run_session` takes a file; a buffered `iter_frames()` window is needed.
4. `AgentLLM` is unused; challenge choice is random. The agent should pick challenges and write
   the verdict text, with the index's similar past sessions in context.
5. `SqliteIndex.search` is keyword overlap; swap in embeddings (NVIDIA hosted or local).
6. `ui/` is empty.

## Demo script (under four minutes)

1. Problem in one sentence an ops or finance person would say (30 s).
2. Live feed: ten seconds, then three on-stage challenges with verdicts (2 min).
3. Architecture slide naming every stack component: YOLO, Cosmos, semantic search, W&B LLM and
   logging, VAST AI OS, CoreWeave (30 s).
4. Who buys it and why: finance and treasury first, then bank fraud teams (30 s).

Fallback decided in advance (1 pm checkpoint): warehouse bottleneck finder, same repo, `challenge/`
replaced by `bottleneck/`. See `HACKATHON_CONTEXT.md`.
