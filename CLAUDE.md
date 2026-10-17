# liveness-agent: working notes for Claude Code

Read `HACKATHON_CONTEXT.md` first (brief, plan, status). Keep both files current. Multi-step
work is tracked in `tasks/todo.md`; corrections go in `tasks/lessons.md`.

## What this is

Passive liveness for video calls. The caller is never asked to do anything. While the call runs,
the agent records the caller's tile in back-to-back windows (8 s) and judges each:

- YOLO pose + YOLO detection (`perception/cues.py`): one face, no jumps, natural micro-movement
  (a held photo moves rigidly), not frozen (consecutive frames differ), no phone/monitor/laptop
  around the face, and the hard moments (head turned, a hand or cup at the face).
- Cosmos judges the frames YOLO picked, hard moments first, with a checklist that works on a
  still frontal face too (prompt v6).
- Window: FAIL on an artefact, a frozen feed or a device; INCONCLUSIVE without natural movement;
  else PASS, with assurance **strong** (clean through a hard moment) or **basic** (clean and alive
  but still). Call: FAIL on the first failed window, PASS after two clean ones (strong if any
  window was), INCONCLUSIVE after six.

The agent (W&B Inference) writes the decision note with similar past windows in context. Every
window is logged (VastDB), archived (VSS), traced (Weave) and mirrored to W&B.

**Pivots on 17 Oct (build day):** first the three-challenge catalogue became one head-circle
challenge with a shuffled order; then (midday) challenges were dropped altogether for passive
watching (Karim). Face swaps break when the head turns (NYU GOTCHA, Euro S&P 2024), so the frames
Cosmos sees are the most-turned moments YOLO measured. Test set: our own call clips as `live`, and
`attack` clips made by face-swapping them, replaying them on a phone screen, or generating them
with video models. Label every clip with how it was made.

## Architecture

```
capture/    webcam or file -> Frame(image, index, t_seconds); sample_frames(); montage()
perception/ Detector: YOLO pose -> Detection(+keypoints); pose.py: keypoints -> head pose, gates
reason/     VisionReasoner: OpenAI-compatible VLM call with frames  <- the Cosmos seam
            prompts.py: COHERENCE_VERSIONS (v1 naive, v2/v3 head-circle, v4 passive, v5 + think)
agent/      loop.run_session(): one window: YOLO gates -> pick_frames -> Cosmos -> combine ->
            Embed1 -> VSS archive -> verdict log; every stage recorded in signals["stack"]
            loop.decide_call() / finish_call(): window verdicts -> call decision + agent note
            llm.py: agent on W&B Inference writes the decision note, always with a fallback
index/      SessionIndex (add/all/get/where/search/similar); sqlite | local | vast  <- swappable
            vast.py: our table hackathon.liveness_sessions in the team VastDB bucket
            embed.py: Cosmos Embed1 (256-dim) vectors; search falls back to keywords without it
eval/       wandb_log (one run per process, a row per session), tracing (Weave ops, no-op without
            a key), vss_archive (upload / search / ask / pull clips on the team VSS)
ui/         FastAPI server + overlay: floating dot over the call, wakes on click or /api/trigger
  web/      React + Vite + Tailwind source (React Bits components in src/reactbits/)
  static/   build output of web/, committed so `make ui` needs no Node
scripts/    check_stack (every sponsor component, OK/FAIL), eval_clips (prompt versions -> W&B),
            smoke_*; network checks live here, not in tests/
config.py   settings dataclass read from .env.team then .env; the only place os.environ is read
```

One window (`agent/loop.py`): `observe()` (YOLO pose, gates, pose track) -> `pick_frames()`
(most-turned moments) -> `judge()` (Cosmos, prompt `PROMPT_VERSION`) -> `combine()` -> `embed()`
-> `vss_archive.upload()` -> `get_index().add()`. A call (`ui/server.py` `/api/watch/*`, or
`scripts/smoke_pipeline.py` on a recording cut into windows): windows until `decide_call()` is not
WATCHING, then `finish_call()` (agent note, decision stored on the last window). Each step appends
`{step, component, latency_s, detail}` to `signals["stack"]`; the session page renders it as "How
this verdict was made". Windows, the decision and the agent calls are Weave ops.

## Conventions

- Python 3.12 via uv. Run everything from the repo root: `uv run python -m scripts.<name>`.
- Model ids, URLs and backends live in `.env`, never in code or prose. Read them via
  `config.settings`. Changing the vision model on the day is two lines in `.env`.
- Every external service degrades instead of failing a session: no Embed1 -> keyword search, no
  agent -> random order and Cosmos's text, no VSS -> no archive, no W&B key -> no logging or
  tracing. Only the vision model and the index are required.
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
make check            # every sponsor component, one line each (ARGS="--write" adds a VastDB row)
make eval             # labelled clips x prompt versions -> W&B (ARGS="--from-vss --index v2")
make ui               # overlay at http://localhost:8765 (Chrome)
make ui-build         # after editing ui/web: rebuild into ui/static and commit both
make test             # offline tests
make lint / make fmt
make smoke-yolo  ARGS="--image data/samples/zidane.jpg"
make smoke-vlm   ARGS="--video clip.mp4 --frames 6"
make smoke-pipeline ARGS="--video clip.mp4 --label live --wandb"   # replays it as a call
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

## Event stack facts (verified 17 Oct, team 10)

- Run the pipeline **on the VM** (`ssh bc-karimbakkali15`, repo at `~/liveness-agent-hack`, uv in
  `~/.local/bin`). The VastDB and VSS hosts only resolve there; from the laptop only the GPU
  endpoints and W&B work. For the UI: `make ui` on the VM, then on the laptop
  `ssh -f -N -L 8765:localhost:8765 bc-karimbakkali15` and open http://localhost:8765 (localhost
  keeps the webcam allowed).
- `.env.team` = `/config/team-10.config` + `WANDB_*` from the VM environment (team `vastdata`,
  project `team-10`) + GPU URLs `http://166.19.38.112:800{1,2,3,4}`. `make check`: 11/11 on the VM.
- Cosmos model id is `nvidia/cosmos3-nano-reasoner` (not `cosmos3-reason`); 5 images max.
- VSS renames uploads to `<timestamp>_<id>.mp4`: labels must be **tags** (`live` / `attack`,
  optional `udlr`). The team tags raw footage `liveness`; our judged sessions are tagged
  `liveness-session`. `/videos/explore` returns `{chunks, total}`, `limit` max under 500.

## Known gaps, in priority order

1. **No real attack clips yet.** Verified on 17 Oct against fakes made from our own clips: 3
   frozen frames FAIL, 3 held photos never cleared (INCONCLUSIVE), live clips PASS strong.
   Untested: a phone or monitor replay (device cue), a face swap, a hand or cup at the face.
2. The still-person thresholds rest on one 3 s still slice (`perception/cues.py`). Record 30 s of
   someone sitting still and talking, run `uv run python -m scripts.calibrate_cues`, re-check.
3. A held photo ends INCONCLUSIVE, not FAIL. Failing "no natural movement for N windows" is a
   one-line change once (2) shows real still callers stay well above the threshold.
4. Cosmos says PASS on photos and frozen frames (it judges what it sees, and a photo of a person
   looks fine); YOLO's cues carry those attacks. Cosmos earns its place on hard moments.
5. Scoring constants are untuned on labelled clips; the overlay is untried on a real Meet call.

## Demo script (under four minutes)

1. Problem in one sentence an ops or finance person would say (30 s).
2. Live call: the agent watches without asking anything and decides; then an attack clip fails
   (2 min). Name each sponsor in one sentence as it acts (YOLO gates and frame choice, Cosmos
   reasoning with cited frames, VastDB log and VSS search, W&B agent note, eval and Weave trace).
   Plan: "Sponsor plan" in `HACKATHON_CONTEXT.md`.
3. Architecture slide naming every stack component: YOLO, Cosmos, semantic search, W&B LLM and
   logging, VAST AI OS, CoreWeave (30 s).
4. Who buys it and why: finance and treasury first, then bank fraud teams (30 s).

Fallback decided in advance (1 pm checkpoint): warehouse bottleneck finder, same repo, `challenge/`
replaced by `bottleneck/`. See `HACKATHON_CONTEXT.md`.
