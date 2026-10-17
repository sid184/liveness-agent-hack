# Hackathon context: VAST Builders Challenge, London

This file is the working context for building our entry. Read it fully before doing anything in this repo. Update the "Where we are now" section as work progresses.

## The event

- **What:** VAST Builders Challenge, London leg ("Real-Time Video Agents Hack - LDN"). Three-city series (NYC, London, SF). Event page: https://luma.com/vastlondon
- **When:** next week (single day). Doors 8:30, keynotes 9:00, build 9:30 to 4:30, demos 4:30, awards 6:30.
- **Hosts / sponsors:** tokens& (organiser), VAST Data, NVIDIA, CoreWeave, Weights & Biases, Cursor.
- **Brief:** "Spend the day building agents to unlock the value of video." Most video sits in storage unwatched; build something that makes it useful.
- **Judges:** two from VAST Data (Brian Verkley, Ram Bansal), one from NVIDIA (Abubakr Karali, Senior Solutions Architect), one from Weights & Biases (Junaid Butt, AI Solutions Engineer).
- **Prizes:** 1st place NVIDIA DGX Spark; also HuggingFace Microduck, Cursor credits, gift cards.
- **Teams:** up to four. We are two (Sid and Karim) and plan to recruit one or two engineers on the morning.
- **Registration:** approval-based, closed 9 Oct. Registered with personal emails. Cursor account on the same email so the organisers' credits land; we build in VS Code with Claude Code regardless.

### The stack they provide on the day

- **Video understanding:** NVIDIA Cosmos (physical-world reasoning vision-language model). Reason over a clip.
- **Semantic search:** natural-language search across hours of footage.
- **Object detection:** YOLO, real-time detection and tracking across frames.
- **General LLMs:** models served via Weights & Biases, for agent logic.
- **Data layer:** VAST AI OS handles ingestion and storage. Models served on CoreWeave GPUs.

Judging is implicit but sponsor-run hackathons reward: (a) using the stack properly, (b) looking shippable, (c) a clear, compelling demo, (d) a real buyer and business reason. Use every stack component visibly and name them on the architecture slide. The NVIDIA judge will notice whether Cosmos is used for what it is built for (physical-world reasoning), so do not point it at talking heads.

## What we are building

### The idea: moment-of-action liveness via physical challenge-response

**Problem (pitch version):** Deepfakes mean a face and a voice on a call no longer prove anything. When someone on a video call asks finance to move £2m, the options today are trust it or hang up and call back. We built the ten-second check that answers the question on the spot: a physical challenge no real-time deepfake can pass, judged by an agent that reasons about the physical world.

**Mechanism:** Instead of detecting fakes (an arms race), make the caller do something physical and unpredictable that a real-time deepfake pipeline cannot render coherently: pass an object in front of the face, turn the head while the camera moves, hold up N fingers at the edge of frame, tilt a card to catch the light. An agent judges whether the physical interaction in the video is coherent (occlusion, lighting, parallax, timing) and returns a verdict with evidence clips.

**Why challenge-response:** it shifts the burden to the attacker. Challenges can change without retraining. Real-time generation is weakest exactly at novel physical interactions.

**Stack mapping:**
- YOLO: detect and track face, hands, objects per frame; feeds the geometric pass criteria (object crosses face box, hand present, timing inside window).
- Cosmos: "Did this physical interaction happen coherently?" reasoning verdict on the clip or sampled frames.
- Semantic search: index past sessions so the agent can retrieve similar attacks and show the system learning.
- W&B LLM: the agent loop (pick challenge, observe, combine YOLO + Cosmos signals, score, write verdict and reasoning). W&B logging for every run, shown on the architecture slide.
- VAST AI OS: on the day, replaces our local storage/search module. Keep that module swappable.

**The thing most teams won't have:** a findings/verdict log the agent writes to and reasons over across sessions, with evidence clips. This is our broader thesis in miniature (the system records its reasoning so the next verdict is better).

**Make it real-time:** at least one part of the demo runs live on a webcam. Heavy analysis may run on short recorded clips.

### Honest product context (for the pitch's "who buys it" slide)

- Yes, it is another verification step. It is only defensible when (1) it replaces a worse step (callbacks, holds, manual review, "call the CFO back") and (2) it fires only on high-stakes actions (new payee, large transfer, executive authorising a payment on a call).
- First buyers: corporate finance and treasury teams (fake-CFO fraud), then bank fraud teams, then platforms.
- Real product also needs device attestation (signed capture path) to defeat virtual-camera injection. Out of scope for the hackathon; mention it as "next" if asked.
- Existing active-liveness vendors (iProov, Onfido, Jumio) verify at onboarding. We verify at the moment of action on live calls with richer challenges.

### Fallback (decided now so nobody argues at 1pm)

**1pm checkpoint:** if the end-to-end path (challenge issued, video in, verdict out) is not working by 1pm on the day, switch to the **warehouse / floor bottleneck finder**: YOLO tracks people, pallets, forklifts; Cosmos describes what is happening per segment; semantic search over footage; agent produces a quantified, prioritised bottleneck list with evidence clips. Same repo structure; only the `challenge/` module is replaced by a `bottleneck/` module. Have 20 to 30 minutes of warehouse or shop-floor footage downloaded in advance.

## Repo layout (target)

```
liveness-agent/
  capture/      webcam or file -> frames (OpenCV)
  challenge/    picks a challenge, defines pass criteria, timestamps
  perception/   YOLO detect + track; face/hand/object boxes per frame
  reason/       Cosmos client: "did this physical interaction happen coherently?"
  index/        embeds sessions, vector search over past sessions (swappable; VAST on the day)
  agent/        W&B-hosted LLM orchestrates: issue -> observe -> score -> verdict + evidence
  ui/           minimal web page: live feed, challenge prompt, verdict
  eval/         real sessions, "attack" clips, W&B logging
  scripts/      one-off smoke tests
  CLAUDE.md     architecture, conventions, demo script (keep current)
  .env.example  variable names only, committed
  .env          real keys, gitignored
```

Conventions: Python 3.11+, one `.env.example`, a `Makefile` or simple task runner, `openai` client package for both NVIDIA and W&B OpenAI-compatible endpoints. Design the storage/search layer as one swappable module.

## Keys and accounts

| Service | Status | Env var | Notes |
|---|---|---|---|
| NVIDIA (build.nvidia.com) | DONE | `NVIDIA_API_KEY` | Testing key, valid to April 2027. **Hosted Cosmos API is disabled** (404 for every account since Feb 2026, confirmed by NVIDIA staff). Fallback vision model when the pod is off: `meta/llama-3.2-11b-vision-instruct`, one image per call, 3 to 7 s. |
| Weights & Biases | **REPLACE** | `WANDB_API_KEY` | Current key belongs to the employer account (email domain instalily.ai, entity `siddharta-instalily`). Create a personal W&B account, new key, and delete the `liveness-agent` test project from the employer entity. Inference works with the key alone. |
| Claude Code | personal account | n/a | Building in VS Code with Claude Code. Personal account only, nothing tied to employer. |
| GitHub | DONE | n/a | gh CLI active account `sid184`; employer account kept but inactive. Repo-local commit identity set to the sid184 noreply address. Remote: https://github.com/sid184/liveness-agent-hack (public, created on the day, Sat 17 Oct 2026). Single `main` branch for the day; everyone commits straight to it. The prep repo `sid184/liveness-agent` (private) is kept as history. |
| Hugging Face | DONE | `HF_TOKEN` | Read token `hackathon-cosmos` on Siddharta184. Licence accepted for `nvidia/Cosmos-Reason2-8B` (verified 200); `Cosmos-Reason2-2B` still 403, accept if wanted. Only needed to pull weights for self-hosting. |
| Cursor | TODO (by Sunday) | n/a | Free tier, same email as registration, so organiser credits land. Not our editor. |
| RunPod (GPU rental) | DONE | `RUNPOD_API_KEY` | Pod `cosmos` (id `exfdzxdrcdk7st`, A40 48GB, $0.59/hr, CA-MTL-1) serves self-hosted `Cosmos-Reason2-8B` with vLLM. **Stopped between sessions**; start it for a session, stop it after (`runpodctl pod start|stop exfdzxdrcdk7st`). Network volume `cosmos` 50GB at `/workspace` keeps the weights and the vLLM venv, about $0.12/day while stopped. Balance $13.79 after tonight. SSH key `~/.ssh/runpod_cosmos` is on the pod. Account was created with the work email; change it under Settings. |
| VAST AI OS, CoreWeave | hackathon-day only | n/a | Not available in advance. Local stand-in: folder of clips + OpenCV + Chroma or LanceDB. |

Local tooling: Python 3.11+, `ffmpeg`, `pip install ultralytics opencv-python openai wandb python-dotenv` (add `chromadb` or `lancedb` for the index module).

## Practice plan this week

1. **Tonight (Sat):** accounts and keys (done), repo (done), first Cosmos call on a test image, first YOLO call on a frame.
2. **Sun:** record ~10 short clips of Sid and Karim doing three challenges (pen past face, head turn while moving phone, N fingers). Record ~5 attack clips (photo on a screen, replayed video, still image). Run Cosmos on each with "did the person pass an object in front of their face?" and learn how it describes physical interaction. That phrasing drives the pass/fail prompts.
3. **Mon to Tue:** wire YOLO tracking to challenge criteria. Combine with Cosmos verdict into one score. Log every run to W&B.
4. **Wed:** agent loop and UI. Live webcam, challenge on screen, verdict in under ten seconds. Record a backup demo video.
5. **Thu:** eval and pitch rehearsal. Sketch the warehouse fallback in the same structure so switching costs an hour.

Open questions to settle with the organisers: are pre-written code and personal footage allowed on the day (most hackathons yes; confirm); and the real latency of hosted Cosmos on a 5 to 10 second clip, which decides whether "real-time" means live or near-live.

## Day plan

- 9:30 to 10:30: one clip through the whole pipeline end to end, ugly. Nothing else until that works.
- 10:30 to 1:00: agent and verdict log on top.
- 1:00: checkpoint. Continue or switch to fallback.
- 1:00 to 3:00: scoring layer, live feed, UI.
- 3:00 to 4:00: freeze. Record backup video. Rehearse pitch three times.
- Pitch (under four minutes): problem in one sentence an ops or finance person would say (30s); live feed ten seconds then three on-stage questions/challenges (2 min); architecture slide naming every stack component (30s); who buys it and why (30s).

Roles: one on video pipeline (capture -> YOLO -> Cosmos -> index), one on agent and verdict log, one on UI and demo, one floating and preparing the pitch. Karim owns the quantification/scoring layer. Sid owns pitch and architecture.

## Where we are now (updated Sat 10 Oct 2026, late evening)

- **Repo skeleton exists, runs, and is on GitHub.** `make test` (offline), `make smoke-yolo`,
  `make smoke-vlm`, `make smoke-pipeline` (clip -> challenge -> YOLO -> VLM -> verdict -> local
  index -> W&B). Layout, conventions and dependency status are in `CLAUDE.md`. Python 3.12 via uv,
  ffmpeg installed. Remote https://github.com/sid184/liveness-agent-hack, single `main` branch.
- **Real Cosmos works, self-hosted.** Hosted Cosmos is disabled (keys table), so we serve
  `nvidia/Cosmos-Reason2-8B` ourselves with vLLM on a RunPod A40 (`scripts/serve_cosmos.sh`).
  Verified tonight end to end: 3 s for 8 frames with a plain prompt, 10 s with the think/answer
  prompt, and it reasons frame by frame ("covers the face from frame 3 to frame 6"). Weights and
  the vLLM install live on the network volume, so a restart is about 4 min with nothing to download.
  The pod is **stopped**. To use it: `runpodctl pod start exfdzxdrcdk7st`, SSH in (IP and port
  change per start; read them from the console), run the serve script under nohup, wait for
  "Application startup complete", run, and stop the pod when done. **Cosmos is the default** in
  `.env` (decided Sun 11 Oct); Llama 3.2 11B Vision on NVIDIA hosted is the fallback when the pod
  is off (comment out the three `VLM_*` lines). Full pod ops are in `CLAUDE.md`.
- **Keys:** NVIDIA ok. Hugging Face ok. RunPod ok. W&B key is the employer account and must be
  replaced before any more logging. GitHub CLI is on the personal account. `.env.dev` is a leftover
  duplicate of `.env` with non-standard names; delete it.
- **Immediate next steps:**
  1. Personal W&B account and key; delete the `liveness-agent` project from the employer entity.
  2. Record the first real webcam clip (ten seconds, pass a phone across the face), start the pod,
     run `make smoke-pipeline ARGS="--video clip.mp4 --challenge object_pass --label live"`, stop
     the pod. Start to stop is about 15 min, under 20 cents.
  3. Invite Karim to the GitHub repo (need his username).
  4. Ask the organisers: is Cosmos provided as an endpoint on the day, and are pre-written code
     and personal footage allowed.
  5. Sunday: record ~10 live clips and ~5 attack clips, run each through the pipeline, then fix the
     COHERENCE prompt: the VLM currently parrots the pass criteria back with confidence 100.
  6. Monday: decide face and hand detection (COCO YOLO has neither), wire the live webcam window.
- **Open decisions:** face/hand model; scoring (Karim). Cosmos is decided: self-hosted on the RunPod
  A40 for dev sessions, the organisers' endpoint on the day if offered, else the pod.
