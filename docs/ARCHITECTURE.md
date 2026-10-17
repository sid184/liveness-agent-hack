# Architecture (as of Sun 11 Oct 2026)

GitHub renders these inline. To edit one, paste the matching `.mmd` file into https://mermaid.live.

## Where it is going: the target workflow

```mermaid
%%{init: {"theme": "neutral", "themeVariables": {"fontFamily": "IBM Plex Mono, Menlo, monospace", "fontSize": "13px"}, "flowchart": {"nodeSpacing": 30, "rankSpacing": 60, "curve": "basis"}}}%%
%% liveness-agent: the target workflow, fully built. Paste the whole file into https://mermaid.live
flowchart LR

subgraph TRIG["Where it starts: the finance workflow"]
  REQ["<b><u>High-stakes request</u></b><br/>on a video call: new payee,<br/>large transfer, credential reset"]
  POL["<b><u>Policy gate</u></b><br/>does this action need a<br/>liveness check? by amount,<br/>payee novelty, caller role"]
  REQ --> POL
end

subgraph SETUP["Session setup: the agent (LLM on W#38;B)"]
  CTX["<b><u>Pull context</u></b><br/>who the caller claims to be,<br/>their past sessions, current<br/>risk level, known attack patterns"]
  PICK["<b><u>Choose the challenge</u></b><br/>object pass, head turn, N fingers,<br/>card tilt; vary the parameters,<br/>never repeat last time's,<br/>escalate when risk is high"]
  CTX --> PICK
end

subgraph LIVE["Live challenge: ten seconds on the call"]
  SHOW["<b><u>Challenge on screen</u></b><br/>one sentence and a countdown;<br/>the caller performs it live"]
  CAP["<b><u>Capture window</u></b><br/>the live feed is recorded for<br/>the window and timestamped;<br/>the camera path is signed so a<br/>virtual camera can't inject footage"]
  SHOW --> CAP
end

subgraph EVID["Evidence gathering: two pairs of eyes, plus memory"]
  YOLO["<b><u>Geometry: YOLO</u></b><br/>face, hand and object boxes<br/>tracked frame to frame: did the<br/>object cross the face, in which<br/>frames, inside the window?<br/>hard measurements that cannot<br/>be talked round"]
  COS["<b><u>Physics: Cosmos</u></b><br/>the clip plus a prompt that<br/>demands evidence: occlusion edge,<br/>lighting, parallax, motion<br/>coherence; must cite frames;<br/>structured findings + confidence"]
  RET["<b><u>Memory: semantic search</u></b><br/>similar past sessions and known<br/>attacks pulled from the verdict<br/>log, with how they turned out"]
  LOGDB[("<b>Verdict log</b><br/>VAST AI OS: every session,<br/>embedded, with evidence clips")]
  ATK[("<b>Attack library</b><br/>confirmed fakes and<br/>replays, labelled")]
  LOGDB --> RET
  ATK --> RET
end

subgraph JUDGE["Verdict: the agent (LLM on W#38;B)"]
  SCORE["<b><u>Score</u></b><br/>YOLO gates first: fail any,<br/>fail the check. Then Cosmos<br/>confidence against a threshold<br/>calibrated on labelled clips"]
  WRITE["<b><u>Write the verdict</u></b><br/>plain English a finance person<br/>can act on, citing frames and<br/>measurements; evidence clip<br/>cut from the window"]
  SCORE --> WRITE
end

DEC{"Verdict?"}

subgraph OUT["Outcomes"]
  PASS["<b><u>PASS</u></b><br/>the action proceeds; a signed<br/>verdict and the evidence clip<br/>are attached to the transaction"]
  RETRY["<b><u>INCONCLUSIVE</u></b><br/>a second, different challenge<br/>is issued; after two tries,<br/>fall back to a callback"]
  FAIL["<b><u>FAIL</u></b><br/>the action is blocked; the fraud<br/>team gets the clip and the<br/>reasoning; the caller sees only<br/>'could not verify'"]
end

subgraph LEARN["Learning loop"]
  REC["<b><u>Record everything</u></b><br/>verdict, reasoning, raw signals<br/>and clip into the verdict log;<br/>mirrored to W#38;B"]
  LBL["<b><u>Label later</u></b><br/>fraud team confirms frauds and<br/>false rejects; attacks join the<br/>library; thresholds recalibrate"]
  DASH["<b><u>Monitor</u></b><br/>W#38;B dashboard: pass rate,<br/>false rejects, time to verdict,<br/>attacks caught, per challenge"]
  REC --> LBL --> DASH
end

POL -- "yes" --> CTX
PICK --> SHOW
CAP --> YOLO
CAP --> COS
PICK -. "what we are about to ask" .-> RET
YOLO --> SCORE
COS --> SCORE
RET --> SCORE
WRITE --> DEC
DEC -- "pass" --> PASS
DEC -- "unsure" --> RETRY
DEC -- "fail" --> FAIL
RETRY -. "new challenge" .-> PICK
PASS --> REC
FAIL --> REC
LBL -. "feeds" .-> ATK
LBL -. "feeds" .-> LOGDB

classDef step fill:#ffffff,stroke:#2f5597,color:#222222
classDef store fill:#ffffff,stroke:#555555,color:#222222
classDef good fill:#eaf7ea,stroke:#2e7d32,color:#1b4d1e
classDef bad fill:#fdecec,stroke:#c0504d,color:#7a2e2c
classDef warn fill:#fff7e6,stroke:#c98a1b,color:#6b4a0f
class REQ,POL,CTX,PICK,SHOW,CAP,YOLO,COS,RET,SCORE,WRITE,REC,LBL,DASH step
class LOGDB,ATK store
class PASS good
class FAIL bad
class RETRY,DEC warn

style TRIG fill:#f3f3f3,stroke:#8a8a8a,color:#444444
style SETUP fill:#eef3fb,stroke:#5b7fb5,color:#2b4a7a
style LIVE fill:#f3f3f3,stroke:#8a8a8a,color:#444444
style EVID fill:#eef3fb,stroke:#5b7fb5,color:#2b4a7a
style JUDGE fill:#fdf3e7,stroke:#d08a2f,color:#6b4a0f
style OUT fill:#f3f3f3,stroke:#8a8a8a,color:#444444
style LEARN fill:#eaf7ea,stroke:#2e7d32,color:#1b4d1e
```

## The big picture today

```mermaid
%%{init: {"theme": "neutral", "themeVariables": {"fontFamily": "IBM Plex Mono, Menlo, monospace", "fontSize": "14px"}}}%%
%% liveness-agent: high-level architecture as it exists on Sun 11 Oct 2026. Paste the whole file into https://mermaid.live
flowchart TB

subgraph IN["INPUT"]
  CLIP["<b>RECORDED CLIP</b> (mp4 on disk)<br/>today: a cartoon test clip · real clips recorded: 0"]:::box
  CAM["<b>WEBCAM</b><br/>streaming code exists, nothing uses it"]:::ghost
end

subgraph RUN["ONE SESSION  (agent/loop.py, started by make smoke-pipeline)"]
  STEPS["<b>one clip's journey</b><br/>① issue a challenge: object pass · head turn · N fingers (random pick)<br/>② sample 6-8 evenly spaced frames from the clip<br/>③ YOLO on each frame (runs on the laptop) → person + object boxes → geometric signal, object pass only<br/>④ vision model judges the frames → OBSERVED · COHERENT · VERDICT · CONFIDENCE<br/>⑤ combine: the vision model decides, YOLO can only veto → PASS or FAIL + score<br/>⑥ write the session record to the verdict log, and to W#38;B if asked"]:::box
end

subgraph EXT["MODELS AND SERVICES"]
  NV["<b>NVIDIA HOSTED API</b><br/>Llama 3.2 11B Vision · the fallback when the pod is off<br/>free · one image per call, so frames are tiled · 3-7 s"]:::box
  RP["<b>COSMOS POD</b> (RunPod A40, ours)<br/>Cosmos-Reason2-8B on vLLM · THE DEFAULT · real video reasoning · 3-10 s<br/>stopped between sessions · 4 min to start · 59 c/hr"]:::box
  WB["<b>WEIGHTS #38; BIASES</b><br/>mirror of every session record, for us to compare runs<br/>key must move to a personal account<br/>text LLM for the agent · built, not used"]:::box
end

subgraph LOG["VERDICT LOG"]
  IDX[("<b>SQLITE DATABASE</b> (source of truth)<br/>data/index/sessions.db, one row per session<br/>challenge · verdict · score · reasoning · label · raw YOLO and VLM output<br/>filter and keyword search exist · nothing reads it back yet")]:::box
  VAST["<b>VAST AI OS</b><br/>replaces this file on the day"]:::ghost
end

subgraph NEXT["NOT BUILT YET"]
  UI["<b>UI</b><br/>live feed · challenge · verdict"]:::ghost
  AG["<b>AGENT</b><br/>LLM picks the challenge, writes the verdict,<br/>reads similar past sessions first"]:::ghost
  FH["<b>FACE + HAND DETECTION</b><br/>stock YOLO has neither"]:::ghost
  EV["<b>EVIDENCE CLIPS</b><br/>the record keeps only the file path"]:::ghost
end

CLIP == "clip path" ==> STEPS
CAM -. "future: live window" .-> STEPS
STEPS == "④ frames + prompt" ==> RP
STEPS -. "④ fallback: comment out 3 lines in .env" .-> NV
STEPS -- "⑥ same record, mirrored" --> WB
STEPS == "⑥ session record" ==> IDX
IDX -. "swap on the day" .-> VAST
IDX -. "future: similar past sessions" .-> AG
STEPS -. "future: verdict on screen" .-> UI

classDef box fill:#ffffff,stroke:#555555,color:#222222
classDef ghost fill:#fafafa,stroke:#999999,color:#666666,stroke-dasharray: 5 4

style IN fill:#f3f3f3,stroke:#8a8a8a,color:#444444
style RUN fill:#fdecec,stroke:#c0504d,color:#7a2e2c
style EXT fill:#eaf1fb,stroke:#5b7fb5,color:#2b4a7a
style LOG fill:#fbf6e4,stroke:#c9a227,color:#6b5410
style NEXT fill:#f1f1f1,stroke:#9a9a9a,color:#555555,stroke-dasharray: 6 4
```

## What happens when one clip runs today

```mermaid
%% liveness-agent: call order of one session (scripts/smoke_pipeline.py -> agent/loop.py run_session). Paste into https://mermaid.live
sequenceDiagram
    autonumber
    participant S as scripts/smoke_pipeline.py
    participant C as challenge.issue()
    participant F as capture.sample_frames()
    participant Y as perception.Detector (YOLO, runs locally)
    participant L as agent/loop.py run_session()
    participant R as reason.VisionReasoner
    participant V as VLM endpoint (NVIDIA hosted Llama by default, or the Cosmos pod)
    participant I as index.SqliteIndex (data/index/sessions.db)
    participant W as eval.wandb_log

    S->>C: issue(challenge_id or random)
    C-->>S: IssuedChallenge(instruction, criteria, params, issued_at, deadline)
    opt wandb flag
        S->>W: start_run(name, config)
    end
    S->>L: run_session(video, challenge, n_frames=6, label)
    L->>F: sample_frames(video, n)
    F-->>L: n evenly spaced Frames (BGR image, index, t_seconds)
    loop each frame
        L->>Y: detect(image, conf=0.25)
        Y-->>L: list of Detection(label, conf, box)
    end
    L->>L: geometric_signal() person frames, object frames, object over person (object_pass only)
    L->>L: format COHERENCE prompt with instruction, criteria, n, duration
    L->>R: describe(frames, prompt)
    R->>R: JPEG encode at 768 px, or tile into one montage if frame_format=montage
    R->>V: POST chat/completions with text plus base64 image_url parts
    V-->>R: message.content (or reasoning_content), usage
    R-->>L: VlmResult(text, latency_s, model, usage, reasoning)
    L->>L: parse_verdict() OBSERVED, COHERENT, VERDICT, CONFIDENCE
    L->>L: combine() VLM decides, geometric fail halves the score, PASS at 0.5 or more
    L->>I: add(SessionRecord)
    I-->>L: one row inserted
    L-->>S: SessionRecord
    S->>S: print verdict, score, raw VLM text, geometric signals
    opt wandb flag
        S->>W: log_session(the same record) then finish()
    end
    Note over L,I: Never called anywhere: AgentLLM, SqliteIndex.search() and where(), iter_frames(), Detector.track(), the deadline
```

A module-by-module inventory, coloured by status, is in `docs/architecture_detailed.mmd`.
