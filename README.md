# liveness-agent

Moment-of-action liveness for video calls: a physical challenge no real-time deepfake can pass,
judged by an agent that reasons about the physical world. Built for the VAST Builders Challenge,
London, October 2026.

- `HACKATHON_CONTEXT.md` is the brief, plan and current status. Read it first.
- `CLAUDE.md` is the architecture, conventions and demo script.

```
make setup            # uv + deps (brew install ffmpeg separately)
cp .env.example .env  # then fill in keys
make test             # offline tests
make smoke-vlm        # one image through the vision reasoner
make smoke-yolo       # one image through YOLO
make smoke-pipeline   # one clip end to end
```
