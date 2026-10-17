#!/usr/bin/env bash
# Serve Cosmos-Reason2 with vLLM on a rented GPU, OpenAI-compatible on port 8000.
#
# Run this ON THE GPU BOX (RunPod / Lambda), not on the laptop. Needs HF_TOKEN in the environment
# and the model licence accepted on huggingface.co. First start downloads ~17GB of weights (8B).
#
#   MODEL=nvidia/Cosmos-Reason2-8B bash scripts/serve_cosmos.sh      # needs >= 32GB GPU (NVIDIA's figure)
#   MODEL=nvidia/Cosmos-Reason2-2B bash scripts/serve_cosmos.sh      # fits 24GB
#
# Verified 10 Oct 2026 on a RunPod A40 (48GB), vLLM 0.31: ~4 min cold start with weights on the
# volume, 3-10 s per 8-frame request. Run it under nohup so a closed browser tab can't kill it:
#   nohup bash scripts/serve_cosmos.sh > /workspace/vllm.log 2>&1 &   # then: tail -f /workspace/vllm.log
# The first run installs vLLM into /workspace/venv (about 15 min over the network volume); later
# runs find it there and start straight away.
#
# Then on the laptop, in .env:
#   VLM_BASE_URL=https://<pod-id>-8000.proxy.runpod.net/v1   (or http://<ip>:8000/v1)
#   VLM_MODEL=nvidia/Cosmos-Reason2-8B
#   VLM_API_KEY=anything                                     (vLLM ignores it unless --api-key is set)
#   VLM_FRAME_FORMAT=images
# and run:  make smoke-vlm ARGS="--video clip.mp4 --frames 8"
set -euo pipefail

MODEL="${MODEL:-nvidia/Cosmos-Reason2-8B}"
MAX_LEN="${MAX_LEN:-16384}"     # NVIDIA recommends 8192-16384; lower it if you hit OOM
PORT="${PORT:-8000}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set; the model is gated on Hugging Face" >&2
  exit 1
fi

# A venv on the network volume survives pod stops; the container disk does not.
VENV="${VENV:-/workspace/venv}"
if [ ! -x "$VENV/bin/vllm" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q "vllm>=0.11.0"
fi

# vLLM compiles kernels at startup and shells out to ninja; make the venv's tools visible.
export PATH="$VENV/bin:$PATH"
"$VENV/bin/python" -c "import ninja" 2>/dev/null || "$VENV/bin/pip" install -q ninja

exec "$VENV/bin/vllm" serve "$MODEL" \
  --max-model-len "$MAX_LEN" \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image": 16, "video": 1}' \
  --host 0.0.0.0 \
  --port "$PORT"
