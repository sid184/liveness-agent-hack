# Lessons

- **Verify hosted model availability with a real call, not the catalogue.** `nvidia/cosmos-reason2-8b`
  appears in `/v1/models` on integrate.api.nvidia.com but every chat call returns 404 "Function not
  found for account". NVIDIA staff confirmed on the developer forum (7 Feb 2026) that the hosted API
  was deliberately disabled. A `/v1/models` listing proves nothing; send a two-token request.
- **Model ids in planning docs go stale fast.** The context file named `cosmos-reason1-7b`, deprecated
  24 Mar 2026. Keep model ids in `.env`, never in code or prose.
- **Check which account a key belongs to, not just that it works.** The W&B key authenticated fine
  but its only entity is the employer team; the first test run landed there. For any "personal
  only" project, query the provider's whoami endpoint before the first write.
- **Hosted VLMs differ in image count per call.** NVIDIA's Llama 3.2 Vision endpoint rejects more
  than one image ("At most 1 image(s) may be provided"). Keep a montage fallback in the reasoner
  and make frame format a config value, not an assumption.
- **Big hosted models can hang, not fail.** Llama 3.2 90B Vision sat for over three minutes with no
  error. Always set a client timeout and `max_retries=1` on demo-path calls.
- **A rented GPU bills while idle.** The RunPod pod cost $0.59/hr from the moment it was deployed,
  serving or not. Stop it (`runpodctl pod stop <id>`, REST `/pods/<id>/stop`, or the console) the
  moment a session pauses, and confirm `desiredStatus: EXITED`. Only the volume bills while stopped.
- **Put installs on the persistent volume, not the container disk.** On RunPod the container disk is
  wiped on stop; `/workspace` (network volume) survives. The vLLM install (8GB, about 15 min over
  NFS) and the weights both live there now, so a restart is about 4 min.
- **`python -m venv --system-site-packages` makes pip skip packages the base image already has.**
  The first venv ended up without vllm because the image's copy satisfied the requirement. Use a
  clean venv when the point is to persist the install.
- **Foreground processes in the RunPod web terminal die with the tab.** Run the server under
  `nohup ... &` and tail a log file; SSH in for anything long.
- **vLLM's `--reasoning-parser qwen3` can put the whole reply in `reasoning_content`.** When the
  model answers without think tags, `content` is empty. The client falls back to the reasoning
  field so callers always get text.
