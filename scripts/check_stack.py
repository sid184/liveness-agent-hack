"""Is every sponsor component reachable and doing its job? One line each, OK or FAIL with why.

    uv run python -m scripts.check_stack          # or: make check
    uv run python -m scripts.check_stack --write  # also write and read back one VastDB test row

Nothing here changes the verdict log or the VSS archive, except --write, which inserts one row
with label `check` into our own VastDB table (never vss-collection).
"""

from __future__ import annotations

import argparse
import time
import traceback
from collections.abc import Callable

import httpx

import scripts._paths  # noqa: F401
from config import settings

results: list[tuple[str, bool, str]] = []


def check(name: str) -> Callable:
    def wrap(fn: Callable[[], str]) -> Callable[[], None]:
        def run() -> None:
            t0 = time.perf_counter()
            try:
                detail = fn()
                ok = True
            except Exception as e:
                detail = f"{type(e).__name__}: {str(e)[:220]}"
                if "--trace" in __import__("sys").argv:
                    traceback.print_exc()
                ok = False
            dt = time.perf_counter() - t0
            results.append((name, ok, detail))
            print(f"{'OK  ' if ok else 'FAIL'}  {name:<34} {dt:5.1f}s  {detail}")

        return run

    return wrap


def _need(*names: str) -> None:
    missing = [n for n in names if not getattr(settings, n)]
    if missing:
        raise RuntimeError(f"not set: {', '.join(missing)} (copy the team config to .env.team)")


@check("Team config (.env.team)")
def team_config() -> str:
    _need("gpu_bearer_token", "ingress_url", "s3_endpoint", "access_key", "vastdb_bucket")
    return f"team {settings.team_username or '?'}, bucket {settings.vastdb_bucket}"


@check("NVIDIA Cosmos (vision judge)")
def cosmos() -> str:
    from reason import VisionReasoner

    r = VisionReasoner()
    from capture.frames import synthetic_test_image

    res = r.describe([synthetic_test_image()], "Reply with the single word: OK", max_tokens=16)
    if not res.text.strip():
        raise RuntimeError("empty reply")
    reply = res.text.strip()[:30]
    return f"{res.model} at {r.client.base_url} replied {reply!r} ({res.latency_s:.1f}s)"


@check("NVIDIA Cosmos Embed1 (search)")
def embed1() -> str:
    from index.embed import DIM, embed

    _need("cosmos_embed1_url")
    v = embed("a person turns their head to the left")
    if v is None:
        raise RuntimeError("no embedding returned (see message above)")
    if len(v) != DIM:
        raise RuntimeError(f"dim {len(v)}, expected {DIM}")
    return f"{settings.embed_model}, dim {len(v)}"


@check("YOLO11 pose (local)")
def yolo() -> str:
    from capture.frames import synthetic_test_image
    from perception import Detector

    d = Detector()
    d.detect(synthetic_test_image())
    return f"{settings.yolo_model} loaded, {len(d.names)} classes"


@check("VAST VastDB (verdict log)")
def vastdb() -> str:
    from index.vast import VastIndex

    idx = VastIndex()
    n = len(idx.all())
    return f"{idx.bucket}/{idx.schema}/{idx.table} reachable, {n} sessions"


def vastdb_write() -> None:
    @check("VAST VastDB write + read back")
    def run() -> str:
        from index import SessionRecord
        from index.vast import VastIndex

        idx = VastIndex()
        rec = SessionRecord("check", "stack check", "ERROR", 0.0, "check_stack test row")
        rec.label = "check"
        idx.add(rec)
        back = idx.get(rec.session_id)
        if back is None:
            raise RuntimeError("row written but not read back")
        return f"session {rec.session_id} round-tripped"

    run()


@check("VAST VSS archive (login + list)")
def vss() -> str:
    from eval import vss_archive

    _need("ingress_url", "team_username", "team_password")
    items = vss_archive.explore(scope="mine", limit=100)
    return f"logged in as {settings.team_username}, {len(items)} indexed uploads visible"


@check("VAST VSS search")
def vss_search() -> str:
    from eval import vss_archive

    _need("ingress_url")
    res = vss_archive.search("a person turning their head", top_k=5) or {}
    hits = res.get("chunk_results") or res.get("results") or []
    return f"{len(hits)} hits for 'a person turning their head'"


@check("W&B key (logging)")
def wandb_key() -> str:
    _need("wandb_api_key")
    import wandb

    viewer = wandb.Api(api_key=settings.wandb_api_key, timeout=20).viewer
    entity = settings.wandb_entity or getattr(viewer, "entity", "?")
    user = getattr(viewer, "username", "?")
    return f"key valid ({user}), logging to {entity}/{settings.wandb_project}"


@check("W&B Inference (agent LLM)")
def wandb_inference() -> str:
    from agent.llm import AgentLLM

    _need("agent_api_key")
    llm = AgentLLM()
    t0 = time.perf_counter()
    text = llm.chat([{"role": "user", "content": "Reply with the single word: OK"}], max_tokens=8)
    return f"{llm.model} replied {text.strip()[:20]!r} ({time.perf_counter() - t0:.1f}s)"


@check("W&B Weave (tracing)")
def weave_check() -> str:
    from eval import tracing

    if not tracing.init():
        raise RuntimeError("weave did not start (WANDB_API_KEY set? WEAVE_ENABLED not 0?)")
    entity = f"{settings.wandb_entity}/" if settings.wandb_entity else ""
    return f"tracing to {entity}{settings.wandb_project}"


@check("CoreWeave GPU endpoints reachable")
def coreweave() -> str:
    _need("cosmos3_reason_url")
    headers = {"Authorization": f"Bearer {settings.gpu_bearer_token}"}
    out = []
    for name, url, path in (
        ("cosmos3", settings.cosmos3_reason_url, "/v1/models"),
        ("embed1", settings.cosmos_embed1_url, "/v1/models"),
        ("yolo", settings.yolo_url, "/healthz"),
    ):
        if not url:
            continue
        r = httpx.get(url.rstrip("/").removesuffix("/v1") + path, headers=headers, timeout=10)
        out.append(f"{name} {r.status_code}")
    return ", ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write one test row to VastDB")
    ap.add_argument("--trace", action="store_true", help="print tracebacks")
    args = ap.parse_args()
    print(
        f"VLM {settings.vlm_model} | index {settings.index_backend} | "
        f"prompt {settings.prompt_version} | agent {settings.agent_model}\n"
    )
    for fn in (
        team_config,
        coreweave,
        cosmos,
        embed1,
        yolo,
        vastdb,
        vss,
        vss_search,
        wandb_key,
        wandb_inference,
        weave_check,
    ):
        fn()
    if args.write:
        vastdb_write()
    bad = [n for n, ok, _ in results if not ok]
    print(
        f"\n{len(results) - len(bad)}/{len(results)} OK"
        + (f"; failing: {', '.join(bad)}" if bad else "")
    )


if __name__ == "__main__":
    main()
