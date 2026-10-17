.PHONY: setup check smoke-vlm smoke-yolo smoke-pipeline eval ui ui-build ui-dev lint fmt test

setup:            ## install Python deps; warn if ffmpeg missing
	uv sync --group dev
	@command -v ffmpeg >/dev/null || echo "ffmpeg not found: brew install ffmpeg"

check:            ## is every sponsor component reachable? one line each (ARGS="--write")
	uv run python -m scripts.check_stack $(ARGS)

smoke-vlm:        ## one image -> vision reasoner    (ARGS="--video clip.mp4")
	uv run python -m scripts.smoke_vlm $(ARGS)

smoke-yolo:       ## one image -> YOLO               (ARGS="--image x.jpg")
	uv run python -m scripts.smoke_yolo $(ARGS)

smoke-pipeline:   ## one clip -> whole pipeline      (ARGS="--video clip.mp4 --label live")
	uv run python -m scripts.smoke_pipeline $(ARGS)

eval:             ## every labelled clip x every prompt version -> W&B  (ARGS="--from-vss")
	uv run python -m scripts.eval_clips $(ARGS)

ui:               ## verifier overlay at http://localhost:8765 (open in Chrome)
	uv run uvicorn ui.server:app --port 8765

ui-build:         ## rebuild the React pages in ui/web into ui/static (committed; needs Node)
	cd ui/web && npm ci && npm run build

ui-dev:           ## hot-reload the React pages at :5173 against a running `make ui`
	cd ui/web && npm run dev

lint:
	uv run ruff check . && uv run ruff format --check .

fmt:
	uv run ruff format . && uv run ruff check --fix .

test:
	uv run pytest -q
