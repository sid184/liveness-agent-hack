.PHONY: setup smoke-vlm smoke-yolo smoke-pipeline lint fmt test

setup:            ## install Python deps; warn if ffmpeg missing
	uv sync --group dev
	@command -v ffmpeg >/dev/null || echo "ffmpeg not found: brew install ffmpeg"

smoke-vlm:        ## one image -> vision reasoner    (ARGS="--video clip.mp4")
	uv run python -m scripts.smoke_vlm $(ARGS)

smoke-yolo:       ## one image -> YOLO               (ARGS="--image x.jpg")
	uv run python -m scripts.smoke_yolo $(ARGS)

smoke-pipeline:   ## one clip -> whole pipeline      (ARGS="--video clip.mp4 --challenge head_turn")
	uv run python -m scripts.smoke_pipeline $(ARGS)

lint:
	uv run ruff check . && uv run ruff format --check .

fmt:
	uv run ruff format . && uv run ruff check --fix .

test:
	uv run pytest -q
