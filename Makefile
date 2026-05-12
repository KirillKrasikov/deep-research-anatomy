TARGET_DIRS := app

format:
	uv run ruff format $(TARGET_DIRS)
	uv run ruff check --fix $(TARGET_DIRS)

lint:
	uv run ruff format --check $(TARGET_DIRS)
	uv run ruff check $(TARGET_DIRS)
	uv run mypy --strict $(TARGET_DIRS)
