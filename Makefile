
.PHONY: check format test

check:
	uv run ruff check wpt_tools tests
	uv run isort --check-only wpt_tools tests
	uv run black --check wpt_tools tests

format:
	uv run ruff check --fix wpt_tools tests
	uv run isort wpt_tools tests
	uv run black wpt_tools tests

test:
	uv run pytest tests/