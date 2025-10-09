
.PHONY: check format test test-cov test-fast test-debug clean docs

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

test-cov:
	uv run pytest tests/ --cov=wpt_tools --cov-report=html

test-fast:
	uv run pytest tests/ -x --tb=short

test-debug:
	uv run pytest tests/ --pdb

docs:
	cd docs && uv run make html

clean:
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .ruff_cache/
	rm -rf docs/build/