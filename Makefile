.PHONY: setup lint format typecheck test test-integration eval eval-live estimate report clean

setup:
	uv sync --frozen --all-extras --all-groups && uv run pre-commit install

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format . && uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest tests/unit

test-integration:
	uv run pytest tests/integration

eval:
	uv run paribus eval --mode replay

eval-live:
	PARIBUS_ENV=live uv run --extra keychain paribus eval --mode record

estimate:
	uv run paribus estimate

report:
	uv run --extra report paribus report

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage
