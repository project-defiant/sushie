.PHONY: test test-only lint type-check clean help

help:
	@echo "Available targets:"
	@echo "  test     - Run unit tests with coverage"
	@echo "  test-only - Run specific test with pattern (e.g., 'make test-only TEST=tests/test_cli.py::test_version')"
	@echo "  lint     - Lint code with ruff and ty"
	@echo "  type-check - Run type checking with mypy"
	@echo "  clean    - Clean up cache and build artifacts"

test:
	uv run pytest --cov sushie --cov-report term-missing --cov-report html --verbose

test-only:
	uv run pytest $(TEST) --verbose

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check .

type-check:
	uv run mypy src/

clean:
	rm -rf .pytest_cache .coverage htmlcov __pycache__ src/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
