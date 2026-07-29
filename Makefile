.PHONY: help dev lint test smoke docker-smoke clean

MAINTAINED_SOURCES := \
	sushie/anndata_output.py \
	sushie/models.py \
	sushie/outputs.py \
	sushie/pipeline_cli.py \
	sushie/preparation.py \
	sushie/stats.py
MAINTAINED_TESTS := \
	tests/test_anndata_output.py \
	tests/test_outputs.py \
	tests/test_pipeline_cli.py \
	tests/test_preparation.py \
	tests/test_stats.py

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

dev: ## Install locked development dependencies
	@uv sync --frozen --extra testing

lint: ## Run formatting and static checks
	@uv run ruff check $(MAINTAINED_SOURCES) $(MAINTAINED_TESTS)
	@uv run ruff format --check $(MAINTAINED_SOURCES) $(MAINTAINED_TESTS)
	@uv run ty check $(MAINTAINED_SOURCES)
test: ## Run the Python test suite
	@uv run pytest -q

smoke: ## Verify the installed command is discoverable
	@uv run sushie --help >/dev/null

docker-smoke: ## Build and run the CLI container smoke test
	@docker build -t sushie:smoke .
	@docker run --rm sushie:smoke --help >/dev/null

clean: ## Remove local Python build and test artifacts
	@rm -rf .venv .pytest_cache .ruff_cache build dist *.egg-info
