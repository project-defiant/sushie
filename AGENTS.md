# Repository Guidelines

## Project Structure

```
sushie/
├── src/sushie/      # Main source code
├── tests/           # Unit tests
├── docs/            # Documentation (Sphinx)
├── data/            # Example data files
└── .github/         # CI workflows
```

## Build & Test Commands

- `uv run pytest` — Run tests with coverage (`--cov sushie --cov-report term-missing --verbose`)
- `uv run pytest -k <pattern>` — Run tests matching pattern (name/expression)
- `uv run pytest tests/test_foo.py::test_bar` — Run specific test function
- `uv run pytest --collect-only` — List all available tests
- `make test-only TEST=<pattern>` — Run specific tests via Makefile
- `uv run ruff check .` — Lint Python code
- `uv run ruff format .` — Format code (Black-compatible)
- `make lint` — Lint with ruff and typer
- `uv run mypy src/` — Type check
- `make type-check` — Type check with mypy
- `uv run ruff check . --fix` — Auto-fix linting issues

Test configuration: `pytest` section in `pyproject.toml`.

## Coding Style

- **Python 3.10–3.11** (via `requires-python = ">=3.10,<3.12"`)
- **Format**: Black-compatible (via `ruff format`)
- **Imports**: isort ordering (standard → third-party → first-party `sushie`)
- **Lint**: Ruff with Google-style docstrings (convention = "google")
- **Types**: Use JAX `Array` and `ArrayLike` from `jax.typing`; type hints recommended
- **Naming**: snake_case for variables/functions; Google-style docstrings with args/returns documented
- **Error handling**: Use `ValueError` for invalid arguments; `assert` for internal checks
- **Logging**: Use `logging` module or `loguru`; prefer `logging.getLogger("sushie")`
- **Quotes**: Single quotes (Ruff Q000); double quotes for docstrings

## Testing

- **Framework**: pytest with coverage
- **Parametrization**: Use `@pytest.mark.parametrize` for multiple input combinations
- **JAX randomness**: Use `jax.random.PRNGKey(seed)` with `split()` for reproducibility
- **Arrays**: Use `jnp` (JAX NumPy) for all array operations
- **Test files**: `tests/test_*.py` with `seed` parameter (default 0)

## JAX-Specific Guidelines

- **X64 precision**: Enable at module startup: `from jax.config import config; config.update("jax_enable_x64", True)`
- **Randomness**: Use `jax.random.PRNGKey(seed)` and `jax.random.split(key, n)` for functional RNG
- **Transformations**: Avoid side effects in functions using `jit`, `grad`, `vmap`
- **Array operations**: Use `jnp.einsum` for tensor contractions; avoid Python loops

## Git Workflow

- **Commit messages**: Imperative mood with scope prefix (`chore:`, `build:`, `docs:`, `refactor:`)
- **Pre-commit hooks**: `trailing-whitespace`, `check-added-large-files`, `check-ast/json/xml/yaml`, `debug-statements`, `end-of-file-fixer`, `requirements-txt-fixer`, `mixed-line-ending`, `isort`, `black`, `flake8`, `mypy`
- **Run before commit**: `ruff check`, `ruff format`, `mypy`, `pytest`

## Project Dependencies

- **Core**: jax, jaxlib, numpy, pandas, scipy
- **Specialized**: cyvcf2, pandas-plink, glimix_core, typer, loguru
- **Dev**: pytest, pytest-cov, ruff, ty
- Add dependencies: `uv add <package>`

## CLI & Entry Point

- Main command: `sushie` (via `sushie:main` in pyproject.toml)
- Use Typer for CLI with type hints; options stored in `OptionStore`

## Troubleshooting

- Test failures: `uv sync` to update dependencies
- Build errors: `uv pip install --no-build-isolation -e .`
- Use `uv run <command>` for all development commands

## Additional Resources

- **Documentation**: https://mancusolab.github.io/sushie/
- **Source**: https://github.com/mancusolab/sushie
- **Issues**: https://github.com/mancusolab/sushie/issues

For detailed installation, see `README.md`.

## Data Formats

- **Genotype**: PLINK (.bed/.bim/.fam), VCF (.vcf), BGEN (.bgen)
- **Phenotype**: TSV/TSV.GZ (subject ID + phenotype value)
- **Covariates**: TSV/TSV.GZ (subject ID + covariates)
- **Summary statistics**: TSV/TSV.GZ or Parquet (.parquet) - use `--parquet` flag
- **LD matrix**: TSV/TSV.GZ (correlation matrix)
- **Prior weights**: TSV/TSV.GZ (SNP ID + prior probability)
