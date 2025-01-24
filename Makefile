# .DEFAULT_GOAL := all
sources = src test

.PHONY: .uv  ## Check that uv is installed
.uv:
	@uv -V || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: install  ## Install the package, dependencies, and pre-commit for local development
install: .uv
	uv sync --frozen --group all --all-extras

.PHONY: rebuild-lockfiles  ## Rebuild lockfiles from scratch, updating all dependencies
rebuild-lockfiles: .uv
	uv lock --upgrade

.PHONY: format  ## Auto-format python source files
format: .uv
	uv run ruff format $(sources)

.PHONY: check-format  ## Auto-format python source files
check-format: .uv
	uv run ruff format --check $(sources)

.PHONY: lint  ## Lint python source files
lint: .uv
	uv run ruff check $(sources)

.PHONY: typing  ## Run the type-checker
typing: .uv
	uv run mypy $(sources)

.PHONY: test  ## Run all tests
test: .uv
	@uv run coverage run -m pytest --durations=10
	@uv run coverage report

.PHONY: benchmark  ## Run all benchmarks
benchmark: .uv
	uv run coverage run -m pytest --durations=10 --benchmark-enable tests/benchmarks

.PHONY: testcov  ## Run tests and generate a coverage report, skipping the type-checker integration tests
testcov: test
	@echo "building coverage html"
	@uv run coverage html

.PHONY: all  ## Run the standard set of checks performed in CI
all: lint format typing typing testcov

.PHONY: clean  ## Clear local caches and build artifacts
clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]'`
	rm -f `find . -type f -name '*~'`
	rm -f `find . -type f -name '.*~'`
	rm -rf .cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -rf build
	rm -rf dist
	rm -rf site
	rm -rf docs/_build
	rm -rf coverage.xml

.PHONY: docs  ## Generate the docs
docs:
	uv run mkdocs serve -a localhost:8001

