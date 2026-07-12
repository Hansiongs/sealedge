.PHONY: help install install-dev test test-fast test-parallel test-cov test-bench test-bench-compare test-bench-save bench-clean lint lint-ruff lint-mypy format clean reproduce reproduce-one reproduce-seal docs-serve docs-build mutate mutate-fast mutate-stats mutate-show mutate-clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package (runtime deps only)
	pip install -e .

install-dev:  ## Install package with dev + docs extras
	pip install -e ".[dev]" mkdocs mkdocs-material mkdocstrings-python mypy mutmut

install-docs:  ## Install docs-only extras (for MkDocs)
	pip install mkdocs mkdocs-material mkdocstrings-python

lint-ruff:  ## Run ruff linter only
	ruff check quant_lib/

lint-mypy:  ## Run mypy type-checker only
	mypy --ignore-missing-imports --no-strict-optional quant_lib/

test:  ## Run all tests serially
	pytest

test-fast:  ## Run fast tests only (skip slow integration tests)
	pytest -m "not slow"

test-parallel:  ## Run tests in parallel using all CPU cores
	pytest -n auto --dist=loadfile

test-cov:  ## Run tests with coverage report (enforces fail_under=70)
	pytest --cov=quant_lib --cov-branch --cov-report=term-missing --cov-report=html
	@echo "HTML report: htmlcov/index.html"

test-bench:  ## Run performance benchmarks (F22)
	pytest --benchmark-only tests/test_perf.py

test-bench-compare:  ## Compare benchmarks against stored baseline
	pytest --benchmark-only --benchmark-compare=2024_01_01 tests/test_perf.py

test-bench-save:  ## Run benchmarks and save as new baseline
	pytest --benchmark-only --benchmark-save=current tests/test_perf.py

bench-clean:  ## Remove stored benchmark history
	rm -rf .benchmarks/

mutate:  ## Run mutation testing on critical modules (candidate.py + commit.py)
	# NOTE: mutmut requires Linux/WSL (no native Windows support).
	# The CI workflow (mutation.yml) runs on Ubuntu weekly.
	# See https://github.com/boxed/mutmut/issues/397
	MUTMUT_RUNNING_IN_BATCH=1 mutmut run --max-children 4

mutate-fast:  ## Run mutation testing with lower parallelism (for local dev)
	# NOTE: mutmut requires Linux/WSL (no native Windows support).
	MUTMUT_RUNNING_IN_BATCH=1 mutmut run --max-children 1

mutate-stats:  ## Show mutation score statistics
	mutmut stats

mutate-show ID:  ## Show details of a specific surviving mutant
	mutmut show $(ID)

mutate-clean:  ## Remove mutmut cache and generated mutants dir
	rm -rf mutants/ .mutmut-cache
	@echo "Cleared mutmut cache."

lint:  ## Run linter (ruff) and type-checker (mypy)
	ruff check quant_lib/
	mypy --ignore-missing-imports --no-strict-optional quant_lib/

format:  ## Auto-format code (ruff)
	ruff format quant_lib/

reproduce:  ## Reproduce paper results (all strategies, n_spa_iters=2000, paper-grade)
	@echo "==> Running scripts/reproduce.py (all 3 strategies, n_spa_iters=2000)..."
	@echo "    Paper-grade: ~53 min measured for all 3 strategies; 1-2 h cold/slow hosts."
	@python scripts/reproduce.py

reproduce-one:  ## Reproduce single experiment (smoke test; same n_spa_iters=2000)
	@echo "==> Running single strategy (e.g., make reproduce-one EXP=vol_compression_v1)..."
	@if [ -z "$(EXP)" ]; then \
		echo "ERROR: specify EXP (e.g., EXP=vol_compression_v1)"; \
		exit 1; \
	fi
	@python scripts/reproduce.py --strategies $(EXP)

reproduce-seal:  ## Claim 1 seal micro-demo (synthetic HMAC; seconds; no market data)
	@echo "==> Running scripts/reproduce_seal_demo.py (synthetic seal lifecycle)..."
	@python scripts/reproduce_seal_demo.py

docs-serve:  ## Serve documentation locally (MkDocs, http://localhost:8000)
	mkdocs serve

docs-build:  ## Build static documentation to site/
	mkdocs build

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache htmlcov/ .coverage .coverage.*
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned build artifacts."

# ════════════════════════════════════════════════════════════════════════
# Notes for Windows users
# ════════════════════════════════════════════════════════════════════════
# The above Makefile is POSIX-compatible. On Windows, you can:
# 1. Use WSL (Windows Subsystem for Linux)
# 2. Use Git Bash (includes make + most unix tools)
# 3. Run commands manually (see individual targets)
#
# Equivalent PowerShell commands:
#   make install        -> pip install -e .
#   make test           -> pytest
#   make test-parallel  -> pytest -n auto --dist=loadfile
#   make test-cov       -> pytest --cov=quant_lib --cov-report=term-missing
#   make lint           -> ruff check quant_lib/
#   make clean          -> Remove-Item -Recurse -Force build,dist,.pytest_cache,htmlcov
