.PHONY: help install install-dev test test-fast test-parallel test-cov test-bench lint format clean reproduce docs mutate mutate-stats mutate-show

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package (runtime deps only)
	pip install -e .

install-dev:  ## Install package with dev + docs + paper extras
	pip install -e ".[dev,docs,paper]"

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
	mutmut run

mutate-stats:  ## Show mutation score statistics
	mutmut stats

mutate-show ID:  ## Show details of a specific surviving mutant
	mutmut show $(ID)

lint:  ## Run linter (ruff)
	ruff check quant_lib/

format:  ## Auto-format code (ruff)
	ruff format quant_lib/

reproduce:  ## Run full pipeline to reproduce paper results
	@echo "==> Running explore (Phase 0-3)..."
	@python -c "from quant_lib import run_explore; r = run_explore('vol_compression_v1'); print('SPA p-value:', r['spa_p_value']); print('Final equity:', r['final_equity'])"
	@echo ""
	@echo "==> Running commit (Phase 4)..."
	@python -c "from quant_lib import run_commit; r = run_commit('vol_compression_v1'); print('Final equity:', r.final_equity); print('PSR:', r.psr); print('Seal broken:', r.seal_broken)"

docs:  ## Serve documentation locally (MkDocs)
	@echo "MkDocs not yet configured."

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
