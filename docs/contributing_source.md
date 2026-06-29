# Contributing to quant_lib

## Table of Contents

- [Development Setup](#development-setup)
- [Code Style & Linting](#code-style--linting)
- [Testing](#testing)
- [CI / GitHub Actions](#ci--github-actions)
- [Adding a New Experiment](#adding-a-new-experiment)
- [Adding a New Strategy](#adding-a-new-strategy)
- [Making a Release](#making-a-release)
- [Pull Request Checklist](#pull-request-checklist)

---

## Development Setup

### Prerequisites

- Python ≥ 3.10
- `pip` ≥ 22 (or `uv` for faster installs)
- Git

### Clone & Install

```bash
git clone https://github.com/Hansiongs/hans-backtest.git
cd hans-backtest

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate        # Windows

# Install with dev extras
pip install -e ".[dev]"
pip install mypy mutmut         # type-checker + mutation testing
```

### Quick Sanity Check

```bash
export QUANT_LIB_HMAC_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
make lint        # ruff + mypy
make test-fast   # fast-test subset
```

---

## Code Style & Linting

### Ruff (linter + formatter)

All code (library + tests) is checked with `ruff`. Run:

```bash
make lint        # full: ruff + mypy
make lint-ruff   # ruff only
make lint-mypy   # mypy only
make format     # auto-format via ruff format
```

Rules:
- Line length 88 (ruff default)
- Single quotes for short strings, double quotes for docstrings
- Imports grouped: stdlib → third-party → `quant_lib` (one blank line between)
- Unused imports / variables are caught by `ruff check` and should be auto-fixed before commit

### mypy (type checker)

The library is fully type-checked (`mypy --strict` level with `--no-strict-optional`).
When adding a new function, annotate all parameters and the return type.

Common patterns:
- `dict[str, list[float]]` instead of bare `dict`
- `Optional[str]` or `str | None` (Python 3.10+ union syntax)
- `from __future__ import annotations` at the top of every module to defer annotation evaluation

### Naming Conventions

- `_` prefix for private / internal-only functions and modules (e.g. `_helper.py`)
- No `_` prefix for public API functions (e.g. `tools/stats.py`)
- Class names: `PascalCase`
- Module names: `snake_case`

---

## Testing

### Test Framework

- **pytest** as the test runner (see `pyproject.toml` for configuration)
- **Hypothesis** for property-based tests (invariant checks)
- **pytest-benchmark** for performance regression detection

### Running Tests

```bash
make test           # all tests serially
make test-fast      # skip @pytest.mark.slow
make test-parallel  # -n auto (all CPU cores)
make test-cov       # + branch coverage (fail_under=70)
make test-bench     # performance benchmarks
```

### Writing Tests

- **File naming**: test files go in `tests/` as `test_<module>.py`
- **Test classes**: group related tests by class, each class covers one module or one feature
- **Fixtures**: use `conftest.py` fixtures (`make_session_candidate`, `_MockCache`, etc.) for shared state
- **Test isolation**: each test must create its own `ResearchSession` with its own `tempfile.TemporaryDirectory`.
  Never share mutable state between tests.
- **HMAC secret**: the conftest `_hmac_secret_is_set` fixture sets `QUANT_LIB_HMAC_SECRET` for every test.
  Tests that verify the missing-secret path should use `os.environ` directly (not `monkeypatch.delenv`)
  to avoid racing with the conftest fixture (see `test_migrate_seals.py` for examples).

### Risk Areas

The framework uses **Numba `@njit`** for hot-path trading in `core/_engine.py`. Numba-compiled functions
are opaque to coverage tools. These paths are tested via black-box integration tests (`test_engine.py`,
`test_engine_matrix.py`). If you modify the engine, run **all** engine tests with `--tb=long` to
verify numeric correctness, not just coverage metrics.

---

## CI / GitHub Actions

Three workflows under `.github/workflows/`:

| Workflow | When | What |
|---|---|---|
| `tests.yml` | PR + push to main | `pytest -n auto --cov-branch` on 8 matrix entries (Py 3.10-3.13 × Ubuntu/Windows) |
| `lint.yml` | PR + push to main | ruff (3 OS) + mypy (Py 3.10-3.12) |
| `mutation.yml` | Weekly Mon 03:00 UTC | `mutmut run --max-children 4` on candidate.py + commit.py |

All checks must pass (or be explicitly waived) before merging a PR.

---

## Adding a New Experiment

Experiments are user-defined strategy configurations. To add one:

1. Create a file in `quant_lib/experiments/`:
   ```python
   # quant_lib/experiments/my_strategy.py
   from quant_lib.audit import for_vol_compression
   from quant_lib.experiments import (
       PeriodConfig, UniverseConfig, StrategyConfig,
       from_hypothesis, register,
   )

   _HYP = for_vol_compression(
       name="my_strategy",
       mechanism="...",
       boundary_conditions="...",
       success_criteria="...",
   )

   register(from_hypothesis(
       name="my_strategy",
       hypothesis=_HYP,
       period=PeriodConfig(
           train_start="2020-01-01",
           train_end="2024-12-31",
       ),
       universe=UniverseConfig(
           symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
       ),
   ))
   ```

2. Verify it registers:
   ```bash
   quant_exp list
   ```

3. Run the explore phase:
   ```bash
   quant_exp explore my_strategy
   ```

---

## Adding a New Strategy

A new strategy variant (e.g. a third type beyond vol_compression / pullback_sniper)
requires changes in 4 locations:

1. **`quant_lib/audit/hypothesis.py`** — add a new `StrategyType` enum member
2. **`quant_lib/core/_engine.py`** — add a `STRATEGY_*` int constant (for Numba comparability)
3. **`quant_lib/experiments/base.py`** — add the name-to-int mapping in `STRATEGY_NAME_TO_INT`
4. **`quant_lib/audit/hypothesis.py`** — add a factory function (like `for_pullback_sniper`)

The int values are part of the Numba ABI and **must be stable** (they are used in
`fast_trade_loop`'s positional parameter list).

---

## Making a Release

### Pre-release Checklist

- [ ] `make lint` — 0 ruff + 0 mypy errors
- [ ] `make test-cov` — all tests pass, coverage ≥ 70%
- [ ] `make mutate` — no regression in mutation score from last release
- [ ] Version bumped in `pyproject.toml`, `quant_lib/__init__.py`, and `CITATION.cff`
- [ ] `CHANGELOG.md` updated with all notable changes
- [ ] `README.md` badges reflect current test count
- [ ] PyPI build + upload (`python -m build && twine upload dist/*`)

### Release Steps

1. Create a release branch:
   ```bash
   git checkout -b release/v0.x.x
   ```

2. Update version in source files:
   - `pyproject.toml` `version = "0.x.x"`
   - `quant_lib/__init__.py` `__version__ = "0.x.x"`
   - `CITATION.cff` `version: 0.x.x` + update date

3. Commit and merge:
   ```bash
   git add -u
   git commit -m "Release v0.x.x"
   git tag -a v0.x.x -m "v0.x.x"
   git push origin v0.x.x
   ```

4. The `publish.yml` workflow (if configured) builds and uploads to PyPI when a tag is pushed.

---

## Pull Request Checklist

Before opening a PR, verify:

- [ ] `make lint` passes (ruff + mypy)
- [ ] `make test-fast` passes
- [ ] `python -m pytest tests/test_<affected_module>.py -x` passes
- [ ] If engine code changed, `python -m pytest tests/test_engine.py -x` additionally passes
- [ ] New code has tests (unit test for each new function, integration test for each new feature)
- [ ] Public API changes are reflected in `__init__.py` `__all__`
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]`
- [ ] `README.md` is updated if user-facing behavior changed
- [ ] No `TODO-ACTUAL-*` tags or stale version strings
