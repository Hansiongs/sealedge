# Testing Guide

This document describes how to run, write, and extend the
quant_lib test suite.

## Quick start

```bash
# Run the full test suite serially
pytest

# Run in parallel (uses all CPU cores)
make test-parallel
# or:
pytest -n auto --dist=loadfile

# Run a specific test file
pytest tests/test_engine.py

# Run a specific test by name
pytest -k test_engine_returns_sl_pct_at_index_8

# Run only fast tests (skip @pytest.mark.slow)
make test-fast
# or:
pytest -m "not slow"

# Run with coverage (must reach 70% to pass CI)
make test-cov
```

## Test file layout

The test suite is organized by source module.  One source module
typically has one (or more) corresponding `test_*.py` file:

```
tests/
├── conftest.py            # Shared fixtures, constants, helpers
├── test_audit.py          # audit/ module
├── test_best_params.py    # research/best_params.py
├── test_cache.py          # research/cache.py
├── test_candidate.py      # research/candidate.py
├── test_chaos.py          # Chaos / fault-injection tests (NEW)
├── test_cli.py            # CLI integration (subprocess)
├── test_cli_list.py       # list_cmd unit tests (NEW)
├── test_cli_main.py       # main.py structure (NEW)
├── test_main_module.py    # `python -m quant_lib` smoke tests (NEW)
├── test_cli_output.py     # OutputManager unit tests (NEW)
├── test_commit.py         # research/commit.py
├── test_config.py         # core/_config.py
├── test_config_validation.py  # experiments/config validation
├── test_conftest.py       # Smoke tests for conftest fixtures (NEW)
├── test_data.py           # core/_data.py
├── test_e2e_happy_path.py # E2E integration
├── test_engine.py         # core/_engine.py (merged from _coverage.py)
├── test_engine_matrix.py  # Parametrized engine matrix
├── test_experiments.py    # quant_lib/experiments/ registry
├── test_features.py       # core/_features.py
├── test_invariants.py     # Property-based tests (NEW)
├── test_metrics.py        # core/_metrics.py
├── test_perf.py           # Performance benchmarks
├── test_perf_regression.py  # CI performance gate (NEW)
├── test_portfolio.py      # core/_portfolio.py
├── test_psr_ess.py        # PSR + ESS
├── test_pullback_sniper.py  # pullback_sniper strategy
├── test_public_api.py     # quant_lib/__init__.py (NEW)
├── test_python_api.py     # Public Python API
├── test_regression_bug3_sl_pct.py   # Bug #3 regression
├── test_regression_hygiene.py         # Sprint 3 hygiene regressions
├── test_regression_marks_seal_seed.py # Bug #4-8 regressions
├── test_reporting.py      # research/reporting.py
├── test_reproducibility.py  # Determinism tests
├── test_research_exceptions.py  # Exception hierarchy
├── test_risk_allocation.py  # risk allocation
├── test_session.py        # research/session.py
├── test_spa.py            # core/_spa.py
├── test_statistics.py     # PSR/FDR/p-value labelling
├── test_status_render.py  # status_cmd rendering (NEW)
├── test_tools.py          # Aggregated tools/ tests (NEW)
├── test_tools_data.py     # tools/data.py (NEW)
├── test_tools_features.py # tools/features.py (NEW)
├── test_tools_portfolio.py  # tools/portfolio.py (NEW)
├── test_tools_stats.py    # tools/stats.py (NEW)
├── test_typing.py         # py.typed marker (NEW)
├── test_universe_filter.py  # Candidate.run_universe filter
├── test_universe_tools.py # tools/universe.py
├── test_wfa.py            # core/_wfa.py
```

## Writing new tests

### Style conventions

- **Behavioural tests preferred.** Tests should exercise the
  public API and assert on observable side effects.  Avoid
  `inspect.getsource()` / AST-based string matching on production
  code — these break under routine refactors.

- **One test class per behaviour.** Group tests that exercise the
  same code path into a single class (e.g., `TestFastTradeLoop`).

- **Use the conftest fixtures.** Shared fixtures in `conftest.py`
  cover the common cases (mock cache, holdout data, session
  builders, etc.).  Re-use them instead of duplicating setup.

- **Avoid `_set_stage` direct calls** in production-code tests.
  Use the public `cand.run_universe()` / `cand.run_edge_testing()`
  / `cand.run_narrowing()` / `cand.mark_ready()` methods.  Reserve
  `_set_stage` for the dedicated state-machine test class.

### Mocking `DataCache`

The `tests.conftest._MockCache` class is the standard way to mock
the Binance data layer.  It provides signal-rich deterministic
data with optional per-symbol overrides:

```python
from tests.conftest import _MockCache

mock = _MockCache()
session.cache = mock
```

For per-symbol custom data, pass `data_lookup={"BTCUSDT": df}`.

### Mocking git / time

For tests that need a specific git commit hash or timestamp, use
`unittest.mock.patch` on `quant_lib.cli._output.get_git_commit`.

## Property-based tests

The framework uses `hypothesis` for property-based testing
(`tests/test_invariants.py`).  These discover edge cases hand-
written tests miss.  Each test asserts an invariant that must
hold for *any* valid input within a domain.

To add a new property test:

```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=0, max_value=100), min_size=1))
def test_my_invariant(values):
    assert my_function(values) >= 0
```

Set `max_examples` to keep CI fast (default in our conftest: 20).

## Chaos / fault-injection tests

`tests/test_chaos.py` exercises failure modes the framework must
survive gracefully: corrupt JSON, NaN/inf inputs, missing files,
concurrent access, etc.  Add new chaos tests whenever you discover
a failure mode in production.

## Performance benchmarks

Two test files track engine performance:

- `tests/test_perf.py` — Ad-hoc benchmarks (manual runs)
- `tests/test_perf_regression.py` — CI gate (regression detection)

To track performance over time:

```bash
# Save baseline (run once)
pytest --benchmark-only --benchmark-save=baseline tests/test_perf.py

# Compare against baseline (run in CI)
pytest --benchmark-only --benchmark-compare=baseline tests/test_perf_regression.py
```

## Mutation testing (optional)

For the highest-value test-coverage signal, run mutation testing
on the core research layer:

```bash
# One-time: install mutmut
pip install mutmut

# Run mutation testing (slow; run manually, not in CI by default)
MUTMUT_RUNNING_IN_BATCH=1 mutmut run --max-children 4
mutmut results
```

The mutmut config is in `pyproject.toml` under `[tool.mutmut]`.

## xdist (parallel execution)

The suite is configured for `pytest-xdist`.  Use:

```bash
pytest -n auto --dist=loadfile
```

The `--dist=loadfile` strategy groups all tests in the same file
into a single worker (avoids cross-file import overhead).

The conftest autouse fixture `_redirect_holdout_paths_to_process_dir`
ensures xdist workers do not race on the shared on-disk holdout
seal file.

## Test isolation

The framework guarantees test isolation.  The autouse fixtures:

- `_isolate_holdout_seal_files` — removes shared seal files
  before/after each test.
- `_redirect_holdout_paths_to_process_dir` — redirects the
  `data_cache/holdout_seals/` path to a per-process temp dir.

If you add a new test that requires shared state, document why
and add the necessary cleanup in the test itself.

## Coverage

The project enforces `fail_under = 70` (see `pyproject.toml`).
Below 70% overall coverage, CI fails.  Coverage is reported via
`make test-cov`.

Branch coverage is tracked via the `--cov-branch` flag (added to
the `test-cov` Makefile target).  Both line and branch coverage
are reported in the terminal output and HTML report.

## CI workflow

GitHub Actions runs the full suite on:
- Python 3.10, 3.11, 3.12
- Ubuntu latest
- With parallel execution (`-n auto --dist=loadfile`)
- With coverage enforcement

See `.github/workflows/tests.yml` for the full config.

## Common pitfalls

- **Forgetting `tempfile.TemporaryDirectory()`** — always use a
  temp dir for tests that touch the filesystem.  Don't write to
  `data_cache/` directly.

- **State leakage between tests** — if a test creates a session
  or candidate, ensure the conftest's `_isolate_holdout_seal_files`
  fixture runs (it's `autouse=True`, so usually automatic).

- **`pytest-randomly` ordering** — the suite is verified to pass
  with random order.  If you add a test that requires fixed order,
  flag it with a comment explaining why.

- **Numba `@njit` code coverage** — `coverage.py` cannot trace
  Numba-compiled code.  The `core/_engine.py` line coverage will
  always be low (~7%).  Test the engine via the public API
  (`fast_trade_loop(*args)`) to cover the actual behaviour.
  Overall line coverage is ~76%, with branch coverage tracked.`
