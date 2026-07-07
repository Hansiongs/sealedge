# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added (Hansen-literal SPA, claim #3 Blocker A fix)

- **Hansen (2005) Eq.6-8 SPA implemented** (`quant_lib/core/_spa.py`,
  `quant_lib/core/_metrics.py`, `quant_lib/research/candidate.py`,
  `quant_lib/research/results.py`, `quant_lib/__init__.py`,
  `quant_lib/core/_wfa.py`): Politis–Romano (1994) stationary block
  bootstrap over per-trial IS loss-differentials `d_k = -r_net_k`,
  Hansen (2005) Eq.7 recenter with nuisance-parameter discarding
  `Ā_k_trunc = Ā_k * 1{Ā_k ≥ 0}`, and Eq.8 cross-strategy maximum
  statistic `T_null_max = max_k T_acc^k_b` — the multiple-testing
  correction (the missing piece from White's Reality Check that the
  legacy circular-permutation path lacked). Numpy-only on `pnl_array`s
  so the existing SPA spy `2*n_iters` invariant holds on BOTH paths.
  Opt-in via `recenter_policy="hansen_literal"` + `trial_r_nets=...`
  + `return_statistics=True`; legacy callers see byte-identical
  3-tuples.

- **Three paper-accepted caveats** (disclosed in
  `docs/methodology.md` §6, not silently "fixed"): honest-power may
  be a negative finding (max-of-K at K~10³-10⁴ may price realistic
  drift out); KS<0.25 is an empirical finite-sample claim not a
  theorem (Hansen N(0,1) under H0 is asymptotic, and the recenter
  injects O(1/B) bias at finite B); the spy `2*n_iters` invariant is
  gated to the LEGACY path (Hansen's pure-numpy regime preserves it
  by construction rather than re-asserting it through `simulate_*`
  calls). A spy sibling test (`tests/test_spa_validation.py`) guards
  this.

- **6 new `TestHansenCalibration` tests** (`tests/test_spa.py`):
  `test_hansen_p_value_uniform_empirically_under_true_null`,
  `test_hansen_power_curve_honest_drift`,
  `test_hansen_p_value_floor_one_over_n_iters_plus_one`,
  `test_hansen_p_value_ceiling_capped_at_one`,
  `test_hansen_recenter_is_bootstrap_distribution_mean`,
  `test_hansen_max_stat_gates_data_snooping`. Each defends a
  load-bearing element of the Hansen (2005) Eq.6-8 specification.

- **Spy sibling test** (`tests/test_spa_validation.py`):
  `test_hansen_path_emits_no_extra_simulate_calls` confirms the
  Hansen path emits zero additional `simulate_*` calls beyond the
  legacy `2*n_iters` baseline, asserting the canonical spy invariant
  holds on both legacy and Hansen paths.

- **`spa_naive_p_value` field** added to `ExploreResult`
  (`quant_lib/research/results.py`) and threaded through
  `run_explore` (`quant_lib/__init__.py`). `spa_p_value` now carries
  the Hansen-corrected p when WFA `trial_r_nets` are available
  (NaN-safe fallback to legacy); the legacy circular-permutation p
  is preserved in `spa_naive_p_value` for transparency.

- **`all_trial_r_nets` field** on `Candidate` and `trial_r_nets` key
  per fold dict (collected in `core/_wfa.py:WalkForwardObjective`,
  stored as `.tolist()` so the WFA reproducibility tests'
  `params_a == params_b` dict equality stays unambiguous).

- **3 new `edge_metrics` keys**: `spa_naive_p_value`,
  `spa_joint_k_trials`, `hansen_fallback` (NaN-safe lens into whether
  the Hansen block exercised the loss-differential bootstrap or
  degraded to legacy).

- **NaN-safe fallback**: callers passing `trial_r_nets=None` /
  empty / `std(d_k) <= 0` / `observed N < 2` degrade to
  `p_hansen = p_naive` with `stats["fallback"] = True` +
  `stats["fallback_reason"]`. Every legacy `trial_r_nets=None`
  caller sees byte-identical 3-tuples.

- **`STATIC["spa_hansen_block_length_override"]`** knob: default 0 →
  use Politis–Romano `p = max(1, round(n_k**(1/3)))` per trial; >0
  forces a fixed expected block length. Disclosed methodology §6.

- **`_stationary_block_bootstrap_resample` primitive**
  (`quant_lib/core/_metrics.py`): Politis–Romano (1994) stationary
  block bootstrap helper (`L ~ Geometric(1/p)`, expected length
  `= p`, circular wrap); sibling to (not a refactor of) the
  existing `run_bootstrap` / `run_trade_bootstrap`. 5-test
  parametric guard in `tests/test_metrics_bootstrap.py`
  (incl. empirical-mean check `mean(L) ≈ p` over 10k draws).

- **Cost-parity fix** in `tests/test_spa.py::_make_observed_from_
  simulator`: 4 diverging positional literals (positions 14/15/17/18)
  replaced with `DEFAULTS[...]` (`weekend_liquidity_penalty`,
  `stress_test_multiplier`, `trend_aligned_risk_mult`,
  `trend_counter_risk_mult`), so simulated observed PnL matches
  the production cost model. Apples-to-apples calibration for both
  legacy and Hansen paths.

### Fixed (Quick-start notebook update)

- `notebooks/01_quick_start.{py,ipynb}` cell-6 / cell-8 were routing
  the post-Sprint-3 `ExploreResult` to the pre-Sprint-3 surface
  (`result.stage`, `result.eligible_symbols`, and the prior
  per-symbol `result.edge_metrics.items()` nested shape). Cell-6
  reworked to print `result.experiment` /
  `result.narrowed_symbols` / both `spa_p_value` and
  `spa_naive_p_value`. Cell-8 reworked to iterate the new flat
  top-level `edge_metrics` keys. Regenerated
  `01_quick_start.ipynb` from its canonical `.py` source via
  `python notebooks/01_quick_start.py`.

### Added (DX & Testing)

- **`quant_exp init` subcommand** (`cli/init_cmd.py`): Scaffolds a new
  experiment file in ``quant_lib/experiments/<name>.py`` from a
  template (vol_compression or pullback_sniper, configurable).
  Also generates a ``.env`` file with a random ``QUANT_LIB_HMAC_SECRET``.
  The experiment is immediately registered and visible via
  ``quant_exp list``. Usage: ``quant_exp init my_strategy --symbols
  BTCUSDT,ETHUSDT``.
  - Supports ``--strategy``, ``--symbols``, ``--mechanism``,
    ``--boundary-conditions``, ``--success-criteria``,
    ``--train-start``, ``--train-end``, ``--holdout-months``,
    and ``--generate-env``/``--no-generate-env``.
  - Validates experiment names (Python identifier rules).
  - `.env` files are gitignored (added to `.gitignore`).
- **CI badges in README.md**: Added GitHub Actions ``status`` badges
  for the ``tests.yml`` and ``lint.yml`` workflows, so the project
  README shows the current CI health at a glance.
- **`STATIC["psr_weight_floor"]`** (default ``0.001``): New
  configuration constant. In the weighted ``prob_sharpe_ratio`` path,
  a diagnostic warning is emitted when any trade weight falls below
  this floor after normalisation, indicating near-degenerate
  concentration that degrades the Kish ESS. The check is non-blocking
  (the existing ESS < 2 step handles hard rejection).

### Changed

- **`Makefile` ``install-dev`` target**: Consolidated to a single
  ``pip install -e ".[dev]" mypy mutmut`` line (was two separate pip
  calls).
- **`Candidate.mark_ready()`**: Added ``min_train_months``
  enforcement (defence-in-depth). Previously the training-minimum
  check only existed in ``commit_to_holdout()`` (commit.py), which
  meant callers that called ``mark_ready()`` directly (bypassing the
  full commit pipeline) could mark a candidate ready with an
  insufficient training period. The check duplicates the logic in
  ``commit_to_holdout``, raising ``NotReadyForCommit`` with an
  actionable message when the training period is too short. The
  duplicate is intentional: the two guards cover different
  entry points and both must pass before a commit can proceed.
- **`quant_lib/core/_portfolio.py`**: Extracted ``_build_events`` and
  ``_init_circuit_breakers`` from the 410-line
  ``simulate_full_portfolio`` monolith (refactored in the previous
  work session). These helpers are pure-data transformations with
  no mutable state.

### Added (Contributing & Packaging)

- **`CONTRIBUTING.md` (new)**: Comprehensive contributor guide covering
  development setup, code style (ruff + mypy), naming conventions,
  testing guidelines, risk areas (Numba, conftest), CI workflows
  explained, experiment/strategy addition walkthrough, release
  checklist, and PR checklist. ``README.md`` now points to this file.
- **`requirements.lock` (new)**: Pinned transitive dependencies
  generated by ``pip-compile`` (from ``pip-tools``) against
  ``pyproject.toml`` with the ``[dev]`` extra. Ensures paper-grade
  reproducibility: ``pip install -r requirements.lock`` installs
  the exact same versions used in CI. All dependencies are pinned
  at specific versions (``pandas==3.0.4``, ``numpy==2.4.6``, etc.).
  Lockfile should be regenerated before each release with
  ``pip-compile --extra=dev --output-file=requirements.lock pyproject.toml``.
- **`.github/workflows/publish.yml` (new)**: PyPI publishing
  workflow. Triggered on GitHub Release creation or manually
  (workflow_dispatch). Uses OIDC trusted publishing (no API keys).
  Two environments:
  - ``pypi`` (release creation): Builds sdist + wheel, uploads to
    `pypi.org/p/quant_lib` via
    ``pypa/gh-action-pypi-publish@release/v1`` with ``id-token: write``.
  - ``testpypi`` (manual dispatch): Uploads to TestPyPI for
    pre-release validation.
  Requires PyPI trusted publisher configuration at the package name
  (``quant_lib``) before first use.
- **`quant_lib/core/_portfolio.py`**: Extracted ``_build_events`` and
  ``_init_circuit_breakers`` helper functions from the 410-line
  ``simulate_full_portfolio`` monolith. Both helpers are pure-data
  transformations with no mutable state, making the event loop's
  entry point shorter and the trade-event pipeline independently
  testable. The public API (``simulate_full_portfolio`` function
  signature) is unchanged; this is a refactoring-only change.
- **`mutation_baseline.txt` (new)**: Documents the process for
  capturing and tracking the mutation-testing score over time.
  The first baseline will be generated by the weekly CI run; manual
  capture on Linux/WSL via ``make mutate && mutmut stats``.
- **`.github/workflows/mutation.yml`**: Added ``Capture mutation
  score`` step that saves ``mutmut stats`` output to an artifact
  alongside the mutant list, so weekly runs produce a trendable
  metric.

### Fixed (Test Pollution)

- **Test pollution from `test_regression_b0_3_warnings.py`**:
  The test ``test_no_quant_lib_module_adds_ignore_filter`` was
  deleting every ``quant_lib.*`` module from ``sys.modules`` and
  re-importing them, with no restoration. The re-import created
  **new class objects** (e.g. ``InvalidStageTransition``,
  ``Hypothesis``, ``ResearchSession``), but tests that ran
  earlier in the suite had already captured references to the
  OLD class objects via ``from quant_lib.X import Y``.

  When the next test in the same suite did
  ``pytest.raises(InvalidStageTransition)``, the check was
  against the OLD class while the production code (now using
  the re-imported module) raised the NEW class. ``OLD != NEW``
  → ``pytest.raises`` failed → test reported as failing even
  though the actual code was correct.

  This produced **highly non-deterministic test pollution**:
  50-90 tests failing per full run, with different tests
  failing each time depending on the order pytest picked.

  **Fix**: Save ``sys.modules`` snapshot before the experiment,
  restore in a ``finally`` block. The new test
  ``test_sys_modules_unchanged_after_reimport_test`` is a
  regression test that verifies all ``quant_lib.*`` modules
  retain their original class identity after the reimport
  experiment. Without the fix, this test fails; with the fix,
  it passes deterministically.

  **Result**: 2 consecutive full-suite runs return
  ``1148 passed, 1 skipped, 0 failed`` (was: ``~1000 passed,
  50-90 failed, 1 skipped`` with non-deterministic patterns).

### Added (CI / DX)

- **`.github/workflows/lint.yml` (new)**: Comprehensive lint
  pipeline. Three jobs:
  - `quick` (Ubuntu, Python 3.12, ~30s): Fast `ruff check` gate
    for PR-opened feedback.
  - `ruff` (matrix Ubuntu/Mac/Windows, Python 3.12): Full
    `ruff check` with `--output-format=github` (PR annotations)
    + `ruff format --check`. Catches path-separator and
    line-ending issues that a Linux-only CI misses.
  - `mypy` (matrix Python 3.10/3.11/3.12): Type-check with
    `--ignore-missing-imports --no-strict-optional` to match
    the developer Makefile.
- **`.github/workflows/mutation.yml` (new)**: Scheduled weekly
  mutation test (Mondays 03:00 UTC) + manual trigger via
  workflow_dispatch. Runs `mutmut run --max-children 4` on the
  F16-scope (`candidate.py` + `commit.py` per `pyproject.toml`).
  Artifacts (mutants/, .mutmut-cache) are uploaded with 14-day
  retention. Catches mutation-score drift before release.
- **`.github/workflows/tests.yml` (updated)**: Existing test
  workflow extended:
  - Added Python 3.13 to the version matrix.
  - Added Windows runner (catches path-separator / fs bugs
    that Ubuntu-only CI misses; the codebase has Windows-aware
    code paths in `conftest.py`, `cli/status_cmd.py`,
    `cli/migrate_seals.py`).
  - `OFFLINE=1` set so network-marked tests are skipped (they
    depend on external crypto APIs and are flaky in CI).
  - Codecov upload only on the canonical (Ubuntu, 3.12) entry
    to avoid duplicate uploads; Windows coverage.xml is
    uploaded as a downloadable artifact for cross-platform diff.
  - Bumped action versions: `actions/checkout@v4`,
    `actions/setup-python@v5`, `actions/upload-artifact@v4`,
    `codecov/codecov-action@v4`.
- **README.md "Continuous Integration" section**: Documents
  the three workflows, their triggers, matrix, and how to run
  the same checks locally via `make lint` / `make test-cov`
  / `make mutate`. Project Structure tree now includes the
  `.github/workflows/` directory.

### Added (Quick Wins)

- **`quant_lib.audit.hypothesis.StrategyType` (IntEnum)**: New
  type-safe identifier for the strategy variant a hypothesis uses.
  Replaces the raw ``STRATEGY_VOL_COMPRESSION = 0`` /
  ``STRATEGY_PULLBACK_SNIPER = 1`` int constants as the recommended
  way to pass ``strategy_type=`` to ``Hypothesis(...)``. Benefits:
  - **Readability**: ``StrategyType.PULLBACK_SNIPER`` vs ``1``.
  - **Type checking**: mypy catches accidental swaps between the
    two strategies.
  - **Self-documentation**: the int values are tied to the engine
    contract via the enum members, so a rename surfaces all call
    sites.
  Backwards compatibility: the legacy int constants are still
  exported (typed as ``int``) and ``Hypothesis`` accepts both forms.
  Internal storage is still ``int`` for JSON-serializable round-trips
  and Numba hot-path compatibility. See `quant_lib/audit/hypothesis.py`
  for the full docstring.
- **Test count badge updated** to 1161 in `README.md` (was 1134, 27
  new tests added in this development cycle).
- **`tests/conftest.py` `_PROCESS_SEAL_DIR` cleanup**: New
  session-scoped autouse fixture removes the per-process seal
  directory (``tempdir/hqs_seals_<pid>``) at end of the test session.
  The `HQS_KEEP_SEAL_DIR=1` env var disables cleanup for
  post-mortem debugging. Each pytest worker (serial or xdist) cleans
  only its own PID-based subdir, so the cleanup is safe under
  parallel test runs.

### Added (Critical Engineering Fixes)

#### Configurable seal directory
- **`quant_lib/research/session.py::ResearchSession`**: New
  `seal_dir` constructor parameter (str, optional) controls where
  HMAC holdout seal files are persisted. Default value
  ``<cache_dir>/holdout_seals`` matches the previous hardcoded path
  for backwards compatibility, but the path can now be overridden
  per-session for non-default cache layouts. This fixes the long-standing
  smell where ``ResearchSession`` wrote to a fixed ``data_cache/holdout_seals``
  path regardless of the caller's `cache_dir`, breaking for users running
  from a different working directory or with a non-default cache layout.
- **`QUANT_LIB_SEAL_DIR` environment variable**: When set, both
  `ResearchSession` (as fallback for `seal_dir`) and `quant_exp status`
  read the seal directory from this env var. This is the supported
  way for tooling (CI scripts, cron jobs, `make reproduce`) to
  redirect the CLI to a non-default seal location.
- **`quant_exp migrate-seals` subcommand** (`quant_lib/cli/migrate_seals.py`):
  Re-signs holdout seal files in place with the current
  `QUANT_LIB_HMAC_SECRET`. Idempotent (already-valid seals are
  skipped), atomic writes (temp file + rename), creates `.bak`
  backups for safe rollback. Supports `--dry-run` and `--seal-dir`.
  Use this when:
  1. Rotating `QUANT_LIB_HMAC_SECRET` and the old seals still exist.
  2. Upgrading from `quant_lib` v0.2.x or earlier (pre-B0.1) where
     seals were not HMAC-signed at all. Without migration, the next
     commit after upgrade will fail with `SealVerificationFailed`.
  3. Recovering from a previous migration with the wrong secret.
  See module docstring for safety notes.

### Changed

- **`ResearchSession` default seal directory**: Now derived from
  `cache_dir` instead of being hardcoded. The on-disk layout
  (``<seal_dir>/holdout_<start>_<end>.json``) is unchanged for
  the default case, so existing data and tests continue to work
  without migration.
- **`quant_exp status`**: Reads seal directory from `QUANT_LIB_SEAL_DIR`
  if set, falling back to `./data_cache/holdout_seals`. The previous
  hardcoded `_SEAL_DIR = Path("data_cache/holdout_seals")` is gone.
- **`tests/conftest.py`**: Replaced the brittle `monkeypatch` of
  `os.path.join` / `os.makedirs` inside `quant_lib.research.session`
  with a clean `monkeypatch.setenv("QUANT_LIB_SEAL_DIR", _PROCESS_SEAL_DIR)`
  redirect. This works because the new env-var fallback in
  `ResearchSession.__init__` honours `QUANT_LIB_SEAL_DIR` directly.
  Removes the technical-debt smell flagged in the v0.3.0 review.
- **`Makefile` `reproduce` target**: Now accepts an optional
  ``EXP=experiment_name`` env var. Default is still
  `vol_compression_v1`, but the target now validates that the
  experiment is actually registered (via `quant_exp list` lookup)
  before running, so a typo produces a clear error instead of an
  opaque ``KeyError`` deep in the call chain.
  Usage: ``make reproduce EXP=my_strategy``.
- **`Makefile` `docs` target**: Previously printed "MkDocs not yet
  configured." (unhelpful). Now lists the actual markdown
  documentation locations and explains how to enable MkDocs
  when desired.

### Fixed (Critical)

- **Holdout seal path no longer hardcoded**: `ResearchSession`
  used to write seals to a hardcoded `data_cache/holdout_seals`
  path regardless of the caller's `cache_dir`. The hardcoded
  path is gone; the directory is now derived from `cache_dir`
  by default and overridable via `seal_dir=` or the
  `QUANT_LIB_SEAL_DIR` env var. The conftest monkeypatch
  workaround is also gone.
- **No more silent fallback to pre-rotation secret**: When the
  HMAC secret is rotated and old seals exist, the old seals
  become unverifiable. Previously the only option was to delete
  the seals and start over. Now `quant_exp migrate-seals` rewrites
  the seals in place with the new secret (with `.bak` backups).

### Added (Test Framework Hardening)

#### Test Coverage Improvements
- **`tests/test_main_module.py` (new)**: Smoke tests for `python -m
  quant_lib` entry point. Covers `quant_lib/cli/main.py` version
  callback and module-level help/version/list behavior.
- **`tests/test_cli_output.py` extended**: Added `TestRenderPureFunctions`
  class with 13 unit tests for `_render_table`, `_render_kv`,
  `_render_chart`, `_render_section` internal helpers. Also added
  4 direct `OutputManager` end-to-end tests covering directory creation,
  `save_metrics`, `save_config`, and `save_html_report` without the
  git mock fixture.
- **`tests/test_cli.py` extended**: Added `TestCLIInternalHelpers`
  class verifying `_looks_like_absolute` helper in both
  `cli/explore.py` and `cli/commit_cmd.py`.
- **`tests/test_public_api.py` extended**: Added
  `TestRunExploreEndToEnd` and `TestRunCommitEndToEnd` classes with
  mocked data layer, validating that the public API's
  `run_explore()` and `run_commit()` return properly structured
  results with all documented fields.

### Changed

- **`Makefile`**: `test-cov` target now includes `--cov-branch` flag
  to collect branch coverage. Previously branch coverage was
  configured in `pyproject.toml` (`branch = true`) but not actually
  collected during test runs.

### Performance

- **Line coverage**: 76.42% (was ~70%, above `fail_under=70` gate).
- **Branch coverage**: Now visible and tracked (1,120 branches,
  previously reported as 0%).

### Removed

- **`tests/test_regression_hygiene.py::TestS31UnusedImportsRemoved`**:
  Removed AST-based test for unused imports (concern delegated to
  `ruff check`, not runtime test).

### Build / Tooling

- **Makefile: added `mutate`, `mutate-stats`, `mutate-show` targets**
  for `mutmut` mutation testing. Run via `make mutate`.

### New Tests (Sprint 4)

- **`tests/test_cli.py::TestCommitAbortPath`**: New test verifying that
  the commit command exits cleanly (code 0) with "Aborted" message when
  the user declines the confirmation prompt.
- **`tests/test_regression_b0_5_indonesian_residue.py` (new)**: Regression
  guard for B0.5 fix (Indonesian residue removed from user-facing labels).
  Verifies no "Kasus" text remains in `reporting.py` or `candidate.py`.
- **`tests/test_integration_real_data.py` (new)**: Integration smoke test
  using real OHLCV data (BTCUSDT 1h, Jan 2024). Validates schema, OHLC
  invariants, and feature computation on realistic data.
- **`tests/fixtures/btcusdt_1h_2024_jan.csv` (new, ~3KB)**: Realistic
  OHLCV fixture for integration test.
- **`tests/fixtures/README.md` (new)**: Documents fixture regeneration.
- **`tools/download_fixture.py` (new)**: Helper script to fetch real
  Binance data for the integration fixture (requires internet).

## [0.3.0] - TBD

### Changed (Architecture)

- **Experiments directory refactor**: User experiment files moved from
  the project root `experiments/` directory into the
  `quant_lib/experiments/` package. This follows standard Python
  packaging conventions: `pip install -e .` now makes the experiments
  available immediately, no `sys.modules` injection is required, and
  the `built_in.discover_experiments()` path resolution no longer
  depends on a hardcoded `parent.parent.parent` traversal.
  - `quant_lib/experiments/built_in.py`: rewrote `_project_root()` as
    `_experiments_dir()` returning the package's own directory.
  - `quant_lib/experiments/__init__.py`: docstring updated to reflect
    new location.
  - README + `docs/testing.md`: paths updated.
  - If a legacy top-level `experiments/` directory still exists at the
    project root, a one-time deprecation warning is logged on
    auto-discovery (does not break functionality, but those files will
    not be auto-discovered until moved).
  - This is a **breaking change** for users with custom
    `experiments/*.py` files at the project root. Move them into
    `quant_lib/experiments/` to restore auto-discovery.

### Fixed

- **B0.2 — `sl_pct <= 0` guard** (`quant_lib/core/_portfolio.py`):
  `simulate_full_portfolio` now raises `ValueError("sl_pct must be > 0,
  got <value>")` at trade entry instead of producing a confusing
  `ZeroDivisionError` deep inside the loop. Regression test
  `tests/test_regression_b0_2_sl_pct.py` covers `sl_pct=0` and
  `sl_pct<0` (3 cases).
- **B0.3 — Global `warnings.filterwarnings("ignore")` removed**
  (`quant_lib/core/_config.py`): importing `quant_lib` (or any
  submodule) no longer mutates the host application's `warnings.filters`
  list. The previous behavior silently hid Numba/Optuna deprecation
  warnings from the user, which violated framework transparency
  principles. Regression test
  `tests/test_regression_b0_3_warnings.py` snapshots and compares
  `warnings.filters` before/after import (3 cases).
- **B0.5 — Indonesian residue in user-facing labels**:
  - `quant_lib/research/reporting.py` edge classification taxonomy:
    "Kasus 1/2/3" → "CONCENTRATED / RANDOM / BROAD_WEAK".
  - `quant_lib/research/candidate.py:428` docstring: "Kasus 3 default"
    → "broad-weak default".

### Removed

- **Dead code: `_resolve_experiment_or_exit`**
  (`quant_lib/cli/main.py`): defined but never called. Each subcommand
  (`explore`, `commit`) implements its own experiment lookup inline.
  Removed along with its test class
  `tests/test_cli_main.py::TestResolveExperimentOrExit`.
- **`hqs_execution.log` no longer created at project root**
  (`quant_lib/core/_logging.py`): the module previously installed a
  `FileHandler("hqs_execution.log", mode="w")` at import time, which
  created/overwrote a 0-byte file at the project root every time
  `quant_lib` was imported. The file handler is now removed: the
  module provides only the `console` and `log` singletons. File
  logging is now exclusively the CLI's responsibility, via
  `quant_lib.utils.logging.setup_logging(verbose, log_file)`, which
  writes to a configurable path (no file written unless explicitly
  requested).
- **Unused `import warnings`**: removed from
  `quant_lib/core/_config.py` as part of the B0.3 fix.

### Added

- **Visualization module** (`quant_lib/research/plotting.py`):
  matplotlib + seaborn styled charts. Six functions total in this
  release (two from Phase 1, four from Phase 2):
  - `plot_equity_curve(daily_equity, initial_capital, output_path=None)`
    — cumulative equity curve with green/red shading above/below
    initial capital.
  - `plot_drawdown_underwater(daily_equity, output_path=None)` — drawdown
    from peak, inverted y-axis.
  - `plot_trade_distribution(r_vals, output_path=None, bins=40)` —
    R-multiple histogram + KDE + mean/median markers, with
    positive/negative count annotation. KDE auto-suppressed when
    `n < 5` or `std == 0`. NaN/Inf values are dropped.
  - `plot_spa_null(random_equities, observed_equity, p_value, output_path=None)`
    — null distribution histogram with observed marker; marker
    color is green for `p < 0.05` (significant) and red otherwise.
    NaN p-value and NaN observed equity are handled gracefully.
  - `plot_per_symbol_equity(per_symbol_equity, output_path=None)` —
    multi-line cumulative equity, one line per symbol. Reveals
    concentration vs breadth. Auto-switches to 2-column legend when
    more than 12 symbols.
  - `plot_wfa_progression(fold_params, output_path=None)` — best_value
    per fold per symbol. Convergence visualization. NaN/Inf `best_value`
    entries are dropped silently; folds with missing `best_value` key
    are skipped.
  Each function accepts dict / Series / DataFrame / array inputs
  (per its documented contract), returns either a file path or a
  base64 data URI, handles empty input gracefully, and closes its
  figure to prevent memory leaks. The `Agg` backend is set at module
  import time, so the module works in headless / CI environments.
  - Optional dependency: `pip install quant_lib[viz]` installs
    matplotlib and seaborn. Without them, importing `plotting.py`
    raises a clear `ImportError` with an install hint. When the
    `quant_exp` CLI is invoked with `--report` but matplotlib is
    missing, a yellow warning is printed and the report is still
    generated (without chart sections).
  - 53 tests in `tests/test_plotting.py` cover input variants,
    output format, edge cases (empty / NaN / Inf / all-same / few
    trades), file saving, and figure cleanup.

- **HTML report generation** (`quant_lib/cli/_output.py::OutputManager.save_html_report`):
  Builds a self-contained single-file HTML report from a list of
  structured `(heading, content)` sections. Charts embedded as
  base64 data URIs (no external file references — fully portable).
  Content dispatch supports: key-value tables, multi-row tables,
  inline charts, raw HTML, and plain strings. All user-supplied
  values are HTML-escaped to prevent injection. Empty subtitles are
  omitted (no stray empty `<p>` tags). 16 new tests in
  `tests/test_cli_output.py` cover table rendering, chart
  embedding, HTML escaping, and ordering.

- **HTML report builders** (`quant_lib/cli/_report.py`):
  `build_explore_report(candidate, session, chart_provider)` and
  `build_commit_report(result, session, chart_provider)` produce
  the structured section list for each CLI subcommand. The
  `chart_provider` callable maps chart names to base64 data URIs
  (or `None` to skip the section). 24 new tests in
  `tests/test_cli_report.py` verify all expected sections,
  chart-conditional inclusion, and edge cases (no trades, no
  by-symbol stats, empty reject breakdown, broken holdout seal).

- **CLI flags** (`quant_exp explore` and `quant_exp commit`):
  - `--report <PATH>`: generate a self-contained HTML report.
    Relative paths are anchored to the run directory (e.g.,
    `--report my.html` → `results/<ts>_<exp>_<mode>/my.html`).
    Absolute paths are used as-is. The report contains all
    text/metrics plus inline charts (unless `--no-plots`).
  - `--no-plots`: skip chart generation. Reports are still
    generated, just without chart sections. Useful for headless
    CI / fast iteration.
  - Both flags are optional; the default CLI behavior (text only
    + `metrics.json` + `config.yaml`) is unchanged for users who
    don't pass them.
  - 6 new tests in `tests/test_cli.py::TestReportFlag` verify
    the flags are registered and accepted.

### Notes

- **Bumped version to 0.3.0** in `pyproject.toml` and
  `quant_lib/__init__.py`. This is a minor version bump (no
  breaking API changes for users of `run_explore` / `run_commit`).
  - The only breaking change is the experiments directory
    refactor (see Changed section above). Users with custom
    `experiments/*.py` files at the project root must move them
    to `quant_lib/experiments/`.
- `pyproject.toml` already added the `[project.optional-dependencies] viz`
  extra in Phase 1; Phase 2 documents the chart output in detail.

### Fixed (B0.4 — risk-weight carry-over, separate PR)

- **B0.4 — PF risk weights now carry from WFA to holdout**: previously,
  `quant_lib/research/candidate.py:run_edge_testing` discarded the
  per-fold summary returned by
  `apply_pf_weighted_risk_allocation` and explicitly set
  `self.risk_weights = {}`. As a result, every holdout trade in
  `commit_to_holdout` was built with
  `candidate.risk_weights.get(sym, 0.01)`, which always returned
  0.01 (the silent fallback). The holdout therefore used a flat
  1%-per-symbol allocation regardless of the per-fold PF-weighted
  allocation produced by the WFA — making the holdout PSR not
  representative of the WFA edge.
  - `quant_lib/core/_risk_allocation.py`: new helper
    `extract_final_fold_weights(risk_summary, eligible_symbols,
    default_weight)` extracts the LAST fold's per-symbol weights as
    a complete mapping. The last fold uses the most prior data and
    is the canonical carry-over to the holdout.
  - `quant_lib/research/candidate.py`: `run_edge_testing` now
    captures the per-fold summary and builds `self.risk_weights`
    from the last fold (with the default for any eligible symbol
    not present in the last fold).
  - `quant_lib/research/commit.py`: the per-trade risk_weight
    construction now reads `candidate.risk_weights[sym]` directly.
    If a symbol is missing (which should not happen for a
    properly-run candidate), the path emits a warning listing the
    missing symbols and uses the default — replacing the previous
     silent `0.01` fallback that masked the bug.
  - 17 new tests in `tests/test_regression_b0_4_risk_weights_carry.py`
    cover the helper, the orchestrator contract, and the
    commit-path carry-over behavior (including a structural
    test that verifies PF-differentiated weights when one
    symbol has all winners and another has all losers).

### Fixed (B0.1 — HMAC-SHA256 seal signature, separate PR)

- **B0.1 — Holdout seals are now cryptographically signed with
  HMAC-SHA256**. Before this fix, the seal JSON file persisted
  only a plain SHA256 of the holdout data and a few metadata
  fields. There was **no integrity check on the seal itself**:
  anyone who could read the seal JSON and the underlying data
  could edit `broken_at` back to `null` to "un-break" a used
  holdout, or construct a forged seal with a known `data_hash`
  for new data. The `data_hash` only proved the *content* of
  the holdout data at sealing time; it did not prove the
  *seal metadata* (`sealed_at`, `broken_at`) was authentic.
  Fix:
  - `quant_lib/audit/holdout.py`: the seal JSON now includes an
    `HMAC-SHA256 signature` field over a canonical JSON
    serialization of the seal state. The secret is loaded from
    the `QUANT_LIB_HMAC_SECRET` environment variable
    (minimum 32 chars; cached on first read; rotation requires
    regenerating all existing seals).
  - New module-level functions: `get_hmac_secret()`,
    `compute_seal_signature(state)`, `verify_seal_signature(state)`,
    and `_reset_hmac_secret_cache()` (for tests).
  - `HoldoutSeal` dataclass: new `signature: Optional[str]` field.
    `to_dict()` includes it only when set; `_save_seal()` adds it
    atomically with the rest of the state.
  - `verify()` re-computes the signature from the on-disk JSON
    (using `hmac.compare_digest` for constant-time comparison)
    **before** any business-logic checks. Missing signature or
    signature mismatch → seal is tampered.
  - `seal()` and `commit_break()` both raise `RuntimeError` if
    the secret is not configured — there is **no insecure
    fallback**. Missing or too-short secret is a configuration
    error, not something to silently work around.
  - `tests/conftest.py`: autouse fixture sets
    `QUANT_LIB_HMAC_SECRET` to a 64-char placeholder for every
    test, with `_reset_hmac_secret_cache()` so per-test secret
    changes are visible.
  - 28 new tests in `tests/test_regression_b0_1_hmac_seal.py`
    cover: secret retrieval (4 cases), signature computation
    (6 cases), signature verification including all tamper
    paths (5 cases), `seal()` requiring the secret (2 cases),
    seal file contents (3 cases), tampering detection (4 cases),
    secret rotation invalidating old seals (1 case), and the
    end-to-end "construct a forged seal" attack scenario (1
    case).
- **Backwards-incompatible change**: any seal file created
  before this fix will be detected as tampered (no signature
  field) and rejected. This is intentional — old seals must be
  regenerated. Existing on-disk seal files in
  `data_cache/holdout_seals/` should be removed after upgrading.

## [0.2.6] - 2026-06-27

### Changed (Documentation — Phase 5)

Phase 5 is a documentation/cleanup release. No production code changed
beyond a single expanded docstring on `commit_to_holdout`.

- **Indonesian → English translations** (7 source files):
  - `quant_lib/audit/__init__.py`: "Prinsip 2" → "Principle 2"
  - `quant_lib/audit/holdout.py`: full module docstring translated
    (3 Indonesian bullet points → English equivalents)
  - `quant_lib/audit/hypothesis.py`: full module docstring translated
    (2 lines of Indonesian principle → English)
  - `quant_lib/audit/journal.py`: full module docstring translated
    (4 lines of Indonesian principles → English)
  - `quant_lib/core/_wfa.py`: line 154 inline comment translated
    (`karena df_is sudah dropna` → `because df_is has already been dropna'd`)
  - `quant_lib/core/_config.py`: line 65 assertion error message
    translated (`untuk` → `for`)
  - `quant_lib/core/_features.py`: line 92 inline comment translated
    (`Mengatasi Leakage pada` → `Address leakage in`)
  - Note: `roadmap_paper.md` (internal working doc) is left as-is
    per project owner decision (not public-facing).

- **Placeholder replacements** (3 files, 6 occurrences): all
  `yourname` / `YourName` / `YourFirstName` placeholders replaced
  with `TODO-ACTUAL-USERNAME` / `TODO-ACTUAL-NAME` /
  `TODO-ACTUAL-FIRSTNAME` for easy `grep` discovery and replacement
  by the project owner:
  - `pyproject.toml`: 3 URL occurrences (lines 55-57)
  - `CITATION.cff`: 2 author placeholders + 1 URL (lines 11, 12, 16)
  - `README.md`: 2 URL occurrences (lines 280, 300)
  - `CITATION.cff` also updated to version 0.2.5 and date 2026-06-27.

- **Test artifact removed**: deleted
  `data_cache/holdout_seals/holdout_2025-01-01_2025-06-30.json`
  (contained an all-zeros SHA256 hash from the original CLI bug
  report session that was using `_skip_holdout_load=True` for
  testing). Will be regenerated on next session init if needed.

- **`commit_to_holdout` docstring expanded** (`research/commit.py`):
  Added explicit "Pre-commit guards" section enumerating the 4
  guards in execution order (state, seal, no-peek hash,
  `min_train_months`). Added Parameters / Returns / Raises sections
  for clarity. This makes the contract explicit for callers and
  future maintainers.

### Tests
- No new tests added in this phase.
- All 571 tests still pass (Phase 1-4 work preserved).

## [0.2.5] - 2026-06-27

### Fixed (Test Coverage — Phase 4)

Hardened existing tests to close coverage gaps identified during
the Phase 1-3 review. No production code changed in this phase;
all updates are test-side.

- **G1 `test_spa_zero_iters` assertion** (`tests/test_spa_coverage.py`):
  Added `assert p == 1.0` to verify the boundary case where
  `n_iters=0`. The Davé 2008 SPA correction gives
  `p = (n_exceed + 1) / (n_iters + 1) = (0 + 1) / (0 + 1) = 1.0`.
  Pre-fix only checked `len(null) == 0`, missing the p-value
  boundary verification.

- **G2 `test_degenerate_anchor_returns_nan_p` tightening**
  (`tests/test_spa_coverage.py`): Spec is clear that degenerate
  anchor (span >= 80% of total) returns NaN p-value. Pre-fix test
  was lenient (`assert np.isnan(p) or p == 1.0` with comment
  "implementation may vary"), masking potential implementation
  drift. Post-fix: strict `assert np.isnan(p)`.

- **G3 bracket TP exit verification** (`tests/test_spa_coverage.py`):
  `test_long_bracket_tp_exit` and `test_short_bracket_tp_exit` now
  verify the exit happens AT the TP level (within 5% of TP target
  price) and that `r_net > 0` (profitable trade). Pre-fix only
  checked `exit_idx >= 0`, so any exit path (bailout, SL, TP) would
  pass. New tests construct deterministic data with a pre-entry
  spike (high for long, low for short) to create a TP target above
  the entry price, ensuring the bracket TP path activates (not
  SL or bailout).

- **G4 `test_ess_more_data_higher_psr` robustness**
  (`tests/test_psr_ess.py`): Pre-fix drew two independent samples
  (n=20 and n=500) which made the assertion RNG-dependent -- if
  the 20-sample happened to have high sample SR by chance, the
  test could fail. Post-fix uses a SHARED PREFIX (first 20 of 500
  samples) so the smaller sample is a strict subset of the
  larger one, making the assertion deterministic.

- **G5 hardcoded `stress_mult=2.0` in tests**
  (5 test files, 23 occurrences total): Replaced all hardcoded
  `stress_mult=2.0` with `stress_mult=DEFAULTS["stress_test_multiplier"]`
  in `test_spa_coverage.py`, `test_engine.py`, `test_engine_coverage.py`,
  `test_pullback_sniper.py`, and `test_sprint1_fixes.py`. If
  `DEFAULTS["stress_test_multiplier"]` changes, tests automatically
  use the new value (no manual update needed). Added `DEFAULTS`
  imports where missing.

### Tests
- Modified 5 existing tests for stronger coverage (G1-G5).
- No new tests added in this phase.
- All 571 tests still pass.

## [0.2.4] - 2026-06-27

### Fixed (LOW severity — Phase 3)

- **A2 `label_p_value` context-specific thresholds** (`core/_testing.py`):
  The `context` parameter is now actually used with separate threshold
  tiers. `"mean_r"` (per-symbol, default) keeps existing 5-tier
  thresholds. `"spa"` (portfolio-level) uses ~2× stricter thresholds
  (PROD at p<0.0025, TRADE at p<0.025, WATCH at p<0.075, NO_EDGE
  at p≥0.15). Unknown contexts fall back to `"mean_r"` with a warning
  log. Each tier has its own interpretation text tailored to the
  context (per-symbol vs. portfolio).

- **A3 ESS<2 NaN guard in WFA path** (`core/_wfa.py`): The weighted
  PSR formula in `WalkForwardObjective.__call__` now also falls back
  to neutral `psr=0.5` when `ess < 2.0`, matching the `var_corr <= 0`
  case. The WFA objective function should not bias the search when
  the PSR formula is unreliable (insufficient effective samples).
  This complements the prob_sharpe_ratio guard added in Phase 1.
  Also added `n_is_months <= 0` guards in `_adaptive_trials()`.

- **D1 atomic seal save** (`audit/holdout.py:commit_break`): The
  `commit_break()` method now performs a SINGLE atomic `_save_seal()`
  call instead of two. Pre-fix, the intermediate state (new hash
  written but `broken_at=None`) was a crash window where the seal
  file on disk had inconsistent state. Post-fix: all fields
  (data_hash + broken_at) are set in memory, then one save writes
  them atomically.

- **E3 `ess` field consistency** (`research/commit.py`): `CommitResult.ess`
  is now `n_trades - 1` (matches the PSR variance denominator) instead
  of `n_trades`. The Kish ESS for uniform weights is `n` (sample count),
  but the PSR formula uses `n-1` as the sample-variance denominator.
  For metadata consistency, the field now matches the formula.

- **B1 SPA NaN guard** (`core/_spa.py:portfolio_spa`): Added a defensive
  guard that returns `NaN` p-value when `observed_final_equity` is NaN
  (e.g., from numerical issues in `simulate_full_portfolio`). Pre-fix,
  the comparison `random_equities >= NaN` would always be False, giving
  `n_exceed=0 -> p_value=1/(N+1)` (misleadingly "significant"). Post-fix:
  explicit NaN return with a warning log.

- **C1 defensive guard in WFA adaptive functions** (`core/_wfa.py`):
  `_get_purge_days()` and `_adaptive_trials()` now guard against
  `n_is_months <= 0` (defensive, return safe defaults of 90 days
  purge and 50 trials respectively). These should not be reached in
  normal flow (run_wfa_per_symbol checks `len(df_is) < 1000` and
  returns -9999 long before), but guard against division-by-zero
  and silly inputs.

- **E4 `min_train_months` enforcement** (`research/commit.py:commit_to_holdout`):
  The commit function now enforces the hypothesis's `min_train_months`
  setting. If the actual training period is shorter than the minimum
  required, `CommitError` is raised with a clear message. A short
  training period produces unreliable frozen params and inflated
  holdout results, so this guard prevents bypassing the WFA minimum
  by going directly to commit.

- **E5 deep-copy holdout data** (`research/session.py:__init__`): The
  `_holdout_data_for_hash` and `_btc_extended_for_features` are now
  stored as `copy.deepcopy()` of the caller's data instead of
  shallow `dict()` copy. Pre-fix, the DataFrame references were
  shared, so a test that mutates a DataFrame in place (e.g.,
  `df.loc[i, "close"] = x`) would silently change the session's
  data without detection. Post-fix: full isolation from caller
  mutation. The perf cost is acceptable since this only happens
  once at session init.

### Tests
- Added 5 new tests for `label_p_value` context behavior (SPA stricter
  thresholds, context-specific tier transitions, unknown context
  fallback, NaN handling with context).
- Added 2 new tests for `_get_purge_days` defensive guard
  (`n_is_months <= 0`).
- Added 1 new test for `_adaptive_trials` defensive guard
  (`n_is_months <= 0`).
- Added 1 new test for SPA NaN guard (`test_spa_observed_nan_returns_nan_pvalue`).
- Added 1 new test for `commit_break` single save
  (`test_commit_break_single_save`).
- Added 1 new test for `ess` field consistency
  (`test_ess_field_equals_n_trades_minus_one`).
- Added 3 new tests for `min_train_months` enforcement (blocks short
  training, allows long training, respects custom hypothesis min).
- Added 1 new test for deep-copy isolation
  (`test_holdout_data_isolated_from_caller_mutation`).
- Total: 15 new tests, 0 updated, 0 removed.
- Test count: 571 passed, 1 skipped (was 556 in 0.2.3).

## [0.2.3] - 2026-06-27

### Fixed (HIGH severity — Phase 1)
- **PSR kurtosis convention** (`core/_testing.py:57`, `core/_wfa.py:192`):
  Changed `fisher=False` (regular kurtosis) to `fisher=True` (excess
  kurtosis) to match Bailey & Lopez de Prado (2012) PSR formula. The
  previous formula overestimated variance by `+3/4 * SR²` offset,
  understating PSR. With excess kurtosis = 0 for normal data, the
  formula now correctly gives `1 - SR²/4` correction instead of
  `1 + SR²/2`. Impact: PSR values become more confident (closer to
  true statistical confidence) for non-zero SR with near-normal data.

- **Weighted PSR consistency** (`core/_testing.py`): `prob_sharpe_ratio`
  with `trade_weights` now uses weighted mean, weighted variance, and
  weighted SR (matching the formula in `core/_wfa.py:179-200`).
  Previously used unweighted SR with ESS-scaled variance, giving
  different results from the WFA inline path. Single formula across
  all code paths.

- **PeriodConfig holdout convention** (`experiments/base.py`):
  Added `holdout_months: int = 6` parameter to `PeriodConfig`.
  `resolve()` now generates POST-training holdout
  `[train_end + 1d, train_end + holdout_months + 1d]` instead of
  embargo-style last 6 months of training. Enforces the no-peek
  guarantee at the framework level. Existing experiments updated
  with `holdout_months=6` explicit. Removed dead `DEFAULT_HOLDOUT_MONTHS`.

- **PSR asymptotic formula guard** (`core/_testing.py`): Bailey's
  variance correction `(kurt-1)/4 * SR²` can make the variance
  negative for high SR with near-normal data. The function now
  clips the correction to a small positive value (1e-8) rather
  than returning NaN, so callers get a usable PSR. Also added
  defensive guard: extreme SR (>1e6) returns NaN (catches
  float-precision near-constant inputs).

### Fixed (MEDIUM severity — Phase 2)
- **Holdout hash coverage** (`research/session.py:_compute_holdout_data_hash`):
  Hash now includes ALL OHLCV columns (`open`, `high`, `low`, `close`,
  `volume`) instead of just `(time, close)`. Tampering with any OHLCV
  column now invalidates the seal. Strategy decisions can depend on
  any of these columns (SL uses high/low; vol_pct_rank uses volume),
  so the seal must cover all of them.

- **BTC extended history integrity** (`research/session.py:_load_holdout_ohlcv`,
  `research/commit.py:commit_to_holdout`): The BTC extended data
  range (`btc_data_start` to `hold_end`, used for EMA warmup features
  at commit time) is now:
  1. **Pre-loaded at session init** and preserved in
     `session._btc_extended_for_features` (was previously discarded).
  2. **Hashed** as part of the seal (was previously NOT hashed,
     allowing silent tampering of EMA source data).
  3. **Reused at commit time** instead of re-fetched, ensuring
     hash consistency between init and commit.
  Modifying any bar in the BTC extended range now aborts the commit
  with `SealVerificationFailed`. New `_btc_extended` testing parameter
  added to `ResearchSession.__init__` for tests that need to verify
  the no-peek guarantee on the extended range.

### Tests
- Added 9 new tests for `PeriodConfig.holdout_months` (default, custom
  values 1/3/6, validation for 0/negative/non-integer, post-training
  invariant, total length preservation, explicit-override behavior).
- Added 3 new tests for `ResearchSession` period validation boundaries
  (one-day-after accepted, same-day rejected, overlap rejected,
  far-future accepted).
- Added 4 new tests for PSR (weighted PSR matches WFA formula,
  kurtosis excess convention, ESS<2 returns NaN, weighted mode
  uses weighted SR).
- Added 3 new tests for OHLCV tampering detection (high, low, open,
  volume columns).
- Added 5 new tests for BTC extended integrity (storage when
  provided, None when not, hash differs with different extended,
  pre-holdout tampering detected, holdout-window tampering detected).
- Updated 4 existing tests to reflect post-training holdout convention
  and corrected annualize behavior under Bailey's formula.
- Total: 30 new tests, 4 updated, 0 removed.
- Test count: 556 passed, 1 skipped (was 542 in 0.2.1).

## [0.2.1] - 2026-06-26

### Added
- **`core/_risk_allocation.py`** — canonical per-fold PF-weighted risk
  rebalancing module. Exposes `apply_pf_weighted_risk_allocation`
  orchestrator plus the 3 private helpers (`_compute_decay_weighted_pnl_loss`,
  `_compute_clamped_factor`, `_rescale_factors_to_total`).
- **`tests/test_risk_allocation.py`** — 30 unit tests covering the
  orchestrator (cold start, multi-fold sequencing, no-fold_key no-op,
  total preservation, log summary) and the 3 private helpers.
- **`StrategyConfig` wiring to `Candidate`**: `ResearchSession.create_candidate`
  now accepts an optional `strategy` parameter. `Candidate.run_edge_testing`
  reads `self.strategy.pf_*` and feeds them to the new orchestrator.
  `run_explore`/`run_commit`/`quant_exp` auto-wire `exp.strategy`.
- **`@pytest.mark.slow` marker** registered in `pyproject.toml` and tagged
  on the slowest E2E/commit tests (`test_e2e_happy_path.py`,
  `test_commit_coverage.py::TestCommitWithRealTrades`). Use
  `make test-fast` to skip them.

### Changed (refactor)
- **Methodology fix**: `Candidate.run_edge_testing` now implements
  per-fold decay-PF risk rebalancing (matches `docs/methodology.md`).
  Previously used a one-time ATR-inverse weight and missed the meta-allocation
  step entirely.
- **Public API shrinkage**: deleted the legacy `run_full_simulation`
  (656-line monolith in `core/_runner.py`). It had no production callers
  and duplicated logic now in `core/_risk_allocation.py` + `Candidate`.
- **`compute_vol_adjusted_weights`** removed (the only other consumer
  was `run_full_simulation`).

### Removed (dead code)
- `live_optuna.py` and `LiveOptuna`/`LiveOptunaResult` re-exports.
- `HoldoutSet.break_seal` (superseded by `commit_break`).
- `HoldoutSet.load` (no production callers).
- `ConfigManager` class (redundant with `StrategyConfig`).
- `compute_frozen_params_best_last` deprecation shim (superseded by
  `pick_best_params_per_symbol`).
- `OutputManager.write_text` (no callers).
- `ComputeEss.compute_ess` (inlined into `objective_psr_ess`).
- `_wfa_worker` and `_wfa_log_dir` (only used by `run_full_simulation`).

### Changed (cleanups)
- Removed ~20 unused imports across 12 source files.
- Removed dead fixture `sample_daily_hl_matrix` from `tests/conftest.py`.
- Removed duplicate `_MockCache` import in `test_commit_coverage.py`.
- Removed trivial/no-op and docstring-presence tests.
- Replaced brittle `test_infrastructure_keys_only` with property check.
- Replaced 12 string-substring tests in `test_research_exceptions.py`
  with 40 parametrized tests validating the exception hierarchy,
  phase formatting, and immediate-parent catchability.
- Consolidated 3-way duplicate `simulate_trailing_stop_trade` tests
  (canonical home: `test_spa_coverage.py`).
- Consolidated 3-way duplicate `EngineArgs` immutability tests
  (canonical home: `test_engine.py`).
- Consolidated sprint-fix files: kept only the non-redundant tests
  (Bugs #3, #4-8) and removed duplicates.
- Updated stale doc references (`REFACTOR_PLAN.md`, `REVIEW_2026-06-25.md`)
  that no longer exist on disk.
- Added `pytest-cov` to dev extras (was missing).
- Updated `stress_test_multiplier` default to 2.0 (matches 0.2.0 reduction).

### Test count
- 561 → 520 (net -41 tests, with significant quality improvements:
  -18 redundant tests removed, +40 meaningful tests added).

### Post-release scan fixes (0.2.2 patch)
Applied after a thorough re-scan of the framework:

- **CLI `main.py`**: removed redundant `from quant_lib.experiments import built_in`
  (auto-discovery is already triggered by `experiments/__init__.py`).
  Also consolidated console import to use `core._logging.console` (matching all
  other CLI files), eliminating two different Rich `Console()` instances in the
  same process.
- **CLI `status_cmd.py`**: rewrote run-name parser using a regex
  (`_RUN_NAME_RE`) instead of `r.name.split("_", 3)`. The old parser truncated
  experiment names with underscores (e.g., `vol_compression_v1` became just
  `vol` with the rest glued to the mode). New parser handles names with any
  number of underscores correctly.
- **CLI `commit_cmd.py`**: removed unused intermediate variable `p`.
- **Tests**: added 5 new tests in `tests/test_cli.py::TestRunNameParsing` for
  the new parser (underscore in exp name, with/without git suffix, invalid
  format). New total: 530 tests.

### Known issue (not fixed in 0.2.2)
- **Logging conflict**: `quant_lib/core/_logging.py` initializes a file
  handler for `hqs_execution.log` at import time. `utils/logging.py:setup_logging`
  uses `force=True` which clears all handlers, so calling `setup_logging()` from
  `explore.py`/`commit_cmd.py` removes the file log. Pre-existing issue,
  scheduled for 0.2.3.

## [0.2.2] - 2026-06-26

### Fixed
- **Bonferroni `discount_ablations`**: ablations now correctly counted as 0.5
  (was subtract, making adjusted alpha too lenient — off by 0.5 per ablation).
  Restores the documented half-weight discount design from 0.2.0.
- **SPA entry-slip formula**: `simulate_trailing_stop_trade` now mirrors
  `fast_trade_loop`'s exact `1.0 + random_draw*(stress_mult-1.0)` formula.
  Previous code was missing the `1.0 +` prefix, so SPA null distribution
  trades had systematically lower entry slippage than real trades. SPA
  p-values are now valid.
- **SPA `stress_mult` default**: changed from hardcoded 2.5 to
  `DEFAULTS["stress_test_multiplier"]` (2.0). Direct `spa_test()` API
  callers now get consistent cost model with the WFA/commit path.
- **CLI `explore`/`commit`**: now pass `strategy=exp.strategy` to
  `session.create_candidate()`. Per-experiment `StrategyConfig` overrides
  (PF weight, leverage, etc.) now apply in the CLI path. Previously
  silently used `StrategyConfig()` defaults.
- **Journal corruption handling**: silent `pass` on corrupt journal file
  replaced with `log.warning()` + continue. Audit-trail corruption is now
  visible instead of silently losing entries.

### Changed
- **PSR formula consolidation**: removed `objective_psr_ess()` (near-duplicate
  of `prob_sharpe_ratio()` for the unweighted case). `prob_sharpe_ratio()`
  now accepts an optional `trade_weights` parameter, handling both weighted
  and unweighted cases. The WFA path in `core/_wfa.py` keeps its own
  inline PSR computation unchanged.
- **EngineArgs docstring**: corrected `auxiliary_features` description
  from "4 ndarrays" to "5 ndarrays" (self-contradictory note removed).

### Documentation
- Fixed 5 stale `core/_runner.py` references in `docs/methodology.md`
  (file deleted in 0.2.1). Replaced with `core/_risk_allocation.py`
  references to current implementation.
- Bumped version in `pyproject.toml`, `quant_lib/__init__.py`,
  `CITATION.cff` from 0.2.1 to 0.2.2.
- Synced test count badge in `README.md` (561 → 520, matching actual).
- Removed stale "verbatim from core/_runner.py" comment in
  `core/_risk_allocation.py:32`.
- Removed stale "moved verbatim from test_runner_integration.py"
  comment in `tests/test_risk_allocation.py:26`.
- Replaced "Run 3" narrative in `tests/test_sprint1_fixes.py:8`
  with proper version reference.

### Tests
- All engine tests: `stress_mult=2.5` → `2.0` (matches production
  `DEFAULTS["stress_test_multiplier"]`). Files: `test_engine.py`,
  `test_engine_coverage.py`, `test_pullback_sniper.py`,
  `test_spa_coverage.py`, `test_sprint1_fixes.py`.
- `test_audit.py`: updated `test_ablation_discounted` for the new
  half-weight add semantics. Added `test_ablation_not_discounted` and
  `test_ablation_only_discounted` for explicit coverage of both paths.
- `test_psr_ess.py`: migrated from deleted `objective_psr_ess` to
  `prob_sharpe_ratio(trade_weights=...)`. Added new tests for annualize
  flag behavior and weighted/unweighted PSR equivalence.
- `test_config.py`: added `test_stress_test_multiplier_value` guard
  (`assert DEFAULTS["stress_test_multiplier"] == 2.0`) to catch
  future DEFAULTS drift.
- `test_spa_coverage.py`: added `TestEntrySlipFormulaRegression` with
  two new tests verifying the entry_slip formula monotonicity and
  finite r_net across stress_mult values.
- `test_wfa_coverage.py:311`: fixed tautology assertion
  (`assert "rsi_oversold" in p or "rsi_oversold" not in p` is always
  true) to assert actual RSI presence in pullback_sniper fold params.

### Cleanup
- Deleted stale `quant_lib/core/__pycache__/_runner.cpython-314.pyc`
  (source deleted in 0.2.1).
- Deleted stale `tests/__pycache__/test_runner_integration.cpython-314-pytest-9.1.1.pyc`
  (test source deleted in 0.2.1).
- Cleared `.pytest_cache/v/cache/nodeids` (40 stale entries for
  deleted `test_runner_integration.py`).

## [0.2.0] - 2025-06-25

### Added
- **`quant_exp` CLI** with 5 subcommands: `list`, `show`, `explore`,
  `commit`, `status`. Built with Typer. Replaces the old argparse
  CLI.
- **Experiment registry** (`quant_lib/experiments/`) with
  `@register` decorator. Users define experiments as Python files
  in `experiments/` (auto-discovered on import).
- **Per-experiment config** (`ExperimentConfig`, `StrategyConfig`,
  `PeriodConfig`, `UniverseConfig`) — frozen dataclasses with
  `__post_init__` validation.
- **High-level Python API**: `from quant_lib import run_explore,
  run_commit` for notebook/interactive use.
- **`pick_best_params_per_symbol()`** in
  `quant_lib/research/best_params.py` — picks fold with highest
  PSR per symbol (Q1 decision).
- **`OutputManager`** (`quant_lib/cli/_output.py`) — manages
  `results/<timestamp>_<name>/` with `metrics.json`, `config.yaml`.
- **`utils/git.py`** — `get_git_commit()` for traceable artifacts.
- **`utils/logging.py`** — `setup_logging()` for CLI.
- **`LICENSE` (MIT)**, **`CITATION.cff`**, **`CHANGELOG.md`**,
  **`Makefile`**, **`docs/methodology.md`** for paper defense.
- **15 CLI tests**, **10 Python API tests**, **10 reproducibility
  tests**, **30 config validation tests** (test_reproducibility.py,
  test_config_validation.py, test_cli.py, test_python_api.py).

### Changed
- **Best-params selection (Q1)**: replaced "frozen from last fold"
  with "best PSR per symbol across all WFA folds" — consistent
  with live trading workflow. Implemented in
  `pick_best_params_per_symbol()`.
- **`STATIC` cleanup**: reduced from 31 to **11 infrastructure
  keys**. Per-experiment config moved to new `DEFAULTS` dict
  (24 keys) and `StrategyConfig`. `asset_risk_weights`,
  `asset_baseline_trades`, `global_position_limit` removed.
- **`ConfigManager`** now merges `STATIC` + `DEFAULTS` for unified
  config access.
- **Version bump**: 0.1.0 → 0.2.0.
- **`compute_frozen_params_best_last`** is deprecated; emits
  `DeprecationWarning`. Use `pick_best_params_per_symbol` instead.

### Removed
- **`black_testing.py`** (311 lines) — legacy demo script.
- **`white_testing.py`** (867 lines) — legacy demo script.
- **`tests/test_testing_smoke.py`** (9 tests) — depended on legacy
  scripts.

### Fixed
- **Holdout seal integrity** (C-2 fix): now
  cryptographically enforced via SHA256 at session creation,
  verified at commit time. Tampering raises
  `SealVerificationFailed`.
- **Pullback_sniper RSI params** (C-3 fix):
  `compute_frozen_params_best_last` is generic across strategies
  and picks up RSI keys automatically (no more dropped RSI on
  pullback_sniper).
- **Stress multiplier default**: 2.5× → 2.0× (more moderate;
  weekend penalty 2.0× stays separate).
- **Windows console encoding**: subprocess CLI tests now set
  `PYTHONIOENCODING=utf-8` to avoid cp1252 crashes.
- **Test isolation**: `test_list_runs_without_crash` re-discovers
  experiments (other tests' fixtures may clear the registry).
- **NoneType in `simulate_full_portfolio`**: `asset_risk_weights`
  may now be `None` (per-asset CB is skipped in that case).
- **NoneType in `portfolio_spa`**: same fix.

### Deprecated
- `compute_frozen_params_best_last` — use
  `pick_best_params_per_symbol` from `quant_lib.research.best_params`.
  Will be removed in v0.3.0.

## [0.1.0] - 2024-XX-XX

Initial release.
that brought the framework to a paper-defensible state.
