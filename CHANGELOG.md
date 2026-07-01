# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed (Sprint 3 — type safety + drift prevention)

This release addresses 7 items from the post-Phase-4 systematic review
focused on type safety, code drift prevention, and infrastructure
hardening. Backward compatibility is preserved on every item.

- **STATIC/DEFAULTS are now TypedDicts** (`quant_lib/core/_config.py`):
  the two config dicts were typed as ``dict[str, Any]``, defeating
  static analysis on all 89+ call sites (``STATIC["fee_taker"]`` was
  ``Any``, not ``float``). Defined ``StaticConfig`` and
  ``DefaultsConfig`` TypedDict schemas and applied them. TypedDict is
  a plain ``dict`` at runtime, so all existing call sites work
  unchanged; only the annotations change. Anti-reintroduction guards
  in ``TestStaticDefaultsTypedDict`` assert schema ↔ literal stay
  in sync.

- **StrategyConfig/DEFAULTS sync guard** (new
  ``TestStrategyConfigStaysInSync``): the two schemas are intentionally
  redundant (StrategyConfig is the user-facing API, DEFAULTS is the
  internal fast-path) and were drifting silently. The new test class
  enforces: every scalar DEFAULTS key has a matching StrategyConfig
  field, every shared key has identical default values, and the
  framework-only exceptions (search_space, default_risk_per_pair,
  etc.) are documented.

- **Mutation baseline + 5pp CI enforcement** (`mutation_baseline.txt`,
  `.github/workflows/mutation.yml`): the mutation workflow previously
  was advisory-only -- no score capture, no enforcement. Added a
  parseable ``BASELINE_SCORE_PCT: TBD`` field to
  `mutation_baseline.txt`. The CI workflow now:
  1. Extracts the current score to ``CURRENT_SCORE_PCT: X.Y`` in
     ``mutation_score.txt``
  2. Parses the baseline score (no-op when ``TBD``)
  3. Fails the run if the drop exceeds the 5pp acceptance threshold
  First weekly run will populate ``BASELINE_SCORE_PCT`` from the
  captured score; subsequent runs enforce automatically.

- **Candidate typed fields** (`quant_lib/research/candidate.py`):
  added type aliases for the previously-untyped ``daily_close_matrix``,
  ``daily_hl_matrix``, ``risk_weights``, ``reject_reasons``,
  ``edge_metrics``, ``frozen_params``, ``fold_params`` fields. Runtime
  is still ``dict``; only static analysis improves. ``report`` field
  changed from ``Optional[object]`` to ``Optional[Any]`` (more precise
  than ``object``, doesn't pretend to know the type).

- **ExploreResult dataclass** (`quant_lib/research/results.py`): the
  return value of ``run_explore`` was a plain ``dict`` while
  ``run_commit`` returned a typed ``CommitResult`` dataclass --
  asymmetric public API. Added ``ExploreResult`` dataclass that
  supports BOTH attribute access (``r.spa_p_value``) for new code
  AND dict-style access (``r["spa_p_value"]``, ``r.keys()``,
  ``r.values()``, ``r.items()``, ``r.get(key, default)``, ``len(r)``,
  ``key in r``, iteration) for backward compatibility with existing
  code that treated the dict-returning API as a dict. Available as
  ``quant_lib.ExploreResult`` (lazy via PEP 562) and
  ``quant_lib.research.ExploreResult``.

- **Shared pipeline helper for CLI/Python API**
  (`quant_lib/research/_pipeline.py`): ``run_explore`` (Python API)
  and ``quant_exp explore`` (CLI) had near-identical session/candidate
  construction logic with subtle differences (CLI used ``session=``
  attribute directly; Python API used local var). Drift was a real
  risk (e.g., one of them could forget the ``strategy=exp.strategy``
  argument). Extracted to ``build_explore_candidate(experiment_name,
  cache_dir)`` -- single source of truth. Both entry points now call
  it.

- **Real daily_equity plumbed through CommitResult**
  (`quant_lib/research/commit.py`, `quant_lib/cli/commit_cmd.py`):
  the Sprint 2 fix removed the synthetic 2-point fake equity curve
  and made the commit HTML chart show "Chart not available". Sprint
  3 completes the fix: added ``CommitResult.daily_equity:
  Optional[dict]`` carrying the REAL daily equity from
  ``commit_to_holdout`` (defaults to ``None`` for BC). The commit
  chart provider now renders the honest chart (or honest
  "Chart not available" when no trades executed).

### Tests added (Sprint 3)

- **`tests/test_sprint3_fixes.py`** (new file, 34 tests across 7
  classes): one test class per Sprint 3 fix.
  - `TestStaticDefaultsTypedDict`: 8 tests (schema ↔ literal sync,
    dict access preserved, mutation preserved)
  - `TestStrategyConfigStaysInSync`: 5 tests (key matching, value
    matching, framework-only exceptions, dict-shape guard)
  - `TestMutationBaselineInfra`: 4 tests (file exists, doc present,
    workflow exists, mutmut scope)
  - `TestCandidateTypedFields`: 3 tests (aliases exported, candidate
    annotations, runtime dicts work)
  - `TestExploreResultDataclass`: 6 tests (lazy resolution, attribute
    access, dict-style BC, keys/values/items, to_dict)
  - `TestSharedPipelineHelper`: 3 tests (importable, KeyError on
    unknown, returns Candidate + ExperimentConfig)
  - `TestCommitResultDailyEquity`: 5 tests (field present, default
    is None, BC instantiation without arg, accepts real dict,
    CLI chart provider uses real equity)
- **`tests/test_public_api.py`**: updated ``TestRunExplore`` to
  reflect new ``ExploreResult`` return type. Dict-style BC access
  asserted alongside new attribute access.

### Test counts

- v0.5.1 baseline: 1245 tests passing
- v0.5.2 (Sprint 1): 1259 tests passing (+14 net new tests)
- v0.5.3 (Sprint 2): 1283 tests passing (+24 net new tests)
- v0.5.4 (Sprint 3): **1317 tests passing** (+34 net new tests,
  +0 regressions)

### Changed (Sprint 2 — review-driven polish)

This release addresses 7 small-to-medium items identified in the
post-Phase-4 systematic review. All changes are backward-compatible
(no behavior change for valid inputs).

- **STRATEGY_* constants deduplicated** (`quant_lib/core/_config.py`,
  `quant_lib/audit/hypothesis.py`, `quant_lib/core/_engine.py`,
  `quant_lib/core/_features.py`): previously triplicated in three
  modules (audit, _features, _engine). Moved to single source in
  `core/_config.py` and imported by all consumers. The IntEnum in
  `audit/hypothesis.py` remains (it's the public type-safe surface)
  but its int values derive from the canonical constants. Anti-
  reintroduction guards via identity checks in
  `tests/test_sprint2_fixes.py::TestStrategyConstantsSingleSource`.

- **README version badge + citation updated** (`README.md`): badge
  was stale at 0.3.0 while pyproject.toml was 0.5.1. Updated both
  the badge and the BibTeX citation block to 0.5.1. Anti-reintro
  guard in `TestReadmeVersionBadge` (asserts README matches
  pyproject.toml version automatically).

- **API reference pages now render members** (`docs/api/{audit,core,
  research,tools}.md`): previously had `members: false` which made
  the mkdocs site render essentially nothing for these pages.
  Changed to `members: summary` so signatures and docstrings are
  visible. `quant_lib.md` already had explicit member list; left
  unchanged. Anti-reintro guard in `TestAPIReferencePages`.

- **HTML equity curve lie removed** (`quant_lib/cli/commit_cmd.py`,
  `tests/test_cli_helpers.py`): the commit HTML report previously
  rendered a SYNTHETIC equity curve built from only 2 points
  (initial + final). The code itself admitted "not a substitute
  for the real equity curve" -- misleading for a results chart.
  Removed `_build_equity_series_from_result`; the commit chart
  provider now returns None for equity_curve / drawdown_underwater
  (renders an honest "Chart not available" placeholder). The
  explore path is unchanged -- it has access to real daily_equity
  via the Candidate. Anti-reintro guard in
  `TestBuildEquitySeriesFromResultRemoved`.

- **`_looks_like_absolute` deduplicated** (new
  `quant_lib/cli/_utils.py`): identical 4-line helper existed in
  both `explore.py` and `commit_cmd.py`. Moved to `cli/_utils.py`
  as `looks_like_absolute`. Both files keep private aliases for
  backward compat with existing tests. Anti-reintro guards in
  `TestLooksLikeAbsoluteDeduplicated`.

- **Notebook 02 API consistency** (`notebooks/02_custom_experiment.
  {py,ipynb}`): the notebook example code had wrong field names
  (`training` instead of `train_start`/`train_end`, `min_volume_usd`
  instead of `min_volume_usdt`) and non-existent Hypothesis fields
  (`expectation`, `holding_period`, `exit_rules`, `invalidation`).
  Rewrote to use the canonical `for_vol_compression` factory +
  correct field names. Verified the example is executable end-to-end
  in `TestNotebook02APIConsistency::test_example_is_executable`.

- **`_safe_mklink` LATEST fallback now has a reader** (`quant_lib/cli/
  _output.py`): previously the OSError fallback wrote a `LATEST`
  marker file but no code in the framework ever read it -- the
  fallback was effectively dead code on Windows-without-admin.
  Added `read_latest_run_dir(results_dir=None)` helper that
  transparently resolves the latest run via symlink OR marker.
  Tests cover symlink, marker, missing-target, and default-dir paths
  in `TestReadLatestRunDir`.

- **Cleaned up stray test artifacts** in `quant_lib/cli/`: removed
  `data_cache/`, `results/`, and `hqs_execution.log` that earlier
  test invocations had accidentally created inside the package
  directory (should be at repo root, not inside a package).

### Tests added (Sprint 2)

- **`tests/test_sprint2_fixes.py`** (new file, 21 tests across 7
  classes): one test class per Sprint 2 fix. Anti-reintroduction
  guards use identity checks (constants), source-level greps (no
  redeclaration), and executable notebook assertions.
  - `TestStrategyConstantsSingleSource`: 5 tests
  - `TestReadmeVersionBadge`: 2 tests
  - `TestAPIReferencePages`: 4 parametrized tests
  - `TestBuildEquitySeriesFromResultRemoved`: 1 test
  - `TestLooksLikeAbsoluteDeduplicated`: 4 tests
  - `TestNotebook02APIConsistency`: 5 tests (incl. 1 end-to-end exec)
  - `TestReadLatestRunDir`: 5 tests
- **`tests/test_cli_helpers.py`**: replaced the
  `TestBuildEquitySeriesFromResult` test with an anti-reintro guard
  `TestBuildEquitySeriesFromResultRemoved` to prevent the misleading
  helper from being resurrected.

### Changed (Sprint 1 — v0.5.2: systematic review fixes)

This release addresses 6 issues identified in the post-Phase-4 systematic
review, prioritized by impact on production-grade claims. Backward
compatibility: the sl_pct behavior change is technically a soft BC
break (raised → skipped); users relying on the prior raise can use
the new `reject_reasons["invalid_sl_pct"]` counter instead.

- **PSR fallback is now sign-preserving**
  (`quant_lib/core/_wfa.py:208-223`): the degenerate-regime fallback
  (var_corr ≤ 0 or ESS < 2 or n_trades < 10) previously returned
  `psr = 0.5` (neutral). This allowed degenerate strategies to
  outcompete mediocre normal ones in Optuna search. Replaced with
  `psr = norm.cdf(w_sr)` — the asymptotic limit of Bailey's formula
  when skew/kurt corrections vanish. Sign is preserved (negative SR
  → psr < 0.5). For NaN w_sr (truly degenerate), Optuna rejects the
  trial — same behavior as before.

- **`commit_break` re-verifies on-disk seal** (`quant_lib/audit/holdout.py`):
  added `self.verify()` call at the top of `commit_break` so an
  in-memory stale `_tampered` flag cannot allow the seal to be broken
  after on-disk tampering (e.g., another process modified the seal
  file between session creation and commit). The verify→save order
  is enforced with a spy test (TestCommitBreakReVerify).

- **Methodology PSR coefficient corrected** (`docs/methodology.md:255`):
  the documented formula `(kurt_excess - 1)/4` was incorrect; the
  code uses `(excess + 2)/4 = (γ₄ - 1)/4` (correct Bailey convention).
  Doc now matches code. Anti-reintroduction guard in
  `tests/test_sprint1_fixes.py::TestMethodologyDocPSR`.

- **Methodology SPA attribution corrected** (`docs/methodology.md:296-302`):
  previously attributed the SPA add-one formula to "Davé (2008)";
  the correct attribution is Phipson & Smyth (2010) — already used
  in `_spa.py:297-303` (CHANGELOG v0.5.1 mentioned the code-side
  fix; doc was inconsistent). Doc now matches code.

- **Portfolio sim skips bad `sl_pct` instead of raising**
  (`quant_lib/core/_portfolio.py:404-418`): the original B0.2 guard
  raised `ValueError` on `sl_pct ≤ 0`, killing the entire backtest
  on a single corrupt trade. Changed to skip + log + increment
  `reject_reasons["invalid_sl_pct"]`. The "no silent
  ZeroDivisionError" property is preserved (the skip happens BEFORE
  the division). B0.2 regression test updated to reflect the new
  contract (asserts `reject_reasons["invalid_sl_pct"] == 1`,
  `executed_trades == []`, no exception). New "mixed valid+invalid"
  regression test confirms one bad trade no longer kills the
  backtest. See `tests/test_regression_b0_2_sl_pct.py` and
  `tests/test_sprint1_fixes.py::TestInvalidSlPctRejectReasons`.

  **Migration:** callers that relied on the raise to detect bad
  trades should now check `reject_reasons["invalid_sl_pct"] > 0`.
  This is a behavior change from v0.5.1; flagged in the v0.5.2
  release notes.

- **CLI persists full traceback on exception** (`quant_lib/cli/explore.py`,
  `quant_lib/cli/commit_cmd.py`): the prior `out.save_metrics(...)`
  saved only `{"status": "failed", "error": str(e)}`, swallowing
  the stack trace. Real bugs (typos, `AttributeError`, `ImportError`)
  were invisible. Now `traceback.format_exc()` is included as
  `metrics["traceback"]` for post-mortem analysis.

- **`quant_lib` import is now fast and lazy** (`quant_lib/__init__.py`):
  eager `from quant_lib import tools, audit, core, research` forced
  Numba JIT compile + pandas/numpy at every `import quant_lib`.
  Replaced with PEP 562 `__getattr__` lazy import: `import quant_lib`
  now takes ~0.004s instead of ~1.4s. Submodules still accessible
  as attributes (`quant_lib.core`, `quant_lib.research`, etc.) and
  `quant_lib.CommitResult` resolves via `__getattr__` for the
  `run_commit` return annotation.

### Tests added (v0.5.2)

- **`tests/test_sprint1_fixes.py`** (new file, 12 tests):
  - `TestPSRFallbackSignPreserving`: anti-reintroduction guard
    verifies the WFA fallback uses `norm.cdf`, not constant `0.5`.
  - `TestCommitBreakReVerify`: spy test confirms `verify()` is
    called before `_save_seal()`; behavior test confirms tampered
    disk seals abort `commit_break`.
  - `TestInvalidSlPctRejectReasons`: structural test for the new
    `reject_reasons["invalid_sl_pct"]` key.
  - `TestCLITracebackPersistence`: end-to-end CLI tests for
    `explore` and `commit` confirming `metrics.json` contains the
    full traceback.
  - `TestLazyImport`: cold-import subprocess test asserts
    `import quant_lib` takes < 1.5s; getattr-based access tests
    confirm submodule backward compat.
  - `TestMethodologyDocPSR`: source-level guards for the doc
    coefficient and SPA attribution fixes.
- **`tests/test_regression_b0_2_sl_pct.py`**: updated to reflect
  the new skip-not-raise contract. Added `test_mixed_valid_and_invalid_trades`
  confirming one bad trade no longer kills the backtest. Total
  tests in file: 7 (up from 3).

### Test counts

- v0.5.1 baseline: 1245 tests passing
- v0.5.2 (Sprint 1): 1259 tests passing (+14 net new tests,
  +1 updated regression test, +0 regressions)
- v0.5.3 (Sprint 2): **1283 tests passing** (+24 net new tests,
  +0 regressions)

### Skipped (deferred to future)

- Bug #29: standardize docstrings to NumPy style across all modules
  (broad refactor, low priority).
- Phase 3 high-risk refactors: `Trade` TypedDict, allocators dedup,
  EngineArgs call-site refactor, entry-slip helper, MTM dedup.
  See CHANGELOG v0.4.1 for the deferred list.

## [0.5.1] - 2026-07-01

### Changed (Polish: v0.5.1 — review-driven minor fixes)

This release addresses 4 issues + 2 test coverage gaps identified in
the post-Phase-4 systematic review. All changes are backward-compatible
(zero behavior change for valid inputs).

- **SPA risk_weight fallback consistency** (`quant_lib/core/_spa.py`):
  replaced hardcoded `0.005` fallback with `DEFAULTS["default_risk_per_pair"]`
  (= 0.01). Matches the convention from Phase 2 Bug #6 fix in
  `simulate_full_portfolio`. No behavior change for normal flow (WFA
  trades always have `risk_weight` set), but prevents inconsistency
  if a non-standard caller creates trades without `risk_weight`.
- **`simulate_trailing_stop_trade` zero-price guard**
  (`quant_lib/core/_engine.py:629`): added `if entry_price > 0.0 else 0.0`
  guard on `sl_pct_pct` calculation, matching the Phase 3 Bug #10
  fix in `fast_trade_loop`. Theoretical (real prices always positive)
  but restores symmetry between the two `@njit` engine functions.
- **Notebook 03 FDR section** (`notebooks/03_interpreting_results.ipynb`):
  converted FDR section from markdown-with-code-block to an actual
  code cell (plus an additional `alpha=0.15` example for comparison).
  Users can now run FDR interactively instead of copy-pasting.
- **Removed redundant inner `import numpy`** (`quant_lib/core/_wfa.py`):
  `_run_engine_on_data` had an inner `import numpy as np` that was
  redundant with the module-level import. Negligible perf impact
  (Python caches imports) but cleaner code.

### Tests added (v0.5.1)

- **SPA all-fail guard with `n_iters > 0`** (`tests/test_spa.py`):
  new test `test_spa_all_iters_produce_zero_equity_returns_p_one`
  covers the gap between `n_iters=0` (empty null array) and
  `n_iters>0` where all iterations produce empty trades (the actual
  all-fail scenario the Phase 3 guard handles). Mocks
  `simulate_full_portfolio` to always return `initial_capital`,
  then verifies `p_value == 1.0` and warning is logged.
- **`load_env_file` edge case tests** (`tests/test_utils_config.py`):
  4 new tests covering CRLF line endings, empty values (`KEY=`),
  spaces around `=` (`KEY = value`), and Unicode values.
  Total tests in file: 14 (up from 10).

### Skipped (deferred to future)

- Bug #29: standardize docstrings to NumPy style across all modules
  (broad refactor, low priority).
- Phase 3 high-risk refactors: `Trade` TypedDict, allocators dedup,
  EngineArgs call-site refactor, entry-slip helper, MTM dedup.
  See CHANGELOG v0.4.1 for the deferred list.

## [0.5.0] - 2026-06-30

### Changed (Phase 4: Documentation & infrastructure — v0.5.0)

This release focuses on **infrastructure hardening** and **discoverability**
rather than algorithmic changes. No behavior change for existing users.

- **`utils/config.py` implemented** (`quant_lib/utils/config.py`): the
  file was previously a stub (``"""Shared utilities."""``). Now exports:
  - ``find_repo_root(start=None)``: locate project root via pyproject.toml.
  - ``load_env_file(env_path=None)``: parse ``.env`` into a dict.
  - ``get_hmac_secret_with_fallback()``: env var → .env fallback chain.

  Re-exported from ``quant_lib.utils`` for convenience:
  ``from quant_lib.utils import find_repo_root, load_env_file``.
- **API reference pages in mkdocs** (`mkdocs.yml`, `docs/api/*.md`):
  added 5 auto-generated API reference pages (`quant_lib`, `core`,
  `research`, `audit`, `tools`). Users can now browse API signatures
  and docstrings via the published docs site.
- **3 starter notebooks** (`notebooks/01_quick_start.ipynb`,
  `02_custom_experiment.ipynb`, `03_interpreting_results.ipynb`):
  written as plain nbformat v4 JSON (no `nbformat` build dependency).
  Each has a corresponding `.py` source file with helper functions for
  regeneration. Includes `notebooks/README.md` with usage instructions.
- **Module-level assert under `-O` fix** (`quant_lib/core/_config.py`):
  replaced `assert STATIC["wfa_purge_days"] >= 30` with
  `if/raise AssertionError` so the invariant check runs even under
  `python -O` (where `assert` statements are stripped).
- **Gap detection ffill ordering comment** (`quant_lib/core/_features.py`):
  added inline comment explaining why ffill must precede null-out of
  gap contamination (reverse order would silently re-fill the gap
  window from pre-gap data). No code change.
- **Mutation baseline workflow** (`mutation_baseline.txt`): expanded
  documentation with Option A (GitHub Actions) and Option B (local)
  capture procedures. Baseline NOT yet captured (requires first
  CI run). Acceptance threshold documented (5pp drop max).

### Skipped in v0.5.0 (deferred)

- **Bug #29**: standardize docstrings to NumPy style across all modules
  (broad refactor, low priority).
- **Bug #31**: enforce mutation threshold in CI workflow (requires
  captured baseline first; will be added in v0.5.x after first
  weekly CI run completes).

### Tests (v0.5.0)

- **`tests/test_utils_config.py`** (NEW, 10 tests):
  - `TestFindRepoRoot` (2 tests): finds project root, returns start when no pyproject.toml
  - `TestLoadEnvFile` (5 tests): simple key=value, comments/blanks ignored, quote stripping, missing file, equals in value
  - `TestGetHmacSecretWithFallback` (3 tests): raises when missing, env var priority, .env fallback

## [0.4.1] - 2026-06-30

### Changed (Phase 3: Refactor & edge cases — v0.4.1)

This release is a **pure refactor** with **no behavior change** for
valid inputs. Existing experiment results are unaffected. The focus is
on adding defensive guards, clarifying misleading variable names, and
providing optional parameters for previously hardcoded assumptions.

- **Zero/negative price guards in `@njit` engine** (`core/_engine.py:223, 453, 470`):
  added `if closes[i] > 0.0` / `if entry_price > 0.0` guards before
  divisions that could produce `inf`/`NaN` in extreme price scenarios
  (theoretical for illiquid pairs). Previously, zero/negative prices
  would propagate NaN through cost calculations and corrupt trade PnL.
  Defensive only — no behavior change for typical crypto pairs.
- **SPA all-fail edge case** (`core/_spa.py:287`): detect when all
  SPA iterations produce empty/zero equity (random_equities all equal
  initial_capital) and return `p_value=1.0` (cannot reject null) instead
  of the misleading `p_value=1/(N+1)` from the Phipson-Bell add-one
  formula. Logs a warning pointing to upstream `simulate_trailing_stop_trade`.
- **`run_trade_bootstrap` accepts `trade_dates`** (`core/_metrics.py:run_trade_bootstrap`):
  new optional parameter `trade_dates: pd.DatetimeIndex | None`. When
  provided, CAGR is annualized using the actual date span
  `(trade_dates[-1] - trade_dates[0]).days` instead of treating
  1 trade ≈ 1 day. The proxy over-inflates CAGR for sparse strategies
  (e.g., 100 trades over 2 years → annualized as 100 days instead of
  ~730 days). Length mismatch with `trade_r_vals` logs a warning and
  falls back to proxy. **Migration:** none required (default preserves
  prior behavior).
- **`_coefficient_of_variation` helper extracted** (`core/_metrics.py`):
  extracted inline CV calculation into a reusable helper function.
  Returns 0.0 when |mean| < 1e-12 (avoids div-by-zero). Used in
  `print_param_stability` (Phase 3 cleanup; remaining 2 inline sites
  will be migrated in a follow-up).
- **Rename `max_idx` → `exit_limit_idx`** (`core/_engine.py:simulate_trailing_stop_trade`):
  the local variable `max_idx` was misleading (sounds like "maximum
  seen index", but is actually the upper bound of the bailout window).
  Renamed to make intent explicit. No behavior change.
- **`_shared_corr_cache` docstring** (`core/_spa.py:portfolio_spa`):
  documented sharing constraints (safe within SPA iterations, unsafe
  across independent backtests). No code change.
- **`_rescale_factors_to_total` docstring** (`core/_risk_allocation.py`):
  expanded notes section explaining the silent failure mode (when all
  PF factors are zero, the fold gets risk_weight=0 and is effectively
  skipped). No behavior change — the function still returns `{}` in
  that case (verified by `tests/test_risk_allocation.py`).

### Skipped in v0.4.1 (deferred to v0.5.0)

These refactors were planned but **deferred** because they require
extensive call-site refactors with non-trivial regression risk:

- **Bug #11**: 33-arg positional call to `fast_trade_loop` → `EngineArgs.as_tuple()`
  (the helper exists; call sites in `_wfa.py` still use positional).
- **Bug #13**: Extract `_apply_pf_weighted_allocation_core` from
  `_risk_allocation.py` (90% OOS/IS duplication).
- **Bug #14**: Extract `_process_liquidations_and_mtm` helper from
  `_portfolio.py` (30-line MTM duplication).
- **Bug #15**: Extract `_compute_entry_slip` `@njit` helper from
  `_engine.py` (4× entry-slip duplication).
- **Bug #18**: `Trade` TypedDict migration (large refactor across 7+ files).
- **Bug #24**: Logger name `"rich"` → `__name__` (would break 4 test
  files that mock the logger by name; needs coordinated update).

These are non-urgent and should be done in a dedicated refactor sprint.

### Tests (v0.4.1)

- **`tests/test_metrics.py`** — Added 7 new tests:
  - `TestTradeBootstrapTradeDates` (3 tests): trade_dates accepted,
    length mismatch logs warning.
  - `TestCoefficientOfVariation` (4 tests): normal values, zero mean,
    negative mean, constant values.

## [0.4.0] - 2026-06-30

### Changed (Phase 2.4: Correctness & dead-code cleanup — v0.4.0)

- **`label_p_value` strict context validation** (`core/_testing.py`):
  unknown context strings (typos, new contexts) now raise ``ValueError``
  instead of silently falling back to "mean_r" with a warning. The
  silent fallback could mask typos like "spa_p" or "sharpe" and produce
  mislabeled reports. Valid contexts: ``{"mean_r", "spa"}``. Updated
  docstring lists common typos and their corrections. **Migration:**
  callers passing other context strings must update to a valid one.
- **`prob_sharpe_ratio` uses sample std (ddof=1)** (`core/_testing.py`):
  unweighted SR now uses ``np.std(returns, ddof=1)`` (sample std) for
  consistency with the variance correction denominator ``n - 1``
  downstream. Prior version used ``np.std(returns)`` (population std,
  ddof=0) which under-reported SR by ~5% for small n (~10-30) and was
  inconsistent with the docstring claim. Weighted path already used
  weighted variance. **Migration:** SR values will shift slightly
  (~2-5% for n < 30, <1% for n > 100). Existing reports can be
  re-generated without re-running experiments.
- **`_portfolio.py` risk_weight fallback uses config** (`core/_portfolio.py`):
  when a trade has no ``risk_weight`` field and ``asset_risk_weights``
  doesn't have the symbol, the fallback now uses
  ``DEFAULTS["default_risk_per_pair"]`` (= 0.01) instead of the
  hardcoded 0.005. This brings the portfolio simulator into
  consistency with the WFA path which already used the config key.
  **Migration:** default-risk trades (no risk_weight, no symbol in
  asset_risk_weights) now risk 1.0% per trade instead of 0.5%. Strategies
  that were on the margin of acceptance/rejection may need re-validation.
- **`Candidate` declares `_is_trades_per_fold_by_sym` as a field**
  (`research/candidate.py`): the attribute is now a dataclass field
  initialized in ``__init__`` rather than lazy-initialized via
  ``hasattr``. No behavior change, but eliminates a subtle bug class
  (e.g., ``.pyc`` cache mismatches, IDE autocomplete failures, hot-path
  ``hasattr`` overhead). Verified by
  ``tests/test_candidate.py::TestCandidateInit::test_init_declares_is_trades_field``.
- **`WalkForwardObjective` rejects empty input** (`core/_wfa.py`):
  ``__init__`` raises ``ValueError`` immediately if ``df_prepped`` is
  empty. Previously an empty input produced silently-corrupt
  ``bar_weights`` (NaN from ``empty_array.mean()``) and only failed
  downstream at ``__call__``. The fail-fast surfaces the real bug
  (empty upstream data) at the source. **Migration:** callers passing
  empty DataFrames must filter or guard upstream.
- **`prepare_data_with_max_time` gains explicit `apply_holdout_ema_to_full`**
  (`core/_features.py`): when ``btc_holdout_start`` is provided, this
  flag controls whether the holdout-constant EMA is applied to the
  entire df (default ``True``, pre-v0.4.0 behavior) or only to bars
  ``>= btc_holdout_start`` (IS bars keep the dynamic EMA, matching the
  WFA path). The new ``False`` mode is opt-in for users who want the IS
  signal to match the WFA training path exactly. **Migration:** none
  required (default preserves prior behavior).

### Tests (v0.4.0)

- **`tests/test_statistics.py`** — Replaced
  ``test_unknown_context_falls_back_to_mean_r`` with
  ``test_unknown_context_raises_value_error`` and added
  ``test_valid_contexts_accepted``.
- **`tests/test_psr_ess.py`** — Added
  ``test_sr_uses_sample_std_ddof1`` (small n) and
  ``test_sr_ddof_consistency_large_n``.
- **`tests/test_features.py`** — Added ``TestApplyHoldoutEMAToFull``
  with 3 tests (default-true, explicit-true matches default,
  explicit-false restores dynamic EMA in IS).
- **`tests/test_wfa.py`** — Added ``TestWalkForwardObjectiveEmptyInput``
  with 2 tests (empty DataFrame raises, non-empty succeeds).
- **`tests/test_candidate.py`** — Added
  ``test_init_declares_is_trades_field`` to verify the dataclass
  field exists.
- **`tests/test_portfolio.py`** — Added
  ``TestRiskWeightFallbackConsistency`` with 2 source-level guards
  (no hardcoded 0.005, DEFAULTS key is used).

### Fixed (v0.3.1 hotfix — Critical statistical bugs)

- **PSR kurtosis formula error** (`core/_testing.py:159`,
  `core/_wfa.py:199`): the Bailey & López de Prado (2012) variance
  correction was using `(excess_kurt - 1) / 4` treating scipy's excess
  kurtosis as if it were regular kurtosis. This understated the variance
  by 3/4 · SR² and systematically **inflated PSR**, especially for
  fat-tailed strategies where excess kurtosis > 0. For normal data
  with SR=2 the formula gave var_correction = 0 (clipped to 1e-8)
  where the correct value is 1.5. Now uses `(excess_kurt + 2) / 4`
  (= `(regular_kurt - 1) / 4` with `regular = excess + 3`), matching
  `deflated_sharpe_ratio` which already used the correct conversion.
  **Impact:** all prior PSR values and strategy selection results
  derived from this formula should be re-validated.

- **`run_bootstrap` DD_Pctile sign convention** (`core/_metrics.py`):
  added `assert max_dd <= 0` to enforce the negative-percentage
  convention (e.g. `-20.0` for -20% drawdown). Previously, callers
  passing positive `max_dd` produced `DD_Pctile = 100%` always because
  the percentile comparison used mixed signs (bootstrap values are
  negative, observed was being treated as positive). Added explicit
  convention docstring with reference to `research/commit.py:500` and
  `research/reporting.py:170` (callers that compute it correctly).
  Existing callers already pass negative values; the assertion catches
  future callers that would silently produce meaningless percentiles.

- **Duplicate fold log message** (`core/_wfa.py:716-729`): the "no
  trades" branch had two identical `console.print` calls back-to-back
  (copy-paste artifact). Each fold with zero OOS trades was logged
  twice. Removed the duplicate.

### Tests (v0.3.1)

- **`tests/test_psr_ess.py`** — Updated `test_weighted_psr_matches_wfa_formula`
  and `test_kurtosis_uses_excess_convention` to reflect the corrected
  formula. Added `test_psr_uses_correct_kurtosis_coefficient` which
  verifies the implementation produces PSR matching the analytical
  corrected formula (and differs from the old buggy formula by more
  than 1e-3) for fat-tailed data with measurable excess kurtosis.
- **`tests/test_wfa.py`** — Added `test_no_trade_fold_prints_exactly_once`
  (regression guard against Bug #3) and `TestWFA_PSRFormulaConsistency`
  with `test_wfa_objective_uses_corrected_kurtosis_formula` (source-level
  guard against re-introducing the (kurt - 1)/4 form).
- **`tests/test_metrics.py`** — Added `TestRunBootstrapConvention` with
  3 tests enforcing the negative-percentage convention and verifying
  DD_Pctile is no longer always 100% for observed negative drift.

### Migration notes (v0.3.1)

- The PSR correction will **reduce PSR for fat-tailed strategies**
  with positive excess kurtosis (typical for crypto). Strategies
  previously classified as TRADE/WATCH may drop to RESEARCH/NO EDGE.
  Run `quant_exp status <experiment>` to re-check tier.
- Bootstrap DD_Pctile values may shift significantly if you re-run
  bootstrap on existing experiment results (previously always 100%).

### Changed (Phase 4: DX & Polish — Trade bootstrap, regime stats, CLI flags)

- **Per-trade bootstrap** (`core/_metrics.py`): new
  `run_trade_bootstrap` performs circular block bootstrap on trade
  R-multiples directly, replacing the daily-return approach for
  trade-based strategies. Daily-return bootstrap overstates
  confidence when n_trades &lt; 200 because most days have zero
  trades; trade bootstrap tests the correct null: "trades are
  independent with replacement." 5 new fields in `CommitResult`:
  `trade_bootstrap_worst5_cagr`, `trade_bootstrap_worst95_dd`,
  `trade_bootstrap_worst5_dd`, `trade_bootstrap_worst1_dd`,
  `trade_bootstrap_block`. NaN when n_trades &lt; 5.
- **Regime stats wired into commit report** (`research/commit.py`):
  `compute_regime_stats` was implemented (`core/_metrics.py:98-117`)
  but never called in the commit path. Now computed and stored in 4
  new `CommitResult` fields: `regime_bull_pf`, `regime_bull_n`,
  `regime_bear_pf`, `regime_bear_n`. NaN when n_trades &lt; 3.
- **CLI flags `--cache-dir` and `--seal-dir`** (`cli/explore.py`,
  `cli/commit_cmd.py`): both the `quant_exp explore` and
  `quant_exp commit` commands now accept `--cache-dir` (default
  `./data_cache`) and `--seal-dir` (default env var fallback,
  then convention). The Python API already supported these via
  `ResearchSession(cache_dir=..., seal_dir=...)`. The CLI now
  matches. The seal_dir is set as `QUANT_LIB_SEAL_DIR` env var
  before the session is created so it propagates to all layers.

### Notes

- The trade bootstrap (Phase 4.1) is additive — all existing
  components (daily-return bootstrap in `run_bootstrap`) remain
  unchanged. The new function targets trade-based strategies where
  daily-return bootstrap is statistically less appropriate.
- CLI flags are optional: existing workflows (no flags) use the
  same defaults as before (`./data_cache`).
- All 1130 unit tests pass.

### Changed (Phase 3: MEDIUM-severity integrity fixes — Atomic writes, path validation, size cap)

- **Atomic journal write** (`audit/journal.py`): `_save_to_disk` now
  writes via `tempfile.mkstemp` + `os.replace` instead of direct
  `open()`. A crash mid-write leaves the original audit trail intact.
  Previously could produce half-written JSON that silently lost the
  decision history — a direct hit to the framework's "no look-ahead"
  value proposition.
- **Atomic seal write** (`audit/holdout.py`): same fix for
  `_save_seal`. Crash mid-write previously could leave a half-written
  JSON whose missing signature would cause `verify()` to fail, a
  false-positive tamper detection.
- **Path validation in data layer** (`core/_data.py`): `symbol` is
  validated against `^[A-Z0-9]{2,20}USDT$` and `interval` against
  `{1m, 5m, 15m, 1h, 4h, 1d}` before use in `os.path.join()`.
  Prevents path traversal through malformed inputs (defense-in-depth;
  callers are trusted in practice but the validation closes the
  hole at the boundary).
- **Response size cap** (`core/_data.py`): `fetch_with_retry` now
  uses `stream=True` + `iter_content` with a 500 MB hard cap.
  Prevents unbounded response bodies from OOM'ing the process.
  The cap is well above any legitimate Binance Vision response.
- **Atomic init file writes** (`cli/init_cmd.py`): new
  `_atomic_file_write` helper used by `quant_exp init` for both
  experiment files and `.env` templates. Crash during scaffold
  creation no longer leaves truncated files.

### Notes

- No behavior change for correct inputs and normal operation. The
  fixes are purely defensive: crash resilience and input validation.
- The `fetch_with_retry` streaming change preserves the existing
  API (`Response` with `._content`). Callers use `res.content`
  identically.
- All 1130 unit tests pass.

### Changed (Phase 2: HIGH-severity fixes — Statistical integrity, look-ahead, market impact)

- **Deflated PSR (Bailey & López de Prado 2014)** added (`core/_testing.py`):
  `deflated_sharpe_ratio()` function adjusts the single-trial PSR for
  the family of Optuna trials that produced the winning params. This
  is the missing piece between PSR (single-trial) and the existing
  Bonferroni correction on `n_commits` (which under-counts the real
  family: `n_symbols × n_folds × n_optuna_trials ≈ 14,400`).
  - 11 new tests in `tests/test_psr_ess.py::TestDeflatedSharpeRatio`.
  - New `deflated_psr` and `n_trials_in_deflated` fields in
    `CommitResult` (with sensible defaults; NaN when n_trials < 2).
  - Computed in `commit_to_holdout` using `n_folds × n_trials_per_fold`
    as the family size and the per-trade Sharpe as the observed SR.
- **Macro trend isolation** (`core/_features.py` + `commit.py`): new
  `btc_holdout_start` parameter on `prepare_data_with_max_time`. When
  set (commit-time only), the asset-level EMA is computed strictly
  on data BEFORE this timestamp, and the last value is forward-filled
  into the holdout. Previously, the holdout's own price action
  influenced the trend signal (state-persistence leak). The trade
  multiplier (1.5x with-trend) could be tuned by looking at the
  holdout; this is now prevented.
  - 3 new tests in `tests/test_features.py::TestMacroTrendHoldoutIsolation`.
  - Default behavior (no `btc_holdout_start`) unchanged for WFA path.
- **Funding data hash** (BC break, Phase 2.3): funding rate data is
  now part of the holdout seal hash. Tampering with funding between
  session creation and commit would silently change trade PnL
  (via the engine's `funding_impact_pct` term); this was previously
  undetectable. Funding is pre-loaded at session init and verified
  at commit time.
  - **BC break**: existing seals (pre-2.3) have hashes computed
    without funding. Users with existing seals must re-create
    sessions to use Phase 2.3+.
  - New `_holdout_funding_for_hash` attribute on `ResearchSession`.
  - New optional `_holdout_funding` parameter on `ResearchSession`
    for tests that pre-load their own data.
  - 5 new tests in `tests/test_session.py::TestFundingDataHash`.
- **Market impact cap** (`core/_config.py` + `core/_portfolio.py`):
  new `market_impact_volume_pct` (default 0.01 = 1% of 24h volume)
  in `DEFAULTS`. Position notional is capped at this fraction of
  daily close (proxy for 24h volume) × leverage. Prevents the trend
  multiplier (1.5x with-trend) from creating unrealistically large
  orders on illiquid assets where live fill price would move
  against the order. Cap is logged once per symbol per run.
  - 4 new tests in `tests/test_portfolio.py::TestMarketImpactCap`.
- **WFA exposes IS trades** (`core/_wfa.py` + `research/candidate.py`):
  `run_wfa_per_symbol` now returns a 3-tuple
  `(oos_trades, fold_params, is_trades_per_fold)`. The new
  `is_trades_per_fold` is a list of per-fold IS trade dicts, generated
  by running `fast_trade_loop` on IS data with the winning Optuna
  params. This is the foundation for fully decoupling the
  meta-allocator (IS-based) from the strategy selector (OOS-based).
  - **Future work**: the PF-weighted risk allocator in
    `_risk_allocation.py` still consumes OOS trades. Switching it
    to consume IS trades is deferred to a follow-up to avoid scope
    creep in this PR. The structural change (3-tuple return) is
    shipped now so downstream consumers can adopt it incrementally.
  - All 6 existing `test_wfa.py` test sites updated to unpack
    the 3-tuple. The candidate.py call site updated.
  - New `candidate._is_trades_per_fold_by_sym` attribute (captured
    for future use by the IS-based allocator).

### Notes

- All 1130 unit tests pass after these changes.
- `ruff check quant_lib/` passes with no errors.
- The headline statistical improvement is the Deflated PSR (Phase 2.1).
  It addresses the FWER under-correction that the previous Bonferroni
  adjustment on `n_commits` missed. A deflated_psr < 0.95 means the
  strategy's edge is likely just the best of N trials under the null,
  not a real edge worth deploying capital.
- The macro trend fix (Phase 2.2) and funding hash (Phase 2.3) close
  silent leak / tampering vectors that were not detected by the
  previous holdout seal mechanism. The seal now covers the full
  no-peek guarantee.
- The market impact cap (Phase 2.4) is a first approximation. A
  more accurate implementation would use actual 24h volume data
  (the daily close proxy is conservative — it underestimates true
  volume, so the cap is tighter than optimal).
- The WFA refactor (Phase 2.5) is structural; the actual
  IS-based PF allocation is a follow-up. Tracked separately.

### Changed (Phase 1: Quick Wins — Stability, Consistency, Look-ahead)

- **Best-params selection is now stability-gated** (`research/best_params.py`):
  Replaces pure "best fold by PSR" with a CV-based stability gate. When
  the best fold's params have mean CV > 30% across folds (UNSTABLE
  rating per `print_param_stability`), the selection falls back to
  per-param MEDIAN across folds. This adds a final safety net against
  lucky outliers in the best fold while preserving the best fold's
  signal when params are stable. CV threshold is configurable via the
  new ``cv_threshold`` parameter on ``pick_best_params_per_symbol``
  (default 0.30). Backward compatible: when CV ≤ threshold, behavior
  is identical to the prior best-per-fold selection.
  - 7 new tests in `tests/test_best_params.py::TestStabilityGate`.
  - 6 existing tests updated to use stable param sets so the gate
    does not spuriously trigger.
- **`kurtosis` field in `CommitResult` now uses `fisher=True`** (excess
  kurtosis, 0 for normal) for consistency with the PSR formula in
  `core/_testing.py:141`. Previously used `fisher=False` (regular
  kurtosis, 3 for normal), making the reported value inconsistent
  with PSR's internal computation. **Breaking for any user reading
  the `kurtosis` field — values shift by 3** (e.g., reported 4.5
  regular → 1.5 excess). No callers observed.
- **`ess` field in `CommitResult` now reports Kish-corrected ESS**
  (uniform weights → equals n_trades). Previously reported `n-1`
  (sample variance df), which was inconsistent with the label "ess".
  No callers observed that depend on the prior convention.
- **Zero-trade commit now reports `psr=NaN`** instead of 0.5 (which
  misleadingly implied "neutral coin flip"). `NaN` correctly conveys
  "no evidence either way."
- **SPA p-value correction renamed**: The "Davé 2008 SPA correction"
  comment is corrected to "Phipson-Bell (2010) add-one correction" —
  the formula `(n_exceed + 1) / (n_iters + 1)` is the standard
  Phipson-Bell add-one for permutation tests, not Davé 2008. Formula
  unchanged. The label was incorrect; the implementation is correct.
- **ATR feature is now `shift(1)`** (`core/_features.py:113`): Today's
  ATR no longer uses today's high/low/close. ATR at bar t is the
  rolling mean of True Range up to bar t-1. This is a real (small)
  look-ahead fix: the engine uses ATR for SL distance at entry bar,
  so today's SL should not depend on today's realized volatility.
- **Error message in `commit_to_holdout` uses configured seal dir**:
  The "race condition" error message now uses `QUANT_LIB_SEAL_DIR` env
  var (falling back to the convention default) instead of a hardcoded
  `data_cache/holdout_seals/` path. No behavior change for default
  setups; correctness for users with custom seal directories.
- **Docstring drift fix** (`commit.py:9`): "best-last-fold per symbol"
  → "stability-gated best fold per symbol" to match actual code
  behavior.

### Notes

- These changes are part of a phased review-driven cleanup. The
  stabilities-gated best-params selection (Phase 1's headline change)
  is now the canonical behavior. No existing functionality is
  removed; only the reporting kurtosis convention, ESS convention,
  and zero-trade PSR semantics change. All 1107 unit tests pass
  after these changes.

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
