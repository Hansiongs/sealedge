# sealedge

Backtesting toolkit for quant strategies with sealed holdouts,
multiple-testing correction, and a reproducible research workflow.

**Distribution name:** `sealedge` (PyPI / repo). **Importable package:**
`quant_lib`. **CLI:** `quant_exp`.

## Why sealedge?

- **No look-ahead**: holdout is sealed at session start with HMAC-SHA256.
  You cannot peek at the holdout data without breaking the seal.
- **Multiple-testing correction**: Bonferroni + Benjamini-Hochberg FDR.
  Every experiment is counted; ablation studies are half-weighted.
- **Probabilistic Sharpe Ratio** (Bailey & Lopez de Prado 2014): accounts
  for skewness and kurtosis, not just mean/variance.
- **Superior Predictive Ability** (two coexisting nulls):
  - **Legacy** (default, `recenter_policy="legacy"`): uniform
    time-anchored permutation of observed trades + Phipson & Smyth
    2010 add-one correction. Stable regression-tested 3-tuple contract;
    the default path that still blocks a made-up edge on trade outcomes.
  - **Hansen-literal** (opt-in, `recenter_policy="hansen_literal"` +
    `trial_r_nets` + `return_statistics=True`): Politis, Romano
    stationary block bootstrap over per-trial IS loss-differentials
    + Hansen (2005) Eq.7 recenter/discarding + Eq.8 cross-strategy
    max-statistic + Phipson-Smyth add-one. The max is the
    multiple-testing correction (the whole point of White's Reality
    Check) the legacy path lacks. Numpy-only `pnl_array` resampling,
    so the SPA spy invariant holds on both paths.
    Hansen-corrected p exposed as `spa_p_value`; legacy p preserved
    in `spa_naive_p_value` for transparency. See
    [`docs/methodology.md`](methodology.md) §6 for the exact null,
    finite-sample divergences, and the three user-accepted caveats
    (finite-sample power of max-of-K may be low; KS<0.25 finite-sample is
    empirical-only; spy `2*n_iters` invariant is gated to the legacy
    path).
- **Walk-Forward Analysis**: Optuna-based per-symbol optimisation with
  L2 regularization toward search-space center.
- **Per-fold PF-weighted risk allocation**: decaying halflife weights
  over WFA folds, preserving total portfolio risk (X1 scheme).
- **State machine**: `hypothesis → universe → edge → narrowed → ready`.
  Every transition is validated; backward transitions are blocked.

## Quick Start

```python
from quant_lib import run_explore, run_commit

CACHE_DIR = "./data_cache"

# Phase 0-3: explore (holdout stays sealed)
result = run_explore("vol_compression_v1", cache_dir=CACHE_DIR)
print(f"SPA p-value: {result['spa_p_value']}")

# Phase 4: commit (irreversible, breaks seal)
commit = run_commit("vol_compression_v1", cache_dir=CACHE_DIR)
print(f"PSR: {commit.psr}")
```

## Architecture

```
experiments/  →  cli/  →  research/  →  audit/  →  tools/  →  core/  →  utils/
   (config)      (Typer)   (session)    (seal)   (public API)  (engine)   (shared)
```

## Quick Links

- [Methodology](methodology.md), paper-grade methodology writeup
- [Testing](testing.md), how to run and write tests
- [Contributing](contributing_source.md), development setup and PR checklist
- [Changelog](changelog_source.md), version history
- [GitHub Repository](https://github.com/Hansiongs/sealedge)
