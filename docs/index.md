# quant_lib

Honest backtesting toolkit for quantitative trading strategies, with
sealed holdout discipline, multiple-testing correction, and
reproducible research guarantees.

## Why quant_lib?

- **No look-ahead**: holdout is sealed at session start with HMAC-SHA256.
  You cannot peek at the holdout data without breaking the seal.
- **Multiple-testing correction**: Bonferroni + Benjamini-Hochberg FDR.
  Every experiment is counted; ablation studies are half-weighted.
- **Probabilistic Sharpe Ratio** (Bailey & Lopez de Prado 2012): accounts
  for skewness and kurtosis, not just mean/variance.
- **Superior Predictive Ability** (Hansen 2005 + Davé 2008): portfolio-level
  test with time-anchored circular permutation.
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

- [Methodology](methodology.md) — paper-grade methodology writeup
- [Testing](testing.md) — how to run and write tests
- [Contributing](contributing_source.md) — development setup and PR checklist
- [Changelog](changelog_source.md) — version history
- [GitHub Repository](https://github.com/Hansiongs/hans-backtest)
