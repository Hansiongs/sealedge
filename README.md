# sealedge

**Sealed-holdout backtesting for crypto strategies.**

sealedge is a Python library for checking whether a trading strategy has a
statistical edge on historical crypto data under sealed-holdout and
multiple-testing defaults. It packages research hygiene (no look-ahead,
HMAC holdouts, SPA/PSR reporting) so results are re-runnable and hard to
tamper with silently.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL_3.0--or--later-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1325_passing-brightgreen.svg)](.github/workflows/tests.yml)
[![CI](https://github.com/Hansiongs/sealedge/actions/workflows/tests.yml/badge.svg)](.github/workflows/tests.yml)
[![Lint](https://github.com/Hansiongs/sealedge/actions/workflows/lint.yml/badge.svg)](.github/workflows/lint.yml)
[![Version 0.5.1](https://img.shields.io/badge/version-0.5.1-blue.svg)](CHANGELOG.md)

## Why sealedge?

The public name is **sealedge**. The importable package is still
**`quant_lib`** (CLI: `quant_exp`). Most backtesters make it easy to peek at
the holdout, tune until it "works", and ship a flattering report. sealedge
pushes back with:

- **Cryptographically-sealed holdouts**, tampering between
  initialization and commit is detected
- **No look-ahead features**, every feature uses `shift(1)`
- **Multiple-testing correction**, Bonferroni (1-indexed) + FDR (BH)
- **PSR + ESS** instead of raw Sharpe (accounts for skew/kurtosis,
  autocorrelation)
- **SPA test** (two coexisting nulls selectable via `portfolio_spa`
  kwargs):
  - **Legacy** (default, `recenter_policy="legacy"`): uniform
    time-anchored circular permutation of observed trades, with
    Phipson & Smyth (2010) add-one correction. Stable regression-tested
    3-tuple contract.
  - **Hansen-literal** (opt-in, `recenter_policy="hansen_literal"` +
    `trial_r_nets=...` + `return_statistics=True`): Politis, Romano
    (1994) stationary block bootstrap over per-trial IS
    loss-differentials + Hansen (2005) Eq.7 recenter/discarding +
    Eq.8 cross-strategy max-statistic + Phipson & Smyth add-one.
    The cross-strategy max is the multiple-testing correction (the
    whole point of White's Reality Check) that the legacy path lacks.
    Numpy-only on `pnl_array`s so the SPA spy invariant holds on
    both paths. Hansen-corrected p is in `spa_p_value`; legacy p is
    preserved in `spa_naive_p_value` for transparency. See
    [docs/methodology.md](docs/methodology.md) §6 for the exact
    null, finite-sample divergences, and the three user-accepted
    caveats (honest-power may be a negative finding;
    KS<0.25 is empirical-only; spy gated to legacy path).

See [docs/methodology.md](docs/methodology.md) for the full method
writeup.

## Quick Start

### 1. Use the Python API (recommended for notebooks)

```python
from quant_lib import run_explore, run_commit

# NOTE: cache_dir is relative to your current working directory.
# Pass an absolute path for reproducible results across working dirs.
CACHE_DIR = "./data_cache"  # or e.g. "/var/quant_cache"

# Phase 0-3: explore (holdout stays sealed)
result = run_explore("vol_compression_v1", cache_dir=CACHE_DIR)
print(f"SPA p-value: {result['spa_p_value']}")
print(f"Final equity: ${result['final_equity']:,.2f}")

# Phase 4: commit (irreversible, breaks seal)
commit = run_commit("vol_compression_v1", cache_dir=CACHE_DIR)
print(f"PSR: {commit.psr}")
print(f"Final equity: ${commit.final_equity:,.2f}")
```

### 2. Use the CLI (recommended for production runs)

```bash
# List registered experiments
$ quant_exp list
┌──────────────────┬────────────────┬────────────────┬─────────────┐
│ Name             │ Strategy       │ Train          │ Symbols     │
├──────────────────┼────────────────┼────────────────┼─────────────┤
│ pullback_sniper… │ pullback_sniper │ 2020-01-01→…  │      3      │
│ vol_compression… │ vol_compression│ 2020-01-01→…  │      3      │
└──────────────────┴────────────────┴────────────────┴─────────────┘

# Show details
$ quant_exp show vol_compression_v1

# Run OOS exploration (Phase 0-3)
$ quant_exp explore vol_compression_v1

# Run final commit (Phase 4, irreversible)
$ quant_exp commit vol_compression_v1

# Show holdout seal status
$ quant_exp status
```

### 3. Use the low-level API (for custom pipelines)

```python
from quant_lib.audit import for_vol_compression
from quant_lib.research.session import ResearchSession
from quant_lib.research.candidate import Candidate
from quant_lib.research.commit import commit_to_holdout

# 1. Build hypothesis (BEFORE looking at data)
hypothesis = for_vol_compression(
    name="vol_breakout_v1",
    mechanism="Volatility compression + volume breakout = momentum",
    boundary_conditions="Fails in strong trends without pullback",
    success_criteria="SPA p < 0.15, PF > 1.3, min 30 trades",
)

# 2. Create research session (seals holdout)
session = ResearchSession(
    training_period=("2020-01-01", "2024-12-31"),
    holdout_period=("2025-01-01", "2025-06-30"),
    symbols=["BTCUSDT", "ETHUSDT"],
    cache_dir="./data_cache",
)

# 3. Run phases
cand = session.create_candidate(hypothesis)
cand.run_universe()
cand.run_edge_testing()
cand.run_narrowing()
cand.mark_ready()

# 4. Commit (irreversible, breaks seal)
result = commit_to_holdout(cand, success_criteria_text="SPA p < 0.15")
print(f"Final equity: ${result.final_equity:,.2f}")
```

## Adding a New Experiment

Create a file in `quant_lib/experiments/` (auto-discovered on import):

```python
# quant_lib/experiments/my_strategy.py
from quant_lib.audit import for_vol_compression
from quant_lib.experiments import (
    PeriodConfig, UniverseConfig, from_hypothesis, register,
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
        # holdout_start/end: auto = [train_end - 6mo, train_end]
    ),
    universe=UniverseConfig(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        min_volume_usdt=50_000_000,
        min_age_days=180,
    ),
))
```

Now your experiment is registered. Run it with `quant_exp explore
my_strategy` or `from quant_lib import run_explore;
run_explore("my_strategy")`.

## Architecture

sealedge is organized in layers. Dependencies only point downward
(higher layers may import lower ones, never the reverse):

```
┌────────────────────────────────────────────────────────────┐
│  quant_lib/experiments/                                     │
│                    User-defined experiment configs          │
│                    (StrategyConfig for per-experiment risk)  │
├────────────────────────────────────────────────────────────┤
│  cli/              quant_exp CLI (list, show, explore,      │
│                    commit, status)                          │
├────────────────────────────────────────────────────────────┤
│  research/         ResearchSession (white-box iterative) +  │
│                    commit_to_holdout (black-box)             │
├────────────────────────────────────────────────────────────┤
│  audit/            Integrity primitives (Hypothesis, seal,   │
│                    ExperimentLog)                             │
├────────────────────────────────────────────────────────────┤
│  tools/            Public API (fetch_klines, walk_forward,   │
│                    spa_test, simulate_portfolio, …)          │
├────────────────────────────────────────────────────────────┤
│  core/             Private implementation (JIT engine, WFA,  │
│                    features, SPA, portfolio, metrics,         │
│                    risk_allocation, …)                        │
├────────────────────────────────────────────────────────────┤
│  utils/            Shared (config, git, logging)             │
└────────────────────────────────────────────────────────────┘
```

### Layer responsibilities

| Layer | Purpose |
|---|---|
| `quant_lib/experiments/` | User-defined experiment configurations (Python files). `StrategyConfig` dataclass enables per-experiment risk-allocation overrides. |
| `cli/` | User-facing CLI (the `quant_exp` command) |
| `research/` | High-level white-box + black-box research workflow. `Candidate` consumes `StrategyConfig` to drive per-fold PF-weighted risk allocation. |
| `audit/` | Integrity primitives (immutable hypothesis, sealed holdout) |
| `tools/` | Composable building blocks for custom pipelines |
| `core/` | JIT-compiled engine, WFA, features, SPA, portfolio, metrics. `core/_risk_allocation.py` houses the canonical per-fold PF risk-rebalancing orchestrator. |
| `utils/` | Shared utilities |

## Methodology

See [docs/methodology.md](docs/methodology.md) for the full
methodology writeup. Highlights:

- **Holdout seal (C-2)**: SHA256 of holdout data at session
  creation, verified at commit time.
- **Purge days**: 30-90 day gap between IS and OOS to prevent
  boundary feature contamination.
- **Best-params selection (Q1)**: per symbol, pick fold with
  highest PSR across all WFA folds (consistent with live trading).
- **Risk allocation**: per-fold decay-weighted PF (halflife 2 folds) +
  clamp [0.5, 1.5] + rescale to preserve total risk. Implemented in
  `core/_risk_allocation.py` and called from `Candidate.run_edge_testing`.
  Tunable per experiment via `StrategyConfig(pf_weight_clamp_floor=..., ...)`.

## Testing

```bash
# Run all 1324 tests (~150s)
make test

# Run fast tests only (skips @pytest.mark.slow)
make test-fast

# With coverage and branch coverage report
make test-cov
# HTML report at htmlcov/index.html

# Lint + type-check (ruff + mypy)
make lint
```

**Test categories** (1324 total):
- Unit tests (per-function)
- Integration tests (component interaction)
- Property-based tests (Hypothesis, ~22 invariants)
- Reproducibility tests (same config + same seed = same output)
- Config validation tests (invalid configs caught at construction)
- CLI smoke tests (subprocess invocation)
- Per-experiment `StrategyConfig` wiring tests
- Per-fold PF-weighted risk allocation tests (`core/_risk_allocation.py`)
- Chaos / fault-injection tests
- Holdout-seal migration tests (`tests/test_migrate_seals.py`)

**Environment variables for testing:**

| Variable | Purpose |
|---|---|
| `QUANT_LIB_HMAC_SECRET` | HMAC key for holdout seals (required, min 32 chars). |
| `QUANT_LIB_SEAL_DIR` | Override the seal directory (default `data_cache/holdout_seals`). |
| `HQS_KEEP_SEAL_DIR=1` | Prevent cleanup of per-process seal temp dir after test session (useful for debugging). |
| `OFFLINE=1` | Skip network tests (marked `@pytest.mark.network`). |

## Project Structure

```
quant_lib/
├── audit/              # Integrity primitives
├── cli/                # quant_exp CLI
├── core/               # Private implementation (JIT engine)
├── research/           # ResearchSession, Candidate, commit
├── experiments/        # (within quant_lib/) User-defined experiment configs
├── tests/              # 1324 tests
├── tools/              # Public composable API
├── utils/              # Shared utilities
├── docs/
│   └── methodology.md  # Paper-grade methodology writeup
├── .github/
│   └── workflows/      # CI: tests.yml + lint.yml + mutation.yml
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE              # GPL-3.0-or-later
├── Makefile
├── pyproject.toml
└── README.md
```

## Continuous Integration

Three GitHub Actions workflows under `.github/workflows/`:

- **`tests.yml`** (PR + push, ~3 min): Runs `pytest -n auto --cov=quant_lib --cov-branch`
  on a matrix of Python 3.10/3.11/3.12/3.13 × Ubuntu/Windows. Network tests
  are skipped via `OFFLINE=1`. Codecov upload on the canonical (Ubuntu, 3.12)
  entry; the Windows coverage.xml is uploaded as a downloadable artifact for
  cross-platform inspection.
- **`lint.yml`** (PR + push, ~30s quick gate + ~2 min full):
  - `quick` job: fast `ruff check` for PR-opened feedback
  - `ruff` job: full `ruff check` + `ruff format --check` on Linux/Mac/Windows
  - `mypy` job: type-check on Python 3.10/3.11/3.12
- **`mutation.yml`** (weekly Monday 03:00 UTC + manual trigger): Runs
  `mutmut run --max-children 4` on the F16-scoped mutation set
  (`quant_lib/research/candidate.py` + `commit.py` per `pyproject.toml`).
  Results are uploaded as artifacts; the scheduled run catches mutation-score
  drift before release.

Run the same checks locally:

```bash
make lint          # ruff + mypy
make test-cov      # full test suite with branch coverage
make mutate        # mutation testing (slow, ~10-30 min)
```

## Documentation

- **[docs/methodology.md](docs/methodology.md)**, Paper-grade
  writeup (formulas, references, justifications).
- **[CHANGELOG.md](CHANGELOG.md)**, Version history.

## Status

This is a research-quality framework. Critical paths have high test
coverage (1324 tests, including property-based, reproducibility, and
config validation). Branch coverage is tracked via `--cov-branch`
(enforced in `make test-cov`). The core engine (`core/_engine.py`)
is exercised by integration tests but the JIT-compiled body is
opaque to coverage tooling.

**Known limitations:**
- Engine line coverage is not measurable (Numba @njit compiles to
  native code). Test behaviours, not lines.
- `core/_data.py` requires network access to Binance Vision
  (mocked in tests)

## Installation

```bash
# Clone the repo
git clone https://github.com/Hansiongs/sealedge.git
cd sealedge

# Install with dev dependencies
make install-dev

# Or with pip
pip install -e ".[dev]"
```

## Citation

If you use sealedge in a paper, please cite it as:

```bibtex
@software{sealedge,
  author = {Winarto, Hansen},
  title = {sealedge: Sealed-holdout backtesting toolkit for quantitative trading strategies},
  version = {0.5.1},
  year = {2026},
  url = {https://github.com/Hansiongs/sealedge},
  license = {GPL-3.0-or-later}
}
```

Or use the [`CITATION.cff`](CITATION.cff) file for automatic
citation generation.

## License

GPL-3.0-or-later, see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup and prerequisites
- Code style guide (ruff, mypy, naming conventions)
- Testing guidelines and risk areas
- How to add new experiments and strategies
- Release checklist and pull-request process

## Acknowledgments

- Bailey & Lopez de Prado for the PSR framework
- Hansen for the SPA test
- Benjamini & Hochberg for the FDR correction
- The Optuna team for the hyperparameter optimization framework
