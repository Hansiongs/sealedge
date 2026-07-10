# sealedge -- replication materials (Phase 2)

This directory contains the **replication materials** for the sealedge JSS
submission. Reviewers should be able to reproduce all paper-claim results
by running **one command**.

## Contents

| File | Purpose |
|------|---------|
| `scripts/reproduce.py` | Single canonical reproduction script. Runs all 3 strategies with paper-grade `n_spa_iters=2000`. |
| `output/` | Default output directory. Per-run subdirs hold `results.json` (machine-readable) + `results.md` (human-readable). |
| `README.md` | This file. |

## Quick start

```bash
# Full paper-grade reproduction (~2.5 h total on a single machine)
python scripts/reproduce.py

# Smoke-test single strategy first (~50 min) before running full pipeline
python scripts/reproduce.py --strategies vol_compression_v1

# Or via Makefile
make reproduce          # all 3 strategies, n_spa_iters=2000
make reproduce-one EXP=vol_compression_v1   # single strategy smoke test
```

Each run writes:

* `output/results.json` -- per-strategy metrics, platform metadata, git
  commit, seed values, dependency versions
* `output/results.md` -- human-readable summary table for reviewer
  cross-check

## Runtime expectations

| Run | Strategy | Time |
|-----|----------|------|
| Single strategy | any of the 3 | ~50 min on a single machine |
| Full pipeline | all 3 | ~2.5 h on a single machine |

These are paper-grade runtimes (Monte Carlo SPA at `n_spa_iters=2000`).
Reviewer time investment is documented honestly in JSS submission.
The single-strategy smoke test is the recommended first check before
committing to the full pipeline.

All results in `output/results.json` are deterministic given the seed
and dependency versions. Run the script on a different machine with the
same `pip install -e ".[dev]"` and you should get identical numbers.

## Platform dependencies

```
numpy>=1.24
pandas>=2.0
scipy>=1.10
numba>=0.58
optuna>=3.0
```

Plus dev dependencies for testing:

```
pytest>=7
pytest-xdist
```

Lock the exact versions in your environment with:

```bash
pip install -e ".[dev]"
pip freeze > pip_freeze.txt
```

Attach `pip_freeze.txt` to the JSS submission alongside the source code.

## Data prerequisites

The reproduction script needs **`data_cache/BTCUSDT_1h_MASTER.csv`**
and similar files for `ETHUSDT` and `SOLUSDT`. These CSVs are
git-ignored (they're 10 MB total and reviewer machines may have
incompatible network access).

**The script does NOT fetch data automatically.** Pre-cache manually:

```python
from quant_lib.core._data import fetch_with_retry
fetch_with_retry("BTCUSDT", "1h", "2019-10-01", "2025-12-31")
fetch_with_retry("ETHUSDT", "1h", "2019-10-01", "2025-12-31")
fetch_with_retry("SOLUSDT", "1h", "2019-10-01", "2025-12-31")  # SOL perp data starts ~2020-09
```

The cached files are then placed in `data_cache/` (or wherever `--cache-dir`
points).

### Why data starts at 2019-10-01

Each registered strategy has `train_start="2020-01-01"`. The framework's
universe filter (`quant_lib.tools.universe.select_universe`) requires a
**90-day volume lookback window** ending at `train_start`. Without
cached data starting 3 months earlier, every symbol fails the
filter and `run_explore` raises `CandidateError`. The reproduction
script catches this upfront and emits a clear error message.

**Workarounds** if you cannot pre-cache 2019-10-01 data:
- Change `train_start` in `quant_lib/experiments/*.py` to a later date
- Modify the experiment's `PeriodConfig` and re-register

## What the script does NOT do

* **Holdout commit** -- the script runs `run_explore` only. The
  holdout seal is preserved for the full paper submission. To verify
  holdout performance, use `python -m quant_lib run_commit <strategy>`
  separately (this irreversibly breaks the seal).
* **Plot generation** -- the paper's figures will be generated
  separately in Phase 3 (manuscript drafting). All numeric metrics
  needed for figures are captured in `results.json`.

## Reviewer checklist

1. Install dependencies: `pip install -e ".[dev]"`
2. Pre-cache data (see "Data prerequisites" above)
3. Set `QUANT_LIB_HMAC_SECRET` (or let the script auto-set it)
4. Run `python scripts/reproduce.py`
5. Verify `output/results.json` shows all 3 strategies with `status: success`
6. Verify `output/results.md` matches the paper's Table X

If any strategy fails, the script exits with code 2 (after writing
partial output). Inspect `results.json` for the per-strategy error
message.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `RuntimeError: HMAC seal secret is not configured` | `QUANT_LIB_HMAC_SECRET` unset | The script auto-sets a default; if running directly, export the env var to any 32+ char string |
| `CandidateError: No symbols passed universe selection` | Cached data starts too recently (see "Data prerequisites") | Pre-cache older data, or update `train_start` in experiment file |
| `ImportError: cannot import name '_imaging' from 'PIL'` | matplotlib/PIL version mismatch | Reinstall: `pip install --upgrade pillow matplotlib` |
| Slow runtime (>1 hour) | Expected for full pipeline on slow hardware | Smoke-test one strategy first: `python scripts/reproduce.py --strategies vol_compression_v1` (same `n_spa_iters=2000`) |

