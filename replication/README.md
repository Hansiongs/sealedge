# sealedge -- replication materials (Phase 2)

This directory contains the **replication materials** for the sealedge JSS
submission. Reviewers should be able to reproduce all paper-claim results
by running **one command**.

## Contents

| File | Purpose |
|------|---------|
| `scripts/reproduce.py` | Single canonical reproduction script. Runs all 3 strategies with paper-grade `n_spa_iters=2000`. |
| `output/` | Default output directory for new runs (`results.json` + `results.md`). Gitignored. |
| `output_paper_grade/` | Committed sample from a paper-grade run (seed 42, `n_spa_iters=2000`) for cross-check. |
| `README.md` | This file. |

## Quick start

```bash
# Full paper-grade reproduction (~1 h class on a typical PC; ~53 min measured on Windows AMD64)
python scripts/reproduce.py

# Smoke-test single strategy first (same n_spa_iters=2000; often ~6–35 min depending on strategy)
python scripts/reproduce.py --strategies vol_compression_v1

# Or via Makefile
make reproduce          # all 3 strategies, n_spa_iters=2000
make reproduce-one EXP=vol_compression_v1   # single strategy smoke test
```

Each run writes (default `--output-dir replication/output`):

* `results.json` -- per-strategy metrics, platform metadata, git
  commit, seed values, dependency versions
* `results.md` -- human-readable summary table for reviewer
  cross-check

Committed reference sample: `replication/output_paper_grade/`.

## Runtime expectations

Measured paper-grade run (`n_spa_iters=2000`, seed 42, Windows AMD64,
see `output_paper_grade/results.json`):

| Run | Strategy | Time (measured) |
|-----|----------|-----------------|
| Single | `pullback_sniper_rsi` | ~6 min |
| Single | `vol_compression_v1` | ~12 min |
| Single | `funding_rate_carry` | ~36 min |
| Full pipeline | all 3 | **~53 min** |

Slower hardware or cold Numba caches can push the full pipeline toward
~1–2 h. There is **no** reduced-`n_spa_iters` fast script: smoke tests
use the same SPA precision on fewer strategies. JSS “about one hour on
a regular PC” is met on the measured box; document machine class in the
manuscript if your machine differs.

All results are deterministic given the seed and dependency versions.
Re-run with the same environment to cross-check `output_paper_grade/`.

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

### Why data starts at 2019-10-01 (cache) vs train_start 2021-07-01

Paper strategies use **`train_start="2021-07-01"`** (SOLUSDT age + lookback
consistency across BTC/ETH/SOL). The framework's universe filter
(`quant_lib.tools.universe.select_universe`) still needs a **90-day
volume lookback** ending at `train_start`, so the cache must start
**before** that window. Fetching from `2019-10-01` is a safe superset
(also covers older exploratory ranges). Without enough history before
`train_start`, symbols fail the filter and `run_explore` raises
`CandidateError`. The reproduction script catches this upfront.

**Workarounds** if you cannot pre-cache early history:
- Ensure cache covers at least ~90 days before `2021-07-01` for each symbol
- Only as last resort: change `train_start` in `quant_lib/experiments/*.py`
  (that diverges from the paper sample — prefer fixing the cache)

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

