# sealedge -- replication materials (Phase 2)

Replication materials for the sealedge JSS paper. Reviewers should be
able to reproduce the paper-claim numbers with one command.

## Contents

| File | Purpose |
|------|---------|
| `scripts/reproduce.py` | SPA explore reproduction. Runs all 3 strategies with paper-grade `n_spa_iters=2000`. |
| `scripts/reproduce_seal_demo.py` | Claim 1 micro-demo: synthetic HMAC seal → verify → tamper fail → one-shot break (seconds; no market data). |
| `output/` | Default output directory for new SPA runs (`results.json` + `results.md`). Gitignored. |
| `output_paper_grade/` | Committed SPA explore sample (seed 42, `n_spa_iters=2000`) for cross-check. |
| `output_seal_demo/` | Committed seal micro-demo sample (`results.json` + `results.md`). |
| `README.md` | This file. |

## Quick start

```bash
# Full paper-grade reproduction (~1 h class on a typical PC; ~53 min measured on Windows AMD64)
python scripts/reproduce.py

# Smoke-test single strategy first (same n_spa_iters=2000; often ~6, 35 min depending on strategy)
python scripts/reproduce.py --strategies vol_compression_v1

# Or via Makefile
make reproduce          # all 3 strategies, n_spa_iters=2000
make reproduce-one EXP=vol_compression_v1   # single strategy smoke test
make reproduce-seal     # Claim 1 synthetic seal micro-demo (~seconds)
```

SPA explore runs write (default `--output-dir replication/output`):

* `results.json` -- per-strategy metrics, platform metadata, git
  commit, seed values, dependency versions
* `results.md` -- human-readable summary table for reviewer
  cross-check

Committed SPA reference sample: `replication/output_paper_grade/`.
Table SPA column is Hansen-path `spa_p_value`. In that sample,
`spa_naive_p_value` is `NaN` for all three strategies (legacy anchor-span
guard ≥80% of calendar) — intentional sentinel, not a failed run.

Seal micro-demo (Claim 1) writes `replication/output_seal_demo/` and does
**not** open registered-experiment holdouts or touch `data_cache/`.

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
~1, 2 h. There is **no** reduced-`n_spa_iters` fast script: smoke tests
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

See also **`data_manifest.md`** (required filenames, coverage rules, optional
local SHA snapshot).

Paper strategies need, under `data_cache/` (or `--cache-dir`):

| Pattern | Symbols | Used by |
|---------|---------|---------|
| `{SYM}_1h_MASTER.csv` | BTCUSDT, ETHUSDT, SOLUSDT | all three SPA strategies |
| `{SYM}_FUNDING_MASTER.csv` | same | `funding_rate_carry` |

CSVs are **gitignored** (multi-MB; network access varies by reviewer).

**The SPA script does NOT fetch data.** Pre-cache with the public helpers:

```python
from quant_lib.tools.data import fetch_klines, fetch_funding

for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
    start = "2020-01-01" if sym != "SOLUSDT" else "2021-01-01"
    fetch_klines(sym, "1h", start, "2025-12-31")
    fetch_funding(sym, "2021-01-01", "2025-12-31")
```

### Coverage vs `train_start=2021-07-01`

Experiments use **`train_start="2021-07-01"`**, **`min_age_days=180`**, and a
**90-day** volume lookback. Preflight in `scripts/reproduce.py` requires 1h
history back to at least **~2021-01-02** (stricter of age floor vs lookback).
Earlier starts (e.g. BTC/ETH from 2020-01-01) are safer. Without coverage,
preflight **exits non-zero before SPA** with the missing file/date named.

Do **not** change `train_start` to paper-match numbers if the cache is short —
extend the cache instead.

### Funding

`funding_rate_carry` also needs funding masters. Missing
`{SYM}_FUNDING_MASTER.csv` fails preflight when that strategy is selected.

## What the SPA script does NOT do

* **Holdout commit on registered experiments** -- `reproduce.py` runs
  `run_explore` only. Registered-experiment seals stay closed. Full
  strategy commit is `run_commit` (irreversible); do not use that path
  for paper-grade SPA cross-check.
* **Claim 1 seal proof** -- use `python scripts/reproduce_seal_demo.py`
  (or `make reproduce-seal`) for synthetic seal/verify/tamper/break.
* **Plot generation** -- numeric metrics for tables live in
  `results.json`; figures are optional for reviewers.

## Reviewer checklist

1. Install dependencies: `pip install -e ".[dev]"`
2. Pre-cache data (see "Data prerequisites" + `data_manifest.md`) — SPA path only
3. Set `QUANT_LIB_HMAC_SECRET` (or let scripts auto-set a demo secret)
4. Run `python scripts/reproduce_seal_demo.py` and confirm
   `output_seal_demo/results.json` has `"ok": true` (all steps pass)
5. Run `python scripts/reproduce.py`
6. Verify `output/results.json` shows all 3 strategies with `status: success`
7. Cross-check SPA table cells against `output_paper_grade/` (Hansen
   `spa_p_value`; legacy naive may be NaN under the span guard)

If any SPA strategy fails, the script exits with code 2 (after writing
partial output). Inspect `results.json` for the per-strategy error
message.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `RuntimeError: HMAC seal secret is not configured` | `QUANT_LIB_HMAC_SECRET` unset | The script auto-sets a default; if running directly, export the env var to any 32+ char string |
| `CandidateError: No symbols passed universe selection` | Cached data starts too recently (see "Data prerequisites") | Pre-cache earlier 1h/funding masters (see Data prerequisites); avoid changing paper train_start |
| `ImportError: cannot import name '_imaging' from 'PIL'` | matplotlib/PIL version mismatch | Reinstall: `pip install --upgrade pillow matplotlib` |
| Slow runtime (>1 hour) | Expected for full pipeline on slow hardware | Smoke-test one strategy first: `python scripts/reproduce.py --strategies vol_compression_v1` (same `n_spa_iters=2000`) |

