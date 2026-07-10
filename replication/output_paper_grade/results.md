# sealedge reproduction -- results summary

Captured: `2026-07-10T03:14:59.939036+00:00`  
Git commit: `1e85e1d555cb36aaad72bde70a4317ca1e65f699`  
Global seed: **42**  
quant_lib version: `0.5.1`  
Python: `3.14.4` on `Windows` (AMD64)

## Strategy results

| Strategy | Status | PSR | SPA p-value | n OOS trades | Final equity | Elapsed (s) |
|----------|--------|-----|-------------|-------------|--------------|-------------|
| `vol_compression_v1` | ✅ success | — | 1.0000 | 438 | 892.8403 | 690.6 |
| `pullback_sniper_rsi` | ✅ success | — | 0.9300 | 180 | 805.8595 | 345.32 |
| `funding_rate_carry` | ✅ success | — | 0.0005 | 3446 | 503.9475 | 2138.06 |

## Reviewer notes

* This script runs **explore** phase only. The holdout seal is NOT broken. To verify holdout performance, use `python -m quant_lib run_commit <strategy>` separately.

* All numbers in `results.json` are deterministic given the global seed (42) and dependency versions.

* Paper-grade default is `n_spa_iters=2000`. For a shorter smoke test with the same SPA precision, run a single strategy: `python scripts/reproduce.py --strategies vol_compression_v1` (or `make reproduce-one EXP=vol_compression_v1`).
