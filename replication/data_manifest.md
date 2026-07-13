# Data cache manifest (paper-grade)

OHLCV/funding CSVs live under repo-root `data_cache/` and are **gitignored**.
This file documents the **minimum layout** for `scripts/reproduce.py` and a
**local integrity snapshot** from the author machine (not a public data deposit).

## Required files

| File | Role |
|------|------|
| `{SYM}_1h_MASTER.csv` | 1h OHLCV for BTCUSDT, ETHUSDT, SOLUSDT |
| `{SYM}_FUNDING_MASTER.csv` | funding rates (required for `funding_rate_carry`) |

Paper experiments use `train_start=2021-07-01`, `min_age_days=180`, and a
90-day volume lookback. Preflight therefore requires history back to at least
**2021-01-02** (180 days before train start); earlier is safer.

## How to obtain (fail-loud if missing)

```python
from quant_lib.tools.data import fetch_klines, fetch_funding

for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
    # BTC/ETH: start early enough for age + lookback; SOL perp ~2021-01
    start = "2020-01-01" if sym != "SOLUSDT" else "2021-01-01"
    fetch_klines(sym, "1h", start, "2025-12-31")
    fetch_funding(sym, "2021-01-01", "2025-12-31")
```

`scripts/reproduce.py` does **not** call these helpers. It only checks that
files exist and cover the lookback window, then exits non-zero with a concrete
message if not.

## Author-machine snapshot (optional cross-check)

Generated from a local `data_cache/` used for the paper sample. Reviewers with
a different exchange dump need not match SHA; they must match **coverage**.

| File | rows | min date | max date | sha256[:16] |
|------|------|----------|----------|-------------|
| `BTCUSDT_1h_MASTER.csv` | 52585 | 2020-01-01 | 2025-12-31 | `3b3fe9a0c6139d34` |
| `ETHUSDT_1h_MASTER.csv` | 52585 | 2020-01-01 | 2025-12-31 | `93043c70a800d410` |
| `SOLUSDT_1h_MASTER.csv` | 43681 | 2021-01-01 | 2025-12-31 | `8bacdbbaaa4fc697` |
| `BTCUSDT_FUNDING_MASTER.csv` | 5206 | 2021-04-01 | 2025-12-31 | `997783c147a00253` |
| `ETHUSDT_FUNDING_MASTER.csv` | 5929 | 2021-01-01 | 2026-05-31 | `7afcc902b2def3d0` |
| `SOLUSDT_FUNDING_MASTER.csv` | 5008 | 2021-07-01 | 2025-12-31 | `da3c188b40bcea2d` |

## Preflight command

```bash
python scripts/reproduce.py --strategies vol_compression_v1
# exits non-zero immediately if cache/coverage fails (before long SPA)
```

