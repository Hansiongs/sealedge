# Test Fixtures

Real data files used by integration tests.

## Files

### `btcusdt_1h_2024_jan.csv`
- Source: Binance public API (`/api/v3/klines`)
- Symbol: BTCUSDT
- Interval: 1h
- Period: 2024-01-18 to 2024-02-01 (14 days, ~336 rows)
- Size: ~14KB
- Purpose: Integration smoke test that pipeline works with real OHLCV data
- Status: Synthetic replica (real data requires network access)

## Regenerate

```bash
python tools/download_fixture.py
```

## Why Frozen?
- Deterministic (no flake)
- Fast (no network at test time)
- Small (~14KB) — safe to commit
- Validates schema is stable
