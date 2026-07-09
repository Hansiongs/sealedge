"""Investigate why funding_rate_carry produces 3446 trades vs 438/180.

Counts actual signal fires per bar for each strategy type, then
compares to expected from threshold configuration.
"""
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault('QUANT_LIB_HMAC_SECRET', 'sealedge-jss-reproduction-32chars-min')

sys.path.insert(0, r'C:\Users\Administrator\Desktop\sealedge')

ROOT = r'C:\Users\Administrator\Desktop\sealedge'

# Use BTCUSDT as representative
df_raw = pd.read_csv(f'{ROOT}/data_cache/BTCUSDT_1h_MASTER.csv')
df_raw['time'] = pd.to_datetime(df_raw['time'])

# Use minimal funding for funding_rate_carry (test data is realistic)
fund_df = pd.read_csv(f'{ROOT}/data_cache/BTCUSDT_FUNDING_MASTER.csv')
fund_df['time'] = pd.to_datetime(fund_df['time'])

# Compute funding_pct_rank the way the engine does
_FUNDING_PCT_WINDOW = 720
funding_pct_rank_raw = (
    fund_df['funding_rate']
    .rolling(_FUNDING_PCT_WINDOW, min_periods=24)
    .rank(pct=True)
)
fund_df['funding_pct_rank'] = funding_pct_rank_raw.shift(1).astype(np.float32)

# Join funding_pct_rank into hourly bars (forward-fill since rank updates only at funding events)
df_with_funding = df_raw.merge(
    fund_df[['time', 'funding_pct_rank']],
    on='time', how='left'
)
df_with_funding['funding_pct_rank'] = df_with_funding['funding_pct_rank'].ffill()
df_with_funding['funding_pct_rank'] = df_with_funding['funding_pct_rank'].fillna(0.5)

# Filter to WFA training period (same as experiments): 2021-07-01 -> 2025-12-31
train_start = pd.Timestamp('2021-07-01')
train_end = pd.Timestamp('2025-12-31')
df_train = df_with_funding[
    (df_with_funding['time'] >= train_start) &
    (df_with_funding['time'] <= train_end)
].copy()
print(f'Training bars (BTCUSDT, 2021-07-01 -> 2025-12-31): {len(df_train)}')

# Count signals per strategy type
funding_pct_rank = df_train['funding_pct_rank'].values

# Default entry thresholds from factory
ENTRY_LOWER_DEFAULT = 0.85  # conservative end of (0.85, 0.95)
ENTRY_UPPER_DEFAULT = 0.95

# Count signals for various entry thresholds
print('\n=== funding_rate_carry signal counts (per bar) ===')
for entry_thresh in [0.85, 0.88, 0.90, 0.92, 0.95]:
    long_signal = funding_pct_rank < (1 - entry_thresh)
    short_signal = funding_pct_rank > entry_thresh
    total_signals = (long_signal | short_signal).sum()
    pct = total_signals / len(funding_pct_rank) * 100
    print(f'  entry_thresh={entry_thresh}: '
          f'long={long_signal.sum():>6}, short={short_signal.sum():>6}, '
          f'total={total_signals:>6} ({pct:.1f}% of bars)')

# Estimate OOS trades (assume WFA produces ~12 folds with ~80 trials each
# per Optuna search). Trial produces one entry when signal fires.
print('\n=== Estimated trades (WFA: ~12 folds * 80 trials = 960 trial runs) ===')
for entry_thresh in [0.85, 0.88, 0.90, 0.92, 0.95]:
    long_signal = funding_pct_rank < (1 - entry_thresh)
    short_signal = funding_pct_rank > entry_thresh
    avg_signal_pct = ((long_signal | short_signal).sum() / len(funding_pct_rank))
    # WFA produces ~12 folds. Per fold, Optuna runs 80 trials.
    # Each trial uses the same signal series. With neutral-zone exit
    # (P40-P70) most trades close within 1-3 days. Average trades per
    # fold = (signal_fires * fold_duration / mean_hold_bars).
    # Rough estimate: signal_fraction * total_bars / mean_hold_bars.
    # Conservative mean_hold = 12 bars (entry + neutral-zone exit).
    mean_hold = 12
    fold_bars = 90 * 24  # 90-day fold
    bars_per_fold = min(fold_bars, len(funding_pct_rank) // 12)
    avg_trades_per_trial = avg_signal_pct * bars_per_fold / mean_hold
    # Trial runs 80 per fold, 12 folds => 960 runs, but each strategy has 1 trial "run" per (fold, param set).
    # Actually WFA in this framework runs N trials per fold and picks best.
    # But each fold's "best" generates OOS trades. So OOS trades ~= trades from 1 selected run per fold.
    print(f'  entry_thresh={entry_thresh}: '
          f'avg_trades/fold={avg_trades_per_trial:.1f} -> '
          f'~12 folds * {avg_trades_per_trial:.1f} = ~{12 * avg_trades_per_trial:.0f} trades')

# Now compare to what we actually got: 3446 trades across all folds/symbols
print('\n=== Observed: 3446 trades for funding_rate_carry (3 symbols) ===')
# Per-symbol estimate if fire rate ~5% of bars per fold:
# 12 folds * X trades/fold * 3 symbols = 3446
# X = 3446 / (12 * 3) = ~96 trades/fold/symbol
# Compare to vol_compression (438 total): 438 / (12*3) = ~12 trades/fold/symbol
# Ratio: 96 / 12 = 8x more
# That's a big ratio. Worth investigating.
print('  3446 / 3 symbols / 12 folds = ~96 trades/fold/symbol')
print('  vs vol_compression: 438 / 3 / 12 = ~12 trades/fold/symbol')
print('  Ratio: 8x more trades per fold')
