"""vol_compression_v1: baseline vol compression breakout strategy.

Entry: vol_pct_rank < 0.20 + close breaks HH_20/LL_20 + pullback confirmation.
Exit: trailing stop at ATR * trail_atr, bailout at 36 bars.
"""
from quant_lib.audit import for_vol_compression
from quant_lib.experiments import (
    PeriodConfig,
    UniverseConfig,
    from_hypothesis,
    register,
)


_HYP = for_vol_compression(
    name="vol_compression_v1",
    mechanism=(
        "Volatility compression (vol_pct_rank < 0.20) followed by volume "
        "breakout with pullback entry generates momentum in liquid "
        "crypto futures"
    ),
    boundary_conditions=(
        "Fails in strong trend without pullback. Also fails for "
        "low-volume coins with noisy breakout signals"
    ),
    success_criteria="SPA p < 0.15, PF > 1.3, min 30 trades, MaxDD < 40%",
)


register(from_hypothesis(
    name="vol_compression_v1",
    hypothesis=_HYP,
    period=PeriodConfig(
        # SOLUSDT perp data on local cache starts 2021-01-01. Two
        # filter requirements push train_start to 2021-07-01:
        #   (1) universe filter needs 90-day volume lookback
        #   (2) universe filter needs 180-day min_age_days
        # BTC/ETH data starts 2020-01-01 but we use the same
        # train_start for consistency across strategies. Costs ~6
        # months of BTC/ETH training data; gains cross-strategy
        # comparability.
        train_start="2021-07-01",
        train_end="2025-12-31",
        # holdout auto-resolves POST-training to [2026-01-01, 2026-07-01]
        # (6-month default; uses train_end + 1d convention for no-peek guarantee)
        holdout_months=6,
    ),
    universe=UniverseConfig(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    ),
))
