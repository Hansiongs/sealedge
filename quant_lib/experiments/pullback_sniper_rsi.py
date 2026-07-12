"""pullback_sniper_rsi: RSI oversold/overbought mean-reversion strategy.

Entry: RSI oversold/overbought + reversal candle.
Exit: trailing stop OR TP at hh_20/ll_20 OR bailout at 36 bars.
"""
from quant_lib.audit import for_pullback_sniper
from quant_lib.experiments import (
    PeriodConfig,
    UniverseConfig,
    from_hypothesis,
    register,
)


_HYP = for_pullback_sniper(
    name="pullback_sniper_rsi",
    mechanism=(
        "RSI oversold/overbought with reversal candle generates "
        "mean-reversion in liquid crypto futures"
    ),
    boundary_conditions="Fails in strong trends without pullback",
    success_criteria="SPA p < 0.15, PF > 1.3, min 30 trades",
)


register(from_hypothesis(
    name="pullback_sniper_rsi",
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
        # (6-month default; uses train_end + 1d convention for no-peek holdout placement)
        holdout_months=6,
    ),
    universe=UniverseConfig(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    ),
))
