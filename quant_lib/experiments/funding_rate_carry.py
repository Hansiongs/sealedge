"""funding_rate_carry: perp funding rate mean-reversion carry strategy.

Entry: funding_pct_rank > entry_thresh -> SHORT
       funding_pct_rank < (1 - entry_thresh) -> LONG
Exit:  funding reverts to neutral zone [exit_low, exit_high] OR
       trailing stop OR bailout
"""
from quant_lib.audit import for_funding_rate_carry
from quant_lib.experiments import (
    PeriodConfig,
    UniverseConfig,
    from_hypothesis,
    register,
)


_HYP = for_funding_rate_carry(
    name="funding_rate_carry",
    mechanism=(
        "Perp funding rates mean-revert to neutral. Shorting when "
        "funding > P90 captures the carry as longs pay shorts; "
        "mirroring when funding < P10 captures the inverse carry. "
        "The edge is the structural mean-reversion of funding rates "
        "to equilibrium, not directional price prediction."
    ),
    boundary_conditions=(
        "Fails during structural funding regimes (e.g., sustained "
        "bull market with persistently positive funding). Fails on "
        "illiquid pairs where funding rate signal is sparse or noisy. "
        "Fails during market stress when funding spikes are not "
        "mean-reverting but persistent."
    ),
    success_criteria="SPA p < 0.15, PF > 1.2, min 30 trades, MaxDD < 35%",
)


register(from_hypothesis(
    name="funding_rate_carry",
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
