"""
Configuration -- STATIC dict, DEFAULTS dict, GLOBAL_SEED, WARMUP_BARS.

Two-layer config after Phase 4 refactor:

- ``STATIC``: Infrastructure-level constants. Exchange costs, test
  parameters, WFA purge (depends on feature lookback), feature windows.
  These rarely change and are framework-level.

- ``DEFAULTS``: Per-experiment defaults that mirror
  ``quant_lib.experiments.base.StrategyConfig``. Used internally by
  runner, wfa, spa, etc. when no Candidate is available. Per-experiment
  overrides happen via ``StrategyConfig`` in experiment files.

Removed from STATIC (Phase 4):
- ``asset_risk_weights``: redundant; per-fold PF-weighted allocation
  in ``core/_risk_allocation.py`` now sets per-trade ``risk_weight``
  directly (no static asset-level weighting needed).
- ``asset_baseline_trades``: per-experiment choice; pass via
  ``StrategyConfig.expected_trades_per_year``.
- ``global_position_limit``: per-experiment choice; pass via
  ``StrategyConfig.global_position_limit``.
"""

from typing import Any

GLOBAL_SEED: int = 42

# ════════════════════════════════════════════════════════════════════════
# STATIC -- Infrastructure constants (rarely change, framework-level)
# ════════════════════════════════════════════════════════════════════════

STATIC: dict[str, Any] = {
    # Feature windows
    "atr_len": 20,                          # ATR rolling window (bars)
    # Exchange costs
    "fee_taker": 0.05,                      # taker fee fraction
    "maintenance_margin_pct": 0.01,         # maintenance margin
    "liquidation_fee_pct": 0.005,           # liquidation fee
    "liquidation_slippage_pct": 0.02,       # extra slip on liquidation
    # Test infrastructure
    "bootstrap_n_sim": 2000,                # bootstrap iterations
    "bootstrap_block_size_min": 20,         # min block size for bootstrap
    "bootstrap_block_size_max": 120,        # max block size
    "spa_n_iters": 2000,                    # SPA permutation iterations
    "spa_equity_warn_threshold_usd": 10.0,  # SPA transparency check
    # WFA -- purge days depend on feature lookback (infrastructure)
    # Purge ≥ max lookback window vol_pct_rank (720 bar 1H = 30 days)
    # to prevent parameter contamination from IS leaking into OOS.
    # NOTE: btc_ema_4800 (span=4800, ~69-day half-life) is not fully
    # converged after 90-day purge (~87% real-data weight). macro_trend
    # is binary (1/-1) and tolerates residual contamination well.
    "wfa_purge_days": 90,                   # MAXIMUM; adaptive via _get_purge_days
}

# NOTE: purge days are now ADAPTIVE via _get_purge_days() -- shrinks as IS grows
# because contamination from boundary artifacts naturally dilutes.
# wfa_purge_days = 90 is MAXIMUM (IS ≤ 15 months), minimum 30 days to
# cover vol_pct_rank rolling window (720 bar 1H).
# Phase 4 (v0.5.0): replaced `assert` with `if/raise AssertionError`
# so the invariant check runs even under `python -O` (where `assert`
# statements are stripped).
if STATIC["wfa_purge_days"] < 30:
    raise AssertionError(
        f"wfa_purge_days ({STATIC['wfa_purge_days']}) must be >= 30 "
        f"(minimum for vol_pct_rank rolling window: 720 bar = 30d max lookback)."
    )

# ════════════════════════════════════════════════════════════════════════
# DEFAULTS -- Per-experiment defaults (mirror StrategyConfig)
# ════════════════════════════════════════════════════════════════════════
# Used internally by runner, wfa, spa, etc. when no Candidate is
# available. Per-experiment overrides go through StrategyConfig in
# ``quant_lib/experiments/base.py``. KEEP IN SYNC with StrategyConfig
# defaults.

DEFAULTS: dict[str, Any] = {
    # Capital
    "initial_capital": 1000.0,
    "leverage": 3.0,
    "global_position_limit": 4,             # max concurrent positions
    # Engine parameters
    "bailout_bars": 36,
    "weekend_liquidity_penalty": 2.0,
    "stress_test_multiplier": 2.0,          # reduced from 2.5 in 0.2.0
    "fixed_rvol_thresh": 2.5,
    "cb_hard_cooldown_hours": 24,
    "fixed_cb_threshold": 0.15,
    # Regularization
    "reg_lambda": 0.05,
    # Trend alignment
    "trend_aligned_risk_mult": 1.5,
    "trend_counter_risk_mult": 0.5,
    # WFA
    "wfa_min_train_months": 12,
    "wfa_decay_halflife_months": 15,
    "wfa_test_months": 3,
    "wfa_trials_per_fold": 80,
    # PSR: minimum trade-weight floor (fraction of 1.0). When the
    # weights post-normalisation are below this threshold, the
    # effective sample size (Kish ESS) approaches 1 and the PSR
    # becomes unreliable. The weighted-PSR code path in
    # core/_testing.py logs a warning when any weight is below this
    # floor, and the ESS < 2.0 guard ultimately catches the worst
    # cases.
    "psr_weight_floor": 0.001,
    # PF-based risk allocation
    "pf_weight_clamp_floor": 0.5,
    "pf_weight_clamp_ceiling": 1.5,
    "pf_decay_halflife_folds": 2,
    "pf_min_trades_for_weight": 10,
    # Optuna search spaces
    "search_space": {
        "vol_pct_thresh": (0.10, 0.40),
        "pullback_bars": (3, 8),
        "trail_atr": (1.5, 5.0),
        "sl_mult": (1.0, 3.0),
    },
    "search_space_pb": {
        "vol_pct_thresh": (0.10, 0.40),  # unused but kept for compat
        "pullback_bars": (3, 8),         # unused but kept for compat
        "trail_atr": (1.5, 5.0),
        "sl_mult": (1.0, 3.0),
        "rsi_oversold": (25, 35),
        "rsi_overbought": (65, 75),
    },
    # Default per-pair risk weight (used in vol-adjusted weights)
    "default_risk_per_pair": 0.01,
    # Default expected trades/year (used when per-experiment override
    # not provided). 30 is a reasonable conservative default.
    "default_expected_trades_per_year": 30,
    # Phase 2.4: Market impact cap. Position size is capped at this
    # fraction of 24h volume. Prevents the trend multiplier (1.5x
    # with-trend) from creating unrealistically large orders on
    # illiquid assets where live fill price would move against the
    # order. 1% is a conservative cap; production trading may use
    # 0.1-0.5% for very liquid pairs.
    "market_impact_volume_pct": 0.01,
}

# WARMUP_BARS -- safe ceiling for all feature lookback windows.
# vol_pct_rank uses realized_vol_24 (window=24, shift=1) on a 720-bar rolling base = 745.
# Rounded to 750 for safety. Bars < WARMUP_BARS retain NaN features and are dropped by
# dropna(subset=critical_cols), so they never reach the trade loop.
WARMUP_BARS = 750
