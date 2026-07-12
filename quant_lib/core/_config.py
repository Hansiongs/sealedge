"""
Config: ``STATIC`` (infra), ``DEFAULTS`` (experiment defaults), seed.

``STATIC`` holds exchange costs, SPA/WFA knobs, feature windows.
``DEFAULTS`` mirrors ``StrategyConfig`` fields for internal callers
without a Candidate. Both are TypedDict-backed mutable dicts so mypy
sees types and tests can still patch values.
"""

from typing import TypedDict

GLOBAL_SEED: int = 42

# ════════════════════════════════════════════════════════════════════════
# Strategy type constants -- SINGLE SOURCE OF TRUTH
# ════════════════════════════════════════════════════════════════════════
# Sprint 2 fix: previously triplicated across ``audit/hypothesis.py``,
# ``core/_features.py``, and ``core/_engine.py``. Now defined here once
# and imported by all consumers. Per the layering contract
# (audit -> core is allowed, never the reverse), ``core/_config.py``
# is the natural home: it's a leaf node (no quant_lib imports of its
# own) and ``audit/`` can safely import from it.
#
# The int values are part of the engine's public ABI: ``fast_trade_loop``
# in ``core/_engine.py`` takes ``strategy_type: int``. They MUST stay
# stable. To add a new strategy, append a constant here AND update the
# ``StrategyType`` IntEnum in ``audit/hypothesis.py`` AND
# ``STRATEGY_NAME_TO_INT`` in ``experiments/base.py``.
STRATEGY_VOL_COMPRESSION: int = 0
STRATEGY_PULLBACK_SNIPER: int = 1
STRATEGY_FUNDING_RATE_CARRY: int = 2


# ════════════════════════════════════════════════════════════════════════
# TypedDict schemas (Sprint 3 fix 3.1)
# ════════════════════════════════════════════════════════════════════════
# TypedDicts are dicts at runtime. The annotation only changes what
# static analyzers (mypy, pyright) see. All existing call sites like
# ``STATIC["fee_taker"]`` work unchanged -- both at runtime and at
# type-check time (mypy now knows the value is ``float``).


class StaticConfig(TypedDict):
    """Schema for ``STATIC``. Infrastructure-level constants.

    Adding a new key here is the type-safe equivalent of adding a row
    to the ``STATIC`` literal below. Both must be kept in sync.
    """

    # Feature windows
    atr_len: int
    # Exchange costs
    fee_taker: float
    maintenance_margin_pct: float
    liquidation_fee_pct: float
    liquidation_slippage_pct: float
    # Test infrastructure
    bootstrap_n_sim: int
    bootstrap_block_size_min: int
    bootstrap_block_size_max: int
    spa_n_iters: int
    spa_equity_warn_threshold_usd: float
    # SPA Hansen-literal stationary-bootstrap expected block length. 0 = use
    # the Politis-Romano default p = max(1, round(n_k ** (1/3))) per trial;
    # >0 forces a fixed expected block length (paper-disclosed calibration
    # knob). Hansen (2005) SPA null uses a
    # Politis-Romano (1994) stationary block bootstrap over IS loss-
    # differentials; this knob overrides the n_k^(1/3) heuristic.
    spa_hansen_block_length_override: int
    # WFA purge (MAXIMUM; adaptive via _get_purge_days)
    wfa_purge_days: int
    # WFA contiguity: IS/OOS folds with internal gaps larger than this
    # are skipped (covers a weekend + emergency maintenance buffer).
    # Phase 4 (v0.5.x): extracted from magic number 48h in core/_wfa.py.
    max_allowed_gap_hours: int
    # Correlation sizing: rolling correlation window for portfolio
    # correlation-aware position sizing. 180 days = 6 months of daily
    # returns, captures crypto market regime cycles. Below
    # ``correlation_min_lookback_days`` the matrix is not computed
    # (insufficient sample). Phase 4 (v0.5.x): extracted from magic
    # numbers in core/_portfolio.py.
    correlation_lookback_days: int
    correlation_min_lookback_days: int


class DefaultsConfig(TypedDict):
    """Schema for ``DEFAULTS``. Per-experiment defaults.

    Mirrors ``quant_lib.experiments.base.StrategyConfig`` -- see
    ``TestStrategyConfigStaysInSync`` in test_sprint3_fixes.py for
    the sync guard. Adding a new key here requires also adding it to
    the ``DEFAULTS`` literal below.
    """

    # Capital
    initial_capital: float
    leverage: float
    global_position_limit: int
    # Engine parameters
    bailout_bars: int
    weekend_liquidity_penalty: float
    stress_test_multiplier: float
    fixed_rvol_thresh: float
    cb_hard_cooldown_hours: int
    fixed_cb_threshold: float
    # Regularization
    reg_lambda: float
    # Trend alignment
    trend_aligned_risk_mult: float
    trend_counter_risk_mult: float
    # WFA
    wfa_min_train_months: int
    wfa_decay_halflife_months: int
    wfa_test_months: int
    wfa_trials_per_fold: int
    # PSR
    psr_weight_floor: float
    # PF-based risk allocation
    pf_weight_clamp_floor: float
    pf_weight_clamp_ceiling: float
    pf_decay_halflife_folds: int
    pf_min_trades_for_weight: int
    # WFA parameter centers (L2 regularization midpoints)
    vol_thresh_center: float
    pullback_bars_center: float
    trail_atr_center: float
    sl_mult_center: float
    rsi_oversold_center: float
    rsi_overbought_center: float
    # WFA parameter scales
    vol_thresh_scale: float
    pullback_bars_scale: float
    trail_atr_scale: float
    sl_mult_scale: float
    rsi_oversold_scale: float
    rsi_overbought_scale: float
    # Optuna search spaces (loosely typed: each strategy picks a subset
    # of keys and adds its own. See docs/methodology.md for the canonical
    # key list per strategy.)
    search_space: dict[str, tuple[float, float]]
    search_space_pb: dict[str, tuple[float, float]]
    # Default per-pair risk weight
    default_risk_per_pair: float
    # Default expected trades/year
    default_expected_trades_per_year: int
    # Market impact cap
    market_impact_volume_pct: float


# ════════════════════════════════════════════════════════════════════════
# STATIC -- Infrastructure constants (rarely change, framework-level)
# ════════════════════════════════════════════════════════════════════════

STATIC: StaticConfig = {
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
    # Hansen-literal SPA stationary-bootstrap expected block length. 0 = use
    # Politis-Romano p = max(1, round(n_k ** (1/3))) per trial (default);
    # >0 = fixed expected block length override (paper-disclosed calibration).
    "spa_hansen_block_length_override": 0,
    "spa_equity_warn_threshold_usd": 10.0,  # SPA transparency check
    # WFA -- purge days depend on feature lookback (infrastructure)
    # Purge ≥ max lookback window vol_pct_rank (720 bar 1H = 30 days)
    # to prevent parameter contamination from IS leaking into OOS.
    # NOTE: btc_ema_4800 (span=4800, ~69-day half-life) is not fully
    # converged after 90-day purge (~87% real-data weight). macro_trend
    # is binary (1/-1) and tolerates residual contamination well.
    "wfa_purge_days": 90,                   # MAXIMUM; adaptive via _get_purge_days
    # WFA contiguity: 48h = weekend (Sat+Sun ≈ 48h) + 0h maintenance
    # buffer. Folds with internal gaps > 48h are skipped to prevent
    # training on fragmented data. Phase 4 (v0.5.x): extracted from
    # hardcoded ``pd.Timedelta(hours=48)`` in core/_wfa.py.
    "max_allowed_gap_hours": 48,
    # Correlation-aware position sizing: 180-day rolling correlation
    # window. Below 30 days the matrix is too noisy to be useful, so
    # the sizing pass is skipped. Phase 4 (v0.5.x): extracted from
    # hardcoded constants in core/_portfolio.py (CORR_LOOKBACK = 180
    # and the 30-bar fallback in the same module).
    "correlation_lookback_days": 180,
    "correlation_min_lookback_days": 30,
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

DEFAULTS: DefaultsConfig = {
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
    # WFA parameter centers (L2 regularization midpoints for optuna search)
    # Extracted from hardcoded values in core/_wfa.py WalkForwardObjective
    "vol_thresh_center": 0.25,
    "pullback_bars_center": 5.5,
    "trail_atr_center": 3.25,
    "sl_mult_center": 2.0,
    "rsi_oversold_center": 30.0,
    "rsi_overbought_center": 70.0,
    # WFA parameter scales (L2 regularization scales for optuna search)
    "vol_thresh_scale": 0.15,
    "pullback_bars_scale": 2.5,
    "trail_atr_scale": 1.75,
    "sl_mult_scale": 1.0,
    "rsi_oversold_scale": 5.0,
    "rsi_overbought_scale": 5.0,
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
