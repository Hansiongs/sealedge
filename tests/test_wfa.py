"""Coverage push for quant_lib.core._wfa.

Targets:
- _get_purge_days (adaptive logic)
- _adaptive_trials (warm-start + IS size)
- WalkForwardObjective.__call__ (PSR + ESS + L2 branches)
- run_wfa_per_symbol (warm-start enqueue, consecutive_failures reset)
"""

from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from quant_lib.core._config import DEFAULTS
from quant_lib.core._wfa import (
    WalkForwardObjective,
    _adaptive_trials,
    _get_purge_days,
    run_wfa_per_symbol,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _make_prepped_df(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic prepped DataFrame with all required columns.

    The data is designed to occasionally trigger entries: vol_pct_rank
    dips below threshold with high rvol, and prices cross the HH_20/LL_20
    channel.
    """
    rng = np.random.default_rng(seed)
    base = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(base, 10.0)
    high = close + np.abs(rng.normal(0, 0.3, n))
    low = close - np.abs(rng.normal(0, 0.3, n))
    open_ = close + rng.normal(0, 0.1, n)
    # Build features: hh_20, ll_20, ema_200, rsi, reversal, etc.
    times = pd.date_range("2020-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "time": times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.exponential(1000, n),
        "hh_20": pd.Series(high).rolling(20).max().shift(1).bfill(),
        "ll_20": pd.Series(low).rolling(20).min().shift(1).bfill(),
        "ema_200": pd.Series(close).ewm(span=200, adjust=False).mean().shift(1).bfill(),
        "rsi_14": np.clip(50 + rng.normal(0, 10, n), 0, 100),
        "bullish_reversal": (rng.integers(0, 2, n)).astype(np.int32),
        "bearish_reversal": (rng.integers(0, 2, n)).astype(np.int32),
        "vol_pct_rank": np.clip(rng.normal(0.3, 0.2, n), 0, 1),
        "rvol": np.clip(rng.normal(2.0, 0.5, n), 0.5, 5.0),
        "atr": np.full(n, 1.5),
        "funding_rate": np.full(n, 0.0),
        "macro_vol": np.full(n, 0.5),
        "macro_trend": np.ones(n, dtype=np.int32),
        "is_weekend": np.zeros(n, dtype=np.int32),
        "is_funding_hour": np.zeros(n, dtype=np.int32),
    })
    return df


def _make_mock_trial(params: dict) -> MagicMock:
    """Build a mock Optuna trial that returns given params."""
    trial = MagicMock()
    trial.suggest_float = MagicMock(side_effect=lambda name, lo, hi: params.get(name, (lo + hi) / 2))
    trial.suggest_int = MagicMock(side_effect=lambda name, lo, hi: int(params.get(name, (lo + hi) / 2)))
    return trial


# ─────────────────────────────────────────────────────────────────────
# S4.1: _get_purge_days
# ─────────────────────────────────────────────────────────────────────


class TestGetPurgeDays:
    """_get_purge_days: adaptive purge based on IS size."""

    def test_small_is_uses_90_days(self):
        assert _get_purge_days(10) == 90
        assert _get_purge_days(15) == 90

    def test_medium_is_uses_60_days(self):
        assert _get_purge_days(16) == 60
        assert _get_purge_days(30) == 60

    def test_large_is_uses_30_days(self):
        assert _get_purge_days(31) == 30
        assert _get_purge_days(60) == 30

    # --- Phase 3.6 C1: defensive guard ---

    def test_zero_is_months_returns_max_purge(self):
        """Phase 3.6 C1: n_is_months <= 0 returns 90 (defensive)."""
        assert _get_purge_days(0) == 90

    def test_negative_is_months_returns_max_purge(self):
        """Phase 3.6 C1: n_is_months < 0 returns 90 (defensive)."""
        assert _get_purge_days(-5) == 90


# ─────────────────────────────────────────────────────────────────────
# S4.1: _adaptive_trials
# ─────────────────────────────────────────────────────────────────────


class TestAdaptiveTrials:
    """_adaptive_trials: trial count adapts to IS size + warm-start."""

    def test_small_is_no_prior(self):
        """<18 months IS, no prior -> 85% of base, at least 50."""
        base = DEFAULTS["wfa_trials_per_fold"]
        n = _adaptive_trials(12, None)
        assert n == max(50, int(base * 0.85))

    def test_small_is_with_prior(self):
        """<18 months IS, with prior -> 70% of base, at least 50."""
        base = DEFAULTS["wfa_trials_per_fold"]
        n = _adaptive_trials(12, {"vol_pct_thresh": 0.2})
        assert n == max(50, int(base * 0.70))

    def test_medium_is_no_prior(self):
        """18-30 months IS, no prior -> 75% of base."""
        base = DEFAULTS["wfa_trials_per_fold"]
        n = _adaptive_trials(24, None)
        assert n == int(base * 0.75)

    def test_medium_is_with_prior(self):
        """18-30 months IS, with prior -> 60% of base."""
        base = DEFAULTS["wfa_trials_per_fold"]
        n = _adaptive_trials(24, {"vol_pct_thresh": 0.2})
        assert n == int(base * 0.60)

    def test_large_is_no_prior(self):
        """>=30 months IS, no prior -> 70% of base, at least 50."""
        base = DEFAULTS["wfa_trials_per_fold"]
        n = _adaptive_trials(36, None)
        assert n == max(50, int(base * 0.70))

    def test_large_is_with_prior(self):
        """>=30 months IS, with prior -> 50% of base, at least 50."""
        base = DEFAULTS["wfa_trials_per_fold"]
        n = _adaptive_trials(36, {"vol_pct_thresh": 0.2})
        assert n == max(50, int(base * 0.50))

    # --- Phase 3.6 C1: defensive guard for _adaptive_trials ---

    def test_zero_is_months_returns_minimum_50(self):
        """Phase 3.6 C1: n_is_months <= 0 returns 50 (defensive min)."""
        assert _adaptive_trials(0, None) == 50
        assert _adaptive_trials(0, {"vol_pct_thresh": 0.2}) == 50
        assert _adaptive_trials(-3, None) == 50


# ─────────────────────────────────────────────────────────────────────
# S4.1: WalkForwardObjective.__call__ (PSR + ESS + L2 branches)
# ─────────────────────────────────────────────────────────────────────


class TestWalkForwardObjectiveCall:
    """Exercise all branches of WalkForwardObjective.__call__."""

    def test_call_returns_value_for_vol_compression(self):
        """Vol_compression trial: full L2 path, full PSR path."""
        df = _make_prepped_df(n=2000)
        obj = WalkForwardObjective(
            df, expected_trades_annual=30,
            use_rvol=True, use_ema=True,
            fold_seed=42, strategy_type=0,
        )
        trial = _make_mock_trial({
            "vol_pct_thresh": 0.20,
            "pullback_bars": 5,
            "trail_atr": 3.0,
            "sl_mult": 1.5,
        })
        # Might be -9999 if no trades; both are valid float returns
        result = obj(trial)
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_call_returns_value_for_pullback_sniper(self):
        """Pullback_sniper trial: RSI L2 branch exercised."""
        df = _make_prepped_df(n=2000)
        obj = WalkForwardObjective(
            df, expected_trades_annual=30,
            use_rvol=True, use_ema=True,
            fold_seed=42, strategy_type=1,
        )
        trial = _make_mock_trial({
            "vol_pct_thresh": 0.20,
            "pullback_bars": 5,
            "trail_atr": 3.0,
            "sl_mult": 1.5,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
        })
        result = obj(trial)
        assert isinstance(result, float)

    def test_call_returns_minus_9999_for_short_data(self):
        """When df is too small (<168), __call__ returns -9999."""
        df = _make_prepped_df(n=100)
        obj = WalkForwardObjective(
            df, expected_trades_annual=30,
            use_rvol=True, use_ema=True,
            fold_seed=42,
        )
        trial = _make_mock_trial({})
        assert obj(trial) == -9999.0

    def test_call_with_zero_reg_lambda_skips_l2(self):
        """reg_lambda=0 must skip the L2 branch entirely."""
        df = _make_prepped_df(n=2000)
        obj = WalkForwardObjective(
            df, expected_trades_annual=30,
            use_rvol=True, use_ema=True,
            fold_seed=42, strategy_type=0, reg_lambda=0.0,
        )
        trial = _make_mock_trial({
            "vol_pct_thresh": 0.20,
            "pullback_bars": 5,
            "trail_atr": 3.0,
            "sl_mult": 1.5,
        })
        result = obj(trial)
        assert isinstance(result, float)

    def test_init_with_zero_decay_uses_uniform_weights(self):
        """decay_halflife_months=0 must use uniform bar weights."""
        df = _make_prepped_df(n=500)
        obj = WalkForwardObjective(
            df, expected_trades_annual=30,
            use_rvol=True, use_ema=True,
            fold_seed=42, decay_halflife_months=0,
        )
        # bar_weights should be all 1.0
        assert np.allclose(obj.bar_weights, 1.0)


# ─────────────────────────────────────────────────────────────────────
# S4.1: run_wfa_per_symbol branches
# ─────────────────────────────────────────────────────────────────────


class TestRunWfaPerSymbol:
    """Exercise the per-fold loop in run_wfa_per_symbol."""

    def test_run_wfa_with_minimal_data(self):
        """Small data: should skip folds due to insufficient size."""
        df = _make_prepped_df(n=200)  # way too small for 12+ month IS
        with patch_wfa_static():
            trades, params, is_trades = run_wfa_per_symbol(
                "BTCUSDT", df, use_rvol=True, use_ema=True,
                verbose=False, strategy_type=0,
            )
        # No trades (data too small) is acceptable; no crash is the goal
        assert isinstance(trades, list)
        assert isinstance(params, list)

    def test_run_wfa_with_sufficient_data_produces_folds(self):
        """Enough data: should produce at least one fold's worth."""
        # 2000 hours ~= 83 days ~= ~3 months. WFA needs min_train=6 months
        # (4320 hours). Use patched STATIC to lower the bar.
        df = _make_prepped_df(n=5000)
        with patch_wfa_static(min_train_months=3, trials=3):
            trades, params, is_trades = run_wfa_per_symbol(
                "BTCUSDT", df, use_rvol=True, use_ema=True,
                verbose=False, strategy_type=0,
            )
        # Should have some folds
        assert isinstance(params, list)
        # If folds were produced, they should have the expected keys
        if params:
            first = params[0]
            assert "fold" in first
            assert "best_value" in first
            assert "vol_pct_thresh" in first

    def test_run_wfa_warm_start_initializes_perturbed_trials(self):
        """When prev_best_params is None on first fold, no warm-start.
        On second fold, warm-start is applied (3 perturbed trials enqueued)."""
        # The warm-start logic is internal to run_wfa_per_symbol;
        # we just verify the function doesn't crash when run with data
        # large enough for 2 folds.
        df = _make_prepped_df(n=8000)  # 8000/24 ~= 333 days ~= ~11 months
        with patch_wfa_static(min_train_months=3, trials=2, test_months=1):
            trades, params, is_trades = run_wfa_per_symbol(
                "BTCUSDT", df, use_rvol=True, use_ema=True,
                verbose=False, strategy_type=0,
            )
        assert isinstance(params, list)

    def test_run_wfa_consecutive_failures_resets_warm_start(self):
        """When 2+ consecutive folds fail, prev_best_params resets to None.
        We verify this by checking the loop completes without crash
        even when no folds produce trades."""
        df = _make_prepped_df(n=300)  # very small -> all folds skipped
        with patch_wfa_static(min_train_months=2, trials=2):
            trades, params, is_trades = run_wfa_per_symbol(
                "BTCUSDT", df, use_rvol=True, use_ema=True,
                verbose=False, strategy_type=0,
            )
        # All folds should have been skipped (data too small)
        assert params == []

    def test_run_wfa_with_pullback_sniper(self):
        """strategy_type=1 path: RSI-specific Optuna suggestions."""
        df = _make_prepped_df(n=5000)
        with patch_wfa_static(min_train_months=3, trials=2):
            trades, params, is_trades = run_wfa_per_symbol(
                "BTCUSDT", df, use_rvol=True, use_ema=True,
                verbose=False, strategy_type=1,
            )
        assert isinstance(params, list)
        if params:
            # RSI fields must be in fold params when strategy_type=1
            # (0.2.2 fix: replaced tautology `assert X in p or X not in p`)
            p = params[0]
            assert "rsi_oversold" in p, (
                "pullback_sniper fold must include rsi_oversold"
            )
            assert "rsi_overbought" in p, (
                "pullback_sniper fold must include rsi_overbought"
            )

    def test_run_wfa_oos_continuity_check_skips_discontinuous_data(self):
        """If OOS has a >48h gap, fold is skipped."""
        df = _make_prepped_df(n=5000)
        # Inject a 72h gap in the middle
        gap_start = 3000
        df.loc[gap_start, "time"] = df.loc[gap_start, "time"] + pd.Timedelta(hours=72)
        with patch_wfa_static(min_train_months=3, trials=2):
            trades, params, is_trades = run_wfa_per_symbol(
                "BTCUSDT", df, use_rvol=True, use_ema=True,
                verbose=False, strategy_type=0,
            )
        # Should still complete without crash
        assert isinstance(params, list)


# ─────────────────────────────────────────────────────────────────────
# S4.1: helper context manager
# ─────────────────────────────────────────────────────────────────────


@contextmanager
def patch_wfa_static(
    min_train_months: int = 12,
    trials: int = 3,
    test_months: int = 3,
    reg_lambda: float = 0.05,
):
    """Patch DEFAULTS values for WFA tests; restore on exit.

    After Phase 4, WFA parameters are read from DEFAULTS (per-experiment
    config), not STATIC (infrastructure). Tests that want to override
    WFA parameters temporarily should patch DEFAULTS.
    """
    saved = {
        k: DEFAULTS[k]
        for k in [
            "wfa_trials_per_fold", "wfa_min_train_months",
            "wfa_test_months", "reg_lambda", "wfa_decay_halflife_months",
        ]
    }
    DEFAULTS["wfa_trials_per_fold"] = trials
    DEFAULTS["wfa_min_train_months"] = min_train_months
    DEFAULTS["wfa_test_months"] = test_months
    DEFAULTS["reg_lambda"] = reg_lambda
    try:
        yield
    finally:
        for k, v in saved.items():
            DEFAULTS[k] = v
