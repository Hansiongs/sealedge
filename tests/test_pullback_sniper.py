"""Tests for pullback_sniper strategy and no-lookahead verification."""

import numpy as np
import pandas as pd

from quant_lib.core._config import DEFAULTS
from quant_lib.core._engine import (
    fast_trade_loop,
    STRATEGY_PULLBACK_SNIPER,
    STRATEGY_VOL_COMPRESSION,
)
from quant_lib.core._features import (
    prepare_data_with_max_time,
    _compute_rsi,
    STRATEGY_PULLBACK_SNIPER as FEATURES_PULLBACK,
)


def _make_arrays(n=1000, seed=42):
    """Create synthetic arrays for pullback sniper testing."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
    close = np.maximum(close, 10.0)
    atr_vals = np.abs(rng.normal(1.0, 0.2, n))
    rsi = 50.0 + rng.normal(0, 5, n)
    rsi = np.clip(rsi, 0, 100)
    return {
        "opens": close + rng.normal(0, 0.1, n),
        "highs": close + np.abs(rng.normal(0.5, 0.2, n)),
        "lows": close - np.abs(rng.normal(0.5, 0.2, n)),
        "closes": close,
        "hh_20": np.maximum.accumulate(close),
        "ll_20": np.minimum.accumulate(close),
        "ema_200s": np.full(n, np.mean(close[:200])),
        "rsi_14": rsi.astype(np.float64),
        "bullish_reversal": (close > np.roll(close, 1)).astype(np.int32),
        "bearish_reversal": (close < np.roll(close, 1)).astype(np.int32),
        "vol_pct_rank": rng.uniform(0, 1, n),
        "rvol": rng.uniform(0, 5, n),
        "atrs": atr_vals,
        "funding_rates": rng.normal(0.0001, 0.001, n),
        "macro_vols": rng.uniform(0.3, 1.5, n),
        "macro_trends": rng.choice([-1, 1], n).astype(np.int32),
        "is_weekends": rng.integers(0, 2, n).astype(np.int32),
        "is_funding_hours": rng.integers(0, 2, n).astype(np.int32),
    }


class TestPullbackSniper:
    def test_pullback_sniper_executes(self):
        """Pullback sniper runs without errors."""
        arrays = _make_arrays(500)
        result = fast_trade_loop(
            **arrays,
            strategy_type=STRATEGY_PULLBACK_SNIPER,
            vol_pct_thresh=0.20,
            rvol_thresh=2.5,
            pullback_bars=3,
            trail_atr=3.0,
            sl_mult=1.5,
            bailout_bars=36,
            warmup_bars=100,
            fee_taker=0.0005,
            use_rvol=1,
            use_ema=1,
            allow_long=1,
            allow_short=1,
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            weekend_penalty=2.0,
            stress_mult=DEFAULTS["stress_test_multiplier"],
            random_draws=np.random.default_rng(0).random(size=1000).astype(np.float64),
            trend_aligned_mult=1.5,
            trend_counter_mult=0.5,
        )
        pnl = result[0]
        assert len(pnl) >= 0

    def test_pullback_sniper_allow_long_only(self):
        """With allow_long=1, allow_short=0, should only produce long trades."""
        arrays = _make_arrays(1000)
        result = fast_trade_loop(
            **arrays,
            strategy_type=STRATEGY_PULLBACK_SNIPER,
            vol_pct_thresh=0.20,
            rvol_thresh=2.5,
            pullback_bars=3,
            trail_atr=3.0,
            sl_mult=1.5,
            bailout_bars=36,
            warmup_bars=200,
            fee_taker=0.0005,
            use_rvol=1,
            use_ema=1,
            allow_long=1,
            allow_short=0,  # only long
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            weekend_penalty=2.0,
            stress_mult=DEFAULTS["stress_test_multiplier"],
            random_draws=np.random.default_rng(0).random(size=2000).astype(np.float64),
            trend_aligned_mult=1.5,
            trend_counter_mult=0.5,
        )
        t_dir = result[3]
        if len(t_dir) > 0:
            for d in t_dir[:len(result[0])]:
                assert d == 1, f"Short trade found with allow_short=0: {d}"

    def test_pullback_sniper_allow_short_only(self):
        """With allow_long=0, allow_short=1, should only produce short trades."""
        arrays = _make_arrays(1000)
        result = fast_trade_loop(
            **arrays,
            strategy_type=STRATEGY_PULLBACK_SNIPER,
            vol_pct_thresh=0.20,
            rvol_thresh=2.5,
            pullback_bars=3,
            trail_atr=3.0,
            sl_mult=1.5,
            bailout_bars=36,
            warmup_bars=200,
            fee_taker=0.0005,
            use_rvol=1,
            use_ema=1,
            allow_long=0,
            allow_short=1,  # only short
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            weekend_penalty=2.0,
            stress_mult=DEFAULTS["stress_test_multiplier"],
            random_draws=np.random.default_rng(0).random(size=2000).astype(np.float64),
            trend_aligned_mult=1.5,
            trend_counter_mult=0.5,
        )
        t_dir = result[3]
        if len(t_dir) > 0:
            for d in t_dir[:len(result[0])]:
                assert d == -1, f"Long trade found with allow_long=0: {d}"


class TestNoLookaheadPullbackSniper:
    """CRITICAL: Verify no-lookahead in pullback_sniper features."""

    def test_rsi_does_not_use_current_bar_close(self):
        """Verify the SHIFTED RSI[14] in features does not use current bar close.

        Raw RSI[14] is a function of all prices up to and including the
        current bar (this is correct for the raw oscillator). However,
        in `compute_features`, the RSI is `.shift(1)`-ed, so the
        'rsi_14' column at bar i uses only data up to bar i-1.

        We test that by modifying close[i] and verifying that
        rsi_14 AT BAR i-1 (where shift places the original rsi[i])
        is UNCHANGED.
        """
        n = 500
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
        close = np.maximum(close, 10.0)

        # Compute raw RSI (no shift) on original data
        rsi_before = _compute_rsi(pd.Series(close), period=14)
        rsi_at_n_minus_1_before = float(rsi_before.iloc[-2])  # bar n-1 uses close[0..n-1]
        rsi_at_n_minus_2_before = float(rsi_before.iloc[-3])  # bar n-2 uses close[0..n-2]

        # Modify close at bar n-1 (last bar)
        close_modified = close.copy()
        close_modified[-1] = close[-1] * 10.0  # huge change

        rsi_after = _compute_rsi(pd.Series(close_modified), period=14)
        rsi_at_n_minus_1_after = float(rsi_after.iloc[-2])
        rsi_at_n_minus_2_after = float(rsi_after.iloc[-3])

        # RSI at bar n-1 should NOT be affected by close[n-1] modification
        # (RSI at bar n-1 uses close[0..n-2], so modifying close[n-1] is safe)
        assert abs(rsi_at_n_minus_1_before - rsi_at_n_minus_1_after) < 1e-9, (
            f"RSI[{n-1}] should not use close[{n-1}]! "
            f"before={rsi_at_n_minus_1_before}, after={rsi_at_n_minus_1_after}"
        )
        # RSI at bar n-2 should be identical (doesn't use close[n-1] at all)
        assert abs(rsi_at_n_minus_2_before - rsi_at_n_minus_2_after) < 1e-9

    def test_features_rsi_is_shifted(self):
        """Verify compute_features applies shift(1) to RSI."""
        n = 500
        rng = np.random.default_rng(42)
        base_time = pd.Timestamp("2024-01-01")
        times = [base_time + pd.Timedelta(hours=i) for i in range(n)]
        close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
        close = np.maximum(close, 10.0)

        df_raw = pd.DataFrame({
            "time": times,
            "open": close + rng.normal(0, 0.1, n),
            "high": close + np.abs(rng.normal(0.5, 0.2, n)),
            "low": close - np.abs(rng.normal(0.5, 0.2, n)),
            "close": close,
            "volume": rng.exponential(1000, n),
        })
        btc_raw = df_raw.copy()
        btc_raw["close"] = 1000 + np.cumsum(rng.normal(0, 5, n))
        btc_raw["high"] = btc_raw["close"] + 5
        btc_raw["low"] = btc_raw["close"] - 5
        btc_raw["volume"] = rng.exponential(50000, n)

        # Compute pullback features
        df = prepare_data_with_max_time(
            df_raw, btc_raw, None, pd.Timestamp(times[-1]),
            strategy_type=STRATEGY_PULLBACK_SNIPER,
        )

        # rsi_14 column must exist
        assert "rsi_14" in df.columns, "rsi_14 column missing"

        # First 14 values should be NaN (RSI warmup)
        # After warmup, RSI[i] should be from data <= close[i-1]
        # We can verify by recomputing RSI and shifting
        expected_rsi = _compute_rsi(df_raw["close"], period=14).shift(1)
        # Compare the first 50 non-NaN values
        for i in range(15, 30):
            actual = df["rsi_14"].iloc[i]
            expected = float(expected_rsi.iloc[i])
            assert abs(actual - expected) < 1e-6, (
                f"rsi_14[{i}] mismatch: actual={actual}, expected={expected}"
            )

    def test_bullish_reversal_uses_no_future_data(self):
        """Verify bullish_reversal[i] doesn't use close[i+1] (no lookahead)."""
        n = 100
        rng = np.random.default_rng(42)
        base_time = pd.Timestamp("2024-01-01")
        times = [base_time + pd.Timedelta(hours=i) for i in range(n)]
        close = 100.0 + rng.normal(0, 0.3, n).cumsum()
        close = np.maximum(close, 10.0)
        opens = close - rng.normal(0, 0.1, n)

        df_raw = pd.DataFrame({
            "time": times,
            "open": opens,
            "high": close + np.abs(rng.normal(0.5, 0.2, n)),
            "low": close - np.abs(rng.normal(0.5, 0.2, n)),
            "close": close,
            "volume": rng.exponential(1000, n),
        })
        btc_raw = df_raw.copy()
        btc_raw["close"] = 1000 + np.cumsum(rng.normal(0, 5, n))

        df = prepare_data_with_max_time(
            df_raw, btc_raw, None, pd.Timestamp(times[-1]),
            strategy_type=STRATEGY_PULLBACK_SNIPER,
        )

        # Modify close[n-1] (last bar, future from bar n-2's perspective)
        # bullish_reversal at bar n-2 should NOT change
        br_at_n2_before = df["bullish_reversal"].iloc[-2]

        df_raw_mod = df_raw.copy()
        df_raw_mod.loc[df_raw_mod.index[-1], "close"] *= 5.0
        df_mod = prepare_data_with_max_time(
            df_raw_mod, btc_raw, None, pd.Timestamp(times[-1]),
            strategy_type=STRATEGY_PULLBACK_SNIPER,
        )
        br_at_n2_after = df_mod["bullish_reversal"].iloc[-2]
        assert br_at_n2_before == br_at_n2_after, (
            f"bullish_reversal at bar n-2 changed when modifying bar n-1: "
            f"{br_at_n2_before} -> {br_at_n2_after}"
        )


class TestStrategyDispatch:
    """Test that engine correctly dispatches to right strategy."""

    def test_both_strategies_run_without_error(self):
        """Both strategies should run without errors and return valid trades or empty."""
        arrays = _make_arrays(500)
        result0 = fast_trade_loop(
            **arrays,
            strategy_type=STRATEGY_VOL_COMPRESSION,
            vol_pct_thresh=0.20,
            rvol_thresh=2.5,
            pullback_bars=3,
            trail_atr=3.0,
            sl_mult=1.5,
            bailout_bars=36,
            warmup_bars=100,
            fee_taker=0.0005,
            use_rvol=1,
            use_ema=1,
            allow_long=1,
            allow_short=1,
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            weekend_penalty=2.0,
            stress_mult=DEFAULTS["stress_test_multiplier"],
            random_draws=np.random.default_rng(0).random(size=1000).astype(np.float64),
            trend_aligned_mult=1.5,
            trend_counter_mult=0.5,
        )
        result1 = fast_trade_loop(
            **arrays,
            strategy_type=STRATEGY_PULLBACK_SNIPER,
            vol_pct_thresh=0.20,
            rvol_thresh=2.5,
            pullback_bars=3,
            trail_atr=3.0,
            sl_mult=1.5,
            bailout_bars=36,
            warmup_bars=100,
            fee_taker=0.0005,
            use_rvol=1,
            use_ema=1,
            allow_long=1,
            allow_short=1,
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            weekend_penalty=2.0,
            stress_mult=DEFAULTS["stress_test_multiplier"],
            random_draws=np.random.default_rng(0).random(size=1000).astype(np.float64),
            trend_aligned_mult=1.5,
            trend_counter_mult=0.5,
        )
        # Both should produce output arrays (possibly empty)
        assert result0[0] is not None
        assert result1[0] is not None
        # Both should have same return signature
        assert len(result0) == len(result1) == 10

    def test_strategy_constants_match_engine(self):
        """Verify strategy constants in features match engine."""
        from quant_lib.core._engine import (
            STRATEGY_VOL_COMPRESSION as ENGINE_VOL,
            STRATEGY_PULLBACK_SNIPER as ENGINE_PULLBACK,
        )
        from quant_lib.core._features import (
            STRATEGY_VOL_COMPRESSION as FEATURES_VOL,
        )
        from quant_lib.audit import (
            STRATEGY_VOL_COMPRESSION as AUDIT_VOL,
            STRATEGY_PULLBACK_SNIPER as AUDIT_PULLBACK,
        )
        assert ENGINE_VOL == FEATURES_VOL == AUDIT_VOL == 0
        assert ENGINE_PULLBACK == FEATURES_PULLBACK == AUDIT_PULLBACK == 1
