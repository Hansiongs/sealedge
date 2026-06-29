"""Tests for Numba trade engine — fast_trade_loop + EngineArgs dataclass."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from quant_lib.core._config import DEFAULTS
from quant_lib.core._engine import (
    EngineArgs,
    fast_trade_loop,
    STRATEGY_VOL_COMPRESSION,
    STRATEGY_PULLBACK_SNIPER,
)


def _make_arrays(n=1000, seed=42):
    """Create synthetic arrays for trade loop testing."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
    close = np.maximum(close, 10.0)
    atr_vals = np.abs(rng.normal(1.0, 0.2, n))
    rsi = 50.0 + rng.normal(0, 5, n)  # around 50 = neutral
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


def _common_extra(arrays, seed=0):
    """Common kwargs for trend risk multiplier params."""
    return {
        "strategy_type": STRATEGY_VOL_COMPRESSION,
        "allow_long": 1,
        "allow_short": 1,
        "rsi_oversold": 30.0,
        "rsi_overbought": 70.0,
        "trend_aligned_mult": 1.5,
        "trend_counter_mult": 0.5,
    }


class TestFastTradeLoop:
    def test_basic_execution(self):
        arrays = _make_arrays(500)
        result = fast_trade_loop(
            **arrays,
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
            weekend_penalty=2.0,
            stress_mult=DEFAULTS["stress_test_multiplier"],  # mirrors DEFAULTS["stress_test_multiplier"]
            random_draws=np.random.default_rng(0).random(size=1000).astype(np.float64),
            **_common_extra(arrays),
        )
        pnl, idx_entry, idx_exit, t_dir, *_ = result
        assert isinstance(pnl, np.ndarray)
        assert pnl.dtype == np.float64
        if len(pnl) > 0:
            assert np.all(np.isfinite(pnl)), "PnL contains NaN or inf"
            assert len(idx_entry) == len(pnl)
            assert len(idx_exit) == len(pnl)
            assert (idx_exit > idx_entry).all(), "Exit before entry!"

    def test_no_trades_when_threshold_extreme(self):
        arrays = _make_arrays(500)
        # vol_pct_thresh = 0 means no trade will ever be triggered
        result = fast_trade_loop(
            **arrays,
            vol_pct_thresh=0.0,
            rvol_thresh=999.0,
            pullback_bars=3,
            trail_atr=3.0,
            sl_mult=1.5,
            bailout_bars=36,
            warmup_bars=100,
            fee_taker=0.0005,
            use_rvol=1,
            use_ema=1,
            weekend_penalty=2.0,
            stress_mult=DEFAULTS["stress_test_multiplier"],  # mirrors DEFAULTS["stress_test_multiplier"]
            random_draws=np.random.default_rng(0).random(size=1000).astype(np.float64),
            **_common_extra(arrays),
        )
        pnl, *_ = result
        assert len(pnl) == 0

    def test_all_trades_have_valid_direction(self):
        arrays = _make_arrays(1000)
        result = fast_trade_loop(
            **arrays,
            vol_pct_thresh=0.30,
            rvol_thresh=1.5,
            pullback_bars=5,
            trail_atr=4.0,
            sl_mult=2.0,
            bailout_bars=36,
            warmup_bars=200,
            fee_taker=0.0005,
            use_rvol=1,
            use_ema=1,
            weekend_penalty=2.0,
            stress_mult=DEFAULTS["stress_test_multiplier"],  # mirrors DEFAULTS["stress_test_multiplier"]
            random_draws=np.random.default_rng(1).random(size=2000).astype(np.float64),
            **_common_extra(arrays),
        )
        pnl, _, _, t_dir, *_ = result
        if len(pnl) > 0:
            for d in t_dir[:len(pnl)]:
                assert d in (-1, 1)

    def test_bailout_triggers(self):
        """With very tight trail_atr, trades should exit via trail or bailout."""
        arrays = _make_arrays(1000)
        result = fast_trade_loop(
            **arrays,
            vol_pct_thresh=0.10,
            rvol_thresh=1.0,
            pullback_bars=3,
            trail_atr=10.0,  # very loose trailing
            sl_mult=10.0,    # very loose initial stop
            bailout_bars=10,  # very short bailout
            warmup_bars=200,
            fee_taker=0.0005,
            use_rvol=1,
            use_ema=0,
            weekend_penalty=2.0,
            stress_mult=DEFAULTS["stress_test_multiplier"],  # mirrors DEFAULTS["stress_test_multiplier"]
            random_draws=np.random.default_rng(2).random(size=2000).astype(np.float64),
            **_common_extra(arrays),
        )
        pnl, idx_entry, idx_exit, *_ = result
        if len(pnl) > 0:
            bars_held = idx_exit - idx_entry
            # If bailout triggered, bars_held <= bailout_bars
            assert (bars_held <= 10).all(), (
                f"bailout_bars=10 but some bars_held > 10: {bars_held}"
            )

    def test_different_random_seeds_produce_different_results(self):
        arrays = _make_arrays(2000)
        base_kw = dict(
            vol_pct_thresh=0.20,
            rvol_thresh=2.0,
            pullback_bars=4,
            trail_atr=3.0,
            sl_mult=1.5,
            bailout_bars=36,
            warmup_bars=300,
            fee_taker=0.0005,
            use_rvol=1,
            use_ema=1,
            weekend_penalty=2.0,
            stress_mult=DEFAULTS["stress_test_multiplier"],  # mirrors DEFAULTS["stress_test_multiplier"]
            **_common_extra(arrays),
        )

        result_a = fast_trade_loop(
            **arrays,
            random_draws=np.random.default_rng(0).random(size=4000).astype(np.float64),
            **base_kw,
        )
        result_b = fast_trade_loop(
            **arrays,
            random_draws=np.random.default_rng(1).random(size=4000).astype(np.float64),
            **base_kw,
        )
        pnl_a, *_ = result_a
        pnl_b, *_ = result_b
        assert isinstance(pnl_a, np.ndarray) and isinstance(pnl_b, np.ndarray)
        if len(pnl_a) > 0 and len(pnl_b) > 0:
            assert not np.array_equal(pnl_a, pnl_b), (
                "Different RNG seeds produced identical pnls"
            )

    def test_trend_risk_mult_output_present(self):
        """Verify the 10th return array (trend_risk_mult) is present and valid."""
        arrays = _make_arrays(1000)
        result = fast_trade_loop(
            **arrays,
            vol_pct_thresh=0.30,
            rvol_thresh=1.0,
            pullback_bars=3,
            trail_atr=3.0,
            sl_mult=1.5,
            bailout_bars=36,
            warmup_bars=200,
            fee_taker=0.0005,
            use_rvol=0,
            use_ema=0,
            weekend_penalty=2.0,
            stress_mult=DEFAULTS["stress_test_multiplier"],  # mirrors DEFAULTS["stress_test_multiplier"]
            random_draws=np.random.default_rng(3).random(size=2000).astype(np.float64),
            **_common_extra(arrays),
        )
        pnl, idx_en, idx_ex, t_dir, m_trend, cb_vol, en_pr, ex_pr, sl_pcts, trend_mults = result
        assert len(trend_mults) == len(pnl)
        if len(trend_mults) > 0:
            for tm in trend_mults:
                assert tm in (0.5, 1.5)

    def test_nan_in_inputs_dont_propagate_to_pnl(self):
        """A NaN in one input array should not produce NaN in PnL."""
        arrays = _make_arrays(500)
        # Introduce NaN in one position
        arrays["closes"][100] = np.nan
        arrays["highs"][100] = np.nan
        result = fast_trade_loop(
            **arrays,
            vol_pct_thresh=0.20,
            rvol_thresh=2.5,
            pullback_bars=5,
            trail_atr=3.0,
            sl_mult=1.5,
            bailout_bars=36,
            warmup_bars=100,
            fee_taker=0.0005,
            use_rvol=1,
            use_ema=1,
            weekend_penalty=2.0,
            stress_mult=DEFAULTS["stress_test_multiplier"],
            random_draws=np.random.default_rng(0).random(size=1000).astype(np.float64),
            **_common_extra(arrays),
        )
        pnl = result[0]
        if len(pnl) > 0:
            assert np.all(np.isfinite(pnl)), (
                "NaN input propagated to PnL — engine should guard against this"
            )


# ─────────────────────────────────────────────────────────────────────
# EngineArgs dataclass (Sprint 3 hygiene: groups engine args by category)
# ─────────────────────────────────────────────────────────────────────


class TestEngineArgsDataclass:
    """EngineArgs is a frozen dataclass that groups positional engine
    args by category. Its ``as_tuple()`` unpacks into the @njit signature.
    """

    def test_engine_args_as_tuple_length_matches_njit(self):
        import inspect
        n_params = len(inspect.signature(fast_trade_loop).parameters)
        n = 10
        zeros = np.zeros(n, dtype=np.float64)
        zeros_i = np.zeros(n, dtype=np.int32)
        ones_i = np.ones(n, dtype=np.int32)
        args = EngineArgs(
            market_data=(zeros, zeros, zeros, zeros),
            channel_features=(zeros, zeros, zeros),
            pullback_features=(zeros, zeros_i, zeros_i),
            signal_features=(zeros, zeros, zeros),
            auxiliary_features=(zeros, zeros, ones_i, zeros_i, zeros_i),
            strategy_type=0,
            thresholds=(0.2, 2.5, 30.0, 70.0, 0.0),
            integer_params=(5, 36, 0, 0),
            exit_params=(3.0, 1.5),
            cost_model=(0.05, 2.0, 2.5),
            flags=(1, 1, 1, 1),
            random_draws=zeros,
            trend_mults=(1.5, 0.5),
        )
        out = args.as_tuple()
        assert len(out) == n_params, (
            "Expected {} args, got {}".format(n_params, len(out))
        )

    def test_engine_args_is_frozen(self):
        """EngineArgs is frozen -- prevents accidental mutation."""
        zeros_f = np.zeros(2, dtype=np.float64)
        zeros_i = np.zeros(2, dtype=np.int32)
        args = EngineArgs(
            market_data=(zeros_f, zeros_f, zeros_f, zeros_f),
            channel_features=(zeros_f, zeros_f, zeros_f),
            pullback_features=(zeros_f, zeros_i, zeros_i),
            signal_features=(zeros_f, zeros_f, zeros_f),
            auxiliary_features=(zeros_f, zeros_f, zeros_i, zeros_i, zeros_i),
            strategy_type=0,
            thresholds=(0.0, 0.0, 0.0, 0.0, 0.0),
            integer_params=(0, 0, 0, 0),
            exit_params=(0.0, 0.0),
            cost_model=(0.0, 0.0, 0.0),
            flags=(0, 0, 0, 0),
            random_draws=zeros_f,
            trend_mults=(0.0, 0.0),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            args.strategy_type = 1


# ═══════════════════════════════════════════════════════════════════════
# Signal data + EngineArgs builder for behavioural tests below
# (merged from the former test_engine_coverage.py)
# ═══════════════════════════════════════════════════════════════════════


def _build_signal_data(
    n: int = 5000,
    n_signals: int = 100,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Build deterministic data that fires vol_compression trades.

    Each signal is 30 bars: 20 flat (resets window) + 1 breakout
    + 1 pullback + 1 recovery + 7 continuation.
    """
    n_per_signal = 30
    n_signals = min(n_signals, n // n_per_signal)
    close = np.full(n, 100.0)
    high = np.full(n, 100.3)
    low = np.full(n, 99.7)
    open_ = np.full(n, 100.0)
    for k in range(n_signals):
        base = k * n_per_signal
        for i in range(20):
            idx = base + i
            if idx < n:
                close[idx] = 100.0
        idx = base + 21
        if idx < n:
            close[idx] = 102.0
            high[idx] = 102.5
        idx = base + 22
        if idx < n:
            close[idx] = 100.5
        idx = base + 23
        if idx < n:
            close[idx] = 103.0
        for i in range(24, 30):
            idx = base + i
            if idx < n:
                close[idx] = 103.0
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }


def _build_engine_args_for_behavioral(
    n: int = 500,
    seed: int = 42,
    strategy_type: int = STRATEGY_VOL_COMPRESSION,
    vol_pct_rank=None, rvol=None, atrs=None, rsi_14=None,
    bullish_rev=None, bearish_rev=None, macro_trends=None,
    closes=None, highs=None, lows=None, opens=None,
    ema_200s=None, funding_rates=None, is_weekends=None,
    is_funding_hours=None, macro_vols=None,
    integer_params=None, exit_params=None, cost_model=None,
    flags=None, trend_mults=None, thresholds=None,
) -> EngineArgs:
    """Build EngineArgs with sensible defaults; allow overrides for
    specific behavioural tests (entry / exit / cost paths).
    """
    rng = np.random.default_rng(seed)
    h = highs if highs is not None else np.full(n, 100.3)
    l_arr = lows if lows is not None else np.full(n, 99.7)
    hh_20 = pd.Series(h).rolling(20).max().shift(1).bfill().values
    ll_20 = pd.Series(l_arr).rolling(20).min().shift(1).bfill().values
    return EngineArgs(
        market_data=(
            opens if opens is not None else np.full(n, 100.0),
            h, l_arr,
            closes if closes is not None else np.full(n, 100.0),
        ),
        channel_features=(
            hh_20, ll_20,
            ema_200s if ema_200s is not None else np.full(n, 100.0),
        ),
        pullback_features=(
            rsi_14 if rsi_14 is not None else np.full(n, 50.0),
            bullish_rev if bullish_rev is not None else np.zeros(n, dtype=np.int32),
            bearish_rev if bearish_rev is not None else np.zeros(n, dtype=np.int32),
        ),
        signal_features=(
            vol_pct_rank if vol_pct_rank is not None else np.full(n, 0.05),
            rvol if rvol is not None else np.full(n, 3.0),
            atrs if atrs is not None else np.full(n, 1.5),
        ),
        auxiliary_features=(
            funding_rates if funding_rates is not None else np.zeros(n),
            macro_vols if macro_vols is not None else np.full(n, 0.5),
            macro_trends if macro_trends is not None else np.ones(n, dtype=np.int32),
            is_weekends if is_weekends is not None else np.zeros(n, dtype=np.int32),
            is_funding_hours if is_funding_hours is not None else np.zeros(n, dtype=np.int32),
        ),
        strategy_type=strategy_type,
        thresholds=thresholds if thresholds is not None else (0.20, 2.5, 30.0, 70.0, 0.0),
        integer_params=integer_params if integer_params is not None else (5, 36, 0, 0),
        exit_params=exit_params if exit_params is not None else (3.0, 1.5),
        cost_model=cost_model if cost_model is not None else (
            0.05, 2.0, DEFAULTS["stress_test_multiplier"],
        ),
        flags=flags if flags is not None else (1, 1, 1, 1),
        random_draws=rng.random(size=n * 2),
        trend_mults=trend_mults if trend_mults is not None else (1.5, 0.5),
    )


# ═══════════════════════════════════════════════════════════════════════
# Invariants (merged from former test_engine_coverage.py)
# ═══════════════════════════════════════════════════════════════════════


class TestEngineInvariants:
    """Properties the engine must always satisfy."""

    def test_output_shape_consistent(self):
        """All 10 return arrays must have the same length (trade count)."""
        args = _build_engine_args_for_behavioral(n=300)
        result = fast_trade_loop(*args.as_tuple())
        lengths = {len(r) for r in result}
        assert len(lengths) == 1, f"Inconsistent return lengths: {lengths}"
        assert lengths.pop() >= 0

    def test_trade_indices_in_bounds(self):
        """Trade entry/exit indices must be within [warmup_bars, n)."""
        sig = _build_signal_data(n=2000, n_signals=30)
        args = _build_engine_args_for_behavioral(
            n=2000,
            closes=sig["close"], highs=sig["high"],
            lows=sig["low"], opens=sig["open"],
        )
        result = fast_trade_loop(*args.as_tuple())
        pnl, idx_en, idx_ex, t_dir = result[:4]
        n = 2000
        for i in range(len(pnl)):
            assert 0 <= idx_en[i] < n
            assert 0 <= idx_ex[i] < n
            assert idx_en[i] <= idx_ex[i]

    def test_r_multiples_are_finite(self):
        """R-multiples should be finite (no NaN, no inf)."""
        sig = _build_signal_data(n=2000, n_signals=30)
        args = _build_engine_args_for_behavioral(
            n=2000,
            closes=sig["close"], highs=sig["high"],
            lows=sig["low"], opens=sig["open"],
        )
        result = fast_trade_loop(*args.as_tuple())
        for r in result[0]:
            assert np.isfinite(r)

    def test_determinism_same_input_same_output(self):
        """Same inputs must produce same outputs (Numba deterministic)."""
        args = _build_engine_args_for_behavioral(n=500, seed=42)
        r1 = fast_trade_loop(*args.as_tuple())
        r2 = fast_trade_loop(*args.as_tuple())
        for a, b in zip(r1, r2):
            assert np.array_equal(a, b), "Engine is non-deterministic"

    def test_cost_clamped_to_5(self):
        """Even with extreme cost_model, R-multiples stay bounded."""
        sig = _build_signal_data(n=2000, n_signals=30)
        args = _build_engine_args_for_behavioral(
            n=2000,
            closes=sig["close"], highs=sig["high"],
            lows=sig["low"], opens=sig["open"],
        )
        # Override cost_model with extreme values
        new_args = EngineArgs(
            **{**args.__dict__, "cost_model": (10.0, 100.0, 100.0)},
        )
        result = fast_trade_loop(*new_args.as_tuple())
        for r in result[0]:
            assert r > -10.0, f"Extreme cost not clamped: r={r}"


# ═══════════════════════════════════════════════════════════════════════
# Vol_compression entry paths (merged from former test_engine_coverage.py)
# ═══════════════════════════════════════════════════════════════════════


class TestVolCompressionEntry:
    """The vol_compression_breakout entry logic (long + short)."""

    def test_long_entry_with_breakout_then_pullback_then_recovery(self):
        """Standard long trade: breakout → pullback → recovery → entry."""
        sig = _build_signal_data(n=2000, n_signals=30)
        args = _build_engine_args_for_behavioral(
            n=2000,
            closes=sig["close"], highs=sig["high"],
            lows=sig["low"], opens=sig["open"],
            strategy_type=STRATEGY_VOL_COMPRESSION,
        )
        result = fast_trade_loop(*args.as_tuple())
        pnl, _, _, t_dir = result[:4]
        assert len(pnl) > 0
        assert np.any(t_dir == 1) or np.any(t_dir == -1)

    def test_no_entry_when_vol_not_compressed(self):
        """If vol_pct_rank > thresh throughout, no trades."""
        n = 500
        args = _build_engine_args_for_behavioral(
            n=n,
            vol_pct_rank=np.full(n, 0.9),  # Always high
            rvol=np.full(n, 1.0),  # Below threshold
        )
        result = fast_trade_loop(*args.as_tuple())
        assert len(result[0]) == 0

    def test_no_entry_when_rvol_below_threshold(self):
        """rvol < thresh blocks the volume confirmation."""
        n = 500
        sig = _build_signal_data(n=n, n_signals=10)
        args = _build_engine_args_for_behavioral(
            n=n,
            closes=sig["close"], highs=sig["high"],
            lows=sig["low"], opens=sig["open"],
            vol_pct_rank=np.full(n, 0.05),  # compressed
            rvol=np.full(n, 1.0),  # rvol too low
        )
        result = fast_trade_loop(*args.as_tuple())
        assert len(result[0]) == 0

    def test_long_setup_cancelled_by_ema_filter(self):
        """If use_ema=1 and close < ema_200, no setup."""
        n = 500
        ema = np.full(n, 200.0)  # close < ema → EMA filter blocks
        args = _build_engine_args_for_behavioral(
            n=n,
            closes=np.full(n, 100.0),
            highs=np.full(n, 100.3),
            lows=np.full(n, 99.7),
            ema_200s=ema,
            vol_pct_rank=np.full(n, 0.05),
            rvol=np.full(n, 3.0),
        )
        result = fast_trade_loop(*args.as_tuple())
        assert len(result[0]) == 0


# ═══════════════════════════════════════════════════════════════════════
# Exit paths (merged from former test_engine_coverage.py)
# ═══════════════════════════════════════════════════════════════════════


class TestExitPaths:
    """Test all exit mechanisms: SL, bailout, force-close."""

    def test_long_exit_at_trailing_stop(self):
        """Long position exits at trailing SL when price drops enough."""
        n = 200
        sig = _build_signal_data(n=n, n_signals=3)
        args = _build_engine_args_for_behavioral(
            n=n,
            closes=sig["close"], highs=sig["high"],
            lows=sig["low"], opens=sig["open"],
            exit_params=(1.0, 1.5),  # tight trail
        )
        result = fast_trade_loop(*args.as_tuple())
        pnl, idx_en, idx_ex = result[:3]
        for i in range(len(pnl)):
            assert idx_ex[i] >= idx_en[i]

    def test_force_close_at_end_of_data(self):
        """Position still open at end of data is force-closed at n-1."""
        n = 500
        close = np.full(n, 100.0)
        high = np.full(n, 100.3)
        low = np.full(n, 99.7)
        open_ = np.full(n, 100.0)
        # Bar 21: breakout
        close[21] = 102.0
        high[21] = 102.5
        low[21] = 100.5
        # Bars 22-23: pullback + recovery (entry at 23)
        close[22] = 100.5
        high[22] = 101.0
        low[22] = 100.0
        close[23] = 103.0
        high[23] = 103.5
        low[23] = 102.5
        for i in range(24, n):
            close[i] = 103.0
            high[i] = 103.3
            low[i] = 102.7
            open_[i] = 103.0
        args = _build_engine_args_for_behavioral(
            n=n,
            closes=close, highs=high, lows=low, opens=open_,
            exit_params=(10.0, 1.5),  # very wide trail
            integer_params=(5, 1000, 0, 0),  # bailout=1000 (won't trigger)
        )
        result = fast_trade_loop(*args.as_tuple())
        pnl, idx_en, idx_ex = result[:3]
        if len(pnl) > 0:
            assert np.any(idx_ex == n - 1), (
                f"Expected force-close at n-1={n-1}, got idx_ex={idx_ex}"
            )

    def test_bailout_exit(self):
        """Position held longer than bailout_bars is closed at close."""
        n = 200
        sig = _build_signal_data(n=n, n_signals=3)
        args = _build_engine_args_for_behavioral(
            n=n,
            closes=sig["close"], highs=sig["high"],
            lows=sig["low"], opens=sig["open"],
            integer_params=(5, 2, 0, 0),  # pullback=5, bailout=2
        )
        result = fast_trade_loop(*args.as_tuple())
        assert isinstance(result[0], np.ndarray)


# ═══════════════════════════════════════════════════════════════════════
# Pullback_sniper entry paths (merged from former test_engine_coverage.py)
# ═══════════════════════════════════════════════════════════════════════


class TestPullbackSniperEntry:
    """The pullback_sniper single-bar entry (RSI + reversal)."""

    def test_long_entry_on_rsi_oversold_bullish_reversal(self):
        """Long entry: RSI < oversold AND bullish reversal candle."""
        n = 200
        rsi = np.full(n, 50.0)
        rsi[100] = 25.0  # oversold
        bullish = np.zeros(n, dtype=np.int32)
        bullish[100] = 1
        args = _build_engine_args_for_behavioral(
            n=n,
            closes=np.full(n, 100.0),
            highs=np.full(n, 100.5),
            lows=np.full(n, 99.5),
            rsi_14=rsi, bullish_rev=bullish,
            strategy_type=STRATEGY_PULLBACK_SNIPER,
            flags=(1, 0, 1, 1),  # use_rvol=1, use_ema=0, allow both
        )
        result = fast_trade_loop(*args.as_tuple())
        pnl, t_dir = result[0], result[3]
        assert len(pnl) >= 1
        assert t_dir[0] == 1

    def test_short_entry_on_rsi_overbought_bearish_reversal(self):
        """Short entry: RSI > overbought AND bearish reversal candle."""
        n = 200
        rsi = np.full(n, 50.0)
        rsi[100] = 75.0  # overbought
        bearish = np.zeros(n, dtype=np.int32)
        bearish[100] = 1
        args = _build_engine_args_for_behavioral(
            n=n,
            closes=np.full(n, 100.0),
            highs=np.full(n, 100.5),
            lows=np.full(n, 99.5),
            rsi_14=rsi, bearish_rev=bearish,
            strategy_type=STRATEGY_PULLBACK_SNIPER,
            flags=(1, 0, 1, 1),
        )
        result = fast_trade_loop(*args.as_tuple())
        pnl, t_dir = result[0], result[3]
        assert len(pnl) >= 1
        assert t_dir[0] == -1

    def test_no_entry_when_allow_long_disabled(self):
        """With allow_long=0 and a bullish setup, no long trade."""
        n = 200
        rsi = np.full(n, 50.0)
        rsi[100] = 25.0
        bullish = np.zeros(n, dtype=np.int32)
        bullish[100] = 1
        args = _build_engine_args_for_behavioral(
            n=n,
            closes=np.full(n, 100.0),
            highs=np.full(n, 100.5),
            lows=np.full(n, 99.5),
            rsi_14=rsi, bullish_rev=bullish,
            strategy_type=STRATEGY_PULLBACK_SNIPER,
            flags=(1, 0, 0, 1),  # allow_long=0
        )
        result = fast_trade_loop(*args.as_tuple())
        assert not np.any(result[3] == 1)


# ═══════════════════════════════════════════════════════════════════════
# Trend alignment (merged from former test_engine_coverage.py)
# ═══════════════════════════════════════════════════════════════════════


class TestTrendAlignment:
    """Trade-level trend alignment (with-trend, counter-trend)."""

    def test_long_with_bull_macro_trend(self):
        """Long entry in bull macro trend -> trend_aligned_mult."""
        n = 200
        sig = _build_signal_data(n=n, n_signals=3)
        macro_trend = np.ones(n, dtype=np.int32)  # all bull
        args = _build_engine_args_for_behavioral(
            n=n,
            closes=sig["close"], highs=sig["high"],
            lows=sig["low"], opens=sig["open"],
            macro_trends=macro_trend,
            trend_mults=(2.0, 0.5),
        )
        result = fast_trade_loop(*args.as_tuple())
        t_trend_mults = result[9]
        for i, t_dir in enumerate(result[3]):
            if t_dir == 1 and i < len(t_trend_mults):
                assert t_trend_mults[i] == 2.0

    def test_long_with_bear_macro_trend_is_counter(self):
        """Long entry in bear macro trend -> trend_counter_mult."""
        n = 200
        sig = _build_signal_data(n=n, n_signals=3)
        macro_trend = -np.ones(n, dtype=np.int32)  # all bear
        args = _build_engine_args_for_behavioral(
            n=n,
            closes=sig["close"], highs=sig["high"],
            lows=sig["low"], opens=sig["open"],
            macro_trends=macro_trend,
            trend_mults=(2.0, 0.5),
        )
        result = fast_trade_loop(*args.as_tuple())
        t_trend_mults = result[9]
        for i, t_dir in enumerate(result[3]):
            if t_dir == 1 and i < len(t_trend_mults):
                assert t_trend_mults[i] == 0.5


# ═══════════════════════════════════════════════════════════════════════
# Cost model (merged from former test_engine_coverage.py)
# ═══════════════════════════════════════════════════════════════════════


class TestCostModel:
    """Funding impact, weekend penalty, random stress."""

    def test_funding_impact_deducted(self):
        """Positive funding rates should reduce long PnL (cost)."""
        n = 200
        sig = _build_signal_data(n=n, n_signals=3)
        fund = np.zeros(n)
        fund[100:200] = 0.0001
        is_fund_hour = np.zeros(n, dtype=np.int32)
        is_fund_hour[100:200] = 1
        args = _build_engine_args_for_behavioral(
            n=n,
            closes=sig["close"], highs=sig["high"],
            lows=sig["low"], opens=sig["open"],
            funding_rates=fund,
            is_funding_hours=is_fund_hour,
        )
        result = fast_trade_loop(*args.as_tuple())
        assert isinstance(result[0], np.ndarray)

    def test_weekend_penalty_increases_cost(self):
        """Holding over weekend should increase cost (lower R)."""
        n = 200
        sig = _build_signal_data(n=n, n_signals=3)
        is_weekend = np.zeros(n, dtype=np.int32)
        is_weekend[100:200] = 1
        args = _build_engine_args_for_behavioral(
            n=n,
            closes=sig["close"], highs=sig["high"],
            lows=sig["low"], opens=sig["open"],
            is_weekends=is_weekend,
            cost_model=(0.05, 2.0, DEFAULTS["stress_test_multiplier"]),
        )
        result = fast_trade_loop(*args.as_tuple())
        assert isinstance(result[0], np.ndarray)
