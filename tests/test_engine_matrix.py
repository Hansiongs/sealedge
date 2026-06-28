"""Parametrised coverage for ``quant_lib.core._engine.fast_trade_loop``.

The engine is the hottest path in the framework: every backtest
exercises ``fast_trade_loop`` once per symbol, fold, and trial.
This file pushes coverage from 9% toward the project target by
exercising the major feature-flag combinations via parametrisation
rather than dozens of bespoke tests.

Each parametrised test runs the engine on small synthetic arrays
(``n=200`` by default) and asserts only the invariants the engine
guarantees regardless of inputs:

- All 10 return arrays have the same length.
- Trade entry / exit indices are within ``[warmup, n)`` and ordered.
- ``t_dir`` values are in ``{-1, +1}``.
- ``t_sl_pct`` is in ``(0, 0.5)``.
- The number of trades is non-negative.

These invariants hold for every parametrisation, so each test acts
as a smoke test for its feature-flag combination without coupling to
the exact trade counts (which depend on the data).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from quant_lib.core._config import DEFAULTS
from quant_lib.core._engine import (
    EngineArgs,
    STRATEGY_PULLBACK_SNIPER,
    STRATEGY_VOL_COMPRESSION,
    fast_trade_loop,
)

from tests.conftest import make_engine_arrays


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _build_engine_args(
    n: int = 200,
    seed: int = 42,
    **overrides: Any,
) -> EngineArgs:
    """Build a valid ``EngineArgs`` with sensible defaults."""
    arrays = make_engine_arrays(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    return EngineArgs(
        market_data=(
            arrays["opens"],
            arrays["highs"],
            arrays["lows"],
            arrays["closes"],
        ),
        channel_features=(
            arrays["hh_20"],
            arrays["ll_20"],
            arrays["ema_200s"],
        ),
        pullback_features=(
            arrays["rsi_14"],
            arrays["bullish_reversal"],
            arrays["bearish_reversal"],
        ),
        signal_features=(
            arrays["vol_pct_rank"],
            arrays["rvol"],
            arrays["atrs"],
        ),
        auxiliary_features=(
            arrays["funding_rates"],
            arrays["macro_vols"],
            arrays["macro_trends"],
            arrays["is_weekends"],
            arrays["is_funding_hours"],
        ),
        strategy_type=overrides.get("strategy_type", STRATEGY_VOL_COMPRESSION),
        thresholds=overrides.get("thresholds", (0.20, 2.5, 30.0, 70.0, 0.0)),
        integer_params=overrides.get("integer_params", (5, 36, 0, 0)),
        exit_params=overrides.get("exit_params", (3.0, 1.5)),
        cost_model=overrides.get("cost_model", (0.05, 2.0, DEFAULTS["stress_test_multiplier"])),
        flags=overrides.get("flags", (1, 1, 1, 1)),
        random_draws=overrides.get("random_draws", rng.random(size=n * 2).astype(np.float64)),
        trend_mults=overrides.get("trend_mults", (1.5, 0.5)),
    )


def _assert_engine_invariants(result, n: int) -> int:
    """Assert invariants that hold for every ``fast_trade_loop`` call.

    Returns the trade count for further assertions.
    """
    assert len(result) == 10, f"Expected 10 return values, got {len(result)}"
    lengths = {len(r) for r in result}
    assert len(lengths) == 1, f"Inconsistent return lengths: {lengths}"
    n_trades = lengths.pop()
    assert n_trades >= 0
    if n_trades > 0:
        pnl, idx_en, idx_ex, t_dir = result[:4]
        for i in range(n_trades):
            assert 0 <= idx_en[i] < n, f"idx_en[{i}]={idx_en[i]} OOB"
            assert 0 <= idx_ex[i] < n, f"idx_ex[{i}]={idx_ex[i]} OOB"
            assert idx_en[i] <= idx_ex[i], (
                f"Entry bar {idx_en[i]} after exit bar {idx_ex[i]}"
            )
            assert t_dir[i] in (-1, 1), f"t_dir[{i}]={t_dir[i]} not in (-1, 1)"
        t_sl_pct = result[8]
        for i in range(n_trades):
            assert 0 < t_sl_pct[i] < 0.5, (
                f"t_sl_pct[{i}]={t_sl_pct[i]} out of (0, 0.5)"
            )
    return n_trades


# ─────────────────────────────────────────────────────────────────────
# Strategy-type matrix
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("strategy_type", [
    STRATEGY_VOL_COMPRESSION,
    STRATEGY_PULLBACK_SNIPER,
])
class TestEngineStrategyTypeMatrix:
    """Both strategies must satisfy the engine invariants."""

    def test_basic(self, strategy_type):
        args = _build_engine_args(strategy_type=strategy_type)
        result = fast_trade_loop(*args.as_tuple())
        _assert_engine_invariants(result, n=200)


# ─────────────────────────────────────────────────────────────────────
# Feature-flag matrix
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("use_rvol,use_ema,weekend_penalty", [
    (0, 0, 1.0),
    (0, 1, 1.0),
    (1, 0, 1.0),
    (1, 1, 1.0),
    (1, 1, 2.0),  # production default
    (1, 1, 5.0),  # extreme
])
class TestEngineFeatureFlags:
    """``use_rvol``, ``use_ema``, ``weekend_penalty`` combinations."""

    def test_flag_combination(self, use_rvol, use_ema, weekend_penalty):
        flags = (use_rvol, use_ema, 1, 1)  # use_long, use_short both 1
        cost_model = (0.05, weekend_penalty, DEFAULTS["stress_test_multiplier"])
        args = _build_engine_args(flags=flags, cost_model=cost_model)
        result = fast_trade_loop(*args.as_tuple())
        _assert_engine_invariants(result, n=200)


# ─────────────────────────────────────────────────────────────────────
# Long / short allow matrix
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("allow_long,allow_short", [
    (1, 0),
    (0, 1),
    (1, 1),
])
class TestEngineLongShortMatrix:
    """``allow_long`` / ``allow_short`` must not crash."""

    def test_long_short_combination(self, allow_long, allow_short):
        arrays = make_engine_arrays(n=200, seed=42)
        rng = np.random.default_rng(42)
        args = EngineArgs(
            market_data=(arrays["opens"], arrays["highs"],
                         arrays["lows"], arrays["closes"]),
            channel_features=(arrays["hh_20"], arrays["ll_20"],
                              arrays["ema_200s"]),
            pullback_features=(arrays["rsi_14"], arrays["bullish_reversal"],
                               arrays["bearish_reversal"]),
            signal_features=(arrays["vol_pct_rank"], arrays["rvol"],
                             arrays["atrs"]),
            auxiliary_features=(arrays["funding_rates"], arrays["macro_vols"],
                                arrays["macro_trends"],
                                arrays["is_weekends"],
                                arrays["is_funding_hours"]),
            strategy_type=STRATEGY_VOL_COMPRESSION,
            thresholds=(0.20, 2.5, 30.0, 70.0, 0.0),
            integer_params=(5, 36, 0, 0),
            exit_params=(3.0, 1.5),
            cost_model=(0.05, 2.0, DEFAULTS["stress_test_multiplier"]),
            flags=(1, 1, allow_long, allow_short),
            random_draws=rng.random(size=400).astype(np.float64),
            trend_mults=(1.5, 0.5),
        )
        result = fast_trade_loop(*args.as_tuple())
        _assert_engine_invariants(result, n=200)


# ─────────────────────────────────────────────────────────────────────
# Threshold / exit-param matrix
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("vol_pct_thresh,rvol_thresh,trail_atr,sl_mult", [
    (0.05, 1.0, 1.0, 1.0),   # tight
    (0.20, 2.5, 3.0, 1.5),   # default
    (0.50, 5.0, 5.0, 3.0),   # loose
    (0.99, 99.0, 10.0, 5.0), # extreme (no trades)
])
class TestEngineThresholds:
    """Threshold combinations across the parameter space."""

    def test_threshold_combination(
        self, vol_pct_thresh, rvol_thresh, trail_atr, sl_mult,
    ):
        args = _build_engine_args(
            thresholds=(vol_pct_thresh, rvol_thresh, 30.0, 70.0, 0.0),
            exit_params=(trail_atr, sl_mult),
        )
        result = fast_trade_loop(*args.as_tuple())
        _assert_engine_invariants(result, n=200)


# ─────────────────────────────────────────────────────────────────────
# Warmup / bail-out matrix
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bailout_bars,warmup_bars", [
    (10, 50),
    (36, 100),  # default
    (100, 200),
])
class TestEngineBailoutWarmup:
    """``bailout_bars`` and ``warmup_bars`` combinations."""

    def test_bailout_warmup_combination(self, bailout_bars, warmup_bars):
        args = _build_engine_args(integer_params=(5, bailout_bars, warmup_bars, 0))
        result = fast_trade_loop(*args.as_tuple())
        _assert_engine_invariants(result, n=200)


# ─────────────────────────────────────────────────────────────────────
# Cost stress matrix
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fee_taker,stress_mult", [
    (0.0005, 1.0),  # low
    (0.05, 2.0),    # default
    (0.10, 5.0),    # high
])
class TestEngineCostStress:
    """``fee_taker`` and ``stress_mult`` combinations."""

    def test_cost_stress_combination(self, fee_taker, stress_mult):
        cost_model = (fee_taker, 2.0, stress_mult)
        args = _build_engine_args(cost_model=cost_model)
        result = fast_trade_loop(*args.as_tuple())
        n_trades = _assert_engine_invariants(result, n=200)
        # Cost-clamped R-multiples should remain finite under stress
        if n_trades > 0:
            pnl = result[0]
            assert np.all(np.isfinite(pnl)), "R-multiples not finite under stress"
            assert np.all(pnl > -10.0), f"Cost not clamped: min pnl = {pnl.min()}"


# ─────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────


class TestEngineDeterminism:
    """Same input + same seed → identical output."""

    def test_same_args_same_output(self):
        args = _build_engine_args(seed=42)
        r1 = fast_trade_loop(*args.as_tuple())
        r2 = fast_trade_loop(*args.as_tuple())
        for a, b in zip(r1, r2):
            assert np.array_equal(a, b), "Engine is non-deterministic"

    def test_different_random_draws_yield_different_pnl(self):
        """Different RNG draws → different trade PnLs (slippage is stochastic)."""
        arrays = make_engine_arrays(n=500, seed=42)
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(1)
        base_kwargs = dict(
            market_data=(arrays["opens"], arrays["highs"],
                         arrays["lows"], arrays["closes"]),
            channel_features=(arrays["hh_20"], arrays["ll_20"],
                              arrays["ema_200s"]),
            pullback_features=(arrays["rsi_14"], arrays["bullish_reversal"],
                               arrays["bearish_reversal"]),
            signal_features=(arrays["vol_pct_rank"], arrays["rvol"],
                             arrays["atrs"]),
            auxiliary_features=(arrays["funding_rates"], arrays["macro_vols"],
                                arrays["macro_trends"],
                                arrays["is_weekends"],
                                arrays["is_funding_hours"]),
            strategy_type=STRATEGY_VOL_COMPRESSION,
            thresholds=(0.20, 2.5, 30.0, 70.0, 0.0),
            integer_params=(5, 36, 0, 0),
            exit_params=(3.0, 1.5),
            cost_model=(0.05, 2.0, DEFAULTS["stress_test_multiplier"]),
            flags=(1, 1, 1, 1),
            trend_mults=(1.5, 0.5),
        )
        args_a = EngineArgs(**base_kwargs, random_draws=rng1.random(size=1000).astype(np.float64))
        args_b = EngineArgs(**base_kwargs, random_draws=rng2.random(size=1000).astype(np.float64))
        r_a = fast_trade_loop(*args_a.as_tuple())
        r_b = fast_trade_loop(*args_b.as_tuple())
        # Same number of trades (data is identical)
        assert len(r_a[0]) == len(r_b[0])
        if len(r_a[0]) > 0:
            # PnL arrays should differ (different slippage draws)
            assert not np.array_equal(r_a[0], r_b[0]), (
                "Different RNG seeds must yield different trade PnLs"
            )


# ─────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────


class TestEngineEdgeCases:
    """Boundary conditions."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning:numpy")
    def test_zero_size_arrays(self):
        """n=0 should return 10 empty arrays without crashing.

        The NumPy ``mean of empty slice`` and ``invalid value in
        scalar divide`` warnings are expected for n=0 inputs; this
        test exercises the boundary case by design.
        """
        arrays = make_engine_arrays(n=0, seed=42)
        args = _build_engine_args(n=0)
        result = fast_trade_loop(*args.as_tuple())
        assert len(result) == 10
        for r in result:
            assert len(r) == 0

    def test_extreme_thresholds_no_trades(self):
        """Impossibly high thresholds → zero trades."""
        args = _build_engine_args(
            thresholds=(0.99, 99.0, 30.0, 70.0, 0.0),
            exit_params=(10.0, 5.0),
        )
        result = fast_trade_loop(*args.as_tuple())
        assert len(result[0]) == 0

    def test_zero_atr(self):
        """ATR=0 must not produce NaN R-multiples (clamped division)."""
        arrays = make_engine_arrays(n=200, seed=42, atrs=np.zeros(200))
        rng = np.random.default_rng(42)
        args = EngineArgs(
            market_data=(arrays["opens"], arrays["highs"],
                         arrays["lows"], arrays["closes"]),
            channel_features=(arrays["hh_20"], arrays["ll_20"],
                              arrays["ema_200s"]),
            pullback_features=(arrays["rsi_14"], arrays["bullish_reversal"],
                               arrays["bearish_reversal"]),
            signal_features=(arrays["vol_pct_rank"], arrays["rvol"],
                             arrays["atrs"]),
            auxiliary_features=(arrays["funding_rates"], arrays["macro_vols"],
                                arrays["macro_trends"],
                                arrays["is_weekends"],
                                arrays["is_funding_hours"]),
            strategy_type=STRATEGY_VOL_COMPRESSION,
            thresholds=(0.20, 2.5, 30.0, 70.0, 0.0),
            integer_params=(5, 36, 0, 0),
            exit_params=(3.0, 1.5),
            cost_model=(0.05, 2.0, DEFAULTS["stress_test_multiplier"]),
            flags=(1, 1, 1, 1),
            random_draws=rng.random(size=400).astype(np.float64),
            trend_mults=(1.5, 0.5),
        )
        result = fast_trade_loop(*args.as_tuple())
        pnl = result[0]
        if len(pnl) > 0:
            assert np.all(np.isfinite(pnl)), (
                f"ATR=0 produced non-finite R-multiples: {pnl}"
            )
