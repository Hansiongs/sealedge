"""Performance benchmarks for the engine and portfolio simulation.

These tests use ``pytest-benchmark`` to track the runtime of the
hot paths in the framework.  Run with::

    pytest --benchmark-only tests/test_perf.py
    pytest --benchmark-compare=2024_01_01 tests/test_perf.py

Each test exercises a small (n=1000) or medium (n=10000) input
on the @njit trade loop.  Baseline times will depend on the host
machine; the goal is regression detection, not absolute speed.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from quant_lib.core._config import DEFAULTS
from quant_lib.core._engine import (
    EngineArgs,
    STRATEGY_VOL_COMPRESSION,
    fast_trade_loop,
)
from quant_lib.core._portfolio import simulate_full_portfolio

from tests.conftest import make_engine_args, make_engine_arrays


class TestEnginePerf:
    """``fast_trade_loop`` perf benchmarks across input sizes."""

    def test_engine_small(self, benchmark):
        """Engine on 1000 bars: should complete in < 50ms typical."""
        args = make_engine_args(n=1000, seed=42)
        result = benchmark(fast_trade_loop, *args.as_tuple())
        # Sanity: result is well-formed
        assert len(result) == 10

    def test_engine_medium(self, benchmark):
        """Engine on 5000 bars: should complete in < 250ms typical."""
        args = make_engine_args(n=5000, seed=42)
        result = benchmark(fast_trade_loop, *args.as_tuple())
        assert len(result) == 10

    @pytest.mark.slow
    def test_engine_large(self, benchmark):
        """Engine on 20000 bars: should complete in < 1.5s typical.

        Marked ``slow`` — opt in with ``--run-slow`` or remove the
        marker in environments where 1.5s is acceptable.
        """
        args = make_engine_args(n=20000, seed=42)
        result = benchmark(fast_trade_loop, *args.as_tuple())
        assert len(result) == 10

    def test_engine_extreme_thresholds_is_fast(self, benchmark):
        """Pathological inputs (no trades) should be faster than the
        signal-rich path because the engine early-exits.
        """
        arrays = make_engine_arrays(n=1000, seed=42)
        rng = np.random.default_rng(42)
        # Force no trades via extreme thresholds
        args = EngineArgs(
            market_data=(arrays["opens"], arrays["highs"],
                         arrays["lows"], arrays["closes"]),
            channel_features=(arrays["hh_20"], arrays["ll_20"], arrays["ema_200s"]),
            pullback_features=(arrays["rsi_14"], arrays["bullish_reversal"],
                               arrays["bearish_reversal"]),
            signal_features=(arrays["vol_pct_rank"], arrays["rvol"], arrays["atrs"]),
            auxiliary_features=(arrays["funding_rates"], arrays["macro_vols"],
                                arrays["macro_trends"], arrays["is_weekends"],
                                arrays["is_funding_hours"]),
            strategy_type=STRATEGY_VOL_COMPRESSION,
            thresholds=(0.99, 99.0, 30.0, 70.0, 0.0),
            integer_params=(5, 36, 0, 0),
            exit_params=(10.0, 5.0),
            cost_model=(0.05, 2.0, DEFAULTS["stress_test_multiplier"]),
            flags=(1, 1, 1, 1),
            random_draws=rng.random(size=2000).astype(np.float64),
            trend_mults=(1.5, 0.5),
        )
        result = benchmark(fast_trade_loop, *args.as_tuple())
        assert len(result[0]) == 0


class TestPortfolioPerf:
    """``simulate_full_portfolio`` perf benchmarks."""

    def test_portfolio_small(self, benchmark):
        """Portfolio sim with 50 trades: should complete in < 50ms."""
        rng = np.random.default_rng(42)
        trades = []
        base_time = pd.Timestamp("2022-01-01")
        for i in range(50):
            entry = base_time + pd.Timedelta(days=i)
            trades.append({
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta(days=2),
                "symbol": "BTCUSDT",
                "r_net": float(rng.normal(0.1, 0.5)),
                "trade_dir": 1,
                "entry_price": 100.0,
                "exit_price": 101.0,
                "sl_pct": 0.02,
                "sl_mult": 1.5,
                "trail_atr": 3.0,
                "m_trend": 1,
                "macro_vol": 0.5,
                "risk_weight": 0.01,
                "trend_risk_mult": 1.0,
            })
        # Build a daily close matrix for the test period
        dates = pd.date_range("2022-01-01", "2022-12-31", freq="D")
        close_matrix = {
            "BTCUSDT": {d: 100.0 + i * 0.1 for i, d in enumerate(dates)},
        }
        eq, daily_eq, executed, reject = benchmark(
            simulate_full_portfolio,
            trades=trades,
            initial_cash=1000.0,
            leverage=3.0,
            mm_pct=0.01,
            position_limit=4,
            cb_hard_cooldown_hours=24,
            fixed_cb_threshold=0.15,
            daily_close_matrix=close_matrix,
            asset_risk_weights={"BTCUSDT": 0.01},
            end_date="2023-01-01",
        )
        assert eq > 0
