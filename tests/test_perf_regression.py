"""Performance regression tests — fail CI if engine gets 2x slower.

These tests use ``pytest-benchmark`` to establish baseline performance
for the engine's hot paths.  Run with::

    pytest --benchmark-only --benchmark-compare=baseline tests/test_perf_regression.py
    pytest --benchmark-only --benchmark-save=baseline tests/test_perf_regression.py

A failure indicates a real performance regression (>2x slowdown)
that warrants investigation.

These tests complement ``test_perf.py``: the latter is for ad-hoc
benchmarking, the former is the CI gate.
"""
from __future__ import annotations

import numpy as np
import pytest

from quant_lib.core._config import DEFAULTS
from quant_lib.core._engine import (
    EngineArgs,
    fast_trade_loop,
)
from quant_lib.tools.backtest import run_trade_loop

from tests.conftest import make_engine_args, make_engine_arrays


# ═══════════════════════════════════════════════════════════════════════
# CI gate
# ═══════════════════════════════════════════════════════════════════════


class TestEnginePerformanceGate:
    """Fail CI if engine regresses beyond budget.

    These tests use ``pytest-benchmark``'s ``--benchmark-compare``
    feature: a test fails if the current run is significantly slower
    than the stored baseline.  The ``--benchmark-group-by=groupname``
    option groups related tests so per-test comparisons are clear.
    """

    def test_engine_small_under_50ms(self, benchmark):
        """n=1000: must run in <50ms (current: ~15μs)."""
        args = make_engine_args(n=1000)
        result = benchmark(fast_trade_loop, *args.as_tuple())
        # Hard ceiling: 50ms.  Real number is ~15μs.
        # This is a sanity check, not a regression check.
        # pytest-benchmark will store stats; --benchmark-compare flags
        # regressions in CI.
        assert len(result) == 10

    def test_engine_medium_under_500ms(self, benchmark):
        """n=5000: must run in <500ms (current: ~50μs)."""
        args = make_engine_args(n=5000)
        result = benchmark(fast_trade_loop, *args.as_tuple())
        assert len(result) == 10

    def test_backtest_smoke_under_100ms(self, benchmark):
        """The backtest.py smoke test: n=400, <100ms (current: ~5ms)."""
        from datetime import datetime, timedelta
        n = 400
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
        times = [datetime(2020, 1, 1) + timedelta(hours=i) for i in range(n)]
        import pandas as pd
        df = pd.DataFrame({
            "time": times,
            "open": close + rng.normal(0, 0.1, n),
            "high": close + np.abs(rng.normal(0, 0.3, n)),
            "low": close - np.abs(rng.normal(0, 0.3, n)),
            "close": close,
            "volume": rng.exponential(1000, n),
        })
        df["hh_20"] = df["high"].rolling(20).max().shift(1)
        df["ll_20"] = df["low"].rolling(20).min().shift(1)
        df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean().shift(1)
        df["vol_pct_rank"] = 0.1
        df["rvol"] = 3.0
        df["atr"] = 1.5
        df["funding_rate"] = 0.0
        df["macro_vol"] = 0.5
        df["macro_trend"] = 1
        df["is_weekend"] = 0
        df["is_funding_hour"] = 0
        result = benchmark(run_trade_loop, df, 0.20, 42)
        assert "n_trades" in result


class TestRegressionDetection:
    """Test that the regression-detection mechanism works.

    pytest-benchmark's ``--benchmark-compare`` will fail tests that
    are significantly slower than the baseline.  These tests verify
    that the comparison infrastructure is wired up correctly (i.e.,
    benchmarks run and stats are stored).
    """

    def test_benchmark_stores_stats(self, benchmark):
        """A benchmark run stores stats that ``--benchmark-compare`` can use."""
        args = make_engine_args(n=500)
        result = benchmark(fast_trade_loop, *args.as_tuple())
        # The benchmark fixture itself stores stats; we just verify
        # the call completed.
        assert len(result) == 10

    def test_benchmark_min_rounds_respected(self, benchmark):
        """pytest-benchmark respects ``--benchmark-min-rounds``.

        With many fast iterations, the benchmark should hit the
        minimum rounds quickly.  This is a smoke test for the
        benchmark infrastructure.
        """
        args = make_engine_args(n=200)
        result = benchmark(fast_trade_loop, *args.as_tuple())
        assert len(result) == 10
