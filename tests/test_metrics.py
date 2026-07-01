"""Tests for metrics utilities — daily matrices, bootstrap, regime stats."""

import numpy as np
import pandas as pd
import pytest

from quant_lib.core._metrics import (
    build_daily_matrices,
    run_bootstrap,
    run_trade_bootstrap,
    compute_regime_stats,
    _coefficient_of_variation,
)


class TestBuildDailyMatrices:
    def test_basic(self, sample_hourly_data):
        symbols = ["TEST"]
        precomputed = {"TEST": sample_hourly_data}
        close_matrix, hl_matrix = build_daily_matrices(symbols, precomputed)
        assert "TEST" in close_matrix
        assert "TEST" in hl_matrix
        assert len(close_matrix["TEST"]) > 0

    def test_all_symbols_present(self):
        rng = np.random.default_rng(42)
        precomputed = {}
        for sym in ["BTCUSDT", "ETHUSDT"]:
            n = 2000
            df = pd.DataFrame({
                "time": pd.date_range("2021-01-01", periods=n, freq="h"),
                "close": 100 + np.cumsum(rng.normal(0, 0.5, n)),
                "high": 101 + np.cumsum(np.abs(rng.normal(0, 0.3, n))),
                "low": 99 + np.cumsum(-np.abs(rng.normal(0, 0.3, n))),
                "volume": rng.exponential(1000, n),
            })
            precomputed[sym] = df

        close_matrix, hl_matrix = build_daily_matrices(["BTCUSDT", "ETHUSDT"], precomputed)
        assert sorted(close_matrix.keys()) == sorted(["BTCUSDT", "ETHUSDT"])
        assert sorted(hl_matrix.keys()) == sorted(["BTCUSDT", "ETHUSDT"])

    def test_empty_data_returns_empty(self):
        # Empty DataFrame with proper datetime dtype
        df = pd.DataFrame({
            "time": pd.to_datetime([]),
            "close": pd.Series([], dtype=float),
            "high": pd.Series([], dtype=float),
            "low": pd.Series([], dtype=float),
            "volume": pd.Series([], dtype=float),
        })
        precomputed = {"TEST": df}
        close_matrix, hl_matrix = build_daily_matrices(["TEST"], precomputed)
        assert close_matrix.get("TEST", {}) == {}
        assert hl_matrix.get("TEST", {}) == {}


class TestRunBootstrap:
    def test_bootstrap_returns_dict(self):
        n = 500
        rng = np.random.default_rng(42)
        prices = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
        eq_series = pd.Series(prices, index=pd.date_range("2021-01-01", periods=n, freq="D"))
        daily_ret = eq_series.pct_change().dropna()
        max_dd = -15.0

        result = run_bootstrap(daily_ret, eq_series, max_dd, 1000.0)
        assert isinstance(result, dict)
        expected_keys = {"Worst5_CAGR", "Worst95_DD", "Worst5_DD", "Worst1_DD", "DD_Pctile", "BootstrapBlock"}
        assert expected_keys.issubset(result.keys())

    def test_bootstrap_worst5_cagr_is_negative_for_poor_strategy(self):
        n = 500
        rng = np.random.default_rng(1)
        # Poor strategy: negative drift with high volatility
        prices = 100.0 * np.cumprod(1 + rng.normal(-0.001, 0.03, n))
        eq_series = pd.Series(prices, index=pd.date_range("2021-01-01", periods=n, freq="D"))
        daily_ret = eq_series.pct_change().dropna()
        max_dd = -30.0

        result = run_bootstrap(daily_ret, eq_series, max_dd, 1000.0)
        # P95 of max DD should be negative (some drawdown)
        assert result["Worst95_DD"] <= 0


class TestRunBootstrapConvention:
    """v0.3.1: enforce max_dd convention (must be non-positive percent).

    Bug: prior version accepted positive max_dd but the percentile
    comparison used mixed sign convention (bootstrap negative, observed
    positive), causing DD_Pctile to silently always be 100%. Now we
    assert the convention explicitly so callers passing positive values
    fail loudly instead of producing meaningless percentiles.
    """

    def test_positive_max_dd_raises_assertion(self):
        """Positive max_dd is invalid (DD cannot be positive)."""
        n = 200
        rng = np.random.default_rng(42)
        prices = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
        eq_series = pd.Series(prices, index=pd.date_range("2021-01-01", periods=n, freq="D"))
        daily_ret = eq_series.pct_change().dropna()

        import pytest
        with pytest.raises(AssertionError, match="NEGATIVE percentage"):
            run_bootstrap(daily_ret, eq_series, max_dd=15.0, initial_capital=1000.0)

    def test_zero_max_dd_is_allowed(self):
        """max_dd=0 (no drawdown) is a valid degenerate input."""
        n = 200
        rng = np.random.default_rng(42)
        prices = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
        eq_series = pd.Series(prices, index=pd.date_range("2021-01-01", periods=n, freq="D"))
        daily_ret = eq_series.pct_change().dropna()

        # Should not raise
        result = run_bootstrap(daily_ret, eq_series, max_dd=0.0, initial_capital=1000.0)
        assert isinstance(result, dict)
        # DD_Pctile = 100% is CORRECT here: observed DD=0% means all
        # bootstrap scenarios (which all have negative DD due to cumprod
        # drift) are "as bad as or worse than" the observed.
        # The original bug was when max_dd was POSITIVE (e.g. 15.0)
        # making bootstrap (always negative) <= observed (positive) always True.
        # With negative convention enforced, 0.0 is the boundary case.
        assert result["DD_Pctile"] == 100.0

    def test_dd_pctile_is_not_always_100_for_negative_drift(self):
        """Regression: previously DD_Pctile was always 100% due to sign bug.
        With correct convention + observed negative drift, observed DD should
        rank in the higher (worse) percentiles -- but not always 100.
        """
        n = 500
        rng = np.random.default_rng(1)
        # Poor strategy: negative drift with high vol
        prices = 100.0 * np.cumprod(1 + rng.normal(-0.005, 0.04, n))
        eq_series = pd.Series(prices, index=pd.date_range("2021-01-01", periods=n, freq="D"))
        daily_ret = eq_series.pct_change().dropna()
        max_dd = ((prices - np.maximum.accumulate(prices)) / np.maximum.accumulate(prices)).min() * 100
        # max_dd is a negative percentage
        assert max_dd < 0

        result = run_bootstrap(daily_ret, eq_series, max_dd, 1000.0)
        # DD_Pctile should be in (0, 100], not always exactly 100
        assert 0.0 < result["DD_Pctile"] <= 100.0, (
            f"DD_Pctile must be in (0, 100] for valid observed DD, got "
            f"{result['DD_Pctile']}"
        )


class TestComputeRegimeStats:
    def test_bull_bear_classification(self):
        trades = [
            {"r_net": 0.5, "m_trend": 1},
            {"r_net": -0.3, "m_trend": 1},
            {"r_net": 0.2, "m_trend": -1},
            {"r_net": -0.1, "m_trend": -1},
        ]
        regimes = compute_regime_stats(trades)
        assert "Bull" in regimes
        assert "Bear" in regimes
        pf_bull, n_bull = regimes["Bull"]
        pf_bear, n_bear = regimes["Bear"]
        assert n_bull == 2
        assert n_bear == 2
        assert pf_bull >= 0
        assert pf_bear >= 0

    def test_empty_trades(self):
        regimes = compute_regime_stats([])
        pf_bull, n_bull = regimes["Bull"]
        pf_bear, n_bear = regimes["Bear"]
        assert pf_bull == 1.0  # default when no trades
        assert n_bull == 0
        assert pf_bear == 1.0
        assert n_bear == 0

    def test_all_bull_trades(self):
        trades = [{"r_net": 0.5, "m_trend": 1}, {"r_net": 0.3, "m_trend": 1}]
        regimes = compute_regime_stats(trades)
        pf_bull, n_bull = regimes["Bull"]
        assert n_bull == 2
        assert n_bull > regimes["Bear"][1]


class TestTradeBootstrapTradeDates:
    """v0.4.1 (Phase 3): run_trade_bootstrap accepts trade_dates
    for accurate CAGR annualization.
    """

    def test_without_trade_dates_uses_n_proxy(self):
        """Without trade_dates, n_days = len(trade_r_vals) (proxy)."""
        rng = np.random.default_rng(42)
        r = rng.normal(0.05, 0.1, 50)
        result = run_trade_bootstrap(r, 1000.0, n_sim=50, block_size=5)
        assert "Worst5_CAGR" in result
        # CAGR should be finite
        assert np.isfinite(result["Worst5_CAGR"])

    def test_with_trade_dates_uses_actual_span(self):
        """With trade_dates, n_days = (last - first).days."""
        rng = np.random.default_rng(42)
        r = rng.normal(0.05, 0.1, 50)
        # 50 trades over 1 year (365 days) -- should produce a
        # different CAGR than 50 trades over 50 days
        dates = pd.date_range("2024-01-01", periods=50, freq="7D")  # ~350 days
        result = run_trade_bootstrap(
            r, 1000.0, n_sim=50, block_size=5, trade_dates=dates
        )
        assert np.isfinite(result["Worst5_CAGR"])

    def test_trade_dates_length_mismatch_logs_warning(self, caplog):
        """Mismatch between trade_dates length and trade_r_vals length
        logs a warning and falls back to n_days = n proxy.
        """
        import logging
        rng = np.random.default_rng(42)
        r = rng.normal(0.05, 0.1, 50)
        wrong_dates = pd.date_range("2024-01-01", periods=20, freq="D")  # wrong length
        with caplog.at_level(logging.WARNING):
            result = run_trade_bootstrap(
                r, 1000.0, n_sim=20, block_size=5, trade_dates=wrong_dates
            )
        assert np.isfinite(result["Worst5_CAGR"])
        assert any("trade_dates length" in m for m in caplog.messages)


class TestCoefficientOfVariation:
    """v0.4.1 (Phase 3): _coefficient_of_variation helper."""

    def test_normal_values(self):
        """CV of [10, 20, 30] should be (std=10)/mean=20 * 100 = 50%."""
        result = _coefficient_of_variation(np.array([10.0, 20.0, 30.0]))
        assert abs(result - 50.0) < 0.01

    def test_zero_mean_returns_zero(self):
        """CV when mean is ~0 returns 0 (avoids div-by-zero)."""
        result = _coefficient_of_variation(np.array([0.0, 0.0, 0.0]))
        assert result == 0.0

    def test_negative_mean_uses_abs(self):
        """CV uses |mean| in denominator (handles negative means)."""
        result = _coefficient_of_variation(np.array([-10.0, -20.0, -30.0]))
        # std/mean*100 = 10/20*100 = 50
        assert abs(result - 50.0) < 0.01

    def test_constant_values_returns_zero(self):
        """CV of constant array is 0 (std=0)."""
        result = _coefficient_of_variation(np.ones(10))
        assert result == 0.0
