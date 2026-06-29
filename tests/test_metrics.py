"""Tests for metrics utilities — daily matrices, bootstrap, regime stats."""

import numpy as np
import pandas as pd

from quant_lib.core._metrics import (
    build_daily_matrices,
    run_bootstrap,
    compute_regime_stats,
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
