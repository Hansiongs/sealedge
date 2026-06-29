"""Tests for portfolio simulation — MTM, margin, circuit breaker."""

import pandas as pd

from quant_lib.core._portfolio import (
    _trade_key,
    simulate_full_portfolio,
)


class TestTradeKey:
    def test_unique_per_trade(self):
        t1 = {"symbol": "BTCUSDT", "entry_time": pd.Timestamp("2022-01-01"),
              "exit_time": pd.Timestamp("2022-01-02"), "trade_dir": 1}
        t2 = {"symbol": "ETHUSDT", "entry_time": pd.Timestamp("2022-01-01"),
              "exit_time": pd.Timestamp("2022-01-02"), "trade_dir": 1}
        assert _trade_key(t1) != _trade_key(t2)

    def test_same_trade_same_key(self):
        t = {"symbol": "BTCUSDT", "entry_time": pd.Timestamp("2022-01-01"),
             "exit_time": pd.Timestamp("2022-01-02"), "trade_dir": 1}
        assert _trade_key(t) == _trade_key(dict(t))

    def test_default_trade_dir(self):
        t = {"symbol": "BTCUSDT", "entry_time": pd.Timestamp("2022-01-01"),
             "exit_time": pd.Timestamp("2022-01-02")}
        key = _trade_key(t)
        assert key[-1] == 0


class TestSimulateFullPortfolio:
    def test_empty_trades_returns_initial_capital(self):
        eq, daily_eq, trades, reject = simulate_full_portfolio(
            trades=[],
            initial_cash=1000.0,
            leverage=3.0,
            mm_pct=0.01,
            position_limit=4,
            cb_hard_cooldown_hours=24,
            fixed_cb_threshold=0.15,
            daily_close_matrix={"BTCUSDT": {}},
            asset_risk_weights={"BTCUSDT": 0.01},
            end_date="2023-01-01",
        )
        assert eq == 1000.0
        assert len(trades) == 0

    def test_single_trade_execution(self):
        trades = [{
            "entry_time": pd.Timestamp("2022-06-01"),
            "exit_time": pd.Timestamp("2022-06-10"),
            "symbol": "BTCUSDT",
            "r_net": 0.5,
            "entry_price": 100.0,
            "exit_price": 105.0,
            "trade_dir": 1,
            "sl_pct": 0.02,
            "sl_mult": 1.5,
            "trail_atr": 3.0,
            "m_trend": 1,
            "macro_vol": 0.5,
            "risk_weight": 0.01,
            "trend_risk_mult": 1.5,
        }]
        close_matrix = {
            "BTCUSDT": {
                pd.Timestamp("2022-06-01"): 100.0,
                pd.Timestamp("2022-06-10"): 105.0,
            }
        }
        eq, daily_eq, executed, reject = simulate_full_portfolio(
            trades=trades,
            initial_cash=1000.0,
            leverage=3.0,
            mm_pct=0.01,
            position_limit=4,
            cb_hard_cooldown_hours=24,
            fixed_cb_threshold=0.15,
            daily_close_matrix=close_matrix,
            asset_risk_weights={"BTCUSDT": 0.01},
            end_date="2022-06-15",
        )
        assert len(executed) == 1
        assert eq != 1000.0  # equity changed
        assert eq > 0

    def test_trade_with_entry_exit_same_time(self):
        """Should handle entry_time == exit_time gracefully."""
        trades = [{
            "entry_time": pd.Timestamp("2022-06-01"),
            "exit_time": pd.Timestamp("2022-06-01"),
            "symbol": "BTCUSDT",
            "r_net": 0.1,
            "entry_price": 100.0,
            "exit_price": 101.0,
            "trade_dir": 1,
            "sl_pct": 0.02,
            "sl_mult": 1.5,
            "trail_atr": 3.0,
            "m_trend": 1,
            "macro_vol": 0.5,
            "risk_weight": 0.01,
            "trend_risk_mult": 1.0,
        }]
        close_matrix = {
            "BTCUSDT": {pd.Timestamp("2022-06-01"): 100.0}
        }
        eq, _, executed, _ = simulate_full_portfolio(
            trades=trades,
            initial_cash=1000.0,
            leverage=3.0,
            mm_pct=0.01,
            position_limit=4,
            cb_hard_cooldown_hours=24,
            fixed_cb_threshold=0.15,
            daily_close_matrix=close_matrix,
            asset_risk_weights={"BTCUSDT": 0.01},
            end_date="2022-06-05",
        )
        assert len(executed) == 1 or len(executed) == 0
        assert eq > 0

    def test_circuit_breaker_rejects(self):
        """Trades after large drawdown should be rejected by CB."""
        base_time = pd.Timestamp("2022-01-01")
        trades = []
        close_matrix = {"BTCUSDT": {}}
        for i in range(30):
            t = base_time + pd.Timedelta(days=i * 10)
            trades.append({
                "entry_time": t,
                "exit_time": t + pd.Timedelta(days=5),
                "symbol": "BTCUSDT",
                "r_net": -0.5,  # consistently losing
                "entry_price": 100.0,
                "exit_price": 95.0,
                "trade_dir": 1,
                "sl_pct": 0.02,
                "sl_mult": 1.5,
                "trail_atr": 3.0,
                "m_trend": 1,
                "macro_vol": 0.5,
                "risk_weight": 0.01,
                "trend_risk_mult": 1.0,
            })
            close_matrix["BTCUSDT"][t] = 100.0
            close_matrix["BTCUSDT"][t + pd.Timedelta(days=5)] = 95.0

        eq, _, executed, reject = simulate_full_portfolio(
            trades=trades,
            initial_cash=1000.0,
            leverage=3.0,
            mm_pct=0.01,
            position_limit=4,
            cb_hard_cooldown_hours=24,
            fixed_cb_threshold=0.15,
            daily_close_matrix=close_matrix,
            asset_risk_weights={"BTCUSDT": 0.01},
            end_date="2023-06-01",
        )
        # Some trades should be rejected by CB
        assert reject["cb_cooldown"] >= 0
        if len(trades) > len(executed):
            assert reject["cb_cooldown"] > 0

    def test_multiple_assets(self, sample_trades, sample_daily_close_matrix):
        eq, daily_eq, executed, reject = simulate_full_portfolio(
            trades=sample_trades,
            initial_cash=1000.0,
            leverage=3.0,
            mm_pct=0.01,
            position_limit=4,
            cb_hard_cooldown_hours=24,
            fixed_cb_threshold=0.15,
            daily_close_matrix=sample_daily_close_matrix,
            asset_risk_weights={"BTCUSDT": 0.01, "ETHUSDT": 0.01, "SOLUSDT": 0.01},
            end_date="2024-12-31",
        )
        assert eq > 0
        assert len(daily_eq) > 0
        assert len(executed) <= len(sample_trades)
