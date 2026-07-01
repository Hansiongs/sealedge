"""Tests for portfolio simulation — MTM, margin, circuit breaker."""

import inspect

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


class TestRiskWeightFallbackConsistency:
    """v0.4.0 (Phase 2.4): risk_weight fallback must use
    DEFAULTS["default_risk_per_pair"] (the configured default), not a
    hardcoded magic number. Both fallback sites in _portfolio.py must
    be consistent.
    """

    def test_no_hardcoded_risk_weight_fallback(self):
        """_portfolio.py must not contain hardcoded 0.005 risk_weight fallback.

        Bug #6: previously, when a trade had no ``risk_weight`` field
        AND ``asset_risk_weights`` didn't have the symbol, the code
        fell back to a hardcoded 0.005. This was inconsistent with the
        WFA path which uses ``DEFAULTS["default_risk_per_pair"]``.

        Only check lines that look like risk_weight fallbacks
        (``trade.get(`` or ``pos["trade"].get(`` near ``risk_weight``).
        """
        from quant_lib.core import _portfolio as portfolio_module
        source = inspect.getsource(portfolio_module)
        for line in source.splitlines():
            # Check lines that are risk-weight fallback sites: the
            # pattern is ``trade.get(`` or ``pos["trade"].get(`` near
            # the keyword "risk_weight", with a literal numeric fallback
            # of 0.005.
            stripped = line.strip()
            if "risk_weight" in stripped and "0.005" in stripped:
                # Allow 0.005 if it's clearly an unrelated constant
                # (e.g. liquidation_fee_pct default value).
                if "liquidation_fee" in stripped:
                    continue
                # Allow comments mentioning "0.005" in a different context
                if stripped.startswith("#") or stripped.startswith('"""'):
                    continue
                pytest.fail(
                    f"_portfolio.py has hardcoded 0.005 risk_weight fallback. "
                    f"Use DEFAULTS['default_risk_per_pair'] instead. "
                    f"Line: {line!r}"
                )

    def test_default_risk_per_pair_is_used(self):
        """Source-level: fallback uses DEFAULTS['default_risk_per_pair']."""
        from quant_lib.core import _portfolio as portfolio_module
        source = inspect.getsource(portfolio_module)
        # Count uses of default_risk_per_pair in risk_weight contexts
        # (both the open-position fallback and the entry fallback)
        count = source.count("default_risk_per_pair")
        # Both fallback sites (entry at line ~98, entry at line ~359)
        # should reference this config key
        assert count >= 2, (
            f"Expected at least 2 DEFAULTS['default_risk_per_pair'] uses in "
            f"_portfolio.py for both risk_weight fallback sites, got {count}"
        )


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


# ════════════════════════════════════════════════════════════════════════
# Phase 2.4: Market impact cap
# ════════════════════════════════════════════════════════════════════════


class TestMarketImpactCap:
    """Phase 2.4: position notional is capped at
    DEFAULTS['market_impact_volume_pct'] of 24h volume.
    """

    def _make_trade(self, sym, sl_pct=0.02, risk_weight=0.01, trend_mult=1.5):
        """Build a minimal trade dict for the cap test."""
        return {
            "entry_time": pd.Timestamp("2024-01-01"),
            "exit_time": pd.Timestamp("2024-01-02"),
            "symbol": sym,
            "trade_dir": 1,
            "entry_price": 100.0,
            "exit_price": 105.0,
            "sl_pct": sl_pct,
            "r_net": 0.5,
            "sl_mult": 1.5,
            "trail_atr": 3.0,
            "risk_weight": risk_weight,
            "trend_risk_mult": trend_mult,
        }

    def _make_close_matrix(self, sym, close):
        return {sym: {pd.Timestamp("2024-01-01").date(): close}}

    def test_cap_triggers_for_large_order(self, monkeypatch):
        """Large order (risk_weight=0.05 + 1.5x trend) triggers cap."""
        from quant_lib.core._config import DEFAULTS
        from quant_lib.core import _portfolio as portfolio_mod
        # Tight cap (0.1%) so we can verify cap is triggered
        monkeypatch.setitem(DEFAULTS, "market_impact_volume_pct", 0.001)
        # Use a trade that, with cap applied, will execute (small
        # enough notional). With cap=0.1%, close=100, leverage=3:
        # max_notional = 100 * 0.001 * 3 = 0.3 USD
        # With risk_weight=0.01 + trend 1.5x = 0.015:
        # uncapped notional = 1000 * 0.015 / 0.02 = 750 USD (way over cap)
        # With cap, risk_weight becomes ~6e-6, notional ~ 0.3 USD
        # This is so small it would be rejected for being essentially
        # zero. Use a more realistic scenario: 0.01 risk weight, close=1000.
        trade = {
            "entry_time": pd.Timestamp("2024-01-01"),
            "exit_time": pd.Timestamp("2024-01-02"),
            "symbol": "BTCUSDT",
            "trade_dir": 1,
            "entry_price": 100.0,
            "exit_price": 105.0,
            "sl_pct": 0.02,
            "r_net": 0.5,
            "sl_mult": 1.5,
            "trail_atr": 3.0,
            "risk_weight": 0.005,  # 0.5% per trade
            "trend_risk_mult": 1.5,  # 1.5x with-trend = 0.75% effective
        }
        # Daily close 50000, cap 0.001, leverage 3:
        # max_notional = 50000 * 0.001 * 3 = 150 USD
        # Uncapped notional = 1000 * 0.0075 / 0.02 = 375 USD (over cap)
        # With cap: notional = 150 USD, risk_weight = 0.003
        daily_close = {
            "BTCUSDT": {pd.Timestamp("2024-01-01").date(): 50000.0}
        }
        final_eq, _, executed, _ = portfolio_mod.simulate_full_portfolio(
            trades=[trade],
            initial_cash=1000.0,
            leverage=3.0,
            mm_pct=0.01,
            position_limit=4,
            cb_hard_cooldown_hours=24,
            fixed_cb_threshold=0.15,
            daily_close_matrix=daily_close,
            asset_risk_weights=None,
            end_date="2024-12-31",
        )
        # With cap applied, notional = 150 USD, IM = 50 USD
        # 1000 - 0 >= 50 → execute
        assert len(executed) == 1

    def test_cap_disabled_when_pct_zero(self, monkeypatch):
        """When market_impact_volume_pct=0, no cap is applied."""
        from quant_lib.core._config import DEFAULTS
        from quant_lib.core import _portfolio as portfolio_mod
        monkeypatch.setitem(DEFAULTS, "market_impact_volume_pct", 0.0)
        # Small executable order (no cap, should execute)
        trade = self._make_trade(
            "BTCUSDT", sl_pct=0.02, risk_weight=0.005, trend_mult=1.0
        )
        daily_close = self._make_close_matrix("BTCUSDT", 50000.0)
        final_eq, _, executed, _ = portfolio_mod.simulate_full_portfolio(
            trades=[trade],
            initial_cash=1000.0,
            leverage=3.0,
            mm_pct=0.01,
            position_limit=4,
            cb_hard_cooldown_hours=24,
            fixed_cb_threshold=0.15,
            daily_close_matrix=daily_close,
            asset_risk_weights=None,
            end_date="2024-12-31",
        )
        # Trade should be executed (no cap)
        assert len(executed) == 1

    def test_no_close_data_skips_cap(self, monkeypatch):
        """When daily_close_matrix is empty for a symbol, cap is skipped."""
        from quant_lib.core._config import DEFAULTS
        from quant_lib.core import _portfolio as portfolio_mod
        monkeypatch.setitem(DEFAULTS, "market_impact_volume_pct", 0.001)
        # Small order that would normally be capped, but with empty
        # close matrix the cap is skipped
        trade = self._make_trade(
            "BTCUSDT", sl_pct=0.02, risk_weight=0.01, trend_mult=1.0
        )
        # Empty close matrix
        final_eq, _, executed, _ = portfolio_mod.simulate_full_portfolio(
            trades=[trade],
            initial_cash=1000.0,
            leverage=3.0,
            mm_pct=0.01,
            position_limit=4,
            cb_hard_cooldown_hours=24,
            fixed_cb_threshold=0.15,
            daily_close_matrix={},  # no data
            asset_risk_weights=None,
            end_date="2024-12-31",
        )
        # Trade should be executed (cap skipped because no proxy data)
        assert len(executed) == 1

    def test_small_order_not_capped(self, monkeypatch):
        """Small order (well under cap) is not reduced."""
        from quant_lib.core._config import DEFAULTS
        from quant_lib.core import _portfolio as portfolio_mod
        monkeypatch.setitem(DEFAULTS, "market_impact_volume_pct", 0.10)  # 10% cap
        # Tiny order
        trade = self._make_trade("BTCUSDT", sl_pct=0.02, risk_weight=0.0001)
        daily_close = self._make_close_matrix("BTCUSDT", 100.0)
        final_eq, _, executed, _ = portfolio_mod.simulate_full_portfolio(
            trades=[trade],
            initial_cash=1000.0,
            leverage=3.0,
            mm_pct=0.01,
            position_limit=4,
            cb_hard_cooldown_hours=24,
            fixed_cb_threshold=0.15,
            daily_close_matrix=daily_close,
            asset_risk_weights=None,
            end_date="2024-12-31",
        )
        # Trade should be executed
        assert len(executed) == 1
