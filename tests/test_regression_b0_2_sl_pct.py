"""Regression tests for B0.2: ``sl_pct <= 0`` guard.

Bug: ``core/_portfolio.py`` computed ``notional = risk_capital / sl_pct``
without a guard. Passing ``sl_pct=0`` caused a silent ``ZeroDivisionError``
deep inside the portfolio simulation loop, producing a confusing traceback
for users who constructed trades with missing or zero stop-loss distances.

Fix: raise ``ValueError`` early at trade entry with a clear message
identifying the offending trade.
"""
from __future__ import annotations


import pandas as pd
import pytest

from quant_lib.core._portfolio import simulate_full_portfolio
from quant_lib.core._config import DEFAULTS


def _trade(sl_pct: float, entry_offset_days: int = 0) -> dict:
    """Build a minimal valid trade dict for portfolio sim."""
    entry = pd.Timestamp("2024-01-01") + pd.Timedelta(days=entry_offset_days)
    exit_t = entry + pd.Timedelta(days=1)
    return {
        "entry_time": entry,
        "exit_time": exit_t,
        "symbol": "BTCUSDT",
        "entry_price": 100.0,
        "exit_price": 101.0,
        "trade_dir": 1,
        "sl_pct": sl_pct,
        "sl_mult": 1.5,
        "r_net": 1.0,
        "risk_weight": 0.01,
        "trend_risk_mult": 1.0,
    }


def _daily_close_matrix(trade: dict) -> dict:
    """Build a minimal daily_close_matrix covering the trade window."""
    sym = trade["symbol"]
    entry = trade["entry_time"]
    exit_t = trade["exit_time"]
    dates = pd.date_range(entry, exit_t, freq="D")
    return {sym: {d: 100.0 for d in dates}}


def _daily_hl_matrix(trade: dict) -> dict:
    sym = trade["symbol"]
    entry = trade["entry_time"]
    exit_t = trade["exit_time"]
    dates = pd.date_range(entry, exit_t, freq="D")
    return {sym: {d: {"high": 101.0, "low": 99.0} for d in dates}}


class TestSlPctGuard:
    """B0.2 fix: ``sl_pct <= 0`` must raise ``ValueError``."""

    def test_sl_pct_zero_raises(self):
        """sl_pct=0 must raise ValueError (not ZeroDivisionError)."""
        trade = _trade(sl_pct=0.0)
        with pytest.raises(ValueError, match="sl_pct must be > 0"):
            simulate_full_portfolio(
                trades=[trade],
                initial_cash=1000.0,
                leverage=3.0,
                mm_pct=DEFAULTS["maintenance_margin_pct"] if "maintenance_margin_pct" in DEFAULTS else 0.01,
                position_limit=4,
                cb_hard_cooldown_hours=24,
                fixed_cb_threshold=0.15,
                daily_close_matrix=_daily_close_matrix(trade),
                asset_risk_weights={trade["symbol"]: 0.01},
                end_date="2024-01-02",
                daily_hl_matrix=_daily_hl_matrix(trade),
            )

    def test_sl_pct_negative_raises(self):
        """sl_pct<0 must also raise (not silently produce nonsense)."""
        trade = _trade(sl_pct=-0.01)
        with pytest.raises(ValueError, match="sl_pct must be > 0"):
            simulate_full_portfolio(
                trades=[trade],
                initial_cash=1000.0,
                leverage=3.0,
                mm_pct=0.01,
                position_limit=4,
                cb_hard_cooldown_hours=24,
                fixed_cb_threshold=0.15,
                daily_close_matrix=_daily_close_matrix(trade),
                asset_risk_weights={trade["symbol"]: 0.01},
                end_date="2024-01-02",
                daily_hl_matrix=_daily_hl_matrix(trade),
            )

    def test_sl_pct_positive_does_not_raise(self):
        """A valid sl_pct must NOT trigger the guard (regression sanity)."""
        trade = _trade(sl_pct=0.02)
        # Should not raise -- the trade is processed normally.
        # The trade may still be rejected (margin, position limit) but
        # the sl_pct guard is not the source.
        try:
            simulate_full_portfolio(
                trades=[trade],
                initial_cash=1000.0,
                leverage=3.0,
                mm_pct=0.01,
                position_limit=4,
                cb_hard_cooldown_hours=24,
                fixed_cb_threshold=0.15,
                daily_close_matrix=_daily_close_matrix(trade),
                asset_risk_weights={trade["symbol"]: 0.01},
                end_date="2024-01-02",
                daily_hl_matrix=_daily_hl_matrix(trade),
            )
        except ValueError as e:
            if "sl_pct" in str(e):
                pytest.fail(f"Valid sl_pct triggered guard: {e}")
