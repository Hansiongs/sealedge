"""Regression tests for B0.2: ``sl_pct <= 0`` guard.

Bug: ``core/_portfolio.py`` computed ``notional = risk_capital / sl_pct``
without a guard. Passing ``sl_pct=0`` caused a silent ``ZeroDivisionError``
deep inside the portfolio simulation loop, producing a confusing traceback
for users who constructed trades with missing or zero stop-loss distances.

Original fix (B0.2): raise ``ValueError`` early at trade entry with a
clear message identifying the offending trade. Loud failure mode.

Sprint 1 update: changed to SKIP the trade and increment
``reject_reasons["invalid_sl_pct"]`` instead of raising. Rationale:
the original raise behavior killed the entire backtest on a single
corrupt trade. In production the framework can produce malformed trades
from upstream bugs (feature prep, optuna params, market data gaps);
one bad trade should not zero out all results. The trade is still
counted (reject_reasons), the warning is still logged once per
symbol per run, and the ZeroDivisionError property is preserved --
the backtest completes with a record of how many trades were skipped.

The "no silent ZeroDivisionError" property is still guaranteed (the
skip happens BEFORE the division), but the loud-fail behavior is
replaced by a soft-skip behavior that allows the rest of the backtest
to complete. See CHANGELOG for the rationale.
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


def _simulate(trade: dict):
    """Run simulate_full_portfolio and return (eq, daily_eq, trades, reasons)."""
    return simulate_full_portfolio(
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


class TestSlPctGuard:
    """B0.2 + Sprint 1: ``sl_pct <= 0`` must be skipped, not raised,
    and must NOT cause a silent ZeroDivisionError."""

    def test_sl_pct_zero_skipped_not_raised(self):
        """sl_pct=0: skip the trade (no ZeroDivisionError, no raise)."""
        trade = _trade(sl_pct=0.0)
        # Must NOT raise (no ValueError, no ZeroDivisionError).
        final_eq, daily_eq, executed, reasons = _simulate(trade)
        # Trade must be skipped (not in executed_trades).
        assert executed == [], f"Bad trade was executed: {executed}"
        # Reject reason must be recorded.
        assert reasons["invalid_sl_pct"] == 1, (
            f"Expected invalid_sl_pct=1, got {reasons}"
        )
        # Equity must equal initial cash (no trades executed).
        assert final_eq == 1000.0, f"Expected initial cash, got {final_eq}"

    def test_sl_pct_negative_skipped_not_raised(self):
        """sl_pct<0: skip the trade (no ZeroDivisionError, no raise)."""
        trade = _trade(sl_pct=-0.01)
        # Must NOT raise.
        final_eq, daily_eq, executed, reasons = _simulate(trade)
        assert executed == [], f"Bad trade was executed: {executed}"
        assert reasons["invalid_sl_pct"] == 1, (
            f"Expected invalid_sl_pct=1, got {reasons}"
        )
        assert final_eq == 1000.0, f"Expected initial cash, got {final_eq}"

    def test_sl_pct_invalid_skips_with_reject_counter(self):
        """End-to-end behavior: invalid sl_pct produces reject_reasons
        counter increment and no executed trade. Log assertions are
        intentionally skipped (loguru API details); the structural
        once-per-symbol guard is in production code at the call site.
        """
        trade = _trade(sl_pct=-0.5)
        # Two consecutive simulations -- the second should not raise,
        # and both should produce the same reject counter.
        _, _, _, r1 = _simulate(trade)
        # Need fresh trade dict per call (simulator may mutate it).
        _, _, _, r2 = _simulate(_trade(sl_pct=-0.5))
        assert r1["invalid_sl_pct"] == 1
        assert r2["invalid_sl_pct"] == 1

    def test_sl_pct_positive_does_not_trigger_guard(self):
        """A valid sl_pct must NOT trigger the invalid_sl_pct guard.

        This is the regression sanity check: positive sl_pct values
        must execute normally and reject_reasons["invalid_sl_pct"]
        must remain 0.
        """
        trade = _trade(sl_pct=0.02)
        try:
            final_eq, daily_eq, executed, reasons = _simulate(trade)
        except Exception as e:
            if "sl_pct" in str(e):
                pytest.fail(f"Valid sl_pct triggered guard: {e}")
            raise
        # No sl_pct violation -- the reject_reasons dict MUST NOT
        # contain an invalid_sl_pct key with value > 0. (Other
        # rejection reasons like margin_insufficient may be non-zero.)
        assert reasons.get("invalid_sl_pct", 0) == 0, (
            f"Valid sl_pct triggered guard: {reasons}"
        )

    def test_mixed_valid_and_invalid_trades(self):
        """Sprint 1 regression: ONE bad trade must not kill the backtest.

        Pre-Sprint-1, this test would raise ValueError. Post-Sprint-1,
        the bad trade is skipped, the good trade executes, and the
        backtest returns normally.
        """
        bad = _trade(sl_pct=0.0, entry_offset_days=0)
        good = _trade(sl_pct=0.02, entry_offset_days=2)
        # Two trades, one bad, one good.
        trades = [bad, good]
        # Build close/hl matrices covering BOTH trade windows.
        all_dates = pd.date_range("2024-01-01", "2024-01-04", freq="D")
        dcm = {"BTCUSDT": {d: 100.0 for d in all_dates}}
        dhl = {"BTCUSDT": {d: {"high": 101.0, "low": 99.0} for d in all_dates}}
        final_eq, _, executed, reasons = simulate_full_portfolio(
            trades=trades,
            initial_cash=1000.0,
            leverage=3.0,
            mm_pct=0.01,
            position_limit=4,
            cb_hard_cooldown_hours=24,
            fixed_cb_threshold=0.15,
            daily_close_matrix=dcm,
            asset_risk_weights={"BTCUSDT": 0.01},
            end_date="2024-01-04",
            daily_hl_matrix=dhl,
        )
        # Bad trade skipped, good trade may or may not execute depending
        # on position_limit/margin rules -- but invalid_sl_pct must be 1.
        assert reasons["invalid_sl_pct"] == 1, (
            f"Expected invalid_sl_pct=1, got {reasons}"
        )


class _NullContext:
    """No-op context manager used as default for pytest.warns(None)."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
