"""Direct unit tests for ``quant_lib.tools.portfolio.simulate_portfolio``."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from quant_lib.tools import portfolio as portfolio_mod
from quant_lib.tools.portfolio import simulate_portfolio


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_trade(symbol: str = "BTCUSDT", r_net: float = 0.5,
                entry: str = "2024-01-01", exit_: str = "2024-01-02"):
    """Build a minimal trade dict."""
    return {
        "entry_time": datetime.fromisoformat(entry),
        "exit_time": datetime.fromisoformat(exit_),
        "symbol": symbol,
        "r_net": r_net,
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
    }


def _make_close_matrix(symbols: list[str], n_days: int = 30,
                       start: str = "2024-01-01") -> dict:
    """Build a {symbol: {date: close}} matrix."""
    from datetime import timedelta
    matrix = {}
    for sym in symbols:
        matrix[sym] = {}
        for i in range(n_days):
            d = (datetime.fromisoformat(start) + timedelta(days=i)).date()
            matrix[sym][d] = 100.0 + i * 0.1
    return matrix


# ═══════════════════════════════════════════════════════════════════════
# daily_close_matrix validation
# ═══════════════════════════════════════════════════════════════════════


class TestSimulatePortfolioMatrixRequired:
    """``daily_close_matrix`` is a required argument (not auto-built)."""

    def test_missing_matrix_raises_value_error(self):
        """If ``daily_close_matrix`` is None, raise ValueError."""
        with pytest.raises(ValueError, match="daily_close_matrix"):
            simulate_portfolio(
                trades=[],
                daily_close_matrix=None,
            )

    def test_error_message_mentions_mtm(self):
        """The ValueError message should hint at the MTM requirement."""
        with pytest.raises(ValueError) as exc_info:
            simulate_portfolio(trades=[], daily_close_matrix=None)
        assert "MTM" in str(exc_info.value) or "mark-to-market" in str(exc_info.value).lower()


# ═══════════════════════════════════════════════════════════════════════
# Argument forwarding
# ═══════════════════════════════════════════════════════════════════════


class TestSimulatePortfolioArgumentForwarding:
    """All arguments are forwarded to ``simulate_full_portfolio``."""

    def test_required_args_forwarded(self):
        """Core required args are forwarded as positional args."""
        trades = [_make_trade()]
        matrix = _make_close_matrix(["BTCUSDT"])
        with patch("quant_lib.tools.portfolio._simulate") as mock_sim:
            mock_sim.return_value = (1000.0, {}, [], {})
            simulate_portfolio(
                trades=trades,
                daily_close_matrix=matrix,
            )
        # The wrapper calls _simulate(*positional_args)
        # _simulate(trades, initial_cash, leverage, mm_pct, position_limit,
        #            cb_hard_cooldown_hours, fixed_cb_threshold, daily_close,
        #            asset_risk_weights, end_date, liquidation_fee_pct,
        #            daily_hl_matrix)
        # Verify a subset of positional args
        call_args = mock_sim.call_args.args
        assert call_args[0] is trades
        assert call_args[7] is matrix

    def test_default_values_forwarded(self):
        """Defaults match the documented function signature."""
        matrix = _make_close_matrix(["BTCUSDT"])
        with patch("quant_lib.tools.portfolio._simulate") as mock_sim:
            mock_sim.return_value = (1000.0, {}, [], {})
            simulate_portfolio(
                trades=[],
                daily_close_matrix=matrix,
            )
        # Default values: initial_cash=1000.0, leverage=3.0, mm_pct=0.01,
        # position_limit=4, cb_hard_cooldown_hours=24, fixed_cb_threshold=0.15
        call_args = mock_sim.call_args.args
        assert call_args[1] == 1000.0
        assert call_args[2] == 3.0
        assert call_args[3] == 0.01
        assert call_args[4] == 4
        assert call_args[5] == 24
        assert call_args[6] == 0.15

    def test_custom_values_forwarded(self):
        """User-provided values are forwarded unchanged."""
        matrix = _make_close_matrix(["BTCUSDT"])
        with patch("quant_lib.tools.portfolio._simulate") as mock_sim:
            mock_sim.return_value = (1000.0, {}, [], {})
            simulate_portfolio(
                trades=[],
                initial_cash=5000.0,
                leverage=5.0,
                mm_pct=0.02,
                position_limit=10,
                cb_hard_cooldown_hours=48,
                fixed_cb_threshold=0.20,
                daily_close_matrix=matrix,
                end_date="2024-12-31",
                liquidation_fee_pct=0.01,
            )
        call_args = mock_sim.call_args.args
        assert call_args[1] == 5000.0
        assert call_args[2] == 5.0
        assert call_args[3] == 0.02
        assert call_args[4] == 10
        assert call_args[5] == 48
        assert call_args[6] == 0.20
        assert call_args[9] == "2024-12-31"
        assert call_args[10] == 0.01

    def test_optional_daily_hl_matrix_forwarded(self):
        """``daily_hl_matrix`` is forwarded (default None)."""
        matrix = _make_close_matrix(["BTCUSDT"])
        hl_matrix = {"BTCUSDT": {}}
        with patch("quant_lib.tools.portfolio._simulate") as mock_sim:
            mock_sim.return_value = (1000.0, {}, [], {})
            simulate_portfolio(
                trades=[],
                daily_close_matrix=matrix,
                daily_hl_matrix=hl_matrix,
            )
        assert mock_sim.call_args.args[11] is hl_matrix

    def test_optional_asset_risk_weights_forwarded(self):
        """``asset_risk_weights`` is forwarded (default None)."""
        matrix = _make_close_matrix(["BTCUSDT"])
        weights = {"BTCUSDT": 0.02}
        with patch("quant_lib.tools.portfolio._simulate") as mock_sim:
            mock_sim.return_value = (1000.0, {}, [], {})
            simulate_portfolio(
                trades=[],
                daily_close_matrix=matrix,
                asset_risk_weights=weights,
            )
        assert mock_sim.call_args.args[8] is weights

    def test_default_end_date_is_documented(self):
        """The default ``end_date`` is the documented string."""
        matrix = _make_close_matrix(["BTCUSDT"])
        with patch("quant_lib.tools.portfolio._simulate") as mock_sim:
            mock_sim.return_value = (1000.0, {}, [], {})
            simulate_portfolio(
                trades=[],
                daily_close_matrix=matrix,
            )
        # Position 9 is end_date
        assert mock_sim.call_args.args[9] == "2026-05-31"


# ═══════════════════════════════════════════════════════════════════════
# Return value
# ═══════════════════════════════════════════════════════════════════════


class TestSimulatePortfolioReturn:
    """The return value is forwarded from the underlying function."""

    def test_returns_underlying_tuple(self):
        matrix = _make_close_matrix(["BTCUSDT"])
        sentinel = (1500.0, {"2024-01-01": 1500.0}, [], {"cb_cooldown": 0})
        with patch("quant_lib.tools.portfolio._simulate", return_value=sentinel):
            result = simulate_portfolio(
                trades=[],
                daily_close_matrix=matrix,
            )
        assert result is sentinel

    def test_returns_four_tuple(self):
        """Return shape: (equity, daily_equity, executed, rejects)."""
        matrix = _make_close_matrix(["BTCUSDT"])
        with patch(
            "quant_lib.tools.portfolio._simulate",
            return_value=(1000.0, {}, [], {}),
        ):
            result = simulate_portfolio(
                trades=[],
                daily_close_matrix=matrix,
            )
        assert isinstance(result, tuple)
        assert len(result) == 4
        equity, daily_eq, executed, rejects = result
        assert isinstance(equity, float)
        assert isinstance(daily_eq, dict)
        assert isinstance(executed, list)
        assert isinstance(rejects, dict)
