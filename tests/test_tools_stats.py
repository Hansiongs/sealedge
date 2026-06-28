"""Direct unit tests for ``quant_lib.tools.stats`` (spa_test, prob_sharpe_ratio, fdr_correct)."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quant_lib.core._config import DEFAULTS, STATIC
from quant_lib.tools import stats as stats_mod
from quant_lib.tools.stats import fdr_correct, prob_sharpe_ratio, spa_test


# ═══════════════════════════════════════════════════════════════════════
# spa_test
# ═══════════════════════════════════════════════════════════════════════


class TestSpaTest:
    """``spa_test`` forwards all arguments to ``portfolio_spa``."""

    def _make_minimal_inputs(self):
        """Build the minimum inputs needed to call spa_test."""
        observed = [
            {
                "entry_time": datetime(2024, 1, 2),
                "exit_time": datetime(2024, 1, 3),
                "symbol": "BTCUSDT",
                "r_net": 0.5, "trade_dir": 1,
                "entry_price": 100.0, "exit_price": 101.0,
                "sl_pct": 0.02, "sl_mult": 1.5, "trail_atr": 3.0,
                "risk_weight": 0.01,
            },
        ]
        times = pd.date_range("2024-01-01", periods=48, freq="h")
        asset_data = {
            "BTCUSDT": pd.DataFrame({
                "time": times,
                "close": 100.0, "atr": 1.5, "high": 101.0, "low": 99.0,
                "funding_rate": 0.0,
                "is_weekend": np.zeros(len(times), dtype=np.int32),
                "is_funding_hour": np.zeros(len(times), dtype=np.int32),
                "macro_trend": np.ones(len(times), dtype=np.int32),
            }),
        }
        close_matrix = {"BTCUSDT": {datetime(2024, 1, 2).date(): 100.0}}
        return observed, asset_data, close_matrix

    def test_forwards_required_args(self):
        """The four required positional args are forwarded."""
        observed, asset_data, close_matrix = self._make_minimal_inputs()
        with patch("quant_lib.tools.stats._portfolio_spa") as mock_spa:
            mock_spa.return_value = (1000.0, np.array([950.0, 1050.0]), 0.5)
            spa_test(observed, asset_data, close_matrix, "2024-01-03")
        # Positional args: (observed_trades, asset_data, daily_close,
        #                    end_date, daily_hl=None, n_iters=..., ...)
        assert mock_spa.call_args.args[0] is observed
        assert mock_spa.call_args.args[1] is asset_data
        assert mock_spa.call_args.args[2] is close_matrix
        assert mock_spa.call_args.args[3] == "2024-01-03"

    def test_default_n_iters_from_static(self):
        """The default ``n_iters`` comes from ``STATIC['spa_n_iters']``."""
        observed, asset_data, close_matrix = self._make_minimal_inputs()
        with patch("quant_lib.tools.stats._portfolio_spa") as mock_spa:
            mock_spa.return_value = (1000.0, np.array([]), 1.0)
            spa_test(observed, asset_data, close_matrix, "2024-01-03")
        # n_iters is a kwarg
        assert mock_spa.call_args.kwargs["n_iters"] == STATIC["spa_n_iters"]

    def test_default_stress_mult_from_defaults(self):
        """``stress_mult`` defaults to ``DEFAULTS['stress_test_multiplier']``."""
        observed, asset_data, close_matrix = self._make_minimal_inputs()
        with patch("quant_lib.tools.stats._portfolio_spa") as mock_spa:
            mock_spa.return_value = (1000.0, np.array([]), 1.0)
            spa_test(observed, asset_data, close_matrix, "2024-01-03")
        assert (
            mock_spa.call_args.kwargs["stress_mult"]
            == DEFAULTS["stress_test_multiplier"]
        )

    def test_default_weekend_penalty_from_defaults(self):
        """``weekend_penalty`` defaults to ``DEFAULTS['weekend_liquidity_penalty']``."""
        observed, asset_data, close_matrix = self._make_minimal_inputs()
        with patch("quant_lib.tools.stats._portfolio_spa") as mock_spa:
            mock_spa.return_value = (1000.0, np.array([]), 1.0)
            spa_test(observed, asset_data, close_matrix, "2024-01-03")
        assert (
            mock_spa.call_args.kwargs["weekend_penalty"]
            == DEFAULTS["weekend_liquidity_penalty"]
        )

    def test_custom_n_iters_forwarded(self):
        observed, asset_data, close_matrix = self._make_minimal_inputs()
        with patch("quant_lib.tools.stats._portfolio_spa") as mock_spa:
            mock_spa.return_value = (1000.0, np.array([]), 1.0)
            spa_test(
                observed, asset_data, close_matrix, "2024-01-03", n_iters=500,
            )
        assert mock_spa.call_args.kwargs["n_iters"] == 500

    def test_optional_daily_hl_matrix_forwarded(self):
        observed, asset_data, close_matrix = self._make_minimal_inputs()
        hl_matrix = {"BTCUSDT": {}}
        with patch("quant_lib.tools.stats._portfolio_spa") as mock_spa:
            mock_spa.return_value = (1000.0, np.array([]), 1.0)
            spa_test(
                observed, asset_data, close_matrix, "2024-01-03",
                daily_hl_matrix=hl_matrix,
            )
        assert mock_spa.call_args.kwargs["daily_hl_matrix"] is hl_matrix

    def test_returns_underlying_tuple(self):
        observed, asset_data, close_matrix = self._make_minimal_inputs()
        sentinel = (1500.0, np.array([1000.0, 1500.0]), 0.05)
        with patch(
            "quant_lib.tools.stats._portfolio_spa",
            return_value=sentinel,
        ):
            result = spa_test(
                observed, asset_data, close_matrix, "2024-01-03",
            )
        assert result is sentinel

    def test_verbose_default_false(self):
        """``verbose`` defaults to False."""
        observed, asset_data, close_matrix = self._make_minimal_inputs()
        with patch("quant_lib.tools.stats._portfolio_spa") as mock_spa:
            mock_spa.return_value = (1000.0, np.array([]), 1.0)
            spa_test(observed, asset_data, close_matrix, "2024-01-03")
        assert mock_spa.call_args.kwargs["verbose"] is False


# ═══════════════════════════════════════════════════════════════════════
# prob_sharpe_ratio
# ═══════════════════════════════════════════════════════════════════════


class TestProbSharpeRatio:
    """``prob_sharpe_ratio`` forwards to ``_psr``."""

    def test_forwards_returns(self):
        """The returns array is forwarded."""
        returns = np.array([0.01, 0.02, 0.03])
        with patch("quant_lib.tools.stats._psr") as mock_psr:
            mock_psr.return_value = (1.0, 0.9)
            prob_sharpe_ratio(returns)
        assert np.array_equal(mock_psr.call_args.args[0], returns)

    def test_default_benchmark_is_zero(self):
        returns = np.array([0.01, 0.02, 0.03])
        with patch("quant_lib.tools.stats._psr") as mock_psr:
            mock_psr.return_value = (1.0, 0.9)
            prob_sharpe_ratio(returns)
        assert mock_psr.call_args.args[1] == 0.0

    def test_default_annualize_is_true(self):
        returns = np.array([0.01, 0.02, 0.03])
        with patch("quant_lib.tools.stats._psr") as mock_psr:
            mock_psr.return_value = (1.0, 0.9)
            prob_sharpe_ratio(returns)
        assert mock_psr.call_args.args[2] is True

    def test_custom_benchmark(self):
        returns = np.array([0.01, 0.02, 0.03])
        with patch("quant_lib.tools.stats._psr") as mock_psr:
            mock_psr.return_value = (1.0, 0.5)
            prob_sharpe_ratio(returns, benchmark=0.5)
        assert mock_psr.call_args.args[1] == 0.5

    def test_custom_annualize(self):
        returns = np.array([0.01, 0.02, 0.03])
        with patch("quant_lib.tools.stats._psr") as mock_psr:
            mock_psr.return_value = (0.5, 0.9)
            prob_sharpe_ratio(returns, annualize=False)
        assert mock_psr.call_args.args[2] is False

    def test_returns_two_tuple(self):
        returns = np.array([0.01, 0.02, 0.03])
        with patch(
            "quant_lib.tools.stats._psr",
            return_value=(1.5, 0.85),
        ):
            sr, psr = prob_sharpe_ratio(returns)
        assert isinstance(sr, float)
        assert isinstance(psr, float)
        assert sr == 1.5
        assert psr == 0.85


# ═══════════════════════════════════════════════════════════════════════
# fdr_correct
# ═══════════════════════════════════════════════════════════════════════


class TestFdrCorrect:
    """``fdr_correct`` forwards to ``_fdr``."""

    def test_forwards_p_values(self):
        p_vals = np.array([0.01, 0.02, 0.03])
        with patch("quant_lib.tools.stats._fdr") as mock_fdr:
            mock_fdr.return_value = (np.array([True, True, True]), np.array([0.02, 0.04, 0.06]))
            fdr_correct(p_vals)
        assert np.array_equal(mock_fdr.call_args.args[0], p_vals)

    def test_default_alpha_is_005(self):
        p_vals = np.array([0.01, 0.02, 0.03])
        with patch("quant_lib.tools.stats._fdr") as mock_fdr:
            mock_fdr.return_value = (np.array([True, True, True]), np.array([0.02, 0.04, 0.06]))
            fdr_correct(p_vals)
        assert mock_fdr.call_args.args[1] == 0.05

    def test_custom_alpha(self):
        p_vals = np.array([0.01, 0.02, 0.03])
        with patch("quant_lib.tools.stats._fdr") as mock_fdr:
            mock_fdr.return_value = (np.array([False, False, False]), np.array([0.1, 0.2, 0.3]))
            fdr_correct(p_vals, alpha=0.10)
        assert mock_fdr.call_args.args[1] == 0.10

    def test_returns_two_tuple(self):
        p_vals = np.array([0.01, 0.02, 0.03])
        with patch(
            "quant_lib.tools.stats._fdr",
            return_value=(np.array([True, False, True]), np.array([0.02, 0.04, 0.06])),
        ):
            rejected, corrected = fdr_correct(p_vals)
        assert isinstance(rejected, np.ndarray)
        assert isinstance(corrected, np.ndarray)
        assert rejected.dtype == bool
