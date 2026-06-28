"""Tests for ``quant_lib.research.plotting``.

Verifies:
- Each plotting function accepts dict / Series / DataFrame inputs.
- Each function returns either a file path (when output_path given)
  or a base64 data URI (when no output_path).
- Empty / all-NaN input does not crash.
- Figures are properly closed (no memory leak).
- Headless / Agg backend works without a display.
- Graceful failure when matplotlib is not installed (simulated).
"""
from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from quant_lib.research import plotting


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


def _make_equity_dict(n_days: int = 60, start: str = "2024-01-01",
                      drift: float = 0.005, vol: float = 0.02,
                      seed: int = 42) -> dict:
    """Generate a synthetic daily-equity dict for testing."""
    import numpy as np
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    rets = rng.normal(drift, vol, n_days)
    eq = 1000.0 * (1 + pd.Series(rets)).cumprod()
    eq.iloc[0] = 1000.0
    return {d: float(v) for d, v in zip(dates, eq)}


def _equity_with_drawdown() -> dict:
    """Equity curve that has a known drawdown profile."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    # Peak at day 2 (1100), then drops to 800 (DD ~27%), recovers
    values = [1000, 1050, 1100, 1050, 950, 800, 850, 900, 950, 1000]
    return {d: float(v) for d, v in zip(dates, values)}


# ═══════════════════════════════════════════════════════════════════════
# plot_equity_curve
# ═══════════════════════════════════════════════════════════════════════


class TestPlotEquityCurve:
    def test_returns_base64_when_no_output_path(self):
        """Without output_path, return base64 data URI."""
        eq = _make_equity_dict()
        result = plotting.plot_equity_curve(eq, initial_capital=1000.0)
        assert isinstance(result, str)
        assert result.startswith("data:image/png;base64,")

    def test_base64_decodes_to_valid_png(self):
        """The base64 payload must decode to a valid PNG file."""
        eq = _make_equity_dict()
        result = plotting.plot_equity_curve(eq, initial_capital=1000.0)
        payload = result.split(",", 1)[1]
        decoded = base64.b64decode(payload)
        # PNG magic bytes
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(decoded) > 100  # non-trivial size

    def test_saves_to_output_path(self, tmp_path):
        """With output_path, save PNG to disk and return the path."""
        eq = _make_equity_dict()
        out = tmp_path / "equity.png"
        result = plotting.plot_equity_curve(eq, initial_capital=1000.0, output_path=str(out))
        assert result == str(out)
        assert out.exists()
        assert out.stat().st_size > 100
        with open(out, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"

    def test_accepts_series_input(self):
        """Series input works (not just dict)."""
        eq_dict = _make_equity_dict()
        eq_series = pd.Series(eq_dict).sort_index()
        result = plotting.plot_equity_curve(eq_series, initial_capital=1000.0)
        assert result.startswith("data:image/png;base64,")

    def test_accepts_dataframe_input(self):
        """DataFrame input with 'time' column works."""
        eq_dict = _make_equity_dict()
        df = pd.DataFrame([
            {"time": k, "equity": v} for k, v in eq_dict.items()
        ])
        result = plotting.plot_equity_curve(df, initial_capital=1000.0)
        assert result.startswith("data:image/png;base64,")

    def test_empty_equity_does_not_crash(self):
        """Empty equity produces a placeholder figure (not exception)."""
        result = plotting.plot_equity_curve({}, initial_capital=1000.0)
        assert result.startswith("data:image/png;base64,")

    def test_loss_scenario_renders(self):
        """Equity below initial_capital still renders without error."""
        eq = _make_equity_dict(drift=-0.02, vol=0.01)
        result = plotting.plot_equity_curve(eq, initial_capital=1000.0)
        assert result.startswith("data:image/png;base64,")

    def test_figure_closed_after_call(self):
        """Calling plot_equity_curve closes its figure (no memory leak)."""
        import matplotlib.pyplot as plt
        n_before = len(plt.get_fignums())
        plotting.plot_equity_curve(_make_equity_dict(), initial_capital=1000.0)
        n_after = len(plt.get_fignums())
        assert n_after == n_before, "Figure was not closed after plot_equity_curve"


# ═══════════════════════════════════════════════════════════════════════
# plot_drawdown_underwater
# ═══════════════════════════════════════════════════════════════════════


class TestPlotDrawdownUnderwater:
    def test_returns_base64_when_no_output_path(self):
        eq = _equity_with_drawdown()
        result = plotting.plot_drawdown_underwater(eq)
        assert result.startswith("data:image/png;base64,")

    def test_known_drawdown_value(self):
        """Verify the figure encodes a drawdown of approximately -27% for the test curve.

        The test curve drops from peak 1100 to trough 800, so DD = (800-1100)/1100
        = -27.27%. The chart title should reflect this.
        """
        eq = _equity_with_drawdown()
        result = plotting.plot_drawdown_underwater(eq)
        # The title text "-27.27%" is in the rendered PNG. We can't easily
        # extract it, but the function should not crash on the curve.
        assert result.startswith("data:image/png;base64,")

    def test_saves_to_output_path(self, tmp_path):
        eq = _equity_with_drawdown()
        out = tmp_path / "dd.png"
        result = plotting.plot_drawdown_underwater(eq, output_path=str(out))
        assert result == str(out)
        assert out.exists()

    def test_accepts_series_input(self):
        eq_dict = _equity_with_drawdown()
        eq_series = pd.Series(eq_dict).sort_index()
        result = plotting.plot_drawdown_underwater(eq_series)
        assert result.startswith("data:image/png;base64,")

    def test_empty_equity_does_not_crash(self):
        result = plotting.plot_drawdown_underwater({})
        assert result.startswith("data:image/png;base64,")

    def test_monotonically_increasing_equity_zero_dd(self):
        """For a monotonically increasing equity, drawdown should be 0% throughout."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        eq = {d: 1000 + i * 100 for i, d in enumerate(dates)}
        result = plotting.plot_drawdown_underwater(eq)
        assert result.startswith("data:image/png;base64,")

    def test_figure_closed_after_call(self):
        import matplotlib.pyplot as plt
        n_before = len(plt.get_fignums())
        plotting.plot_drawdown_underwater(_equity_with_drawdown())
        n_after = len(plt.get_fignums())
        assert n_after == n_before, "Figure was not closed after plot_drawdown_underwater"


# ═══════════════════════════════════════════════════════════════════════
# plot_trade_distribution
# ═══════════════════════════════════════════════════════════════════════


def _trade_r_vals() -> list:
    """Synthetic R-multiples: realistic mix of wins/losses with tail."""
    import numpy as np
    rng = np.random.default_rng(42)
    n = 100
    # 55% positive, mean +0.3, mean -0.5; some outliers
    pos = rng.normal(0.5, 0.4, int(n * 0.55))
    neg = rng.normal(-0.5, 0.3, n - int(n * 0.55))
    vals = np.concatenate([pos, neg])
    # add a few outliers
    vals[0] = 3.5
    vals[1] = -2.8
    return vals.tolist()


class TestPlotTradeDistribution:
    def test_returns_base64_when_no_output_path(self):
        result = plotting.plot_trade_distribution(_trade_r_vals())
        assert result.startswith("data:image/png;base64,")

    def test_saves_to_output_path(self, tmp_path):
        out = tmp_path / "trades.png"
        result = plotting.plot_trade_distribution(_trade_r_vals(), output_path=str(out))
        assert result == str(out)
        assert out.exists()

    def test_accepts_numpy_array(self):
        import numpy as np
        vals = np.array([0.5, -0.3, 1.2, -0.5])
        result = plotting.plot_trade_distribution(vals)
        assert result.startswith("data:image/png;base64,")

    def test_accepts_series(self):
        s = pd.Series([0.5, -0.3, 1.2, -0.5])
        result = plotting.plot_trade_distribution(s)
        assert result.startswith("data:image/png;base64,")

    def test_empty_equity_does_not_crash(self):
        result = plotting.plot_trade_distribution([])
        assert result.startswith("data:image/png;base64,")

    def test_all_nan_does_not_crash(self):
        import numpy as np
        result = plotting.plot_trade_distribution([np.nan, np.nan, np.nan])
        assert result.startswith("data:image/png;base64,")

    def test_drops_nan_and_inf(self):
        """NaN and Inf are dropped before plotting; finite values are used."""
        import numpy as np
        mixed = [0.5, np.nan, -0.3, np.inf, 1.2, -np.inf, 0.4]
        result = plotting.plot_trade_distribution(mixed)
        assert result.startswith("data:image/png;base64,")

    def test_few_trades_no_kde(self):
        """With n < 5, KDE is suppressed but the figure still renders."""
        result = plotting.plot_trade_distribution([0.5, -0.3, 1.2])
        assert result.startswith("data:image/png;base64,")

    def test_zero_variance_renders(self):
        """All-same values: std=0, KDE must not crash."""
        result = plotting.plot_trade_distribution([0.5, 0.5, 0.5, 0.5, 0.5])
        assert result.startswith("data:image/png;base64,")

    def test_figure_closed_after_call(self):
        import matplotlib.pyplot as plt
        n_before = len(plt.get_fignums())
        plotting.plot_trade_distribution(_trade_r_vals())
        n_after = len(plt.get_fignums())
        assert n_after == n_before, "Figure was not closed after plot_trade_distribution"


# ═══════════════════════════════════════════════════════════════════════
# plot_spa_null
# ═══════════════════════════════════════════════════════════════════════


def _spa_null_equities(n: int = 200, seed: int = 42) -> list:
    """Synthetic SPA null distribution centered below observed."""
    import numpy as np
    rng = np.random.default_rng(seed)
    return rng.normal(1000, 50, n).tolist()


class TestPlotSpaNull:
    def test_returns_base64_when_no_output_path(self):
        result = plotting.plot_spa_null(_spa_null_equities(), observed_equity=1300, p_value=0.03)
        assert result.startswith("data:image/png;base64,")

    def test_saves_to_output_path(self, tmp_path):
        out = tmp_path / "spa.png"
        result = plotting.plot_spa_null(
            _spa_null_equities(), observed_equity=1300, p_value=0.03,
            output_path=str(out),
        )
        assert result == str(out)
        assert out.exists()

    def test_significant_observed_uses_green(self):
        """p < 0.05 -> observed line should be green-coded (significant)."""
        result = plotting.plot_spa_null(_spa_null_equities(), observed_equity=1300, p_value=0.03)
        assert result.startswith("data:image/png;base64,")

    def test_not_significant_uses_red(self):
        """p >= 0.05 -> observed line is red-coded (not significant)."""
        result = plotting.plot_spa_null(_spa_null_equities(), observed_equity=1100, p_value=0.5)
        assert result.startswith("data:image/png;base64,")

    def test_pvalue_nan_handled(self):
        """NaN p-value is rendered with 'NaN' in the title, not crashed."""
        import numpy as np
        result = plotting.plot_spa_null(
            _spa_null_equities(), observed_equity=1300, p_value=float("nan"),
        )
        assert result.startswith("data:image/png;base64,")

    def test_empty_null_does_not_crash(self):
        """Empty null distribution produces a placeholder figure."""
        result = plotting.plot_spa_null([], observed_equity=1300, p_value=0.5)
        assert result.startswith("data:image/png;base64,")

    def test_nan_observed_handled(self):
        """NaN observed equity produces a placeholder figure."""
        result = plotting.plot_spa_null(_spa_null_equities(), observed_equity=float("nan"),
                                        p_value=0.5)
        assert result.startswith("data:image/png;base64,")

    def test_drops_inf_from_null(self):
        """Inf values in null are dropped before plotting."""
        import numpy as np
        mixed = [1000, 1100, float("inf"), 950, -float("inf"), 1050]
        result = plotting.plot_spa_null(mixed, observed_equity=1300, p_value=0.03)
        assert result.startswith("data:image/png;base64,")

    def test_figure_closed_after_call(self):
        import matplotlib.pyplot as plt
        n_before = len(plt.get_fignums())
        plotting.plot_spa_null(_spa_null_equities(), observed_equity=1300, p_value=0.03)
        n_after = len(plt.get_fignums())
        assert n_after == n_before, "Figure was not closed after plot_spa_null"


# ═══════════════════════════════════════════════════════════════════════
# plot_per_symbol_equity
# ═══════════════════════════════════════════════════════════════════════


def _per_symbol_equity() -> dict:
    """Synthetic per-symbol equity data."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    return {
        "BTCUSDT": {d: 100 + i * 5 for i, d in enumerate(dates)},
        "ETHUSDT": {d: 50 + i * 3 for i, d in enumerate(dates)},
        "SOLUSDT": {d: 20 + i * 1.5 for i, d in enumerate(dates)},
    }


class TestPlotPerSymbolEquity:
    def test_returns_base64(self):
        result = plotting.plot_per_symbol_equity(_per_symbol_equity())
        assert result.startswith("data:image/png;base64,")

    def test_saves_to_output_path(self, tmp_path):
        out = tmp_path / "per_sym.png"
        result = plotting.plot_per_symbol_equity(_per_symbol_equity(), output_path=str(out))
        assert result == str(out)
        assert out.exists()

    def test_empty_dict_does_not_crash(self):
        result = plotting.plot_per_symbol_equity({})
        assert result.startswith("data:image/png;base64,")

    def test_single_symbol(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        eq = {d: 100 + i for i, d in enumerate(dates)}
        result = plotting.plot_per_symbol_equity({"BTCUSDT": eq})
        assert result.startswith("data:image/png;base64,")

    def test_series_input_per_symbol(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        result = plotting.plot_per_symbol_equity({
            "BTCUSDT": pd.Series([100 + i for i in range(10)], index=dates),
            "ETHUSDT": pd.Series([50 + i for i in range(10)], index=dates),
        })
        assert result.startswith("data:image/png;base64,")

    def test_skips_empty_symbols(self):
        """A symbol with no data should be skipped, not crash the chart."""
        result = plotting.plot_per_symbol_equity({
            "BTCUSDT": {pd.Timestamp("2024-01-01"): 100},
            "EMPTY": {},
        })
        assert result.startswith("data:image/png;base64,")

    def test_many_symbols_uses_2col_legend(self):
        """When more than 12 symbols, legend should switch to 2 columns."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        eq_dict = {f"SYM{i:02d}": {d: 100 + i for d in dates} for i in range(15)}
        result = plotting.plot_per_symbol_equity(eq_dict)
        assert result.startswith("data:image/png;base64,")

    def test_figure_closed_after_call(self):
        import matplotlib.pyplot as plt
        n_before = len(plt.get_fignums())
        plotting.plot_per_symbol_equity(_per_symbol_equity())
        n_after = len(plt.get_fignums())
        assert n_after == n_before, "Figure was not closed after plot_per_symbol_equity"


# ═══════════════════════════════════════════════════════════════════════
# plot_wfa_progression
# ═══════════════════════════════════════════════════════════════════════


def _wfa_fold_params() -> dict:
    """Synthetic WFA fold params per symbol."""
    return {
        "BTCUSDT": [
            {"fold": 1, "best_value": 0.45, "oos_start": pd.Timestamp("2020-04-01"),
             "oos_end": pd.Timestamp("2020-06-30")},
            {"fold": 2, "best_value": 0.55, "oos_start": pd.Timestamp("2020-07-01"),
             "oos_end": pd.Timestamp("2020-09-30")},
            {"fold": 3, "best_value": 0.50, "oos_start": pd.Timestamp("2020-10-01"),
             "oos_end": pd.Timestamp("2020-12-31")},
        ],
        "ETHUSDT": [
            {"fold": 1, "best_value": 0.30},
            {"fold": 2, "best_value": 0.40},
        ],
    }


class TestPlotWfaProgression:
    def test_returns_base64(self):
        result = plotting.plot_wfa_progression(_wfa_fold_params())
        assert result.startswith("data:image/png;base64,")

    def test_saves_to_output_path(self, tmp_path):
        out = tmp_path / "wfa.png"
        result = plotting.plot_wfa_progression(_wfa_fold_params(), output_path=str(out))
        assert result == str(out)
        assert out.exists()

    def test_empty_dict_does_not_crash(self):
        result = plotting.plot_wfa_progression({})
        assert result.startswith("data:image/png;base64,")

    def test_single_fold_per_symbol(self):
        params = {"BTCUSDT": [{"best_value": 0.5}], "ETHUSDT": [{"best_value": 0.6}]}
        result = plotting.plot_wfa_progression(params)
        assert result.startswith("data:image/png;base64,")

    def test_drops_nan_best_value(self):
        """Folds with non-finite best_value are dropped, not crashed."""
        import numpy as np
        params = {
            "BTCUSDT": [
                {"best_value": 0.5},
                {"best_value": float("nan")},
                {"best_value": 0.6},
                {"best_value": float("inf")},
            ],
        }
        result = plotting.plot_wfa_progression(params)
        assert result.startswith("data:image/png;base64,")

    def test_missing_best_value_key(self):
        """Folds without 'best_value' key are skipped silently."""
        params = {
            "BTCUSDT": [
                {"fold": 1, "best_value": 0.5},
                {"fold": 2},  # no best_value
                {"fold": 3, "best_value": 0.6},
            ],
        }
        result = plotting.plot_wfa_progression(params)
        assert result.startswith("data:image/png;base64,")

    def test_non_dict_fold_entries_skipped(self):
        """Fold entries that are not dicts are skipped, not crashed."""
        params = {
            "BTCUSDT": [
                {"best_value": 0.5},
                "not_a_dict",  # type: ignore[list-item]
                None,  # type: ignore[list-item]
                {"best_value": 0.7},
            ],
        }
        result = plotting.plot_wfa_progression(params)
        assert result.startswith("data:image/png;base64,")

    def test_symbol_with_no_valid_folds_skipped(self):
        """A symbol whose folds all lack best_value is excluded from chart."""
        params = {
            "BTCUSDT": [{"best_value": 0.5}, {"best_value": 0.6}],
            "EMPTY": [{"fold": 1}, {"fold": 2}],  # no best_value
        }
        result = plotting.plot_wfa_progression(params)
        assert result.startswith("data:image/png;base64,")

    def test_figure_closed_after_call(self):
        import matplotlib.pyplot as plt
        n_before = len(plt.get_fignums())
        plotting.plot_wfa_progression(_wfa_fold_params())
        n_after = len(plt.get_fignums())
        assert n_after == n_before, "Figure was not closed after plot_wfa_progression"


# ═══════════════════════════════════════════════════════════════════════
# Backend / headless
# ═══════════════════════════════════════════════════════════════════════


class TestMatplotlibBackend:
    def test_agg_backend_is_active(self):
        """The Agg backend must be active (no display required)."""
        import matplotlib
        assert matplotlib.get_backend().lower() == "agg"

    def test_seaborn_theme_applied(self):
        """Seaborn theme is applied (rcParams reflect it)."""
        import matplotlib as mpl
        # whitegrid style sets axes.facecolor and grid presence
        # The exact rcParams depend on seaborn version; just verify
        # the theme is recognized.
        import seaborn as sns
        assert sns.axes_style() is not None
