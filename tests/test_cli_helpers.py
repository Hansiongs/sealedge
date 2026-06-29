"""Unit tests for CLI helper functions.

Targets uncovered lines in:
- explore.py: _try_save_html_report, _make_chart_provider,
              _per_symbol_equity_from_trades, _looks_like_absolute
- commit_cmd.py: _try_save_html_report, _make_chart_provider,
                 _build_equity_series_from_result, _looks_like_absolute
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quant_lib.cli.explore import (
    _looks_like_absolute as explore_absolute,
    _per_symbol_equity_from_trades,
)
from quant_lib.cli.commit_cmd import (
    _looks_like_absolute as commit_absolute,
    _build_equity_series_from_result,
)
from quant_lib.research.commit import CommitResult


# ═══════════════════════════════════════════════════════════════════════
# _looks_like_absolute (explore.py + commit_cmd.py)
# ═══════════════════════════════════════════════════════════════════════

class TestLooksLikeAbsolute:
    """Both explore and commit_cmd have identical _looks_like_absolute.

    On Windows, ``os.path.isabs`` behaves differently:
    - ``/tmp/foo`` → False (not a Windows absolute path)
    - ``C:\\foo`` → True
    - ``foo.html`` → False
    """

    def test_relative_path(self):
        assert explore_absolute("report.html") is False
        assert commit_absolute("report.html") is False

    def test_relative_subdir_path(self):
        assert explore_absolute("subdir/report.html") is False
        assert commit_absolute("subdir/report.html") is False

    def test_absolute_windows_path(self):
        assert explore_absolute("C:\\Users\\report.html") is True
        assert commit_absolute("C:\\Users\\report.html") is True

    def test_empty_string(self):
        assert explore_absolute("") is False
        assert commit_absolute("") is False

    @pytest.mark.skipif(os.name != "posix", reason="Unix-only absolute path test")
    def test_absolute_unix_path_posix(self):
        assert explore_absolute("/tmp/report.html") is True
        assert commit_absolute("/tmp/report.html") is True


# ═══════════════════════════════════════════════════════════════════════
# _per_symbol_equity_from_trades (explore.py)
# ═══════════════════════════════════════════════════════════════════════

class TestPerSymbolEquityFromTrades:
    """_per_symbol_equity_from_trades(cand) -> {sym: {date: cum_r}}."""

    @staticmethod
    def _mock_candidate(executed_trades: list | None = None):
        cand = MagicMock(spec=["executed_trades"])
        cand.executed_trades = executed_trades or []
        return cand

    def test_no_trades_returns_empty(self):
        cand = self._mock_candidate([])
        result = _per_symbol_equity_from_trades(cand)
        assert result == {}

    def test_none_trades_returns_empty(self):
        cand = self._mock_candidate(None)
        result = _per_symbol_equity_from_trades(cand)
        assert result == {}

    def test_single_symbol_single_trade(self):
        cand = self._mock_candidate([
            {"symbol": "BTCUSDT", "exit_time": pd.Timestamp("2025-01-15"), "r_net": 1.5},
        ])
        result = _per_symbol_equity_from_trades(cand)
        assert "BTCUSDT" in result
        assert len(result["BTCUSDT"]) == 1

    def test_two_symbols_cumulative(self):
        cand = self._mock_candidate([
            {"symbol": "BTCUSDT", "exit_time": pd.Timestamp("2025-01-15"), "r_net": 1.0},
            {"symbol": "ETHUSDT", "exit_time": pd.Timestamp("2025-01-16"), "r_net": 2.0},
            {"symbol": "BTCUSDT", "exit_time": pd.Timestamp("2025-01-17"), "r_net": 0.5},
        ])
        result = _per_symbol_equity_from_trades(cand)
        assert "BTCUSDT" in result
        assert "ETHUSDT" in result
        btc = pd.Series(result["BTCUSDT"])
        assert btc.iloc[-1] == pytest.approx(1.5)

    def test_skip_trade_with_missing_symbol(self):
        cand = self._mock_candidate([
            {"symbol": None, "exit_time": pd.Timestamp("2025-01-15"), "r_net": 1.0},
        ])
        result = _per_symbol_equity_from_trades(cand)
        assert result == {}

    def test_skip_trade_with_missing_exit_time(self):
        cand = self._mock_candidate([
            {"symbol": "BTCUSDT", "exit_time": None, "r_net": 1.0},
        ])
        result = _per_symbol_equity_from_trades(cand)
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════
# _build_equity_series_from_result (commit_cmd.py)
# ═══════════════════════════════════════════════════════════════════════

class TestBuildEquitySeriesFromResult:

    def test_builds_two_point_series(self):
        result = CommitResult(
            candidate_name="test", commit_idx=1,
            holdout_period=("2025-01-01", "2025-06-30"),
            timestamp="2025-01-01T00:00:00",
            initial_capital=1000.0, final_equity=1100.0,
            equity_pct=10.0, cagr_pct=21.0, max_dd_pct=5.0,
            n_raw_trades=10, n_executed_trades=8, n_rejected=2,
            reject_breakdown={},
            n_trades=8, win_rate=62.5, avg_r=0.5, median_r=0.3,
            std_r=1.0, best_r=2.5, worst_r=-1.5,
            profit_factor=1.8, avg_bars_held=12.0,
            sharpe_r=0.5, psr=0.85, psr_ess=0.85,
            skew=0.2, kurtosis=3.5, ess=8.0,
            bonferroni_alpha=0.075, fdr_alpha=0.15,
            by_symbol_stats={},
            seal_hash_before="", seal_hash_after="a" * 64,
            seal_broken=True, success_criteria_text="",
        )
        series = _build_equity_series_from_result(result)
        assert len(series) == 2
        keys = sorted(series.keys())
        assert series[keys[0]] == 1000.0
        assert series[keys[1]] == 1100.0


# ═══════════════════════════════════════════════════════════════════════
# _make_chart_provider (explore.py + commit_cmd.py)
# ═══════════════════════════════════════════════════════════════════════

class TestExploreChartProvider:

    def test_no_plots_returns_none_provider(self):
        from quant_lib.cli.explore import _make_chart_provider
        cand = MagicMock()
        session = MagicMock()
        provider = _make_chart_provider(cand, session, no_plots=True)
        assert provider("equity_curve") is None
        assert provider("anything") is None


class TestCommitChartProvider:

    def test_no_plots_returns_none_provider(self):
        from quant_lib.cli.commit_cmd import _make_chart_provider
        cand = MagicMock()
        result = MagicMock()
        session = MagicMock()
        provider = _make_chart_provider(cand, result, session, no_plots=True)
        assert provider("equity_curve") is None
        assert provider("trade_distribution") is None


# ═══════════════════════════════════════════════════════════════════════
# _try_save_html_report (explore.py + commit_cmd.py)
# ═══════════════════════════════════════════════════════════════════════
# These functions call build_explore_report / build_commit_report first,
# which need fully populated mocks. We patch those at module level.

class TestExploreTrySaveHTMLReport:

    def test_save_with_default_report(self):
        from quant_lib.cli.explore import _try_save_html_report
        out = MagicMock()
        out.path = Path(tempfile.mkdtemp())
        out.save_html_report = MagicMock()

        cand = MagicMock()
        session = MagicMock()

        # build_explore_report returns a known list
        with patch("quant_lib.cli._report.build_explore_report", return_value=[]):
            _try_save_html_report(
                out=out, title="Explore Test",
                cand=cand, session=session,
                report="", no_plots=True,
            )
        out.save_html_report.assert_called_once()

    def test_save_with_relative_report(self):
        from quant_lib.cli.explore import _try_save_html_report
        out = MagicMock()
        out.path = Path(tempfile.mkdtemp())
        out.save_html_report = MagicMock()

        cand = MagicMock()
        session = MagicMock()

        with patch("quant_lib.cli._report.build_explore_report", return_value=[]):
            _try_save_html_report(
                out=out, title="Test", cand=cand, session=session,
                report="custom.html", no_plots=True,
            )
        out.save_html_report.assert_called_once()

    def test_save_html_report_raises(self):
        from quant_lib.cli.explore import _try_save_html_report
        out = MagicMock()
        out.path = Path(tempfile.mkdtemp())
        out.save_html_report.side_effect = RuntimeError("write failed")

        cand = MagicMock()
        session = MagicMock()

        with patch("quant_lib.cli._report.build_explore_report", return_value=[]):
            # Should not raise — exception is caught
            _try_save_html_report(
                out=out, title="Test", cand=cand, session=session,
                report="fail_report.html", no_plots=True,
            )


class TestCommitTrySaveHTMLReport:

    def test_save_with_default_report(self):
        from quant_lib.cli.commit_cmd import _try_save_html_report
        out = MagicMock()
        out.path = Path(tempfile.mkdtemp())
        out.save_html_report = MagicMock()

        cand = MagicMock()
        result = MagicMock()
        session = MagicMock()

        with patch("quant_lib.cli._report.build_commit_report", return_value=[]):
            _try_save_html_report(
                out=out, title="Commit Test",
                cand=cand, result=result, session=session,
                report="", no_plots=True,
            )
        out.save_html_report.assert_called_once()

    def test_save_html_report_raises(self):
        from quant_lib.cli.commit_cmd import _try_save_html_report
        out = MagicMock()
        out.path = Path(tempfile.mkdtemp())
        out.save_html_report.side_effect = RuntimeError("write failed")

        cand = MagicMock()
        result = MagicMock()
        session = MagicMock()

        with patch("quant_lib.cli._report.build_commit_report", return_value=[]):
            _try_save_html_report(
                out=out, title="Test", cand=cand, result=result,
                session=session, report="fail.html", no_plots=True,
            )

    def test_save_with_absolute_report_commit(self):
        """Absolute report path: mkdir parent, copy from run dir."""
        from quant_lib.cli.commit_cmd import _try_save_html_report
        tmpdir = Path(tempfile.mkdtemp())
        out = MagicMock()
        out.path = tmpdir / "run_dir"
        out.path.mkdir(parents=True, exist_ok=True)
        (out.path / "report.html").write_text("<html></html>")
        out.save_html_report = MagicMock()

        cand = MagicMock()
        result = MagicMock()
        session = MagicMock()
        abs_target = str(tmpdir / "output" / "final_report.html")

        with patch("quant_lib.cli._report.build_commit_report", return_value=[]):
            _try_save_html_report(
                out=out, title="Test", cand=cand, result=result,
                session=session, report=abs_target, no_plots=True,
            )
        out.save_html_report.assert_called_once()


# ── explore try_save_html_report with absolute path ──

class TestExploreAbsoluteReportPath:
    """Cover explore.py _try_save_html_report absolute path branch."""

    def test_absolute_report_path(self):
        from quant_lib.cli.explore import _try_save_html_report
        tmpdir = Path(tempfile.mkdtemp())
        out = MagicMock()
        out.path = tmpdir / "run_dir"
        out.path.mkdir(parents=True, exist_ok=True)
        (out.path / "report.html").write_text("<html></html>")
        out.save_html_report = MagicMock()

        cand = MagicMock()
        session = MagicMock()
        abs_target = str(tmpdir / "final.html")

        with patch("quant_lib.cli._report.build_explore_report", return_value=[]):
            _try_save_html_report(
                out=out, title="Test", cand=cand, session=session,
                report=abs_target, no_plots=True,
            )
        out.save_html_report.assert_called_once()


# ── commit_cmd chart provider with plotting available ──

class TestCommitChartProviderWithPlotting:
    """Cover _make_chart_provider in commit_cmd.py (lines 224-253)."""

    def test_commit_chart_provider_with_plotting(self):
        from quant_lib.cli.commit_cmd import _make_chart_provider

        cand = MagicMock()
        result = MagicMock()
        result.initial_capital = 1000.0
        result.final_equity = 1100.0
        result.holdout_period = ("2025-01-01", "2025-06-30")
        session = MagicMock()
        session.initial_capital = 1000.0

        provider = _make_chart_provider(cand, result, session, no_plots=False)
        # Code path exercised — no crash expected
        _ = provider("equity_curve")
        _ = provider("drawdown_underwater")
        _ = provider("trade_distribution")
        unknown = provider("nonexistent")
        assert unknown is None

    def test_commit_chart_provider_catches_exception(self):
        from quant_lib.cli.commit_cmd import _make_chart_provider

        cand = MagicMock()
        result = MagicMock()
        result.initial_capital = 1000.0
        result.final_equity = 1100.0
        result.holdout_period = ("2025-01-01", "2025-06-30")
        session = MagicMock()
        session.initial_capital = 1000.0

        provider = _make_chart_provider(cand, result, session, no_plots=False)
        val = provider("equity_curve")
        assert val is None or isinstance(val, str)


# ── explore chart provider with plotting available ──

class TestExploreChartProviderWithPlotting:
    """Cover _make_chart_provider with plotting available (lines 212-245)."""

    def test_chart_provider_with_plotting_mock(self):
        from quant_lib.cli.explore import _make_chart_provider

        cand = MagicMock()
        cand.daily_equity = {"2025-01-01": 1000.0}
        cand.executed_trades = [
            {"symbol": "BTCUSDT", "exit_time": pd.Timestamp("2025-01-15"), "r_net": 1.0},
        ]
        cand.fold_params = {"BTCUSDT": {"fold_1": {"pf": 1.5}}}
        session = MagicMock()
        session.initial_capital = 1000.0

        provider = _make_chart_provider(cand, session, no_plots=False)
        # Calls go to real plotting module — code path exercised, no crash.
        _ = provider("equity_curve")
        _ = provider("drawdown_underwater")
        _ = provider("trade_distribution")
        _ = provider("per_symbol_equity")
        _ = provider("wfa_progression")
        unknown = provider("nonexistent")
        assert unknown is None

    def test_chart_provider_catches_exception(self):
        """When a chart function raises, provider catches and returns None."""
        from quant_lib.cli.explore import _make_chart_provider

        cand = MagicMock()
        cand.daily_equity = {"2025-01-01": "not_a_float"}  # breaks real plotting
        cand.executed_trades = []
        cand.fold_params = {}
        session = MagicMock()
        session.initial_capital = 1000.0

        provider = _make_chart_provider(cand, session, no_plots=False)
        # Should not crash — exception is caught, None returned
        result = provider("equity_curve")
        assert result is None
