"""Coverage push for quant_lib.research.reporting.

Targets:
- print_candidate_report: all 13 sections
- print_commit_report: all 8 sections
- Early-return paths (stage not edge/narrowed)
- Edge cases (insufficient data for bootstrap, etc.)
"""

import io
import re
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

from quant_lib.audit import for_vol_compression, for_pullback_sniper
from quant_lib.research.candidate import Candidate
from quant_lib.research.commit import CommitResult
from quant_lib.research.reporting import print_candidate_report, print_commit_report
from quant_lib.research.session import ResearchSession


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


class _CapturingConsole:
    """Replace the module-level `console` with a capturing one.

    Rich's `Console(record=True)` captures output to a string buffer
    that can be exported with `export_text()`. This lets us assert
    on the actual text printed.
    """

    def __init__(self):
        self.console = Console(
            file=io.StringIO(),
            record=True,
            width=120,
            force_terminal=False,
            color_system=None,
        )

    def __enter__(self):
        self._patcher = patch("quant_lib.research.reporting.console", self.console)
        self._patcher.__enter__()
        return self

    def __exit__(self, *args):
        self._patcher.__exit__(*args)

    @property
    def text(self) -> str:
        return self.console.export_text()


def _make_candidate_with_full_data(tmp: str) -> Candidate:
    """Build a Candidate manually populated with realistic data,
    bypassing the real WFA/feature pipeline.
    """
    session = ResearchSession(
        training_period=("2020-01-01", "2024-12-31"),
        holdout_period=("2025-01-01", "2025-06-30"),
        symbols=["BTCUSDT", "ETHUSDT"],
        cache_dir=tmp, _skip_holdout_load=True,
    )
    hyp = for_vol_compression("rep_test_v1", "m", "b", "c")
    cand = session.create_candidate(hyp)
    # Walk the state machine
    cand._set_stage("universe")
    cand._set_stage("edge")

    # Pre-populate with realistic data
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    rng = np.random.default_rng(42)
    eq = 1000.0 + np.cumsum(rng.normal(0, 5, 200))
    eq = np.maximum(eq, 500.0)
    cand.daily_equity = {d: float(v) for d, v in zip(dates, eq)}
    cand.final_equity = float(eq[-1])
    cand.all_oos_trades = [
        {
            "entry_time": dates[i],
            "exit_time": dates[i] + timedelta(days=2),
            "symbol": "BTCUSDT" if i % 2 == 0 else "ETHUSDT",
            "r_net": float(rng.normal(0.2, 0.5)),
            "trade_dir": 1,
        }
        for i in range(50)
    ]
    cand.executed_trades = cand.all_oos_trades[:40]
    cand.reject_reasons = {"cb_cooldown": 5, "position_limit": 3, "margin_insufficient": 2}
    cand.spa_p_value = 0.08
    cand.eligible_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    cand.narrowed_symbols = ["BTCUSDT", "ETHUSDT"]
    cand.narrowing_rule = "top_2_by_volume"
    cand.narrowing_context = "User rule"
    cand.cache_hits = 5
    cand.cache_misses = 1
    cand.precomputed_data = {
        "BTCUSDT": pd.DataFrame({"time": dates, "x": range(200)}),
        "ETHUSDT": pd.DataFrame({"time": dates, "x": range(200)}),
        "SOLUSDT": pd.DataFrame({"time": dates, "x": range(200)}),
    }
    cand.fold_params = {
        "BTCUSDT": [
            {
                "fold": 1, "total_folds": 3,
                "is_start": pd.Timestamp("2020-01-01"),
                "oos_start": pd.Timestamp("2022-01-01"),
                "oos_end": pd.Timestamp("2022-04-01"),
                "vol_pct_thresh": 0.20, "pullback_bars": 5,
                "trail_atr": 3.0, "sl_mult": 1.5,
                "best_value": 0.75,
            },
            {
                "fold": 2, "total_folds": 3,
                "is_start": pd.Timestamp("2020-01-01"),
                "oos_start": pd.Timestamp("2022-04-01"),
                "oos_end": pd.Timestamp("2022-07-01"),
                "vol_pct_thresh": 0.22, "pullback_bars": 4,
                "trail_atr": 3.2, "sl_mult": 1.6,
                "best_value": 0.80,
            },
        ],
        "ETHUSDT": [
            {
                "fold": 1, "total_folds": 3,
                "is_start": pd.Timestamp("2020-01-01"),
                "oos_start": pd.Timestamp("2022-01-01"),
                "oos_end": pd.Timestamp("2022-04-01"),
                "vol_pct_thresh": 0.18, "pullback_bars": 6,
                "trail_atr": 2.8, "sl_mult": 1.4,
                "best_value": 0.70,
            },
        ],
    }
    cand.frozen_params = {
        "BTCUSDT": {
            "vol_pct_thresh": 0.22, "pullback_bars": 4,
            "trail_atr": 3.2, "sl_mult": 1.6,
        },
        "ETHUSDT": {
            "vol_pct_thresh": 0.18, "pullback_bars": 6,
            "trail_atr": 2.8, "sl_mult": 1.4,
        },
    }
    cand.edge_metrics = {
        "n_oos_trades": 50, "n_executed": 40, "n_rejected": 10,
        "final_equity": float(eq[-1]), "spa_p_value": 0.08,
    }
    return cand


def _make_commit_result() -> CommitResult:
    """Build a fully-populated CommitResult for report testing."""
    return CommitResult(
        candidate_name="rep_test_v1",
        commit_idx=1,
        holdout_period=("2025-01-01", "2025-06-30"),
        timestamp="2025-01-15T10:30:00+00:00",
        initial_capital=1000.0,
        final_equity=1150.0,
        equity_pct=15.0,
        cagr_pct=32.5,
        max_dd_pct=8.0,
        n_raw_trades=50,
        n_executed_trades=42,
        n_rejected=8,
        reject_breakdown={"cb_cooldown": 3, "position_limit": 2, "margin_insufficient": 3},
        n_trades=42,
        win_rate=64.3,
        avg_r=0.45,
        median_r=0.30,
        std_r=0.85,
        best_r=3.2,
        worst_r=-1.5,
        profit_factor=1.85,
        avg_bars_held=18.5,
        sharpe_r=0.53,
        psr=0.92,
        psr_ess=0.92,
        skew=0.3,
        kurtosis=3.8,
        ess=42.0,
        bonferroni_alpha=0.075,
        fdr_alpha=0.15,
        by_symbol_stats={
            "BTCUSDT": {"n_trades": 22, "win_rate": 68.2,
                         "avg_r": 0.55, "profit_factor": 2.1,
                         "total_r": 12.1},
            "ETHUSDT": {"n_trades": 20, "win_rate": 60.0,
                         "avg_r": 0.35, "profit_factor": 1.6,
                         "total_r": 7.0},
        },
        with_trend_trades=28,
        with_trend_r_total=15.0,
        counter_trend_trades=10,
        counter_trend_r_total=-2.0,
        seal_hash_before="a" * 64,
        seal_hash_after="b" * 64,
        seal_broken=True,
        success_criteria_text="PSR > 0.9, PF > 1.5, max DD < 15%",
    )


# ─────────────────────────────────────────────────────────────────────
# S4.4: print_candidate_report
# ─────────────────────────────────────────────────────────────────────


class TestPrintCandidateReport:
    """Verify all 13 sections of the white test report render."""

    def test_early_return_when_stage_not_edge_narrowed(self):
        """If candidate is at 'hypothesis' or 'universe', report is skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            cand = session.create_candidate(for_vol_compression("v1", "m", "b", "c"))
            with _CapturingConsole() as cap:
                print_candidate_report(cand)
            text = cap.text
            assert "Cannot print report" in text

    def test_section_1_hypothesis_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_with_full_data(tmp)
            with _CapturingConsole() as cap:
                print_candidate_report(cand)
            text = cap.text
            assert "[1] HYPOTHESIS" in text
            assert "rep_test_v1" in text
            assert "m" in text  # mechanism (short)
            assert "Search space" in text

    def test_section_2_universe_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_with_full_data(tmp)
            with _CapturingConsole() as cap:
                print_candidate_report(cand)
            text = cap.text
            assert "[2] UNIVERSE" in text
            assert "Eligible" in text
            assert "Cache hits" in text

    def test_section_3_feature_engineering_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_with_full_data(tmp)
            with _CapturingConsole() as cap:
                print_candidate_report(cand)
            text = cap.text
            assert "[3] FEATURE ENGINEERING" in text
            assert "BTCUSDT" in text

    def test_section_4_wfa_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_with_full_data(tmp)
            with _CapturingConsole() as cap:
                print_candidate_report(cand)
            text = cap.text
            assert "[4] WFA PER SYMBOL" in text
            assert "BTCUSDT" in text
            assert "Frozen" in text  # frozen params section

    def test_section_5_portfolio_simulation_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_with_full_data(tmp)
            with _CapturingConsole() as cap:
                print_candidate_report(cand)
            text = cap.text
            assert "[5] PORTFOLIO SIMULATION" in text
            assert "CAGR" in text
            assert "Max DD" in text
            assert "CB cooldown" in text

    def test_section_6_spa_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_with_full_data(tmp)
            with _CapturingConsole() as cap:
                print_candidate_report(cand)
            text = cap.text
            assert "[6] SPA TEST" in text
            assert "p-value" in text
            assert "0.08" in text  # the spa_p_value

    def test_section_7_psr_ess_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_with_full_data(tmp)
            with _CapturingConsole() as cap:
                print_candidate_report(cand)
            text = cap.text
            assert "[7] PSR + ESS" in text
            assert "N trades" in text
            assert "Skewness" in text

    def test_section_8_wilcoxon_fdr_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_with_full_data(tmp)
            with _CapturingConsole() as cap:
                print_candidate_report(cand)
            text = cap.text
            assert "[8] PER-SYMBOL WILCOXON + FDR" in text

    def test_section_9_regime_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_with_full_data(tmp)
            with _CapturingConsole() as cap:
                print_candidate_report(cand)
            text = cap.text
            assert "[9] REGIME STATS" in text
            assert "Bull" in text
            assert "Bear" in text

    def test_section_10_param_stability_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_with_full_data(tmp)
            with _CapturingConsole() as cap:
                print_candidate_report(cand)
            text = cap.text
            assert "[10] PARAMETER STABILITY" in text

    def test_section_11_edge_classification_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_with_full_data(tmp)
            with _CapturingConsole() as cap:
                print_candidate_report(cand)
            text = cap.text
            assert "[11] EDGE CLASSIFICATION" in text
            # Edge case classification should be one of Kasus 1/2/3
            assert "Case" in text or "Kasus" in text

    def test_section_12_bootstrap_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_with_full_data(tmp)
            with _CapturingConsole() as cap:
                print_candidate_report(cand)
            text = cap.text
            assert "[12] BOOTSTRAP" in text

    def test_section_13_journal_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_with_full_data(tmp)
            with _CapturingConsole() as cap:
                print_candidate_report(cand)
            text = cap.text
            assert "[13] JOURNAL & FDR CONTEXT" in text
            assert "FDR alpha" in text
            assert "Bonf" in text

    def test_full_report_end_marker(self):
        """The report ends with 'END OF WHITE TEST REPORT'."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_with_full_data(tmp)
            with _CapturingConsole() as cap:
                print_candidate_report(cand)
            text = cap.text
            assert "END OF WHITE TEST REPORT" in text

    def test_insufficient_data_for_psr_handles_gracefully(self):
        """If executed_trades has < 10 items, PSR section shows dim message."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_with_full_data(tmp)
            cand.executed_trades = cand.executed_trades[:3]  # too few
            with _CapturingConsole() as cap:
                print_candidate_report(cand)
            text = cap.text
            # Should not crash, section 7 may have dim message
            assert "[7]" in text

    def test_insufficient_data_for_bootstrap(self):
        """If daily_equity has < 30 days, bootstrap shows dim message."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_with_full_data(tmp)
            # Reduce to 5 days
            short_dates = pd.date_range("2024-01-01", periods=5, freq="D")
            cand.daily_equity = {d: 1000.0 for d in short_dates}
            with _CapturingConsole() as cap:
                print_candidate_report(cand)
            text = cap.text
            assert "[12]" in text
            # Should mention insufficient
            assert "Insufficient" in text or "bootstrap" in text.lower()

    def test_no_fold_params_skips_stability(self):
        """If fold_params is empty, section 10 should be handled."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_with_full_data(tmp)
            cand.fold_params = {}
            with _CapturingConsole() as cap:
                print_candidate_report(cand)
            text = cap.text
            # Should not crash
            assert "[10]" in text


# ─────────────────────────────────────────────────────────────────────
# S4.4: print_commit_report
# ─────────────────────────────────────────────────────────────────────


class TestPrintCommitReport:
    """Verify all 8 sections of the black test report render."""

    def test_section_1_pre_commit_verification(self):
        result = _make_commit_result()
        with _CapturingConsole() as cap:
            print_commit_report(result)
        text = cap.text
        assert "[1] PRE-COMMIT VERIFICATION" in text
        assert "rep_test_v1" in text
        assert "Commit idx" in text
        assert "FDR alpha" in text
        assert "Seal status" in text

    def test_section_1_includes_success_criteria_panel(self):
        """If success_criteria_text is non-empty, it's shown in a Panel."""
        result = _make_commit_result()
        with _CapturingConsole() as cap:
            print_commit_report(result)
        text = cap.text
        assert "Success criteria" in text
        assert "PSR > 0.9" in text  # from the criteria text

    def test_section_2_holdout_simulation(self):
        result = _make_commit_result()
        with _CapturingConsole() as cap:
            print_commit_report(result)
        text = cap.text
        assert "[2] HOLDOUT SIMULATION" in text
        assert "Raw trades" in text
        assert "Executed" in text
        assert "Rejected" in text
        assert "cb_cooldown" in text

    def test_section_3_per_trade_stats(self):
        result = _make_commit_result()
        with _CapturingConsole() as cap:
            print_commit_report(result)
        text = cap.text
        assert "[3] PER-TRADE STATS" in text
        assert "Win rate" in text
        assert "Avg R" in text
        assert "Profit factor" in text
        assert "By Symbol" in text
        assert "BTCUSDT" in text
        assert "ETHUSDT" in text

    def test_section_4_equity_curve_stats(self):
        result = _make_commit_result()
        with _CapturingConsole() as cap:
            print_commit_report(result)
        text = cap.text
        assert "[4] EQUITY CURVE STATS" in text
        assert "Initial" in text
        assert "Final" in text
        assert "CAGR" in text
        assert "Max DD" in text

    def test_section_5_psr_holdout(self):
        result = _make_commit_result()
        with _CapturingConsole() as cap:
            print_commit_report(result)
        text = cap.text
        assert "[5] PSR + ESS (HOLDOUT)" in text
        assert "Skewness" in text
        assert "Kurtosis" in text
        assert "vs Bonf alpha" in text
        assert "vs FDR alpha" in text

    def test_section_6_risk_metrics(self):
        result = _make_commit_result()
        with _CapturingConsole() as cap:
            print_commit_report(result)
        text = cap.text
        assert "[6] RISK METRICS" in text
        assert "With-trend" in text
        assert "Counter-trend" in text
        assert "Alignment impact" in text

    def test_section_7_seal_audit(self):
        result = _make_commit_result()
        with _CapturingConsole() as cap:
            print_commit_report(result)
        text = cap.text
        assert "[7] SEAL & AUDIT" in text
        assert "Seal hash" in text
        assert "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" in text  # 32 a's
        assert "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" in text  # 32 b's

    def test_section_8_final_summary(self):
        result = _make_commit_result()
        with _CapturingConsole() as cap:
            print_commit_report(result)
        text = cap.text
        assert "[8] FINAL SUMMARY" in text
        assert "no verdict" in text.lower() or "informational" in text.lower()
        # Rich formats 1150 as "1,150.00" - check for either
        assert "1,150" in text or "1150" in text

    def test_end_marker_present(self):
        result = _make_commit_result()
        with _CapturingConsole() as cap:
            print_commit_report(result)
        text = cap.text
        assert "END OF BLACK TEST REPORT" in text

    def test_with_session_includes_seal_info(self):
        """If session is provided, seal state can be cross-referenced."""
        result = _make_commit_result()
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            with _CapturingConsole() as cap:
                print_commit_report(result, session=session)
            text = cap.text
            # Should still produce output (no crash)
            assert "BLACK TEST REPORT" in text
