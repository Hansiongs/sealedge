"""Unit tests for the HTML report builders in ``quant_lib.cli._report``.

The builders take duck-typed ``candidate``, ``session``, and
``result`` objects, plus a ``chart_provider`` callable. Tests use
lightweight stubs to keep the test surface small.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import pytest

from quant_lib.cli._report import build_explore_report, build_commit_report


# ═══════════════════════════════════════════════════════════════════════
# Stubs
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class StubHypothesis:
    name: str = "test_exp"
    strategy_name: str = "vol_compression"
    mechanism: str = "Test mechanism"
    boundary_conditions: str = "Test boundary"
    success_criteria: str = "SPA p < 0.15"
    entry_logic: str = "test entry"
    exit_logic: str = "test exit"
    min_train_months: int = 12


@dataclass
class StubJournal:
    hypothesis_name: str = "test_exp"
    entries: list = field(default_factory=list)
    initial_alpha: float = 0.05

    @property
    def n_experiments(self) -> int:
        return sum(1 for e in self.entries if e.get("category") in ("improve", "explore"))

    @property
    def n_bugfixes(self) -> int:
        return sum(1 for e in self.entries if e.get("category") == "bugfix")

    @property
    def n_ablations(self) -> int:
        return sum(1 for e in self.entries if e.get("category") == "ablation")

    def adjusted_alpha(self) -> float:
        return self.initial_alpha / (self.n_experiments + 1)


@dataclass
class StubHoldoutSet:
    sealed: bool = True

    def is_sealed(self) -> bool:
        return self.sealed


@dataclass
class StubSession:
    training_period: tuple = ("2020-01-01", "2024-12-31")
    holdout_period: tuple = ("2025-01-01", "2025-06-30")
    symbols: list = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    initial_capital: float = 1000.0
    holdout_set: StubHoldoutSet = field(default_factory=StubHoldoutSet)
    journal: StubJournal = field(default_factory=StubJournal)


@dataclass
class StubCandidate:
    hypothesis: StubHypothesis = field(default_factory=StubHypothesis)
    eligible_symbols: list = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    narrowed_symbols: list = field(default_factory=lambda: ["BTCUSDT"])
    executed_trades: list = field(default_factory=list)
    daily_equity: dict = field(default_factory=dict)
    final_equity: float = 1100.0
    spa_p_value: float = 0.05
    n_oos_trades: int = 0
    n_executed: int = 0
    n_rejected: int = 0
    cache_hits: int = 5
    cache_misses: int = 2
    reject_reasons: dict = field(default_factory=lambda: {
        "cb_cooldown": 0, "position_limit": 1, "margin_insufficient": 2,
    })
    fold_params: dict = field(default_factory=dict)

    @property
    def session(self) -> StubSession:
        return _shared_session


_shared_session = StubSession()


@dataclass
class StubCommitResult:
    candidate_name: str = "test_exp"
    commit_idx: int = 1
    holdout_period: tuple = ("2025-01-01", "2025-06-30")
    timestamp: str = "2025-07-01T00:00:00+00:00"
    initial_capital: float = 1000.0
    final_equity: float = 1200.0
    equity_pct: float = 20.0
    cagr_pct: float = 40.0
    max_dd_pct: float = -15.0
    n_raw_trades: int = 50
    n_executed_trades: int = 45
    n_rejected: int = 5
    reject_breakdown: dict = field(default_factory=lambda: {
        "cb_cooldown": 1, "position_limit": 2, "margin_insufficient": 2,
    })
    n_trades: int = 45
    win_rate: float = 55.0
    avg_r: float = 0.3
    median_r: float = 0.2
    std_r: float = 1.0
    best_r: float = 4.5
    worst_r: float = -2.1
    profit_factor: float = 1.6
    avg_bars_held: float = 24.0
    sharpe_r: float = 0.3
    psr: float = 0.85
    psr_ess: float = 0.85
    skew: float = 0.1
    kurtosis: float = 3.0
    ess: float = 44.0
    bonferroni_alpha: float = 0.025
    fdr_alpha: float = 0.15
    by_symbol_stats: dict = field(default_factory=lambda: {
        "BTCUSDT": {"n_trades": 25, "win_rate": 60.0, "avg_r": 0.4,
                    "profit_factor": 1.8, "total_r": 10.0},
        "ETHUSDT": {"n_trades": 20, "win_rate": 50.0, "avg_r": 0.2,
                    "profit_factor": 1.3, "total_r": 4.0},
    })
    with_trend_trades: int = 30
    with_trend_r_total: float = 12.0
    counter_trend_trades: int = 15
    counter_trend_r_total: float = -2.0
    seal_hash_before: str = "0" * 64
    seal_hash_after: str = "1" * 64
    seal_broken: bool = True
    success_criteria_text: str = "SPA p < 0.15, PF > 1.3"


def _no_charts(_name: str) -> Optional[str]:
    """Chart provider that returns None for every chart (no-plots mode)."""
    return None


def _all_charts(_name: str) -> str:
    """Chart provider that returns a fake base64 URI for every chart."""
    return "data:image/png;base64,FAKE"


# ═══════════════════════════════════════════════════════════════════════
# build_explore_report
# ═══════════════════════════════════════════════════════════════════════


class TestBuildExploreReport:
    def test_returns_list_of_tuples(self):
        cand = StubCandidate()
        session = cand.session
        sections = build_explore_report(cand, session, _no_charts)
        assert isinstance(sections, list)
        for s in sections:
            assert isinstance(s, tuple)
            assert len(s) == 2

    def test_includes_hypothesis_section(self):
        cand = StubCandidate()
        sections = build_explore_report(cand, cand.session, _no_charts)
        headings = [h for h, _ in sections]
        assert "Hypothesis" in headings

    def test_includes_period_section(self):
        cand = StubCandidate()
        sections = build_explore_report(cand, cand.session, _no_charts)
        headings = [h for h, _ in sections]
        assert any("Period" in h for h in headings)

    def test_includes_portfolio_section(self):
        cand = StubCandidate()
        sections = build_explore_report(cand, cand.session, _no_charts)
        headings = [h for h, _ in sections]
        assert any("Portfolio" in h for h in headings)

    def test_includes_spa_section(self):
        cand = StubCandidate()
        sections = build_explore_report(cand, cand.session, _no_charts)
        headings = [h for h, _ in sections]
        assert any("SPA" in h for h in headings)

    def test_includes_journal_section(self):
        cand = StubCandidate()
        sections = build_explore_report(cand, cand.session, _no_charts)
        headings = [h for h, _ in sections]
        assert any("Journal" in h for h in headings)

    def test_charts_skipped_when_provider_returns_none(self):
        cand = StubCandidate()
        sections = build_explore_report(cand, cand.session, _no_charts)
        # No section with chart dispatch tuples when provider returns None
        for heading, content in sections:
            if isinstance(content, tuple) and len(content) == 2:
                if content[0] == "chart":
                    pytest.fail(f"Chart section {heading!r} should be skipped")

    def test_charts_included_when_provider_returns_data(self):
        cand = StubCandidate()
        cand.executed_trades = [
            {"r_net": 0.5, "exit_time": pd.Timestamp("2024-12-01"), "symbol": "BTCUSDT"},
            {"r_net": -0.3, "exit_time": pd.Timestamp("2024-12-02"), "symbol": "BTCUSDT"},
        ]
        sections = build_explore_report(cand, cand.session, _all_charts)
        chart_sections = [
            (h, c) for h, c in sections
            if isinstance(c, tuple) and c[0] == "chart"
        ]
        # At least 4 charts: equity, drawdown, trade dist, per-symbol, wfa
        # (wfa only if fold_params has data)
        assert len(chart_sections) >= 4

    def test_trade_distribution_only_when_trades_exist(self):
        cand = StubCandidate()
        cand.executed_trades = []
        sections = build_explore_report(cand, cand.session, _all_charts)
        # No trade_distribution chart when no executed trades
        chart_names = [
            c[1] for h, c in sections
            if isinstance(c, tuple) and c[0] == "chart"
        ]
        # All chart calls return FAKE; we just verify the count
        # (no trades means fewer chart sections)
        assert len(chart_names) == 4  # equity, drawdown, per_symbol, wfa (no fold data)

    def test_no_trades_skips_distribution_chart(self):
        cand = StubCandidate()
        cand.executed_trades = []
        # Provide a chart provider that tags which chart is requested
        requested = []

        def tracking_provider(name):
            requested.append(name)
            return "data:image/png;base64,FAKE"

        sections = build_explore_report(cand, cand.session, tracking_provider)
        assert "trade_distribution" not in requested
        assert "equity_curve" in requested
        assert "drawdown_underwater" in requested

    def test_reject_reasons_missing_keys_handled(self):
        """reject_reasons missing keys must not crash; treated as 0."""
        cand = StubCandidate()
        cand.reject_reasons = {}  # empty
        sections = build_explore_report(cand, cand.session, _no_charts)
        # The "Portfolio Simulation" section is built from the empty dict
        # (all counts default to 0 via .get(key, 0)). Values are formatted
        # as strings; check the int value via int() round-trip.
        for h, content in sections:
            if "Portfolio" in h:
                for label, val in content:
                    if label.strip().startswith(("CB", "Position", "Margin")):
                        assert int(val) == 0

    def test_holdout_seal_status_in_journal_section(self):
        cand = StubCandidate()
        session = cand.session
        session.holdout_set.sealed = True
        sections = build_explore_report(cand, session, _no_charts)
        for h, content in sections:
            if "Journal" in h and isinstance(content, list):
                kv = dict(content)
                assert kv.get("Holdout status") == "SEALED"

        session.holdout_set.sealed = False
        sections = build_explore_report(cand, session, _no_charts)
        for h, content in sections:
            if "Journal" in h and isinstance(content, list):
                kv = dict(content)
                assert kv.get("Holdout status") == "BROKEN"


# ═══════════════════════════════════════════════════════════════════════
# build_commit_report
# ═══════════════════════════════════════════════════════════════════════


class TestBuildCommitReport:
    def test_returns_list_of_tuples(self):
        result = StubCommitResult()
        session = StubSession()
        sections = build_commit_report(result, session, _no_charts)
        assert isinstance(sections, list)
        for s in sections:
            assert isinstance(s, tuple)
            assert len(s) == 2

    def test_includes_precommit_section(self):
        result = StubCommitResult()
        sections = build_commit_report(result, StubSession(), _no_charts)
        headings = [h for h, _ in sections]
        assert any("Pre-Commit" in h for h in headings)

    def test_includes_success_criteria_section(self):
        result = StubCommitResult()
        sections = build_commit_report(result, StubSession(), _no_charts)
        # Success criteria text exists -> section added
        found = False
        for h, content in sections:
            if "Success Criteria" in h and isinstance(content, str):
                assert "SPA p < 0.15" in content
                found = True
        assert found

    def test_no_success_criteria_omits_section(self):
        result = StubCommitResult()
        result.success_criteria_text = ""
        sections = build_commit_report(result, StubSession(), _no_charts)
        for h, _ in sections:
            assert "Success Criteria" not in h

    def test_includes_rejection_breakdown_table(self):
        result = StubCommitResult()
        sections = build_commit_report(result, StubSession(), _no_charts)
        # Find the rejection breakdown section
        found_table = None
        for h, content in sections:
            if "Rejection" in h and isinstance(content, tuple) and content[0] == "table":
                found_table = content[1]
        assert found_table is not None
        # First row is header
        assert found_table[0] == ["Reason", "Count"]
        # Each non-header row has 2 cells
        for row in found_table[1:]:
            assert len(row) == 2

    def test_rejection_breakdown_empty_dict(self):
        result = StubCommitResult()
        result.reject_breakdown = {}
        sections = build_commit_report(result, StubSession(), _no_charts)
        for h, content in sections:
            if "Rejection" in h and isinstance(content, tuple) and content[0] == "table":
                # Should have a "no rejections" placeholder row
                rows = content[1]
                assert any("(no rejections)" in str(r) for r in rows)

    def test_includes_by_symbol_table(self):
        result = StubCommitResult()
        sections = build_commit_report(result, StubSession(), _no_charts)
        for h, content in sections:
            if h == "By Symbol" and isinstance(content, tuple) and content[0] == "table":
                rows = content[1]
                # Header + 2 symbol rows
                assert rows[0] == ["Symbol", "N", "WR", "Avg R", "PF", "Total R"]
                assert len(rows) == 3
                symbols = {row[0] for row in rows[1:]}
                assert symbols == {"BTCUSDT", "ETHUSDT"}
                return
        pytest.fail("By Symbol section not found")

    def test_no_by_symbol_stats_omits_section(self):
        result = StubCommitResult()
        result.by_symbol_stats = {}
        sections = build_commit_report(result, StubSession(), _no_charts)
        for h, _ in sections:
            assert h != "By Symbol"

    def test_includes_seal_section(self):
        result = StubCommitResult()
        sections = build_commit_report(result, StubSession(), _no_charts)
        for h, content in sections:
            if "Seal" in h and "Audit" in h:
                kv = dict(content)
                assert "Seal status" in kv
                assert kv["Seal status"] == "BROKEN"
                return
        pytest.fail("Seal & Audit section not found")

    def test_includes_final_summary(self):
        result = StubCommitResult()
        sections = build_commit_report(result, StubSession(), _no_charts)
        for h, _ in sections:
            if "Final Summary" in h:
                return
        pytest.fail("Final Summary section not found")

    def test_charts_skipped_when_no_charts_provider(self):
        result = StubCommitResult()
        sections = build_commit_report(result, StubSession(), _no_charts)
        for h, content in sections:
            if isinstance(content, tuple) and content[0] == "chart":
                pytest.fail(f"Chart {h} should be skipped")

    def test_charts_included_when_provider_returns_data(self):
        result = StubCommitResult()
        sections = build_commit_report(result, StubSession(), _all_charts)
        chart_sections = [
            (h, c) for h, c in sections
            if isinstance(c, tuple) and c[0] == "chart"
        ]
        # 3 charts: equity, drawdown, trade_distribution
        assert len(chart_sections) == 3
