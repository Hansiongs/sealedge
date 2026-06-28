"""Tests for ResearchSession."""

import os
import tempfile
import pytest
from datetime import datetime

from quant_lib.audit import Hypothesis, for_vol_compression, for_pullback_sniper
from quant_lib.research.session import ResearchSession
from quant_lib.research.exceptions import InvalidPeriod


class TestResearchSession:
    def test_create_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT", "ETHUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            assert session.training_period == ("2020-01-01", "2024-12-31")
            assert session.holdout_period == ("2025-01-01", "2025-06-30")
            assert session.symbols == ["BTCUSDT", "ETHUSDT"]

    def test_session_seals_holdout_at_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            assert session.holdout_set.is_sealed() is True

    def test_session_creates_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            assert session.journal is not None
            assert len(session.journal.entries) >= 1  # at least init log

    def test_invalid_period_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(InvalidPeriod):
                ResearchSession(
                    training_period=("2025-01-01", "2025-12-31"),
                    holdout_period=("2025-01-01", "2025-06-30"),  # before training end
                    symbols=["BTCUSDT"],
                    cache_dir=tmp, _skip_holdout_load=True,
                )

    def test_holdout_one_day_after_train_accepted(self):
        """Boundary case: hold_start = train_end + 1 day must pass.

        PeriodConfig auto-resolution generates this pattern via
        train_end + pd.Timedelta(days=1). Validation must allow it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            assert session.training_period[1] == "2024-12-31"
            assert session.holdout_period[0] == "2025-01-01"

    def test_holdout_same_day_as_train_rejected(self):
        """hold_start == train_end must fail (strict comparison)."""
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(InvalidPeriod):
                ResearchSession(
                    training_period=("2020-01-01", "2024-12-31"),
                    holdout_period=("2024-12-31", "2025-06-30"),  # same day
                    symbols=["BTCUSDT"],
                    cache_dir=tmp, _skip_holdout_load=True,
                )

    def test_holdout_overlapping_train_rejected(self):
        """hold_start < train_end must fail (overlap = look-ahead)."""
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(InvalidPeriod):
                ResearchSession(
                    training_period=("2020-01-01", "2024-12-31"),
                    holdout_period=("2024-06-30", "2024-12-31"),  # last 6mo of train
                    symbols=["BTCUSDT"],
                    cache_dir=tmp, _skip_holdout_load=True,
                )

    def test_holdout_far_after_train_accepted(self):
        """Far-future holdout (e.g., 1 year gap) must pass validation."""
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2026-01-01", "2026-06-30"),  # 1y gap
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            assert session.holdout_period == ("2026-01-01", "2026-06-30")

    def test_create_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            hyp = for_vol_compression(
                "v1", "mech", "boundary", "criteria"
            )
            cand = session.create_candidate(hyp)
            assert cand.hypothesis.name == "v1"
            assert cand.stage == "hypothesis"
            assert len(session.candidates) == 1

    def test_create_candidate_auto_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            initial_log_count = len(session.journal.entries)
            hyp = for_vol_compression("v1", "mech", "boundary", "criteria")
            session.create_candidate(hyp)
            assert len(session.journal.entries) > initial_log_count

    def test_create_candidate_validates_hypothesis(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            bad_hyp = Hypothesis(
                name="bad",
                mechanism="",  # missing
                boundary_conditions="x",
                success_criteria="x",
                entry_logic="x",
                exit_logic="x",
            )
            with pytest.raises(Exception):
                session.create_candidate(bad_hyp)

    def test_create_multiple_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            for i in range(3):
                hyp = for_vol_compression(f"v{i}", "m", "b", "c")
                session.create_candidate(hyp)
            assert len(session.candidates) == 3

    def test_n_commits_starts_at_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            assert session.n_commits == 0

    def test_current_bonferroni_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                bonferroni_base=0.15,
                cache_dir=tmp, _skip_holdout_load=True,
            )
            # 0 commits: alpha = 0.15 / (0+1) = 0.15
            assert session.current_bonferroni_alpha == pytest.approx(0.15, rel=1e-6)
            # Simulate 1 commit
            from quant_lib.research.session import SessionCommitRecord
            session._commits.append(SessionCommitRecord(
                candidate_name="t", timestamp="2024", final_equity=1000,
                equity_pct=0, n_trades=0, psr=0.5, seal_hash="x",
                success_criteria_text=""
            ))
            # 1 commit: alpha = 0.15 / (1+1) = 0.075
            assert session.current_bonferroni_alpha == pytest.approx(0.075, rel=1e-6)

    def test_adjusted_alpha_for_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                bonferroni_base=0.15,
                cache_dir=tmp, _skip_holdout_load=True,
            )
            assert session.adjusted_alpha_for_commit(1) == pytest.approx(0.15, rel=1e-6)
            assert session.adjusted_alpha_for_commit(2) == pytest.approx(0.075, rel=1e-6)
            assert session.adjusted_alpha_for_commit(3) == pytest.approx(0.05, rel=1e-6)

    def test_fdr_alpha_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            assert session.fdr_alpha == 0.15  # DEFAULT_FDR_ALPHA

    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            s = session.summary()
            assert "ResearchSession" in s
            assert "2020-01-01" in s
            assert "2025-01-01" in s
            assert "candidates=0" in s
            assert "commits=0" in s

    def test_repr(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            assert repr(session) == session.summary()

    def test_record_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            hyp = for_vol_compression("v1", "m", "b", "c")
            cand = session.create_candidate(hyp)
            session.record_commit(
                candidate=cand,
                final_equity=1500.0,
                equity_pct=50.0,
                n_trades=42,
                psr=0.85,
                seal_hash="abc123",
                success_criteria_text="equity > 20%",
            )
            assert session.n_commits == 1
            assert session._commits[0].candidate_name == "v1"
            assert session._commits[0].success_criteria_text == "equity > 20%"
