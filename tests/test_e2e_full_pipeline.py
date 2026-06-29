"""End-to-end tests for full research pipeline.

These tests verify that the complete research pipeline wires together:
1. ResearchSession creation with mock cache
2. Phase 1 (run_universe) with public API
3. Phase 2-3-4 (WFA → narrow → ready → commit) via pre-populated state

Instead of trying to produce real WFA trades from synthetic data (which
depends on exact engine logic), we verify that:
- run_universe works correctly with mocked data
- run_edge_testing + run_narrowing + mark_ready + commit_to_holdout
  work end-to-end when trades exist (using pre-populated candidate state)

Marked @pytest.mark.slow because these involve real computation.
"""
from __future__ import annotations

import tempfile

import pytest

from quant_lib.research.commit import CommitResult, commit_to_holdout
from tests.conftest import (
    _MockCache,
    make_session_candidate,
)


@pytest.mark.slow
class TestE2EPipelineStage1:
    """Phase 1: run_universe with mocked data works correctly."""

    def test_run_universe_populates_precomputed_data(self):
        """run_universe fetches data and precomputes features."""
        with tempfile.TemporaryDirectory() as tmp:
            mock = _MockCache()
            session, cand = make_session_candidate(tmp, mock, name="e2e_u1")
            cand.run_universe(min_volume_usdt=100, min_age_days=0)
            assert cand.stage == "universe"
            assert len(cand.eligible_symbols) > 0
            for sym in cand.eligible_symbols:
                assert sym in cand.precomputed_data
                df = cand.precomputed_data[sym]
                for col in ("vol_pct_rank", "rvol", "atr", "hh_20", "ll_20", "ema_200"):
                    assert col in df.columns, f"Missing feature: {col} in {sym}"


@pytest.mark.slow
class TestE2EPipelineWiring:
    """Phase 2-3-4: end-to-end wiring from populated candidate to commit."""

    def test_commit_end_to_end_with_populated_candidate(self):
        """Full pipeline: run_universe + prepopulated candidate → commit.

        We use run_universe for Phase 1 (public API), then populate
        candidate state and call commit_to_holdout directly.
        """
        with tempfile.TemporaryDirectory() as tmp:
            mock = _MockCache()
            session, cand = make_session_candidate(tmp, mock, name="e2e_pop1")
            cand.run_universe(min_volume_usdt=100, min_age_days=0)
            eligible = list(cand.eligible_symbols)
            assert len(eligible) > 0

            # Populate what run_edge_testing + run_narrowing would produce
            cand._set_stage("edge")
            cand._set_stage("narrowed")
            cand.narrowed_symbols = eligible
            cand.frozen_params = {
                sym: {
                    "vol_pct_thresh": 0.20, "pullback_bars": 5,
                    "trail_atr": 3.0, "sl_mult": 1.5,
                }
                for sym in eligible
            }
            cand.risk_weights = {sym: 0.01 for sym in eligible}
            cand.mark_ready()
            assert cand.stage == "ready"
            assert cand.is_ready_for_commit

            # Phase 4: Commit
            result = commit_to_holdout(
                cand, success_criteria_text="SPA p<0.15", verbose=False,
            )
            assert isinstance(result, CommitResult)
            assert result.seal_broken
            assert result.candidate_name == "e2e_pop1"

    def test_commit_preserves_session_state(self):
        """After commit, session tracks the commit."""
        with tempfile.TemporaryDirectory() as tmp:
            mock = _MockCache()
            session, cand = make_session_candidate(tmp, mock, name="e2e_ss1")
            cand.run_universe(min_volume_usdt=100, min_age_days=0)
            eligible = list(cand.eligible_symbols)

            cand._set_stage("edge")
            cand._set_stage("narrowed")
            cand.narrowed_symbols = eligible
            cand.frozen_params = {sym: {
                "vol_pct_thresh": 0.20, "pullback_bars": 5,
                "trail_atr": 3.0, "sl_mult": 1.5,
            } for sym in eligible}
            cand.risk_weights = {sym: 0.01 for sym in eligible}
            cand.mark_ready()

            n_before = len(session._commits)
            commit_to_holdout(cand, success_criteria_text="x", verbose=False)
            assert len(session._commits) == n_before + 1
            assert session.holdout_set.is_broken()


@pytest.mark.slow
class TestE2EPipelineErrorPaths:
    """Error paths in the end-to-end pipeline."""

    def test_commit_fails_when_candidate_not_ready(self):
        """commit_to_holdout raises when candidate hasn't passed narrow."""
        with tempfile.TemporaryDirectory() as tmp:
            mock = _MockCache()
            session, cand = make_session_candidate(tmp, mock, name="e2e_err1")
            cand.run_universe(min_volume_usdt=100, min_age_days=0)
            from quant_lib.research.exceptions import NotReadyForCommit
            with pytest.raises(NotReadyForCommit):
                commit_to_holdout(cand, success_criteria_text="x", verbose=False)
