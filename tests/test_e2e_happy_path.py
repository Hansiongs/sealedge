"""End-to-end integration tests for the new research flow.

These tests verify that the research flow layers wire together correctly:
- ResearchSession
- Candidate (state machine + phases)
- commit_to_holdout

The full pipeline is exercised in pieces. For full-pipeline smoke testing
with real data, use the Python API (see README.md) or the `quant_exp` CLI.

NOTE: The WFA + SPA tests are NOT exercised here because they require
carefully-tuned data that produces trades. Those paths are covered by
test_wfa_coverage.py and test_spa_coverage.py separately.
"""

import tempfile
from contextlib import contextmanager

import pandas as pd
import pytest

from quant_lib.audit import for_vol_compression
from quant_lib.core._config import STATIC
from quant_lib.research.candidate import Candidate
from quant_lib.research.commit import CommitResult, commit_to_holdout
from quant_lib.research.session import ResearchSession
from quant_lib.research.exceptions import (
    InvalidStageTransition,
    NotReadyForCommit,
)

# M-2: re-export shared _MockCache for backward compat with tests
# that import from this module (e.g., external scripts).
from tests.conftest import _MockCache  # noqa: F401


@contextmanager
def _patch_statics(**overrides):
    """Temporarily override STATIC values for the duration of the test."""
    saved = {k: STATIC.get(k) for k in overrides}
    for k, v in overrides.items():
        STATIC[k] = v
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                STATIC.pop(k, None)
            else:
                STATIC[k] = v


def _make_session_candidate(
    tmp: str, mock: _MockCache, name: str = "e2e_v1"
) -> tuple:
    """Create a ResearchSession + Candidate with mock cache.

    C-2 fix: provide synthetic holdout data via _holdout_data so the
    session init can hash it without network access. The mock cache
    is used for other calls (run_universe, etc.).
    """
    # Build synthetic holdout data (1H bars covering the holdout period)
    times = pd.date_range("2025-01-01", "2025-06-30", freq="h")
    holdout_data = {
        sym: pd.DataFrame({
            "time": times,
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.0, "volume": 1000.0,
        })
        for sym in ["BTCUSDT", "ETHUSDT"]
    }
    session = ResearchSession(
        training_period=("2020-01-01", "2024-12-31"),
        holdout_period=("2025-01-01", "2025-06-30"),
        symbols=["BTCUSDT", "ETHUSDT"],
        cache_dir=tmp,
        btc_data_start="2019-06-01",
        _holdout_data=holdout_data,
    )
    session.cache = mock
    hyp = for_vol_compression(name, "m", "b", "c")
    cand = session.create_candidate(hyp)
    return session, cand


def _prep_candidate_for_commit(
    cand: Candidate, tmp: str, mock: _MockCache, frozen_params: dict
) -> None:
    """Run run_universe then skip WFA by setting state directly.

    Populates all the fields commit_to_holdout expects, with
    caller-supplied frozen_params.
    """
    cand.run_universe(min_volume_usdt=100, min_age_days=0)
    cand._set_stage("edge")
    cand._set_stage("narrowed")
    cand.narrowed_symbols = list(cand.eligible_symbols)
    cand.frozen_params = frozen_params
    # risk_weights must be set (commit.py uses them)
    cand.risk_weights = {sym: 0.01 for sym in cand.narrowed_symbols}


# ─────────────────────────────────────────────────────────────────────
# Layer 1: Session + Candidate creation
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestSessionCandidateCreation:
    def test_session_creates_and_seals_holdout(self):
        """Session at init seals the holdout (audit invariant)."""
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
            symbols=["BTCUSDT"],
            cache_dir=tmp, _skip_holdout_load=True,
            )
            assert session.holdout_set.is_sealed()
            assert not session.holdout_set.is_broken()

    def test_candidate_starts_at_hypothesis_stage(self):
        """Newly created candidate is in 'hypothesis' stage."""
        with tempfile.TemporaryDirectory() as tmp:
            _, cand = _make_session_candidate(tmp, _MockCache())
            assert cand.stage == "hypothesis"
            assert cand.is_ready_for_commit is False

    def test_multiple_candidates_managed_by_session(self):
        """Session tracks all candidates; each has unique name."""
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
            symbols=["BTCUSDT"],
            cache_dir=tmp, _skip_holdout_load=True,
            )
            for i in range(3):
                h = for_vol_compression(f"v{i}", "m", "b", "c")
                session.create_candidate(h)
            assert len(session.candidates) == 3


# ─────────────────────────────────────────────────────────────────────
# Layer 2: Phase 1 (universe)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestPhase1Universe:
    def test_run_universe_fetches_data_and_computes_features(self):
        """run_universe populates precomputed_data with features."""
        with tempfile.TemporaryDirectory() as tmp:
            _, cand = _make_session_candidate(tmp, _MockCache())
            cand.run_universe(min_volume_usdt=100, min_age_days=0)
            assert cand.stage == "universe"
            assert "BTCUSDT" in cand.precomputed_data
            df = cand.precomputed_data["BTCUSDT"]
            for col in ("hh_20", "ll_20", "vol_pct_rank", "rvol", "atr"):
                assert col in df.columns, f"Missing feature: {col}"


# ─────────────────────────────────────────────────────────────────────
# Layer 3: State machine invariants
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestStateMachineInvariants:
    def test_cannot_skip_stages(self):
        """Cannot run_edge_testing directly from 'hypothesis' stage."""
        with tempfile.TemporaryDirectory() as tmp:
            _, cand = _make_session_candidate(tmp, _MockCache())
            with pytest.raises(InvalidStageTransition):
                cand.run_edge_testing()

    def test_cannot_go_backward(self):
        """Cannot transition from 'narrowed' to 'universe' (backward)."""
        with tempfile.TemporaryDirectory() as tmp:
            _, cand = _make_session_candidate(tmp, _MockCache())
            cand.run_universe(min_volume_usdt=100, min_age_days=0)
            cand._set_stage("edge")
            cand._set_stage("narrowed")
            with pytest.raises(InvalidStageTransition):
                cand._set_stage("universe")

    def test_mark_ready_validates_prerequisites(self):
        """mark_ready refuses if narrowed_symbols is empty."""
        with tempfile.TemporaryDirectory() as tmp:
            _, cand = _make_session_candidate(tmp, _MockCache())
            cand.run_universe(min_volume_usdt=100, min_age_days=0)
            cand._set_stage("edge")
            cand._set_stage("narrowed")
            cand.narrowed_symbols = []
            with pytest.raises(NotReadyForCommit):
                cand.mark_ready()

    def test_mark_ready_succeeds_when_prepared(self):
        """mark_ready works when candidate is fully prepared."""
        with tempfile.TemporaryDirectory() as tmp:
            _, cand = _make_session_candidate(tmp, _MockCache())
            cand.run_universe(min_volume_usdt=100, min_age_days=0)
            cand._set_stage("edge")
            cand._set_stage("narrowed")
            cand.narrowed_symbols = ["BTCUSDT", "ETHUSDT"]
            cand.frozen_params = {
                "BTCUSDT": {"vol_pct_thresh": 0.2, "pullback_bars": 5,
                              "trail_atr": 3.0, "sl_mult": 1.5},
                "ETHUSDT": {"vol_pct_thresh": 0.2, "pullback_bars": 5,
                              "trail_atr": 3.0, "sl_mult": 1.5},
            }
            cand.mark_ready()
            assert cand.stage == "ready"
            assert cand.is_ready_for_commit


# ─────────────────────────────────────────────────────────────────────
# Layer 4: commit_to_holdout (pre-populated candidate)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestCommitToHoldout:
    def test_commit_with_no_trades_returns_valid_zero_result(self):
        """If engine produces 0 trades, commit returns a valid zero-result."""
        with tempfile.TemporaryDirectory() as tmp:
            session, cand = _make_session_candidate(
                tmp, _MockCache(), name="zero_v1"
            )
            # Very tight params that produce 0 trades
            _prep_candidate_for_commit(
                cand, tmp, _MockCache(),
                frozen_params={
                    "BTCUSDT": {
                        "vol_pct_thresh": 0.01, "pullback_bars": 50,
                        "trail_atr": 0.5, "sl_mult": 0.5,
                    },
                },
            )
            cand.narrowed_symbols = ["BTCUSDT"]  # Force to BTC only
            cand.frozen_params = {
                "BTCUSDT": {
                    "vol_pct_thresh": 0.01, "pullback_bars": 50,
                    "trail_atr": 0.5, "sl_mult": 0.5,
                },
            }
            cand.mark_ready()
            result = commit_to_holdout(
                cand, success_criteria_text="x", verbose=False,
            )
            assert isinstance(result, CommitResult)
            assert result.candidate_name == "zero_v1"
            assert result.n_trades == 0
            assert result.equity_pct == 0.0
            assert result.cagr_pct == 0.0
            assert result.max_dd_pct == 0.0
            assert result.seal_broken
            assert session.holdout_set.is_broken()

    def test_commit_records_in_session_and_journal(self):
        """After commit, session has a commit record and journal has entry."""
        with tempfile.TemporaryDirectory() as tmp:
            session, cand = _make_session_candidate(
                tmp, _MockCache(), name="audit_v1"
            )
            _prep_candidate_for_commit(
                cand, tmp, _MockCache(),
                frozen_params={
                    "BTCUSDT": {"vol_pct_thresh": 0.2, "pullback_bars": 5,
                                  "trail_atr": 3.0, "sl_mult": 1.5},
                    "ETHUSDT": {"vol_pct_thresh": 0.2, "pullback_bars": 5,
                                  "trail_atr": 3.0, "sl_mult": 1.5},
                },
            )
            cand.mark_ready()
            journal_before = len(session.journal.entries)
            commit_to_holdout(
                cand, success_criteria_text="test criteria", verbose=False,
            )
            assert len(session._commits) == 1
            assert session._commits[0].candidate_name == "audit_v1"
            assert session._commits[0].success_criteria_text == "test criteria"
            assert len(session.journal.entries) > journal_before

    def test_second_commit_on_same_session_raises(self):
        """The holdout seal can only be broken once per session."""
        with tempfile.TemporaryDirectory() as tmp:
            session, cand = _make_session_candidate(
                tmp, _MockCache(), name="first_v1"
            )
            _prep_candidate_for_commit(
                cand, tmp, _MockCache(),
                frozen_params={
                    "BTCUSDT": {"vol_pct_thresh": 0.2, "pullback_bars": 5,
                                  "trail_atr": 3.0, "sl_mult": 1.5},
                },
            )
            cand.mark_ready()
            commit_to_holdout(
                cand, success_criteria_text="first", verbose=False,
            )
            # Create a second candidate, try to commit
            cand2 = session.create_candidate(
                for_vol_compression("second_v1", "m", "b", "c")
            )
            cand2._set_stage("universe")
            cand2._set_stage("edge")
            cand2._set_stage("narrowed")
            cand2.narrowed_symbols = ["BTCUSDT"]
            cand2.frozen_params = cand.frozen_params
            cand2.risk_weights = cand.risk_weights
            cand2.precomputed_data = cand.precomputed_data
            cand2.mark_ready()
            from quant_lib.research.exceptions import (
                CommitError,
                HoldoutAlreadyBroken,
            )
            with pytest.raises((HoldoutAlreadyBroken, CommitError)):
                commit_to_holdout(
                    cand2, success_criteria_text="second", verbose=False,
                )
