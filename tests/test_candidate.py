"""Tests for Candidate class."""

import tempfile
import pytest

from quant_lib.audit import for_vol_compression, for_pullback_sniper
from quant_lib.experiments.base import StrategyConfig
from quant_lib.research.session import ResearchSession
from quant_lib.research.exceptions import (
    NotReadyForCommit,
    InvalidStageTransition,
)


class TestCandidateInit:
    def test_init_sets_initial_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            hyp = for_vol_compression("v1", "m", "b", "c")
            cand = session.create_candidate(hyp)
            assert cand.stage == "hypothesis"
            assert cand.is_ready_for_commit is False
            assert cand.n_oos_trades == 0
            assert cand.n_executed == 0
            assert cand.n_rejected == 0

    def test_init_declares_is_trades_field(self):
        """v0.4.0 (Phase 2.4): _is_trades_per_fold_by_sym must exist
        at __init__ as a dataclass field (not lazily via hasattr).

        This avoids .pyc cache mismatch bugs and makes the field
        visible to type checkers / IDE autocomplete.
        """
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            cand = session.create_candidate(for_vol_compression("v1", "m", "b", "c"))
            # Field must be in dataclass fields (not lazily attached)
            from dataclasses import fields
            field_names = {f.name for f in fields(cand)}
            assert "_is_trades_per_fold_by_sym" in field_names, (
                f"_is_trades_per_fold_by_sym must be a dataclass field, not "
                f"lazily attached. Got fields: {field_names}"
            )
            # Initial value must be empty dict (not missing attribute)
            assert cand._is_trades_per_fold_by_sym == {}

    def test_default_strategy_is_strategy_config(self):
        """Without explicit strategy, Candidate uses StrategyConfig() defaults."""
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            cand = session.create_candidate(for_vol_compression("v1", "m", "b", "c"))
            assert isinstance(cand.strategy, StrategyConfig)
            # Verify framework defaults (must match StrategyConfig dataclass).
            assert cand.strategy.pf_weight_clamp_floor == 0.5
            assert cand.strategy.pf_weight_clamp_ceiling == 1.5
            assert cand.strategy.pf_decay_halflife_folds == 2
            assert cand.strategy.pf_min_trades_for_weight == 10

    def test_explicit_strategy_propagated(self):
        """session.create_candidate(hyp, strategy=...) stores the custom config."""
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            custom = StrategyConfig(
                pf_weight_clamp_floor=0.3,
                pf_weight_clamp_ceiling=1.2,
                pf_decay_halflife_folds=3,
                pf_min_trades_for_weight=5,
            )
            cand = session.create_candidate(
                for_vol_compression("v1", "m", "b", "c"),
                strategy=custom,
            )
            assert cand.strategy.pf_weight_clamp_floor == 0.3
            assert cand.strategy.pf_weight_clamp_ceiling == 1.2
            assert cand.strategy.pf_decay_halflife_folds == 3
            assert cand.strategy.pf_min_trades_for_weight == 5


class TestCandidateStateMachine:
    def test_assert_ready_raises_when_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            cand = session.create_candidate(for_vol_compression("v1", "m", "b", "c"))
            with pytest.raises(NotReadyForCommit):
                cand.assert_ready()

    def test_repr(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            cand = session.create_candidate(for_vol_compression("vol_v1", "m", "b", "c"))
            r = repr(cand)
            assert "vol_v1" in r
            assert "hypothesis" in r
            assert "vol_compression" in r

    def test_repr_pullback_sniper(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            cand = session.create_candidate(for_pullback_sniper("pbs_v1", "m", "b", "c"))
            r = repr(cand)
            assert "pullback_sniper" in r

    def test_narrowed_symbols_initially_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            cand = session.create_candidate(for_vol_compression("v1", "m", "b", "c"))
            assert cand.narrowed_symbols == []

    def test_frozen_params_initially_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            cand = session.create_candidate(for_vol_compression("v1", "m", "b", "c"))
            assert cand.frozen_params == {}

    def test_equity_change_pct_initially_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            cand = session.create_candidate(for_vol_compression("v1", "m", "b", "c"))
            assert cand.equity_change_pct == 0.0


class TestCandidateStageValidation:
    def test_set_stage_forward_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            cand = session.create_candidate(for_vol_compression("v1", "m", "b", "c"))
            # hypothesis -> universe
            cand._set_stage("universe")
            assert cand.stage == "universe"
            # universe -> edge
            cand._set_stage("edge")
            assert cand.stage == "edge"

    def test_set_stage_backward_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            cand = session.create_candidate(for_vol_compression("v1", "m", "b", "c"))
            cand._set_stage("universe")
            with pytest.raises(InvalidStageTransition):
                cand._set_stage("hypothesis")  # backward

    def test_set_stage_skip_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            cand = session.create_candidate(for_vol_compression("v1", "m", "b", "c"))
            with pytest.raises(InvalidStageTransition):
                cand._set_stage("edge")  # skip universe

    def test_assert_stage_at_least(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            cand = session.create_candidate(for_vol_compression("v1", "m", "b", "c"))
            cand._set_stage("universe")
            cand._assert_stage_at_least("hypothesis")  # ok
            with pytest.raises(InvalidStageTransition):
                cand._assert_stage_at_least("edge")  # too far
