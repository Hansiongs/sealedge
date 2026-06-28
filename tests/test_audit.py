"""Tests for audit module — Hypothesis, ExperimentLog, HoldoutSet."""

import json
import tempfile
from datetime import datetime, timezone

import pandas as pd
import pytest

from quant_lib.audit.hypothesis import Hypothesis
from quant_lib.audit.journal import ExperimentLog
from quant_lib.audit.holdout import HoldoutSet, HoldoutSeal


class TestHypothesis:
    def test_valid_hypothesis_passes_validation(self):
        h = Hypothesis(
            name="test",
            mechanism="Mean reversion in 1h timeframe",
            boundary_conditions="Fails during strong trends",
            success_criteria="SPA p < 0.15, PF > 1.3",
            entry_logic="RSI < 30",
            exit_logic="Trailing stop",
        )
        missing = h.validate()
        assert missing == []

    def test_invalid_hypothesis_missing_fields(self):
        h = Hypothesis(
            name="test",
            mechanism="",
            boundary_conditions="",
            success_criteria="",
            entry_logic="",
            exit_logic="",
        )
        missing = h.validate()
        assert "mechanism" in missing
        assert "boundary_conditions" in missing
        assert "success_criteria" in missing

    def test_summary_contains_name(self):
        h = Hypothesis(
            name="momentum_v1",
            mechanism="Breakout momentum strategy",
            boundary_conditions="Range-bound markets",
            success_criteria="PF > 1.5",
            entry_logic="Close > HH_20",
            exit_logic="ATR trailing stop",
        )
        assert "momentum_v1" in h.summary()
        assert "Breakout" in h.summary()

    def test_timestamp_auto_set(self):
        h = Hypothesis(
            name="test",
            mechanism="test",
            boundary_conditions="test",
            success_criteria="test",
            entry_logic="test",
            exit_logic="test",
        )
        assert h.timestamp is not None
        assert h.timestamp.tzinfo is not None

    def test_to_dict_serializable(self):
        h = Hypothesis(
            name="test",
            mechanism="test mechanism",
            boundary_conditions="test conditions",
            success_criteria="test criteria",
            entry_logic="entry",
            exit_logic="exit",
        )
        d = h.to_dict()
        assert d["name"] == "test"
        assert d["mechanism"] == "test mechanism"
        assert "timestamp" in d

    def test_frozen_immutable(self):
        h = Hypothesis(
            name="test", mechanism="m", boundary_conditions="b",
            success_criteria="s", entry_logic="e", exit_logic="x",
        )
        with pytest.raises(AttributeError):
            h.name = "new_name"


class TestExperimentLog:
    def test_empty_log(self):
        log = ExperimentLog("test_hyp")
        assert log.n_experiments == 0
        assert log.n_bugfixes == 0
        assert log.n_ablations == 0
        assert log.adjusted_alpha() == 0.05

    def test_log_run_increases_experiments(self):
        log = ExperimentLog("test_hyp")
        log.log_run("first run", category="explore")
        assert log.n_experiments == 1
        assert log.adjusted_alpha() == 0.05 / 2

    def test_bugfix_not_counted(self):
        log = ExperimentLog("test_hyp")
        log.log_run("test", category="explore")
        log.log_modify("fixed bug", category="bugfix")
        assert log.n_experiments == 1
        assert log.n_bugfixes == 1

    def test_ablation_discounted(self):
        """Ablation should be counted at half weight (discounted).

        0.2.2 fix: was subtract (bug — ablations are disjoint from
        n_experiments, so subtracting undercounted tests and made
        adjusted_alpha too lenient). Now adds ablations at half weight,
        which is the documented "discount" intent.
        """
        log = ExperimentLog("test_hyp")
        log.log_run("ablation 1", category="ablation")
        log.log_run("explore 1", category="explore")
        # n_experiments = 1 (explore only — ablation is disjoint)
        # n_ablations = 1
        # n_tests = 1 + 0.5*1 = 1.5
        # adjusted = 0.05 / (1.5 + 1) = 0.02
        assert log.n_ablations == 1
        assert log.adjusted_alpha() == pytest.approx(0.05 / 2.5, rel=1e-6)

    def test_ablation_not_discounted(self):
        """When discount_ablations=False, only improve+explore count."""
        log = ExperimentLog("test_hyp")
        log.log_run("ablation 1", category="ablation")
        log.log_run("explore 1", category="explore")
        # n_tests = 1 (ablation excluded entirely)
        # adjusted = 0.05 / (1 + 1) = 0.025
        assert log.adjusted_alpha(discount_ablations=False) == pytest.approx(
            0.05 / 2, rel=1e-6,
        )

    def test_ablation_only_discounted(self):
        """Only ablations (no explore/improve): discount still applies.
        n_tests = 0 + 0.5*1 = 0.5, adjusted = 0.05/1.5 = 0.0333...
        But n_tests <= 0 is False (0.5 > 0), so we use 0.5/1.5.
        Wait — the code has `if n_tests <= 0: return initial_alpha`.
        For 0.5, 0.5 <= 0 is False, so it goes to the division.
        adjusted = 0.05 / (0.5 + 1) = 0.0333.
        """
        log = ExperimentLog("test_hyp")
        log.log_run("ablation 1", category="ablation")
        assert log.n_experiments == 0
        assert log.n_ablations == 1
        assert log.adjusted_alpha() == pytest.approx(0.05 / 1.5, rel=1e-6)

    def test_log_modify(self):
        log = ExperimentLog("test_hyp")
        entry = log.log_modify("changed params", category="improve")
        assert entry.type == "modify"
        assert entry.category == "improve"
        assert log.n_experiments == 1

    def test_summary_contains_key_info(self):
        log = ExperimentLog("test_hyp")
        log.log_run("first run", category="explore")
        s = log.summary()
        assert "test_hyp" in s
        assert "Experiments" in s
        assert "1" in s

    def test_multiple_entries(self):
        log = ExperimentLog("test_hyp")
        for i in range(5):
            log.log_run(f"run {i}", category="explore")
        assert log.n_experiments == 5
        assert len(log.entries) == 5

        adj = log.adjusted_alpha()
        expected = 0.05 / 6  # 5 experiments + 1
        assert adj == pytest.approx(expected, rel=1e-6)


class TestHoldoutSet:
    # C-2: seal() now requires a real data_hash. Tests that don't care
    # about hash content use a deterministic fake.
    _FAKE_HASH = "0" * 64

    def test_seal_and_verify(self):
        hs = HoldoutSet("test", "2025-01-01", "2025-12-31")
        assert not hs.is_sealed()
        hs.seal(data_hash=self._FAKE_HASH)
        assert hs.is_sealed()
        assert hs.verify()
        assert hs._seal.data_hash == self._FAKE_HASH

    def test_seal_requires_data_hash(self):
        """seal() must reject empty data_hash (C-2 fix)."""
        hs = HoldoutSet("test", "2025-01-01", "2025-12-31")
        # No-arg: TypeError (signature enforces required arg)
        with pytest.raises(TypeError):
            hs.seal()
        # Empty string: ValueError (signature accepts but validator rejects)
        with pytest.raises(ValueError):
            hs.seal(data_hash="")
        with pytest.raises(ValueError):
            hs.seal(data_hash=None)

    def test_commit_break_succeeds(self):
        hs = HoldoutSet("test", "2025-01-01", "2025-12-31")
        hs.seal(data_hash=self._FAKE_HASH)
        was_intact, _, hash_after = hs.commit_break(self._FAKE_HASH)
        assert was_intact
        assert hash_after == self._FAKE_HASH
        assert hs.is_broken()
        assert not hs.is_sealed()

    def test_cannot_break_twice(self):
        hs = HoldoutSet("test", "2025-01-01", "2025-12-31")
        hs.seal(data_hash=self._FAKE_HASH)
        was_intact_1, _, _ = hs.commit_break(self._FAKE_HASH)
        was_intact_2, _, _ = hs.commit_break(self._FAKE_HASH)
        assert was_intact_1
        assert not was_intact_2

    def test_cannot_reseal_after_break(self):
        hs = HoldoutSet("test", "2025-01-01", "2025-12-31")
        hs.seal(data_hash=self._FAKE_HASH)
        hs.commit_break(self._FAKE_HASH)
        with pytest.raises(RuntimeError):
            hs.seal(data_hash=self._FAKE_HASH)

    # --- Phase 3.3 D1: single atomic _save_seal() ---

    def test_commit_break_single_save(self):
        """Phase 3.3 D1: commit_break must call _save_seal() exactly once.

        Pre-fix, _save_seal() was called twice (once for new hash,
        once for broken_at). The intermediate state (new hash but
        broken_at=None on disk) was a race window. Post-fix: single
        atomic save with all fields set before the write.
        """
        import tempfile as _tempfile
        import json as _json

        with _tempfile.TemporaryDirectory() as tmp:
            seal_path = f"{tmp}/seal.json"
            hs = HoldoutSet("test", "2025-01-01", "2025-12-31", seal_path=seal_path)
            hs.seal(data_hash=self._FAKE_HASH)

            # Track _save_seal calls
            original_save = hs._save_seal
            save_count = [0]
            def counting_save():
                save_count[0] += 1
                original_save()
            hs._save_seal = counting_save

            hs.commit_break("new_hash_xyz")
            # Must be exactly 1 call (not 2)
            assert save_count[0] == 1, (
                f"commit_break should call _save_seal() exactly once, "
                f"got {save_count[0]} calls (Phase 3.3 D1 fix)"
            )

            # Verify file on disk has BOTH new hash AND broken_at
            with open(seal_path, "r") as f:
                saved = _json.load(f)
            assert saved["data_hash"] == "new_hash_xyz"
            assert saved["broken_at"] is not None

    def test_boundary(self):
        hs = HoldoutSet("test", "2025-01-01", "2025-12-31")
        assert hs.boundary() == ("2025-01-01", "2025-12-31")

    def test_summary(self):
        hs = HoldoutSet("test", "2025-01-01", "2025-12-31")
        hs.seal(data_hash=self._FAKE_HASH)
        s = hs.summary()
        assert "SEALED" in s
        assert "2025-01-01" in s

    def test_verify_fails_after_break(self):
        hs = HoldoutSet("test", "2025-01-01", "2025-12-31")
        hs.seal(data_hash=self._FAKE_HASH)
        hs.commit_break(self._FAKE_HASH)
        assert not hs.verify()


class TestHoldoutSeal:
    def test_round_trip_dict(self):
        seal = HoldoutSeal(
            start="2025-01-01", end="2025-12-31",
            sealed_at="2025-01-01T00:00:00",
        )
        d = seal.to_dict()
        seal2 = HoldoutSeal.from_dict(d)
        assert seal2.start == seal.start
        assert seal2.end == seal.end
        assert seal2.sealed_at == seal.sealed_at
