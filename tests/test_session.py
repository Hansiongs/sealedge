"""Tests for ResearchSession."""

import tempfile
import pytest
import pandas as pd

from quant_lib.audit import Hypothesis, for_vol_compression
from quant_lib.research.session import ResearchSession, _compute_holdout_data_hash
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


class TestResearchSessionSealDir:
    """Tests for the new ``seal_dir`` constructor parameter and the
    ``QUANT_LIB_SEAL_DIR`` environment variable fallback.

    Background: the seal directory used to be hardcoded to
    ``data_cache/holdout_seals`` regardless of the caller's
    ``cache_dir``. That broke for users running from a different
    working directory or with a non-default cache layout. The fix
    introduces explicit configuration (constructor arg + env var).
    """

    def test_default_seal_dir_derives_from_cache_dir(self, monkeypatch):
        """Without explicit config, ``seal_dir`` defaults to
        ``<cache_dir>/holdout_seals``."""
        # Clear the env var fallback so this test is deterministic.
        monkeypatch.delenv("QUANT_LIB_SEAL_DIR", raising=False)
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp,
                _skip_holdout_load=True,
            )
            # os.path.join is platform-aware; on Windows it uses "\".
            import os
            assert session.seal_dir == os.path.join(tmp, "holdout_seals")
            # And the directory is created on init.
            assert os.path.isdir(session.seal_dir)

    def test_explicit_seal_dir_overrides_default(self):
        """Passing ``seal_dir=`` overrides the default derivation."""
        import os
        with tempfile.TemporaryDirectory() as tmp:
            custom = os.path.join(tmp, "my_seals")
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp,
                seal_dir=custom,
                _skip_holdout_load=True,
            )
            assert session.seal_dir == custom
            assert os.path.isdir(custom)

    def test_env_var_overrides_default(self, monkeypatch):
        """``QUANT_LIB_SEAL_DIR`` env var overrides the default but
        is itself overridden by an explicit ``seal_dir=`` argument.
        """
        import os
        with tempfile.TemporaryDirectory() as tmp:
            env_path = os.path.join(tmp, "env_seals")
            monkeypatch.setenv("QUANT_LIB_SEAL_DIR", env_path)
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp,
                _skip_holdout_load=True,
            )
            assert session.seal_dir == env_path

    def test_explicit_seal_dir_beats_env_var(self, monkeypatch):
        """Explicit ``seal_dir=`` argument takes precedence over
        the env var. This is important for tests that want to
        isolate even when the env var is set globally.
        """
        import os
        with tempfile.TemporaryDirectory() as tmp:
            env_path = os.path.join(tmp, "env_seals")
            explicit = os.path.join(tmp, "explicit_seals")
            monkeypatch.setenv("QUANT_LIB_SEAL_DIR", env_path)
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp,
                seal_dir=explicit,
                _skip_holdout_load=True,
            )
            assert session.seal_dir == explicit

    def test_seal_file_lands_in_seal_dir(self, monkeypatch):
        """The HMAC seal JSON is persisted to ``seal_dir``, not to
        a hardcoded path. This is the regression test for the
        original smell.
        """
        import json
        from quant_lib.audit.holdout import verify_seal_signature
        import os
        monkeypatch.delenv("QUANT_LIB_SEAL_DIR", raising=False)
        with tempfile.TemporaryDirectory() as tmp:
            custom = os.path.join(tmp, "custom_seals")
            # The session is constructed for its side effect of writing
            # the seal file; we only need to verify the file lands in
            # ``custom``, not the session object itself.
            _session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp,
                seal_dir=custom,
                _skip_holdout_load=True,
            )
            expected_file = os.path.join(
                custom, "holdout_2025-01-01_2025-06-30.json"
            )
            assert os.path.isfile(expected_file)
            # And the file is properly HMAC-signed.
            with open(expected_file, "r") as f:
                state = json.load(f)
            assert verify_seal_signature(state)


# ════════════════════════════════════════════════════════════════════════
# Phase 2.3: Funding data hash (BC break)
# ════════════════════════════════════════════════════════════════════════


class TestFundingDataHash:
    """Phase 2.3: funding rate data is now part of the holdout seal hash.

    This is a BC break: existing seals (pre-2.3) have hashes computed
    without funding. Users with existing seals must re-create sessions.
    """

    def test_funding_changes_hash(self):
        """Adding funding data must change the hash."""
        ohlcv = {
            "BTCUSDT": pd.DataFrame({
                "time": [1, 2, 3],
                "close": [100.0, 101.0, 102.0],
                "volume": [10.0, 20.0, 30.0],
            })
        }
        funding_a = {
            "BTCUSDT": pd.DataFrame({
                "time": [1, 2, 3],
                "funding_rate": [0.0001, 0.0002, 0.0003],
            })
        }
        funding_b = {
            "BTCUSDT": pd.DataFrame({
                "time": [1, 2, 3],
                "funding_rate": [0.0002, 0.0003, 0.0004],  # different
            })
        }
        hash_no = _compute_holdout_data_hash(ohlcv)
        hash_a = _compute_holdout_data_hash(ohlcv, funding_data=funding_a)
        hash_b = _compute_holdout_data_hash(ohlcv, funding_data=funding_b)
        assert hash_no != hash_a, "Funding must change hash"
        assert hash_a != hash_b, "Different funding must change hash"

    def test_funding_none_preserves_backward_compat(self):
        """When funding_data=None, hash matches the pre-2.3 behavior."""
        ohlcv = {
            "BTCUSDT": pd.DataFrame({
                "time": [1, 2, 3],
                "close": [100.0, 101.0, 102.0],
            })
        }
        # No funding arg
        hash1 = _compute_holdout_data_hash(ohlcv)
        # Explicit None
        hash2 = _compute_holdout_data_hash(ohlcv, funding_data=None)
        assert hash1 == hash2, "None funding should match no funding"

    def test_funding_empty_dict_treated_as_none(self):
        """Empty funding dict is treated as no funding."""
        ohlcv = {
            "BTCUSDT": pd.DataFrame({
                "time": [1, 2, 3],
                "close": [100.0, 101.0, 102.0],
            })
        }
        hash_none = _compute_holdout_data_hash(ohlcv)
        hash_empty = _compute_holdout_data_hash(ohlcv, funding_data={})
        assert hash_none == hash_empty

    def test_funding_columns_filtered(self):
        """Only time and funding_rate columns are hashed (others ignored)."""
        ohlcv = {
            "BTCUSDT": pd.DataFrame({
                "time": [1, 2, 3],
                "close": [100.0, 101.0, 102.0],
            })
        }
        # Funding with extra columns that should be ignored
        funding_extra = {
            "BTCUSDT": pd.DataFrame({
                "time": [1, 2, 3],
                "funding_rate": [0.0001, 0.0002, 0.0003],
                "extra_ignored": [99, 98, 97],
            })
        }
        funding_clean = {
            "BTCUSDT": pd.DataFrame({
                "time": [1, 2, 3],
                "funding_rate": [0.0001, 0.0002, 0.0003],
            })
        }
        hash_extra = _compute_holdout_data_hash(ohlcv, funding_data=funding_extra)
        hash_clean = _compute_holdout_data_hash(ohlcv, funding_data=funding_clean)
        assert hash_extra == hash_clean, "Non-canonical columns should not affect hash"

    def test_funding_per_symbol(self):
        """Different funding per symbol gives different hash."""
        ohlcv = {
            "BTCUSDT": pd.DataFrame({
                "time": [1, 2, 3], "close": [100.0, 101.0, 102.0]
            }),
            "ETHUSDT": pd.DataFrame({
                "time": [1, 2, 3], "close": [50.0, 51.0, 52.0]
            }),
        }
        funding_btc = {
            "BTCUSDT": pd.DataFrame({"time": [1, 2, 3], "funding_rate": [0.0001, 0.0002, 0.0003]}),
            "ETHUSDT": pd.DataFrame({"time": [1, 2, 3], "funding_rate": [0.0004, 0.0005, 0.0006]}),
        }
        # Swap funding between symbols
        funding_swapped = {
            "BTCUSDT": pd.DataFrame({"time": [1, 2, 3], "funding_rate": [0.0004, 0.0005, 0.0006]}),
            "ETHUSDT": pd.DataFrame({"time": [1, 2, 3], "funding_rate": [0.0001, 0.0002, 0.0003]}),
        }
        h1 = _compute_holdout_data_hash(ohlcv, funding_data=funding_btc)
        h2 = _compute_holdout_data_hash(ohlcv, funding_data=funding_swapped)
        assert h1 != h2, "Per-symbol funding must affect hash"

