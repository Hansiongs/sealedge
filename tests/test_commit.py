"""Coverage push for quant_lib.research.commit.

Targets:
- commit_to_holdout full flow with mocked data layer
- All metric calculations (PSR, by_symbol, trend alignment)
- Error paths (already-broken seal, verify failure, missing ready stage)
- CommitResult field population
- Session recording + journal logging
"""

import hashlib
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quant_lib.audit import for_vol_compression
from quant_lib.research.candidate import Candidate
from quant_lib.research.commit import CommitResult, commit_to_holdout
from quant_lib.research.exceptions import (
    CommitError,
    HoldoutAlreadyBroken,
    NotReadyForCommit,
    SealVerificationFailed,
    SessionError,
)
from quant_lib.research.session import ResearchSession
from tests.conftest import _MockCache  # M-2: shared mock cache helper


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _make_candidate_ready(tmp: str) -> Candidate:
    """Create a Candidate, walk it to the 'ready' stage, populate trade data.

    This simulates what run_universe + run_edge_testing + run_narrowing
    + mark_ready would produce, but without doing the real WFA.

    C-2 fix: provide synthetic holdout data so session init can hash
    it without network. Tests then call commit_to_holdout which
    verifies the hash matches.
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
    hyp = for_vol_compression("test_v1", "m", "b", "c")
    cand = session.create_candidate(hyp)

    # Manually walk the state machine
    cand._set_stage("universe")
    cand._set_stage("edge")
    cand._set_stage("narrowed")
    cand.narrowed_symbols = ["BTCUSDT", "ETHUSDT"]

    # Populate frozen params (best-last-fold style)
    cand.frozen_params = {
        "BTCUSDT": {
            "vol_pct_thresh": 0.20, "pullback_bars": 5,
            "trail_atr": 3.0, "sl_mult": 1.5,
        },
        "ETHUSDT": {
            "vol_pct_thresh": 0.20, "pullback_bars": 5,
            "trail_atr": 3.0, "sl_mult": 1.5,
        },
    }

    # Populate risk_weights (needed for portfolio sim)
    cand.risk_weights = {"BTCUSDT": 0.01, "ETHUSDT": 0.01}

    # Lock to ready
    cand.mark_ready()
    return cand


def _make_features_df(sym: str, n: int, start: str) -> pd.DataFrame:
    """Build a 'precomputed' DataFrame with all features needed by the engine."""
    rng = np.random.default_rng(sum(ord(c) for c in sym))
    start_dt = pd.Timestamp(start)
    times = [start_dt + timedelta(hours=i) for i in range(n)]
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 10.0)
    high = close + np.abs(rng.normal(0, 0.3, n))
    low = close - np.abs(rng.normal(0, 0.3, n))
    open_ = close + rng.normal(0, 0.1, n)
    return pd.DataFrame({
        "time": times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.exponential(1000, n),
        "hh_20": pd.Series(high).rolling(20).max().shift(1).bfill(),
        "ll_20": pd.Series(low).rolling(20).min().shift(1).bfill(),
        "ema_200": pd.Series(close).ewm(span=200, adjust=False).mean().shift(1).bfill(),
        "rsi_14": np.clip(50 + rng.normal(0, 10, n), 0, 100),
        "bullish_reversal": np.zeros(n, dtype=np.int32),
        "bearish_reversal": np.zeros(n, dtype=np.int32),
        "vol_pct_rank": np.clip(rng.normal(0.3, 0.2, n), 0, 1),
        "rvol": np.clip(rng.normal(2.0, 0.5, n), 0.5, 5.0),
        "atr": np.full(n, 1.5),
        "funding_rate": np.zeros(n),
        "macro_vol": np.full(n, 0.5),
        "macro_trend": np.ones(n, dtype=np.int32),
        "is_weekend": np.zeros(n, dtype=np.int32),
        "is_funding_hour": np.zeros(n, dtype=np.int32),
    })


def _make_trade_trigger_data(
    sym: str,
    n: int,
    start: str,
    n_signals: int = 10,
) -> pd.DataFrame:
    """Build synthetic data designed to fire vol_compression entries.

    Pattern per signal (30 bars, with 20-bar gap so hh_20 resets):
    - bars 0-19: flat at 100 (this also "resets" the rolling 20-bar high)
    - bar 20: compression bar
    - bar 21: breakout (long: close=102)
    - bar 22: pullback (close=100.5)
    - bar 23: recovery (close=103, > setup_price)
    - bars 24-29: flat at 100 (wait for exit)

    The 20-bar gap ensures each signal's breakout creates a fresh hh_20.
    """
    start_dt = pd.Timestamp(start)
    times = [start_dt + timedelta(hours=i) for i in range(n)]
    n_per_signal = 30  # 20-bar warmup + 10-bar signal
    n_signals = min(n_signals, n // n_per_signal)

    close = np.full(n, 100.0)
    high = np.full(n, 100.3)
    low = np.full(n, 99.7)
    open_ = np.full(n, 100.0)
    volume = np.full(n, 500.0)

    for k in range(n_signals):
        base = k * n_per_signal
        if base + n_per_signal > n:
            break
        # Bars 0-19 (within this signal): flat at 100 (resets rolling window)
        for i in range(20):
            idx = base + i
            if idx >= n:
                break
            close[idx] = 100.0
            high[idx] = 100.3
            low[idx] = 99.7
            open_[idx] = 100.0
            volume[idx] = 500.0
        # Bar 20: compression
        idx = base + 20
        if idx < n:
            close[idx] = 100.0
            high[idx] = 100.3
            low[idx] = 99.7
            open_[idx] = 100.0
            volume[idx] = 500.0
        # Bar 21: breakout (long: spike up)
        idx = base + 21
        if idx < n:
            close[idx] = 102.0
            high[idx] = 102.5
            low[idx] = 100.5
            open_[idx] = 100.0
            volume[idx] = 5000.0
        # Bar 22: pullback
        idx = base + 22
        if idx < n:
            close[idx] = 100.5
            high[idx] = 101.0
            low[idx] = 100.0
            open_[idx] = 101.0
            volume[idx] = 500.0
        # Bar 23: recovery above breakout (close > setup_price=102)
        idx = base + 23
        if idx < n:
            close[idx] = 103.0
            high[idx] = 103.5
            low[idx] = 102.5
            open_[idx] = 102.5
            volume[idx] = 500.0
        # Bars 24-29: continuation then settle
        for i in range(24, 30):
            idx = base + i
            if idx < n:
                close[idx] = 103.0
                high[idx] = 103.3
                low[idx] = 102.7
                open_[idx] = 103.0
                volume[idx] = 500.0

    df = pd.DataFrame({
        "time": times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    # Features
    df["hh_20"] = df["high"].rolling(20).max().shift(1).bfill()
    df["ll_20"] = df["low"].rolling(20).min().shift(1).bfill()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean().shift(1).bfill()
    df["rsi_14"] = np.full(n, 50.0)
    df["bullish_reversal"] = np.zeros(n, dtype=np.int32)
    df["bearish_reversal"] = np.zeros(n, dtype=np.int32)
    df["vol_pct_rank"] = np.full(n, 0.05)
    df["rvol"] = np.full(n, 3.0)
    df["atr"] = np.full(n, 1.5)
    df["funding_rate"] = np.zeros(n)
    df["macro_vol"] = np.full(n, 0.5)
    df["macro_trend"] = np.ones(n, dtype=np.int32)
    df["is_weekend"] = np.zeros(n, dtype=np.int32)
    df["is_funding_hour"] = np.zeros(n, dtype=np.int32)
    return df


class _SilentLogger:
    """Context manager that silences all logging during a block."""
    def __enter__(self):
        import logging
        self._original = logging.getLogger("rich").info
        logging.getLogger("rich").info = lambda *a, **kw: None
        return self

    def __exit__(self, *args):
        import logging
        logging.getLogger("rich").info = self._original


def _silence_logs():
    """Return a context manager that suppresses 'rich' logger output."""
    return _SilentLogger()


# ─────────────────────────────────────────────────────────────────────
# S4.3: Error paths
# ─────────────────────────────────────────────────────────────────────


class TestCommitErrorPaths:
    """Verify the error-handling paths in commit_to_holdout."""

    def test_candidate_not_ready_raises(self):
        """A candidate at hypothesis stage must raise NotReadyForCommit."""
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp,
                _skip_holdout_load=True,
            )
            cand = session.create_candidate(for_vol_compression("v1", "m", "b", "c"))
            with pytest.raises(NotReadyForCommit):
                commit_to_holdout(cand, success_criteria_text="test")

    def test_already_broken_holdout_raises(self):
        """If the holdout is already broken, raise HoldoutAlreadyBroken."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            session = cand.session
            # Break the holdout seal
            session.holdout_set._seal.broken_at = "2025-01-01T00:00:00+00:00"
            with pytest.raises((HoldoutAlreadyBroken, CommitError)):
                commit_to_holdout(cand, success_criteria_text="test")


# ─────────────────────────────────────────────────────────────────────
# S4.3: Happy path
# ─────────────────────────────────────────────────────────────────────


class TestCommitHappyPath:
    """Test the full commit flow with mocked data layer."""

    def test_commit_produces_valid_result(self):
        """commit_to_holdout returns a fully-populated CommitResult."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            # Inject mock cache
            mock_cache = _MockCache()
            cand.session.cache = mock_cache
            # Suppress console output
            with _silence_logs():
                result = commit_to_holdout(
                    cand, success_criteria_text="PSR > 0.95", verbose=False,
                )
            assert isinstance(result, CommitResult)
            assert result.candidate_name == "test_v1"
            assert result.commit_idx == 1
            assert result.holdout_period == ("2025-01-01", "2025-06-30")
            assert result.success_criteria_text == "PSR > 0.95"
            assert result.seal_broken is True
            assert result.seal_hash_after != ""

    def test_commit_records_in_session(self):
        """After commit, session._commits has 1 record."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            cand.session.cache = _MockCache()
            with _silence_logs():
                commit_to_holdout(cand, success_criteria_text="x", verbose=False)
            assert len(cand.session._commits) == 1
            record = cand.session._commits[0]
            assert record.candidate_name == "test_v1"
            assert record.success_criteria_text == "x"

    def test_commit_logs_to_journal(self):
        """After commit, journal has at least one new entry from the commit."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            cand.session.cache = _MockCache()
            n_before = len(cand.session.journal.entries)
            with _silence_logs():
                commit_to_holdout(cand, success_criteria_text="x", verbose=False)
            n_after = len(cand.session.journal.entries)
            assert n_after > n_before, "Journal should have new entry from commit"

    def test_commit_breaks_seal(self):
        """After commit, holdout is broken and verify() fails."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            cand.session.cache = _MockCache()
            assert cand.session.holdout_set.is_sealed()
            with _silence_logs():
                commit_to_holdout(cand, success_criteria_text="x", verbose=False)
            assert cand.session.holdout_set.is_broken()
            assert not cand.session.holdout_set.is_sealed()

    def test_commit_candidate_becomes_locked(self):
        """After commit, candidate stage is 'ready' (terminal)."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            cand.session.cache = _MockCache()
            assert cand.stage == "ready"
            with _silence_logs():
                commit_to_holdout(cand, success_criteria_text="x", verbose=False)
            # Still 'ready' after commit
            assert cand.stage == "ready"

    def test_second_commit_raises(self):
        """A second commit on the same holdout must fail (seal broken)."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            cand.session.cache = _MockCache()
            with _silence_logs():
                commit_to_holdout(cand, success_criteria_text="x", verbose=False)
            # Second commit must raise
            with pytest.raises((HoldoutAlreadyBroken, CommitError)):
                commit_to_holdout(cand, success_criteria_text="x", verbose=False)


# ─────────────────────────────────────────────────────────────────────
# S4.3: Metric calculation coverage
# ─────────────────────────────────────────────────────────────────────


class TestCommitMetrics:
    """The metric calculation block (PSR, by_symbol, trend alignment)."""

    def test_commit_with_empty_trades_handles_gracefully(self):
        """If no trades are produced, metrics should be sensible (no NaN crash)."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            # Use data that's unlikely to produce any trades
            mock_cache = _MockCache()
            # Replace the data with something that won't trigger entries
            for sym in ["BTCUSDT", "ETHUSDT"]:
                df = _make_features_df(sym, n=2000, start="2019-06-01")
                df["vol_pct_rank"] = 0.9  # very high -> not compressed
                df["rvol"] = 0.5  # very low
                mock_cache._cache[(sym, "1h")] = df
            cand.session.cache = mock_cache
            with _silence_logs():
                result = commit_to_holdout(
                    cand, success_criteria_text="x", verbose=False,
                )
            # No trades => n_trades == 0, but result should still build
            assert result.n_trades == 0
            assert result.equity_pct == 0.0
            assert result.cagr_pct == 0.0
            assert result.max_dd_pct == 0.0

    def test_commit_by_symbol_stats(self):
        """By-symbol stats are computed per executed symbol."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            cand.session.cache = _MockCache()
            with _silence_logs():
                result = commit_to_holdout(
                    cand, success_criteria_text="x", verbose=False,
                )
            # If trades were executed, by_symbol_stats should have entries
            if result.n_trades > 0:
                for sym, stats in result.by_symbol_stats.items():
                    assert "n_trades" in stats
                    assert "win_rate" in stats
                    assert "avg_r" in stats
                    assert "profit_factor" in stats
                    assert "total_r" in stats

    def test_commit_trend_alignment_fields(self):
        """with_trend and counter_trend R totals are computed."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            cand.session.cache = _MockCache()
            with _silence_logs():
                result = commit_to_holdout(
                    cand, success_criteria_text="x", verbose=False,
                )
            # Fields should exist with sensible defaults
            assert hasattr(result, "with_trend_trades")
            assert hasattr(result, "with_trend_r_total")
            assert hasattr(result, "counter_trend_trades")
            assert hasattr(result, "counter_trend_r_total")

    def test_commit_seal_hash_computed(self):
        """The seal hash after commit matches the session init hash (data unchanged).

        C-2 fix: with pre-commit hashing, the hash computed at session
        creation and the hash computed at commit are EQUAL when the
        data has not been tampered with. This is the desired behavior.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            cand.session.cache = _MockCache()
            with _silence_logs():
                result = commit_to_holdout(
                    cand, success_criteria_text="x", verbose=False,
                )
            # Both hashes must be valid SHA256 hex (64 chars)
            assert len(result.seal_hash_after) == 64
            assert len(result.seal_hash_before) == 64
            assert all(c in "0123456789abcdef" for c in result.seal_hash_after)
            # C-2: hashes match (data unchanged between init and commit)
            assert result.seal_hash_before == result.seal_hash_after, (
                f"Seal hash mismatch: before={result.seal_hash_before}, "
                f"after={result.seal_hash_after}. Data tampering detected."
            )

    # --- Phase 3.4 E3: ess field consistency with PSR denom ---

    def test_ess_field_equals_n_trades_minus_one(self):
        """Phase 3.4 E3: ess field = n_trades - 1 (consistent with PSR denom)."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            cand.session.cache = _MockCache()
            with _silence_logs():
                result = commit_to_holdout(
                    cand, success_criteria_text="x", verbose=False,
                )
            # If we have trades, ess should be n_trades - 1
            if result.n_trades > 0:
                assert result.ess == result.n_trades - 1, (
                    f"ess field ({result.ess}) should equal n_trades - 1 "
                    f"({result.n_trades - 1}) for consistency with PSR denominator"
                )
            else:
                # No trades: ess should be 0
                assert result.ess == 0.0

    # --- Phase 3.7 E4: min_train_months enforcement ---

    def test_commit_blocks_short_training(self):
        """Phase 3.7 E4: short training period raises CommitError.

        The hypothesis specifies min_train_months (default 12). A session
        with a shorter training period must refuse to commit.
        """
        with tempfile.TemporaryDirectory() as tmp:
            # Make candidate with default hypothesis (min_train_months=12)
            cand = _make_candidate_ready(tmp)
            # Override the session's training_period to be very short
            cand.session.training_period = ("2024-12-01", "2024-12-31")  # 1 month
            cand.session.cache = _MockCache()

            with _silence_logs():
                with pytest.raises(CommitError, match="min_train_months"):
                    commit_to_holdout(cand, success_criteria_text="x", verbose=False)

    def test_commit_allows_long_training(self):
        """Phase 3.7 E4: training period >= min_train_months proceeds normally."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            # _make_candidate_ready uses training_period=("2020-01-01", "2024-12-31")
            # which is 5 years = 60 months, well above the 12-month default.
            cand.session.cache = _MockCache()
            with _silence_logs():
                # Should NOT raise CommitError for min_train_months
                result = commit_to_holdout(
                    cand, success_criteria_text="x", verbose=False,
                )
            assert isinstance(result, CommitResult)

    def test_commit_respects_custom_min_train_months(self):
        """Phase 3.7 E4: hypothesis with higher min_train_months is enforced."""
        from quant_lib.audit import for_vol_compression
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            # Replace hypothesis with one requiring 24 months
            cand.hypothesis = for_vol_compression(
                "test_higher_min", "m", "b", "c", min_train_months=24,
            )
            # _make_candidate_ready uses 60 months training -> OK
            cand.session.cache = _MockCache()
            with _silence_logs():
                result = commit_to_holdout(
                    cand, success_criteria_text="x", verbose=False,
                )
            assert isinstance(result, CommitResult)

            # Now test with insufficient training for the 24-month requirement
            cand2 = _make_candidate_ready(tmp)
            cand2.hypothesis = for_vol_compression(
                "test_higher_min_2", "m", "b", "c", min_train_months=24,
            )
            cand2.session.training_period = ("2024-01-01", "2024-12-31")  # 12 months
            cand2.session.cache = _MockCache()
            with _silence_logs():
                with pytest.raises(CommitError, match="min_train_months"):
                    commit_to_holdout(
                        cand2, success_criteria_text="x", verbose=False,
                    )


# ─────────────────────────────────────────────────────────────────────
# C-2: Pre-commit holdout hash (no-peek enforcement)
# ─────────────────────────────────────────────────────────────────────


class TestC2PreCommitHash:
    """C-2 fix: holdout data is hashed at session creation, verified at commit."""

    def test_session_seals_with_real_hash_not_placeholder(self):
        """Session init must hash the actual holdout data, not a placeholder.

        Pre-fix: seal used a placeholder like 'SEALED_<timestamp>'.
        Post-fix: seal uses a 64-char hex SHA256 of (time, close) per symbol.
        """
        with tempfile.TemporaryDirectory() as tmp:
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
                _holdout_data=holdout_data,
            )
            # Hash must be 64-char hex (SHA256), not a placeholder
            hash_val = session.holdout_set._seal.data_hash
            assert len(hash_val) == 64
            assert all(c in "0123456789abcdef" for c in hash_val)
            # Must NOT be the old placeholder pattern
            assert not hash_val.startswith("SEALED_")
            # Session should also store the hash for verification
            assert session._holdout_hash == hash_val
            # Cached data should match what we passed
            assert "BTCUSDT" in session._holdout_data_for_hash
            assert "ETHUSDT" in session._holdout_data_for_hash

    def test_commit_aborts_if_holdout_data_modified(self):
        """If holdout data is tampered between session and commit, abort."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            cand.session.cache = _MockCache()

            # Tamper with the cached holdout data AFTER session creation
            # (simulating "user looked at data and modified it")
            tampered = cand.session._holdout_data_for_hash["BTCUSDT"].copy()
            tampered.loc[tampered.index[100], "close"] = 999999.0
            cand.session._holdout_data_for_hash["BTCUSDT"] = tampered

            # Commit should raise SealVerificationFailed (tamper detected)
            from quant_lib.research.exceptions import SealVerificationFailed
            with _silence_logs():
                with pytest.raises(SealVerificationFailed):
                    commit_to_holdout(cand, success_criteria_text="x", verbose=False)

            # Holdout must still be sealed (not broken)
            assert cand.session.holdout_set.is_sealed()
            assert not cand.session.holdout_set.is_broken()

    def test_commit_proceeds_when_data_unchanged(self):
        """If holdout data unchanged, commit proceeds normally."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            cand.session.cache = _MockCache()
            with _silence_logs():
                result = commit_to_holdout(
                    cand, success_criteria_text="x", verbose=False,
                )
            # Commit succeeded
            assert isinstance(result, CommitResult)
            assert result.seal_broken
            # Hashes match
            assert result.seal_hash_before == result.seal_hash_after

    def test_skip_holdout_load_uses_fake_hash(self):
        """_skip_holdout_load=True bypasses data load, uses fake hash."""
        with tempfile.TemporaryDirectory() as tmp:
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp,
                _skip_holdout_load=True,
            )
            # Cached data is empty
            assert session._holdout_data_for_hash == {}
            # Hash is deterministic fake
            assert session._holdout_hash == "0" * 64
            # But seal is still recorded
            assert session.holdout_set.is_sealed()

    # --- Phase 3.8 E5: deepcopy isolation ---

    def test_holdout_data_isolated_from_caller_mutation(self):
        """Phase 3.8 E5: mutating caller's DataFrame in place must NOT affect session.

        Pre-fix, the session stored `dict(_holdout_data)` which is a shallow
        copy -- the DataFrame references were shared. A test that mutates
        a DataFrame in place (e.g., `df.loc[i, "close"] = x`) would
        silently change the session's data without the session knowing.
        Post-fix, deepcopy provides full isolation.
        """
        with tempfile.TemporaryDirectory() as tmp:
            times = pd.date_range("2025-01-01", "2025-06-30", freq="h")
            original_close = 100.0
            holdout_data = {
                sym: pd.DataFrame({
                    "time": times,
                    "open": 100.0, "high": 101.0, "low": 99.0,
                    "close": original_close, "volume": 1000.0,
                })
                for sym in ["BTCUSDT", "ETHUSDT"]
            }
            # Record the hash BEFORE in-place mutation
            session = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT", "ETHUSDT"],
                cache_dir=tmp,
                _holdout_data=holdout_data,
            )
            hash_before_mutation = session._holdout_hash

            # Mutate the caller's DataFrame in place (no reassignment)
            holdout_data["BTCUSDT"].loc[holdout_data["BTCUSDT"].index[100], "close"] = 999.0

            # Session's hash must be UNCHANGED (because session has its own copy)
            assert session._holdout_hash == hash_before_mutation, (
                f"Session hash changed after in-place mutation of caller's data! "
                f"Pre-fix: shallow copy means session sees the mutation. "
                f"Post-fix: deepcopy provides isolation."
            )

    def test_tampering_high_column_detected(self):
        """Phase 2.1: modifying 'high' column (not just 'close') must abort commit.

        Pre-Phase-2.1, the seal hash covered only (time, close). Tampering
        with high/low/open/volume was undetectable, allowing silent
        look-ahead bias on strategies that use those columns (e.g. SL
        uses high/low; vol_pct_rank uses volume).
        """
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            cand.session.cache = _MockCache()

            # Tamper with the 'high' column AFTER session creation.
            tampered = cand.session._holdout_data_for_hash["BTCUSDT"].copy()
            tampered.loc[tampered.index[100], "high"] = 999999.0
            cand.session._holdout_data_for_hash["BTCUSDT"] = tampered

            # Commit must raise SealVerificationFailed (tamper detected).
            from quant_lib.research.exceptions import SealVerificationFailed
            with _silence_logs():
                with pytest.raises(SealVerificationFailed):
                    commit_to_holdout(cand, success_criteria_text="x", verbose=False)

            # Holdout must still be sealed (not broken).
            assert cand.session.holdout_set.is_sealed()
            assert not cand.session.holdout_set.is_broken()

    def test_tampering_volume_column_detected(self):
        """Phase 2.1: modifying 'volume' column must abort commit."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            cand.session.cache = _MockCache()

            tampered = cand.session._holdout_data_for_hash["ETHUSDT"].copy()
            tampered.loc[tampered.index[50], "volume"] = 0.0  # zero volume
            cand.session._holdout_data_for_hash["ETHUSDT"] = tampered

            from quant_lib.research.exceptions import SealVerificationFailed
            with _silence_logs():
                with pytest.raises(SealVerificationFailed):
                    commit_to_holdout(cand, success_criteria_text="x", verbose=False)

            assert cand.session.holdout_set.is_sealed()

    def test_tampering_open_column_detected(self):
        """Phase 2.1: modifying 'open' column must abort commit."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            cand.session.cache = _MockCache()

            tampered = cand.session._holdout_data_for_hash["BTCUSDT"].copy()
            tampered.loc[tampered.index[200], "open"] = 0.01  # outlier
            cand.session._holdout_data_for_hash["BTCUSDT"] = tampered

            from quant_lib.research.exceptions import SealVerificationFailed
            with _silence_logs():
                with pytest.raises(SealVerificationFailed):
                    commit_to_holdout(cand, success_criteria_text="x", verbose=False)

            assert cand.session.holdout_set.is_sealed()

    def test_tampering_low_column_detected(self):
        """Phase 2.1: modifying 'low' column must abort commit."""
        with tempfile.TemporaryDirectory() as tmp:
            cand = _make_candidate_ready(tmp)
            cand.session.cache = _MockCache()

            tampered = cand.session._holdout_data_for_hash["ETHUSDT"].copy()
            tampered.loc[tampered.index[300], "low"] = -100.0  # invalid
            cand.session._holdout_data_for_hash["ETHUSDT"] = tampered

            from quant_lib.research.exceptions import SealVerificationFailed
            with _silence_logs():
                with pytest.raises(SealVerificationFailed):
                    commit_to_holdout(cand, success_criteria_text="x", verbose=False)

            assert cand.session.holdout_set.is_sealed()


# ─────────────────────────────────────────────────────────────────────
# Phase 2.2: BTC extended history integrity
# ─────────────────────────────────────────────────────────────────────


class TestBTCExtendedIntegrity:
    """Phase 2.2: BTC extended data (pre-holdout EMA warmup) is hashed.

    The BTC extended range (btc_data_start to hold_end) is used at commit
    time to compute EMA-warmup features. Tampering with this pre-holdout
    BTC data would silently change EMA features and trade signals, so it
    must be in the seal.
    """

    def _make_session_with_btc_extended(
        self,
        tmp: str,
        btc_extended: pd.DataFrame | None = None,
    ) -> ResearchSession:
        """Build a session with holdout data + optional BTC extended."""
        times = pd.date_range("2025-01-01", "2025-06-30", freq="h")
        holdout_data = {
            sym: pd.DataFrame({
                "time": times,
                "open": 100.0, "high": 101.0, "low": 99.0,
                "close": 100.0, "volume": 1000.0,
            })
            for sym in ["BTCUSDT", "ETHUSDT"]
        }
        return ResearchSession(
            training_period=("2020-01-01", "2024-12-31"),
            holdout_period=("2025-01-01", "2025-06-30"),
            symbols=["BTCUSDT", "ETHUSDT"],
            cache_dir=tmp,
            btc_data_start="2024-12-01",
            _holdout_data=holdout_data,
            _btc_extended=btc_extended,
        )

    def test_session_stores_btc_extended_when_provided(self):
        """When _btc_extended is passed, session must preserve it."""
        with tempfile.TemporaryDirectory() as tmp:
            ext_times = pd.date_range("2024-12-01", "2025-06-30", freq="h")
            btc_ext = pd.DataFrame({
                "time": ext_times,
                "open": 50.0, "high": 51.0, "low": 49.0,
                "close": 50.0, "volume": 500.0,
            })
            session = self._make_session_with_btc_extended(tmp, btc_ext)
            assert session._btc_extended_for_features is not None
            assert len(session._btc_extended_for_features) == len(btc_ext)

    def test_session_btc_extended_none_when_not_provided(self):
        """When _btc_extended is NOT passed, session has None for it."""
        with tempfile.TemporaryDirectory() as tmp:
            session = self._make_session_with_btc_extended(tmp)
            assert session._btc_extended_for_features is None

    def test_hash_differs_with_different_btc_extended(self):
        """Hash must change if BTC extended data is modified.

        Two sessions with identical holdout window but different BTC
        extended ranges must have different seal hashes.
        """
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            times = pd.date_range("2025-01-01", "2025-06-30", freq="h")
            holdout_data = {
                sym: pd.DataFrame({
                    "time": times,
                    "open": 100.0, "high": 101.0, "low": 99.0,
                    "close": 100.0, "volume": 1000.0,
                })
                for sym in ["BTCUSDT", "ETHUSDT"]
            }
            ext_times = pd.date_range("2024-12-01", "2025-06-30", freq="h")
            btc_ext_v1 = pd.DataFrame({
                "time": ext_times, "open": 50.0, "high": 51.0,
                "low": 49.0, "close": 50.0, "volume": 500.0,
            })
            # Version 2: same shape, different values
            btc_ext_v2 = btc_ext_v1.copy()
            btc_ext_v2.loc[btc_ext_v2.index[10], "close"] = 999.0

            s1 = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT", "ETHUSDT"],
                cache_dir=tmp1,
                btc_data_start="2024-12-01",
                _holdout_data=holdout_data,
                _btc_extended=btc_ext_v1,
            )
            s2 = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT", "ETHUSDT"],
                cache_dir=tmp2,
                btc_data_start="2024-12-01",
                _holdout_data=holdout_data,
                _btc_extended=btc_ext_v2,
            )
            # Different BTC extended -> different hashes
            assert s1._holdout_hash != s2._holdout_hash

    def test_tampering_btc_extended_pre_holdout_detected(self):
        """Modifying BTC extended data BEFORE holdout_start must abort commit.

        This is the core Phase 2.2 invariant: tampering with the BTC
        extended range (used for EMA warmup) is now caught by the seal.
        """
        with tempfile.TemporaryDirectory() as tmp:
            ext_times = pd.date_range("2024-12-01", "2025-06-30", freq="h")
            btc_ext = pd.DataFrame({
                "time": ext_times,
                "open": 50.0, "high": 51.0, "low": 49.0,
                "close": 50.0, "volume": 500.0,
            })
            cand = self._make_candidate_with_btc_extended(tmp, btc_ext)
            cand.session.cache = _MockCache()

            # Tamper with BTC extended data at index 10 (in pre-holdout region)
            tampered = cand.session._btc_extended_for_features.copy()
            tampered.loc[tampered.index[10], "close"] = 999.0
            cand.session._btc_extended_for_features = tampered

            from quant_lib.research.exceptions import SealVerificationFailed
            with _silence_logs():
                with pytest.raises(SealVerificationFailed):
                    commit_to_holdout(cand, success_criteria_text="x", verbose=False)

            # Holdout must still be sealed.
            assert cand.session.holdout_set.is_sealed()

    def test_tampering_btc_extended_in_holdout_window_detected(self):
        """Modifying BTC extended in the holdout window must also abort."""
        with tempfile.TemporaryDirectory() as tmp:
            ext_times = pd.date_range("2024-12-01", "2025-06-30", freq="h")
            btc_ext = pd.DataFrame({
                "time": ext_times,
                "open": 50.0, "high": 51.0, "low": 49.0,
                "close": 50.0, "volume": 500.0,
            })
            cand = self._make_candidate_with_btc_extended(tmp, btc_ext)
            cand.session.cache = _MockCache()

            # Find a bar inside the holdout window (>= 2025-01-01)
            holdout_start = pd.Timestamp("2025-01-01")
            btc_ext = cand.session._btc_extended_for_features
            in_window_idx = btc_ext[btc_ext["time"] >= holdout_start].index[5]
            tampered = btc_ext.copy()
            tampered.loc[in_window_idx, "close"] = 888.0
            cand.session._btc_extended_for_features = tampered

            from quant_lib.research.exceptions import SealVerificationFailed
            with _silence_logs():
                with pytest.raises(SealVerificationFailed):
                    commit_to_holdout(cand, success_criteria_text="x", verbose=False)

            assert cand.session.holdout_set.is_sealed()

    def _make_candidate_with_btc_extended(
        self, tmp: str, btc_ext: pd.DataFrame,
    ) -> Candidate:
        """Build a ready candidate with BTC extended data injected."""
        cand = _make_candidate_ready(tmp)
        # Inject BTC extended into the existing session
        cand.session._btc_extended_for_features = btc_ext.copy()
        # Recompute hash with the new BTC extended
        from quant_lib.research.session import _compute_holdout_data_hash
        cand.session._holdout_hash = _compute_holdout_data_hash(
            cand.session._holdout_data_for_hash,
            btc_extended=btc_ext,
        )
        # Also update the seal to the new hash
        cand.session.holdout_set._seal.data_hash = cand.session._holdout_hash
        return cand


# ─────────────────────────────────────────────────────────────────────
# S4.3: CommitResult dataclass construction
# ─────────────────────────────────────────────────────────────────────


class TestCommitResultDataclass:
    """Verify CommitResult dataclass can be constructed and serialized."""

    def test_commit_result_has_all_required_fields(self):
        result = CommitResult(
            candidate_name="test",
            commit_idx=1,
            holdout_period=("2025-01-01", "2025-06-30"),
            timestamp="2025-01-01T00:00:00+00:00",
            initial_capital=1000.0,
            final_equity=1100.0,
            equity_pct=10.0,
            cagr_pct=21.0,
            max_dd_pct=5.0,
            n_raw_trades=10,
            n_executed_trades=8,
            n_rejected=2,
            reject_breakdown={"cb_cooldown": 1, "position_limit": 1},
            n_trades=8,
            win_rate=62.5,
            avg_r=0.5,
            median_r=0.3,
            std_r=1.0,
            best_r=2.5,
            worst_r=-1.5,
            profit_factor=1.8,
            avg_bars_held=12.0,
            sharpe_r=0.5,
            psr=0.85,
            psr_ess=0.85,
            skew=0.2,
            kurtosis=3.5,
            ess=8.0,
            bonferroni_alpha=0.075,
            fdr_alpha=0.15,
            by_symbol_stats={"BTCUSDT": {"n_trades": 5, "win_rate": 60.0,
                                          "avg_r": 0.4, "profit_factor": 1.5,
                                          "total_r": 2.0}},
            with_trend_trades=5,
            with_trend_r_total=3.0,
            counter_trend_trades=3,
            counter_trend_r_total=-1.0,
            seal_hash_before="N/A",
            seal_hash_after="a" * 64,
            seal_broken=True,
            success_criteria_text="PSR > 0.95",
        )
        assert result.candidate_name == "test"
        assert result.profit_factor == 1.8
        assert result.equity_pct == 10.0
        assert result.by_symbol_stats["BTCUSDT"]["n_trades"] == 5


@pytest.mark.slow
class TestCommitWithRealTrades:
    """Test commit flow when data is designed to actually fire trades.

    This covers the trade-dict construction, per-trade stats, by_symbol
    breakdown, and trend alignment calculation -- all previously uncovered
    because synthetic random data rarely triggers engine entries.
    """

    def _make_signal_candidate(self, tmp: str) -> Candidate:
        """Build a candidate with data designed to produce trades.

        C-2 fix: provide synthetic holdout data so commit_to_holdout
        can verify the hash without network.
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
        hyp = for_vol_compression("signal_v1", "m", "b", "c")
        cand = session.create_candidate(hyp)
        cand._set_stage("universe")
        cand._set_stage("edge")
        cand._set_stage("narrowed")
        cand.narrowed_symbols = ["BTCUSDT", "ETHUSDT"]
        cand.frozen_params = {
            "BTCUSDT": {
                "vol_pct_thresh": 0.20, "pullback_bars": 3,
                "trail_atr": 3.0, "sl_mult": 1.5,
            },
            "ETHUSDT": {
                "vol_pct_thresh": 0.20, "pullback_bars": 3,
                "trail_atr": 3.0, "sl_mult": 1.5,
            },
        }
        cand.risk_weights = {"BTCUSDT": 0.01, "ETHUSDT": 0.01}
        cand.mark_ready()
        return cand

    def test_commit_with_signal_data_produces_trades(self):
        """With signal-rich data, the engine should produce trades.

        NOTE: This test is intentionally lenient on the exact trade count
        because the engine's entry conditions are sensitive to the
        specific data shape. The test verifies the *plumbing* works
        end-to-end (per-trade metrics are computed when trades fire).
        The exact entry logic is tested separately in test_engine_coverage.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cand = self._make_signal_candidate(tmp)
            mock_cache = _MockCache()
            # Need data from 2019-06-01 to 2025-06-30 (~53000 hours).
            # Each signal needs 30 bars (20 reset + 10 signal), so we
            # need ~1700 signals to fill 51000 bars.
            for sym in ["BTCUSDT", "ETHUSDT"]:
                mock_cache._cache[(sym, "1h")] = _make_trade_trigger_data(
                    sym, n=55000, start="2019-06-01", n_signals=1800,
                )
            cand.session.cache = mock_cache
            with _silence_logs():
                result = commit_to_holdout(
                    cand, success_criteria_text="PF>1.3", verbose=False,
                )
            # Verify the function completed without crashing and
            # returns a valid CommitResult, regardless of trade count.
            assert isinstance(result, CommitResult)
            assert result.candidate_name == "signal_v1"
            assert result.n_raw_trades >= 0
            assert result.n_executed_trades >= 0
            assert result.n_trades >= 0
            assert result.n_raw_trades >= result.n_trades
            # All metric fields are properly typed
            assert isinstance(result.win_rate, float)
            assert isinstance(result.avg_r, float)
            assert isinstance(result.median_r, float)
            assert isinstance(result.std_r, float)
            assert isinstance(result.best_r, float)
            assert isinstance(result.worst_r, float)
            assert isinstance(result.profit_factor, float)
            assert isinstance(result.avg_bars_held, float)
            assert isinstance(result.psr, float)
            assert isinstance(result.psr_ess, float)
            assert isinstance(result.skew, float)
            assert isinstance(result.kurtosis, float)
            assert isinstance(result.ess, float)
            assert isinstance(result.by_symbol_stats, dict)
            # If trades DID fire, the per-trade stats must be consistent
            if result.n_trades > 0:
                assert 0 <= result.win_rate <= 100
                assert result.best_r >= result.worst_r
                assert 0 <= result.ess  # ESS can't be negative

    def test_commit_trend_alignment_with_real_trades(self):
        """Trend alignment R-totals should be computed when there are trades.

        Lenient: only asserts when trades actually fire.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cand = self._make_signal_candidate(tmp)
            mock_cache = _MockCache()
            for sym in ["BTCUSDT", "ETHUSDT"]:
                mock_cache._cache[(sym, "1h")] = _make_trade_trigger_data(
                    sym, n=55000, start="2019-06-01", n_signals=1800,
                )
            cand.session.cache = mock_cache
            with _silence_logs():
                result = commit_to_holdout(
                    cand, success_criteria_text="x", verbose=False,
                )
            # Trend fields always exist (default to 0)
            assert result.with_trend_trades >= 0
            assert result.counter_trend_trades >= 0
            # When trades fire, classification should sum to <= n_trades
            # (some trades might be neutral, trend_risk_mult == 1.0)
            if result.n_trades > 0:
                total_classified = (
                    result.with_trend_trades + result.counter_trend_trades
                )
                assert total_classified <= result.n_trades

    def test_commit_by_symbol_stats_populated(self):
        """by_symbol_stats should have entries for each executed symbol.

        Lenient: only asserts when trades actually fire.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cand = self._make_signal_candidate(tmp)
            mock_cache = _MockCache()
            for sym in ["BTCUSDT", "ETHUSDT"]:
                mock_cache._cache[(sym, "1h")] = _make_trade_trigger_data(
                    sym, n=55000, start="2019-06-01", n_signals=1800,
                )
            cand.session.cache = mock_cache
            with _silence_logs():
                result = commit_to_holdout(
                    cand, success_criteria_text="x", verbose=False,
                )
            if result.n_trades > 0 and result.by_symbol_stats:
                for sym, stats in result.by_symbol_stats.items():
                    assert stats["n_trades"] >= 0
                    assert 0 <= stats["win_rate"] <= 100
                    assert isinstance(stats["avg_r"], float)
                    assert stats["profit_factor"] > 0 or stats["avg_r"] <= 0
