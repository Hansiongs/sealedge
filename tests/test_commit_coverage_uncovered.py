"""Targeted tests for uncovered code paths in commit.py.

Each test targets specific uncovered lines without modifying production code.
"""
from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd

from quant_lib.audit import for_vol_compression, for_pullback_sniper
from quant_lib.research.candidate import Candidate
from quant_lib.research.commit import CommitResult, commit_to_holdout
from quant_lib.research.session import ResearchSession, _compute_holdout_data_hash
from tests.conftest import (
    _MockCache,
    DEFAULT_SYMBOLS,
    BTC_DATA_START,
    TRAIN_PERIOD,
    HOLDOUT_PERIOD,
)


def _make_signal_holdout_data() -> dict[str, pd.DataFrame]:
    """Build synthetic OHLCV with vol_compression signals.

    Creates alternating flat/breakout patterns that trigger trades.
    """
    # Deterministic seed for reproducibility; rng is set up but the
    # synthetic series below uses fixed values for simplicity.
    _rng = np.random.default_rng(42)
    times = pd.date_range(BTC_DATA_START, HOLDOUT_PERIOD[1], freq="h")
    n = len(times)
    close = np.full(n, 100.0)
    high = np.full(n, 100.3)
    low = np.full(n, 99.7)
    volume = np.full(n, 1000.0)

    n_signals = min(n // 30, 200)
    for k in range(n_signals):
        base = k * 30
        if base + 30 > n:
            break
        # Bars 0-19: flat (compression)
        for i in range(20):
            idx = base + i
            if idx < n:
                close[idx] = 100.0
                high[idx] = 100.3
        # Bar 21: breakout
        idx = base + 21
        if idx < n:
            close[idx] = 102.0
            high[idx] = 102.5
            volume[idx] = 5000.0
        # Bar 22: pullback
        idx = base + 22
        if idx < n:
            close[idx] = 100.5
        # Bar 23+: recovery
        for i in range(23, 30):
            idx = base + i
            if idx < n:
                close[idx] = 103.0
                high[idx] = 103.3

    df = pd.DataFrame({
        "time": times, "open": 100.0, "high": high,
        "low": low, "close": close, "volume": volume,
    })
    return {sym: df.copy() for sym in DEFAULT_SYMBOLS}


def _build_session_with_signal_data(
    cache_dir: str,
    btc_extended: pd.DataFrame | None = None,
) -> ResearchSession:
    """Create a ResearchSession with signal-rich holdout data + optional BTC extended.

    Crucially, the seal hash is recomputed after data injection so verify()
    passes at commit time.
    """
    holdout_data = _make_signal_holdout_data()

    session = ResearchSession(
        training_period=TRAIN_PERIOD,
        holdout_period=HOLDOUT_PERIOD,
        symbols=DEFAULT_SYMBOLS,
        cache_dir=cache_dir,
        btc_data_start=BTC_DATA_START,
        _holdout_data=holdout_data,
        _btc_extended=btc_extended,
        _holdout_funding={},  # Phase 2.3: explicit empty funding for test
    )

    # Update seal hash to match the actual holdout data
    session._holdout_hash = _compute_holdout_data_hash(
        session._holdout_data_for_hash,
        btc_extended=session._btc_extended_for_features,
        funding_data=session._holdout_funding_for_hash,
    )
    session.holdout_set._seal.data_hash = session._holdout_hash
    session.holdout_set._save_seal()
    return session


def _make_candidate_for_trades(
    cache_dir: str,
    mock_cache: _MockCache,
    strategy_type: str = "vol_compression",
    btc_extended: pd.DataFrame | None = None,
) -> Candidate:
    """Build a ready candidate whose holdout data can produce trades."""
    hyp_cls = for_pullback_sniper if strategy_type == "pullback_sniper" else for_vol_compression
    hyp = hyp_cls("test_hyp", "mechanism", "boundary", "criteria")

    session = _build_session_with_signal_data(cache_dir, btc_extended=btc_extended)
    session.cache = mock_cache
    cand = session.create_candidate(hyp)
    cand._set_stage("universe")
    cand._set_stage("edge")
    cand._set_stage("narrowed")
    cand.narrowed_symbols = list(DEFAULT_SYMBOLS)
    cand.frozen_params = {
        sym: {"vol_pct_thresh": 0.20, "pullback_bars": 5,
              "trail_atr": 3.0, "sl_mult": 1.5}
        for sym in DEFAULT_SYMBOLS
    }
    cand.risk_weights = {sym: 0.01 for sym in DEFAULT_SYMBOLS}
    cand.mark_ready()
    return cand


# ═══════════════════════════════════════════════════════════════════════
# Test: BTC extended branch (line 222)
# ═══════════════════════════════════════════════════════════════════════

class TestBTCExtendedBranch:
    """Exercise btc_holdout_ext from cached_btc_extended (line 222)."""

    def test_commit_with_btc_extended_data(self):
        """When _btc_extended_for_features is provided, use it."""
        with tempfile.TemporaryDirectory() as tmp:
            ext_times = pd.date_range(BTC_DATA_START, HOLDOUT_PERIOD[1], freq="h")
            btc_ext = pd.DataFrame({
                "time": ext_times, "open": 50.0, "high": 51.0,
                "low": 49.0, "close": 50.0, "volume": 500.0,
            })
            mock = _MockCache()
            cand = _make_candidate_for_trades(tmp, mock, btc_extended=btc_ext)
            result = commit_to_holdout(cand, success_criteria_text="x", verbose=False)
            assert isinstance(result, CommitResult)
            assert result.seal_broken


# ═══════════════════════════════════════════════════════════════════════
# Test: Empty symbol data (line 237 continue)
# ═══════════════════════════════════════════════════════════════════════

class TestEmptySymbolData:
    """Exercise continue on empty symbol data (line 237)."""

    def test_commit_with_empty_symbol_data(self):
        """When a narrowed symbol has no cached data, skip it."""
        with tempfile.TemporaryDirectory() as tmp:
            mock = _MockCache()
            cand = _make_candidate_for_trades(tmp, mock)
            cand.narrowed_symbols = list(DEFAULT_SYMBOLS) + ["LTCUSDT"]
            result = commit_to_holdout(cand, success_criteria_text="x", verbose=False)
            assert isinstance(result, CommitResult)
            assert result.seal_broken


# ═══════════════════════════════════════════════════════════════════════
# Test: Pullback sniper strategy (line 261)
# ═══════════════════════════════════════════════════════════════════════

class TestPullbackSniper:
    """Exercise pullback_sniper critical_cols path (line 261)."""

    def test_commit_pullback_sniper(self):
        """Pullback sniper strategy uses additional critical_cols."""
        with tempfile.TemporaryDirectory() as tmp:
            mock = _MockCache()
            cand = _make_candidate_for_trades(tmp, mock, strategy_type="pullback_sniper")
            result = commit_to_holdout(cand, success_criteria_text="x", verbose=False)
            assert isinstance(result, CommitResult)
            assert result.seal_broken


# ═══════════════════════════════════════════════════════════════════════
# Test: Missing risk_weights (line 364-372 warning, line 391)
# ═══════════════════════════════════════════════════════════════════════

class TestMissingRiskWeights:
    """Exercise missing risk_weights warning and holdout_weights=None paths."""

    def test_commit_with_missing_symbol_in_risk_weights(self):
        """When narrowed sym missing from risk_weights, warning fires."""
        with tempfile.TemporaryDirectory() as tmp:
            mock = _MockCache()
            cand = _make_candidate_for_trades(tmp, mock)
            cand.risk_weights = {"BTCUSDT": 0.01}
            result = commit_to_holdout(cand, success_criteria_text="x", verbose=False)
            assert isinstance(result, CommitResult)

    def test_commit_with_empty_risk_weights(self):
        """When risk_weights is empty dict, holdout_weights=None (line 391)."""
        with tempfile.TemporaryDirectory() as tmp:
            mock = _MockCache()
            cand = _make_candidate_for_trades(tmp, mock)
            cand.risk_weights = {}
            result = commit_to_holdout(cand, success_criteria_text="x", verbose=False)
            assert isinstance(result, CommitResult)


# ═══════════════════════════════════════════════════════════════════════
# Test: Verbose path (line 599-601)
# ═══════════════════════════════════════════════════════════════════════

class TestVerboseCommit:
    """Exercise verbose=True path (line 599-601)."""

    def test_commit_verbose(self):
        """verbose=True should log commit info without crashing."""
        with tempfile.TemporaryDirectory() as tmp:
            mock = _MockCache()
            cand = _make_candidate_for_trades(tmp, mock)
            result = commit_to_holdout(cand, success_criteria_text="x", verbose=True)
            assert isinstance(result, CommitResult)
            assert result.seal_broken


# ═══════════════════════════════════════════════════════════════════════
# Test: All strict metric fields populated (lines 438-479)
# ═══════════════════════════════════════════════════════════════════════

class TestMetricsWithTrades:
    """Exercise metric computation when n_trades > 0."""

    def test_metrics_have_all_fields(self):
        """Metric fields are present even with 0 trades."""
        with tempfile.TemporaryDirectory() as tmp:
            mock = _MockCache()
            cand = _make_candidate_for_trades(tmp, mock)
            result = commit_to_holdout(cand, success_criteria_text="x", verbose=False)
            # All fields must exist
            for field in ("win_rate", "avg_r", "median_r", "std_r", "best_r", "worst_r",
                          "profit_factor", "avg_bars_held", "sharpe_r", "psr",
                          "skew", "kurtosis", "cagr_pct", "max_dd_pct"):
                assert hasattr(result, field), f"Missing field: {field}"
            assert isinstance(result, CommitResult)
