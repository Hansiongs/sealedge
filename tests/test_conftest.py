"""Smoke tests for the conftest.py fixtures and helpers.

These tests verify that the shared fixtures in ``tests/conftest.py``
can be instantiated without crashing, and that the helper functions
return well-defined structures.  Acts as a fast smoke check that the
test infrastructure is healthy.
"""


import numpy as np
import pandas as pd

from tests.conftest import (
    BTC_SEED,
    DAILY_CLOSE_SEED,
    DEFAULT_N_BARS_BTC,
    DEFAULT_N_BARS_HOURLY,
    DEFAULT_N_BARS_TRADE,
    DEFAULT_SYMBOLS,
    FUNDING_SEED,
    GLOBAL_SEED,
    HOLDOUT_PERIOD,
    HOLDOUT_PERIOD_ALT,
    HOLDOUT_PERIOD_FAR,
    HOURLY_SEED,
    TRAIN_PERIOD,
    TRADES_SEED,
    _MockCache,
    common_engine_extra,
    make_candidate_ready,
    make_engine_arrays,
    make_session_candidate,
    make_synthetic_holdout_data,
    patch_statics,
    walk_to_narrowed,
)


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════


class TestSharedConstants:
    """Project-wide date / seed / count constants are well-defined."""

    def test_train_period_is_valid(self):
        train_start, train_end = TRAIN_PERIOD
        assert pd.Timestamp(train_start) < pd.Timestamp(train_end)

    def test_holdout_after_train(self):
        train_end = pd.Timestamp(TRAIN_PERIOD[1])
        hold_start = pd.Timestamp(HOLDOUT_PERIOD[0])
        assert hold_start > train_end, "Holdout must be after train end"

    def test_holdout_variants_after_train(self):
        train_end = pd.Timestamp(TRAIN_PERIOD[1])
        for period in (HOLDOUT_PERIOD, HOLDOUT_PERIOD_ALT, HOLDOUT_PERIOD_FAR):
            start = pd.Timestamp(period[0])
            assert start > train_end

    def test_default_symbols_is_non_empty(self):
        assert len(DEFAULT_SYMBOLS) > 0
        assert all(isinstance(s, str) for s in DEFAULT_SYMBOLS)

    def test_seeds_are_integers(self):
        for seed in (GLOBAL_SEED, HOURLY_SEED, BTC_SEED, FUNDING_SEED,
                     TRADES_SEED, DAILY_CLOSE_SEED):
            assert isinstance(seed, int)

    def test_n_bars_are_positive(self):
        for n in (DEFAULT_N_BARS_HOURLY, DEFAULT_N_BARS_BTC,
                  DEFAULT_N_BARS_TRADE):
            assert n > 0


# ═══════════════════════════════════════════════════════════════════════
# Engine array factory
# ═══════════════════════════════════════════════════════════════════════


class TestMakeEngineArrays:
    """``make_engine_arrays`` returns a complete dict for the engine."""

    def test_basic_invocation(self):
        arrays = make_engine_arrays()
        expected_keys = {
            "opens", "highs", "lows", "closes",
            "hh_20", "ll_20", "ema_200s",
            "rsi_14", "bullish_reversal", "bearish_reversal",
            "vol_pct_rank", "rvol", "atrs",
            "funding_rates", "macro_vols", "macro_trends",
            "is_weekends", "is_funding_hours",
        }
        assert set(arrays.keys()) == expected_keys

    def test_arrays_have_correct_length(self):
        n = 500
        arrays = make_engine_arrays(n=n)
        for name, arr in arrays.items():
            assert len(arr) == n, f"{name} has length {len(arr)} != {n}"

    def test_overrides_take_effect(self):
        n = 100
        closes = np.full(n, 99.0)
        arrays = make_engine_arrays(n=n, closes=closes)
        assert np.all(arrays["closes"] == 99.0)

    def test_deterministic_with_same_seed(self):
        a1 = make_engine_arrays(seed=42, n=100)
        a2 = make_engine_arrays(seed=42, n=100)
        for key in a1:
            assert np.array_equal(a1[key], a2[key]), f"Mismatch on {key}"


class TestCommonEngineExtra:
    """``common_engine_extra`` returns a strategy dict."""

    def test_returns_required_keys(self):
        arrays = make_engine_arrays(n=100)
        result = common_engine_extra(arrays)
        assert "strategy_type" in result
        assert "allow_long" in result
        assert "allow_short" in result
        assert "rsi_oversold" in result
        assert "rsi_overbought" in result
        assert "trend_aligned_mult" in result
        assert "trend_counter_mult" in result


# ═══════════════════════════════════════════════════════════════════════
# Holdout data factory
# ═══════════════════════════════════════════════════════════════════════


class TestMakeSyntheticHoldoutData:
    """``make_synthetic_holdout_data`` returns a per-symbol DataFrame dict."""

    def test_default_symbols(self):
        data = make_synthetic_holdout_data()
        assert set(data.keys()) == set(DEFAULT_SYMBOLS)

    def test_custom_symbols(self):
        data = make_synthetic_holdout_data(symbols=["BTCUSDT"])
        assert set(data.keys()) == {"BTCUSDT"}

    def test_each_dataframe_has_required_columns(self):
        data = make_synthetic_holdout_data()
        for sym, df in data.items():
            for col in ("time", "open", "high", "low", "close", "volume"):
                assert col in df.columns, f"{sym} missing {col}"

    def test_custom_date_range(self):
        data = make_synthetic_holdout_data(
            start="2024-01-01", end="2024-01-02",
        )
        for df in data.values():
            assert df["time"].min() >= pd.Timestamp("2024-01-01")
            assert df["time"].max() <= pd.Timestamp("2024-01-02")


# ═══════════════════════════════════════════════════════════════════════
# Session / candidate factories
# ═══════════════════════════════════════════════════════════════════════


class TestMakeSessionCandidate:
    """``make_session_candidate`` returns a session + candidate."""

    def test_returns_session_and_candidate(self, tmp_path):
        mock = _MockCache()
        session, cand = make_session_candidate(
            tmp_path, mock, name="smoke_v1",
        )
        assert session is not None
        assert cand is not None
        assert cand.stage == "hypothesis"

    def test_candidate_attached_to_session(self, tmp_path):
        mock = _MockCache()
        session, cand = make_session_candidate(tmp_path, mock)
        assert cand.session is session


class TestMakeCandidateReady:
    """``make_candidate_ready`` returns a ready candidate."""

    def test_returns_ready_candidate(self, tmp_path):
        cand = make_candidate_ready(tmp_path, name="ready_v1")
        assert cand.stage == "ready"
        assert cand.is_ready_for_commit is True
        assert cand.narrowed_symbols  # non-empty
        assert cand.frozen_params  # non-empty

    def test_risk_weights_populated(self, tmp_path):
        cand = make_candidate_ready(tmp_path)
        for sym in cand.narrowed_symbols:
            assert sym in cand.risk_weights


# ═══════════════════════════════════════════════════════════════════════
# _MockCache
# ═══════════════════════════════════════════════════════════════════════


class TestMockCache:
    """``_MockCache`` produces signal-rich data on demand."""

    def test_get_klines_returns_dataframe(self):
        cache = _MockCache()
        df = cache.get_klines("BTCUSDT", "1h", None, None)
        assert isinstance(df, pd.DataFrame)
        assert "time" in df.columns
        assert "close" in df.columns

    def test_get_klines_respects_date_filter(self):
        cache = _MockCache()
        start = pd.Timestamp("2020-01-01")
        end = pd.Timestamp("2020-01-31")
        df = cache.get_klines("BTCUSDT", "1h", start, end)
        assert df["time"].min() >= start
        assert df["time"].max() <= end

    def test_invalidate_specific_symbol(self):
        cache = _MockCache(n_bars=200)
        cache.get_klines("BTCUSDT", "1h", None, None)  # populate
        cache.invalidate(symbol="BTCUSDT")
        # Re-fetching should re-create the data
        df = cache.get_klines("BTCUSDT", "1h", None, None)
        assert len(df) > 0

    def test_invalidate_all(self):
        cache = _MockCache(n_bars=200)
        cache.get_klines("BTCUSDT", "1h", None, None)
        cache.invalidate()
        # After invalidation, stats should be reset
        # (the next call is a miss)
        before_misses = cache._misses
        cache.get_klines("ETHUSDT", "1h", None, None)
        assert cache._misses == before_misses + 1

    def test_get_funding_returns_none(self):
        """``get_funding`` always returns None (mock doesn't model
        funding rates; production code is expected to fall back to
        a funding-less path).
        """
        cache = _MockCache()
        result = cache.get_funding("BTCUSDT", None, None)
        assert result is None

    def test_get_funding_args_accepted(self):
        """``get_funding`` accepts (symbol, start, end) and returns None."""
        cache = _MockCache()
        result = cache.get_funding(
            "BTCUSDT", "2024-01-01", "2024-01-31",
        )
        assert result is None

    def test_get_funding_does_not_touch_state(self):
        """``get_funding`` does not modify cache state."""
        cache = _MockCache()
        before = cache._misses
        cache.get_funding("BTCUSDT", None, None)
        # No miss recorded (funding has no cache lookup)
        assert cache._misses == before

    def test_data_lookup_overrides(self):
        """``data_lookup`` parameter replaces per-symbol data."""
        custom_df = pd.DataFrame({
            "time": pd.date_range("2020-01-01", periods=10, freq="h"),
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.0, "volume": 1000.0,
        })
        cache = _MockCache(data_lookup={"BTCUSDT": custom_df})
        df = cache.get_klines("BTCUSDT", "1h", None, None)
        assert len(df) == 10


# ═══════════════════════════════════════════════════════════════════════
# patch_statics context manager
# ═══════════════════════════════════════════════════════════════════════


class TestPatchStatics:
    """``patch_statics`` overrides STATIC values for the duration of a block."""

    def test_override_and_restore(self):
        from quant_lib.core._config import STATIC
        original = STATIC.get("spa_equity_warn_threshold_usd", 0.0)
        with patch_statics(spa_equity_warn_threshold_usd=999.0):
            assert STATIC["spa_equity_warn_threshold_usd"] == 999.0
        # Restored
        assert STATIC.get("spa_equity_warn_threshold_usd") == original

    def test_new_key_removed_on_exit(self):
        from quant_lib.core._config import STATIC
        assert "test_patch_key" not in STATIC
        with patch_statics(test_patch_key=42):
            assert STATIC["test_patch_key"] == 42
        # Removed (not in original)
        assert "test_patch_key" not in STATIC


# ═══════════════════════════════════════════════════════════════════════
# walk_to_narrowed
# ═══════════════════════════════════════════════════════════════════════


class TestWalkToNarrowed:
    """``walk_to_narrowed`` drives a candidate to the narrowed stage."""

    def test_walks_to_narrowed(self, tmp_path):
        mock = _MockCache()
        _, cand = make_session_candidate(tmp_path, mock, name="walk_v1")
        assert cand.stage == "hypothesis"
        walk_to_narrowed(cand, narrowed_symbols=["BTCUSDT"])
        assert cand.stage == "narrowed"
        assert cand.narrowed_symbols == ["BTCUSDT"]

    def test_no_narrowed_symbols(self, tmp_path):
        mock = _MockCache()
        _, cand = make_session_candidate(tmp_path, mock)
        walk_to_narrowed(cand)  # no symbol override
        assert cand.stage == "narrowed"
        assert cand.narrowed_symbols == []  # empty default

    def test_custom_frozen_params(self, tmp_path):
        mock = _MockCache()
        _, cand = make_session_candidate(tmp_path, mock)
        params = {"BTCUSDT": {"vol_pct_thresh": 0.5}}
        walk_to_narrowed(cand, frozen_params=params)
        assert cand.frozen_params == params
