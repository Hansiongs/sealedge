"""Shared fixtures, constants, and helpers for quant_lib tests.

Provides:
- Project-wide constants (HOLDOUT_PERIOD, TRAIN_PERIOD, etc.) to avoid
  hardcoded date strings scattered across test files.
- Synthetic OHLCV fixtures for parameterised engine tests.
- ``_MockCache`` as the single source of truth for the in-memory DataCache
  used by E2E, commit-flow, and universe-filter tests.
- Reusable candidate/session builders (kept as private helpers to avoid
  breaking the public ``research`` API surface).
- An autouse fixture that isolates the on-disk holdout seal / journal
  between tests so the suite is safe to run with ``pytest-xdist``.

Test Order Independence (F17)
------------------------------
The framework is verified to be order-independent: ``pytest-randomly``
randomises test order, and the suite passes regardless.  The
``_isolate_holdout_seal_files`` autouse fixture removes the shared
seal file at the start and end of every test, so concurrent or
out-of-order tests cannot interfere with each other.  This is part
of the framework's contract — any new test that requires fixed
ordering must be flagged with a comment explaining why.
"""
from __future__ import annotations

import glob
import os
from contextlib import contextmanager
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytest

# Type-only imports used in conftest signatures. Kept at module
# scope (not TYPE_CHECKING) because some conftest factories use
# these types at runtime via forward-reference strings.
from quant_lib.core._engine import EngineArgs  # noqa: E402
from quant_lib.research.candidate import Candidate  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# B0.1: HMAC seal secret for tests
# ─────────────────────────────────────────────────────────────────────
# All test sessions need ``QUANT_LIB_HMAC_SECRET`` set before any
# seal operation runs. We set it at conftest-load time (before any
# test collection imports ``quant_lib``) so the cached secret is
# available immediately. The value is 64 chars (well over the 32-char
# minimum enforced by ``get_hmac_secret``). Production deployments
# should set this to a cryptographically random string of their own.
os.environ.setdefault("QUANT_LIB_HMAC_SECRET", "x" * 64)


@pytest.fixture(autouse=True)
def _hmac_secret_is_set(monkeypatch):
    """Autouse fixture: ensure ``QUANT_LIB_HMAC_SECRET`` is set for
    every test. Also clears the cached secret so a test that
    intentionally changes the env var sees a fresh value.

    Most tests just need the secret present; specific tests that
    verify the missing-secret error path use ``monkeypatch.delenv``
    inside the test body, which will trigger our re-cache logic.
    """
    from quant_lib.audit.holdout import _reset_hmac_secret_cache
    monkeypatch.setenv("QUANT_LIB_HMAC_SECRET", "x" * 64)
    _reset_hmac_secret_cache()
    yield
    _reset_hmac_secret_cache()


# ═══════════════════════════════════════════════════════════════════════
# Shared constants (F13)
# ═══════════════════════════════════════════════════════════════════════
# Centralising these prevents date-drift when the framework's holdout
# convention moves.  Tests should import these names rather than embed
# the literal strings.

TRAIN_PERIOD: tuple[str, str] = ("2020-01-01", "2024-12-31")
HOLDOUT_PERIOD: tuple[str, str] = ("2025-01-01", "2025-06-30")
HOLDOUT_PERIOD_ALT: tuple[str, str] = ("2025-07-01", "2025-12-31")
HOLDOUT_PERIOD_FAR: tuple[str, str] = ("2026-01-01", "2026-06-30")
BTC_DATA_START: str = "2019-06-01"
DEFAULT_SYMBOLS: list[str] = ["BTCUSDT", "ETHUSDT"]

DEFAULT_N_BARS_HOURLY: int = 1000
DEFAULT_N_BARS_BTC: int = 2000
DEFAULT_N_BARS_TRADE: int = 50

GLOBAL_SEED: int = 42
HOURLY_SEED: int = 42
BTC_SEED: int = 7
FUNDING_SEED: int = 99
TRADES_SEED: int = 42
DAILY_CLOSE_SEED: int = 42


# ═══════════════════════════════════════════════════════════════════════
# Synthetic OHLCV fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_hourly_data() -> pd.DataFrame:
    """Generate 1000 bars of synthetic hourly OHLCV data."""
    np.random.seed(HOURLY_SEED)
    n = DEFAULT_N_BARS_HOURLY
    base_time = pd.Timestamp("2021-01-01")
    times = [base_time + pd.Timedelta(hours=i) for i in range(n)]

    close = 100.0 + np.cumsum(np.random.normal(0, 0.5, n))
    close = np.maximum(close, 10.0)
    high = close + np.abs(np.random.normal(0, 0.3, n))
    low = close - np.abs(np.random.normal(0, 0.3, n))
    open_ = close - np.random.normal(0, 0.2, n)
    volume = np.random.exponential(1000, n)

    return pd.DataFrame({
        "time": times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def sample_btc_data() -> pd.DataFrame:
    """Generate extended BTC data for feature computation."""
    np.random.seed(BTC_SEED)
    n = DEFAULT_N_BARS_BTC
    base_time = pd.Timestamp("2019-06-01")
    times = [base_time + pd.Timedelta(hours=i) for i in range(n)]

    close = 10000.0 + np.cumsum(np.random.normal(0, 50, n))
    close = np.maximum(close, 1000.0)
    high = close + np.abs(np.random.normal(0, 30, n))
    low = close - np.abs(np.random.normal(0, 30, n))

    return pd.DataFrame({
        "time": times,
        "open": close - np.random.normal(0, 20, n),
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.exponential(50000, n),
    })


@pytest.fixture
def sample_funding_data() -> pd.DataFrame:
    """Generate synthetic funding rate data."""
    np.random.seed(FUNDING_SEED)
    n = DEFAULT_N_BARS_HOURLY
    base_time = pd.Timestamp("2021-01-01")
    times_8h = [base_time + pd.Timedelta(hours=i * 8) for i in range(n // 8 + 1)]
    rates = np.random.normal(0.0001, 0.001, len(times_8h))
    return pd.DataFrame({
        "time": times_8h,
        "funding_rate": rates,
    })


@pytest.fixture
def sample_trades() -> list[dict]:
    """Generate synthetic OOS trade records."""
    np.random.seed(TRADES_SEED)
    n = DEFAULT_N_BARS_TRADE
    trades = []
    base_time = pd.Timestamp("2022-06-01")
    for i in range(n):
        entry = base_time + pd.Timedelta(hours=i * 100)
        exit_ = entry + pd.Timedelta(hours=np.random.randint(5, 50))
        r_net = np.random.normal(0.1, 0.5)
        trades.append({
            "entry_time": entry,
            "exit_time": exit_,
            "symbol": np.random.choice(["BTCUSDT", "ETHUSDT", "SOLUSDT"]),
            "r_net": r_net,
            "entry_price": 100.0 + np.random.normal(0, 5),
            "exit_price": 100.0 + np.random.normal(0, 5),
            "trade_dir": 1 if r_net > 0 else -1,
            "sl_pct": 0.02,
            "sl_mult": 1.5,
            "trail_atr": 3.0,
            "m_trend": np.random.choice([1, -1]),
            "macro_vol": np.random.uniform(0.3, 1.5),
            "risk_weight": 0.01,
            "trend_risk_mult": np.random.choice([0.5, 1.5]),
        })
    return trades


@pytest.fixture
def sample_daily_close_matrix() -> dict[str, dict]:
    """Generate synthetic daily close matrix for portfolio sim."""
    np.random.seed(DAILY_CLOSE_SEED)
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    matrix: dict[str, dict] = {}
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        prices = 100.0 + np.cumsum(np.random.normal(0, 0.5, len(dates)))
        matrix[sym] = {d: float(p) for d, p in zip(dates, prices)}
    return matrix


# ═══════════════════════════════════════════════════════════════════════
# Engine array factory (F8) — shared across engine, pullback, sprint
# ═══════════════════════════════════════════════════════════════════════


def make_engine_arrays(
    n: int = 1000,
    seed: int = HOURLY_SEED,
    *,
    closes: Optional[np.ndarray] = None,
    highs: Optional[np.ndarray] = None,
    lows: Optional[np.ndarray] = None,
    opens: Optional[np.ndarray] = None,
    atrs: Optional[np.ndarray] = None,
    rsi: Optional[np.ndarray] = None,
    rvol: Optional[np.ndarray] = None,
    vol_pct_rank: Optional[np.ndarray] = None,
    bullish_rev: Optional[np.ndarray] = None,
    bearish_rev: Optional[np.ndarray] = None,
    ema_200s: Optional[np.ndarray] = None,
    funding_rates: Optional[np.ndarray] = None,
    macro_vols: Optional[np.ndarray] = None,
    macro_trends: Optional[np.ndarray] = None,
    is_weekends: Optional[np.ndarray] = None,
    is_funding_hours: Optional[np.ndarray] = None,
) -> dict:
    """Factory for engine input arrays.

    Centralises the construction of synthetic arrays used by
    ``fast_trade_loop`` and ``simulate_trailing_stop_trade``.  Callers
    can override any individual array; non-overridden arrays are
    generated deterministically from a seeded RNG so test outcomes
    stay reproducible.

    Returns
    -------
    dict
        Mapping of parameter name → numpy array ready to be unpacked
        into ``fast_trade_loop(**arrays, ...)``.
    """
    rng = np.random.default_rng(seed)
    if closes is None:
        closes = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
        closes = np.maximum(closes, 10.0)
    if highs is None:
        highs = closes + np.abs(rng.normal(0.5, 0.2, n))
    if lows is None:
        lows = closes - np.abs(rng.normal(0.5, 0.2, n))
    if opens is None:
        opens = closes + rng.normal(0, 0.1, n)
    if atrs is None:
        atrs = np.abs(rng.normal(1.0, 0.2, n))
    if rsi is None:
        rsi = 50.0 + rng.normal(0, 5, n)
        rsi = np.clip(rsi, 0, 100)
    if rvol is None:
        rvol = rng.uniform(0, 5, n)
    if vol_pct_rank is None:
        vol_pct_rank = rng.uniform(0, 1, n)
    if bullish_rev is None:
        bullish_rev = (closes > np.roll(closes, 1)).astype(np.int32)
    if bearish_rev is None:
        bearish_rev = (closes < np.roll(closes, 1)).astype(np.int32)
    if ema_200s is None:
        ema_200s = np.full(n, np.mean(closes[: min(200, n)]))
    if funding_rates is None:
        funding_rates = rng.normal(0.0001, 0.001, n)
    if macro_vols is None:
        macro_vols = rng.uniform(0.3, 1.5, n)
    if macro_trends is None:
        macro_trends = rng.choice([-1, 1], n).astype(np.int32)
    if is_weekends is None:
        is_weekends = rng.integers(0, 2, n).astype(np.int32)
    if is_funding_hours is None:
        is_funding_hours = rng.integers(0, 2, n).astype(np.int32)

    return {
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "hh_20": np.maximum.accumulate(closes),
        "ll_20": np.minimum.accumulate(closes),
        "ema_200s": ema_200s,
        "rsi_14": rsi.astype(np.float64),
        "bullish_reversal": bullish_rev,
        "bearish_reversal": bearish_rev,
        "vol_pct_rank": vol_pct_rank,
        "rvol": rvol,
        "atrs": atrs,
        "funding_rates": funding_rates,
        "macro_vols": macro_vols,
        "macro_trends": macro_trends,
        "is_weekends": is_weekends,
        "is_funding_hours": is_funding_hours,
    }


def common_engine_extra(arrays: dict, seed: int = 0) -> dict:
    """Common kwargs for trend-risk-multiplier parameters."""
    _ = seed  # kept for API parity; no RNG calls here today
    return {
        "strategy_type": 0,  # STRATEGY_VOL_COMPRESSION
        "allow_long": 1,
        "allow_short": 1,
        "rsi_oversold": 30.0,
        "rsi_overbought": 70.0,
        "trend_aligned_mult": 1.5,
        "trend_counter_mult": 0.5,
    }


# ═══════════════════════════════════════════════════════════════════════
# Session / Candidate builders (F8) — shared by E2E + commit tests
# ═══════════════════════════════════════════════════════════════════════


def make_synthetic_holdout_data(
    symbols: list[str] = DEFAULT_SYMBOLS,
    start: str = HOLDOUT_PERIOD[0],
    end: str = HOLDOUT_PERIOD[1],
) -> dict[str, pd.DataFrame]:
    """Build a 1-H-bar holdout dataset covering the period.

    Returns
    -------
    dict[str, DataFrame]
        Mapping ``symbol → DataFrame`` with the columns required by
        ``ResearchSession`` to hash the holdout without network access.
    """
    times = pd.date_range(start, end, freq="h")
    return {
        sym: pd.DataFrame({
            "time": times,
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.0, "volume": 1000.0,
        })
        for sym in symbols
    }


def make_session_candidate(
    tmp: str,
    mock: "_MockCache",
    name: str = "e2e_v1",
    *,
    training_period: tuple[str, str] = TRAIN_PERIOD,
    holdout_period: tuple[str, str] = HOLDOUT_PERIOD,
    symbols: list[str] = DEFAULT_SYMBOLS,
    provide_holdout_data: bool = True,
):
    """Create a ``ResearchSession`` + ``Candidate`` with a mock cache.

    Importing the candidate type lazily keeps ``conftest.py`` import-time
    free of the heavy ``quant_lib.research`` dependency graph.
    """
    from quant_lib.audit import for_vol_compression
    from quant_lib.research.session import ResearchSession

    if provide_holdout_data:
        from quant_lib.audit import for_vol_compression as _  # noqa: F401  (forces audit import)
        _holdout_data = make_synthetic_holdout_data(
            symbols=symbols,
            start=holdout_period[0],
            end=holdout_period[1],
        )
        session = ResearchSession(
            training_period=training_period,
            holdout_period=holdout_period,
            symbols=symbols,
            cache_dir=tmp,
            btc_data_start=BTC_DATA_START,
            _holdout_data=_holdout_data,
        )
    else:
        session = ResearchSession(
            training_period=training_period,
            holdout_period=holdout_period,
            symbols=symbols,
            cache_dir=tmp,
            _skip_holdout_load=True,
        )
    session.cache = mock
    hyp = for_vol_compression(name, "m", "b", "c")
    cand = session.create_candidate(hyp)
    return session, cand


def make_candidate_ready(
    tmp: str,
    mock: Optional["_MockCache"] = None,
    name: str = "test_v1",
) -> "Candidate":
    """Build a Candidate walked to ``ready`` with frozen params populated.

    Used by commit-coverage tests.  When ``mock`` is provided the
    ``ResearchSession.cache`` attribute is replaced; otherwise the
    default cache (constructed at session init) is left untouched.
    """
    from quant_lib.audit import for_vol_compression
    from quant_lib.research.session import ResearchSession

    _holdout_data = make_synthetic_holdout_data()
    session = ResearchSession(
        training_period=TRAIN_PERIOD,
        holdout_period=HOLDOUT_PERIOD,
        symbols=DEFAULT_SYMBOLS,
        cache_dir=tmp,
        btc_data_start=BTC_DATA_START,
        _holdout_data=_holdout_data,
    )
    cand = session.create_candidate(for_vol_compression(name, "m", "b", "c"))

    cand._set_stage("universe")
    cand._set_stage("edge")
    cand._set_stage("narrowed")
    cand.narrowed_symbols = list(DEFAULT_SYMBOLS)

    cand.frozen_params = {
        sym: {
            "vol_pct_thresh": 0.20, "pullback_bars": 5,
            "trail_atr": 3.0, "sl_mult": 1.5,
        }
        for sym in DEFAULT_SYMBOLS
    }
    cand.risk_weights = {sym: 0.01 for sym in DEFAULT_SYMBOLS}
    cand.mark_ready()
    if mock is not None:
        session.cache = mock
    return cand


def walk_to_narrowed(
    cand,
    *,
    narrowed_symbols: Optional[list[str]] = None,
    frozen_params: Optional[dict] = None,
) -> None:
    """Walk a candidate to ``narrowed`` stage via the public state machine.

    This is the test-side helper for setting up a candidate in
    ``narrowed`` without invoking the full WFA pipeline.  It uses
    the private ``_set_stage`` because that is the only path that
    supports setting an arbitrary stage without running the
    full ``run_universe`` / ``run_edge_testing`` / ``run_narrowing``
    pipeline (which require a working data cache, an Optuna study,
    and on-disk fold artefacts).

    Callers that need to test the state machine *itself* (e.g.,
    the public ``mark_ready`` and stage-transition exceptions)
    should still use ``_set_stage`` directly so they can drive
    each transition individually.
    """
    cand._set_stage("universe")
    cand._set_stage("edge")
    cand._set_stage("narrowed")
    if narrowed_symbols is not None:
        cand.narrowed_symbols = list(narrowed_symbols)
    if frozen_params is not None:
        cand.frozen_params = dict(frozen_params)


# ═══════════════════════════════════════════════════════════════════════
# STATIC patcher (shared by commit / SPA / WFA tests)
# ═══════════════════════════════════════════════════════════════════════


@contextmanager
def patch_statics(**overrides):
    """Temporarily override ``STATIC`` values for the duration of a test.

    Restores original values on exit.  Keys whose saved value is ``None``
    (i.e., absent before the patch) are removed rather than re-inserted.
    """
    from quant_lib.core._config import STATIC

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


# ═══════════════════════════════════════════════════════════════════════
# _MockCache — in-memory DataCache returning signal-rich data
# ═══════════════════════════════════════════════════════════════════════


class _MockCache:
    """In-memory DataCache returning signal-rich data for any range.

    Used by:
    - test_e2e_happy_path.py (commit flow)
    - test_commit_coverage.py (commit flow)
    - test_universe_filter.py (universe selection)

    Replaces the prior test-to-test import pattern. Tests can override
    the per-symbol data via the `data_lookup` parameter (used by
    universe filter tests to inject low-volume / zero-price cases).

    Parameters
    ----------
    n_bars : int
        Number of hourly bars to generate per symbol (default 54000
        ≈ 6.16 years, spans 2019-06-01 to ~2025-08-01).
    data_lookup : dict, optional
        {sym: DataFrame} for per-symbol override. Useful for
        injecting custom data (e.g., low-volume, zero-price) to
        exercise universe filtering edge cases.
    """

    def __init__(self, n_bars: int = 54000, data_lookup: dict | None = None):
        self._n_bars = n_bars
        self._data_lookup = data_lookup
        self._cache: dict = {}
        self._hits = 0
        self._misses = 0

    def get_klines(self, symbol, interval, start, end):
        key = (symbol, interval)
        if key not in self._cache:
            self._misses += 1
            if self._data_lookup is not None and symbol in self._data_lookup:
                self._cache[key] = self._data_lookup[symbol].copy()
            else:
                self._cache[key] = self._build_signal_data()
        else:
            self._hits += 1
        df = self._cache[key]
        if start is not None:
            df = df[df["time"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["time"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)

    def get_funding(self, symbol, start, end):
        return None

    def invalidate(self, symbol=None):
        if symbol is None:
            self._cache.clear()
        else:
            for k in list(self._cache.keys()):
                if k[0] == symbol:
                    del self._cache[k]

    def _build_signal_data(self) -> pd.DataFrame:
        """Build deterministic data with vol_compression signals."""
        np.random.seed(HOURLY_SEED)
        n = self._n_bars
        n_per_signal = 30
        n_signals = min(n // n_per_signal, 100)
        close = np.full(n, 100.0)
        high = np.full(n, 100.3)
        low = np.full(n, 99.7)
        open_ = np.full(n, 100.0)
        for k in range(n_signals):
            base = k * n_per_signal
            if base + n_per_signal > n:
                break
            for i in range(20):
                idx = base + i
                if idx < n:
                    close[idx] = 100.0
            close[min(base + 21, n - 1)] = 102.0
            high[min(base + 21, n - 1)] = 102.5
            close[min(base + 22, n - 1)] = 100.5
            close[min(base + 23, n - 1)] = 103.0
        return pd.DataFrame({
            "time": pd.date_range(BTC_DATA_START, periods=n, freq="h"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.where(np.arange(n) % 30 < 21, 500.0, 5000.0),
        })


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# make_engine_args — factory for EngineArgs (DRY for perf/regression tests)
# ═══════════════════════════════════════════════════════════════════════


def make_engine_args(n: int = 1000, seed: int = HOURLY_SEED) -> "EngineArgs":
    """Factory for EngineArgs (used by bench + behavioral tests).

    Centralises the construction of EngineArgs with sensible defaults.
    Overrides mirror those of ``make_engine_arrays``.
    """
    from quant_lib.core._engine import EngineArgs
    from quant_lib.core._config import DEFAULTS
    arrays = make_engine_arrays(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    return EngineArgs(
        market_data=(
            arrays["opens"], arrays["highs"],
            arrays["lows"], arrays["closes"],
        ),
        channel_features=(
            arrays["hh_20"], arrays["ll_20"], arrays["ema_200s"],
        ),
        pullback_features=(
            arrays["rsi_14"], arrays["bullish_reversal"],
            arrays["bearish_reversal"],
        ),
        signal_features=(
            arrays["vol_pct_rank"], arrays["rvol"], arrays["atrs"],
        ),
        auxiliary_features=(
            arrays["funding_rates"], arrays["macro_vols"],
            arrays["macro_trends"], arrays["is_weekends"],
            arrays["is_funding_hours"],
        ),
        strategy_type=0,
        thresholds=(0.20, 2.5, 30.0, 70.0, 0.0),
        integer_params=(5, 36, 0, 0),
        exit_params=(3.0, 1.5),
        cost_model=(0.05, 2.0, DEFAULTS["stress_test_multiplier"]),
        flags=(1, 1, 1, 1),
        random_draws=rng.random(size=n * 2).astype(np.float64),
        trend_mults=(1.5, 0.5),
    )


# ═══════════════════════════════════════════════════════════════════════
# make_klines_df — factory for OHLCV + features DataFrame
# ═══════════════════════════════════════════════════════════════════════


def make_klines_df(n: int = 2000, start: str = "2019-06-01",
                   seed: int = HOURLY_SEED, with_features: bool = True) -> pd.DataFrame:
    """Factory for OHLCV + features DataFrame (klines format).

    Used by tests that need a DataFrame in the format returned by
    ``DataCache.get_klines`` (with optional precomputed features).
    """
    rng = np.random.default_rng(seed)
    start_dt = pd.Timestamp(start)
    times = [start_dt + timedelta(hours=i) for i in range(n)]
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 10.0)
    high = close + np.abs(rng.normal(0, 0.3, n))
    low = close - np.abs(rng.normal(0, 0.3, n))
    open_ = close + rng.normal(0, 0.1, n)
    df = pd.DataFrame({
        "time": times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.exponential(1000, n),
    })
    if with_features:
        df["hh_20"] = pd.Series(high).rolling(20).max().shift(1).bfill()
        df["ll_20"] = pd.Series(low).rolling(20).min().shift(1).bfill()
        df["ema_200"] = pd.Series(close).ewm(span=200, adjust=False).mean().shift(1).bfill()
        df["rsi_14"] = np.clip(50 + rng.normal(0, 10, n), 0, 100)
        df["bullish_reversal"] = np.zeros(n, dtype=np.int32)
        df["bearish_reversal"] = np.zeros(n, dtype=np.int32)
        df["vol_pct_rank"] = np.clip(rng.normal(0.3, 0.2, n), 0, 1)
        df["rvol"] = np.clip(rng.normal(2.0, 0.5, n), 0.5, 5.0)
        df["atr"] = np.full(n, 1.5)
        df["funding_rate"] = np.zeros(n)
        df["macro_vol"] = np.full(n, 0.5)
        df["macro_trend"] = np.ones(n, dtype=np.int32)
        df["is_weekend"] = np.zeros(n, dtype=np.int32)
        df["is_funding_hour"] = np.zeros(n, dtype=np.int32)
    return df


# Backwards-compat aliases (used by older test files)
# ═══════════════════════════════════════════════════════════════════════
# These names are kept for the current sprint so we can migrate test
# files incrementally without breaking their imports.  New code should
# import the canonical names above.
# (Aliases removed 2026-06-29: audit confirmed all 7 names are unused
# outside conftest. Test files define local helpers with the same
# underscore-prefixed names; conftest shadowing was misleading.)


# ═══════════════════════════════════════════════════════════════════════
# Per-test isolation of the on-disk holdout seal / journal (F1 / xdist)
# ═══════════════════════════════════════════════════════════════════════
# ``ResearchSession`` persists holdout seals and journals to
# ``<seal_dir>/holdout_<start>_<end>.json``.  When the same period is
# used across many tests (which is the case for HOLDOUT_PERIOD) and
# multiple workers run concurrently under pytest-xdist, they race on
# the same file and tests fail with ``SealVerificationFailed`` because
# the seal was broken by a peer worker.
#
# The autouse fixture below:
# 1. Sets ``QUANT_LIB_SEAL_DIR`` env var to a per-process temp directory
#    for the duration of the test, so concurrent workers write to
#    separate seal files. ``ResearchSession`` reads this env var via
#    its default-derivation logic (see session.py:seal_dir fallback).
# 2. Cleans up the per-process seal directory at session teardown so
#    no orphaned files accumulate across runs.
# This keeps the suite deterministic in serial AND in parallel runs.

# Per-process directory for holdout seal / journal files. Computed
# once per xdist worker (or once per serial run). Each xdist worker
# has its own PID-based subdir, so cleanup at session end is safe
# (it only ever touches the current process's seals).
import shutil as _shutil  # noqa: E402  (kept local to conftest)
import tempfile as _tempfile  # noqa: E402  (kept local to conftest)
_PROCESS_SEAL_DIR = os.path.join(
    _tempfile.gettempdir(),
    "hqs_seals_" + str(os.getpid()),
)
os.makedirs(_PROCESS_SEAL_DIR, exist_ok=True)


@pytest.fixture(scope="session", autouse=True)
def _cleanup_process_seal_dir():
    """Remove the per-process seal dir at session end.

    Each pytest worker (serial or xdist) creates its own
    ``hqs_seals_<pid>`` directory in the OS temp dir. Without
    cleanup these accumulate to no functional harm, but they
    clutter ``%TEMP%`` (Windows) or ``/tmp`` (POSIX) and can
    confuse post-mortem inspection. To preserve the directory
    for debugging, set ``HQS_KEEP_SEAL_DIR=1`` in the environment
    before running tests.
    """
    yield
    if os.environ.get("HQS_KEEP_SEAL_DIR") == "1":
        return
    if _PROCESS_SEAL_DIR and os.path.isdir(_PROCESS_SEAL_DIR):
        try:
            _shutil.rmtree(_PROCESS_SEAL_DIR, ignore_errors=True)
        except OSError:
            # Best-effort cleanup. Don't fail the test session over
            # a leftover temp dir; the OS will reap it eventually.
            pass


def _remove_seal_files(holdout_period) -> None:
    """Remove the holdout seal JSON for the given (start, end) period.

    We only remove the seal file here (not the journal). The journal
    is append-only and concurrent writes from xdist workers are
    benign (each worker writes its own session log; readers tolerate
    trailing partial records).  Removing the seal is what matters
    because ``HoldoutSet.verify()`` re-reads it on every commit.
    """
    if holdout_period is None:
        return
    start, end = holdout_period
    for path in glob.glob(
        os.path.join(_PROCESS_SEAL_DIR, f"holdout_{start}_{end}.json")
    ):
        try:
            os.remove(path)
        except OSError:
            pass


@pytest.fixture(autouse=True)
def _redirect_holdout_paths_to_process_dir(monkeypatch):
    """Force ``ResearchSession`` and ``quant_exp status`` to read/write
    seals / journals under ``_PROCESS_SEAL_DIR`` so that pytest-xdist
    workers do not race on the shared on-disk location.

    The redirect is applied before each test and reverted after.
    """
    monkeypatch.setenv("QUANT_LIB_SEAL_DIR", _PROCESS_SEAL_DIR)
    yield
    # monkeypatch restores env automatically




@pytest.fixture(autouse=True)
def _reset_data_module_state():
    """Reset ``quant_lib.core._data`` globals after each test.

    Prevents state pollution from ``test_data.py::TestEnsureDataExists``
    which sets ``_DATA_DIR_INITIALIZED = True`` (via ``try/finally``)
    but never restores it to ``False``.  Left in the ``True`` state,
    subsequent tests that rely on lazy directory creation (via
    ``_ensure_data_dir``) may silently skip the creation step and fail
    if ``DATA_DIR`` does not exist.

    Also catches any other test that temporarily modifies these globals
    and forgets to restore them (``test_data.py::TestDataDir`` resets
    ``_DATA_DIR_INITIALIZED`` in ``setup_method`` but only for its own
    test class).
    """
    import quant_lib.core._data as d
    yield
    d.DATA_DIR = "data_cache"
    d._DATA_DIR_INITIALIZED = False


@pytest.fixture(autouse=True)
def _isolate_holdout_seal_files(request):
    """Remove stale seal JSON for the test's holdout period.

    Defaults to HOLDOUT_PERIOD, the period used by virtually every
    commit / E2E test.  Tests that use a different period can request
    the ``holdout_period_for_isolation`` marker to override the
    default.
    """
    marker = request.node.get_closest_marker("holdout_period")
    if marker is not None and marker.args:
        period = marker.args[0]
    else:
        period = HOLDOUT_PERIOD
    _remove_seal_files(period)
    yield
    _remove_seal_files(period)


def pytest_runtest_setup(item):
    """Skip tests marked with ``@pytest.mark.network`` when
    ``OFFLINE`` environment variable is set (e.g., ``$env:OFFLINE=1``
    on Windows or ``OFFLINE=1`` on Unix).

    Usage:
        @pytest.mark.network
        def test_needs_internet(self): ...
    """
    if os.environ.get("OFFLINE") and item.get_closest_marker("network"):
        pytest.skip("skipping network test (OFFLINE=1)")
