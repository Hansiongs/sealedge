"""Tests for Candidate.run_universe universe-selection filter (S6.4)."""

import tempfile
import pandas as pd
import pytest

from quant_lib.audit import for_vol_compression
from quant_lib.research.session import ResearchSession
from tests.conftest import _MockCache  # M-2: shared cache helper


def _make_candidate(tmp, data_lookup, symbols, train_start, train_end):
    """Build a Candidate with a mock cache that returns the given data per symbol."""
    session = ResearchSession(
        training_period=(train_start, train_end),
        holdout_period=("2025-01-01", "2025-06-30"),
        symbols=symbols,
        cache_dir=tmp, _skip_holdout_load=True,
    )
    session.cache = _MockCache(data_lookup=data_lookup)
    hyp = for_vol_compression("uv_v1", "m", "b", "c")
    return session.create_candidate(hyp)


def _make_synth_klines(n_hours, base_price=100.0, base_volume=500.0, start="2019-06-01"):
    """Build a synthetic hourly klines DataFrame with required columns.

    Defaults to 24*5000 hours (~5.7 years) starting 2019-06-01 to cover
    any reasonable test scenario (180+ days age, 90d lookback).
    """
    times = pd.date_range(start, periods=n_hours, freq="h")
    return pd.DataFrame({
        "time": times,
        "open": base_price,
        "high": base_price * 1.01,
        "low": base_price * 0.99,
        "close": base_price,
        "volume": base_volume,
    })


class TestUniverseVolumeAgeFilter:
    """run_universe now applies real volume + age criteria (not just 'has data')."""

    def test_high_volume_symbol_passes(self):
        """Symbol with high USDT volume and sufficient age passes filter."""
        with tempfile.TemporaryDirectory() as tmp:
            # 24*5000 = 5.7y; covers 2019-06-01 to ~2025-03-04
            df = _make_synth_klines(24 * 5000, base_price=30000, base_volume=100.0)
            cand = _make_candidate(
                tmp,
                data_lookup={"BTCUSDT": df, "HIGHUSDT": df},
                symbols=["BTCUSDT", "HIGHUSDT"],
                train_start="2021-01-01", train_end="2024-12-31",
            )
            cand.run_universe(min_volume_usdt=1_000_000, min_age_days=180)
            # Both symbols: $30k * 100 * 24 = $72M/day (passes), age 1000d+ (passes)
            assert set(cand.eligible_symbols) == {"BTCUSDT", "HIGHUSDT"}

    def test_low_volume_symbol_excluded(self):
        """Symbol with low USDT volume is excluded from eligible list."""
        with tempfile.TemporaryDirectory() as tmp:
            df_low = _make_synth_klines(24 * 5000, base_price=10.0, base_volume=10.0)
            cand = _make_candidate(
                tmp,
                data_lookup={"LOWUSDT": df_low},
                symbols=["LOWUSDT"],
                train_start="2021-01-01", train_end="2024-12-31",
            )
            with pytest.raises(Exception) as exc_info:
                cand.run_universe(min_volume_usdt=50_000_000, min_age_days=180)
            assert "No symbols passed" in str(exc_info.value)

    def test_too_young_symbol_excluded(self):
        """Symbol listed less than min_age_days before start is excluded."""
        with tempfile.TemporaryDirectory() as tmp:
            # Data starts 60 days before train_start (less than 180d)
            df = _make_synth_klines(24 * 200, base_price=30000, base_volume=100.0, start="2020-11-01")
            cand = _make_candidate(
                tmp,
                data_lookup={"YOUNGUSDT": df},
                symbols=["YOUNGUSDT"],
                train_start="2021-01-01", train_end="2024-12-31",
            )
            with pytest.raises(Exception) as exc_info:
                cand.run_universe(min_volume_usdt=1_000_000, min_age_days=180)
            assert "No symbols passed" in str(exc_info.value)

    def test_mixed_eligible_and_excluded(self):
        """Some symbols pass, some fail; only passing symbols returned."""
        with tempfile.TemporaryDirectory() as tmp:
            df_high = _make_synth_klines(24 * 5000, base_price=30000, base_volume=100.0)
            df_low = _make_synth_klines(24 * 5000, base_price=10.0, base_volume=10.0)
            cand = _make_candidate(
                tmp,
                data_lookup={"HIGHUSDT": df_high, "LOWUSDT": df_low},
                symbols=["HIGHUSDT", "LOWUSDT"],
                train_start="2021-01-01", train_end="2024-12-31",
            )
            cand.run_universe(min_volume_usdt=1_000_000, min_age_days=180)
            assert "HIGHUSDT" in cand.eligible_symbols
            assert "LOWUSDT" not in cand.eligible_symbols

    def test_threshold_boundary_inclusive(self):
        """Symbol whose volume exactly equals the threshold passes (= not strictly less)."""
        with tempfile.TemporaryDirectory() as tmp:
            # 24 * 100 * price = 2400 * price. We want exactly 1_000_000 USDT/day.
            # 1_000_000 / 2400 = 416.666... Use price 416.666..., volume 100.
            df = _make_synth_klines(24 * 5000, base_price=1_000_000 / 2400, base_volume=100.0)
            cand = _make_candidate(
                tmp,
                data_lookup={"BOUNDUSDT": df},
                symbols=["BOUNDUSDT"],
                train_start="2021-01-01", train_end="2024-12-31",
            )
            # With exact threshold: median should be >= 1M
            cand.run_universe(min_volume_usdt=1_000_000, min_age_days=180)
            assert "BOUNDUSDT" in cand.eligible_symbols


class TestUniverseFilterEdgeCases:
    """Edge cases: empty data, missing columns, zero price."""

    def test_empty_dataframe_excluded(self):
        """Symbol with empty precomputed data is excluded."""
        with tempfile.TemporaryDirectory() as tmp:
            df_empty = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
            cand = _make_candidate(
                tmp,
                data_lookup={"EMPTYUSDT": df_empty},
                symbols=["EMPTYUSDT"],
                train_start="2021-01-01", train_end="2024-12-31",
            )
            with pytest.raises(Exception) as exc_info:
                cand.run_universe()
            assert "No symbols passed" in str(exc_info.value)

    def test_zero_price_excluded(self):
        """Symbol with close=0 (degenerate) is excluded (would be infinite USDT)."""
        with tempfile.TemporaryDirectory() as tmp:
            df_zero = _make_synth_klines(24 * 5000, base_price=0.0, base_volume=100.0)
            cand = _make_candidate(
                tmp,
                data_lookup={"ZEROPUSDT": df_zero},
                symbols=["ZEROPUSDT"],
                train_start="2021-01-01", train_end="2024-12-31",
            )
            with pytest.raises(Exception) as exc_info:
                cand.run_universe(min_volume_usdt=1, min_age_days=0)
            assert "No symbols passed" in str(exc_info.value)

    def test_low_thresholds_allow_synthetic_data(self):
        """With min_volume_usdt=0 and min_age_days=0, normal data passes."""
        with tempfile.TemporaryDirectory() as tmp:
            df = _make_synth_klines(24 * 5000, base_price=100.0, base_volume=500.0)
            cand = _make_candidate(
                tmp,
                data_lookup={"NORMUSDT": df},
                symbols=["NORMUSDT"],
                train_start="2020-01-15", train_end="2020-04-30",
            )
            cand.run_universe(min_volume_usdt=0, min_age_days=0)
            assert "NORMUSDT" in cand.eligible_symbols
