"""Direct unit tests for ``quant_lib.tools.universe``.

The ``tools.universe`` module provides the mechanical universe
selection helpers used by the framework's Phase 1 (universe filter)
and Phase 3 (narrowing) stages.  These tests exercise the public
functions directly with synthetic data, isolating them from the
``Candidate.run_universe`` integration path tested in
``test_universe_filter.py``.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quant_lib.tools.universe import filter_by_volume_rank, select_universe


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _make_klines_df(
    n_bars: int = 5000,
    start: str = "2019-06-01",
    base_price: float = 100.0,
    base_volume: float = 1000.0,
    freq: str = "h",
) -> pd.DataFrame:
    """Build a synthetic hourly klines DataFrame."""
    times = pd.date_range(start, periods=n_bars, freq=freq)
    return pd.DataFrame({
        "time": times,
        "open": base_price,
        "high": base_price * 1.01,
        "low": base_price * 0.99,
        "close": base_price,
        "volume": base_volume,
    })


# ─────────────────────────────────────────────────────────────────────
# filter_by_volume_rank
# ─────────────────────────────────────────────────────────────────────


class TestFilterByVolumeRank:
    """``filter_by_volume_rank`` keeps the top-N symbols by median volume."""

    def test_basic_top_n(self):
        df_high = _make_klines_df(base_volume=5000.0)
        df_low = _make_klines_df(base_volume=100.0)
        result = filter_by_volume_rank(
            ["HIGH", "LOW"],
            {"HIGH": df_high, "LOW": df_low},
            top_n=1,
        )
        assert result == ["HIGH"], "High-volume symbol should be selected"

    def test_returns_top_n_in_order(self):
        df1 = _make_klines_df(base_volume=100.0)
        df2 = _make_klines_df(base_volume=200.0)
        df3 = _make_klines_df(base_volume=300.0)
        result = filter_by_volume_rank(
            ["A", "B", "C"],
            {"A": df1, "B": df2, "C": df3},
            top_n=2,
        )
        assert set(result) == {"B", "C"}
        # Top-2 should be C and B (highest volumes)
        assert result[0] == "C", "Highest volume should be first"

    def test_top_n_larger_than_input(self):
        df = _make_klines_df(base_volume=100.0)
        result = filter_by_volume_rank(
            ["A"], {"A": df}, top_n=5,
        )
        assert result == ["A"]

    def test_empty_input(self):
        result = filter_by_volume_rank([], {}, top_n=3)
        assert result == []

    def test_uses_median_not_mean(self):
        """An outlier bar should not affect ranking (median is robust)."""
        # All bars volume=100 except one outlier at 100_000
        times = pd.date_range("2020-01-01", periods=100, freq="h")
        df_robust = pd.DataFrame({
            "time": times,
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "volume": [100.0] * 99 + [100_000.0],
        })
        df_consistent = pd.DataFrame({
            "time": times,
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "volume": [200.0] * 100,
        })
        result = filter_by_volume_rank(
            ["ROBUST", "CONSISTENT"],
            {"ROBUST": df_robust, "CONSISTENT": df_consistent},
            top_n=1,
        )
        # Median of [100]*99 + [100_000] = 100
        # Median of [200]*100 = 200
        # So CONSISTENT should win despite the outlier
        assert result == ["CONSISTENT"], (
            f"Median should make CONSISTENT win; got {result}"
        )


# ─────────────────────────────────────────────────────────────────────
# select_universe (using mock for ensure_data_exists)
# ─────────────────────────────────────────────────────────────────────


class TestSelectUniverse:
    """``select_universe`` uses mechanical (volume + age) criteria only.

    We mock ``ensure_data_exists`` to return a CSV path with synthetic
    data so the function runs end-to-end without network access.
    """

    def test_high_volume_old_symbol_passes(self, tmp_path, monkeypatch):
        """Symbol with high volume + sufficient age passes filter."""
        df = _make_klines_df(
            n_bars=24 * 5000, base_price=30000, base_volume=100.0,
        )
        # 100 * 30000 * 24 = 72M USDT/day, age ~5.7y
        csv_path = tmp_path / "BTCUSDT.csv"
        df.to_csv(csv_path, index=False)
        # Mock ensure_data_exists
        monkeypatch.setattr(
            "quant_lib.tools.universe.ensure_data_exists",
            lambda *a, **kw: str(csv_path),
        )
        result = select_universe(
            ["BTCUSDT"],
            start_date="2021-01-01",
            end_date="2024-12-31",
            min_volume_usdt=1_000_000,
            min_age_days=180,
            verbose=False,
        )
        assert "BTCUSDT" in result

    def test_low_volume_symbol_excluded(self, tmp_path, monkeypatch):
        """Symbol with low USDT volume is excluded."""
        df = _make_klines_df(
            n_bars=24 * 5000, base_price=10.0, base_volume=10.0,
        )
        # 10 * 10 * 24 = 2.4k USDT/day (way below 50M)
        csv_path = tmp_path / "LOWUSDT.csv"
        df.to_csv(csv_path, index=False)
        monkeypatch.setattr(
            "quant_lib.tools.universe.ensure_data_exists",
            lambda *a, **kw: str(csv_path),
        )
        result = select_universe(
            ["LOWUSDT"],
            start_date="2021-01-01",
            end_date="2024-12-31",
            min_volume_usdt=50_000_000,
            min_age_days=180,
            verbose=False,
        )
        assert "LOWUSDT" not in result

    def test_too_young_symbol_excluded(self, tmp_path, monkeypatch):
        """Symbol listed less than min_age_days before start is excluded."""
        df = _make_klines_df(
            n_bars=24 * 200, base_price=30000, base_volume=100.0,
            start="2020-11-01",
        )
        # 60 days before train_start < 180 day threshold
        csv_path = tmp_path / "YOUNGUSDT.csv"
        df.to_csv(csv_path, index=False)
        monkeypatch.setattr(
            "quant_lib.tools.universe.ensure_data_exists",
            lambda *a, **kw: str(csv_path),
        )
        result = select_universe(
            ["YOUNGUSDT"],
            start_date="2021-01-01",
            end_date="2024-12-31",
            min_volume_usdt=1_000_000,
            min_age_days=180,
            verbose=False,
        )
        assert "YOUNGUSDT" not in result

    def test_data_unavailable_continues_to_next(self, tmp_path, monkeypatch):
        """If ensure_data_exists raises for one symbol, others still process."""
        # First symbol raises, second returns valid CSV
        df = _make_klines_df(
            n_bars=24 * 5000, base_price=30000, base_volume=100.0,
        )
        csv_path = tmp_path / "GOODUSDT.csv"
        df.to_csv(csv_path, index=False)

        def _mock_ensure(sym, *args, **kwargs):
            if sym == "BADUSDT":
                raise RuntimeError("network error")
            return str(csv_path)
        monkeypatch.setattr(
            "quant_lib.tools.universe.ensure_data_exists", _mock_ensure,
        )
        result = select_universe(
            ["BADUSDT", "GOODUSDT"],
            start_date="2021-01-01",
            end_date="2024-12-31",
            min_volume_usdt=1_000_000,
            min_age_days=180,
            verbose=False,
        )
        assert "BADUSDT" not in result
        assert "GOODUSDT" in result

    def test_returns_sorted(self, tmp_path, monkeypatch):
        """Eligible symbols are returned in sorted order (deterministic)."""
        df = _make_klines_df(
            n_bars=24 * 5000, base_price=30000, base_volume=100.0,
        )
        monkeypatch.setattr(
            "quant_lib.tools.universe.ensure_data_exists",
            lambda *a, **kw: (
                _write_and_return(tmp_path, a[0], df)
            ),
        )
        result = select_universe(
            ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            start_date="2021-01-01",
            end_date="2024-12-31",
            min_volume_usdt=1_000_000,
            min_age_days=180,
            verbose=False,
        )
        # Must be sorted
        assert result == sorted(result), f"Result not sorted: {result}"

    def test_verbose_log_messages(self, tmp_path, monkeypatch, caplog):
        """verbose=True logs per-symbol diagnostics."""
        df = _make_klines_df(
            n_bars=24 * 5000, base_price=30000, base_volume=100.0,
        )
        csv_path = tmp_path / "BTCUSDT.csv"
        df.to_csv(csv_path, index=False)
        monkeypatch.setattr(
            "quant_lib.tools.universe.ensure_data_exists",
            lambda *a, **kw: str(csv_path),
        )
        import logging
        with caplog.at_level(logging.INFO, logger="rich"):
            select_universe(
                ["BTCUSDT"],
                start_date="2021-01-01",
                end_date="2024-12-31",
                min_volume_usdt=1_000_000,
                min_age_days=180,
                verbose=True,
            )
        # Should log eligibility info
        assert any("Eligible" in r.message or "Universe" in r.message
                   for r in caplog.records)

    def test_verbose_false_does_not_log_per_symbol(self, tmp_path, monkeypatch, caplog):
        """verbose=False suppresses per-symbol log lines."""
        df = _make_klines_df(
            n_bars=24 * 5000, base_price=30000, base_volume=100.0,
        )
        csv_path = tmp_path / "BTCUSDT.csv"
        df.to_csv(csv_path, index=False)
        monkeypatch.setattr(
            "quant_lib.tools.universe.ensure_data_exists",
            lambda *a, **kw: str(csv_path),
        )
        import logging
        with caplog.at_level(logging.INFO, logger="rich"):
            select_universe(
                ["BTCUSDT"],
                start_date="2021-01-01",
                end_date="2024-12-31",
                min_volume_usdt=1_000_000,
                min_age_days=180,
                verbose=False,
            )
        # verbose=False: no per-symbol "Eligible" lines
        eligible_msgs = [r for r in caplog.records if "Eligible" in r.message]
        assert not eligible_msgs, (
            f"verbose=False should not log per-symbol: {[r.message for r in eligible_msgs]}"
        )

    def test_zero_volume_data_excluded(self, tmp_path, monkeypatch):
        """Symbol with all-zero volume is excluded (median = 0 < min)."""
        df = _make_klines_df(
            n_bars=24 * 5000, base_price=30000, base_volume=0.0,
        )
        csv_path = tmp_path / "ZEROVOLUSDT.csv"
        df.to_csv(csv_path, index=False)
        monkeypatch.setattr(
            "quant_lib.tools.universe.ensure_data_exists",
            lambda *a, **kw: str(csv_path),
        )
        result = select_universe(
            ["ZEROVOLUSDT"],
            start_date="2021-01-01",
            end_date="2024-12-31",
            min_volume_usdt=1_000_000,
            min_age_days=180,
            verbose=False,
        )
        assert "ZEROVOLUSDT" not in result

    def test_insufficient_lookback_data_excluded(self, tmp_path, monkeypatch):
        """Symbol with <24 hourly bars in lookback window is excluded."""
        # Only 50 bars → lookback yields 50 < 24... wait, 50 > 24. Use 12.
        df = _make_klines_df(
            n_bars=12, base_price=30000, base_volume=100.0,
            start="2024-12-01",
        )
        csv_path = tmp_path / "SHORTUSDT.csv"
        df.to_csv(csv_path, index=False)
        monkeypatch.setattr(
            "quant_lib.tools.universe.ensure_data_exists",
            lambda *a, **kw: str(csv_path),
        )
        result = select_universe(
            ["SHORTUSDT"],
            start_date="2024-12-01",
            end_date="2024-12-31",
            min_volume_usdt=1_000_000,
            min_age_days=0,  # don't reject on age
            volume_lookback_days=90,
            verbose=False,
        )
        assert "SHORTUSDT" not in result


def _write_and_return(tmp_path, sym, df) -> str:
    """Helper: write a per-symbol CSV under tmp_path and return its path."""
    csv_path = tmp_path / f"{sym}.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)
