"""Smoke test against a real Binance fixture (small, deterministic).

Validates that the framework's data ingestion + feature computation
paths work on real-world data, not just synthetic.
"""
from pathlib import Path
import pandas as pd
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "btcusdt_1h_2024_jan.csv"


@pytest.fixture(scope="module")
def real_btc():
    if not FIXTURE.exists():
        pytest.skip(
            f"Fixture not found: {FIXTURE}\n"
            f"Generate via: python tools/download_fixture.py"
        )
    return pd.read_csv(FIXTURE, parse_dates=["time"])


class TestRealDataFixtureSchema:
    def test_fixture_loads(self, real_btc):
        assert len(real_btc) > 40

    def test_required_columns_present(self, real_btc):
        for col in ("time", "open", "high", "low", "close", "volume"):
            assert col in real_btc.columns

    def test_no_null_values(self, real_btc):
        for col in ("open", "high", "low", "close", "volume"):
            assert real_btc[col].notna().all()

    def test_ohlc_invariants(self, real_btc):
        assert (real_btc["high"] >= real_btc[["open", "close"]].max(axis=1)).all()
        assert (real_btc["low"] <= real_btc[["open", "close"]].min(axis=1)).all()

    def test_time_is_hourly(self, real_btc):
        diffs = real_btc["time"].diff().dropna()
        assert (diffs >= pd.Timedelta("55min")).all()
        assert (diffs <= pd.Timedelta("1h 5min")).all()

    def test_realistic_price_range(self, real_btc):
        assert 20000 < real_btc["close"].min() < 100000
        assert 20000 < real_btc["close"].max() < 100000

    def test_realistic_volume(self, real_btc):
        assert (real_btc["volume"] > 0).all()
        assert 500 < real_btc["volume"].median() < 2000


class TestRealDataFeatures:
    def test_features_compute_without_error(self, real_btc):
        df = real_btc.copy()
        df["hh_20"] = df["high"].rolling(20).max().shift(1).bfill()
        df["ll_20"] = df["low"].rolling(20).min().shift(1).bfill()
        df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean().shift(1).bfill()
        assert df["hh_20"].notna().all()
        assert df["ll_20"].notna().all()
        assert df["ema_200"].notna().all()

    def test_data_cache_constructs(self, real_btc, tmp_path):
        from quant_lib.research.cache import DataCache
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        real_btc.to_csv(cache_dir / "BTCUSDT_1h_2024-01.csv", index=False)
        cache = DataCache(cache_dir=cache_dir, ttl_days=365)
        assert cache is not None
