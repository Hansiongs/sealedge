"""Tests for DataCache (7-day TTL)."""

import os
import tempfile
import json
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from quant_lib.research.cache import DataCache


class TestDataCache:
    def test_init_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp, ttl_days=7)
            assert os.path.exists(tmp)
            assert os.path.exists(os.path.join(tmp, "_meta"))

    def test_default_ttl_is_7_days(self):
        cache = DataCache()
        assert cache.ttl.days == 7

    def test_custom_ttl(self):
        cache = DataCache(ttl_days=14)
        assert cache.ttl.days == 14

    def test_stats_initial(self):
        cache = DataCache()
        stats = cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    def test_is_fresh_returns_false_for_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            assert cache._is_fresh("BTCUSDT", "klines_1h") is False

    def test_is_fresh_returns_true_for_recent_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            # Create meta file with recent timestamp
            meta_file = cache._meta_path("BTCUSDT", "klines_1h")
            meta = {
                "symbol": "BTCUSDT",
                "kind": "klines_1h",
                "path": "/tmp/BTCUSDT_1h_MASTER.csv",
                "cached_at": datetime.now().isoformat(),
                "ttl_days": 7,
            }
            with open(meta_file, "w") as f:
                json.dump(meta, f)
            assert cache._is_fresh("BTCUSDT", "klines_1h") is True

    def test_is_fresh_returns_false_for_expired(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp, ttl_days=7)
            # Create meta file with timestamp 8 days ago
            meta_file = cache._meta_path("BTCUSDT", "klines_1h")
            old_time = (datetime.now() - timedelta(days=8)).isoformat()
            meta = {
                "symbol": "BTCUSDT",
                "kind": "klines_1h",
                "path": "/tmp/BTCUSDT_1h_MASTER.csv",
                "cached_at": old_time,
                "ttl_days": 7,
            }
            with open(meta_file, "w") as f:
                json.dump(meta, f)
            assert cache._is_fresh("BTCUSDT", "klines_1h") is False

    def test_invalidate_specific_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            # Create meta files for 2 symbols
            for sym in ["BTCUSDT", "ETHUSDT"]:
                meta_file = cache._meta_path(sym, "klines_1h")
                meta = {
                    "symbol": sym, "kind": "klines_1h",
                    "path": "/tmp", "cached_at": datetime.now().isoformat(),
                }
                with open(meta_file, "w") as f:
                    json.dump(meta, f)
            assert os.path.exists(cache._meta_path("BTCUSDT", "klines_1h"))
            assert os.path.exists(cache._meta_path("ETHUSDT", "klines_1h"))
            cache.invalidate("BTCUSDT")
            assert not os.path.exists(cache._meta_path("BTCUSDT", "klines_1h"))
            assert os.path.exists(cache._meta_path("ETHUSDT", "klines_1h"))

    def test_invalidate_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            for sym in ["BTCUSDT", "ETHUSDT"]:
                meta_file = cache._meta_path(sym, "klines_1h")
                meta = {
                    "symbol": sym, "kind": "klines_1h",
                    "path": "/tmp", "cached_at": datetime.now().isoformat(),
                }
                with open(meta_file, "w") as f:
                    json.dump(meta, f)
            cache.invalidate()
            assert not os.path.exists(cache._meta_path("BTCUSDT", "klines_1h"))
            assert not os.path.exists(cache._meta_path("ETHUSDT", "klines_1h"))

    def test_save_meta_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            fake_csv = os.path.join(tmp, "BTCUSDT_1h_MASTER.csv")
            with open(fake_csv, "w") as f:
                f.write("time,close\n2024-01-01,100\n")
            cache._save_meta("BTCUSDT", "klines_1h", fake_csv)
            assert os.path.exists(cache._meta_path("BTCUSDT", "klines_1h"))

    def test_repr(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp, ttl_days=7)
            r = repr(cache)
            assert "DataCache" in r
            assert "ttl" in r


class TestDataCacheFetchPaths:
    """Test the actual fetch paths (cache miss → fetch → save)."""

    def test_get_klines_cache_miss_fetches_and_saves(self):
        """First call to get_klines should fetch and save meta."""
        import tempfile
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            # Create the CSV that ensure_data_exists would have written
            csv_path = os.path.join(tmp, "BTCUSDT_1h_MASTER.csv")
            df = pd.DataFrame({
                "time": pd.date_range("2024-01-01", periods=10, freq="h"),
                "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000,
            })
            df.to_csv(csv_path, index=False)
            with patch("quant_lib.research.cache.ensure_data_exists") as mock_fetch:
                mock_fetch.return_value = csv_path
                result = cache.get_klines("BTCUSDT", "1h", "2024-01-01", "2024-12-31")
            mock_fetch.assert_called_once_with(
                "BTCUSDT", "1h", "2024-01-01", "2024-12-31"
            )
            assert len(result) == 10
            assert cache._misses == 1
            assert cache._hits == 0
            # Meta was saved
            assert os.path.exists(cache._meta_path("BTCUSDT", "klines_1h"))

    def test_get_klines_cache_hit_skips_fetch(self):
        """Subsequent call with fresh cache skips the fetch."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            # Create CSV + fresh meta
            csv_path = os.path.join(tmp, "BTCUSDT_1h_MASTER.csv")
            pd.DataFrame({
                "time": pd.date_range("2024-01-01", periods=5, freq="h"),
                "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000,
            }).to_csv(csv_path, index=False)
            meta_file = cache._meta_path("BTCUSDT", "klines_1h")
            with open(meta_file, "w") as f:
                json.dump({
                    "symbol": "BTCUSDT", "kind": "klines_1h",
                    "path": csv_path,
                    "cached_at": datetime.now().isoformat(),
                }, f)
            with patch("quant_lib.research.cache.ensure_data_exists") as mock_fetch:
                result = cache.get_klines("BTCUSDT", "1h", "2024-01-01", "2024-12-31")
            # Fetch NOT called
            mock_fetch.assert_not_called()
            assert len(result) == 5
            assert cache._hits == 1
            assert cache._misses == 0

    def test_get_klines_cache_hit_returns_full_cached_data(self):
        """Cache hit returns the full cached file (caller filters as needed).

        The cache's contract is to return cached data efficiently; the
        start/end params are used only to FETCH the data, not to filter
        the cache hit. Callers (e.g., commit.py) do their own filtering
        after the cache returns.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            csv_path = os.path.join(tmp, "BTCUSDT_1h_MASTER.csv")
            pd.DataFrame({
                "time": pd.date_range("2024-01-01", periods=10, freq="h"),
                "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000,
            }).to_csv(csv_path, index=False)
            meta_file = cache._meta_path("BTCUSDT", "klines_1h")
            with open(meta_file, "w") as f:
                json.dump({
                    "symbol": "BTCUSDT", "kind": "klines_1h",
                    "path": csv_path,
                    "cached_at": datetime.now().isoformat(),
                }, f)
            # Request a small range, but the cache returns all 10 rows
            # (start/end are not applied on cache hit)
            result = cache.get_klines(
                "BTCUSDT", "1h", "2024-01-01", "2024-01-02",
            )
            assert len(result) == 10

    def test_get_funding_cache_miss_returns_dataframe(self):
        """When ensure_funding_exists returns a path, cache reads CSV."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            csv_path = os.path.join(tmp, "BTCUSDT_FUNDING_MASTER.csv")
            pd.DataFrame({
                "time": pd.date_range("2024-01-01", periods=5, freq="8h"),
                "funding_rate": 0.0001,
            }).to_csv(csv_path, index=False)
            with patch("quant_lib.research.cache.ensure_funding_exists") as mock_fetch:
                mock_fetch.return_value = csv_path
                result = cache.get_funding("BTCUSDT", "2024-01-01", "2024-12-31")
            assert result is not None
            assert len(result) == 5
            assert "funding_rate" in result.columns
            assert cache._misses == 1

    def test_get_funding_returns_none_when_no_data(self):
        """When ensure_funding_exists returns None, cache returns None."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            with patch("quant_lib.research.cache.ensure_funding_exists") as mock_fetch:
                mock_fetch.return_value = None
                result = cache.get_funding("BTCUSDT", "2024-01-01", "2024-12-31")
            assert result is None

    def test_get_funding_cache_hit_skips_fetch(self):
        """Fresh meta for funding → skip fetch."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            csv_path = os.path.join(tmp, "BTCUSDT_FUNDING_MASTER.csv")
            pd.DataFrame({
                "time": pd.date_range("2024-01-01", periods=5, freq="8h"),
                "funding_rate": 0.0001,
            }).to_csv(csv_path, index=False)
            meta_file = cache._meta_path("BTCUSDT", "funding")
            with open(meta_file, "w") as f:
                json.dump({
                    "symbol": "BTCUSDT", "kind": "funding",
                    "path": csv_path,
                    "cached_at": datetime.now().isoformat(),
                }, f)
            with patch("quant_lib.research.cache.ensure_funding_exists") as mock_fetch:
                result = cache.get_funding("BTCUSDT", "2024-01-01", "2024-12-31")
            mock_fetch.assert_not_called()
            assert len(result) == 5
            assert cache._hits == 1

    def test_is_fresh_returns_false_for_corrupt_json(self):
        """Corrupt meta file → treated as not fresh."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            meta_file = cache._meta_path("BTCUSDT", "klines_1h")
            with open(meta_file, "w") as f:
                f.write("{not valid json")
            assert cache._is_fresh("BTCUSDT", "klines_1h") is False

    def test_is_fresh_returns_false_for_missing_key(self):
        """Meta file missing 'cached_at' key → treated as not fresh."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            meta_file = cache._meta_path("BTCUSDT", "klines_1h")
            with open(meta_file, "w") as f:
                json.dump({"symbol": "BTCUSDT", "kind": "klines_1h"}, f)
            assert cache._is_fresh("BTCUSDT", "klines_1h") is False
