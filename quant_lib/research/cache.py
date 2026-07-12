"""
Data cache with configurable TTL (default 7 days).

Thin wrapper over ``core._data.ensure_data_exists`` plus metadata,
invalidation, and per-symbol cache stats.
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from quant_lib.core._data import ensure_data_exists, ensure_funding_exists


class DataCache:
    """Caches Binance klines + funding with configurable TTL.

    Parameters
    ----------
    cache_dir : str
        Directory to store cached files. Default: "data_cache".
    ttl_days : int
        Time-to-live in days. Cache entries older than this trigger re-fetch.
        Default: 7.
    """

    def __init__(self, cache_dir: str = "data_cache", ttl_days: int = 7):
        self.cache_dir = cache_dir
        self.ttl = timedelta(days=ttl_days)
        self._hits = 0
        self._misses = 0

        os.makedirs(self.cache_dir, exist_ok=True)
        self._meta_dir = os.path.join(self.cache_dir, "_meta")
        os.makedirs(self._meta_dir, exist_ok=True)

    def _meta_path(self, symbol: str, kind: str) -> str:
        """Get metadata file path for a symbol/kind."""
        return os.path.join(self._meta_dir, f"{symbol}_{kind}.json")

    def _is_fresh(self, symbol: str, kind: str) -> bool:
        """Check if cached file is still within TTL."""
        meta_file = self._meta_path(symbol, kind)
        if not os.path.exists(meta_file):
            return False
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
            cached_at = datetime.fromisoformat(meta["cached_at"])
            return (datetime.now() - cached_at) < self.ttl
        except (json.JSONDecodeError, KeyError, ValueError):
            return False

    def _save_meta(self, symbol: str, kind: str, path: str) -> None:
        """Save cache metadata."""
        meta = {
            "symbol": symbol,
            "kind": kind,
            "path": path,
            "cached_at": datetime.now().isoformat(),
            "ttl_days": self.ttl.days,
        }
        # Compute file hash for tamper detection
        if os.path.exists(path):
            with open(path, "rb") as f:
                meta["file_hash"] = hashlib.sha256(f.read()).hexdigest()
        with open(self._meta_path(symbol, kind), "w") as f:
            json.dump(meta, f, indent=2)

    def get_klines(
        self, symbol: str, interval: str, start: str, end: str
    ) -> pd.DataFrame:
        """Get klines from cache if fresh, else fetch from Binance.

        Returns
        -------
        pd.DataFrame
            Cached or freshly-fetched klines.
        """
        if self._is_fresh(symbol, f"klines_{interval}"):
            cache_path = os.path.join(
                self.cache_dir, f"{symbol}_{interval}_MASTER.csv"
            )
            if os.path.exists(cache_path):
                self._hits += 1
                df = pd.read_csv(cache_path)
                df["time"] = pd.to_datetime(df["time"])
                return df

        self._misses += 1
        path = ensure_data_exists(symbol, interval, start, end)
        self._save_meta(symbol, f"klines_{interval}", path)
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        return df

    def get_funding(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Get funding rate data from cache if fresh, else fetch.

        Returns None if no funding data available for the symbol.
        """
        if self._is_fresh(symbol, "funding"):
            cache_path = os.path.join(
                self.cache_dir, f"{symbol}_FUNDING_MASTER.csv"
            )
            if os.path.exists(cache_path):
                self._hits += 1
                df = pd.read_csv(cache_path)
                df["time"] = pd.to_datetime(df["time"])
                return df

        self._misses += 1
        path = ensure_funding_exists(symbol, start, end)
        if path is None:
            return None
        self._save_meta(symbol, "funding", path)
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        return df

    def invalidate(self, symbol: Optional[str] = None) -> None:
        """Invalidate cache (all or per-symbol).

        Parameters
        ----------
        symbol : str or None
            If None, invalidate all cache entries.
            If str, invalidate only that symbol.
        """
        if symbol is None:
            for f in os.listdir(self._meta_dir):
                if f.endswith(".json"):
                    os.remove(os.path.join(self._meta_dir, f))
        else:
            for f in os.listdir(self._meta_dir):
                if f.startswith(f"{symbol}_") and f.endswith(".json"):
                    os.remove(os.path.join(self._meta_dir, f))

    @property
    def stats(self) -> dict:
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }

    def __repr__(self) -> str:
        return (
            f"DataCache(dir={self.cache_dir}, ttl={self.ttl.days}d, "
            f"hits={self._hits}, misses={self._misses})"
        )
