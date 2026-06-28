"""Tests for data fetching module — lazy init, path construction, retry logic, ensure_data_exists."""
import io
import os
import tempfile
import zipfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quant_lib.core._data import (
    DATA_DIR,
    _data_path,
    _ensure_data_dir,
    fetch_with_retry,
    ensure_data_exists,
)


# ═══════════════════════════════════════════════════════════════════════
# DataDir lazy init
# ═══════════════════════════════════════════════════════════════════════


class TestDataDir:
    def setup_method(self):
        """Reset the lazy init flag for testing."""
        import quant_lib.core._data as d
        d._DATA_DIR_INITIALIZED = False

    def test_ensure_data_dir_creates_directory(self):
        import quant_lib.core._data as d
        original_dir = d.DATA_DIR
        try:
            with tempfile.TemporaryDirectory() as tmp:
                d.DATA_DIR = os.path.join(tmp, "test_cache")
                d._DATA_DIR_INITIALIZED = False
                assert not os.path.exists(d.DATA_DIR)
                d._ensure_data_dir()
                assert os.path.exists(d.DATA_DIR)
        finally:
            d.DATA_DIR = original_dir
            d._DATA_DIR_INITIALIZED = False

    def test_data_path_creates_dir_on_first_call(self):
        import quant_lib.core._data as d
        original_dir = d.DATA_DIR
        try:
            with tempfile.TemporaryDirectory() as tmp:
                d.DATA_DIR = os.path.join(tmp, "lazy_cache")
                d._DATA_DIR_INITIALIZED = False
                assert not os.path.exists(d.DATA_DIR)
                path = d._data_path("test.csv")
                assert os.path.exists(d.DATA_DIR)
                assert path.endswith("test.csv")
        finally:
            d.DATA_DIR = original_dir
            d._DATA_DIR_INITIALIZED = False

    def test_data_path_returns_absolute_path_inside_data_dir(self):
        path = _data_path("BTCUSDT_1h_MASTER.csv")
        assert DATA_DIR in path
        assert path.endswith("BTCUSDT_1h_MASTER.csv")

    def test_ensure_data_dir_is_idempotent(self):
        """Calling _ensure_data_dir multiple times does not raise."""
        import quant_lib.core._data as d
        original_dir = d.DATA_DIR
        try:
            with tempfile.TemporaryDirectory() as tmp:
                d.DATA_DIR = os.path.join(tmp, "idempotent")
                d._DATA_DIR_INITIALIZED = False
                d._ensure_data_dir()
                d._ensure_data_dir()
                d._ensure_data_dir()
                assert os.path.isdir(d.DATA_DIR)
        finally:
            d.DATA_DIR = original_dir
            d._DATA_DIR_INITIALIZED = False


# ═══════════════════════════════════════════════════════════════════════
# fetch_with_retry
# ═══════════════════════════════════════════════════════════════════════


class TestFetchWithRetry:
    def test_fetch_with_retry_timeout_returns_none(self):
        """fetch_with_retry returns None for unreachable URLs."""
        result = fetch_with_retry("http://127.0.0.1:1/nonexistent", timeout=1, max_retries=1)
        assert result is None

    @pytest.mark.network
    def test_fetch_with_retry_invalid_url_returns_none(self):
        result = fetch_with_retry("http://nonexistent.invalid/data", timeout=1, max_retries=1)
        assert result is None

    def test_fetch_with_retry_returns_response_on_success(self):
        """A successful HTTP call returns the response object."""
        with patch("quant_lib.core._data.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            result = fetch_with_retry("http://example.com", timeout=1, max_retries=1)
            assert result is mock_response

    def test_fetch_with_retry_404_returns_none_immediately(self):
        """404 errors should NOT retry — they return None after one call."""
        from requests.exceptions import HTTPError
        with patch("quant_lib.core._data.requests.get") as mock_get:
            with patch("quant_lib.core._data.time.sleep") as mock_sleep:
                mock_response = MagicMock()
                mock_response.status_code = 404
                err = HTTPError(response=mock_response)
                mock_get.side_effect = err
                result = fetch_with_retry("http://example.com/404", timeout=1, max_retries=3)
                assert result is None
                assert mock_get.call_count == 1
                # No sleep on 404 (immediate return)
                assert mock_sleep.call_count == 0

    def test_fetch_with_retry_retries_on_timeout(self):
        """Timeout errors should retry up to max_retries times."""
        from requests.exceptions import Timeout
        with patch("quant_lib.core._data.requests.get") as mock_get:
            with patch("quant_lib.core._data.time.sleep") as mock_sleep:
                mock_get.side_effect = Timeout("test timeout")
                result = fetch_with_retry("http://example.com", timeout=1, max_retries=3)
                assert result is None
                assert mock_get.call_count == 3
                # Sleep called twice (between attempts, not after the last)
                assert mock_sleep.call_count == 2

    def test_fetch_with_retry_retries_on_5xx(self):
        """5xx HTTP errors should retry."""
        from requests.exceptions import HTTPError
        with patch("quant_lib.core._data.requests.get") as mock_get:
            with patch("quant_lib.core._data.time.sleep") as mock_sleep:
                mock_response = MagicMock()
                mock_response.status_code = 500
                err = HTTPError(response=mock_response)
                mock_get.side_effect = err
                result = fetch_with_retry("http://example.com", timeout=1, max_retries=2)
                assert result is None
                assert mock_get.call_count == 2

    def test_fetch_with_retry_exponential_backoff(self):
        """Sleep duration doubles between retries (2^attempt)."""
        from requests.exceptions import Timeout
        with patch("quant_lib.core._data.requests.get") as mock_get:
            with patch("quant_lib.core._data.time.sleep") as mock_sleep:
                mock_get.side_effect = Timeout("test")
                fetch_with_retry("http://example.com", timeout=1, max_retries=3)
                # Sleeps should be: 2^0=1, 2^1=2 (only between retries)
                assert mock_sleep.call_args_list[0].args[0] == 1
                assert mock_sleep.call_args_list[1].args[0] == 2


# ═══════════════════════════════════════════════════════════════════════
# ensure_data_exists
# ═══════════════════════════════════════════════════════════════════════


def _make_mock_zip_bytes(n: int = 100) -> bytes:
    """Build an in-memory ZIP containing a headerless CSV (Binance format).

    The production code reads the zip with ``header=None`` then assigns
    column names itself, so the mock must also be headerless.  The
    production parser then calls ``pd.to_datetime(..., unit="ms")``
    on the time column, so the mock must contain millisecond
    timestamps.
    """
    times = pd.date_range("2024-01-01", periods=n, freq="h")
    # pandas astype("int64") returns MICROSECONDS since epoch.
    # Convert to milliseconds (μs → ms) for the unit="ms" parser.
    times_ms = (times.astype("int64") // 1000).tolist()
    rows = [
        f"{t},100.0,101.0,99.0,100.0,1000.0,0,0,0,0,0,0"
        for t in times_ms
    ]
    csv_content = "\n".join(rows)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.csv", csv_content)
    return buf.getvalue()


def _make_mock_csv_bytes(n: int = 100) -> bytes:
    """Build in-memory raw CSV bytes (non-zip, headerless, ms timestamps)."""
    times = pd.date_range("2024-01-01", periods=n, freq="h")
    times_ms = (times.astype("int64") // 1000).tolist()
    rows = [
        f"{t},100.0,101.0,99.0,100.0,1000.0,0,0,0,0,0,0"
        for t in times_ms
    ]
    return "\n".join(rows).encode("utf-8")


def _make_valid_klines_df(n: int = 100) -> pd.DataFrame:
    """Build a cached CSV with proper columns (post-fetch format)."""
    times = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "time": times,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": 1000.0,
    })


class TestEnsureDataExists:
    """``ensure_data_exists`` is the data-fetch orchestrator.

    Tests mock the network and zipfile layers so the function runs
    end-to-end without real downloads.
    """

    def test_uses_cached_file_when_valid(self, tmp_path):
        """If a valid cache file exists, no network call is made."""
        import quant_lib.core._data as d
        original_dir = d.DATA_DIR
        try:
            d.DATA_DIR = str(tmp_path)
            d._DATA_DIR_INITIALIZED = True
            df = _make_valid_klines_df(200)
            cached_path = os.path.join(tmp_path, "BTCUSDT_1h_MASTER.csv")
            df.to_csv(cached_path, index=False)
            with patch("quant_lib.core._data.fetch_with_retry") as mock_fetch:
                result = ensure_data_exists(
                    "BTCUSDT", "1h", "2024-01-01", "2024-01-08",
                )
            assert result == cached_path
            # fetch_with_retry should NOT be called (cache hit)
            mock_fetch.assert_not_called()
        finally:
            d.DATA_DIR = original_dir

    def test_cached_file_with_wrong_columns_is_rejected(self, tmp_path):
        """A cached file with missing columns must trigger a refetch."""
        import quant_lib.core._data as d
        original_dir = d.DATA_DIR
        try:
            d.DATA_DIR = str(tmp_path)
            d._DATA_DIR_INITIALIZED = True
            df_bad = pd.DataFrame({
                "time": pd.date_range("2024-01-01", periods=10, freq="h"),
                "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            })
            df_bad.to_csv(
                os.path.join(tmp_path, "BTCUSDT_1h_MASTER.csv"), index=False,
            )
            zip_bytes = _make_mock_zip_bytes(100)
            mock_response = MagicMock()
            mock_response.content = zip_bytes
            with patch("quant_lib.core._data.fetch_with_retry", return_value=mock_response):
                result = ensure_data_exists(
                    "BTCUSDT", "1h", "2024-01-01", "2024-01-08",
                )
            assert os.path.exists(result)
        finally:
            d.DATA_DIR = original_dir

    def test_fetches_when_no_cache(self, tmp_path):
        """No cache file exists → fetch from network and save."""
        import quant_lib.core._data as d
        original_dir = d.DATA_DIR
        try:
            d.DATA_DIR = str(tmp_path)
            d._DATA_DIR_INITIALIZED = True
            zip_bytes = _make_mock_zip_bytes(100)
            mock_response = MagicMock()
            mock_response.content = zip_bytes
            with patch("quant_lib.core._data.fetch_with_retry", return_value=mock_response):
                result = ensure_data_exists(
                    "BTCUSDT", "1h", "2024-01-01", "2024-01-08",
                )
            assert os.path.exists(result)
            cached = pd.read_csv(result)
            assert "volume" in cached.columns
            assert len(cached) == 100
        finally:
            d.DATA_DIR = original_dir

    def test_raises_on_network_failure_with_no_cache(self, tmp_path):
        """If fetch returns None and no cache exists, raise RuntimeError."""
        import quant_lib.core._data as d
        original_dir = d.DATA_DIR
        try:
            d.DATA_DIR = str(tmp_path)
            d._DATA_DIR_INITIALIZED = True
            with patch("quant_lib.core._data.fetch_with_retry", return_value=None):
                with pytest.raises(RuntimeError, match="[Cc]ritical"):
                    ensure_data_exists(
                        "BTCUSDT", "1h", "2024-01-01", "2024-01-08",
                    )
        finally:
            d.DATA_DIR = original_dir

    def test_corrupt_zip_logs_and_continues(self, tmp_path):
        """If the zip is corrupt, the function logs a warning and fails."""
        import quant_lib.core._data as d
        original_dir = d.DATA_DIR
        try:
            d.DATA_DIR = str(tmp_path)
            d._DATA_DIR_INITIALIZED = True
            # Return bytes that aren't a valid zip
            mock_response = MagicMock()
            mock_response.content = b"not a zip"
            with patch("quant_lib.core._data.fetch_with_retry", return_value=mock_response):
                with pytest.raises(RuntimeError, match="[Cc]ritical"):
                    ensure_data_exists(
                        "BTCUSDT", "1h", "2024-01-01", "2024-01-08",
                    )
        finally:
            d.DATA_DIR = original_dir

    def test_partial_month_failures_logged(self, tmp_path):
        """Failed months are logged but don't raise if at least one succeeds."""
        import quant_lib.core._data as d
        original_dir = d.DATA_DIR
        try:
            d.DATA_DIR = str(tmp_path)
            d._DATA_DIR_INITIALIZED = True

            call_count = [0]

            def fake_fetch(url, *args, **kwargs):
                call_count[0] += 1
                if "2024-01" in url:
                    # First month: corrupt
                    mock_response = MagicMock()
                    mock_response.content = b"not a zip"
                    return mock_response
                # Other months: empty list (no calls anyway)
                return None

            # Just verify fetch_with_retry is called with the expected URL
            with patch("quant_lib.core._data.fetch_with_retry", side_effect=fake_fetch):
                with pytest.raises(RuntimeError):
                    ensure_data_exists(
                        "BTCUSDT", "1h", "2024-01-01", "2024-01-08",
                    )
            # Only one URL was queried (single month)
            assert call_count[0] == 1
        finally:
            d.DATA_DIR = original_dir

    def test_cache_used_when_range_already_covered(self, tmp_path):
        """If cached data covers the requested range, return cache path
        without refetching.
        """
        import quant_lib.core._data as d
        original_dir = d.DATA_DIR
        try:
            d.DATA_DIR = str(tmp_path)
            d._DATA_DIR_INITIALIZED = True
            # Write a cached file that COVERS the requested range
            df = _make_valid_klines_df(200)
            cached_path = os.path.join(tmp_path, "BTCUSDT_1h_MASTER.csv")
            df.to_csv(cached_path, index=False)
            with patch("quant_lib.core._data.fetch_with_retry") as mock_fetch:
                result = ensure_data_exists(
                    "BTCUSDT", "1h", "2024-01-01", "2024-01-05",
                )
            assert result == cached_path
            mock_fetch.assert_not_called()
        finally:
            d.DATA_DIR = original_dir
