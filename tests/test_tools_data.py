"""Direct unit tests for ``quant_lib.tools.data`` (fetch_klines, fetch_funding).

These tests mock the underlying ``_data`` module functions to verify
the ``tools.data`` layer correctly delegates and processes results.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd



def _fetch_klines(*args, **kwargs):
    """Lazy import wrapper — prevents stale ``__globals__`` after
    ``test_regression_b0_3_warnings.py`` destroys and re-imports
    all ``quant_lib.*`` modules (which orphans the function object's
    ``__globals__`` and breaks ``monkeypatch.setattr``).
    """
    from quant_lib.tools.data import fetch_klines as _fk
    return _fk(*args, **kwargs)


def _fetch_funding(*args, **kwargs):
    """Lazy import wrapper — same rationale as ``_fetch_klines``."""
    from quant_lib.tools.data import fetch_funding as _ff
    return _ff(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# fetch_klines
# ═══════════════════════════════════════════════════════════════════════


class TestFetchKlines:
    """``fetch_klines`` delegates to ``_ensure_data_exists`` and
    parses the resulting CSV into a DataFrame with a datetime index.
    """

    def test_returns_dataframe_with_expected_columns(
        self, tmp_path, monkeypatch,
    ):
        """``fetch_klines`` returns a DataFrame with the canonical
        OHLCV columns + datetime ``time``.
        """
        # Write a synthetic CSV
        times = pd.date_range("2024-01-01", periods=10, freq="h")
        df_in = pd.DataFrame({
            "time": times,
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.0, "volume": 1000.0,
        })
        csv_path = tmp_path / "BTCUSDT_1h_MASTER.csv"
        df_in.to_csv(csv_path, index=False)

        mock_ensure = MagicMock(return_value=str(csv_path))
        monkeypatch.setattr("quant_lib.tools.data._ensure_data_exists", mock_ensure)
        result = _fetch_klines("BTCUSDT", "1h", "2024-01-01", "2024-01-08")

        assert isinstance(result, pd.DataFrame)
        for col in ("time", "open", "high", "low", "close", "volume"):
            assert col in result.columns

    def test_time_column_is_datetime(self, tmp_path, monkeypatch):
        """``time`` is converted to datetime dtype."""
        times = pd.date_range("2024-01-01", periods=5, freq="h")
        df_in = pd.DataFrame({
            "time": times, "open": 100.0, "high": 101.0,
            "low": 99.0, "close": 100.0, "volume": 1000.0,
        })
        csv_path = tmp_path / "BTCUSDT_1h_MASTER.csv"
        df_in.to_csv(csv_path, index=False)
        mock_ensure = MagicMock(return_value=str(csv_path))
        monkeypatch.setattr("quant_lib.tools.data._ensure_data_exists", mock_ensure)
        result = _fetch_klines("BTCUSDT", "1h", "2024-01-01", "2024-01-01")
        assert pd.api.types.is_datetime64_any_dtype(result["time"])

    def test_default_parameters(self, tmp_path, monkeypatch):
        """Default start_date / end_date are passed through."""
        times = pd.date_range("2024-01-01", periods=3, freq="h")
        df_in = pd.DataFrame({
            "time": times, "open": 100.0, "high": 101.0,
            "low": 99.0, "close": 100.0, "volume": 1000.0,
        })
        csv_path = tmp_path / "ETHUSDT_1h_MASTER.csv"
        df_in.to_csv(csv_path, index=False)
        mock_ensure = MagicMock(return_value=str(csv_path))
        monkeypatch.setattr("quant_lib.tools.data._ensure_data_exists", mock_ensure)
        _fetch_klines("ETHUSDT")
        # Verify the function was called (we don't assert on the
        # specific default values — those are tested in test_data.py)
        assert mock_ensure.called

    def test_interval_passed_through(self, tmp_path, monkeypatch):
        """The ``interval`` argument is passed to ``_ensure_data_exists``."""
        times = pd.date_range("2024-01-01", periods=3, freq="4h")
        df_in = pd.DataFrame({
            "time": times, "open": 100.0, "high": 101.0,
            "low": 99.0, "close": 100.0, "volume": 1000.0,
        })
        csv_path = tmp_path / "BTCUSDT_4h_MASTER.csv"
        df_in.to_csv(csv_path, index=False)
        mock_ensure = MagicMock(return_value=str(csv_path))
        monkeypatch.setattr("quant_lib.tools.data._ensure_data_exists", mock_ensure)
        _fetch_klines("BTCUSDT", interval="4h")
        call_args = mock_ensure.call_args
        assert "4h" in call_args.args


# ═══════════════════════════════════════════════════════════════════════
# fetch_funding
# ═══════════════════════════════════════════════════════════════════════


class TestFetchFunding:
    """``fetch_funding`` delegates to ``_ensure_funding_exists``."""

    def test_returns_dataframe_on_success(self, tmp_path, monkeypatch):
        """When ``_ensure_funding_exists`` returns a path,
        a DataFrame is returned with ``time`` as datetime.
        """
        times = pd.date_range("2024-01-01", periods=5, freq="8h")
        df_in = pd.DataFrame({
            "time": times,
            "funding_rate": [0.0001, -0.0001, 0.0002, 0.0, -0.0001],
        })
        csv_path = tmp_path / "BTCUSDT_FUNDING_MASTER.csv"
        df_in.to_csv(csv_path, index=False)
        mock_ensure = MagicMock(return_value=str(csv_path))
        monkeypatch.setattr("quant_lib.tools.data._ensure_funding_exists", mock_ensure)
        result = _fetch_funding("BTCUSDT", "2024-01-01", "2024-01-02")
        assert isinstance(result, pd.DataFrame)
        assert "funding_rate" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["time"])

    def test_returns_none_when_data_unavailable(self, monkeypatch):
        """If ``_ensure_funding_exists`` returns None, ``fetch_funding``
        returns None (no exception raised).
        """
        mock_ensure = MagicMock(return_value=None)
        monkeypatch.setattr("quant_lib.tools.data._ensure_funding_exists", mock_ensure)
        result = _fetch_funding("BTCUSDT", "2024-01-01", "2024-01-02")
        assert result is None

    def test_default_parameters(self, monkeypatch):
        """Default start/end dates are passed through."""
        mock_ensure = MagicMock(return_value=None)
        monkeypatch.setattr("quant_lib.tools.data._ensure_funding_exists", mock_ensure)
        _fetch_funding("ETHUSDT")
        assert mock_ensure.called

    def test_time_column_is_datetime(self, tmp_path, monkeypatch):
        """The ``time`` column is always converted to datetime."""
        times = pd.date_range("2024-01-01", periods=3, freq="8h")
        df_in = pd.DataFrame({
            "time": times,
            "funding_rate": [0.0, 0.0001, 0.0],
        })
        csv_path = tmp_path / "BTCUSDT_FUNDING_MASTER.csv"
        df_in.to_csv(csv_path, index=False)
        mock_ensure = MagicMock(return_value=str(csv_path))
        monkeypatch.setattr("quant_lib.tools.data._ensure_funding_exists", mock_ensure)
        result = _fetch_funding("BTCUSDT", "2024-01-01", "2024-01-01")
        assert pd.api.types.is_datetime64_any_dtype(result["time"])

    def test_funding_path_returned_by_underlying(self, tmp_path, monkeypatch):
        """Verify the underlying function's return value is used as the
        CSV path (not ignored / not re-fetched).
        """
        times = pd.date_range("2024-01-01", periods=3, freq="8h")
        df_in = pd.DataFrame({
            "time": times,
            "funding_rate": [0.0] * 3,
        })
        csv_path = tmp_path / "unique_path.csv"
        df_in.to_csv(csv_path, index=False)
        mock_ensure = MagicMock(return_value=str(csv_path))
        monkeypatch.setattr("quant_lib.tools.data._ensure_funding_exists", mock_ensure)
        result = _fetch_funding("BTCUSDT", "2024-01-01", "2024-01-02")
        # The unique path is the one read
        assert len(result) == 3
        # Verify the mock was called with our specific args
        assert mock_ensure.call_args.args[0] == "BTCUSDT"
