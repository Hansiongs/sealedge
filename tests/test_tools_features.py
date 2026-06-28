"""Direct unit tests for ``quant_lib.tools.features`` (compute_features, build_matrices)."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quant_lib.tools import features as features_mod
from quant_lib.tools.features import build_matrices, compute_features


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_klines_df(n: int = 200, freq: str = "h", start: str = "2024-01-01",
                    base_price: float = 100.0, base_volume: float = 1000.0) -> pd.DataFrame:
    times = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({
        "time": times,
        "open": base_price, "high": base_price * 1.01,
        "low": base_price * 0.99, "close": base_price,
        "volume": base_volume,
    })


def _make_feature_df(n: int = 200, freq: str = "h", start: str = "2024-01-01") -> pd.DataFrame:
    """Build a DataFrame in the format returned by compute_features."""
    times = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({
        "time": times,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": 1000.0,
        "hh_20": 100.0, "ll_20": 100.0, "log_ret": 0.0,
        "realized_vol_24": 0.01, "vol_pct_rank": 0.5,
        "sma_vol_24": 0.01, "rvol": 1.0, "atr": 1.5,
        "ema_200": 100.0, "macro_vol": 0.5,
        "macro_trend": 1, "is_weekend": 0,
        "funding_rate": 0.0, "funding_missing": 0, "is_funding_hour": 0,
    })


# ═══════════════════════════════════════════════════════════════════════
# compute_features
# ═══════════════════════════════════════════════════════════════════════


class TestComputeFeatures:
    """``compute_features`` is a thin wrapper around the core
    feature pipeline.  We mock the underlying call to verify the
    wrapper's behaviour (default ``max_time``, argument forwarding).
    """

    def test_default_max_time_uses_latest_of_df_raw(self):
        """When ``max_time`` is None, the wrapper uses ``df_raw['time'].max()``."""
        df_raw = _make_klines_df(n=100, start="2024-01-01")
        df_btc = _make_klines_df(n=200, start="2023-12-01")
        expected_max = df_raw["time"].max()

        with patch("quant_lib.tools.features._prepare") as mock_prep:
            mock_prep.return_value = df_raw
            compute_features(df_raw, df_btc)

        # Positional args: (df_raw, df_btc, df_funding, max_time)
        # 4th positional arg is max_time
        assert mock_prep.call_args.args[3] == expected_max

    def test_explicit_max_time_passed_through(self):
        """An explicit ``max_time`` is forwarded to the underlying call."""
        df_raw = _make_klines_df(n=100)
        df_btc = _make_klines_df(n=200)
        explicit_max = pd.Timestamp("2024-01-15T12:00:00")

        with patch("quant_lib.tools.features._prepare") as mock_prep:
            mock_prep.return_value = df_raw
            compute_features(df_raw, df_btc, max_time=explicit_max)

        # 4th positional arg is max_time
        assert mock_prep.call_args.args[3] == explicit_max

    def test_funding_data_optional(self):
        """``df_funding_raw=None`` is passed through (no defaulting)."""
        df_raw = _make_klines_df(n=100)
        df_btc = _make_klines_df(n=200)

        with patch("quant_lib.tools.features._prepare") as mock_prep:
            mock_prep.return_value = df_raw
            compute_features(df_raw, df_btc, df_funding_raw=None)

        # 3rd positional arg is df_funding_raw
        assert mock_prep.call_args.args[2] is None

    def test_funding_data_passed_through(self):
        """Non-None funding data is passed through unchanged."""
        df_raw = _make_klines_df(n=100)
        df_btc = _make_klines_df(n=200)
        df_fund = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=10, freq="8h"),
            "funding_rate": [0.0001] * 10,
        })

        with patch("quant_lib.tools.features._prepare") as mock_prep:
            mock_prep.return_value = df_raw
            compute_features(df_raw, df_btc, df_funding_raw=df_fund)

        # 3rd positional arg is df_funding_raw
        assert mock_prep.call_args.args[2] is df_fund

    def test_returns_underlying_result(self):
        """The function returns what the underlying call returns."""
        df_raw = _make_klines_df(n=100)
        df_btc = _make_klines_df(n=200)
        sentinel = _make_feature_df(n=100)

        with patch(
            "quant_lib.tools.features._prepare",
            return_value=sentinel,
        ):
            result = compute_features(df_raw, df_btc)
        assert result is sentinel


# ═══════════════════════════════════════════════════════════════════════
# build_matrices
# ═══════════════════════════════════════════════════════════════════════


class TestBuildMatrices:
    """``build_matrices`` is a thin wrapper around ``_build_matrices``."""

    def test_passes_symbols_and_precomputed_through(self):
        symbols = ["BTCUSDT", "ETHUSDT"]
        precomputed = {
            sym: _make_feature_df(n=200) for sym in symbols
        }

        with patch("quant_lib.tools.features._build_matrices") as mock_bm:
            mock_bm.return_value = ({}, {})
            build_matrices(symbols, precomputed)

        # Both args are forwarded
        mock_bm.assert_called_once_with(symbols, precomputed)

    def test_returns_underlying_result(self):
        symbols = ["BTCUSDT"]
        precomputed = {"BTCUSDT": _make_feature_df(n=100)}
        sentinel_close = {"BTCUSDT": {}}
        sentinel_hl = {"BTCUSDT": {}}

        with patch(
            "quant_lib.tools.features._build_matrices",
            return_value=(sentinel_close, sentinel_hl),
        ):
            close, hl = build_matrices(symbols, precomputed)
        assert close is sentinel_close
        assert hl is sentinel_hl

    def test_empty_symbols(self):
        """Empty symbols list is passed through (underlying returns empty dicts)."""
        with patch(
            "quant_lib.tools.features._build_matrices",
            return_value=({}, {}),
        ) as mock_bm:
            close, hl = build_matrices([], {})
        assert close == {}
        assert hl == {}
        mock_bm.assert_called_once_with([], {})

    def test_returns_two_tuple(self):
        """``build_matrices`` returns a (close, hl) tuple."""
        symbols = ["BTCUSDT", "ETHUSDT"]
        precomputed = {sym: _make_feature_df(n=200) for sym in symbols}
        with patch(
            "quant_lib.tools.features._build_matrices",
            return_value=({}, {}),
        ):
            result = build_matrices(symbols, precomputed)
        assert isinstance(result, tuple)
        assert len(result) == 2
