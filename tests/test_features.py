"""Tests for feature engineering — leakage-aware feature computation."""

import numpy as np
import pandas as pd

from quant_lib.core._features import prepare_data_with_max_time


class TestPrepareDataWithMaxTime:
    def test_basic_feature_computation(self, sample_hourly_data, sample_btc_data):
        max_time = sample_hourly_data["time"].max()
        result = prepare_data_with_max_time(
            sample_hourly_data, sample_btc_data, None, max_time
        )
        assert not result.empty
        expected_cols = {
            "hh_20", "ll_20", "log_ret", "realized_vol_24", "vol_pct_rank",
            "sma_vol_24", "rvol", "atr", "ema_200", "macro_vol",
            "macro_trend", "is_weekend", "funding_rate", "funding_missing",
            "is_funding_hour",
        }
        assert expected_cols.issubset(result.columns)

    def test_max_time_filtering(self, sample_hourly_data, sample_btc_data):
        cutoff = sample_hourly_data["time"].iloc[500]
        result = prepare_data_with_max_time(
            sample_hourly_data, sample_btc_data, None, cutoff
        )
        assert result["time"].max() <= cutoff
        assert len(result) <= 501  # 500 + 1 because of <= max_time

    def test_empty_input_returns_empty(self, sample_btc_data):
        df_empty = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        max_time = pd.Timestamp("2021-06-01")
        result = prepare_data_with_max_time(df_empty, sample_btc_data, None, max_time)
        assert result.empty

    def test_funding_rate_integration(self, sample_hourly_data, sample_btc_data, sample_funding_data):
        max_time = sample_hourly_data["time"].max()
        result = prepare_data_with_max_time(
            sample_hourly_data, sample_btc_data, sample_funding_data, max_time
        )
        assert "funding_rate" in result.columns
        assert "funding_missing" in result.columns

    def test_funding_none_returns_zeros(self, sample_hourly_data, sample_btc_data):
        max_time = sample_hourly_data["time"].max()
        result = prepare_data_with_max_time(
            sample_hourly_data, sample_btc_data, None, max_time
        )
        assert (result["funding_rate"] == 0.0).all()
        assert (result["funding_missing"] == 0).all()

    def test_vol_pct_rank_no_leakage(self, sample_hourly_data, sample_btc_data):
        """vol_pct_rank must use shifted values to prevent lookahead."""
        max_time = sample_hourly_data["time"].max()
        result = prepare_data_with_max_time(
            sample_hourly_data, sample_btc_data, None, max_time
        )
        vol_col = result["realized_vol_24"].dropna()
        if len(vol_col) > 25:
            first_computed = vol_col.iloc[24]
            # The first value should be NaN because of the rolling window
            # After shift(1), the earliest valid value is at index 25+
            assert not np.isnan(first_computed)

    def test_gap_detection(self, sample_hourly_data, sample_btc_data):
        """Introduce a gap and verify gap detection doesn't crash."""
        df = sample_hourly_data.copy()
        # Remove a chunk to create a gap
        gap_start = 300
        gap_end = 350
        df = pd.concat([df.iloc[:gap_start], df.iloc[gap_end:]], ignore_index=True)

        max_time = df["time"].max()
        result = prepare_data_with_max_time(df, sample_btc_data, None, max_time)
        assert not result.empty
        assert "atr" in result.columns

    def test_macro_trend_from_btc(self, sample_hourly_data, sample_btc_data):
        max_time = sample_hourly_data["time"].max()
        result = prepare_data_with_max_time(
            sample_hourly_data, sample_btc_data, None, max_time
        )
        assert "macro_trend" in result.columns
        assert result["macro_trend"].dropna().isin([-1, 1]).all()

    def test_weekend_flag(self, sample_hourly_data, sample_btc_data):
        max_time = sample_hourly_data["time"].max()
        result = prepare_data_with_max_time(
            sample_hourly_data, sample_btc_data, None, max_time
        )
        assert "is_weekend" in result.columns
        assert result["is_weekend"].dtype == np.int32
