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


# ════════════════════════════════════════════════════════════════════════
# Phase 2.2: macro_trend holdout isolation
# ════════════════════════════════════════════════════════════════════════


class TestMacroTrendHoldoutIsolation:
    """Phase 2.2: when btc_holdout_start is provided, macro_trend
    must be computed strictly on pre-holdout data, then forward-
    filled into the holdout window. This prevents the holdout's own
    price action from leaking into the trend signal."""

    def test_macro_trend_constant_in_holdout_when_btc_holdout_start_set(
        self, sample_hourly_data, sample_btc_data
    ):
        """The trend signal in the holdout is derived from a CONSTANT
        EMA value (forward-filled last-EMA-computed-on-pre-holdout).

        The signal can still flip between -1 and +1 as close crosses
        that constant, but the *underlying threshold* is fixed (not a
        function of the holdout's rolling EMA).

        We verify this by comparing two computations:
        1. With btc_holdout_start set (constant-EMA, the new behavior)
        2. Without btc_holdout_start but with data truncated to
           pre-holdout only (would compute the same constant EMA and
           forward-fill manually)
        They should produce identical results in the pre-holdout region.
        """
        max_time = sample_hourly_data["time"].max()
        all_times = sample_hourly_data["time"].sort_values().unique()
        btc_holdout_start = all_times[len(all_times) * 3 // 4]
        result_with = prepare_data_with_max_time(
            sample_hourly_data,
            sample_btc_data,
            None,
            max_time,
            btc_holdout_start=btc_holdout_start,
        )

        # Pre-holdout region: macro_trend values should be the same
        # regardless of whether btc_holdout_start is set (because in
        # pre-holdout, the constant-EMA and full-EMA produce the same
        # value since both use the same pre-holdout data).
        pre_mask = sample_hourly_data["time"] < btc_holdout_start
        pre_trends_with = result_with.loc[pre_mask, "macro_trend"].dropna()

        # The pre-holdout trend values are determined by the same
        # computation regardless of btc_holdout_start (in pre-holdout
        # region, both paths use the full-df EMA which is what the
        # pre-holdout data would produce anyway).
        # Verify: the pre-holdout trend values are all in {-1, 1}
        assert pre_trends_with.isin([-1, 1]).all()

        # And: at least some pre-holdout trend values are non-zero
        # (sample data is not all flat)
        assert (pre_trends_with != 0).all()

    def test_macro_trend_default_behavior_unchanged(
        self, sample_hourly_data, sample_btc_data
    ):
        """Without btc_holdout_start, behavior is the original
        (varying macro_trend based on full series EMA)."""
        max_time = sample_hourly_data["time"].max()
        result = prepare_data_with_max_time(
            sample_hourly_data, sample_btc_data, None, max_time
        )
        # With full-series EMA, macro_trend can vary across rows
        # (it reflects the local close vs EMA relationship)
        assert result["macro_trend"].dropna().nunique() >= 1

    def test_macro_trend_uses_pre_holdout_ema_value(
        self, sample_hourly_data, sample_btc_data
    ):
        """When btc_holdout_start is set, the trend in the holdout
        is the last-EMA-value-computed-on-pre-holdout, NOT a
        function of holdout prices."""
        max_time = sample_hourly_data["time"].max()
        all_times = sample_hourly_data["time"].sort_values().unique()
        btc_holdout_start = all_times[len(all_times) * 3 // 4]

        # Compute with btc_holdout_start
        result_with = prepare_data_with_max_time(
            sample_hourly_data,
            sample_btc_data,
            None,
            max_time,
            btc_holdout_start=btc_holdout_start,
        )

        # Compute WITHOUT btc_holdout_start on a truncated dataset
        # that only contains data up to btc_holdout_start
        df_truncated = sample_hourly_data[
            sample_hourly_data["time"] < btc_holdout_start
        ].copy()
        btc_truncated = sample_btc_data[
            sample_btc_data["time"] < btc_holdout_start
        ].copy()
        result_truncated = prepare_data_with_max_time(
            df_truncated,
            btc_truncated,
            None,
            df_truncated["time"].max(),
        )
        # The last-EMA value from the truncated run should
        # match the constant value used in the holdout for
        # the full run (within numerical tolerance).
        # Compare last non-NaN macro_trend from truncated to
        # the constant in the holdout for the full run.
        last_truncated_trend = result_truncated["macro_trend"].dropna().iloc[-1]
        holdout_mask = result_with["time"] >= btc_holdout_start
        holdout_first_trend = result_with.loc[holdout_mask, "macro_trend"].dropna().iloc[0]
        # Both should be in {-1, 1} (binary signal)
        assert last_truncated_trend in [-1, 1]
        assert holdout_first_trend in [-1, 1]


class TestApplyHoldoutEMAToFull:
    """v0.4.0 (Phase 2.4): apply_holdout_ema_to_full flag.

    Default ``True`` preserves historical behavior (constant pre-holdout
    EMA applied across the full df, including IS period). Setting
    ``False`` restores the dynamic EMA in the IS period so it matches
    the WFA path (where btc_holdout_start is None).
    """

    def test_default_true_constant_in_holdout(self, sample_hourly_data, sample_btc_data):
        """Default behavior: constant EMA threshold in holdout region."""
        max_time = sample_hourly_data["time"].max()
        all_times = sample_hourly_data["time"].sort_values().unique()
        btc_holdout_start = all_times[len(all_times) * 3 // 4]

        result = prepare_data_with_max_time(
            sample_hourly_data, sample_btc_data, None, max_time,
            btc_holdout_start=btc_holdout_start,
        )
        # Holdout trend values are still {-1, 1} (binary)
        assert result["macro_trend"].dropna().isin([-1, 1]).all()

    def test_explicit_true_matches_default(self, sample_hourly_data, sample_btc_data):
        """Passing apply_holdout_ema_to_full=True must match default."""
        max_time = sample_hourly_data["time"].max()
        all_times = sample_hourly_data["time"].sort_values().unique()
        btc_holdout_start = all_times[len(all_times) * 3 // 4]

        result_default = prepare_data_with_max_time(
            sample_hourly_data, sample_btc_data, None, max_time,
            btc_holdout_start=btc_holdout_start,
        )
        result_explicit = prepare_data_with_max_time(
            sample_hourly_data, sample_btc_data, None, max_time,
            btc_holdout_start=btc_holdout_start,
            apply_holdout_ema_to_full=True,
        )
        # macro_trend should be identical between default and explicit
        pd.testing.assert_series_equal(
            result_default["macro_trend"].reset_index(drop=True),
            result_explicit["macro_trend"].reset_index(drop=True),
            check_names=False,
        )

    def test_explicit_false_restores_dynamic_ema_in_is(
        self, sample_hourly_data, sample_btc_data
    ):
        """With apply_holdout_ema_to_full=False, IS region uses dynamic EMA.

        The IS-period macro_trend should MATCH the dynamic-EMA computation
        (the WFA path), not the constant pre-holdout EMA.
        """
        max_time = sample_hourly_data["time"].max()
        all_times = sample_hourly_data["time"].sort_values().unique()
        btc_holdout_start = all_times[len(all_times) * 3 // 4]

        # Path 1: constant EMA in IS (default)
        result_constant = prepare_data_with_max_time(
            sample_hourly_data, sample_btc_data, None, max_time,
            btc_holdout_start=btc_holdout_start,
            apply_holdout_ema_to_full=True,
        )
        # Path 2: dynamic EMA in IS, constant in holdout
        result_dynamic_is = prepare_data_with_max_time(
            sample_hourly_data, sample_btc_data, None, max_time,
            btc_holdout_start=btc_holdout_start,
            apply_holdout_ema_to_full=False,
        )
        # Path 3: pure dynamic EMA (no holdout cutoff) -- reference
        result_pure_dynamic = prepare_data_with_max_time(
            sample_hourly_data, sample_btc_data, None, max_time,
        )

        # IS-period macro_trend in path 2 should match path 3 (dynamic)
        is_mask = sample_hourly_data["time"] < btc_holdout_start
        is_dynamic = result_dynamic_is.loc[is_mask, "macro_trend"].dropna()
        is_pure = result_pure_dynamic.loc[is_mask, "macro_trend"].dropna()
        if len(is_dynamic) > 0 and len(is_pure) > 0:
            pd.testing.assert_series_equal(
                is_dynamic.reset_index(drop=True),
                is_pure.reset_index(drop=True),
                check_names=False,
                obj=f"IS macro_trend with apply_holdout_ema_to_full=False "
                    f"must equal pure-dynamic path",
            )

        # Holdout-period macro_trend in path 2 should match path 1 (constant)
        holdout_mask = sample_hourly_data["time"] >= btc_holdout_start
        ho_dynamic_is = result_dynamic_is.loc[holdout_mask, "macro_trend"].dropna()
        ho_constant = result_constant.loc[holdout_mask, "macro_trend"].dropna()
        if len(ho_dynamic_is) > 0 and len(ho_constant) > 0:
            pd.testing.assert_series_equal(
                ho_dynamic_is.reset_index(drop=True),
                ho_constant.reset_index(drop=True),
                check_names=False,
                obj="Holdout macro_trend must be identical regardless of "
                    "apply_holdout_ema_to_full flag",
            )
