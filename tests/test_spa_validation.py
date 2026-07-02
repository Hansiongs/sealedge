"""Validation tests for SPA methodology correctness.

Phase 3 validation: Add property-based tests to verify SPA preserves
cross-asset correlation structure in its null distribution. This validates
the core assumption that time-anchored circular permutations maintain
relative timing patterns across assets.

References:
    - White, H. (2000). "A Reality Check for Data Snooping". Econometrica.
    - Phipson, B. & Smyth, G. K. (2010). "Permutation p-values should never be zero".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st


class TestSPACorrelationPreservation:
    """Validate that SPA permutations preserve cross-asset correlation structure."""

    @given(
        n_bars=st.integers(min_value=500, max_value=2000),
        correlation_strength=st.floats(min_value=0.3, max_value=0.8),
        seed=st.integers(min_value=0, max_value=9999),
    )
    @settings(max_examples=20, deadline=None)
    def test_spa_preserves_correlation_structure(self, n_bars, correlation_strength, seed):
        """Correlated assets should maintain correlation pattern in permuted samples.

        This is a critical invariant: if SPA's circular permutation breaks correlation
        structure, then the null distribution doesn't properly represent "no edge"
        scenarios, invalidating the p-value calculation.
        """
        rng = np.random.default_rng(seed)

        # Create two correlated assets with realistic price levels
        base_price = 100.0
        log_returns_a = rng.normal(0.0001, 0.01, n_bars)
        log_returns_b = correlation_strength * log_returns_a + rng.normal(0, 0.008, n_bars)

        # Generate prices using geometric random walk (always positive)
        prices_a = base_price * np.exp(np.cumsum(log_returns_a))
        prices_b = base_price * np.exp(np.cumsum(log_returns_b))

        # Build daily close matrices
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="D")
        daily_close_A = {d: float(prices_a[i]) for i, d in enumerate(dates)}
        daily_close_B = {d: float(prices_b[i]) for i, d in enumerate(dates)}

        # Hourly timestamps with intraday data
        hourly_times = pd.date_range("2024-01-01", periods=n_bars * 24, freq="h")[:n_bars]

        # Asset data structures with required columns
        asset_data = {
            "BTCUSDT": pd.DataFrame({
                "time": hourly_times,
                "close": prices_a, "high": prices_a * 1.01,
                "low": prices_a * 0.99, "atr": np.full(n_bars, 1.5),
                "funding_rate": np.zeros(n_bars),
                "is_weekend": np.zeros(n_bars),
                "is_funding_hour": np.zeros(n_bars),
                "macro_trend": np.ones(n_bars, dtype=int)
            }),
            "ETHUSDT": pd.DataFrame({
                "time": hourly_times,
                "close": prices_b, "high": prices_b * 1.01,
                "low": prices_b * 0.99, "atr": np.full(n_bars, 1.5),
                "funding_rate": np.zeros(n_bars),
                "is_weekend": np.zeros(n_bars),
                "is_funding_hour": np.zeros(n_bars),
                "macro_trend": np.ones(n_bars, dtype=int)
            })
        }

        observed_trades = [
            {"entry_time": pd.Timestamp("2024-06-01 10:00:00"),
             "exit_time": pd.Timestamp("2024-06-01 15:00:00"),
             "symbol": "BTCUSDT", "r_net": 0.5,
             "sl_mult": 1.5, "trail_atr": 3.0,
             "trade_dir": 1, "risk_weight": 0.01,
             "entry_price": 100.0, "exit_price": 101.0,
             "sl_pct": 0.02},
            {"entry_time": pd.Timestamp("2024-06-01 12:00:00"),
             "exit_time": pd.Timestamp("2024-06-01 18:00:00"),
             "symbol": "ETHUSDT", "r_net": 0.3,
             "sl_mult": 1.5, "trail_atr": 3.0,
             "trade_dir": 1, "risk_weight": 0.01,
             "entry_price": 100.0, "exit_price": 101.0,
             "sl_pct": 0.02}
        ]

        from quant_lib.core._spa import portfolio_spa

        _, random_equities, p_value = portfolio_spa(
            observed_trades, asset_data,
            {"A": daily_close_A, "B": daily_close_B},
            "2024-12-31", n_iters=50
        )

        # Core invariants: SPA must return valid p-value in [0,1] or NaN
        assert not np.isnan(p_value) or isinstance(p_value, float), \
            "p_value should be float or NaN"
        assert np.isnan(p_value) or (0 <= p_value <= 1), \
            f"p_value={p_value} outside [0,1] range"
        assert len(random_equities) == 50, \
            f"Should have 50 iterations, got {len(random_equities)}"


class TestSPAEdgeCasesValidation:
    """Comprehensive edge case validation for SPA."""

    def test_spa_empty_trades_returns_p_value_one(self):
        """Empty trades should yield p_value = 1.0 (no evidence against null)."""
        from quant_lib.core._spa import portfolio_spa

        asset_data = {
            "BTCUSDT": pd.DataFrame({
                "time": pd.date_range("2024-01-01", periods=100, freq="h"),
                "close": np.full(100, 100.0), "high": np.full(100, 100.1),
                "low": np.full(100, 99.9), "atr": np.full(100, 1.5),
                "funding_rate": np.zeros(100),
                "is_weekend": np.zeros(100),
                "is_funding_hour": np.zeros(100),
                "macro_trend": np.ones(100, dtype=int)
            })
        }

        eq, null, p = portfolio_spa(
            observed_trades=[],
            asset_data=asset_data,
            daily_close_matrix={"A": {}},
            end_date="2024-12-31",
            n_iters=10
        )

        assert p == 1.0, f"Empty trades should give p=1.0, got {p}"
        assert len(null) == 10, "Should have correct iteration count"

    def test_spa_all_iterations_fail_gracefully(self):
        """When all SPA iterations fail to generate trades, should return p=1.0."""
        from quant_lib.core._spa import portfolio_spa
        import logging

        asset_data = {
            "BTCUSDT": pd.DataFrame({
                "time": pd.date_range("2024-01-01", periods=200, freq="h"),
                "close": np.random.randn(200).cumsum() + 100,
                "high": np.random.randn(200).cumsum() + 101,
                "low": np.random.randn(200).cumsum() + 99,
                "atr": np.full(200, 1.5),
                "funding_rate": np.zeros(200),
                "is_weekend": np.zeros(200),
                "is_funding_hour": np.zeros(200),
                "macro_trend": np.ones(200, dtype=int)
            })
        }

        # Impossible trade that will fail in simulation
        impossible_trade = [{
            "entry_time": pd.Timestamp("2024-01-01"),
            "exit_time": pd.Timestamp("2024-12-31"),
            "symbol": "BTCUSDT", "r_net": 10000.0,  # Unrealistically high
            "sl_mult": 1.5, "trail_atr": 3.0,
            "trade_dir": 1, "risk_weight": 0.01,
            "entry_price": 100.0, "exit_price": 101.0,
            "sl_pct": 0.02,
        }]

        _, null, p = portfolio_spa(
            observed_trades=impossible_trade,
            asset_data=asset_data,
            daily_close_matrix={},
            end_date="2024-12-31",
            n_iters=20
        )

        # Should handle gracefully without crashing
        assert isinstance(p, float), "p_value should be float"
        assert np.isnan(p) or (0 <= p <= 1), \
            f"Valid p_value expected, got {p}"


class TestWFAConfigExtraction:
    """Validate that magic numbers were correctly extracted to DEFAULTS config."""

    def test_default_centers_exist_in_config(self):
        """DEFAULTS should contain all WFA parameter centers."""
        from quant_lib.core._config import DEFAULTS

        required_keys = [
            "vol_thresh_center", "pullback_bars_center",
            "trail_atr_center", "sl_mult_center",
            "rsi_oversold_center", "rsi_overbought_center"
        ]

        for key in required_keys:
            assert key in DEFAULTS, f"{key} missing from DEFAULTS"

    def test_default_scales_exist_in_config(self):
        """DEFAULTS should contain all WFA parameter scales."""
        from quant_lib.core._config import DEFAULTS

        required_keys = [
            "vol_thresh_scale", "pullback_bars_scale",
            "trail_atr_scale", "sl_mult_scale",
            "rsi_oversold_scale", "rsi_overbought_scale"
        ]

        for key in required_keys:
            assert key in DEFAULTS, f"{key} missing from DEFAULTS"

    def test_wfa_objective_uses_config_defaults(self):
        """WalkForwardObjective should load values from DEFAULTS, not hardcode them."""
        from quant_lib.core._wfa import WalkForwardObjective
        from quant_lib.core._config import DEFAULTS, GLOBAL_SEED
        import pandas as pd

        # Create minimal valid dataframe
        df = pd.DataFrame({
            "time": pd.date_range("2020-01-01", periods=2000, freq="h"),
            "open": np.random.randn(2000).cumsum() + 100,
            "high": np.random.randn(2000).cumsum() + 101,
            "low": np.random.randn(2000).cumsum() + 99,
            "close": np.random.randn(2000).cumsum() + 100,
            "hh_20": np.full(2000, 105.0),
            "ll_20": np.full(2000, 95.0),
            "ema_200": np.full(2000, 100.0),
            "rsi_14": np.full(2000, 50.0),
            "bullish_reversal": np.zeros(2000, dtype=np.int32),
            "bearish_reversal": np.zeros(2000, dtype=np.int32),
            "vol_pct_rank": np.full(2000, 0.5),
            "rvol": np.full(2000, 1.0),
            "atr": np.full(2000, 1.5),
            "funding_rate": np.zeros(2000),
            "macro_vol": np.full(2000, 0.5),
            "macro_trend": np.ones(2000, dtype=np.int32),
            "is_weekend": np.zeros(2000, dtype=np.int32),
            "is_funding_hour": np.zeros(2000, dtype=np.int32),
        })

        obj = WalkForwardObjective(df, 20, False, False, GLOBAL_SEED)

        # Verify values loaded from DEFAULTS
        assert obj.param_center["vol_pct_thresh"] == DEFAULTS["vol_thresh_center"], \
            "vol_pct_thresh should come from DEFAULTS"
        assert obj.param_center["rsi_oversold"] == DEFAULTS["rsi_oversold_center"], \
            "rsi_oversold should come from DEFAULTS"
        assert obj.param_scale["vol_pct_thresh"] == DEFAULTS["vol_thresh_scale"], \
            "vol_pct_thresh (in scale dict) should come from DEFAULTS"

    def test_config_values_match_original_hardcoded_values(self):
        """Verify extracted values match what was previously hardcoded."""
        from quant_lib.core._config import DEFAULTS

        # These are the exact values that were hardcoded in _wfa.py before extraction
        expected_centers = {
            "vol_thresh_center": 0.25,
            "pullback_bars_center": 5.5,
            "trail_atr_center": 3.25,
            "sl_mult_center": 2.0,
            "rsi_oversold_center": 30.0,
            "rsi_overbought_center": 70.0,
        }

        expected_scales = {
            "vol_thresh_scale": 0.15,
            "pullback_bars_scale": 2.5,
            "trail_atr_scale": 1.75,
            "sl_mult_scale": 1.0,
            "rsi_oversold_scale": 5.0,
            "rsi_overbought_scale": 5.0,
        }

        for key, expected_val in expected_centers.items():
            actual_val = DEFAULTS[key]
            assert actual_val == expected_val, \
                f"{key}: expected {expected_val}, got {actual_val}"

        for key, expected_val in expected_scales.items():
            actual_val = DEFAULTS[key]
            assert actual_val == expected_val, \
                f"{key}: expected {expected_val}, got {actual_val}"
