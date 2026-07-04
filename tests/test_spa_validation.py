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


class TestSPACorrelationPreservation:
    """Validate that SPA permutations preserve cross-asset correlation structure."""

    def test_spa_preserves_correlation_structure(self):
        """Cross-asset relative timing is preserved under circular permutation.

        This is the critical invariant that justifies the time-anchored
        null in ``portfolio_spa``: every iteration draws ONE shared
        ``anchor_offset`` (``_spa.py:199``) and shifts ALL observed
        trades by it -- each null trade's entry hour is
        ``(anchor_offset + relative_offset) % total_hours``. Because the
        *same* offset is added to every trade, the relative spacing
        between trades is preserved, which keeps cross-asset
        co-occurrence structure in the null distribution. A per-trade
        INDEPENDENT anchor (the regression this test guards against)
        would destroy it and silently invalidate the p-value.

        This is a PRODUCTION-observed regression test, not a spec test:
        it spies on ``simulate_trailing_stop_trade`` (the production
        symbol ``portfolio_spa`` actually calls per null trade) to
        recover the real null entry bar index ``idx`` for every (iter,
        trade) pair, converts ``idx`` to an entry timestamp, and
        asserts the relative-gap invariant against the production null
        timestamps. If a future change moves the ``anchor_offset`` draw
        inside the per-trade loop (per-trade independent anchors), the
        recovered null relative gaps no longer equal the observed gap
        and the test goes RED.

        Invariant (mod ``total_hours``): for every iteration, the two
        trades' null entry-hour gap equals the observed gap::

            (null_entry_B - null_entry_A) % total_hours
                == (rel_off_B - rel_off_A) % total_hours
        """
        from unittest.mock import patch
        from quant_lib.core import _spa as spa_mod
        from quant_lib.core._spa import portfolio_spa

        seed = 7
        gap_hours = 5.0
        n_bars = 2000
        n_iters = 30

        rng = np.random.default_rng(seed)
        log_returns_a = rng.normal(0.0001, 0.01, n_bars)
        log_returns_b = 0.6 * log_returns_a + rng.normal(0, 0.008, n_bars)
        prices_a = 100.0 * np.exp(np.cumsum(log_returns_a))
        prices_b = 100.0 * np.exp(np.cumsum(log_returns_b))

        daily_close_A = {d: float(prices_a[i]) for i, d in enumerate(
            pd.date_range("2024-01-01", periods=n_bars, freq="D"))}
        daily_close_B = {d: float(prices_b[i]) for i, d in enumerate(
            pd.date_range("2024-01-01", periods=n_bars, freq="D"))}

        hourly_times = pd.date_range("2024-01-01", periods=n_bars, freq="h")[:n_bars]
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

        first_entry = hourly_times[100]
        second_entry = first_entry + pd.Timedelta(hours=gap_hours)
        observed_trades = [
            {"entry_time": first_entry,
             "exit_time": first_entry + pd.Timedelta(hours=5),
             "symbol": "BTCUSDT", "r_net": 0.5,
             "sl_mult": 1.5, "trail_atr": 3.0,
             "trade_dir": 1, "risk_weight": 0.01,
             "entry_price": 100.0, "exit_price": 101.0,
             "sl_pct": 0.02},
            {"entry_time": second_entry,
             "exit_time": second_entry + pd.Timedelta(hours=5),
             "symbol": "ETHUSDT", "r_net": 0.3,
             "sl_mult": 1.5, "trail_atr": 3.0,
             "trade_dir": 1, "risk_weight": 0.01,
             "entry_price": 100.0, "exit_price": 101.0,
             "sl_pct": 0.02}
        ]

        # Observed relative gap (hours), the quantity the null must
        # preserve.
        first_entry_ts = min(t["entry_time"] for t in observed_trades)
        relative_offsets = [
            (t["entry_time"] - first_entry_ts).total_seconds() / 3600.0
            for t in observed_trades
        ]
        observed_rel_gap = relative_offsets[1] - relative_offsets[0]

        global_start = max(v["time"].iloc[0] for v in asset_data.values())
        # end_date sits just past the last data bar (2024-03-24 07:00) so
        # the data-gap region is < 7 days -- otherwise portfolio_spa's
        # searchsorted clamps anchors that land in the gap via
        # ``idx % (max_valid_idx+1)`` (see _spa.py:214) and the recovered
        # null entry timestamps no longer reflect the true anchor hours,
        # which would corrupt the relative-timing invariant for a reason
        # unrelated to the shared-anchor property under test.
        _data_end_ts = pd.Timestamp("2024-03-25")
        _min_asset_end = min(v["time"].iloc[-1] for v in asset_data.values())
        global_end = max(_data_end_ts, _min_asset_end)
        total_hours = (global_end - global_start) / np.timedelta64(1, "h")

        # --- Spy on the per-null-trade simulation call ---
        # ``portfolio_spa`` calls ``simulate_trailing_stop_trade`` (imported
        # into the ``_spa`` module namespace) inside the per-iteration,
        # per-trade loop. We wrap the real symbol so PnL behaviour is
        # unchanged, but record ``idx`` (the null entry bar index) for
        # each call. Calls are emitted in order: all of iteration k's
        # trades (trade 0, then trade 1) before iteration k+1, because
        # the outer loop is ``for it in range(n_iters)`` and the inner
        # is ``for i, t in enumerate(observed_trades)``. With
        # ``n_bars=2000`` and durations <= 5h, ``max_valid_idx`` is
        # always > 0, so no pre-call ``continue`` truncates an iteration
        # below two calls -- every iteration emits exactly two calls.
        recorded_idx: list[int] = []
        real_simulate = spa_mod.simulate_trailing_stop_trade

        def _spy(high, low, close, atr, funding, is_fund_h, is_wknd,
                 macro_trend, idx, direction, sl_mult, trail_atr,
                 bailout, fee_taker, weekend_penalty, stress_mult,
                 rand_draw, trend_aligned, trend_counter):
            recorded_idx.append(int(idx))
            return real_simulate(
                high, low, close, atr, funding, is_fund_h, is_wknd,
                macro_trend, idx, direction, sl_mult, trail_atr,
                bailout, fee_taker, weekend_penalty, stress_mult,
                rand_draw, trend_aligned, trend_counter)

        with patch.object(spa_mod, "simulate_trailing_stop_trade", _spy):
            _, _, p_value = portfolio_spa(
                observed_trades, asset_data,
                {"A": daily_close_A, "B": daily_close_B},
                end_date="2024-03-25", n_iters=n_iters, rng_seed=int(seed),
            )

        # Precondition: the run was non-degenerate (no early NaN return
        # from the degenerate-anchor guard), otherwise the spy captured
        # no data and the invariant is unexercised.
        assert not np.isnan(p_value), (
            "Setup precondition failed: portfolio_spa returned NaN "
            "(degenerate-anchor guard fired). Adjust the trade spacing "
            "/ n_bars so the anchor space is non-degenerate, otherwise "
            "the relative-timing invariant below is never exercised."
        )
        # Each iteration emits exactly two ``simulate_trailing_stop_trade``
        # calls (one per observed trade); n_iters iterations => 2*n_iters.
        #
        # Phase 7: this ASSUMPTION holds only on the LEGACY path
        # (``trial_r_nets is None`` -- uniform time-anchored permutation).
        # The Hansen-literal path (Phase 5, ``recenter_policy="hansen_literal"``
        # + ``trial_r_nets`` + ``return_statistics=True``) is NUMPY-ONLY on
        # ``pnl_array``s -- it does NOT drive ``anchor_offset`` / ``rand_draw``
        # (which use ``rng_spa`` here) and emits ZERO ``simulate_*`` calls.
        # Hence the Hansen path ALSO leaves ``len(recorded_idx) == 2*n_iters``
        # intact by construction (see ``test_hansen_path_emits_no_extra_*``).
        # The sibling test guards against a future Phase-5 regression where
        # the Hansen block accidentally touches the simulate path -- that
        # would shift ``rng_spa`` and break this legacy assumption under
        # ``trial_r_nets=None``. DO NOT relax the assertion: if a sibling
        # spy FAILS here, fix the Phase 5 implementation, not the assert.
        assert len(recorded_idx) == 2 * n_iters, (
            f"Spy captured {len(recorded_idx)} calls, expected "
            f"{2 * n_iters}. If a future change adds an early pre-call "
            f"`continue` path that truncates iterations, this grouping "
            f"assumption breaks and the test must pair on iteration "
            f"labels instead."
        )

        # --- Verify the relative-gap invariant against PRODUCTION null
        # entry timestamps recovered from the spy's recorded bar idx. ---
        # Trade i's symbol is observed_trades[i]["symbol"]; we read its
        # entry timestamp from that asset's ``time`` column at the
        # recorded idx. Pairs are (call a=trade0, call a+1=trade1) per
        # iteration.
        invariant_holds = 0
        for k in range(n_iters):
            idx_a = recorded_idx[2 * k]
            idx_b = recorded_idx[2 * k + 1]
            t_a = asset_data["BTCUSDT"]["time"].iloc[idx_a]
            t_b = asset_data["ETHUSDT"]["time"].iloc[idx_b]
            # Hours since global_start, on the same axis as total_hours.
            h_a = (t_a - global_start) / np.timedelta64(1, "h")
            h_b = (t_b - global_start) / np.timedelta64(1, "h")
            null_rel_gap_mod = (float(h_b) - float(h_a)) % total_hours
            if abs(null_rel_gap_mod - (observed_rel_gap % total_hours)) < 1e-9:
                invariant_holds += 1

        assert invariant_holds == n_iters, (
            f"Cross-asset relative-timing invariant broken: only "
            f"{invariant_holds}/{n_iters} production iterations preserved "
            f"the observed gap of {observed_rel_gap}h (mod "
            f"{total_hours:.0f}h). SPA's circular permutation MUST draw "
            f"ONE shared anchor offset per iteration (_spa.py:199) so "
            f"cross-asset co-occurrence is preserved. A per-trade "
            f"independent anchor draw would destroy this -- the null "
            f"would no longer represent the 'no edge' scenario the "
            f"p-value assumes, silently invalidating claim #3."
        )

    def test_hansen_path_emits_no_extra_simulate_calls(self):
        """Spy-gating infrastructure (Phase 7) -- NOT one of the 6
        Hansen calibration tests.

        Documents that the Hansen-literal SPA path
        (``recenter_policy="hansen_literal"`` + ``trial_r_nets`` +
        ``return_statistics=True``) is NUMPY-ONLY on ``pnl_array``s: it
        resamples pre-collected trial IS PnL series via the
        Politis-Romano stationary block bootstrap (Phase 1 primitive) and
        does NOT drive ``anchor_offset`` / ``rand_draw`` (legacy ``rng_spa``
        consumers) or call ``simulate_trailing_stop_trade`` /
        ``simulate_full_portfolio``. Hence a fresh spy on
        ``simulate_trailing_stop_trade`` records the SAME number of
        calls as the legacy circular-permutation path on the SAME
        observed trades (n_iters * n_observed_trades = ``2*n_iters``
        here).

        If this FAILS, Phase 5's Hansen block accidentally invoked
        ``simulate_*`` -- that would shift the legacy ``rng_spa``
        stream and break the existing ``len(recorded_idx)==2*n_iters``
        invariant under ``trial_r_nets=None``. DO NOT relax the
        assertion: fix the Phase 5 implementation.
        """
        from unittest.mock import patch
        from quant_lib.core import _spa as spa_mod
        from quant_lib.core._spa import portfolio_spa

        seed = 11
        n_bars = 800
        n_iters = 12
        n_observed = 2

        rng = np.random.default_rng(seed)
        log_returns = rng.normal(0.0001, 0.01, n_bars)
        prices = 100.0 * np.exp(np.cumsum(log_returns))
        hourly_times = pd.date_range("2024-01-01", periods=n_bars, freq="h")
        daily_close = {
            d: float(prices[i])
            for i, d in enumerate(pd.date_range("2024-01-01", periods=n_bars, freq="D"))
        }
        asset_data = {
            "BTCUSDT": pd.DataFrame({
                "time": hourly_times,
                "close": prices, "high": prices * 1.01,
                "low": prices * 0.99, "atr": np.full(n_bars, 1.5),
                "funding_rate": np.zeros(n_bars),
                "is_weekend": np.zeros(n_bars),
                "is_funding_hour": np.zeros(n_bars),
                "macro_trend": np.ones(n_bars, dtype=int),
            }),
        }
        first_entry = hourly_times[100]
        observed_trades = [
            {
                "entry_time": first_entry,
                "exit_time": first_entry + pd.Timedelta(hours=5),
                "symbol": "BTCUSDT", "r_net": 0.5,
                "sl_mult": 1.5, "trail_atr": 3.0,
                "trade_dir": 1, "risk_weight": 0.01,
                "entry_price": 100.0, "exit_price": 101.0,
                "sl_pct": 0.02,
            }
            for _ in range(n_observed)
        ]
        # Synthetic IS PnL arrays for the Hansen null (claim #3).
        trials = [
            np.random.default_rng(seed + k).normal(0.0, 0.1, 30)
            for k in range(5)
        ]
        recorded_idx = []

        def spy_simulate(*args, **kwargs):
            recorded_idx.append(kwargs.get("idx", args[8]))
            return 0, 0.0, 0.0, 1.0

        with patch.object(spa_mod, "simulate_trailing_stop_trade", side_effect=spy_simulate):
            result = portfolio_spa(
                observed_trades, asset_data,
                {"BTCUSDT": daily_close},
                end_date="2024-03-25",
                n_iters=n_iters,
                rng_seed=seed,
                trial_r_nets=trials,
                recenter_policy="hansen_literal",
                return_statistics=True,
            )
        # Hansen path emits ZERO extra simulate_* calls beyond the
        # legacy 2*n_iters. The Hansen block resamples pnl arrays in
        # numpy-only (Phase 1 primitive) on the final return path,
        # AFTER the legacy SPA loop has produced its 2*n_iters calls.
        assert len(recorded_idx) == 2 * n_iters, (
            f"Hansen path emitted unexpected simulate_* calls: "
            f"saw {len(recorded_idx)}, expected {2 * n_iters} (matching "
            f"the legacy 2*n_iters). This is the Phase 5 spy-invariant "
            f"guard -- a Hansen block that touches the simulate path "
            f"would shift the legacy rng_spa stream under "
            f"trial_r_nets=None and silently invalidate the claim #3 "
            f"anti-data-snooping null."
        )
        # And the return is the 4-tuple opt-in shape with stats dict.
        assert len(result) == 4
        assert "p_hansen" in result[3]
        assert "p_naive" in result[3]


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
