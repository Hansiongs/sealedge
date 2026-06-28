"""Tests for core configuration -- STATIC and DEFAULTS dicts integrity.

After Phase 4 cleanup:
- STATIC: infrastructure constants (rarely change)
- DEFAULTS: per-experiment defaults (mirror StrategyConfig)
- Removed: asset_risk_weights, asset_baseline_trades, global_position_limit
"""

from quant_lib.core._config import DEFAULTS, GLOBAL_SEED, STATIC, WARMUP_BARS


class TestStaticConfig:
    """Validate infrastructure STATIC dict (post-Phase-4)."""

    def test_global_seed_is_integer(self):
        assert isinstance(GLOBAL_SEED, int)
        assert GLOBAL_SEED == 42

    def test_warmup_bars_positive(self):
        assert WARMUP_BARS >= 750

    def test_wfa_purge_days_assertion(self):
        assert STATIC["wfa_purge_days"] >= 30

    def test_bootstrap_params_valid(self):
        assert STATIC["bootstrap_n_sim"] >= 100
        assert STATIC["bootstrap_block_size_min"] >= 5

    def test_bootstrap_block_size_max_greater_than_min(self):
        assert STATIC["bootstrap_block_size_max"] > STATIC["bootstrap_block_size_min"]

    def test_spa_equity_warn_threshold_positive(self):
        assert STATIC["spa_equity_warn_threshold_usd"] > 0.0

    def test_infrastructure_keys_only(self):
        """STATIC should only contain infrastructure constants.

        Per-experiment config lives in DEFAULTS / StrategyConfig.
        This is a property check (not a literal-set comparison) so
        that adding a new infrastructure key doesn't require a test
        update.
        """
        for key in STATIC:
            # Per-experiment keys (moved to DEFAULTS in Phase 4) must
            # not appear here. The blocked-prefix list mirrors the
            # category names used in StrategyConfig.
            assert not key.startswith("default_"), (
                f"STATIC key '{key}' looks like a per-experiment default; "
                f"should live in DEFAULTS / StrategyConfig"
            )

    def test_removed_keys_not_in_static(self):
        """Removed keys (asset_risk_weights, asset_baseline_trades,
        global_position_limit) must NOT be in STATIC anymore.
        """
        for removed in (
            "asset_risk_weights",
            "asset_baseline_trades",
            "global_position_limit",
        ):
            assert removed not in STATIC, (
                f"{removed} should be removed from STATIC (Phase 4 cleanup)"
            )


class TestDefaultsConfig:
    """Validate DEFAULTS dict (per-experiment defaults)."""

    def test_search_space_has_all_params(self):
        required = {"vol_pct_thresh", "pullback_bars", "trail_atr", "sl_mult"}
        assert required.issubset(DEFAULTS["search_space"].keys())

    def test_search_space_bounds_valid(self):
        ss = DEFAULTS["search_space"]
        for name, (lo, hi) in ss.items():
            assert lo < hi, f"{name}: low={lo} >= high={hi}"

    def test_key_numeric_params_in_range(self):
        """Per-experiment params are valid."""
        # fee_taker stays in STATIC (infrastructure)
        assert 0 < STATIC["fee_taker"] < 1
        # leverage, global_position_limit, wfa_trials_per_fold moved to DEFAULTS
        assert DEFAULTS["leverage"] >= 1
        assert DEFAULTS["global_position_limit"] >= 1
        assert DEFAULTS["wfa_trials_per_fold"] > 0

    def test_reg_lambda_non_negative(self):
        assert DEFAULTS["reg_lambda"] >= 0.0

    def test_weekend_penalty_ge_1(self):
        assert DEFAULTS["weekend_liquidity_penalty"] >= 1.0

    def test_stress_mult_ge_1(self):
        assert DEFAULTS["stress_test_multiplier"] >= 1.0

    def test_stress_test_multiplier_value(self):
        """Production stress_test_multiplier must be 2.0 (not 2.5).

        0.2.2 fix: previously _spa.py and tools/stats.py hardcoded 2.5
        while production used DEFAULTS=2.0. Now SPA/tools default to
        DEFAULTS too, so this guard ensures they stay in sync.
        """
        assert DEFAULTS["stress_test_multiplier"] == 2.0

    def test_pf_weight_clamp_keys(self):
        assert "pf_weight_clamp_floor" in DEFAULTS
        assert "pf_weight_clamp_ceiling" in DEFAULTS
        assert 0 < DEFAULTS["pf_weight_clamp_floor"] < 1.0
        assert DEFAULTS["pf_weight_clamp_floor"] < DEFAULTS["pf_weight_clamp_ceiling"]
        assert DEFAULTS["pf_weight_clamp_ceiling"] > 1.0

    def test_pf_decay_halflife_positive(self):
        assert DEFAULTS["pf_decay_halflife_folds"] >= 1

    def test_pf_min_trades_for_weight_positive(self):
        assert DEFAULTS["pf_min_trades_for_weight"] >= 1

    def test_default_risk_per_pair_positive(self):
        """DEFAULTS["default_risk_per_pair"] is the new replacement for
        the removed asset_risk_weights dict."""
        assert DEFAULTS["default_risk_per_pair"] > 0
        assert DEFAULTS["default_risk_per_pair"] < 0.1  # sanity bound

    def test_default_expected_trades_positive(self):
        """DEFAULTS["default_expected_trades_per_year"] replaces the
        removed asset_baseline_trades dict."""
        assert DEFAULTS["default_expected_trades_per_year"] > 0

    def test_search_space_pb_has_rsi_keys(self):
        ss_pb = DEFAULTS["search_space_pb"]
        assert "rsi_oversold" in ss_pb
        assert "rsi_overbought" in ss_pb
