"""Tests for statistical testing utilities — PSR, FDR, p-value labeling."""

import numpy as np

from quant_lib.core._testing import (
    prob_sharpe_ratio,
    fdr_correction,
    label_p_value,
)


class TestProbSharpeRatio:
    def test_positive_returns(self):
        # Seed for deterministic test (otherwise 500-sample PSR varies)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 500)
        sr, psr = prob_sharpe_ratio(returns, benchmark=0.0, annualize=True)
        assert np.isfinite(sr)
        assert 0 <= psr <= 1

    def test_negative_returns_low_psr(self):
        # Seed for deterministic test. With 500 samples from N(-0.001, 0.02),
        # a single random draw can occasionally give a sample mean close to
        # zero, making PSR > 0.5. Seeding makes the test reproducible.
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.001, 0.02, 500)
        sr, psr = prob_sharpe_ratio(returns, benchmark=0.0, annualize=True)
        assert sr < 0
        assert psr < 0.5

    def test_insufficient_data(self):
        returns = np.array([0.01, 0.02])
        sr, psr = prob_sharpe_ratio(returns, benchmark=0.0, annualize=True)
        assert np.isnan(sr)
        assert np.isnan(psr)

    def test_zero_std(self):
        returns = np.ones(50) * 0.01
        sr, psr = prob_sharpe_ratio(returns, benchmark=0.0, annualize=True)
        assert np.isnan(sr)
        assert np.isnan(psr)

    def test_benchmark_effect(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.02, 500)
        _, psr_low = prob_sharpe_ratio(returns, benchmark=0.001, annualize=True)
        _, psr_high = prob_sharpe_ratio(returns, benchmark=0.0001, annualize=True)
        # Lower benchmark should give higher PSR
        if not np.isnan(psr_low) and not np.isnan(psr_high):
            assert psr_high >= psr_low

    def test_annualize_flag(self):
        rng = np.random.default_rng(99)
        daily_ret = rng.normal(0.0005, 0.02, 500)
        sr_ann, _ = prob_sharpe_ratio(daily_ret, benchmark=0.0, annualize=True)
        sr_daily, _ = prob_sharpe_ratio(daily_ret, benchmark=0.0, annualize=False)
        if np.isfinite(sr_ann) and np.isfinite(sr_daily):
            assert abs(sr_ann) > abs(sr_daily)  # annualized SR is scaled up


class TestFDRCorrection:
    def test_all_null(self):
        p_vals = np.array([0.5, 0.7, 0.3, 0.8, 0.6])
        rejected, corrected = fdr_correction(p_vals, alpha=0.05)
        assert not rejected.any()
        assert len(corrected) == len(p_vals)

    def test_all_significant(self):
        p_vals = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        rejected, corrected = fdr_correction(p_vals, alpha=0.05)
        assert rejected.all()
        assert (corrected <= 0.05).all()

    def test_mixed_significance(self):
        p_vals = np.array([0.001, 0.5, 0.002, 0.7, 0.03])
        rejected, corrected = fdr_correction(p_vals, alpha=0.05)
        assert rejected.sum() > 0
        assert rejected.sum() < len(p_vals)

    def test_no_input(self):
        p_vals = np.array([])
        rejected, corrected = fdr_correction(p_vals, alpha=0.05)
        assert len(rejected) == 0
        assert len(corrected) == 0

    def test_single_value(self):
        p_vals = np.array([0.01])
        rejected, corrected = fdr_correction(p_vals, alpha=0.05)
        assert rejected[0]
        assert corrected[0] <= 0.05

    def test_adjusted_values_not_exceeding_1(self):
        p_vals = np.array([0.9, 0.95, 0.8])
        _, corrected = fdr_correction(p_vals, alpha=0.05)
        assert (corrected <= 1.0).all()
        assert (corrected >= 0).all()

    def test_output_shapes_match(self):
        p_vals = np.array([0.01, 0.02, 0.03, 0.04])
        rejected, corrected = fdr_correction(p_vals, alpha=0.05)
        assert rejected.shape == (4,)
        assert corrected.shape == (4,)


class TestLabelPValue:
    def test_nan_returns_unreliable(self):
        label, conf, interp = label_p_value(None)
        assert "UNRELIABLE" in label

        label, conf, interp = label_p_value(float("nan"))
        assert "UNRELIABLE" in label

    def test_prod_level(self):
        label, conf, interp = label_p_value(0.001)
        assert "PROD" in label

    def test_trade_level(self):
        label, conf, interp = label_p_value(0.03)
        assert "TRADE" in label

    def test_watch_level(self):
        label, conf, interp = label_p_value(0.10)
        assert "WATCH" in label

    def test_research_level(self):
        label, conf, interp = label_p_value(0.20)
        assert "RESEARCH" in label

    def test_no_edge_level(self):
        label, conf, interp = label_p_value(0.50)
        assert "NO EDGE" in label

    def test_boundary_005(self):
        label, _, _ = label_p_value(0.049)
        assert "TRADE" in label

        label, _, _ = label_p_value(0.051)
        assert "WATCH" in label

    def test_boundary_015(self):
        label, _, _ = label_p_value(0.149)
        assert "WATCH" in label

        label, _, _ = label_p_value(0.151)
        assert "RESEARCH" in label

    # --- Phase 3.1 A2: context-specific thresholds ---

    def test_spa_context_stricter_thresholds(self):
        """SPA context uses ~2x stricter thresholds than mean_r.

        p=0.02 is TRADE in mean_r (0.005-0.05) but TRADE in spa (0.0025-0.025).
        p=0.03 is TRADE in mean_r (0.005-0.05) but WATCH in spa (0.025-0.075).
        """
        # p=0.02: mean_r -> TRADE, spa -> TRADE
        mean_r_label, _, _ = label_p_value(0.02, context="mean_r")
        spa_label, _, _ = label_p_value(0.02, context="spa")
        assert "TRADE" in mean_r_label
        assert "TRADE" in spa_label

        # p=0.03: mean_r -> TRADE, spa -> WATCH
        mean_r_label, _, _ = label_p_value(0.03, context="mean_r")
        spa_label, _, _ = label_p_value(0.03, context="spa")
        assert "TRADE" in mean_r_label
        assert "WATCH" in spa_label

    def test_spa_context_prod_threshold(self):
        """SPA context requires p < 0.0025 for PROD (vs 0.005 for mean_r)."""
        # p=0.003: mean_r -> PROD (0.003 < 0.005), spa -> TRADE (0.003 in [0.0025, 0.025))
        # This asymmetry shows SPA requires stronger evidence for PROD.
        mean_r_label, _, _ = label_p_value(0.003, context="mean_r")
        spa_label, _, _ = label_p_value(0.003, context="spa")
        assert "PROD" in mean_r_label
        assert "TRADE" in spa_label

        # p=0.001: both contexts -> PROD
        mean_r_label, _, _ = label_p_value(0.001, context="mean_r")
        spa_label, _, _ = label_p_value(0.001, context="spa")
        assert "PROD" in mean_r_label
        assert "PROD" in spa_label

    def test_spa_context_no_edge_threshold(self):
        """SPA context: NO EDGE at p >= 0.15 (vs 0.30 for mean_r)."""
        # p=0.20: mean_r -> RESEARCH, spa -> NO EDGE
        mean_r_label, _, _ = label_p_value(0.20, context="mean_r")
        spa_label, _, _ = label_p_value(0.20, context="spa")
        assert "RESEARCH" in mean_r_label
        assert "NO EDGE" in spa_label

    def test_unknown_context_falls_back_to_mean_r(self, caplog):
        """Unknown context strings fall back to mean_r tiers with a warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            label, conf, interp = label_p_value(0.03, context="bogus")
        # Falls back to mean_r behavior
        assert "TRADE" in label
        # Logs a warning about unknown context
        assert any("unknown context" in m for m in caplog.messages)

    def test_nan_with_spa_context(self):
        """NaN p-value with spa context returns UNRELIABLE (with spa mention)."""
        label, conf, interp = label_p_value(float("nan"), context="spa")
        assert "UNRELIABLE" in label
        assert "spa" in interp.lower()
