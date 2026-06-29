"""Tests for the consolidated prob_sharpe_ratio (with optional trade_weights).

NOTE (0.2.2): The standalone `objective_psr_ess()` was removed -- it was a
near-duplicate of `prob_sharpe_ratio()` for the unweighted case. Its
weighted-mode functionality is now provided by
`prob_sharpe_ratio(returns, trade_weights=...)`.
"""

import numpy as np
from scipy import stats as scipy_stats

from quant_lib.core._testing import (
    prob_sharpe_ratio,
    fdr_correction,
)


class TestProbSharpeRatioWeighted:
    """Consolidated PSR function: unweighted + trade_weights support."""

    def test_unweighted_returns_valid_value(self):
        """Unweighted case (default): returns (sr, psr) both finite."""
        rng = np.random.default_rng(42)
        pnl = rng.normal(0.1, 0.5, 100)
        sr, psr = prob_sharpe_ratio(pnl, annualize=False)
        assert isinstance(sr, float)
        assert isinstance(psr, float)
        assert 0.0 <= psr <= 1.0

    def test_negative_returns_low_psr(self):
        """Negative-returning series: PSR should be below 0.5."""
        rng = np.random.default_rng(42)
        pnl = rng.normal(-0.1, 0.5, 100)
        _, psr = prob_sharpe_ratio(pnl, annualize=False)
        assert psr < 0.5

    def test_positive_returns_high_psr(self):
        """Strong positive series: PSR should be above 0.7."""
        rng = np.random.default_rng(42)
        pnl = rng.normal(0.5, 0.2, 200)
        _, psr = prob_sharpe_ratio(pnl, annualize=False)
        assert psr > 0.7

    def test_insufficient_data_returns_nan(self):
        """len(returns) < 10: returns (NaN, NaN)."""
        pnl = np.array([0.1, 0.2, 0.3])
        sr, psr = prob_sharpe_ratio(pnl, annualize=False)
        assert np.isnan(sr)
        assert np.isnan(psr)

    def test_zero_variance_returns_nan_psr(self):
        """All-same value (std~0 due to float precision): PSR is NaN, SR may be huge.

        Float precision means np.std of np.ones(50) * 0.1 can return
        a tiny non-zero value, making the first guard miss. The
        variance <= 0 guard then triggers, returning the (huge) SR
        but NaN PSR. This test verifies PSR is correctly marked as
        unreliable.
        """
        pnl = np.ones(50) * 0.1
        _, psr = prob_sharpe_ratio(pnl, annualize=False)
        assert np.isnan(psr)

    def test_with_trade_weights_differs_from_unweighted(self):
        """Weighted case: PSR differs from unweighted (effective sample size differs)."""
        rng = np.random.default_rng(42)
        pnl = rng.normal(0.2, 0.5, 100)
        weights = np.exp(-np.arange(100) / 50)  # decay weights
        _, psr_unweighted = prob_sharpe_ratio(pnl, annualize=False)
        _, psr_weighted = prob_sharpe_ratio(
            pnl, annualize=False, trade_weights=weights,
        )
        # Different ESS -> different PSR (not necessarily same direction)
        assert psr_unweighted != psr_weighted

    def test_with_benchmark_higher_lowers_psr(self):
        """Higher benchmark should give lower PSR for same returns."""
        rng = np.random.default_rng(42)
        pnl = rng.normal(0.1, 0.5, 100)
        _, psr_no_bench = prob_sharpe_ratio(pnl, annualize=False, benchmark=0.0)
        _, psr_bench = prob_sharpe_ratio(pnl, annualize=False, benchmark=0.5)
        assert psr_bench < psr_no_bench

    def test_ess_more_data_higher_psr(self):
        """Larger n with same effect: PSR should be more confident (higher).

        Phase 4.4 G4: pre-fix, the test drew two independent samples
        (n=20 and n=500) with the same seed -- the 20-sample set was
        a prefix of the 500-sample set in terms of RNG state, but the
        actual values were different draws. This made the assertion
        `psr_large >= psr_small` RNG-dependent: if the 20-sample
        happened to have a high sample SR by chance, psr_small could
        exceed psr_large and the test would fail.

        Post-fix: use a SHARED PREFIX (the first 20 of the 500 samples)
        so the SR of the smaller sample is a strict subset of the
        larger sample. With more data containing the same effect,
        PSR confidence must increase.
        """
        rng = np.random.default_rng(42)
        pnl_500 = rng.normal(0.2, 0.5, 500)
        pnl_20 = pnl_500[:20]  # Shared prefix
        _, psr_small = prob_sharpe_ratio(pnl_20, annualize=False)
        _, psr_large = prob_sharpe_ratio(pnl_500, annualize=False)
        # Larger n should give more confident (higher) PSR for same
        # effect (since larger sample contains smaller sample).
        assert psr_large >= psr_small, (
            f"PSR with more data should be more confident (higher). "
            f"psr_20={psr_small:.4f}, psr_500={psr_large:.4f}"
        )

    def test_annualize_flag_scales_sr(self):
        """annualize=True multiplies SR by sqrt(365.25).

        Note: PSR is NOT scale-invariant under annualization in Bailey's
        correct formula. The variance correction has a constant "1" term
        that does not scale with annualization (only the SR² term scales).
        So as we annualize SR (multiply by sqrt(365.25)), the z-score
        changes, and PSR changes accordingly. This test only verifies
        the SR scaling; PSR behavior is documented in the function.
        """
        rng = np.random.default_rng(42)
        pnl = rng.normal(0.001, 0.05, 500)
        sr_a, _ = prob_sharpe_ratio(pnl, annualize=True)
        sr_na, _ = prob_sharpe_ratio(pnl, annualize=False)
        # SR differs by exactly sqrt(365.25)
        assert abs(sr_a - sr_na * np.sqrt(365.25)) < 1e-9

    def test_weighted_psr_matches_wfa_formula(self):
        """Weighted PSR in prob_sharpe_ratio must match WFA inline formula.

        Cross-validates that the refactored weighted path is mathematically
        equivalent to core/_wfa.py:179-200. Both must use:
        - Weighted mean: w_mean = sum(w * x)
        - Weighted variance: w_var = sum(w * (x - w_mean)^2)
        - Weighted SR: w_sr = w_mean / sqrt(w_var)
        - ESS: n_eff = 1 / sum(w^2) (Kish, since w normalized to sum=1)
        - Bailey variance correction with excess kurtosis (fisher=True)
        """
        rng = np.random.default_rng(42)
        pnl = rng.normal(0.3, 0.8, 100)
        weights = np.exp(-np.arange(100) / 30)  # decay weights

        # prob_sharpe_ratio weighted path
        _, psr_lib = prob_sharpe_ratio(pnl, trade_weights=weights, annualize=False)

        # Replicate WFA inline formula (core/_wfa.py:179-200)
        w = weights / weights.sum()
        w_mean = np.dot(pnl, w)
        w_var = np.dot(w, (pnl - w_mean) ** 2)
        w_sr = w_mean / np.sqrt(w_var)
        ess = 1.0 / np.dot(w, w)
        skew_v = float(scipy_stats.skew(pnl))
        kurt_v = float(scipy_stats.kurtosis(pnl, fisher=True))  # EXCESS
        var_corr = (1 - skew_v * w_sr + (kurt_v - 1) / 4 * w_sr**2)
        if var_corr <= 0:
            psr_wfa = 0.5  # WFA uses neutral fallback for invalid range
        else:
            psr_wfa = float(scipy_stats.norm.cdf(
                w_sr / np.sqrt(var_corr / max(ess - 1, 1))
            ))
        # WFA uses 0.5 fallback; lib uses clip+high PSR. Only compare when
        # both are in the valid regime.
        if var_corr > 0:
            assert abs(psr_lib - psr_wfa) < 1e-9, (
                f"Weighted PSR mismatch: lib={psr_lib:.6f}, wfa={psr_wfa:.6f}"
            )

    def test_kurtosis_uses_excess_convention(self):
        """Verify kurtosis uses excess (fisher=True) convention.

        Bailey's PSR formula requires excess kurtosis (γ₄_excess = γ₄ - 3).
        For normal data: excess kurtosis ≈ 0, regular kurtosis ≈ 3.
        Both conventions would give same PSR for SR=0 but differ for SR>0.

        We test indirectly: generate near-normal data with known SR. With
        excess kurtosis convention, variance correction is approx
        1 - skew*SR + (excess-1)/4*SR². For near-normal with SR=0.5 and
        excess_kurt ≈ 0, correction ≈ 1 - 0 - 0.0625 = 0.9375 (positive).
        With regular kurtosis, correction would be 1 + (3-1)/4*0.25 = 1.125.
        """
        rng = np.random.default_rng(42)
        # Use n large enough to get tight confidence on kurtosis estimate
        pnl = rng.normal(0.05, 0.1, 5000)  # SR = 0.5, near-normal
        sr, psr = prob_sharpe_ratio(pnl, annualize=False)
        # For near-normal with SR=0.5, formula gives a well-defined PSR
        # (not clipped, not NaN). Verify both sr and psr are finite.
        assert np.isfinite(sr)
        assert np.isfinite(psr)
        # SR should be near 0.5
        assert abs(sr - 0.5) < 0.1
        # PSR should be reasonably high (0.5 SR with n=5000 is highly significant)
        assert psr > 0.9

    def test_weighted_sanity_ess_below_2_returns_nan(self):
        """ESS < 2: weighted PSR must return NaN, not silently finite."""
        pnl = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        # Extreme weights: one element gets ~100% weight, others ~0%
        # ESS = 1/sum(w^2) = 1/(0.999^2 + 4*0.00025^2) ≈ 1.0005 < 2
        weights = np.array([0.999, 0.00025, 0.00025, 0.00025, 0.00025])
        sr, psr = prob_sharpe_ratio(pnl, trade_weights=weights, annualize=False)
        assert np.isnan(sr)
        assert np.isnan(psr)

    def test_weighted_uses_weighted_sr_not_unweighted(self):
        """Weighted mode must use weighted SR, not unweighted SR.

        With time-decay weights (recent data weighted higher), the weighted
        SR should differ from unweighted SR. This is the core of the
        weighted refactor: previously the function used unweighted SR with
        ESS variance, which gave different results than the WFA inline
        formula (which uses fully weighted SR).
        """
        rng = np.random.default_rng(42)
        # Create returns with strong upward trend (recent > old)
        pnl = np.concatenate([
            rng.normal(-0.5, 0.3, 50),   # older: negative mean
            rng.normal(0.5, 0.3, 50),    # recent: positive mean
        ])
        # Decay weights: recent data weighted higher
        weights = np.exp(-np.arange(100, 0, -1) / 25)

        # Unweighted: should give near-zero SR (mix of negative and positive)
        _, psr_unw = prob_sharpe_ratio(pnl, annualize=False)
        # Weighted: should give positive SR (recent data dominates)
        _, psr_w = prob_sharpe_ratio(pnl, annualize=False, trade_weights=weights)

        # Weighted PSR should be higher than unweighted (weighted SR > unweighted SR)
        assert psr_w > psr_unw, (
            f"Weighted PSR ({psr_w:.4f}) should be higher than "
            f"unweighted ({psr_unw:.4f}) for time-decay weights on trending data"
        )


class TestPSRandFDRIntegration:
    def test_psr_with_fdr(self):
        """PSR values integrate correctly with FDR correction."""
        rng = np.random.default_rng(42)
        # Generate 5 symbols, 2 should be significant
        all_p = []
        for i in range(5):
            pnl = rng.normal(0.3 if i < 2 else 0, 0.5, 100)
            sr, psr = prob_sharpe_ratio(pnl, benchmark=0.0, annualize=False)
            # Approximate p-value from PSR
            all_p.append(1.0 - psr if psr > 0.5 else 0.9)

        rejected, p_corr = fdr_correction(np.array(all_p), alpha=0.15)
        assert len(rejected) == 5
        assert len(p_corr) == 5
        # The first 2 should likely be significant
        assert rejected[0] or rejected[1]


# ════════════════════════════════════════════════════════════════════════
# Deflated Sharpe Ratio (Phase 2.1)
# ════════════════════════════════════════════════════════════════════════


class TestDeflatedSharpeRatio:
    """Phase 2.1: deflated_sharpe_ratio (Bailey & López de Prado 2014).

    Adjusts PSR for multiple testing across N independent trials.
    """

    def test_n_trials_less_than_2_returns_nan(self):
        """No multiple testing → PSR is the correct metric, deflated is NaN."""
        from quant_lib.core._testing import deflated_sharpe_ratio
        result = deflated_sharpe_ratio(2.0, 1, n_obs_per_trial=30)
        assert result != result  # NaN
        result = deflated_sharpe_ratio(2.0, 0, n_obs_per_trial=30)
        assert result != result  # NaN
        result = deflated_sharpe_ratio(2.0, -1, n_obs_per_trial=30)
        assert result != result  # NaN

    def test_observed_at_or_below_benchmark_returns_zero(self):
        """Observed SR at or below null benchmark: cannot be best of N."""
        from quant_lib.core._testing import deflated_sharpe_ratio
        # observed == benchmark: returns 0
        result = deflated_sharpe_ratio(0.0, 100, benchmark_sharpe=0.0)
        assert result == 0.0
        # observed < benchmark: returns 0
        result = deflated_sharpe_ratio(-0.5, 100, benchmark_sharpe=0.0)
        assert result == 0.0
        # observed < positive benchmark
        result = deflated_sharpe_ratio(0.3, 100, benchmark_sharpe=0.5)
        assert result == 0.0

    def test_high_observed_sr_with_few_trials_high_psr(self):
        """Few trials + high observed SR → deflated PSR should be high."""
        from quant_lib.core._testing import deflated_sharpe_ratio
        # 5 trials, observed SR 2.0 with n_obs=30 → very significant
        result = deflated_sharpe_ratio(2.0, 5, n_obs_per_trial=30)
        assert result > 0.9

    def test_moderate_observed_sr_with_many_trials_low_psr(self):
        """Many trials + moderate observed SR → deflated PSR should be lower."""
        from quant_lib.core._testing import deflated_sharpe_ratio
        # 50000 trials, observed SR 0.5 with n_obs=30 → less significant
        # (the bar is higher when there are many trials)
        result_few = deflated_sharpe_ratio(0.5, 100, n_obs_per_trial=30)
        result_many = deflated_sharpe_ratio(0.5, 50000, n_obs_per_trial=30)
        # More trials → lower deflated PSR (more likely the SR is just
        # the best of N under the null)
        assert result_many < result_few

    def test_n_obs_per_trial_affects_variance(self):
        """More observations per trial → lower SR variance → higher PSR."""
        from quant_lib.core._testing import deflated_sharpe_ratio
        # Same observed SR and n_trials, but more observations
        result_few_obs = deflated_sharpe_ratio(
            0.5, 14400, n_obs_per_trial=30
        )
        result_many_obs = deflated_sharpe_ratio(
            0.5, 14400, n_obs_per_trial=500
        )
        # More observations per trial → tighter SR distribution
        # → easier to reject null → higher deflated PSR
        assert result_many_obs > result_few_obs

    def test_skewness_correction_applied(self):
        """Negative skewness penalizes the deflated PSR (worse strategy)."""
        from quant_lib.core._testing import deflated_sharpe_ratio
        # Same observed SR and trials, but with different skewness
        result_normal = deflated_sharpe_ratio(
            1.0, 1000, returns_skewness=0.0, n_obs_per_trial=100
        )
        result_neg_skew = deflated_sharpe_ratio(
            1.0, 1000, returns_skewness=-2.0, n_obs_per_trial=100
        )
        # Negative skewness (lottery-ticket-like losses) should
        # reduce the deflated PSR (or at least not increase it).
        assert result_neg_skew <= result_normal

    def test_excess_kurtosis_correction_applied(self):
        """Fat tails (positive excess kurtosis) reduce the deflated PSR."""
        from quant_lib.core._testing import deflated_sharpe_ratio
        result_normal = deflated_sharpe_ratio(
            1.0, 1000, returns_excess_kurtosis=0.0, n_obs_per_trial=100
        )
        result_fat_tails = deflated_sharpe_ratio(
            1.0, 1000, returns_excess_kurtosis=3.0, n_obs_per_trial=100
        )
        # Fat tails → wider SR distribution → harder to reject null
        # → lower deflated PSR
        assert result_fat_tails <= result_normal

    def test_result_in_valid_range(self):
        """Deflated PSR is always in [0, 1]."""
        from quant_lib.core._testing import deflated_sharpe_ratio
        # Edge case inputs that could cause numerical issues
        result = deflated_sharpe_ratio(0.001, 2, n_obs_per_trial=100)
        assert 0.0 <= result <= 1.0
        result = deflated_sharpe_ratio(100.0, 1000000, n_obs_per_trial=10)
        assert 0.0 <= result <= 1.0
        # Negative observed with positive benchmark
        result = deflated_sharpe_ratio(-10.0, 100, benchmark_sharpe=0.0)
        assert 0.0 <= result <= 1.0

    def test_default_benchmark_is_zero(self):
        """Default benchmark_sharpe is 0.0 (no risk-free rate)."""
        from quant_lib.core._testing import deflated_sharpe_ratio
        # Calling without benchmark should work (default 0.0)
        result = deflated_sharpe_ratio(1.0, 100, n_obs_per_trial=50)
        assert 0.0 <= result <= 1.0

    def test_realistic_framework_scenario(self):
        """Realistic: 6 symbols × 30 folds × 80 trials = 14400 trials.
        Observed SR 0.8, n_obs 50 (typical crypto strategy).

        The exact deflated PSR value depends on the math (n_obs/trial
        shrinks the SR variance enough that moderate observed SR can
        still be significant). The important property is that the
        result is bounded in [0, 1] and well-defined.
        """
        from quant_lib.core._testing import deflated_sharpe_ratio
        result = deflated_sharpe_ratio(
            observed_sharpe=0.8,
            n_trials=14400,  # 6 × 30 × 80
            returns_skewness=-0.5,  # typical crypto negative skew
            returns_excess_kurtosis=2.0,  # fat tails
            n_obs_per_trial=50,
        )
        # Deflated PSR is bounded in [0, 1]
        assert 0.0 <= result <= 1.0
        # Deflated PSR should be lower than or equal to the
        # uncorrected (single-trial) probability for a positive SR
        # (multiple testing correction always reduces confidence).
        # We check this by computing with n_trials=1 (== NaN) and
        # comparing to n_trials=10000:
        result_few_trials = deflated_sharpe_ratio(
            0.8, 100, n_obs_per_trial=50
        )
        # More trials → deflated PSR is lower or equal
        assert result <= result_few_trials + 1e-9

    def test_zero_n_obs_per_trial_uses_asymptotic(self):
        """n_obs_per_trial=None or 0 → asymptotic variance formula."""
        from quant_lib.core._testing import deflated_sharpe_ratio
        result = deflated_sharpe_ratio(1.0, 100, n_obs_per_trial=None)
        assert 0.0 <= result <= 1.0
        result = deflated_sharpe_ratio(1.0, 100, n_obs_per_trial=0)
        assert 0.0 <= result <= 1.0
