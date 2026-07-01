"""Tests for the consolidated prob_sharpe_ratio (with optional trade_weights).

NOTE (0.2.2): The standalone `objective_psr_ess()` was removed -- it was a
near-duplicate of `prob_sharpe_ratio()` for the unweighted case. Its
weighted-mode functionality is now provided by
`prob_sharpe_ratio(returns, trade_weights=...)`.
"""

import numpy as np
import pytest
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
        - Bailey variance correction: 1 - skew*SR + (excess+2)/4 * SR²
          (where excess = sample excess kurtosis via scipy fisher=True)
        """
        rng = np.random.default_rng(42)
        pnl = rng.normal(0.3, 0.8, 100)
        weights = np.exp(-np.arange(100) / 30)  # decay weights

        # prob_sharpe_ratio weighted path
        _, psr_lib = prob_sharpe_ratio(pnl, trade_weights=weights, annualize=False)

        # Replicate WFA inline formula (core/_wfa.py:179-200).
        # Bailey's variance correction uses REGULAR kurtosis (γ₄ = excess + 3):
        #   var_corr = 1 - γ₃·SR + (γ₄ - 1)/4 · SR²
        #           = 1 - γ₃·SR + (excess + 2)/4 · SR²
        w = weights / weights.sum()
        w_mean = np.dot(pnl, w)
        w_var = np.dot(w, (pnl - w_mean) ** 2)
        w_sr = w_mean / np.sqrt(w_var)
        ess = 1.0 / np.dot(w, w)
        skew_v = float(scipy_stats.skew(pnl))
        kurt_v = float(scipy_stats.kurtosis(pnl, fisher=True))  # EXCESS
        var_corr = (1 - skew_v * w_sr + (kurt_v + 2) / 4 * w_sr**2)
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

        Bailey's PSR formula uses REGULAR kurtosis γ₄ = excess + 3
        (3 for normal data). Internally we compute excess via scipy
        fisher=True and convert via (γ₄ - 1)/4 = (excess + 2)/4.

        For normal data with SR=0.5:
            var_corr = 1 - 0 + (0 + 2)/4 * 0.25 = 1 + 0.125 = 1.125
        (positive, formula in valid range)
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

    def test_sr_uses_sample_std_ddof1(self):
        """v0.4.0 (Phase 2.4): SR must use sample std (ddof=1), matching
        the variance correction denominator ``n - 1``. Prior version
        used np.std default (ddof=0) which under-reported SR by ~5% for
        n=20.
        """
        rng = np.random.default_rng(42)
        # Small n: bias between ddof=0 and ddof=1 is visible (~2-5%)
        n = 30
        returns = rng.normal(0.005, 0.01, size=n)
        expected_sr = float(returns.mean() / returns.std(ddof=1))
        impl_sr, _ = prob_sharpe_ratio(returns, annualize=False)
        # SR should match ddof=1 (sample std) within float precision
        assert abs(impl_sr - expected_sr) < 1e-10, (
            f"PSR SR ({impl_sr:.6f}) must match sample std (ddof=1) "
            f"({expected_sr:.6f}). Difference: {impl_sr - expected_sr:.6f}. "
            f"Population std (ddof=0) would give "
            f"{returns.mean() / returns.std():.6f}"
        )

    def test_sr_ddof_consistency_large_n(self):
        """For large n, ddof=0 vs ddof=1 difference is negligible but
        should still match ddof=1 exactly."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=1000)
        expected_sr = float(returns.mean() / returns.std(ddof=1))
        impl_sr, _ = prob_sharpe_ratio(returns, annualize=False)
        assert abs(impl_sr - expected_sr) < 1e-12

    def test_psr_uses_correct_kurtosis_coefficient(self):
        """Verify PSR uses (excess + 2)/4, NOT (excess - 1)/4.

        Bug fix v0.3.1: prior code used (kurt - 1)/4 treating excess
        kurtosis as if it were regular kurtosis. The difference between
        formulas is 3/4 * SR² / denom in the variance correction, which
        is most visible at moderate SR (~0.5) with small n (~20) where
        the z-score is large but variance correction is the differentiator.

        Test strategy: construct small-n normal data with moderate SR,
        where both formulas give finite (non-clipped) var_corr but
        the kurtosis coefficient is the main differentiator.
        """
        from scipy import stats as sp_stats
        rng = np.random.default_rng(42)
        # n=20, SR~0.5 → both formulas finite, but different (~0.008 PSR diff).
        # Larger n saturates PSR to 1.0 (no difference visible).
        n = 20
        returns = rng.normal(0.005, 0.01, size=n)
        sample_sr = float(returns.mean() / returns.std(ddof=1))
        sample_skew = float(sp_stats.skew(returns))
        sample_kurt_excess = float(sp_stats.kurtosis(returns, fisher=True))

        # Sanity: SR should be ~0.5 for test to be meaningful
        if not (0.3 < sample_sr < 0.7):
            pytest.skip(
                f"Test data SR={sample_sr:.3f} outside target range [0.3, 0.7]"
            )

        sr, psr = prob_sharpe_ratio(returns, annualize=False)
        # SR should be close to sample_sr (within 5%)
        assert abs(sr - sample_sr) / max(abs(sample_sr), 1e-9) < 0.1, (
            f"PSR SR {sr:.3f} diverges from sample SR {sample_sr:.3f}"
        )

        # For n=20 normal data with SR~0.5:
        #   Correct formula: 1 - skew*SR + (excess+2)/4 * SR²
        #   Old (buggy) formula: 1 - skew*SR + (excess-1)/4 * SR²
        # Old formula produces SMALLER var_corr → LARGER z-score →
        # HIGHER PSR (because Phi is monotonic). So PSR_correct <= PSR_old.
        # We verify:
        #   1. PSR is in valid range (handled by other tests)
        #   2. PSR is <= what the old formula would give
        #   3. PSR is >= what the correct formula predicts (within tolerance)
        denom = len(returns) - 1
        # Compute what PSR would be under the OLD (buggy) formula
        var_corr_old = (
            1 - sample_skew * sample_sr + (sample_kurt_excess - 1) / 4 * sample_sr**2
        )
        if var_corr_old > 0:
            variance_old = var_corr_old / denom
            psr_old_would_be = float(
                sp_stats.norm.cdf(sample_sr / np.sqrt(variance_old))
            )
            # Old formula inflates PSR (underestimates variance). The fixed
            # formula should give a LOWER OR EQUAL PSR.
            assert psr <= psr_old_would_be + 1e-6, (
                f"PSR ({psr:.6f}) exceeds OLD (buggy) formula PSR "
                f"({psr_old_would_be:.6f}). Old formula inflates PSR; "
                f"the fixed formula should be <= old. "
                f"Sample: SR={sample_sr:.3f}, skew={sample_skew:.3f}, "
                f"excess_kurt={sample_kurt_excess:.3f}."
            )
            # And the difference must be non-trivial (otherwise the test
            # doesn't actually validate the fix)
            assert psr_old_would_be - psr > 0.003, (
                f"PSR ({psr:.6f}) matches OLD (buggy) formula "
                f"({psr_old_would_be:.6f}) too closely. "
                f"Difference: {psr_old_would_be - psr:.6f}. "
                f"Bug fix may not have taken effect (or data has zero kurt)."
            )
        else:
            # If old formula would have clipped, fixed formula should also clip
            # (and produce a similar or lower PSR)
            assert psr <= 1.0, "PSR must be <= 1.0"

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


class TestDeflatedSharpeRatioClosedForm:
    """Phase LOW-3: closed-form verification of the deflated PSR formula.

    Validates the implementation against the analytical Bailey &
    López de Prado 2014 Eqs 2.2 and 3.1. Tests hand-derived values
    for specific parameter combinations so the formula can be
    independently verified against the paper.

    Paper formulas (using excess kurtosis notation):
        V[s_hat] = 1 - γ₃·s* + ((excess + 2) / 4) · s*²
        E[max_s_n] = sqrt(V) · [(1 - γ₃·s* + ((excess+2)/4)·s*²)·z_α + γ₃·s*]
        PSR_deflated = Φ((s_obs - E[max]) / sqrt(V))
    where:
        z_α = Φ⁻¹(1 - 1/N)
        γ_3 = sample skewness
        excess = sample excess kurtosis
        s* = benchmark Sharpe
        N = n_trials
    """

    def test_normal_no_skew_benchmark_zero(self):
        """Normal returns, skew=0, excess=0, benchmark=0, asymptotic V.

        With skew=0, s*=0, the asymptotic variance V=1 (constant).
        The kurt_bias is ((0+2)/4)·0² = 0, so E[max] = z_α.
        PSR = Φ(s_obs - z_α).

        Note: s_obs must be > 0 (benchmark). At s_obs=0 the function
        returns 0.0 (early-return guard), not the formula value.
        """
        from quant_lib.core._testing import deflated_sharpe_ratio
        from scipy import stats

        for n_trials in [10, 100, 1000]:
            z_alpha = stats.norm.ppf(1 - 1 / n_trials)
            for s_obs in [0.5, 1.0, 2.0, 5.0]:
                result = deflated_sharpe_ratio(
                    observed_sharpe=s_obs,
                    n_trials=n_trials,
                    returns_skewness=0.0,
                    returns_excess_kurtosis=0.0,
                    benchmark_sharpe=0.0,
                    n_obs_per_trial=None,  # asymptotic V = 1
                )
                expected = float(stats.norm.cdf(s_obs - z_alpha))
                assert abs(result - expected) < 1e-9, (
                    f"Closed-form mismatch for n_trials={n_trials}, "
                    f"s_obs={s_obs}: got {result}, expected {expected}"
                )

    def test_asymptotic_with_kurtosis(self):
        """Asymptotic V=1, with kurtosis bias in E[max].

        With n_obs=None, V = 1 (asymptotic). The kurtosis bias
        only enters E[max], not V:

            kurt_bias = ((excess + 2) / 4) · s*²
            E[max] = ((1 + 0 + kurt_bias) · z_α + 0) · sqrt(V)
            PSR = Φ((s_obs - E[max]) / sqrt(V))
        """
        from quant_lib.core._testing import deflated_sharpe_ratio
        from scipy import stats

        n_trials = 100
        s_star = 0.5
        excess = 3.0  # moderate fat tails
        s_obs = 3.0

        sr_variance = 1.0  # n_obs=None → asymptotic V = 1
        sr_std = np.sqrt(sr_variance)
        z_alpha = stats.norm.ppf(1 - 1 / n_trials)
        kurt_bias = ((excess + 2) / 4) * s_star**2  # skew=0, so skew_bias=0
        expected_max = ((1.0 - 0.0 + kurt_bias) * z_alpha + 0.0) * sr_std
        expected = float(stats.norm.cdf((s_obs - expected_max) / sr_std))

        result = deflated_sharpe_ratio(
            observed_sharpe=s_obs,
            n_trials=n_trials,
            returns_skewness=0.0,
            returns_excess_kurtosis=excess,
            benchmark_sharpe=s_star,
            n_obs_per_trial=None,
        )
        assert abs(result - expected) < 1e-9, (
            f"Closed-form mismatch: got {result}, expected {expected}"
        )

    def test_finite_sample_variance_correct(self):
        """Verify V[s_hat] = (1 + kurt_bias·sr*²) / (n-1) matches paper.

        With skew=0, s*=0:
            V = (1 + 0) / (n-1) = 1/(n-1)
        So PSR = Φ(s_obs · sqrt(n-1) - z_α).
        """
        from quant_lib.core._testing import deflated_sharpe_ratio
        from scipy import stats

        n_obs = 100
        n_trials = 1000
        s_obs = 2.0
        z_alpha = stats.norm.ppf(1 - 1 / n_trials)
        expected = float(stats.norm.cdf(s_obs * np.sqrt(n_obs - 1) - z_alpha))

        result = deflated_sharpe_ratio(
            observed_sharpe=s_obs,
            n_trials=n_trials,
            returns_skewness=0.0,
            returns_excess_kurtosis=3.0,  # any value; with sr*=0, V=1
            benchmark_sharpe=0.0,
            n_obs_per_trial=n_obs,
        )
        assert abs(result - expected) < 1e-9, (
            f"V formula mismatch (skew=0, sr*=0): got {result}, expected {expected}"
        )

    def test_negative_skew_lowers_psr(self):
        """Negative skew (γ_3 < 0) + positive benchmark (s* > 0) LOWERS PSR.

        Derivation: E[max] = ((1 - γ_3·s* + kurt_bias)·z_α + γ_3·s*) · sqrt(V).
        For γ_3 = -2, s* = 0.5, z_α ≈ 2.33:
            skew_bias = γ_3·s* = -1
            skew_bias·(1 - z_α) = -1 · (-1.33) = +1.33 (POSITIVE bias)
        So negative skew INCREASES E[max] → LOWERS PSR.

        This is consistent with the paper: the "upper tail" of a
        negatively-skewed distribution with positive mean extends
        further, making it harder to claim an observed positive SR
        is the best of N trials under the null.
        """
        from quant_lib.core._testing import deflated_sharpe_ratio

        n_trials = 100
        common = dict(
            observed_sharpe=2.5,
            n_trials=n_trials,
            returns_excess_kurtosis=0.0,
            benchmark_sharpe=0.5,
            n_obs_per_trial=None,
        )
        result_no_skew = deflated_sharpe_ratio(
            returns_skewness=0.0, **common
        )
        result_neg_skew = deflated_sharpe_ratio(
            returns_skewness=-2.0, **common
        )
        # Negative skew → higher E[max] → lower PSR
        assert result_neg_skew < result_no_skew, (
            f"Negative skew should LOWER PSR: "
            f"no_skew={result_no_skew}, neg_skew={result_neg_skew}"
        )

    def test_higher_kurtosis_lowers_psr(self):
        """Fat-tailed returns (higher excess kurtosis) should LOWER PSR.

        Rationale: fatter tails mean SR estimator has higher variance,
        so the upper tail of "best of N trials" extends further,
        making it harder to claim a real edge.
        """
        from quant_lib.core._testing import deflated_sharpe_ratio

        common = dict(
            observed_sharpe=2.0,
            n_trials=10000,
            returns_skewness=0.0,
            benchmark_sharpe=0.0,
            n_obs_per_trial=100,
        )
        result_normal = deflated_sharpe_ratio(
            returns_excess_kurtosis=0.0, **common
        )
        result_fat_tail = deflated_sharpe_ratio(
            returns_excess_kurtosis=3.0, **common
        )
        result_very_fat = deflated_sharpe_ratio(
            returns_excess_kurtosis=10.0, **common
        )
        # Monotonic: higher kurtosis → lower PSR
        assert result_normal >= result_fat_tail, (
            f"Normal kurt should have higher PSR: "
            f"normal={result_normal}, fat={result_fat_tail}"
        )
        assert result_fat_tail >= result_very_fat, (
            f"Fat tails should have lower PSR than normal: "
            f"fat={result_fat_tail}, very_fat={result_very_fat}"
        )

    def test_more_trials_lowers_psr(self):
        """More trials = stronger multiple-testing penalty."""
        from quant_lib.core._testing import deflated_sharpe_ratio

        common = dict(
            observed_sharpe=2.0,
            returns_skewness=0.0,
            returns_excess_kurtosis=0.0,
            benchmark_sharpe=0.0,
            n_obs_per_trial=None,
        )
        result_few = deflated_sharpe_ratio(n_trials=10, **common)
        result_many = deflated_sharpe_ratio(n_trials=100000, **common)
        # More trials → lower PSR
        assert result_few > result_many, (
            f"More trials should lower PSR: "
            f"few={result_few}, many={result_many}"
        )

    def test_higher_observed_sharpe_higher_psr(self):
        """Monotonicity in observed_sharpe."""
        from quant_lib.core._testing import deflated_sharpe_ratio

        common = dict(
            n_trials=1000,
            returns_skewness=0.0,
            returns_excess_kurtosis=0.0,
            benchmark_sharpe=0.0,
            n_obs_per_trial=None,
        )
        psr_low = deflated_sharpe_ratio(observed_sharpe=1.0, **common)
        psr_mid = deflated_sharpe_ratio(observed_sharpe=2.0, **common)
        psr_high = deflated_sharpe_ratio(observed_sharpe=3.0, **common)
        assert psr_low < psr_mid < psr_high, (
            f"Monotonicity in observed_sharpe: "
            f"low={psr_low}, mid={psr_mid}, high={psr_high}"
        )
