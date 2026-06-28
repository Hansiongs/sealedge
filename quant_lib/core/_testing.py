"""
Statistical testing utilities -- PSR, FDR, p-value labeling.

Extracted from Hans_Quant_Systems.py:
  - prob_sharpe_ratio (lines 1850-1874)
  - _fdr_correction (lines 2798-2853)
  - _label_p_value (lines 2856-2894)
"""

import numpy as np
from numpy import ndarray
import scipy.stats as stats

from quant_lib.core._logging import log


def prob_sharpe_ratio(
    returns: ndarray,
    benchmark: float = 0.0,
    annualize: bool = True,
    trade_weights: ndarray | None = None,
) -> tuple[float, float]:
    """Probabilistic Sharpe Ratio with optional effective sample size (ESS).

    The PSR adjusts the Sharpe ratio for skewness and kurtosis, more
    valid for crypto returns (fat-tailed, asymmetric). Implements the
    Bailey & Lopez de Prado (2012) formula:

        PSR = Phi((SR - SR*) / sqrt(Var_correction / (n_eff - 1)))
        Var_correction = 1 - gamma_3 * SR + ((gamma_4 - 1) / 4) * SR^2

    where gamma_3 = skewness and gamma_4 = EXCESS kurtosis (0 for normal).
    See quant_lib.core._testing._PSR_FORMULA_NOTES for derivation notes.

    Parameters
    ----------
    returns : np.ndarray
        Return series (R-multiples or daily returns).
    benchmark : float
        Benchmark Sharpe ratio. Default 0.
    annualize : bool
        Whether to annualize the output SR. Default True. Set False for
        R-multiples (per-trade returns).
    trade_weights : np.ndarray, optional
        Per-element weights (e.g., time-decay). When provided, computes
        WEIGHTED mean, weighted variance, weighted SR (matching the
        formula in ``core/_wfa.py``). Effective sample size (Kish ESS)
        is used in the variance denominator. When None, uses
        unweighted sample mean/std with ``n - 1`` denominator.

    Returns
    -------
    (sharpe_ratio, psr) : tuple of float
        PSR in [0, 1] (probability that true SR exceeds benchmark).
        Returns (NaN, NaN) for insufficient data, zero variance, or
        ESS < 2 in weighted mode.
    """
    n = len(returns)

    # ── Step 1: Compute SR (weighted if trade_weights provided) ──
    if trade_weights is None:
        # Unweighted path
        if n < 10 or np.std(returns) == 0:
            log.warning(
                f"PSR: insufficient data ({n} returns, "
                f"std={np.std(returns):.4f}). Returning SR=NaN, PSR=NaN."
            )
            return float("nan"), float("nan")
        sr = float(np.mean(returns) / np.std(returns))
        # Sanity guard: extreme SR (e.g. >1e6) usually means near-constant
        # input that bypassed the std==0 check via float precision. Such
        # inputs produce meaningless PSR values, so return NaN.
        if not np.isfinite(sr) or abs(sr) > 1e6:
            log.warning(
                f"PSR: extreme SR ({sr:.2e}) suggests degenerate input "
                f"(near-constant returns). Returning SR=NaN, PSR=NaN."
            )
            return float("nan"), float("nan")
        denom = n - 1
    else:
        # Weighted path: weighted mean, weighted variance, weighted SR
        # (matches the formula in core/_wfa.py:179-200)
        w = np.asarray(trade_weights, dtype=np.float64)
        w_sum = w.sum()
        if w_sum <= 0:
            log.warning(
                f"PSR: trade_weights sum to {w_sum} (<= 0). Returning NaN."
            )
            return float("nan"), float("nan")
        w = w / w_sum  # normalize to sum=1
        w_mean = float(np.dot(returns, w))
        w_var = float(np.dot(w, (returns - w_mean) ** 2))
        if n < 10 or w_var <= 0:
            log.warning(
                f"PSR weighted: insufficient data ({n} returns) or "
                f"zero weighted variance ({w_var:.4f}). Returning NaN."
            )
            return float("nan"), float("nan")
        sr = w_mean / np.sqrt(w_var)
        if not np.isfinite(sr) or abs(sr) > 1e6:
            log.warning(
                f"PSR weighted: extreme SR ({sr:.2e}) suggests degenerate "
                f"input. Returning SR=NaN, PSR=NaN."
            )
            return float("nan"), float("nan")
        # Kish ESS: n_eff = (sum w)^2 / sum(w^2) = 1 / sum(w^2) since w sums to 1
        ess = 1.0 / float(np.dot(w, w))
        if ess < 2.0:
            log.warning(
                f"PSR weighted: ESS={ess:.2f} < 2 (insufficient effective "
                f"samples). Returning NaN."
            )
            return float("nan"), float("nan")
        denom = ess - 1.0

    # ── Step 2: Compute higher moments (excess kurtosis = Bailey convention) ──
    # NOTE: kurt uses fisher=True (excess kurtosis, 0 for normal). This
    # matches Bailey & Lopez de Prado (2012) formula. Prior versions used
    # fisher=False (regular kurtosis, 3 for normal) which overestimated
    # variance by +3/4 * SR^2 / denom, understating PSR.
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=True))  # excess kurtosis

    # ── Step 3: Apply annualization to SR and benchmark if requested ──
    ann_factor = np.sqrt(365.25)
    benchmark_daily = benchmark / ann_factor if annualize else benchmark
    if annualize:
        sr = sr * ann_factor

    # ── Step 4: Compute Bailey's PSR variance correction ──
    # Bailey's formula is an asymptotic expansion valid for moderate SR.
    # For very high SR with near-normal data, the correction term
    # (kurt-1)/4 * SR^2 can make the variance correction negative
    # (e.g., normal data with SR > 2: correction = 1 - SR^2/4 < 0).
    # In that regime, the formula is outside its valid range. We clip
    # the variance correction to a small positive value rather than
    # returning NaN, so callers get a usable PSR. The PSR will be very
    # high (close to 1) reflecting the high observed SR -- the same
    # intuitive result the original (incorrectly-robust) formula gave.
    variance_correction = 1 - skew * sr + ((kurt - 1) / 4) * sr**2
    if variance_correction <= 0:
        log.warning(
            f"PSR variance correction <= 0 (skew={skew:.2f}, kurt={kurt:.2f}, "
            f"sr={sr:.2f}) -- Bailey's asymptotic formula outside its valid "
            f"range. Clipping to minimum. Returning PSR close to 1.0."
        )
        variance_correction = 1e-8
    variance = variance_correction / denom
    sr_std = np.sqrt(variance)
    psr = float(stats.norm.cdf((sr - benchmark_daily) / sr_std))
    # PSR can technically exceed 1 due to numerical issues with very
    # high z-scores; clamp to [0, 1] for safety.
    if psr > 1.0:
        psr = 1.0
    elif psr < 0.0:
        psr = 0.0
    return sr, psr


def fdr_correction(p_values: ndarray, alpha: float = 0.05) -> tuple[ndarray, ndarray]:
    """Benjamini-Hochberg FDR correction for multiple comparison.

    Parameters
    ----------
    p_values : array-like of float
        Raw p-values from N independent hypotheses.
    alpha : float
        Target false discovery rate (default 0.05).

    Returns
    -------
    rejected : np.ndarray (bool)
        True for hypotheses rejected at FDR alpha.
    p_corrected : np.ndarray (float)
        Adjusted p-values (q-values), clamped to [0, 1].
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.float64)

    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]

    thresholds = np.arange(1, n + 1) * alpha / n
    below = sorted_p <= thresholds + 1e-15

    if below.any():
        k = int(np.where(below)[0][-1])
        reject_threshold = sorted_p[k]
        rejected_sorted = sorted_p <= reject_threshold
    else:
        reject_threshold = 0.0
        rejected_sorted = np.zeros(n, dtype=bool)

    adjusted_sorted = np.minimum(1.0, sorted_p * n / np.arange(1, n + 1))
    # Enforce monotonicity (step-up): running MINIMUM from the RIGHT
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]

    rejected = np.empty(n, dtype=bool)
    adjusted = np.empty(n, dtype=np.float64)
    rejected[sorted_idx] = rejected_sorted
    adjusted[sorted_idx] = adjusted_sorted

    return rejected, adjusted


# Context-specific threshold tiers. Each entry: (threshold_upper_bound,
# label, rich_style, confidence_band, interpretation). The threshold is
# the upper bound of that tier (lower tier is strictly less). Tiers must
# be sorted by ascending threshold.
#
# - "mean_r": per-symbol R-multiple PSR. Same thresholds pre-0.2.3.
# - "spa":    portfolio-level SPA p-value. Stricter thresholds (SPA is
#             already a multi-symbol test; we want higher confidence
#             before claiming a real portfolio-wide edge).
_LABEL_TIERS: dict[str, list[tuple[float, str, str, str, str]]] = {
    "mean_r": [
        (0.005, "PROD",     "bright_green", "> 99.5%",  "Holy grail -- very rare, suspect data snooping"),
        (0.05,  "TRADE",    "green",        "95-99.5%", "Tradeable -- worth paper trading / small live allocation"),
        (0.15,  "WATCH",    "yellow",       "85-95%",   "Watchlist -- signal present, needs further validation"),
        (0.30,  "RESEARCH", "orange3",      "70-85%",   "Research -- trace of signal, too weak to trade"),
        (float("inf"), "NO EDGE", "red",   "< 70%",    "No edge -- pure noise, move on"),
    ],
    "spa": [
        # SPA is multi-symbol + multi-fold, so we want stronger evidence
        # to declare a real edge. Tiers shifted ~2x stricter than mean_r.
        (0.0025, "PROD",     "bright_green", "> 99.75%", "Portfolio holy grail -- very rare, suspect data snooping"),
        (0.025,  "TRADE",    "green",        "97.5-99.75%", "Portfolio tradeable -- paper trade first"),
        (0.075,  "WATCH",    "yellow",       "92.5-97.5%", "Watchlist -- multi-symbol signal detected"),
        (0.15,   "RESEARCH", "orange3",      "85-92.5%",   "Research -- portfolio signal too weak"),
        (float("inf"), "NO EDGE", "red",   "< 85%",     "No edge at portfolio level"),
    ],
}


def label_p_value(p: float | None, context: str = "mean_r") -> tuple[str, str, str]:
    """Crypto-adjusted p-value label, confidence, and brief interpretation.

    Different contexts use different threshold tiers:
    - "mean_r" (default): per-symbol R-multiple PSR thresholds.
    - "spa": portfolio-level SPA p-value thresholds (stricter, since
      SPA is a multi-symbol test that already accounts for multiple
      comparisons).

    Unknown context strings fall back to "mean_r" tiers (with a
    debug log) so the function is robust to typos.

    Parameters
    ----------
    p : float or None
        P-value to label. NaN/None returns UNRELIABLE.
    context : str
        "mean_r" for per-symbol (5-tier) or "spa" for Portfolio SPA.

    Returns
    -------
    label : str
        Rich-markup label string.
    confidence : str
        Confidence range.
    interpretation : str
        Brief interpretation.
    """
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return (
            "[bold grey]UNRELIABLE[/]",
            "N/A",
            f"p-value is NaN ({context} context)",
        )

    tiers = _LABEL_TIERS.get(context)
    if tiers is None:
        # Unknown context: log once and fall back to mean_r (defensive).
        log.warning(
            f"label_p_value: unknown context '{context}', "
            f"falling back to 'mean_r' tiers."
        )
        tiers = _LABEL_TIERS["mean_r"]

    for upper, label_name, style, conf, interp in tiers:
        if p < upper:
            return (
                f"[bold {style}]{label_name}[/]",
                conf,
                interp,
            )
    # Unreachable (last tier has upper=inf), but defensive return.
    return "[bold red]NO EDGE[/]", "< 70%", "No edge -- pure noise, move on"


# NOTE (0.2.2): Removed `objective_psr_ess()` -- it was a near-duplicate of
# `prob_sharpe_ratio()` for the unweighted case. The functionality is now
# provided by `prob_sharpe_ratio(trade_weights=...)`. Callers updated:
# - quant_lib/research/commit.py: use prob_sharpe_ratio(annualize=False)
# - tests/test_psr_ess.py: now tests prob_sharpe_ratio with trade_weights
# The WFA path in core/_wfa.py keeps its own inline PSR computation.
