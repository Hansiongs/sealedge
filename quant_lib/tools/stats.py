"""
Statistical testing tools -- SPA, PSR, FDR correction.
"""

from numpy import ndarray
import pandas as pd

from quant_lib.core._spa import portfolio_spa as _portfolio_spa
from quant_lib.core._testing import (
    prob_sharpe_ratio as _psr,
    fdr_correction as _fdr,
)
from quant_lib.core._config import STATIC, DEFAULTS


def spa_test(
    observed_trades: list[dict],
    asset_data: dict[str, pd.DataFrame],
    daily_close_matrix: dict[str, dict],
    end_date: str,
    daily_hl_matrix: dict[str, dict] | None = None,
    n_iters: int = STATIC["spa_n_iters"],
    initial_capital: float = 1000.0,
    leverage: float = 3.0,
    mm_pct: float = 0.01,
    position_limit: int = 4,
    cb_hard_cooldown_hours: int = 24,
    fixed_cb_threshold: float = 0.15,
    rng_seed: int = 42,
    verbose: bool = False,
    liquidation_fee_pct: float = 0.005,
    fee_taker: float = 0.05,
    # NOTE (0.2.2): Was hardcoded 2.5. Now mirrors DEFAULTS so direct
    # spa_test() callers get the same cost model as the WFA/commit path.
    stress_mult: float = DEFAULTS["stress_test_multiplier"],
    weekend_penalty: float = DEFAULTS["weekend_liquidity_penalty"],
    asset_risk_weights: dict[str, float] | None = None,
) -> tuple[float, ndarray, float]:
    """Run Portfolio SPA (Superior Predictive Ability) test.

    Tests whether the observed strategy edge is genuine or random
    using time-anchored circular permutation across all assets.

    The null hypothesis replicates the strategy's trailing stop exit
    mechanism on randomly-timed entries, isolating entry-timing edge
    from exit-mechanism edge.

    Parameters
    ----------
    observed_trades : list of dict
        All OOS trade signals from walk_forward. Must include sl_mult
        and trail_atr fields for trailing stop replication.
    asset_data : dict of str -> pd.DataFrame
        Per-symbol data slices with columns: time, close, atr, high, low,
        funding_rate, is_weekend, is_funding_hour, macro_trend.
    daily_close_matrix : dict
        {symbol: {date: close}} for portfolio simulation.
    end_date : str
        End date for simulation (YYYY-MM-DD).

    Returns
    -------
    observed_equity : float
        Final equity from baseline simulation.
    null_equities : np.ndarray
        Equity from each permutation iteration.
    p_value : float
        SPA p-value (Phipson & Smyth 2010 add-one corrected).
    """
    if asset_risk_weights is None:
        asset_risk_weights = None  # Let portfolio_spa handle None

    return _portfolio_spa(
        observed_trades,
        asset_data,
        daily_close_matrix,
        end_date,
        daily_hl_matrix=daily_hl_matrix,
        n_iters=n_iters,
        initial_capital=initial_capital,
        leverage=leverage,
        mm_pct=mm_pct,
        position_limit=position_limit,
        cb_hard_cooldown_hours=cb_hard_cooldown_hours,
        fixed_cb_threshold=fixed_cb_threshold,
        rng_seed=rng_seed,
        verbose=verbose,
        liquidation_fee_pct=liquidation_fee_pct,
        fee_taker=fee_taker,
        stress_mult=stress_mult,
        weekend_penalty=weekend_penalty,
        asset_risk_weights=asset_risk_weights,
    )


def prob_sharpe_ratio(
    returns: ndarray,
    benchmark: float = 0.0,
    annualize: bool = True,
) -> tuple[float, float]:
    """Compute Probabilistic Sharpe Ratio.

    PSR measures the probability that the true Sharpe ratio exceeds
    the benchmark, accounting for return skewness and kurtosis.

    Parameters
    ----------
    returns : np.ndarray
        Daily return series.
    benchmark : float
        Benchmark Sharpe ratio (annualised).
    annualize : bool
        Whether to annualize the output SR.

    Returns
    -------
    sharpe_ratio : float
    psr : float
        Probability (0-1) that true SR > benchmark.
    """
    return _psr(returns, benchmark, annualize)


def fdr_correct(p_values: ndarray, alpha: float = 0.05) -> tuple[ndarray, ndarray]:
    """Apply Benjamini-Hochberg FDR correction to multiple p-values.

    Parameters
    ----------
    p_values : array-like
        Raw p-values from multiple hypothesis tests.
    alpha : float
        Target false discovery rate.

    Returns
    -------
    rejected : np.ndarray (bool)
        Which hypotheses are rejected at FDR alpha.
    p_corrected : np.ndarray (float)
        Adjusted p-values (q-values).
    """
    return _fdr(p_values, alpha)
