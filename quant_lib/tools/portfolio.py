"""
Portfolio simulation tools -- MTM, margin, circuit breaker.
"""

from typing import Any

from quant_lib.core._portfolio import simulate_full_portfolio as _simulate


def simulate_portfolio(
    trades: list[dict[str, Any]],
    initial_cash: float = 1000.0,
    leverage: float = 3.0,
    mm_pct: float = 0.01,
    position_limit: int = 4,
    cb_hard_cooldown_hours: int = 24,
    fixed_cb_threshold: float = 0.15,
    daily_close_matrix: dict[str, dict] | None = None,
    asset_risk_weights: dict[str, float] | None = None,
    end_date: str = "2026-05-31",
    liquidation_fee_pct: float = 0.005,
    daily_hl_matrix: dict[str, dict] | None = None,
) -> tuple[float, dict, list, dict[str, int]]:
    """Run portfolio-level simulation with position sizing, margin, and CB.

    Simulates trade execution across a portfolio of assets with:
    - Equity-based position sizing
    - Maintenance margin checks (intraday via HL matrix)
    - Per-asset circuit breaker (drawdown-based cooldown)
    - Rolling correlation-aware risk scaling

    Parameters
    ----------
    trades : list of dict
        Trade records with fields: entry_time, exit_time, symbol, r_net,
        trade_dir, sl_pct, entry_price, exit_price, risk_weight.
    initial_cash : float
        Starting capital.
    leverage : float
        Maximum leverage.
    mm_pct : float
        Maintenance margin fraction.
    position_limit : int
        Maximum concurrent positions.
    cb_hard_cooldown_hours : int
        Circuit breaker cooldown in hours.
    fixed_cb_threshold : float
        Drawdown % threshold for CB trigger.
    daily_close_matrix : dict, optional
        {symbol: {date: close}} for MTM. Built from data if None.
    asset_risk_weights : dict, optional
        {symbol: risk_weight}. Uses STATIC defaults if None.
    end_date : str
        End date for MTM tail fill.
    liquidation_fee_pct : float
        Fee fraction on liquidation.
    daily_hl_matrix : dict, optional
        {symbol: {date: {high, low}}} for intraday liquidation check.

    Returns
    -------
    final_equity : float
    daily_equity : dict
    executed_trades : list
    reject_reasons : dict
    """
    if daily_close_matrix is None:
        raise ValueError("daily_close_matrix is required for MTM simulation")
    # If asset_risk_weights is None, simulate_full_portfolio will skip
    # the per-asset circuit breaker (see core/_portfolio.py:148).

    return _simulate(
        trades,
        initial_cash,
        leverage,
        mm_pct,
        position_limit,
        cb_hard_cooldown_hours,
        fixed_cb_threshold,
        daily_close_matrix,
        asset_risk_weights,
        end_date,
        liquidation_fee_pct,
        daily_hl_matrix,
    )
