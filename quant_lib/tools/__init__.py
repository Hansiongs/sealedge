"""
quant_lib.tools -- public composable API.

Building blocks for custom pipelines. You own the control flow.

Usage::

    from quant_lib.tools import fetch_klines, compute_features, walk_forward, spa_test
"""

from quant_lib.tools.data import fetch_klines, fetch_funding
from quant_lib.tools.features import compute_features, build_matrices
from quant_lib.tools.backtest import walk_forward, run_trade_loop
from quant_lib.tools.portfolio import simulate_portfolio
from quant_lib.tools.stats import spa_test, prob_sharpe_ratio, fdr_correct
from quant_lib.tools.universe import select_universe

__all__ = [
    "fetch_klines",
    "fetch_funding",
    "compute_features",
    "build_matrices",
    "walk_forward",
    "run_trade_loop",
    "simulate_portfolio",
    "spa_test",
    "prob_sharpe_ratio",
    "fdr_correct",
    "select_universe",
]
