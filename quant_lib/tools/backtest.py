"""
Walk-forward analysis and trade-loop helpers (public tools API).
"""

import numpy as np
import pandas as pd

from quant_lib.core._wfa import run_wfa_per_symbol as _run_wfa
from quant_lib.core._engine import fast_trade_loop as _trade_loop, EngineArgs
from quant_lib.core._config import STATIC, DEFAULTS


def walk_forward(
    symbol: str,
    precomputed_df: pd.DataFrame,
    use_rvol: bool = True,
    use_ema: bool = True,
    verbose: bool = True,
    reg_lambda: float = 0.05,
    strategy_type: int = 0,
    allow_long: bool = True,
    allow_short: bool = True,
    search_space: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run walk-forward optimization for one symbol.

    Executes expanding-window WFA with Optuna TPE optimization.
    Collects all out-of-sample (OOS) trades across folds.

    Parameters
    ----------
    symbol : str
        Trading pair symbol.
    precomputed_df : pd.DataFrame
        Precomputed features (from compute_features).
    use_rvol : bool
        Apply volume confirmation filter.
    use_ema : bool
        Apply EMA200 trend filter.
    verbose : bool
        Print fold progress.
    reg_lambda : float
        L2 regularization strength for parameter stability.
    strategy_type : int
        0 = vol_compression, 1 = pullback_sniper.
    allow_long : bool
        Allow long entries.
    allow_short : bool
        Allow short entries.
    search_space : dict or None
        Per-hypothesis Optuna search space. If None, uses STATIC defaults.

    Returns
    -------
    trades : list of dict
        All OOS trade records with fields:
        entry_time, exit_time, symbol, r_net, trade_dir, sl_pct, etc.
    fold_params : list of dict
        Optimal parameters per fold with stability metadata.
    """
    return _run_wfa(
        symbol, precomputed_df, use_rvol, use_ema, verbose, reg_lambda,
        strategy_type=strategy_type,
        allow_long=1 if allow_long else 0,
        allow_short=1 if allow_short else 0,
        search_space=search_space,
    )


def run_trade_loop(
    df: pd.DataFrame,
    vol_pct_thresh: float = 0.20,
    pullback_bars: int = 5,
    trail_atr: float = 3.0,
    sl_mult: float = 1.5,
    use_rvol: bool = True,
    use_ema: bool = True,
    seed: int = 42,
    strategy_type: int = 0,
    allow_long: bool = True,
    allow_short: bool = True,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
) -> dict[str, object]:
    """Run the Numba trade loop on a single DataFrame.

    Useful for quick strategy evaluation without WFA.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-complete DataFrame (from compute_features).
    vol_pct_thresh : float
        Volatility percentile entry threshold (vol_compression only).
    pullback_bars : int
        Pullback confirmation bars (vol_compression only).
    trail_atr : float
        Trailing stop ATR multiple.
    sl_mult : float
        Initial stop ATR multiple.
    use_rvol : bool
        Apply volume filter.
    use_ema : bool
        Apply EMA200 filter.
    seed : int
        RNG seed for cost model stochastic component.
    strategy_type : int
        0 = vol_compression, 1 = pullback_sniper.
    allow_long : bool
        Whether to allow long entries.
    allow_short : bool
        Whether to allow short entries.
    rsi_oversold : float
        RSI oversold threshold (pullback_sniper only).
    rsi_overbought : float
        RSI overbought threshold (pullback_sniper only).

    Returns
    -------
    dict with keys:
        pnl, entry_times, exit_times, trade_dir, n_trades
    """
    critical_cols = [
        "open", "high", "low", "close",
        "hh_20", "ll_20", "ema_200",
        "vol_pct_rank", "rvol", "atr",
        "funding_rate", "macro_vol", "macro_trend",
        "is_weekend", "is_funding_hour",
    ]
    df_clean = df.dropna(subset=critical_cols).copy()

    rng = np.random.default_rng(seed)
    random_draws = rng.random(size=len(df_clean) * 2).astype(np.float64)

    n = len(df_clean)
    zeros_f = np.zeros(n, dtype=np.float64)
    zeros_i = np.zeros(n, dtype=np.int32)
    rsi_14 = df_clean["rsi_14"].values if "rsi_14" in df_clean.columns else zeros_f
    bullish_rev = df_clean["bullish_reversal"].values if "bullish_reversal" in df_clean.columns else zeros_i
    bearish_rev = df_clean["bearish_reversal"].values if "bearish_reversal" in df_clean.columns else zeros_i

    # Build args via EngineArgs dataclass (recommended pattern).
    # The dataclass groups args by category for readability while
    # remaining a no-op wrapper around the @njit positional call.
    engine_args = EngineArgs(
        market_data=(
            df_clean["open"].values,
            df_clean["high"].values,
            df_clean["low"].values,
            df_clean["close"].values,
        ),
        channel_features=(
            df_clean["hh_20"].values,
            df_clean["ll_20"].values,
            df_clean["ema_200"].values,
        ),
        pullback_features=(rsi_14, bullish_rev, bearish_rev),
        signal_features=(
            df_clean["vol_pct_rank"].values,
            df_clean["rvol"].values,
            df_clean["atr"].values,
            # Phase 2: funding_pct_rank slot (safe default 0.5 = neutral).
            # backtest.py currently exercises vol_compression and
            # pullback_sniper; the funding-carry branch never triggers.
            df_clean["funding_pct_rank"].values if "funding_pct_rank" in df_clean.columns else np.full(len(df_clean), 0.5, dtype=np.float64),
        ),
        auxiliary_features=(
            df_clean["funding_rate"].values,
            df_clean["macro_vol"].values,
            df_clean["macro_trend"].values,
            df_clean["is_weekend"].values,
            df_clean["is_funding_hour"].values,
        ),
        strategy_type=strategy_type,
        thresholds=(
            vol_pct_thresh,
            DEFAULTS["fixed_rvol_thresh"],
            rsi_oversold,
            rsi_overbought,
            # Phase 2: funding_rate_carry thresholds (safe defaults).
            0.90, 0.40, 0.60, 0.0,
        ),
        integer_params=(
            pullback_bars,
            DEFAULTS["bailout_bars"],
            0,  # warmup_bars
            0,  # unused slot
        ),
        exit_params=(trail_atr, sl_mult),
        cost_model=(
            STATIC["fee_taker"],
            DEFAULTS["weekend_liquidity_penalty"],
            DEFAULTS["stress_test_multiplier"],
        ),
        flags=(
            1 if use_rvol else 0,
            1 if use_ema else 0,
            1 if allow_long else 0,
            1 if allow_short else 0,
        ),
        random_draws=random_draws,
        trend_mults=(
            DEFAULTS["trend_aligned_risk_mult"],
            DEFAULTS["trend_counter_risk_mult"],
        ),
    )
    result = _trade_loop(*engine_args.as_tuple())

    pnl_array = result[0]
    idx_en = result[1]
    idx_ex = result[2]
    t_dir = result[3]
    trend_mults = result[9]

    return {
        "pnl": pnl_array,
        "entry_times": df_clean["time"].iloc[idx_en].tolist(),
        "exit_times": df_clean["time"].iloc[idx_ex].tolist(),
        "trade_dir": t_dir[:len(pnl_array)],
        "trend_risk_mult": trend_mults[:len(pnl_array)],
        "n_trades": len(pnl_array),
    }
