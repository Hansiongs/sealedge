"""
Feature engineering tools -- compute strategy features from raw data.
"""

import pandas as pd

from quant_lib.core._features import prepare_data_with_max_time as _prepare
from quant_lib.core._metrics import build_daily_matrices as _build_matrices


def compute_features(
    df_raw: pd.DataFrame,
    df_btc_raw: pd.DataFrame,
    df_funding_raw: pd.DataFrame | None = None,
    max_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute all strategy features from raw OHLCV data.

    Features calculated:
    - hh_20 / ll_20 (20-period channel)
    - vol_pct_rank (normalized volatility percentile)
    - rvol (relative volume)
    - ATR (average true range)
    - ema_200 (trend filter)
    - macro_trend (BTC EMA 4800 regime)
    - macro_vol (annualised volatility)
    - is_weekend / is_funding_hour flags

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw klines with columns: time, open, high, low, close, volume.
    df_btc_raw : pd.DataFrame
        BTC klines (extended history for EMA warmup).
    df_funding_raw : pd.DataFrame or None
        Funding rate data (optional).
    max_time : pd.Timestamp or None
        Cutoff time. If None, uses the latest time in df_raw.

    Returns
    -------
    pd.DataFrame
        Input data augmented with all feature columns.
    """
    if max_time is None:
        max_time = df_raw["time"].max()
    return _prepare(df_raw, df_btc_raw, df_funding_raw, max_time)


def build_matrices(
    symbols: list[str], precomputed_data: dict[str, pd.DataFrame]
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Build daily close and high/low matrices from precomputed hourly data.

    Parameters
    ----------
    symbols : list of str
        Universe symbols.
    precomputed_data : dict of str -> pd.DataFrame
        Precomputed features per symbol (output of compute_features).

    Returns
    -------
    daily_close_matrix : dict
        {symbol: {date: close_price}}
    daily_hl_matrix : dict
        {symbol: {date: {high: ..., low: ...}}}
    """
    return _build_matrices(symbols, precomputed_data)
