"""
Data fetch/cache helpers (Binance Vision and local cache).
"""

import pandas as pd

from quant_lib.core._data import ensure_data_exists as _ensure_data_exists
from quant_lib.core._data import ensure_funding_exists as _ensure_funding_exists


def fetch_klines(
    symbol: str,
    interval: str = "1h",
    start_date: str = "2021-01-01",
    end_date: str = "2026-05-31",
) -> pd.DataFrame:
    """Fetch OHLCV klines from Binance Vision.

    Downloads monthly ZIP files from Binance public data, caches to CSV.
    Subsequent calls reuse the cache if the date range is covered.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. "BTCUSDT".
    interval : str
        Kline interval, e.g. "1h", "4h", "1d".
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Columns: time, open, high, low, close, volume
    """
    path = _ensure_data_exists(symbol, interval, start_date, end_date)
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_funding(
    symbol: str,
    start_date: str = "2021-01-01",
    end_date: str = "2026-05-31",
) -> pd.DataFrame | None:
    """Fetch historical funding rates from Binance Vision.

    Parameters
    ----------
    symbol : str
        Trading pair.
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame or None
        Columns: time, funding_rate. None if no data available.
    """
    path = _ensure_funding_exists(symbol, start_date, end_date)
    if path is None:
        return None
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    return df
