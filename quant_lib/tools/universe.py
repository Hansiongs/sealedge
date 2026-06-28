"""
Universe construction tools -- mechanical, point-in-time asset selection.

Based on Phase 1 of the framework: selection criteria must be
mechanical, point-in-time, and independent of strategy performance.
"""

import pandas as pd

from quant_lib.core._data import ensure_data_exists
from quant_lib.core._logging import log


def select_universe(
    candidates: list[str],
    start_date: str,
    end_date: str,
    interval: str = "1h",
    min_volume_usdt: float = 50_000_000.0,
    min_age_days: int = 180,
    volume_lookback_days: int = 90,
    verbose: bool = True,
) -> list[str]:
    """Mechanically select universe based on volume and age criteria.

    Point-in-time criteria (no strategy performance involved):
    - Listing age >= min_age_days before start_date
    - Trailing volume_lookback_days median daily volume >= min_volume_usdt
    - Available OHLCV data for the full period

    Parameters
    ----------
    candidates : list of str
        Candidate symbols to evaluate (e.g. from Binance USDT perpetual list).
    start_date : str
        Backtest start date (YYYY-MM-DD).
    end_date : str
        Backtest end date (YYYY-MM-DD).
    interval : str
        Data interval for volume calculation.
    min_volume_usdt : float
        Minimum median daily volume in USDT.
    min_age_days : int
        Minimum days since listing.
    volume_lookback_days : int
        Lookback window for volume calculation.
    verbose : bool
        Print selection diagnostics.

    Returns
    -------
    list of str
        Symbols that pass all criteria.
    """
    eligible = []
    start_dt = pd.Timestamp(start_date)

    for sym in candidates:
        try:
            csv_path = ensure_data_exists(sym, interval, start_date, end_date)
        except (RuntimeError, Exception) as e:
            log.warning(f"[{sym}] Data unavailable: {e}")
            continue

        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")

        # Age check: must be listed at least min_age_days before start_date
        first_bar = df["time"].min()
        age_at_start = (start_dt - first_bar).days
        if age_at_start < min_age_days:
            if verbose:
                log.info(
                    f"[{sym}] Excluded: age at start = {age_at_start}d < {min_age_days}d "
                    f"(listing: {first_bar.date()})"
                )
            continue

        # Volume check: median daily volume over lookback window
        df_vol = df[df["time"] >= start_dt - pd.Timedelta(days=volume_lookback_days)]
        df_vol = df_vol[df_vol["time"] <= start_dt]

        if len(df_vol) < 24:  # minimum 1 day of hourly data
            if verbose:
                log.info(f"[{sym}] Excluded: insufficient volume data ({len(df_vol)} bars)")
            continue

        # Approximate daily volume from hourly data
        # Sum hourly volume in chunks of 24 bars as a proxy for daily volume
        daily_vol = df_vol["volume"].rolling(24).sum().dropna()
        if len(daily_vol) == 0:
            if verbose:
                log.info(f"[{sym}] Excluded: no valid daily volume estimate")
            continue

        # Use close price at volume lookback point to approximate USDT value
        ref_price = df_vol["close"].iloc[-1]
        daily_vol_usdt = daily_vol * ref_price
        median_daily_vol = daily_vol_usdt.median()

        if median_daily_vol < min_volume_usdt:
            if verbose:
                log.info(
                    f"[{sym}] Excluded: median daily vol = "
                    f"${median_daily_vol:,.0f} < ${min_volume_usdt:,.0f}"
                )
            continue

        eligible.append(sym)
        if verbose:
            log.info(
                f"[{sym}] Eligible: age={age_at_start}d, "
                f"vol=${median_daily_vol:,.0f}"
            )

    log.info(
        f"Universe selection complete: {len(eligible)}/{len(candidates)} symbols eligible"
    )
    return sorted(eligible)


def filter_by_volume_rank(
    symbols: list[str],
    precomputed_data: dict[str, pd.DataFrame],
    top_n: int = 6,
) -> list[str]:
    """Filter universe to top-N by median volume (non-performance criterion).

    Uses precomputed data volume -- does NOT use any strategy output.
    Suitable for Phase 3 narrowing when operational constraints limit
    the number of tradeable assets.

    Parameters
    ----------
    symbols : list of str
        Current eligible universe.
    precomputed_data : dict of str -> pd.DataFrame
        Precomputed features per symbol (must have 'volume' column).
    top_n : int
        Number of symbols to keep.

    Returns
    -------
    list of str
        Top-N symbols by median hourly volume.
    """
    vol_map = {}
    for sym in symbols:
        df = precomputed_data[sym]
        vol_map[sym] = df["volume"].median()

    sorted_syms = sorted(vol_map, key=vol_map.get, reverse=True)
    return sorted_syms[:top_n]
