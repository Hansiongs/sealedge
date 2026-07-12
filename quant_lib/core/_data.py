"""
Data fetch: Binance Vision klines and funding rates into DATA_DIR.
"""

import os
import re
import tempfile
import requests
import zipfile
import io
import time
from typing import Optional
import pandas as pd
from requests import Response

from quant_lib.core._logging import log


# Phase 3.3: Input validation for cache paths. Symbols must match
# Binance's naming convention (uppercase, alphanumeric, up to 20
# chars, ending in USDT). Intervals must be one of the standard
# Binance kline intervals. Rejecting bad inputs early prevents
# path traversal and creates cleaner error messages.
_VALID_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}USDT$")
_VALID_INTERVALS = frozenset({"1m", "5m", "15m", "1h", "4h", "1d"})


def _validate_symbol(symbol: str) -> str:
    """Validate symbol format. Raises ValueError on invalid input."""
    if not isinstance(symbol, str) or not _VALID_SYMBOL_RE.match(symbol):
        raise ValueError(
            f"Invalid symbol {symbol!r}. Must match pattern "
            f"[A-Z0-9]{{2,20}}USDT (e.g., BTCUSDT, ETHUSDT)."
        )
    return symbol


def _validate_interval(interval: str) -> str:
    """Validate interval format. Raises ValueError on invalid input."""
    if interval not in _VALID_INTERVALS:
        raise ValueError(
            f"Invalid interval {interval!r}. "
            f"Must be one of: {sorted(_VALID_INTERVALS)}."
        )
    return interval

# All cached CSV files go here (auto-created)
DATA_DIR = "data_cache"

_DATA_DIR_INITIALIZED = False


def _ensure_data_dir() -> None:
    """Lazy-init the data cache directory, called only when needed."""
    global _DATA_DIR_INITIALIZED
    if not _DATA_DIR_INITIALIZED:
        os.makedirs(DATA_DIR, exist_ok=True)
        _DATA_DIR_INITIALIZED = True


def _data_path(filename: str) -> str:
    """Prefix a filename with the data directory."""
    _ensure_data_dir()
    return os.path.join(DATA_DIR, filename)


# Maximum response body size (500 MB). A single month of 1H klines
# is ~2 MB compressed, ~10 MB uncompressed. 500 MB is a generous
# safety cap that will never be hit by legitimate data but prevents
# a malicious or compromised server from OOM'ing the process.
_MAX_RESPONSE_BYTES = 500 * 1024 * 1024


def fetch_with_retry(
    url: str,
    timeout: int = 15,
    max_retries: int = 3,
    max_size: int = _MAX_RESPONSE_BYTES,
) -> Optional[Response]:
    """Helper for HTTP fetch with retry logic, proper logging, and size cap.

    Phase 3.4: Uses ``stream=True`` + ``iter_content`` with a hard
    size cap (default 500 MB) to prevent unbounded response bodies
    from a malicious or compromised server. The cap is well above
    any legitimate Binance Vision response (a full year of 1H klines
    is ~120 MB uncompressed).

    Parameters
    ----------
    url : str
        The URL to fetch.
    timeout : int
        Per-attempt timeout in seconds. Default 15.
    max_retries : int
        Number of retries on transient failures. Default 3.
    max_size : int
        Maximum response body in bytes. Default 500 MB.

    Returns
    -------
    Response or None
        The response object with ``._content`` populated, or None on
        transient failure or size cap exceeded.
    """
    for attempt in range(max_retries):
        try:
            res = requests.get(url, timeout=timeout, stream=True)
            res.raise_for_status()
            # Stream content with size cap
            content = bytearray()
            for chunk in res.iter_content(chunk_size=1024 * 1024):
                content.extend(chunk)
                if len(content) > max_size:
                    log.error(
                        f"Response too large (>{max_size:,} bytes): {url}"
                    )
                    return None
            res._content = bytes(content)
            return res
        except requests.exceptions.Timeout:
            log.warning(f"Timeout on attempt {attempt+1}/{max_retries}: {url}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            log.warning(f"HTTP {e.response.status_code} on attempt {attempt+1}: {url}")
        except requests.exceptions.ConnectionError:
            log.warning(f"Connection error on attempt {attempt+1}/{max_retries}: {url}")
        except Exception as e:
            log.error(f"Unexpected error fetching {url}: {type(e).__name__}: {e}")
            return None
        if attempt < max_retries - 1:
            time.sleep(2**attempt)
    return None


def ensure_data_exists(symbol: str, interval: str, start_date: str, end_date: str) -> str:
    _validate_symbol(symbol)
    _validate_interval(interval)
    _ensure_data_dir()
    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    output_filename = f"{symbol}_{interval}_MASTER.csv"
    output_path = _data_path(output_filename)

    need_refetch = False
    if os.path.exists(output_path):
        try:
            df_check = pd.read_csv(output_path, nrows=5)
            expected_cols = {"time", "open", "high", "low", "close", "volume"}
            if expected_cols.issubset(df_check.columns):
                try:
                    df_time = pd.read_csv(output_path, usecols=["time"])
                    cached_times = pd.to_datetime(df_time["time"], errors="coerce").dropna()
                    if len(cached_times) > 0:
                        cached_min = cached_times.min()
                        cached_max = cached_times.max()
                        if cached_min > start_dt or cached_max < end_dt:
                            log.warning(
                                f"[{symbol}] Cached data ({cached_min.date()} to "
                                f"{cached_max.date()}) doesn't cover requested range "
                                f"({start_date} to {end_date}). Re-fetching..."
                            )
                            need_refetch = True
                        if not need_refetch and len(cached_times) > 10:
                            diffs = cached_times.sort_values().diff().dropna()
                            median_gap = diffs.median()
                            if median_gap > pd.Timedelta(hours=2):
                                log.warning(
                                    f"[{symbol}] Cached data has continuity issues -- "
                                    f"median gap = {median_gap}. Some months may be "
                                    f"missing. Gap is permanent; delete the CSV "
                                    f"manually to force a clean re-fetch."
                                )
                    if len(cached_times) == 0:
                        log.warning(
                            f"[{symbol}] Cached file has header but no data rows. "
                            f"Re-fetching..."
                        )
                        need_refetch = True
                except Exception:
                    log.warning(f"[{symbol}] Cannot validate cache date range, re-fetching...")
                    need_refetch = True
                if not need_refetch:
                    log.info(f"[{symbol}] Using cached data: {output_filename} [{DATA_DIR}/]")
                    return output_path
            else:
                log.warning(f"[{symbol}] Cached file malformed, re-fetching...")
                need_refetch = True
        except Exception as e:
            log.warning(f"[{symbol}] Cannot read cache ({e}), re-fetching...")
            need_refetch = True
        if need_refetch:
            try:
                os.remove(output_path)
            except OSError:
                pass

    log.info(f"[yellow]{symbol}[/]: Klines not found. Fetching from Binance Vision...")
    months = pd.date_range(start_dt.replace(day=1), end_dt, freq="MS")
    all_data = []
    failed_months = []

    for dt in months:
        url = (
            f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{dt.strftime('%Y-%m')}.zip"
        )
        res = fetch_with_retry(url)
        if res is None:
            failed_months.append(dt.strftime("%Y-%m"))
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    all_data.append(pd.read_csv(f, header=None))
        except zipfile.BadZipFile:
            log.warning(f"Corrupt zip for {symbol} {dt.strftime('%Y-%m')}, skipping.")
            failed_months.append(dt.strftime("%Y-%m"))
        except Exception as e:
            log.error(f"Parse error for {symbol} {dt.strftime('%Y-%m')}: {e}")
            failed_months.append(dt.strftime("%Y-%m"))

    if failed_months:
        log.warning(f"[{symbol}] Failed months (skipped): {failed_months}")

    if not all_data:
        raise RuntimeError(
            f"Critical Failure: No Kline data fetched for {symbol}. "
            f"Check network or Binance Vision availability."
        )

    df = pd.concat(all_data, ignore_index=True)
    df.columns = [
        "time", "open", "high", "low", "close", "volume",
        "close_time", "qv", "cnt", "tbv", "tbqv", "ign",
    ]
    df = df[["time", "open", "high", "low", "close", "volume"]]
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df[(df["time"] >= start_dt) & (df["time"] <= end_dt)]
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv", prefix=f"{symbol}_", dir=DATA_DIR,
    ) as _tf:
        _tmp_path = _tf.name
        df.to_csv(_tf, index=False)
    os.replace(_tmp_path, output_path)
    return output_path


def ensure_funding_exists(symbol: str, start_date: str, end_date: str) -> Optional[str]:
    _validate_symbol(symbol)
    _ensure_data_dir()
    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    output_filename = f"{symbol}_FUNDING_MASTER.csv"
    output_path = _data_path(output_filename)

    if os.path.exists(output_path):
        try:
            df_check = pd.read_csv(output_path, nrows=5)
            expected_cols = {"time", "funding_rate"}
            if expected_cols.issubset(df_check.columns):
                try:
                    df_time = pd.read_csv(output_path, usecols=["time"])
                    cached_times = pd.to_datetime(df_time["time"], errors="coerce").dropna()
                    if len(cached_times) > 0:
                        cached_min = cached_times.min()
                        cached_max = cached_times.max()
                        if cached_min > start_dt or cached_max < end_dt:
                            log.warning(
                                f"[{symbol}] Cached funding data ({cached_min.date()} to "
                                f"{cached_max.date()}) doesn't cover requested range. "
                                f"Re-fetching..."
                            )
                            os.remove(output_path)
                        else:
                            return output_path
                    else:
                        log.warning(f"[{symbol}] Cached funding file has no data rows. Re-fetching...")
                        os.remove(output_path)
                except Exception:
                    log.warning(f"[{symbol}] Cannot validate funding cache date range, re-fetching...")
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass
            else:
                log.warning(f"[{symbol}] Cached funding file malformed, re-fetching...")
                try:
                    os.remove(output_path)
                except OSError:
                    pass
        except Exception as e:
            log.warning(f"[{symbol}] Cannot read funding cache ({e}), re-fetching...")
            try:
                os.remove(output_path)
            except OSError:
                pass

    log.info(f"[yellow]{symbol}[/]: Funding Rate not found. Fetching from Binance Vision...")
    months = pd.date_range(start_dt.replace(day=1), end_dt, freq="MS")
    all_data = []

    for dt in months:
        url = (
            f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/"
            f"{symbol}-fundingRate-{dt.strftime('%Y-%m')}.zip"
        )
        res = fetch_with_retry(url)
        if res is None:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f)
                    df.columns = df.columns.str.strip().str.lower()
                    c_map = {}
                    for c in df.columns:
                        if "funding" in c and "rate" in c:
                            c_map[c] = "funding_rate"
                        elif "calc" in c and "time" in c:
                            c_map[c] = "calc_time"
                        elif "funding" in c and "time" in c:
                            c_map[c] = "calc_time"
                    df.rename(columns=c_map, inplace=True)
                    all_data.append(df)
        except Exception as e:
            log.warning(
                f"[{symbol}] Failed to parse funding zip for {dt.strftime('%Y-%m')}: "
                f"{type(e).__name__}: {e}"
            )

    if not all_data:
        return None

    df = pd.concat(all_data, ignore_index=True)
    if "calc_time" not in df.columns or "funding_rate" not in df.columns:
        return None

    df["calc_time"] = pd.to_numeric(df["calc_time"], errors="coerce")
    df = df.dropna(subset=["calc_time"])
    df["time"] = pd.to_datetime(df["calc_time"], unit="ms").dt.round("h")
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df = df[["time", "funding_rate"]].drop_duplicates(subset=["time"])
    df = df[(df["time"] >= start_dt) & (df["time"] <= end_dt)]
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv", prefix=f"{symbol}_funding_", dir=DATA_DIR,
    ) as _tf:
        _tmp_path = _tf.name
        df.to_csv(_tf, index=False)
    os.replace(_tmp_path, output_path)
    return output_path
