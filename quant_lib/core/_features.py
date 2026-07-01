"""
Feature engineering -- leakage-aware feature computation.

Extracted from Hans_Quant_Systems.py:
  - prepare_data_with_max_time (lines 416-650)
"""

import pandas as pd
import numpy as np

from quant_lib.core._config import (
    STATIC,
    WARMUP_BARS,
    STRATEGY_VOL_COMPRESSION,
    STRATEGY_PULLBACK_SNIPER,
)
from quant_lib.core._logging import log

# Sentinel for gap detection threshold
_MAX_ALLOWED_GAP_HOURS: int = 2

# Default pullback_sniper config
_DEFAULT_RSI_PERIOD = 14
_DEFAULT_EMA_PULLBACK = 20
_TARGET_VOL_FOR_SPAN = 0.50  # 50% annualized vol baseline for span calc
_MIN_SPAN = 2400
_MAX_SPAN = 7200

# Per-asset macro_trend baseline span (1H bars). 4800 ~= 200 days ~= 6.6 months.
# Chosen to be slow enough that macro regime doesn't flip on noise.
# Scaled by (asset_vol_median / _TARGET_VOL_FOR_SPAN) so high-vol assets
# get a shorter span (faster adaptation) and vice versa.
_DEFAULT_SPAN_EMA = 4800.0


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index) without lookahead.

    RSI uses Wilder's smoothing method:
      gain = max(close - close.shift(1), 0)
      loss = max(close.shift(1) - close, 0)
      avg_gain = ewm(gain, alpha=1/period, adjust=False).mean()
      avg_loss = ewm(loss, alpha=1/period, adjust=False).mean()
      RS = avg_gain / avg_loss
      RSI = 100 - 100 / (1 + RS)

    Returns raw RSI at bar i (uses close[0..i]). The caller must
    apply .shift(1) to use it as a non-lookahead signal.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def prepare_data_with_max_time(
    df_raw: pd.DataFrame,
    df_btc_raw: pd.DataFrame,
    df_funding_raw: pd.DataFrame | None,
    max_time: pd.Timestamp,
    strategy_type: int = STRATEGY_VOL_COMPRESSION,
    rsi_period: int = _DEFAULT_RSI_PERIOD,
    btc_holdout_start: pd.Timestamp | None = None,
    apply_holdout_ema_to_full: bool = True,
) -> pd.DataFrame:
    """
    Compute ALL features using only data <= max_time.
    NO bfill -- any remaining NaN in the early warm‑up period will be dropped
    later when we slice IS/OOS windows (which always start after >720 hours).

    Phase 2.2: When ``btc_holdout_start`` is provided, ``macro_trend``
    (and the asset-level EMA) are computed strictly on data BEFORE this
    timestamp, then forward-filled. Default ``None`` (preserves prior
    behavior: compute on full df, no holdout cutoff).

    Parameters
    ----------
    strategy_type : int
        0 = vol_compression (default), 1 = pullback_sniper.
    rsi_period : int
        RSI period for pullback_sniper. Default 14.
    btc_holdout_start : pd.Timestamp | None
        If provided, compute macro_trend EMA strictly on data BEFORE this
        timestamp and forward-fill.
    apply_holdout_ema_to_full : bool
        When ``btc_holdout_start`` is provided, controls whether the
        holdout-constant EMA is applied to the **entire** ``df`` (default
        ``True``, the historical behavior) or only to bars ``>= btc_holdout_start``
        (IS bars keep their own dynamic EMA, matching the WFA path).

        The two modes differ in the IS period:
        - ``True`` (default, pre-v0.4.0 behavior): IS-period macro_trend
          uses the constant pre-holdout EMA value, NOT the dynamic per-bar
          EMA. This is the historical behavior; useful when you want
          the entire backtest to use a regime signal consistent with what
          the holdout will see.
        - ``False`` (new, opt-in): IS-period macro_trend uses the
          standard dynamic EMA (matches the WFA path where ``btc_holdout_start``
          is None). Use this when comparing WFA training results against
          holdout results -- the signals will then be consistent across
          the boundary.

        See CHANGELOG v0.4.0 for rationale.
    """
    df = df_raw[df_raw["time"] <= max_time].copy()
    if df.empty:
        return df

    df_btc = df_btc_raw[df_btc_raw["time"] <= max_time].copy()

    if "taker_buy_volume" in df.columns:
        df.rename(columns={"taker_buy_volume": "tbv"}, inplace=True)

    df["hh_20"] = df["high"].shift(1).rolling(window=20).max()
    df["ll_20"] = df["low"].shift(1).rolling(window=20).min()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    # -- FIX #1: Address leakage in vol_pct_rank --
    # Shift applied BEFORE rolling window min/max to ensure
    # the price at bar t does not leak into the extreme bounds.
    df["realized_vol_24"] = df["log_ret"].rolling(window=24).std().shift(1)
    vol_min_720 = df["realized_vol_24"].rolling(window=720).min()
    vol_max_720 = df["realized_vol_24"].rolling(window=720).max()
    df["vol_pct_rank"] = np.where(
        vol_max_720 > vol_min_720,
        (df["realized_vol_24"] - vol_min_720) / (vol_max_720 - vol_min_720),
        np.nan
    )

    df["sma_vol_24"] = df["volume"].shift(1).rolling(window=24).mean()
    df["rvol"] = df["volume"] / (df["sma_vol_24"] + 1e-10)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1)),
        ),
    )
    # ATR is used for SL distance at entry bar (engine line 287, 326,
    # 387, 416). Use yesterday's ATR to avoid current-bar look-ahead:
    # today's SL shouldn't depend on today's realized volatility.
    # Phase 1.5 fix: add shift(1).
    df["atr"] = df["tr"].rolling(window=STATIC["atr_len"]).mean().shift(1)
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean().shift(1)
    df["macro_vol"] = df["log_ret"].rolling(window=30 * 24).std().shift(1) * np.sqrt(
        365 * 24
    )

    # Per-asset MACRO TREND (volatility-adjusted EMA)
    # Replaces BTC-only macro_trend so each symbol uses its own regime.
    # span is scaled by relative volatility vs target baseline.
    #
    # Phase 2.2: When ``btc_holdout_start`` is provided, the asset EMA
    # is computed on data STRICTLY before this timestamp, and the
    # last value is forward-filled into the holdout. This prevents
    # the holdout's own price action from influencing the trend
    # signal (state-persistence leak). When ``btc_holdout_start`` is
    # None (default), the original behavior is preserved: EMA is
    # computed on the full df (used in WFA path, where look-ahead
    # within the IS is acceptable and necessary for proper regime
    # detection across the full training period).
    if btc_holdout_start is not None:
        # Compute vol scaling on pre-holdout data only
        df_pre = df[df["time"] < btc_holdout_start]
        if len(df_pre) > 0:
            asset_vol_annual_pre = (
                df_pre["log_ret"].rolling(window=30 * 24).std().shift(1)
                * np.sqrt(365 * 24)
            )
            asset_vol_median = (
                float(asset_vol_annual_pre.median())
                if len(asset_vol_annual_pre.dropna()) > 0
                else _TARGET_VOL_FOR_SPAN
            )
        else:
            asset_vol_median = _TARGET_VOL_FOR_SPAN
        asset_vol_median = max(asset_vol_median, 0.1)
        span_scaled = _DEFAULT_SPAN_EMA * (asset_vol_median / _TARGET_VOL_FOR_SPAN)
        span_clamped = int(np.clip(span_scaled, _MIN_SPAN, _MAX_SPAN))
        # Compute EMA only on pre-holdout data, take last value, ffill
        if len(df_pre) > 0:
            asset_ema_pre = (
                df_pre["close"].ewm(span=span_clamped, adjust=False).mean()
            )
            last_ema = (
                float(asset_ema_pre.iloc[-1])
                if len(asset_ema_pre) > 0
                else float(df_pre["close"].mean())
            )
        else:
            # Edge case: no pre-holdout data, use mean of full df
            last_ema = float(df["close"].mean())
        # Forward-fill the constant into the full df
        df["macro_trend"] = np.where(
            df["close"] > last_ema, 1, -1
        ).astype(np.int32)
        # Phase 2.4 (v0.4.0): when apply_holdout_ema_to_full=False,
        # restore the dynamic-EMA macro_trend in the IS period so it
        # matches the WFA path. We compute the dynamic EMA on IS data
        # ONLY (df_pre) -- this mirrors what ``prepare_data_with_max_time``
        # would produce when called from the WFA path (where df is already
        # filtered by max_time to IS only). The holdout period keeps the
        # constant pre-holdout EMA so the holdout signal remains
        # contamination-free.
        if not apply_holdout_ema_to_full and len(df_pre) > 0:
            is_ema_dynamic = df_pre["close"].ewm(
                span=span_clamped, adjust=False
            ).mean()
            is_dynamic_trend = np.where(
                df_pre["close"] > is_ema_dynamic, 1, -1
            ).astype(np.int32)
            # Assign to the IS rows in the full df (align by time).
            # df_pre preserves the original df row order, so positional
            # assignment is safe (both arrays have the same length and
            # correspond to the IS portion).
            df.loc[df["time"] < btc_holdout_start, "macro_trend"] = (
                is_dynamic_trend
            )
    else:
        # Original behavior: compute on full df (WFA path)
        asset_vol_annual = (
            df["log_ret"].rolling(window=30 * 24).std().shift(1) * np.sqrt(365 * 24)
        )
        asset_vol_median = float(asset_vol_annual.median()) if len(asset_vol_annual.dropna()) > 0 else _TARGET_VOL_FOR_SPAN
        asset_vol_median = max(asset_vol_median, 0.1)  # avoid div by zero
        span_scaled = _DEFAULT_SPAN_EMA * (asset_vol_median / _TARGET_VOL_FOR_SPAN)
        span_clamped = int(np.clip(span_scaled, _MIN_SPAN, _MAX_SPAN))
        asset_ema = df["close"].ewm(span=span_clamped, adjust=False).mean()
        df["macro_trend"] = np.where(df["close"] > asset_ema, 1, -1).astype(np.int32)
    # Shift to prevent lookahead at entry bar (uses yesterday's regime)
    df["macro_trend"] = df["macro_trend"].shift(1)
    # drop intermediate asset_vol_annual (not needed downstream)
    asset_vol_annual = None  # free

    # -- Compute weekend flag early (independent of funding data) --
    df["is_weekend"] = df["time"].dt.dayofweek.isin([5, 6]).astype(np.int32)

    # -- Funding merge & is_funding_hour from actual data --
    if df_funding_raw is not None and not df_funding_raw.empty:
        df_fund = df_funding_raw[df_funding_raw["time"] <= max_time].copy()
        df = pd.merge(df, df_fund[["time", "funding_rate"]], on="time", how="left")

        fund_hour_counts = df_funding_raw["time"].dt.hour.value_counts()
        common_hours = sorted(
            fund_hour_counts[fund_hour_counts >= 5].index.tolist()
        )
        if len(common_hours) >= 3:
            gaps = np.diff(common_hours).astype(np.int32)
            pos = gaps[gaps > 0]
            if len(pos) > 0:
                gap_bins = np.bincount(pos)
                dominant = int(gap_bins.argmax())
                if dominant in (4, 8):
                    funding_hours_list = list(range(0, 24, dominant))
                else:
                    funding_hours_list = common_hours
            else:
                funding_hours_list = common_hours
        else:
            funding_hours_list = [0, 8, 16]

        df["is_funding_hour"] = (
            (df["time"].dt.hour.isin(funding_hours_list))
            & (df["time"].dt.minute == 0)
        ).astype(np.int32)

        n_funding_hours = int(df["is_funding_hour"].sum())
        df["funding_missing"] = (
            df["funding_rate"].isna() & (df["is_funding_hour"] == 1)
        ).astype(np.int32)
        n_missing = int(df["funding_missing"].sum())
        if n_missing > 0:
            pct = n_missing / n_funding_hours * 100 if n_funding_hours > 0 else 0.0
            log.warning(
                f"[Funding] {n_missing}/{n_funding_hours} funding-hour bars "
                f"({pct:.1f}%) genuinely missing -> imputed to 0."
            )
        df["funding_rate"] = df["funding_rate"].fillna(0.0)
    else:
        df["funding_rate"] = 0.0
        df["funding_missing"] = 0
        df["is_funding_hour"] = (
            (df["time"].dt.hour.isin([0, 8, 16])) & (df["time"].dt.minute == 0)
        ).astype(np.int32)

    # -- Pullback sniper features (only if strategy_type=1) --
    if strategy_type == STRATEGY_PULLBACK_SNIPER:
        # RSI: computed then shift(1) to prevent lookahead at entry.
        # rsi_14[i] uses data up to bar i-1 (yesterday), so decision
        # at close[i] is leakage-free.
        rsi_raw = _compute_rsi(df["close"], period=rsi_period)
        df["rsi_14"] = rsi_raw.shift(1).astype(np.float32)

        # EMA 20 for pullback reference (already shift(1) for anti-lookahead)
        df["ema_20"] = df["close"].ewm(span=_DEFAULT_EMA_PULLBACK, adjust=False).mean().shift(1)

        # Bullish reversal: current bar's close > open AND close > prev close
        # (uses only current and previous bar -- no lookahead)
        df["bullish_reversal"] = (
            (df["close"] > df["open"]) & (df["close"] > df["close"].shift(1))
        ).astype(np.int32)
        df["bearish_reversal"] = (
            (df["close"] < df["open"]) & (df["close"] < df["close"].shift(1))
        ).astype(np.int32)
    else:
        # vol_compression doesn't need these -- leave None/NaN
        df["rsi_14"] = np.nan
        df["ema_20"] = np.nan
        df["bullish_reversal"] = 0
        df["bearish_reversal"] = 0

    # -- Gap detection (before ffill) --
    gap_end_idxs = []
    contamination_window = WARMUP_BARS
    signal_cols = ["vol_pct_rank", "rvol", "atr", "hh_20", "ll_20", "ema_200"]
    if strategy_type == STRATEGY_PULLBACK_SNIPER:
        signal_cols.extend(["rsi_14", "ema_20", "bullish_reversal", "bearish_reversal"])
    if len(df) > 1:
        time_diffs = df["time"].diff().dropna()
        median_diff = time_diffs.median()
        if pd.notna(median_diff):
            expected = pd.Timedelta("1h")
            max_allowed = expected * 2
            large_gaps = time_diffs[time_diffs > max_allowed]
            if len(large_gaps) > 0:
                n_gaps = len(large_gaps)
                max_gap = large_gaps.max()
                gap_hours = max_gap.total_seconds() / 3600
                log.warning(
                    f"[GAP] {n_gaps} discontinuity(ies) detected (max gap = "
                    f"{gap_hours:.0f}h). "
                    f"ATR/rvol/vol_pct_rank are unreliable at those boundaries."
                )
                gap_end_idxs = df.index[df["time"].diff() > max_allowed].tolist()

    # -- ffill (fill feature-computation NaN) --
    # Phase 4 (v0.5.0): added comment explaining ordering.
    # ORDER MATTERS: ffill FIRST propagates the last valid pre-gap
    # value forward into the gap window. This is desired for normal
    # feature NaNs at the warmup head (not contamination -- we want
    # the engine to see the most recent valid feature).
    df.ffill(inplace=True)

    # -- Null-out gap contamination (after ffill) --
    # AFTER ffill, we null-out the contamination window at each gap
    # boundary. This overwrites the (now forward-filled) gap bars
    # with NaN, preventing pre-gap values from leaking into the gap
    # window. If we reversed the order (null-out THEN ffill), the
    # ffill would re-fill the gap window from pre-gap data, which
    # would be silent data leakage.
    for pos in gap_end_idxs:
        loc = df.index.get_loc(pos)
        end_loc = min(loc + contamination_window, len(df))
        for col in signal_cols:
            if col in df.columns:
                df.iloc[loc:end_loc, df.columns.get_loc(col)] = np.nan

    # -- BTC series gap check --
    if len(df_btc) > 1:
        btc_diffs = df_btc["time"].diff().dropna()
        if pd.notna(btc_diffs.median()):
            btc_large = btc_diffs[btc_diffs > pd.Timedelta(hours=2)]
            if len(btc_large) > 0:
                log.warning(
                    f"[BTC-GAP] {len(btc_large)} gap(s) in BTC series "
                    f"(max = {btc_large.max().total_seconds()/3600:.0f}h). "
                    f"macro_trend/ema_200 may be stale at merged timestamps."
                )

    return df
