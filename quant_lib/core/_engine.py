"""
Numba trade engine -- fast_trade_loop.

Extracted from Hans_Quant_Systems.py lines 656-967.

This module contains the JIT-compiled trade loop. No dependencies on
other core modules -- all parameters are passed explicitly as arrays/floats.

Supports three strategies via strategy_type:
  - 0 = vol_compression_breakout (original)
  - 1 = pullback_sniper (RSI + reversal)
  - 2 = funding_rate_carry (perp funding carry, mean-reversion)
"""

import numpy as np
from numba import njit
from numpy import ndarray
from dataclasses import dataclass
from typing import Tuple

# Sprint 2 fix: import the single-source-of-truth strategy type
# constants from ``core/_config.py`` instead of redeclaring. The
# constants are also re-exported from ``audit/hypothesis.py`` for
# the audit layer; importing from ``_config`` (a sibling leaf
# module) keeps the dependency graph acyclic.
from quant_lib.core._config import (  # noqa: I001
    STRATEGY_VOL_COMPRESSION,
    STRATEGY_PULLBACK_SNIPER,
    STRATEGY_FUNDING_RATE_CARRY,
)


@dataclass(frozen=True)
class EngineArgs:
    """Grouped arguments for :func:`fast_trade_loop`.

    Why this exists:
        The underlying ``@njit`` function takes 35 positional parameters
        (Numba does not support dataclass/object args). This dataclass
        captures the *natural groupings* -- market data, signal features,
        strategy params, cost model, trend-risk mults -- so call sites
        can be written/read more clearly. Use :meth:`as_tuple` to expand
        into the positional args expected by the JIT-compiled function.

    Fields
    ------
    market_data : tuple of 4 ndarrays
        (opens, highs, lows, closes)
    channel_features : tuple of 3 ndarrays
        (hh_20, ll_20, ema_200s)
    pullback_features : tuple of 3 ndarrays
        (rsi_14, bullish_reversal, bearish_reversal)
    signal_features : tuple of 4 ndarrays
        (vol_pct_rank, rvol, atrs, funding_pct_rank)
    auxiliary_features : tuple of 5 ndarrays
        (funding_rates, macro_vols, macro_trends, is_weekends, is_funding_hours)
    strategy_type : int
        0 = vol_compression, 1 = pullback_sniper, 2 = funding_rate_carry.
    thresholds : tuple of 8 floats
        (vol_pct_thresh, rvol_thresh, rsi_oversold, rsi_overbought,
         funding_entry_pct, funding_exit_low, funding_exit_high, _)
    integer_params : tuple of 4 ints
        (pullback_bars, bailout_bars, warmup_bars, _)
    exit_params : tuple of 2 floats
        (trail_atr, sl_mult)
    cost_model : tuple of 3 floats
        (fee_taker, weekend_penalty, stress_mult)
    flags : tuple of 4 ints (0/1)
        (use_rvol, use_ema, allow_long, allow_short)
    random_draws : ndarray
        Pre-generated uniform [0, 1) draws for slippage noise.
    trend_mults : tuple of 2 floats
        (trend_aligned_mult, trend_counter_mult)
    """
    market_data: Tuple[ndarray, ndarray, ndarray, ndarray]
    channel_features: Tuple[ndarray, ndarray, ndarray]
    pullback_features: Tuple[ndarray, ndarray, ndarray]
    signal_features: Tuple[ndarray, ndarray, ndarray, ndarray]
    auxiliary_features: Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]
    strategy_type: int
    thresholds: Tuple[float, float, float, float, float, float, float, float]
    integer_params: Tuple[int, int, int, int]
    exit_params: Tuple[float, float]
    cost_model: Tuple[float, float, float]
    flags: Tuple[int, int, int, int]
    random_draws: ndarray
    trend_mults: Tuple[float, float]

    def as_tuple(self) -> tuple:
        """Expand to the positional-args tuple expected by fast_trade_loop.

        Order MUST match the @njit function signature exactly.

        Backwards-compatibility (Phase 2 funding_rate_carry extension):
        - ``signal_features`` accepts 3-tuple (legacy, no funding_pct_rank)
          or 4-tuple (Phase 2+, with funding_pct_rank). When 3-tuple is
          passed, ``funding_pct_rank`` defaults to a constant 0.5 array
          (neutral funding regime -- never triggers entry/exit).
        - ``thresholds`` accepts 5-tuple (legacy) or 8-tuple (Phase 2+,
          with funding_entry_pct, funding_exit_low, funding_exit_high,
          padding). When 5-tuple is passed, the three funding thresholds
          default to safe neutral values (0.90, 0.40, 0.60).
        """
        opens, highs, lows, closes = self.market_data
        hh_20, ll_20, ema_200s = self.channel_features
        rsi_14, bullish_rev, bearish_rev = self.pullback_features
        # signal_features: 3-tuple legacy or 4-tuple Phase 2+
        if len(self.signal_features) == 4:
            vol_pct_rank, rvol, atrs, funding_pct_rank = self.signal_features
        else:
            vol_pct_rank, rvol, atrs = self.signal_features
            # Default: constant 0.5 (neutral funding regime).
            # Used only by tests/legacy callers that haven't migrated
            # to the Phase 2 4-tuple shape.
            funding_pct_rank = np.full(
                len(closes), 0.5, dtype=np.float64
            )
        funding_rates, macro_vols, macro_trends, is_weekends, is_funding_hours = (
            self.auxiliary_features
        )
        # thresholds: 5-tuple legacy or 8-tuple Phase 2+
        if len(self.thresholds) == 8:
            (v_thresh, r_thresh, rsi_oversold, rsi_overbought,
             funding_entry_pct, funding_exit_low, funding_exit_high, _) = (
                self.thresholds
            )
        else:
            v_thresh, r_thresh, rsi_oversold, rsi_overbought, _ = self.thresholds
            # Defaults: safe neutral funding thresholds
            # (0.90 entry, 0.40-0.60 neutral zone). Tests using the legacy
            # 5-tuple shape continue to work without modification.
            funding_entry_pct = 0.90
            funding_exit_low = 0.40
            funding_exit_high = 0.60
        pullback_bars, bailout_bars, warmup_bars, _ = self.integer_params
        trail_atr, sl_mult = self.exit_params
        fee_taker, weekend_penalty, stress_mult = self.cost_model
        use_rvol, use_ema, allow_long, allow_short = self.flags
        trend_aligned_mult, trend_counter_mult = self.trend_mults
        return (
            opens, highs, lows, closes,
            hh_20, ll_20, ema_200s,
            rsi_14, bullish_rev, bearish_rev,
            vol_pct_rank, rvol, atrs, funding_pct_rank,
            funding_rates, macro_vols, macro_trends, is_weekends, is_funding_hours,
            self.strategy_type,
            v_thresh, r_thresh,
            pullback_bars,
            trail_atr, sl_mult,
            bailout_bars, warmup_bars,
            fee_taker,
            use_rvol, use_ema, allow_long, allow_short,
            rsi_oversold, rsi_overbought,
            funding_entry_pct, funding_exit_low, funding_exit_high,
            weekend_penalty, stress_mult,
            self.random_draws,
            trend_aligned_mult, trend_counter_mult,
        )


@njit
def fast_trade_loop(
    opens: ndarray,
    highs: ndarray,
    lows: ndarray,
    closes: ndarray,
    hh_20: ndarray,
    ll_20: ndarray,
    ema_200s: ndarray,
    rsi_14: ndarray,
    bullish_reversal: ndarray,
    bearish_reversal: ndarray,
    vol_pct_rank: ndarray,
    rvol: ndarray,
    atrs: ndarray,
    funding_pct_rank: ndarray,
    funding_rates: ndarray,
    macro_vols: ndarray,
    macro_trends: ndarray,
    is_weekends: ndarray,
    is_funding_hours: ndarray,
    strategy_type: int,
    vol_pct_thresh: float,
    rvol_thresh: float,
    pullback_bars: int,
    trail_atr: float,
    sl_mult: float,
    bailout_bars: int,
    warmup_bars: int,
    fee_taker: float,
    use_rvol: int,
    use_ema: int,
    allow_long: int,
    allow_short: int,
    rsi_oversold: float,
    rsi_overbought: float,
    funding_entry_pct: float,
    funding_exit_low: float,
    funding_exit_high: float,
    weekend_penalty: float,
    stress_mult: float,
    random_draws: ndarray,
    trend_aligned_mult: float,
    trend_counter_mult: float,
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    n = len(closes)
    trade_pnl = np.zeros(n, dtype=np.float64)
    trade_idx_exit = np.zeros(n, dtype=np.int32)
    trade_idx_entry = np.zeros(n, dtype=np.int32)
    trade_dir = np.zeros(n, dtype=np.int32)

    t_macro_trend = np.zeros(n, dtype=np.int32)
    t_cb_vol = np.zeros(n, dtype=np.float64)
    t_entry_price = np.zeros(n, dtype=np.float64)
    t_exit_price = np.zeros(n, dtype=np.float64)
    t_sl_pct = np.zeros(n, dtype=np.float64)
    t_trend_risk_mult = np.ones(n, dtype=np.float64)

    trade_count = 0
    trade_state, entry_bar_idx = 0, 0
    sl_level, entry_price, sl_dist_val = 0.0, 0.0, 0.0
    highest_price, lowest_price, current_entry_slip = 0.0, 0.0, 0.0
    current_trend_mult = 1.0

    setup_dir = 0
    setup_price = 0.0
    setup_timer = 0
    pulled_back = False

    rng_idx = 0
    max_draws = len(random_draws)

    for i in range(warmup_bars, n):
        # -- Exit logic --
        if trade_state != 0:
            bars_held = i - entry_bar_idx
            exit_triggered, exit_price_val = False, 0.0
            if trade_state == 1:
                highest_price = max(highest_price, highs[i])
                sl_level = max(sl_level, highest_price - (atrs[i] * trail_atr))
                if lows[i] < sl_level:
                    exit_triggered, exit_price_val = True, sl_level
                # Bracket TP for pullback_sniper: exit at hh_20 (resistance)
                if not exit_triggered and strategy_type == STRATEGY_PULLBACK_SNIPER:
                    if highs[i] >= hh_20[i]:
                        exit_triggered, exit_price_val = True, hh_20[i]
                # Neutral-zone exit for funding_rate_carry: exit when
                # funding_pct_rank returns to neutral [exit_low, exit_high].
                # We longed when funding was low; now funding has reverted
                # to neutral -- our carry thesis is realized.
                if not exit_triggered and strategy_type == STRATEGY_FUNDING_RATE_CARRY:
                    if funding_pct_rank[i] >= funding_exit_low and funding_pct_rank[i] <= funding_exit_high:
                        exit_triggered, exit_price_val = True, closes[i]
            else:
                lowest_price = min(lowest_price, lows[i])
                sl_level = min(sl_level, lowest_price + (atrs[i] * trail_atr))
                if highs[i] > sl_level:
                    exit_triggered, exit_price_val = True, sl_level
                # Bracket TP for pullback_sniper: exit at ll_20 (support)
                if not exit_triggered and strategy_type == STRATEGY_PULLBACK_SNIPER:
                    if lows[i] <= ll_20[i]:
                        exit_triggered, exit_price_val = True, ll_20[i]
                # Neutral-zone exit for funding_rate_carry (short side):
                # we shorted when funding was high; funding has now
                # reverted to neutral -- carry thesis realized.
                if not exit_triggered and strategy_type == STRATEGY_FUNDING_RATE_CARRY:
                    if funding_pct_rank[i] >= funding_exit_low and funding_pct_rank[i] <= funding_exit_high:
                        exit_triggered, exit_price_val = True, closes[i]
            if not exit_triggered and bars_held >= bailout_bars:
                exit_triggered, exit_price_val = True, closes[i]

            if exit_triggered:
                accumulated_funding = 0.0
                for idx in range(entry_bar_idx, i + 1):
                    if is_funding_hours[idx] == 1:
                        accumulated_funding += funding_rates[idx]
                funding_impact_pct = (
                    (accumulated_funding * 100.0)
                    if trade_state == 1
                    else -(accumulated_funding * 100.0)
                )

                atr_pct = (atrs[i] / closes[i]) * 100.0 if closes[i] > 0.0 else 0.0
                base_exit_slip = min(max(0.010 * (atr_pct / 0.5), 0.005), 0.15)

                # NOTE: Weekend penalty application is asymmetric (H-4 doc).
                # Exit slippage penalty applies if the trade HELD OVER any
                # weekend in [entry_bar_idx, i+1]. This captures gap risk
                # realized at exit. Entry slippage (in entry blocks below)
                # applies only if the entry BAR itself is a weekend bar,
                # capturing low-liquidity entry conditions. These two
                # mechanisms cover distinct weekend-related cost sources.
                held_over_weekend = np.sum(is_weekends[entry_bar_idx : i + 1]) > 0
                pen = weekend_penalty if held_over_weekend else 1.0

                random_stress = 1.0 + (
                    random_draws[rng_idx % max_draws] * (stress_mult - 1.0)
                )
                rng_idx += 1

                exit_slip = base_exit_slip * random_stress * pen

                gross_r = (
                    ((exit_price_val - entry_price) / sl_dist_val)
                    if trade_state == 1
                    else ((entry_price - exit_price_val) / sl_dist_val)
                )
                sl_pct_pct = (sl_dist_val / entry_price) * 100.0 if entry_price > 0.0 else 0.0

                cost_total = (
                    (fee_taker * 2) + current_entry_slip + exit_slip
                ) / sl_pct_pct + (funding_impact_pct / sl_pct_pct)
                cost_total = min(cost_total, 5.0)
                trade_pnl[trade_count] = gross_r - cost_total
                trade_idx_exit[trade_count] = i
                trade_idx_entry[trade_count] = entry_bar_idx
                trade_dir[trade_count] = trade_state
                t_macro_trend[trade_count] = macro_trends[entry_bar_idx]
                t_cb_vol[trade_count] = macro_vols[entry_bar_idx]
                t_entry_price[trade_count] = entry_price
                t_exit_price[trade_count] = exit_price_val
                t_sl_pct[trade_count] = sl_pct_pct / 100.0
                t_trend_risk_mult[trade_count] = current_trend_mult
                trade_count += 1
                trade_state = 0
                continue

        # -- Entry logic --
        if trade_state == 0:
            is_macro_safe = True

            if setup_timer > 0:
                setup_timer -= 1
                if setup_dir == 1:
                    if closes[i] < hh_20[i]:
                        pulled_back = True
                    is_ema_ok = closes[i] > ema_200s[i] if use_ema else True
                    if (
                        pulled_back
                        and closes[i] > setup_price
                        and is_ema_ok
                        and is_macro_safe
                    ):
                        if i + 1 < n:
                            trade_state = 1
                            entry_price = closes[i]
                            sl_dist_val = atrs[i] * sl_mult
                            sl_level = entry_price - sl_dist_val
                            highest_price = lowest_price = entry_price
                            entry_bar_idx = i
                            base_entry_slip = min(
                                max(
                                    0.010 * ((atrs[i] / entry_price) * 100.0 / 0.5),
                                    0.005,
                                ),
                                0.10,
                            )
                            random_stress = 1.0 + (
                                random_draws[rng_idx % max_draws] * (stress_mult - 1.0)
                            )
                            rng_idx += 1
                            pen_en = weekend_penalty if is_weekends[i] == 1 else 1.0
                            current_entry_slip = (
                                base_entry_slip * random_stress * pen_en
                            )
                            trend_aligned = (macro_trends[entry_bar_idx] == 1)
                            current_trend_mult = (
                                trend_aligned_mult if trend_aligned else trend_counter_mult
                            )
                            setup_timer = 0
                            setup_dir = 0
                            pulled_back = False
                elif setup_dir == -1:
                    if closes[i] > ll_20[i]:
                        pulled_back = True
                    is_ema_ok = closes[i] < ema_200s[i] if use_ema else True
                    if (
                        pulled_back
                        and closes[i] < setup_price
                        and is_ema_ok
                        and is_macro_safe
                    ):
                        if i + 1 < n:
                            trade_state = -1
                            entry_price = closes[i]
                            sl_dist_val = atrs[i] * sl_mult
                            sl_level = entry_price + sl_dist_val
                            highest_price = lowest_price = entry_price
                            entry_bar_idx = i
                            base_entry_slip = min(
                                max(
                                    0.010 * ((atrs[i] / entry_price) * 100.0 / 0.5),
                                    0.005,
                                ),
                                0.10,
                            )
                            random_stress = 1.0 + (
                                random_draws[rng_idx % max_draws] * (stress_mult - 1.0)
                            )
                            rng_idx += 1
                            pen_en = weekend_penalty if is_weekends[i] == 1 else 1.0
                            current_entry_slip = (
                                base_entry_slip * random_stress * pen_en
                            )
                            trend_aligned = (macro_trends[entry_bar_idx] == -1)
                            current_trend_mult = (
                                trend_aligned_mult if trend_aligned else trend_counter_mult
                            )
                            setup_timer = 0
                            setup_dir = 0
                            pulled_back = False
                if setup_timer == 0 and trade_state == 0:
                    setup_dir = 0
            else:
                if strategy_type == STRATEGY_VOL_COMPRESSION:
                    # Vol compression: require compressed vol + rvol confirmation
                    # before setting up a breakout watch.
                    is_compressed = vol_pct_rank[i] < vol_pct_thresh
                    is_vol_confirmed = rvol[i] > rvol_thresh if use_rvol else True
                    if is_compressed and is_vol_confirmed and is_macro_safe:
                        if closes[i] > hh_20[i] and (
                            closes[i] > ema_200s[i] if use_ema else True
                        ):
                            setup_dir = 1
                            setup_price = closes[i]
                            setup_timer = pullback_bars
                            pulled_back = False
                        elif closes[i] < ll_20[i] and (
                            closes[i] < ema_200s[i] if use_ema else True
                        ):
                            setup_dir = -1
                            setup_price = closes[i]
                            setup_timer = pullback_bars
                            pulled_back = False
                elif strategy_type == STRATEGY_PULLBACK_SNIPER:
                    # Pullback Sniper: RSI oversold/overbought + reversal candle
                    # No setup/pullback wait -- single-bar entry (Q2=A1).
                    # Independent of vol_compression conditions: the
                    # RSI+reversal signal is the primary trigger.
                    is_macro_safe = True
                    if allow_long == 1 and rsi_14[i] < rsi_oversold and bullish_reversal[i] == 1:
                        is_ema_ok = closes[i] > ema_200s[i] if use_ema == 1 else True
                        if is_ema_ok and is_macro_safe:
                            if i + 1 < n:
                                trade_state = 1
                                entry_price = closes[i]
                                sl_dist_val = atrs[i] * sl_mult
                                sl_level = entry_price - sl_dist_val
                                highest_price = lowest_price = entry_price
                                entry_bar_idx = i
                                base_entry_slip = min(
                                    max(
                                        0.010 * ((atrs[i] / entry_price) * 100.0 / 0.5),
                                        0.005,
                                    ),
                                    0.10,
                                )
                                random_stress = 1.0 + (
                                    random_draws[rng_idx % max_draws] * (stress_mult - 1.0)
                                )
                                rng_idx += 1
                                pen_en = weekend_penalty if is_weekends[i] == 1 else 1.0
                                current_entry_slip = (
                                    base_entry_slip * random_stress * pen_en
                                )
                                trend_aligned = (macro_trends[entry_bar_idx] == 1)
                                current_trend_mult = (
                                    trend_aligned_mult if trend_aligned else trend_counter_mult
                                )
                    elif allow_short == 1 and rsi_14[i] > rsi_overbought and bearish_reversal[i] == 1:
                        is_ema_ok = closes[i] < ema_200s[i] if use_ema == 1 else True
                        if is_ema_ok and is_macro_safe:
                            if i + 1 < n:
                                trade_state = -1
                                entry_price = closes[i]
                                sl_dist_val = atrs[i] * sl_mult
                                sl_level = entry_price + sl_dist_val
                                highest_price = lowest_price = entry_price
                                entry_bar_idx = i
                                base_entry_slip = min(
                                    max(
                                        0.010 * ((atrs[i] / entry_price) * 100.0 / 0.5),
                                        0.005,
                                    ),
                                    0.10,
                                )
                                random_stress = 1.0 + (
                                    random_draws[rng_idx % max_draws] * (stress_mult - 1.0)
                                )
                                rng_idx += 1
                                pen_en = weekend_penalty if is_weekends[i] == 1 else 1.0
                                current_entry_slip = (
                                    base_entry_slip * random_stress * pen_en
                                )
                                trend_aligned = (macro_trends[entry_bar_idx] == -1)
                                current_trend_mult = (
                                    trend_aligned_mult if trend_aligned else trend_counter_mult
                                )
                elif strategy_type == STRATEGY_FUNDING_RATE_CARRY:
                    # Funding rate carry: enter when funding_pct_rank crosses
                    # an extreme threshold, exit when it reverts to neutral
                    # zone. No setup/pullback wait (single-bar entry, like
                    # pullback_sniper). The neutral-zone exit is the funding-
                    # rate analog of pullback_sniper's TP bracket.
                    #
                    # Entry conditions:
                    #   funding_pct_rank[i] > funding_entry_pct -> SHORT
                    #     (longs are paying; we sell carry to them)
                    #   funding_pct_rank[i] < (1 - funding_entry_pct) -> LONG
                    #     (shorts are paying; we buy carry from them)
                    is_macro_safe = True
                    if allow_short == 1 and funding_pct_rank[i] > funding_entry_pct:
                        is_ema_ok = closes[i] < ema_200s[i] if use_ema == 1 else True
                        if is_ema_ok and is_macro_safe:
                            if i + 1 < n:
                                trade_state = -1
                                entry_price = closes[i]
                                sl_dist_val = atrs[i] * sl_mult
                                sl_level = entry_price + sl_dist_val
                                highest_price = lowest_price = entry_price
                                entry_bar_idx = i
                                base_entry_slip = min(
                                    max(
                                        0.010 * ((atrs[i] / entry_price) * 100.0 / 0.5),
                                        0.005,
                                    ),
                                    0.10,
                                )
                                random_stress = 1.0 + (
                                    random_draws[rng_idx % max_draws] * (stress_mult - 1.0)
                                )
                                rng_idx += 1
                                pen_en = weekend_penalty if is_weekends[i] == 1 else 1.0
                                current_entry_slip = (
                                    base_entry_slip * random_stress * pen_en
                                )
                                # Shorts profit in downtrend (macro=-1)
                                trend_aligned = (macro_trends[entry_bar_idx] == -1)
                                current_trend_mult = (
                                    trend_aligned_mult if trend_aligned else trend_counter_mult
                                )
                    elif allow_long == 1 and funding_pct_rank[i] < (1.0 - funding_entry_pct):
                        is_ema_ok = closes[i] > ema_200s[i] if use_ema == 1 else True
                        if is_ema_ok and is_macro_safe:
                            if i + 1 < n:
                                trade_state = 1
                                entry_price = closes[i]
                                sl_dist_val = atrs[i] * sl_mult
                                sl_level = entry_price - sl_dist_val
                                highest_price = lowest_price = entry_price
                                entry_bar_idx = i
                                base_entry_slip = min(
                                    max(
                                        0.010 * ((atrs[i] / entry_price) * 100.0 / 0.5),
                                        0.005,
                                    ),
                                    0.10,
                                )
                                random_stress = 1.0 + (
                                    random_draws[rng_idx % max_draws] * (stress_mult - 1.0)
                                )
                                rng_idx += 1
                                pen_en = weekend_penalty if is_weekends[i] == 1 else 1.0
                                current_entry_slip = (
                                    base_entry_slip * random_stress * pen_en
                                )
                                # Longs profit in uptrend (macro=1)
                                trend_aligned = (macro_trends[entry_bar_idx] == 1)
                                current_trend_mult = (
                                    trend_aligned_mult if trend_aligned else trend_counter_mult
                                )

    # -- Force-close remaining position at array end --
    if trade_state != 0:
        accumulated_funding = 0.0
        for idx in range(entry_bar_idx, n):
            if is_funding_hours[idx] == 1:
                accumulated_funding += funding_rates[idx]
        funding_impact_pct = (
            (accumulated_funding * 100.0)
            if trade_state == 1
            else -(accumulated_funding * 100.0)
        )

        exit_price_val = closes[n - 1]
        atr_pct = (atrs[n - 1] / closes[n - 1]) * 100.0 if closes[n - 1] > 0.0 else 0.0
        base_exit_slip = min(max(0.010 * (atr_pct / 0.5), 0.005), 0.15)

        held_over_weekend = np.sum(is_weekends[entry_bar_idx:n]) > 0
        pen = weekend_penalty if held_over_weekend else 1.0

        random_stress = 1.0 + (
            random_draws[rng_idx % max_draws] * (stress_mult - 1.0)
        )
        rng_idx += 1
        exit_slip = base_exit_slip * random_stress * pen

        gross_r = (
            ((exit_price_val - entry_price) / sl_dist_val)
            if trade_state == 1
            else ((entry_price - exit_price_val) / sl_dist_val)
        )
        sl_pct_pct = (sl_dist_val / entry_price) * 100.0 if entry_price > 0.0 else 0.0

        cost_total = (
            (fee_taker * 2) + current_entry_slip + exit_slip
        ) / sl_pct_pct + (funding_impact_pct / sl_pct_pct)
        cost_total = min(cost_total, 5.0)

        trade_pnl[trade_count] = gross_r - cost_total
        trade_idx_exit[trade_count] = n - 1
        trade_idx_entry[trade_count] = entry_bar_idx
        trade_dir[trade_count] = trade_state
        t_macro_trend[trade_count] = macro_trends[entry_bar_idx]
        t_cb_vol[trade_count] = macro_vols[entry_bar_idx]
        t_entry_price[trade_count] = entry_price
        t_exit_price[trade_count] = exit_price_val
        t_sl_pct[trade_count] = sl_pct_pct / 100.0
        t_trend_risk_mult[trade_count] = current_trend_mult
        trade_count += 1
        trade_state = 0

    return (
        trade_pnl[:trade_count],
        trade_idx_entry[:trade_count],
        trade_idx_exit[:trade_count],
        trade_dir[:trade_count],
        t_macro_trend[:trade_count],
        t_cb_vol[:trade_count],
        t_entry_price[:trade_count],
        t_exit_price[:trade_count],
        t_sl_pct[:trade_count],
        t_trend_risk_mult[:trade_count],
    )


@njit
def simulate_trailing_stop_trade(
    highs: ndarray,
    lows: ndarray,
    closes: ndarray,
    atrs: ndarray,
    funding_rates: ndarray,
    is_funding_hours: ndarray,
    is_weekends: ndarray,
    macro_trends: ndarray,
    entry_idx: int,
    direction: int,
    sl_mult: float,
    trail_atr: float,
    bailout_bars: int,
    fee_taker: float,
    weekend_penalty: float,
    stress_mult: float,
    random_draw: float,
    trend_aligned_mult: float,
    trend_counter_mult: float,
    hh_20: ndarray = None,
    ll_20: ndarray = None,
    use_bracket: int = 0,
) -> Tuple[int, float, float, float]:
    """Simulate a single forced-entry trade with trailing stop exit.

    Used by SPA to replicate the strategy's exit mechanism on random entries.
    Returns (exit_idx, exit_price, r_net, trend_risk_mult).

    If use_bracket=1 and hh_20/ll_20 are provided, also exits at TP level
    (hh_20 for long, ll_20 for short) -- matches pullback_sniper bracket.
    """
    n = len(closes)
    if entry_idx >= n or entry_idx < 0:
        return -1, 0.0, 0.0, 1.0

    entry_price = closes[entry_idx]
    sl_dist = atrs[entry_idx] * sl_mult
    sl_level = (
        entry_price - sl_dist if direction == 1 else entry_price + sl_dist
    )
    highest_price = entry_price
    lowest_price = entry_price

    atr_pct_entry = (atrs[entry_idx] / entry_price) * 100.0
    base_entry_slip = min(max(0.010 * (atr_pct_entry / 0.5), 0.005), 0.10)
    pen_en = weekend_penalty if is_weekends[entry_idx] == 1 else 1.0
    # NOTE (0.2.2): Mirror fast_trade_loop's exact entry_slip formula so SPA
    # null distribution mirrors real cost. Previous code was missing the
    # "1.0 +" prefix, making entry_slip = base * random_draw * stress_mult
    # (no baseline). Real trades use base * (1.0 + random_draw*(stress-1)).
    random_stress = 1.0 + (random_draw * (stress_mult - 1.0))
    entry_slip = base_entry_slip * random_stress * pen_en

    trend_aligned = (
        (direction == 1 and macro_trends[entry_idx] == 1)
        or (direction == -1 and macro_trends[entry_idx] == -1)
    )
    trend_risk_mult = (
        trend_aligned_mult if trend_aligned else trend_counter_mult
    )

    exit_idx = -1
    exit_price_val = 0.0
    # Phase 3 (v0.4.1): renamed from `max_idx` (misleading -- sounds like
    # "maximum seen", but it's actually the UPPER bound of the bailout
    # window). `exit_limit_idx` makes the intent explicit.
    exit_limit_idx = min(entry_idx + bailout_bars, n)

    for i in range(entry_idx, exit_limit_idx):
        if direction == 1:
            highest_price = max(highest_price, highs[i])
            sl_level = max(sl_level, highest_price - (atrs[i] * trail_atr))
            if lows[i] < sl_level:
                exit_idx = i
                exit_price_val = sl_level
                break
            # Bracket TP for pullback_sniper
            if use_bracket == 1 and hh_20 is not None:
                if highs[i] >= hh_20[i]:
                    exit_idx = i
                    exit_price_val = hh_20[i]
                    break
        else:
            lowest_price = min(lowest_price, lows[i])
            sl_level = min(sl_level, lowest_price + (atrs[i] * trail_atr))
            if highs[i] > sl_level:
                exit_idx = i
                exit_price_val = sl_level
                break
            # Bracket TP for pullback_sniper
            if use_bracket == 1 and ll_20 is not None:
                if lows[i] <= ll_20[i]:
                    exit_idx = i
                    exit_price_val = ll_20[i]
                    break

    if exit_idx == -1:
        exit_idx = exit_limit_idx - 1
        exit_price_val = closes[exit_idx]

    accumulated_funding = 0.0
    for idx in range(entry_idx, exit_idx + 1):
        if is_funding_hours[idx] == 1:
            accumulated_funding += funding_rates[idx]
    funding_impact_pct = (
        (accumulated_funding * 100.0) if direction == 1
        else -(accumulated_funding * 100.0)
    )

    atr_pct_exit = (atrs[exit_idx] / exit_price_val) * 100.0
    base_exit_slip = min(max(0.010 * (atr_pct_exit / 0.5), 0.005), 0.15)
    held_over_weekend = False
    for idx in range(entry_idx, exit_idx + 1):
        if is_weekends[idx] == 1:
            held_over_weekend = True
            break
    exit_pen = weekend_penalty if held_over_weekend else 1.0
    exit_slip = base_exit_slip * (1.0 + random_draw * (stress_mult - 1.0)) * exit_pen

    gross_r = (
        ((exit_price_val - entry_price) / sl_dist) if direction == 1
        else ((entry_price - exit_price_val) / sl_dist)
    )
    sl_pct_pct = (sl_dist / entry_price) * 100.0 if entry_price > 0.0 else 0.0

    cost_total = (
        (fee_taker * 2.0) + entry_slip + exit_slip
    ) / sl_pct_pct + (funding_impact_pct / sl_pct_pct)
    cost_total = min(cost_total, 5.0)
    r_net = gross_r - cost_total

    return exit_idx, exit_price_val, r_net, trend_risk_mult
