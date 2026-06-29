"""
Portfolio SPA -- Superior Predictive Ability test with circular permutation.

Extracted from Hans_Quant_Systems.py:
  - portfolio_spa (lines 1489-1844)
"""

import numpy as np
from numpy import ndarray
import pandas as pd

from quant_lib.core._config import STATIC, DEFAULTS
from quant_lib.core._logging import log, console
from quant_lib.core._portfolio import _trade_key, simulate_full_portfolio
from quant_lib.core._engine import simulate_trailing_stop_trade

_AssetDataDict = dict[str, pd.DataFrame]


def portfolio_spa(
    observed_trades: list[dict],
    asset_data: _AssetDataDict,
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
    # NOTE (0.2.2): Was hardcoded 2.5 (stale, predates 0.2.0 default change).
    # Now mirrors DEFAULTS["stress_test_multiplier"] so direct spa_test() callers
    # get the same cost model as the WFA path (which passes DEFAULTS through).
    stress_mult: float = DEFAULTS["stress_test_multiplier"],
    weekend_penalty: float = DEFAULTS["weekend_liquidity_penalty"],
    asset_risk_weights: dict[str, float] | None = None,
) -> tuple[float, ndarray, float]:
    """Portfolio-level SPA (Superior Predictive Ability) test.

    Tests whether the observed strategy edge is genuine or random,
    using time-anchored circular permutation across all assets.
    """
    aw = asset_risk_weights  # None is allowed; portfolio sim will skip per-asset CB

    if not observed_trades:
        return initial_capital, np.zeros(n_iters), 1.0

    # Defensive filter -- trades MUST have sl_mult
    n_no_sl = sum(1 for t in observed_trades if t.get("sl_mult") is None)
    if n_no_sl > 0:
        observed_trades = [t for t in observed_trades if t.get("sl_mult") is not None]
        if not observed_trades:
            return initial_capital, np.zeros(n_iters), 1.0
        log.warning(
            f"SPA: {n_no_sl}/{n_no_sl + len(observed_trades)} trades "
            f"missing sl_mult -- excluded from permutation."
        )

    rng_spa = np.random.default_rng(rng_seed)

    # Pre-compute correlation data ONCE for all 500+ SPA iterations
    _precomputed_sym_list = sorted(daily_close_matrix.keys())
    _precomputed_daily_returns = None
    if len(_precomputed_sym_list) >= 2:
        _ret_series_list = []
        for _sym in _precomputed_sym_list:
            _s = pd.Series(daily_close_matrix[_sym]).sort_index().pct_change().dropna()
            _ret_series_list.append(_s)
        _ret_df = pd.concat(_ret_series_list, axis=1, keys=_precomputed_sym_list).dropna()
        if len(_ret_df) > 30:
            _precomputed_daily_returns = _ret_df
        else:
            _precomputed_daily_returns = None
            _precomputed_sym_list = []
    else:
        _precomputed_sym_list = []

    # Cross-iteration correlation cache shared with simulate_full_portfolio.
    # Same convention as in _portfolio.py (dict | None). Annotation
    # needed because mypy can't infer dict literal in this scope.
    _shared_corr_cache: dict | None = {}

    # 1. Simulate observed trades for baseline equity
    observed_final_equity = simulate_full_portfolio(
        observed_trades,
        initial_capital,
        leverage,
        mm_pct,
        position_limit,
        cb_hard_cooldown_hours,
        fixed_cb_threshold,
        daily_close_matrix,
        aw,
        end_date=end_date,
        liquidation_fee_pct=liquidation_fee_pct,
        daily_hl_matrix=daily_hl_matrix,
        _precomputed_daily_returns=_precomputed_daily_returns,
        _precomputed_sym_list=_precomputed_sym_list,
        _shared_corr_cache=_shared_corr_cache,
    )[0]

    # Temporal anchoring: preserve cross-asset co-occurrence correlation
    first_entry = min(t["entry_time"] for t in observed_trades)
    relative_offsets = {}
    durations_h = {}
    trade_keys = []

    for t in observed_trades:
        tid = _trade_key(t)
        trade_keys.append(tid)
        relative_offsets[tid] = (t["entry_time"] - first_entry).total_seconds() / 3600.0
        dur = int((t["exit_time"] - t["entry_time"]).total_seconds() / 3600.0)
        durations_h[tid] = max(1, dur)

    # Valid time range for anchor
    global_start = max(v["time"].iloc[0] for v in asset_data.values())

    _data_end_date = pd.Timestamp(end_date)
    _data_lasts = {sym: v["time"].iloc[-1] for sym, v in asset_data.items()}
    _min_asset_end = min(_data_lasts.values())
    global_end = max(_data_end_date, _min_asset_end)
    _gap_days = (global_end - _min_asset_end).total_seconds() / 86400
    if _gap_days > 7:
        log.warning(
            f"SPA data gap: asset min end = {_min_asset_end.date()}, "
            f"END_DATE = {_data_end_date.date()} "
            f"(gap={_gap_days:.0f}d). "
            f"Some SPA iterations may have fewer trades for short-data assets."
        )
    global_start_np = np.datetime64(global_start)
    total_hours = (global_end - global_start) / np.timedelta64(1, "h")

    span_hours = max(relative_offsets.values()) + max(durations_h.values())
    max_anchor = max(0, total_hours - span_hours)

    random_equities = np.zeros(n_iters) if n_iters > 0 else np.array([])

    # Defensive guard (Phase 3.5 B1): if observed_final_equity is NaN
    # (e.g. from numerical issues in simulate_full_portfolio), the SPA
    # p-value comparison `random_equities >= NaN` would always be False,
    # giving n_exceed=0 -> p_value=1/(N+1) (misleadingly "significant").
    # Return NaN p-value explicitly so callers can detect the issue.
    if np.isnan(observed_final_equity):
        log.warning(
            "SPA: observed_final_equity is NaN (numerical issue in "
            "portfolio simulation). Returning NaN p-value."
        )
        return observed_final_equity, random_equities, float("nan")

    anchor_ratio = span_hours / total_hours * 100 if total_hours > 0 else 0
    log.info(
        f"SPA anchor space: total_hours={total_hours:.0f}h, "
        f"span_hours={span_hours:.0f}h, max_anchor={max_anchor:.0f}h, "
        f"anchor_ratio={anchor_ratio:.1f}%"
    )

    # Degenerate anchor guard
    if total_hours > 0 and span_hours >= total_hours * 0.8:
        log.warning(
            f"SPA DEGENERATE: anchor_ratio={anchor_ratio:.0f}% "
            f"(span={span_hours:.0f}h / total={total_hours:.0f}h >= 80%). "
            f"Circular permutation creates near-identical null -> "
            f"p-value UNRELIABLE. Returning NaN."
        )
        return observed_final_equity, random_equities, float("nan")

    times_hours_map = {
        sym: (asset_data[sym]["time"].values - global_start_np) / np.timedelta64(1, "h")
        for sym in asset_data
    }

    if total_hours <= 0:
        log.error("SPA: total_hours <= 0, no valid time range for permutation.")
        return observed_final_equity, random_equities, float("nan")

    for it in range(n_iters):
        anchor_offset = rng_spa.uniform(0, total_hours)
        random_trades = []

        for i, t in enumerate(observed_trades):
            sym = t["symbol"]
            df_sym = asset_data[sym]
            tid = trade_keys[i]

            target_entry_hour = (anchor_offset + relative_offsets[tid]) % total_hours
            dur_h = durations_h[tid]

            idx = int(np.searchsorted(times_hours_map[sym], target_entry_hour))
            max_valid_idx = len(df_sym) - dur_h - 1
            if max_valid_idx < 0:
                continue
            if idx > max_valid_idx:
                idx = idx % (max_valid_idx + 1)

            sl_mult_val = t.get("sl_mult", 1.5)
            trail_atr_val = t.get("trail_atr", 3.0)
            direction = int(t.get("trade_dir", 1))

            rand_draw = float(rng_spa.random())

            exit_idx, exit_price, net_r, trend_mult = simulate_trailing_stop_trade(
                df_sym["high"].values,
                df_sym["low"].values,
                df_sym["close"].values,
                df_sym["atr"].values,
                df_sym["funding_rate"].values,
                df_sym["is_funding_hour"].values,
                df_sym["is_weekend"].values,
                df_sym["macro_trend"].values,
                idx,
                direction,
                sl_mult_val,
                trail_atr_val,
                DEFAULTS["bailout_bars"],
                fee_taker,
                weekend_penalty,
                stress_mult,
                rand_draw,
                DEFAULTS["trend_aligned_risk_mult"],
                DEFAULTS["trend_counter_risk_mult"],
            )

            if exit_idx < 0:
                continue

            entry_price = df_sym["close"].iloc[idx]
            atr_entry = df_sym["atr"].iloc[idx]
            sl_dist = atr_entry * sl_mult_val
            sl_pct = sl_dist / entry_price

            random_trades.append({
                "entry_time": df_sym["time"].iloc[idx],
                "exit_time": df_sym["time"].iloc[exit_idx],
                "symbol": sym,
                "trade_dir": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "sl_pct": sl_pct,
                "r_net": net_r,
                "risk_weight": t.get(
                    "risk_weight",
                    aw.get(sym, 0.005) if aw else 0.005,
                ),
                "trend_risk_mult": trend_mult,
            })

        if random_trades:
            eq, _, _, _ = simulate_full_portfolio(
                random_trades,
                initial_capital,
                leverage,
                mm_pct,
                position_limit,
                cb_hard_cooldown_hours,
                fixed_cb_threshold,
                daily_close_matrix,
                aw,
                end_date=end_date,
                liquidation_fee_pct=liquidation_fee_pct,
                daily_hl_matrix=daily_hl_matrix,
                _precomputed_daily_returns=_precomputed_daily_returns,
                _precomputed_sym_list=_precomputed_sym_list,
                _shared_corr_cache=_shared_corr_cache,
            )
            random_equities[it] = eq
        else:
            random_equities[it] = initial_capital

        if verbose and (it + 1) % max(1, n_iters // 10) == 0:
            pct = (it + 1) / n_iters * 100
            console.print(f"   SPA progress: {pct:.0f}% ({it+1}/{n_iters})")

    # Phipson-Bell (2010) add-one correction. Note: this is NOT a
    # proper Hansen 2005 SPA null -- the null here is uniform time-
    # anchored permutation of observed trades, which preserves cross-
    # asset co-occurrence structure. This correction is the standard
    # add-one for permutation tests (Phipson & Smyth 2010). The
    # previous label "Davé 2008" was incorrect; the formula is the
    # same but the attribution is to Phipson-Bell.
    n_exceed = int(np.sum(random_equities >= observed_final_equity))
    p_value = (n_exceed + 1) / (n_iters + 1)
    return observed_final_equity, random_equities, p_value
