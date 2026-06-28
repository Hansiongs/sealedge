"""
Portfolio simulation -- MTM, margin, liquidation, CB, SPA baseline.

Extracted from Hans_Quant_Systems.py:
  - _trade_key (lines 973-977)
  - _mtm_and_margin_check (lines 980-1073)
  - simulate_full_portfolio (lines 1076-1483)
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Any

from quant_lib.core._config import STATIC

_LiqTrade = dict[str, Any]
_Position = dict[str, Any]
_RejectReasons = dict[str, int]


def _trade_key(t: dict[str, Any]) -> tuple[str, Any, Any, int]:
    """Composite key for trade deduplication -- replaces fragile id().
    Uses (symbol, entry_time, exit_time, trade_dir) which is unique per trade
    in a given backtest run."""
    return (t["symbol"], t["entry_time"], t["exit_time"], t.get("trade_dir", 0))


def _mtm_and_margin_check(
    pos: _Position, date: pd.Timestamp, daily_close_matrix: dict[str, dict],
    daily_hl_matrix: dict[str, dict] | None,
    mm_pct: float, liquidation_fee_pct: float, asset_risk_weights: dict[str, float],
) -> tuple[bool, float, _LiqTrade | None, float]:
    """Process MTM and margin call for one position on a given date."""
    if hasattr(date, "tz") and date.tz is not None:
        date = date.tz_localize(None)
    sym = pos["trade"]["symbol"]
    cur_price = daily_close_matrix.get(sym, {}).get(date, pos["trade"]["entry_price"])
    dir_mult = 1 if pos["trade"]["trade_dir"] == 1 else -1
    pnl = (
        ((cur_price - pos["trade"]["entry_price"]) / pos["trade"]["entry_price"])
        * pos["notional"]
        * dir_mult
    )

    liq_triggered = False
    maintenance_margin = pos["notional"] * mm_pct

    if daily_hl_matrix is not None:
        hl = daily_hl_matrix.get(sym, {}).get(date)
        if hl is not None:
            daily_high, daily_low = hl["high"], hl["low"]
            if pos["trade"]["trade_dir"] == 1:
                liq_price = pos["trade"]["entry_price"] * (
                    1 - (pos["initial_margin"] - maintenance_margin) / pos["notional"]
                )
                if daily_low <= liq_price:
                    liq_triggered = True
                    cur_price = liq_price
            else:
                liq_price = pos["trade"]["entry_price"] * (
                    1 + (pos["initial_margin"] - maintenance_margin) / pos["notional"]
                )
                if daily_high >= liq_price:
                    liq_triggered = True
                    cur_price = liq_price

    if liq_triggered:
        sl_mult_val = pos["trade"].get("sl_mult", 1.5)
        atr_pct_approx = pos["trade"]["sl_pct"] * 100.0 / sl_mult_val
        liq_slip = STATIC["liquidation_slippage_pct"] * (atr_pct_approx / 0.5)
        liq_slip = min(liq_slip, 0.20)
        if pos["trade"]["trade_dir"] == 1:
            cur_price = cur_price * (1 - liq_slip)
        else:
            cur_price = cur_price * (1 + liq_slip)
        pnl = (
            ((cur_price - pos["trade"]["entry_price"]) / pos["trade"]["entry_price"])
            * pos["notional"]
            * dir_mult
        )
        liq_fee = abs(pos["notional"]) * liquidation_fee_pct
        pnl -= liq_fee
        risk_cap = pos["risk_capital"]
        r_net = pnl / risk_cap if risk_cap != 0 else 0.0
        liq_trade = {
            "entry_time": pos["trade"]["entry_time"],
            "exit_time": date,
            "symbol": sym,
            "r_net": r_net,
            "entry_price": pos["trade"]["entry_price"],
            "exit_price": cur_price,
            "trade_dir": pos["trade"]["trade_dir"],
            "sl_pct": pos["trade"]["sl_pct"],
            "m_trend": pos["trade"].get("m_trend", 0),
            "macro_vol": pos["trade"].get("macro_vol", 0.0),
            "risk_weight": pos["trade"].get("risk_weight", asset_risk_weights.get(sym, 0.005)),
            "pnl_usd": pnl,
            "liquidated": True,
        }
        return True, pnl, liq_trade, cur_price
    else:
        return False, pnl, None, cur_price


def simulate_full_portfolio(
    trades: list[dict[str, Any]],
    initial_cash: float,
    leverage: float,
    mm_pct: float,
    position_limit: int,
    cb_hard_cooldown_hours: int,
    fixed_cb_threshold: float,
    daily_close_matrix: dict[str, dict],
    asset_risk_weights: dict[str, float],
    end_date: str,
    liquidation_fee_pct: float = 0.005,
    daily_hl_matrix: dict[str, dict] | None = None,
    _precomputed_daily_returns: pd.DataFrame | None = None,
    _precomputed_sym_list: list | None = None,
    _shared_corr_cache: dict | None = None,
) -> tuple[float, dict, list, _RejectReasons]:
    events = []
    for t in trades:
        events.append({"time": t["entry_time"], "type": "ENTRY", "trade": t})
        ext = t["exit_time"]
        if ext == t["entry_time"]:
            ext += timedelta(seconds=1)
        events.append({"time": ext, "type": "EXIT", "trade": t})
    events.sort(key=lambda x: (x["time"], 0 if x["type"] == "EXIT" else 1, x["trade"]["symbol"]))

    cash = initial_cash
    peak_equity = cash
    open_positions = {}
    daily_equity = {}
    current_day = None
    executed_trades = []
    reject_reasons = {"cb_cooldown": 0, "position_limit": 0, "margin_insufficient": 0}

    # Per-asset circuit breaker
    _per_asset_cb_threshold = fixed_cb_threshold
    asset_initial_alloc = {}
    asset_realized_pnl = {}
    asset_peak_equity = {}
    cb_asset_active = {}
    cb_asset_until = {}
    if asset_risk_weights:
        total_rw = sum(asset_risk_weights.values())
        for sym, rw in asset_risk_weights.items():
            alloc = initial_cash * (rw / total_rw)
            asset_initial_alloc[sym] = alloc
            asset_realized_pnl[sym] = 0.0
            asset_peak_equity[sym] = alloc
            cb_asset_active[sym] = False
            cb_asset_until[sym] = pd.Timestamp.min

    # Rolling correlation for sizing (no lookahead)
    if _precomputed_daily_returns is not None and _precomputed_sym_list is not None:
        _daily_returns_df = _precomputed_daily_returns
        _sym_list = _precomputed_sym_list
        _sym_to_idx = {s: i for i, s in enumerate(_sym_list)}
    else:
        _sym_list = sorted(daily_close_matrix.keys())
        _sym_to_idx = {s: i for i, s in enumerate(_sym_list)}
        _daily_returns_df = None
        if len(_sym_list) >= 2:
            ret_series_list = []
            for sym in _sym_list:
                s = (
                    pd.Series(daily_close_matrix[sym])
                    .sort_index()
                    .pct_change()
                    .dropna()
                )
                ret_series_list.append(s)
            ret_df = pd.concat(ret_series_list, axis=1, keys=_sym_list).dropna()
            if len(ret_df) > 30:
                _daily_returns_df = ret_df
            else:
                _daily_returns_df = None
                _sym_list = []
                _sym_to_idx = {}
        else:
            _sym_list = []
            _sym_to_idx = {}

    _corr_cache = _shared_corr_cache if _shared_corr_cache is not None else {}
    CORR_LOOKBACK = 180

    for event in events:
        curr_time = event["time"]
        trade = event["trade"]
        tr_id = _trade_key(trade)
        day_key = curr_time.replace(hour=0, minute=0, second=0)
        if hasattr(day_key, "tz") and day_key.tz is not None:
            day_key = day_key.tz_localize(None)

        # Day boundary: finalize previous event day's equity
        if current_day is not None and day_key > current_day:
            eod_prev = cash + sum(
                p.get("current_pnl_usd", 0.0) for p in open_positions.values()
            )
            daily_equity[current_day] = eod_prev
            peak_equity = max(peak_equity, eod_prev)

        # Gap day MTM fill + current day margin check
        if current_day is not None and day_key >= current_day:
            gap_start = current_day + timedelta(days=1)
            gap_end = day_key - timedelta(days=1)
            for d in pd.date_range(gap_start, gap_end, freq="D"):
                unrealized = 0.0
                for pos_id, pos in list(open_positions.items()):
                    is_liq, pnl, liq_trade, _ = _mtm_and_margin_check(
                        pos, d, daily_close_matrix, daily_hl_matrix,
                        mm_pct, liquidation_fee_pct, asset_risk_weights,
                    )
                    if is_liq:
                        executed_trades.append(liq_trade)
                        cash += pnl
                        if liq_trade["symbol"] in asset_realized_pnl:
                            asset_realized_pnl[liq_trade["symbol"]] += pnl
                        del open_positions[pos_id]
                    else:
                        pos["current_pnl_usd"] = pnl
                        unrealized += pnl
                daily_equity[d] = cash + unrealized
                peak_equity = max(peak_equity, daily_equity[d])

            unrealized = 0.0
            for pos_id, pos in list(open_positions.items()):
                is_liq, pnl, liq_trade, _ = _mtm_and_margin_check(
                    pos, day_key, daily_close_matrix, daily_hl_matrix,
                    mm_pct, liquidation_fee_pct, asset_risk_weights,
                )
                if is_liq:
                    executed_trades.append(liq_trade)
                    cash += pnl
                    if liq_trade["symbol"] in asset_realized_pnl:
                        asset_realized_pnl[liq_trade["symbol"]] += pnl
                    del open_positions[pos_id]
                else:
                    pos["current_pnl_usd"] = pnl
                    unrealized += pnl
        elif current_day is None:
            daily_equity[day_key] = cash
            peak_equity = max(peak_equity, cash)

        current_day = day_key

        if event["type"] == "EXIT":
            if tr_id in open_positions:
                pos = open_positions.pop(tr_id)
                pnl_usd = pos["risk_capital"] * trade["r_net"]
                cash += pnl_usd
                trade["pnl_usd"] = pnl_usd
                executed_trades.append(trade)
                if trade["symbol"] in asset_realized_pnl:
                    asset_realized_pnl[trade["symbol"]] += pnl_usd

                post_exit_equity = cash + sum(
                    p.get("current_pnl_usd", 0.0) for p in open_positions.values()
                )
                peak_equity = max(peak_equity, post_exit_equity)
        else:  # ENTRY
            current_equity = cash + sum(
                p.get("current_pnl_usd", 0.0) for p in open_positions.values()
            )
            peak_equity = max(peak_equity, current_equity)

            # Per-asset CB
            sym = trade["symbol"]
            current_asset_val = asset_initial_alloc.get(sym, 0) + asset_realized_pnl.get(sym, 0.0)
            for pos in open_positions.values():
                if pos["trade"]["symbol"] == sym:
                    current_asset_val += pos.get("current_pnl_usd", 0.0)
            asset_peak_equity[sym] = max(
                asset_peak_equity.get(sym, current_asset_val), current_asset_val
            )
            asset_dd = (
                (asset_peak_equity[sym] - current_asset_val) / asset_peak_equity[sym]
                if asset_peak_equity[sym] > 0 else 0
            )

            if not cb_asset_active.get(sym, False) and asset_dd > _per_asset_cb_threshold:
                cb_asset_active[sym] = True
                cb_asset_until[sym] = curr_time + timedelta(hours=cb_hard_cooldown_hours)
            if cb_asset_active.get(sym, False):
                if curr_time > cb_asset_until.get(sym, pd.Timestamp.min):
                    cb_asset_active[sym] = False
                else:
                    reject_reasons["cb_cooldown"] += 1
                    continue

            if len(open_positions) >= position_limit:
                reject_reasons["position_limit"] += 1
                continue

            risk_weight = trade.get(
                "risk_weight",
                asset_risk_weights.get(trade["symbol"], 0.005) if asset_risk_weights else 0.005,
            )

            # Trend-aligned risk multiplier (P0-B1 fix)
            # Scales position size: 1.5x with-trend, 0.5x counter-trend
            trend_risk_mult = trade.get("trend_risk_mult", 1.0)
            risk_weight = risk_weight * trend_risk_mult

            # Rolling correlation-aware sizing (signed)
            if _daily_returns_df is not None and open_positions and len(_sym_list) >= 2:
                new_sym = trade["symbol"]
                trade_dir_new = trade.get("trade_dir", 1)
                if new_sym in _sym_to_idx:
                    new_idx = _sym_to_idx[new_sym]
                    if day_key not in _corr_cache:
                        past_rets = _daily_returns_df[_daily_returns_df.index < day_key]
                        if len(past_rets) >= CORR_LOOKBACK:
                            window = past_rets.iloc[-CORR_LOOKBACK:]
                            _corr_cache[day_key] = window.corr().values.astype(np.float64)
                        elif len(past_rets) >= 30:
                            _corr_cache[day_key] = past_rets.corr().values.astype(np.float64)
                        else:
                            _corr_cache[day_key] = None
                    corr_matrix_roll = _corr_cache.get(day_key)
                    if corr_matrix_roll is not None:
                        signed_corrs = []
                        for pos_id, pos in open_positions.items():
                            open_sym = pos["trade"]["symbol"]
                            open_dir = pos["trade"].get("trade_dir", 1)
                            if open_sym in _sym_to_idx:
                                open_idx = _sym_to_idx[open_sym]
                                signed_corrs.append(
                                    corr_matrix_roll[new_idx, open_idx]
                                    * trade_dir_new
                                    * open_dir
                                )
                        if signed_corrs:
                            avg_signed = float(np.mean(signed_corrs))
                            risk_weight = risk_weight / (1.0 + max(0.0, avg_signed))

            sl_pct = trade["sl_pct"]
            if sl_pct <= 0:
                raise ValueError(
                    f"sl_pct must be > 0, got {sl_pct} (trade symbol={sym}, "
                    f"entry_time={trade.get('entry_time')})"
                )
            risk_capital = current_equity * risk_weight
            notional = risk_capital / sl_pct
            initial_margin = notional / leverage

            total_im_used = 0.0
            for p in open_positions.values():
                sym_p = p["trade"]["symbol"]
                pos_size_units = p["notional"] / p["trade"]["entry_price"]
                current_price = daily_close_matrix.get(sym_p, {}).get(
                    day_key, p["trade"]["entry_price"]
                )
                current_notional = pos_size_units * current_price
                total_im_used += current_notional / leverage

            if current_equity - total_im_used < initial_margin:
                reject_reasons["margin_insufficient"] += 1
                continue

            open_positions[tr_id] = {
                "trade": trade,
                "risk_capital": risk_capital,
                "notional": notional,
                "initial_margin": initial_margin,
                "current_pnl_usd": 0.0,
            }

    # Finalize last event day's equity
    if current_day is not None:
        eod_last = cash + sum(
            p.get("current_pnl_usd", 0.0) for p in open_positions.values()
        )
        daily_equity[current_day] = eod_last
        peak_equity = max(peak_equity, eod_last)

    # Fill MTM for remaining days after the last event through END_DATE
    final_end = pd.Timestamp(end_date)
    if current_day is not None:
        for d in pd.date_range(current_day + timedelta(days=1), final_end, freq="D"):
            unrealized = 0.0
            for pos_id, pos in list(open_positions.items()):
                is_liq, pnl, liq_trade, _ = _mtm_and_margin_check(
                    pos, d, daily_close_matrix, daily_hl_matrix,
                    mm_pct, liquidation_fee_pct, asset_risk_weights,
                )
                if is_liq:
                    executed_trades.append(liq_trade)
                    cash += pnl
                    if liq_trade["symbol"] in asset_realized_pnl:
                        asset_realized_pnl[liq_trade["symbol"]] += pnl
                    del open_positions[pos_id]
                else:
                    pos["current_pnl_usd"] = pnl
                    unrealized += pnl
            daily_equity[d] = cash + unrealized
            peak_equity = max(peak_equity, daily_equity[d])

    final_equity = cash + sum(
        p.get("current_pnl_usd", 0.0) for p in open_positions.values()
    )

    return final_equity, daily_equity, executed_trades, reject_reasons
