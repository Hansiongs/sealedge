"""
Helper utilities -- matrices, bootstrap, regime stats, param diagnostics.

Extracted from Hans_Quant_Systems.py:
  - build_daily_matrices (lines 2057-2088)
  - _run_bootstrap (lines 2523-2565)
  - _compute_regime_stats (lines 2568-2590)
  - _print_param_stability (lines 2593-2795)
"""

import pandas as pd
import numpy as np
from rich.table import Table

from quant_lib.core._config import STATIC, GLOBAL_SEED
from quant_lib.core._logging import log, console

_DailyCloseMatrix = dict[str, dict]
_DailyHLMatrix = dict[str, dict]


def build_daily_matrices(
    symbols: list[str], precomputed_data: dict[str, pd.DataFrame]
) -> tuple[_DailyCloseMatrix, _DailyHLMatrix]:
    """Build daily close and high/low matrices from precomputed hourly data."""
    daily_close_matrix = {}
    daily_hl_matrix = {}
    for sym in symbols:
        df_temp = precomputed_data[sym].dropna(subset=["close", "high", "low"])
        if hasattr(df_temp["time"].dt, "tz") and df_temp["time"].dt.tz is not None:
            df_temp = df_temp.copy()
            df_temp["time"] = df_temp["time"].dt.tz_localize(None)
        close_daily = df_temp.set_index("time")["close"].resample("D").last()
        high_daily = df_temp.set_index("time")["high"].resample("D").max()
        low_daily = df_temp.set_index("time")["low"].resample("D").min()
        if close_daily.isna().sum() > len(close_daily) * 0.01:
            log.warning(
                f"[{sym}] >1% missing daily bars -- ffill propagates stale prices, "
                f"daily matrices may be unreliable."
            )
        close_daily = close_daily.ffill()
        high_daily = high_daily.ffill()
        low_daily = low_daily.ffill()
        daily_close_matrix[sym] = close_daily.to_dict()
        daily_hl_matrix[sym] = {
            d: {"high": high_daily.loc[d], "low": low_daily.loc[d]}
            for d in close_daily.index
        }
    return daily_close_matrix, daily_hl_matrix


def run_bootstrap(
    daily_ret: pd.Series, eq_series: pd.Series, max_dd: float, initial_capital: float
) -> dict[str, float]:
    """Circular block bootstrap for worst-case CAGR and DD estimates."""
    n_sim = STATIC["bootstrap_n_sim"]
    n_ret = len(daily_ret)
    block_size = max(
        STATIC["bootstrap_block_size_min"],
        min(STATIC["bootstrap_block_size_max"], n_ret // 20),
    )
    rng_boot = np.random.default_rng(GLOBAL_SEED + 12345)
    ending_equities = []
    max_dds_boot = []
    ret_arr = daily_ret.values
    n_blocks_needed = int(np.ceil(n_ret / block_size))

    for _ in range(n_sim):
        starts = rng_boot.integers(0, n_ret, size=n_blocks_needed)
        samp_ret = np.concatenate(
            [np.take(ret_arr, np.arange(s, s + block_size) % n_ret) for s in starts]
        )[:n_ret]
        eq_sim = initial_capital * np.cumprod(1 + samp_ret)
        ending_equities.append(eq_sim[-1])
        roll_max = np.maximum.accumulate(eq_sim)
        max_dds_boot.append(((eq_sim - roll_max) / roll_max).min())

    worst5_cagr = (np.percentile(ending_equities, 5) / initial_capital) ** (
        365.25 / len(eq_series)
    ) - 1
    worst5_cagr *= 100
    worst95_dd = np.percentile(max_dds_boot, 95) * 100
    observed_dd_decimal = max_dd / 100.0
    dd_pctile = np.sum(np.array(max_dds_boot) <= observed_dd_decimal) / n_sim * 100
    dd_worst5 = np.percentile(max_dds_boot, 5) * 100
    dd_worst1 = np.percentile(max_dds_boot, 1) * 100

    return {
        "Worst5_CAGR": worst5_cagr,
        "Worst95_DD": worst95_dd,
        "Worst5_DD": dd_worst5,
        "Worst1_DD": dd_worst1,
        "DD_Pctile": dd_pctile,
        "BootstrapBlock": block_size,
    }


def run_trade_bootstrap(
    trade_r_vals: np.ndarray,
    initial_capital: float,
    n_sim: int = 2000,
    block_size: int = 5,
) -> dict[str, float]:
    """Circular block bootstrap on TRADE R-multiples (Phase 4.1).

    More appropriate for trade-based strategies than the daily-return
    bootstrap (``run_bootstrap``). Resamples trade R-multiples in
    blocks to preserve autocorrelation within winning/losing streaks,
    then computes the equity outcome for each simulated set.

    Why this replaces the daily-return approach:
    - For strategies with n_trades < 200, daily returns are sparse
      (most days have no trades). Bootstrapping daily returns
      overstates confidence because the zero-return days inflate
      the apparent sample size.
    - Trade R-multiples are the fundamental unit of strategy edge.
      Bootstrapping them directly tests the null "trades are
      independent with replacement" which is the correct null for
      a strategy selector.

    Parameters
    ----------
    trade_r_vals : np.ndarray
        1D array of trade R-multiples (per-trade returns).
    initial_capital : float
        Starting capital (used to convert R-multiples to USD equity
        for CAGR/Worst5 metrics).
    n_sim : int
        Number of bootstrap simulations. Default 2000.
    block_size : int
        Block size for circular block bootstrap (preserves short-
        range serial correlation). Default 5 trades.

    Returns
    -------
    dict
        ``Worst5_CAGR``: 5th percentile annualized CAGR.
        ``Worst95_DD``: 95th percentile max drawdown.
        ``Worst5_DD``: 5th percentile max drawdown.
        ``Worst1_DD``: 1st percentile max drawdown.
        ``Block``: block size used.
        All values NaN if len(trade_r_vals) < 5.
    """
    n = len(trade_r_vals)
    if n < 5:
        return {
            "Worst5_CAGR": float("nan"),
            "Worst95_DD": float("nan"),
            "Worst5_DD": float("nan"),
            "Worst1_DD": float("nan"),
            "Block": block_size,
        }

    r_arr = np.asarray(trade_r_vals, dtype=np.float64)
    rng = np.random.default_rng(GLOBAL_SEED + 99999)
    n_blocks = int(np.ceil(n / block_size))
    n_days = n  # proxy for CAGR: treat 1 trade ≈ 1 day
    ending_equities = []
    max_dds = []

    for _ in range(n_sim):
        starts = rng.integers(0, n, size=n_blocks)
        samp = np.concatenate(
            [r_arr[np.arange(s, s + block_size) % n] for s in starts]
        )[:n]
        # Convert R-multiples to equity: eq[t] = eq[t-1] * (1 + samp[t])
        eq = initial_capital * np.cumprod(1 + samp)
        ending_equities.append(eq[-1])
        roll_max = np.maximum.accumulate(eq)
        max_dds.append(((eq - roll_max) / roll_max).min())

    ending_arr = np.array(ending_equities)
    dd_arr = np.array(max_dds)

    worst5_cagr = (np.percentile(ending_arr, 5) / initial_capital) ** (
        365.25 / max(n_days, 1)
    ) - 1
    worst5_cagr *= 100

    return {
        "Worst5_CAGR": worst5_cagr,
        "Worst95_DD": float(np.percentile(dd_arr, 95) * 100),
        "Worst5_DD": float(np.percentile(dd_arr, 5) * 100),
        "Worst1_DD": float(np.percentile(dd_arr, 1) * 100),
        "Block": block_size,
    }


def compute_regime_stats(executed_trades: list[dict]) -> dict[str, tuple[float, int]]:
    """Compute profit factor per macro regime (Bull/Bear based on BTC macro trend)."""
    bull_trades = [t for t in executed_trades if t.get("m_trend") == 1]
    bear_trades = [t for t in executed_trades if t.get("m_trend") == -1]

    def _regime_pf(trades):
        if not trades:
            return 1.0, 0
        g_lr = abs(sum(x["r_net"] for x in trades if x["r_net"] <= 0))
        pfr = (
            sum(x["r_net"] for x in trades if x["r_net"] > 0) / g_lr
            if g_lr > 0
            else 1.0
        )
        return pfr, len(trades)

    return {
        "Bull": _regime_pf(bull_trades),
        "Bear": _regime_pf(bear_trades),
    }


def print_param_stability(
    all_fold_params: dict[str, list[dict]],
    symbols: list[str],
) -> None:
    """Print fold-by-fold parameter stability analysis with CV metrics."""
    PARAM_NAMES = ["vol_pct_thresh", "pullback_bars", "trail_atr", "sl_mult"]

    if not all_fold_params:
        return

    for sym in symbols:
        folds = all_fold_params.get(sym, [])
        if len(folds) < 2:
            continue

        tbl = Table(
            show_header=True,
            header_style="bold cyan",
            title=f"[bold]{sym}[/] -- Fold Parameters (λ={STATIC.get('reg_lambda', 0.05)})",
            box=None,
        )
        tbl.add_column("Fold", style="bold white")
        tbl.add_column("IS Start", justify="right")
        tbl.add_column("OOS Period", justify="right")
        for pn in PARAM_NAMES:
            tbl.add_column(pn, justify="right")
        tbl.add_column("Best Val", justify="right")

        for fp in folds:
            oos_label = (
                f"{fp['oos_start'].strftime('%b %y')}-{fp['oos_end'].strftime('%b %y')}"
            )
            vals = []
            for pn in PARAM_NAMES:
                v = fp.get(pn, float("nan"))
                if isinstance(v, float):
                    vals.append(f"{v:.3f}" if pn != "pullback_bars" else f"{int(v)}")
                else:
                    vals.append(str(v))
            tbl.add_row(
                f"{fp['fold']}/{fp['total_folds']}",
                fp["is_start"].strftime("%b %y"),
                oos_label,
                *vals,
                f"{fp['best_value']:.3f}",
            )
        console.print(tbl)

        # Stability metrics. Explicit type annotation: dict literal
        # comprehension infers list[Unknown] which fails mypy strict.
        param_vals: dict[str, list[float]] = {pn: [] for pn in PARAM_NAMES}
        for fp in folds:
            for pn in PARAM_NAMES:
                v = fp.get(pn)
                if v is not None:
                    param_vals[pn].append(v)

        stbl_tbl = Table(
            show_header=True,
            header_style="bold magenta",
            title=(
                f"[bold]{sym}[/] -- Stability Metrics "
                f"(CV < 15% = stable, 15-30% = moderate, >30% = unstable)"
            ),
        )
        stbl_tbl.add_column("Parameter", style="bold white")
        stbl_tbl.add_column("Mean", justify="right")
        stbl_tbl.add_column("Std", justify="right")
        stbl_tbl.add_column("Min", justify="right")
        stbl_tbl.add_column("Max", justify="right")
        stbl_tbl.add_column("CV%", justify="right")
        stbl_tbl.add_column("Rating", justify="center")

        for pn in PARAM_NAMES:
            # Renamed from `vals` to avoid conflict with the outer
            # `vals: list[str]` declared at line 152 (table rows).
            param_vals_for_pn: list[float] = param_vals[pn]
            if len(param_vals_for_pn) < 2:
                continue
            mean_v = float(np.mean(param_vals_for_pn))
            std_v = float(np.std(param_vals_for_pn, ddof=1))
            min_v = float(np.min(param_vals_for_pn))
            max_v = float(np.max(param_vals_for_pn))
            cv = (std_v / abs(mean_v)) * 100 if abs(mean_v) > 1e-10 else 0.0

            if cv < 15:
                rating = "[bold green]STABLE[/]"
                cv_str = f"[green]{cv:.1f}%[/]"
            elif cv < 30:
                rating = "[bold yellow]MODERATE[/]"
                cv_str = f"[yellow]{cv:.1f}%[/]"
            else:
                rating = "[bold red]UNSTABLE[/]"
                cv_str = f"[red]{cv:.1f}%[/]"

            if pn == "pullback_bars":
                mean_str = f"{mean_v:.2f}"
                std_str = f"{std_v:.2f}"
                min_str = f"{min_v:.0f}"
                max_str = f"{max_v:.0f}"
            else:
                mean_str = f"{mean_v:.3f}"
                std_str = f"{std_v:.3f}"
                min_str = f"{min_v:.3f}"
                max_str = f"{max_v:.3f}"

            stbl_tbl.add_row(pn, mean_str, std_str, min_str, max_str, cv_str, rating)

        if len(folds) >= 2:
            cvs = []
            for pn in PARAM_NAMES:
                # Renamed from `vals` to avoid conflict with the
                # `list[str]` declared at line 152 (table rows).
                cv_inner: list[float] = param_vals[pn]
                if len(cv_inner) >= 2:
                    cv_p = (
                        float(np.std(cv_inner, ddof=1))
                        / abs(float(np.mean(cv_inner)))
                    ) * 100
                    cvs.append(cv_p)
            if cvs:
                mean_cv = float(np.mean(cvs))
                if mean_cv < 15:
                    comp_rating = f"[bold green]STABLE[/] (mean CV={mean_cv:.1f}%)"
                elif mean_cv < 30:
                    comp_rating = f"[bold yellow]MODERATE[/] (mean CV={mean_cv:.1f}%)"
                else:
                    comp_rating = f"[bold red]UNSTABLE[/] (mean CV={mean_cv:.1f}%)"
                stbl_tbl.add_row(
                    "[bold]Composite[/]", "", "", "", "", "", comp_rating,
                )
        console.print(stbl_tbl)

    # Cross-symbol summary
    console.print("\n[bold]Cross-Symbol Parameter Stability Summary:[/]")
    summary_tbl = Table(show_header=True, header_style="bold yellow")
    summary_tbl.add_column("Symbol", style="bold white")
    for pn in PARAM_NAMES:
        summary_tbl.add_column(f"{pn} CV%", justify="right")
    summary_tbl.add_column("Mean CV%", justify="right")
    summary_tbl.add_column("Overall", justify="center")

    for sym in symbols:
        folds = all_fold_params.get(sym, [])
        if len(folds) < 2:
            continue
        cvs = []
        row_vals = []
        for pn in PARAM_NAMES:
            # fp.get(pn) is float | None; the None filter narrows to float
            # but mypy can't see through the comprehension. Renamed from
            # `vals` to avoid conflict with the outer `vals: list[str]`
            # used for the table row (line 152).
            param_vals_inner: list[float] = [
                fp.get(pn) for fp in folds if fp.get(pn) is not None
            ]
            if len(param_vals_inner) >= 2:
                cv_p = (
                    float(np.std(param_vals_inner, ddof=1))
                    / abs(float(np.mean(param_vals_inner)))
                ) * 100
            else:
                cv_p = 0.0
            cvs.append(cv_p)
            if cv_p < 15:
                row_vals.append(f"[green]{cv_p:.1f}%[/]")
            elif cv_p < 30:
                row_vals.append(f"[yellow]{cv_p:.1f}%[/]")
            else:
                row_vals.append(f"[red]{cv_p:.1f}%[/]")

        mean_cv = float(np.mean(cvs))
        if mean_cv < 15:
            overall = f"[green]STABLE ({mean_cv:.1f}%)[/]"
        elif mean_cv < 30:
            overall = f"[yellow]MODERATE ({mean_cv:.1f}%)[/]"
        else:
            overall = f"[red]UNSTABLE ({mean_cv:.1f}%)[/]"
        summary_tbl.add_row(sym, *row_vals, f"{mean_cv:.1f}%", overall)

    console.print(summary_tbl)

    # Warning if unstable
    unstable_symbols = []
    for sym in symbols:
        folds = all_fold_params.get(sym, [])
        if len(folds) < 2:
            continue
        cvs = []
        for pn in ["vol_pct_thresh", "trail_atr", "sl_mult"]:
            # Same pattern as above; renamed to `unstable_vals` to
            # avoid conflict with the `param_vals_inner` declared
            # earlier in the function.
            unstable_vals: list[float] = [
                fp.get(pn) for fp in folds if fp.get(pn) is not None
            ]
            if len(unstable_vals) >= 2:
                cv_p = (
                    float(np.std(unstable_vals, ddof=1))
                    / abs(float(np.mean(unstable_vals)))
                ) * 100
                cvs.append(cv_p)
        avg_cv = float(np.mean(cvs)) if cvs else 0
        if avg_cv > 30:
            unstable_symbols.append((sym, avg_cv))

    if unstable_symbols:
        msg = ", ".join(f"{s} ({c:.0f}%)" for s, c in unstable_symbols)
        log.warning(
            f"Parameter stability: [{msg}] show high CV% across folds. "
            f"This suggests per-fold overfitting -- SPA p-value may overstate "
            f"edge. Consider increasing reg_lambda or inspecting fold params."
        )
