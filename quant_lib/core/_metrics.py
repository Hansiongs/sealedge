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
from typing import Any
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


def print_param_stability(all_fold_params: dict[str, list[dict]], symbols: list[str]) -> None:
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

        # Stability metrics
        param_vals = {pn: [] for pn in PARAM_NAMES}
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
            vals = param_vals[pn]
            if len(vals) < 2:
                continue
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1))
            min_v = float(np.min(vals))
            max_v = float(np.max(vals))
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
                vals = param_vals[pn]
                if len(vals) >= 2:
                    cv_p = (float(np.std(vals, ddof=1)) / abs(float(np.mean(vals)))) * 100
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
            vals = [fp.get(pn) for fp in folds if fp.get(pn) is not None]
            if len(vals) >= 2:
                cv_p = (float(np.std(vals, ddof=1)) / abs(float(np.mean(vals)))) * 100
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
            vals = [fp.get(pn) for fp in folds if fp.get(pn) is not None]
            if len(vals) >= 2:
                cv_p = (float(np.std(vals, ddof=1)) / abs(float(np.mean(vals)))) * 100
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
