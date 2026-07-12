"""
Rich console reports for explore (white) and commit (black).

No auto-verdict: print metrics; the user interprets them.

Explore report sections include hypothesis, universe, WFA, portfolio,
SPA and related OOS diagnostics, FDR context, and other sections.
Commit report sections include seal verification, holdout simulation,
holdout PSR/ESS, risk metrics, and audit summary.
"""

from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from quant_lib.core._testing import prob_sharpe_ratio, label_p_value
from quant_lib.core._metrics import compute_regime_stats, run_bootstrap, print_param_stability
from scipy import stats as scipy_stats

if TYPE_CHECKING:
    from quant_lib.research.candidate import Candidate


console = Console()


# ─────────────────────────────────────────────────────────────────────
# White Test Report
# ─────────────────────────────────────────────────────────────────────


def print_candidate_report(candidate: "Candidate") -> None:
    """Print the full 13-section white test report for a candidate."""
    if candidate.stage not in ("edge", "narrowed"):
        console.print(
            f"[yellow]Cannot print report: candidate at stage '{candidate.stage}' "
            f"(need 'edge' or 'narrowed')[/]"
        )
        return

    h = candidate.hypothesis
    session = candidate.session

    console.print()
    console.rule(f"[bold cyan]WHITE TEST REPORT: {h.name}[/]")
    console.print()

    # ── 1. Hypothesis ──
    _print_section_header(1, "HYPOTHESIS")
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column(style="bold", width=20)
    tbl.add_column()
    tbl.add_row("Name", h.name)
    tbl.add_row("Strategy", h.strategy_name)
    tbl.add_row("Mechanism", h.mechanism)
    tbl.add_row("Boundary", h.boundary_conditions)
    tbl.add_row("Success Crit.", h.success_criteria)
    tbl.add_row("Entry", h.entry_logic)
    tbl.add_row("Exit", h.exit_logic)
    if h.universe_rules:
        tbl.add_row("Universe", h.universe_rules)
    tbl.add_row("Min train mo.", str(h.min_train_months))
    tbl.add_row("Search space", str(h.merged_search_space()))
    tbl.add_row("Static overrides", str(h.merged_static_overrides()))
    tbl.add_row("Strategy params", str(h.merged_strategy_params()))
    tbl.add_row("Timestamp", h.timestamp.strftime("%Y-%m-%d %H:%M UTC"))
    console.print(tbl)
    console.print()

    # ── 2. Universe ──
    _print_section_header(2, "UNIVERSE")
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column(style="bold", width=20)
    tbl.add_column()
    tbl.add_row("Candidates", str(len(session.symbols)))
    tbl.add_row("Eligible", f"{len(candidate.eligible_symbols)}/{len(session.symbols)}")
    tbl.add_row("Narrowed", f"{len(candidate.narrowed_symbols)}/{len(candidate.eligible_symbols)}")
    tbl.add_row("Narrowing rule", candidate.narrowing_rule or "(not run)")
    tbl.add_row("Cache hits", str(candidate.cache_hits))
    tbl.add_row("Cache misses", str(candidate.cache_misses))
    console.print(tbl)
    console.print()

    # ── 3. Feature Engineering ──
    _print_section_header(3, "FEATURE ENGINEERING")
    tbl = Table(box=box.SIMPLE)
    tbl.add_column("Symbol", style="bold")
    tbl.add_column("Rows", justify="right")
    tbl.add_column("Cols", justify="right")
    for sym, df in candidate.precomputed_data.items():
        tbl.add_row(sym, f"{len(df):,}", str(len(df.columns)))
    console.print(tbl)
    console.print()

    # ── 4. WFA per symbol ──
    _print_section_header(4, "WFA PER SYMBOL")
    for sym in candidate.eligible_symbols:
        folds = candidate.fold_params.get(sym, [])
        sym_trades = [t for t in candidate.all_oos_trades if t["symbol"] == sym]
        tbl = Table(box=box.SIMPLE, show_header=True, padding=(0, 2),
                    title=f"[bold]{sym}[/] -- {len(folds)} folds, {len(sym_trades)} OOS trades")
        tbl.add_column("Fold")
        tbl.add_column("IS Start")
        tbl.add_column("OOS Period")
        tbl.add_column("vol_pct", justify="right")
        tbl.add_column("pullback", justify="right")
        tbl.add_column("trail_atr", justify="right")
        tbl.add_column("sl_mult", justify="right")
        tbl.add_column("rsi_oversold", justify="right")
        tbl.add_column("rsi_overbought", justify="right")
        tbl.add_column("Best Val", justify="right")
        for fp in folds:
            tbl.add_row(
                f"{fp.get('fold', '?')}/{fp.get('total_folds', '?')}",
                fp.get("is_start").strftime("%b %y") if fp.get("is_start") else "?",
                f"{fp.get('oos_start', '').strftime('%b %y')}-{fp.get('oos_end', '').strftime('%b %y')}",
                f"{fp.get('vol_pct_thresh', 0):.3f}",
                f"{int(fp.get('pullback_bars', 0))}",
                f"{fp.get('trail_atr', 0):.3f}",
                f"{fp.get('sl_mult', 0):.3f}",
                f"{fp.get('rsi_oversold', 0):.2f}" if fp.get('rsi_oversold') else "-",
                f"{fp.get('rsi_overbought', 0):.2f}" if fp.get('rsi_overbought') else "-",
                f"{fp.get('best_value', 0):.3f}",
            )
        console.print(tbl)
        if sym in candidate.frozen_params:
            fp = candidate.frozen_params[sym]
            console.print(
                f"  [green]Frozen (last fold):[/] "
                f"vol_pct={fp.get('vol_pct_thresh', 0):.3f}, "
                f"pullback={int(fp.get('pullback_bars', 0))}, "
                f"trail_atr={fp.get('trail_atr', 0):.3f}, "
                f"sl_mult={fp.get('sl_mult', 0):.3f}"
                + (f", rsi_oversold={fp.get('rsi_oversold', 0):.2f}, "
                   f"rsi_overbought={fp.get('rsi_overbought', 0):.2f}"
                   if fp.get('rsi_oversold') else "")
            )
    console.print()

    # ── 5. Portfolio Simulation ──
    _print_section_header(5, "PORTFOLIO SIMULATION")
    if len(candidate.daily_equity) > 1:
        eq_series = pd.Series(candidate.daily_equity).sort_index()
        cagr = ((eq_series.iloc[-1] / eq_series.iloc[0]) ** (365.25 / len(eq_series)) - 1) * 100
        max_dd = ((eq_series - eq_series.cummax()) / eq_series.cummax()).min() * 100
    else:
        cagr = max_dd = 0.0
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column(style="bold", width=20)
    tbl.add_column()
    tbl.add_row("Initial capital", f"${session.initial_capital:,.2f}")
    tbl.add_row("Final equity", f"${candidate.final_equity:,.2f}")
    tbl.add_row("CAGR", f"{cagr:+.2f}%")
    tbl.add_row("Max DD", f"{max_dd:.2f}%")
    tbl.add_row("OOS trades", str(candidate.n_oos_trades))
    tbl.add_row("Executed", str(candidate.n_executed))
    tbl.add_row("Rejected", str(candidate.n_rejected))
    rej = candidate.reject_reasons
    tbl.add_row("  CB cooldown", str(rej.get("cb_cooldown", 0)))
    tbl.add_row("  Position limit", str(rej.get("position_limit", 0)))
    tbl.add_row("  Margin insufficient", str(rej.get("margin_insufficient", 0)))
    console.print(tbl)
    console.print()

    # ── 6. SPA Test ──
    _print_section_header(6, "SPA TEST")
    spa_p = candidate.spa_p_value
    spa_label, spa_conf, spa_interp = label_p_value(spa_p, context="spa")
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column(style="bold", width=20)
    tbl.add_column()
    tbl.add_row("p-value", f"{spa_p:.4f} {spa_label}")
    tbl.add_row("Confidence", spa_conf)
    tbl.add_row("Interpretation", spa_interp)
    console.print(tbl)
    console.print()

    # ── 7. PSR + ESS ──
    _print_section_header(7, "PSR + ESS")
    r_vals = np.array([t["r_net"] for t in candidate.executed_trades])
    if len(r_vals) >= 10:
        sr, psr_val = prob_sharpe_ratio(r_vals, annualize=False)
        skew_v = float(scipy_stats.skew(r_vals))
        kurt_v = float(scipy_stats.kurtosis(r_vals, fisher=False))
        ess_v = float(len(r_vals))
        tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        tbl.add_column(style="bold", width=20)
        tbl.add_column()
        tbl.add_row("N trades", str(len(r_vals)))
        tbl.add_row("SR (R-multiple)", f"{sr:.4f}")
        tbl.add_row("Skewness", f"{skew_v:.4f}")
        tbl.add_row("Kurtosis", f"{kurt_v:.4f}")
        tbl.add_row("ESS", f"{ess_v:.1f}")
        tbl.add_row("PSR", f"{psr_val:.4f}")
        console.print(tbl)
    else:
        console.print(f"  [dim]Insufficient trades ({len(r_vals)} < 10) for PSR[/]")
    console.print()

    # ── 8. Per-Symbol Wilcoxon + FDR ──
    _print_section_header(8, "PER-SYMBOL WILCOXON + FDR")
    raw_p = {}
    for sym in candidate.eligible_symbols:
        sym_trades = [t for t in candidate.executed_trades if t["symbol"] == sym]
        if len(sym_trades) >= 5:
            r_arr = np.array([t["r_net"] for t in sym_trades])
            _, p = scipy_stats.wilcoxon(r_arr, alternative="greater")
            raw_p[sym] = p
        else:
            raw_p[sym] = 1.0
    if raw_p:
        sym_list = list(raw_p.keys())
        p_arr = np.array([raw_p[s] for s in sym_list])
        from quant_lib.core._testing import fdr_correction
        rejected, p_corr = fdr_correction(p_arr, alpha=0.15)
        tbl = Table(box=box.SIMPLE)
        tbl.add_column("Symbol", style="bold")
        tbl.add_column("N trades", justify="right")
        tbl.add_column("WR", justify="right")
        tbl.add_column("p_raw", justify="right")
        tbl.add_column("p_FDR", justify="right")
        tbl.add_column("Status")
        for i, sym in enumerate(sym_list):
            sym_trades = [t for t in candidate.executed_trades if t["symbol"] == sym]
            wr = f"{sum(1 for t in sym_trades if t['r_net'] > 0) / len(sym_trades) * 100:.0f}%" if sym_trades else "-"
            status = "[green]SIG[/]" if rejected[i] else "[red]NS[/]"
            tbl.add_row(
                sym, str(len(sym_trades)), wr,
                f"{p_arr[i]:.4f}", f"{p_corr[i]:.4f}", status
            )
        console.print(tbl)
    console.print()

    # ── 9. Regime Stats ──
    _print_section_header(9, "REGIME STATS")
    regime = compute_regime_stats(candidate.executed_trades)
    tbl = Table(box=box.SIMPLE)
    tbl.add_column("Regime", style="bold")
    tbl.add_column("N trades", justify="right")
    tbl.add_column("Profit Factor", justify="right")
    for reg, (pf, n) in regime.items():
        tbl.add_row(reg, str(n), f"{pf:.2f}")
    console.print(tbl)
    console.print()

    # ── 10. Parameter Stability ──
    _print_section_header(10, "PARAMETER STABILITY")
    if candidate.fold_params:
        try:
            print_param_stability(candidate.fold_params, candidate.eligible_symbols)
        except Exception:
            console.print("  [dim]Stability table not available (insufficient folds)[/]")
    console.print()

    # ── 11. Edge Classification ──
    _print_section_header(11, "EDGE CLASSIFICATION")
    n_sig_raw = sum(1 for p in raw_p.values() if p < 0.05) if raw_p else 0
    n_sig_fdr = int(rejected.sum()) if raw_p else 0
    if n_sig_fdr == 0 and spa_p >= 0.15:
        case = "RANDOM"
    elif n_sig_fdr >= 1 and spa_p < 0.15:
        case = "CONCENTRATED"
    elif n_sig_fdr == 0 and spa_p < 0.15:
        case = "BROAD_WEAK"
    else:
        case = "UNCLASSIFIED"
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column(style="bold", width=20)
    tbl.add_column()
    tbl.add_row("Case", case)
    tbl.add_row("SPA p", f"{spa_p:.4f}")
    tbl.add_row("Symbols total", str(len(candidate.eligible_symbols)))
    tbl.add_row("Significant (raw)", f"{n_sig_raw}")
    tbl.add_row("Significant (FDR)", f"{n_sig_fdr}")
    console.print(tbl)
    console.print()

    # ── 12. Bootstrap ──
    _print_section_header(12, "BOOTSTRAP")
    if len(candidate.daily_equity) > 30:
        try:
            daily_ret = pd.Series(candidate.daily_equity).sort_index().pct_change().dropna()
            eq_series = pd.Series(candidate.daily_equity).sort_index()
            boot = run_bootstrap(daily_ret, eq_series, max_dd, session.initial_capital)
            tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            tbl.add_column(style="bold", width=25)
            tbl.add_column()
            tbl.add_row("Block size", str(boot.get("BootstrapBlock", "?")))
            tbl.add_row("Worst 5% CAGR", f"{boot['Worst5_CAGR']:.2f}%")
            tbl.add_row("Worst 5% DD", f"{boot['Worst5_DD']:.2f}%")
            tbl.add_row("Worst 1% DD", f"{boot['Worst1_DD']:.2f}%")
            tbl.add_row("Worst 95% DD", f"{boot['Worst95_DD']:.2f}%")
            tbl.add_row("Observed DD pctile", f"{boot['DD_Pctile']:.1f}%")
            console.print(tbl)
        except Exception as e:
            console.print(f"  [dim]Bootstrap failed: {e}[/]")
    else:
        console.print("  [dim]Insufficient data for bootstrap[/]")
    console.print()

    # ── 13. Journal & FDR context ──
    _print_section_header(13, "JOURNAL & FDR CONTEXT")
    j = session.journal
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column(style="bold", width=20)
    tbl.add_column()
    tbl.add_row("Journal name", j.hypothesis_name)
    tbl.add_row("N entries", str(len(j.entries)))
    tbl.add_row("N experiments", str(j.n_experiments))
    tbl.add_row("N bugfixes", str(j.n_bugfixes))
    tbl.add_row("N ablations", str(j.n_ablations))
    tbl.add_row("Base alpha", f"{j.initial_alpha:.4f}")
    tbl.add_row("Adjusted alpha", f"{j.adjusted_alpha():.4f}")
    tbl.add_row("Session candidates", str(len(session.candidates)))
    tbl.add_row("Session commits", str(session.n_commits))
    tbl.add_row("Next Bonf alpha", f"{session.current_bonferroni_alpha:.4f}")
    tbl.add_row("FDR alpha", f"{session.fdr_alpha:.4f}")
    tbl.add_row("Holdout status", "SEALED" if session.holdout_set.is_sealed() else "BROKEN")
    tbl.add_row("Holdout period", f"{session.holdout_period[0]} -> {session.holdout_period[1]}")
    console.print(tbl)
    console.print()
    console.rule("[bold cyan]END OF WHITE TEST REPORT[/]")


# ─────────────────────────────────────────────────────────────────────
# Black Test Report
# ─────────────────────────────────────────────────────────────────────


def print_commit_report(result, session=None) -> None:
    """Print the full 8-section black test report for a commit."""
    console.print()
    console.rule(f"[bold red]BLACK TEST REPORT: COMMIT #{result.commit_idx} -- {result.candidate_name}[/]")
    console.print()

    # ── 1. Pre-commit Verification ──
    _print_section_header(1, "PRE-COMMIT VERIFICATION", color="red")
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column(style="bold", width=20)
    tbl.add_column()
    tbl.add_row("Candidate", result.candidate_name)
    tbl.add_row("Commit idx", str(result.commit_idx))
    tbl.add_row("Holdout period", f"{result.holdout_period[0]} -> {result.holdout_period[1]}")
    tbl.add_row("Initial capital", f"${result.initial_capital:,.2f}")
    tbl.add_row("Bonferroni alpha", f"{result.bonferroni_alpha:.4f}")
    tbl.add_row("FDR alpha", f"{result.fdr_alpha:.4f}")
    tbl.add_row("Seal status", "[red]BROKEN[/]" if result.seal_broken else "[green]SEALED[/]")
    tbl.add_row("Timestamp", result.timestamp)
    console.print(tbl)
    if result.success_criteria_text:
        console.print()
        console.print(
            Panel(
                f"[bold]Success criteria (user-defined):[/]\n{result.success_criteria_text}",
                border_style="yellow",
            )
        )
    console.print()

    # ── 2. Holdout Simulation ──
    _print_section_header(2, "HOLDOUT SIMULATION", color="red")
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column(style="bold", width=20)
    tbl.add_column()
    tbl.add_row("Initial capital", f"${result.initial_capital:,.2f}")
    tbl.add_row("Final equity", f"${result.final_equity:,.2f}")
    tbl.add_row("Raw trades", str(result.n_raw_trades))
    tbl.add_row("Executed", str(result.n_executed_trades))
    tbl.add_row("Rejected", str(result.n_rejected))
    rej = result.reject_breakdown
    for k, v in rej.items():
        tbl.add_row(f"  {k}", str(v))
    console.print(tbl)
    console.print()

    # ── 3. Per-Trade Stats ──
    _print_section_header(3, "PER-TRADE STATS", color="red")
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column(style="bold", width=20)
    tbl.add_column()
    tbl.add_row("N trades", str(result.n_trades))
    tbl.add_row("Win rate", f"{result.win_rate:.1f}%")
    tbl.add_row("Avg R", f"{result.avg_r:+.3f}")
    tbl.add_row("Median R", f"{result.median_r:+.3f}")
    tbl.add_row("Std R", f"{result.std_r:.3f}")
    tbl.add_row("Best trade", f"{result.best_r:+.2f} R")
    tbl.add_row("Worst trade", f"{result.worst_r:+.2f} R")
    tbl.add_row("Profit factor", f"{result.profit_factor:.2f}")
    tbl.add_row("Avg bars held", f"{result.avg_bars_held:.1f}")
    console.print(tbl)
    if result.by_symbol_stats:
        console.print()
        tbl2 = Table(box=box.SIMPLE, title="By Symbol")
        tbl2.add_column("Symbol", style="bold")
        tbl2.add_column("N", justify="right")
        tbl2.add_column("WR", justify="right")
        tbl2.add_column("Avg R", justify="right")
        tbl2.add_column("PF", justify="right")
        tbl2.add_column("Total R", justify="right")
        for sym, stats in result.by_symbol_stats.items():
            tbl2.add_row(
                sym, str(stats["n_trades"]),
                f"{stats['win_rate']:.0f}%",
                f"{stats['avg_r']:+.3f}",
                f"{stats['profit_factor']:.2f}",
                f"{stats['total_r']:+.1f}",
            )
        console.print(tbl2)
    console.print()

    # ── 4. Equity Curve Stats ──
    _print_section_header(4, "EQUITY CURVE STATS", color="red")
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column(style="bold", width=20)
    tbl.add_column()
    tbl.add_row("Initial", f"${result.initial_capital:,.2f}")
    tbl.add_row("Final", f"${result.final_equity:,.2f}")
    tbl.add_row("Equity %", f"{result.equity_pct:+.2f}%")
    tbl.add_row("CAGR (annualized)", f"{result.cagr_pct:+.2f}%")
    tbl.add_row("Max DD", f"{result.max_dd_pct:.2f}%")
    tbl.add_row("Sharpe (R)", f"{result.sharpe_r:.3f}")
    console.print(tbl)
    console.print()

    # ── 5. PSR + ESS Holdout ──
    _print_section_header(5, "PSR + ESS (HOLDOUT)", color="red")
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column(style="bold", width=20)
    tbl.add_column()
    tbl.add_row("N trades", str(result.n_trades))
    tbl.add_row("SR (R-multiple)", f"{result.sharpe_r:.4f}")
    tbl.add_row("Skewness", f"{result.skew:.4f}")
    tbl.add_row("Kurtosis", f"{result.kurtosis:.4f}")
    tbl.add_row("ESS", f"{result.ess:.1f}")
    tbl.add_row("PSR", f"{result.psr:.4f}")
    tbl.add_row("vs Bonf alpha", f"{result.psr:.4f} >= {result.bonferroni_alpha:.4f}")
    tbl.add_row("vs FDR alpha", f"{result.psr:.4f} >= {result.fdr_alpha:.4f}")
    console.print(tbl)
    console.print()

    # ── 6. Risk Metrics ──
    _print_section_header(6, "RISK METRICS", color="red")
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column(style="bold", width=25)
    tbl.add_column()
    tbl.add_row("With-trend trades (1.5x)", str(result.with_trend_trades))
    tbl.add_row("With-trend total R", f"{result.with_trend_r_total:+.1f}")
    tbl.add_row("Counter-trend trades (0.5x)", str(result.counter_trend_trades))
    tbl.add_row("Counter-trend total R", f"{result.counter_trend_r_total:+.1f}")
    impact = result.with_trend_r_total + result.counter_trend_r_total
    no_alignment = impact - (
        (result.with_trend_r_total / 1.5) + (result.counter_trend_r_total / 0.5)
    )
    tbl.add_row("Alignment impact (delta R)", f"{no_alignment:+.1f}")
    console.print(tbl)
    console.print()

    # ── 7. Seal & Audit ──
    _print_section_header(7, "SEAL & AUDIT", color="red")
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column(style="bold", width=25)
    tbl.add_column()
    tbl.add_row("Seal hash (before)", result.seal_hash_before[:32] + "...")
    tbl.add_row("Seal hash (after)", result.seal_hash_after[:32] + "...")
    tbl.add_row("Seal status", "[red]BROKEN[/]" if result.seal_broken else "[green]SEALED[/]")
    console.print(tbl)
    console.print()

    # ── 8. Final Summary (no verdict) ──
    _print_section_header(8, "FINAL SUMMARY (informational only, no verdict)", color="red")
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column(style="bold", width=20)
    tbl.add_column()
    tbl.add_row("Hypothesis", result.candidate_name)
    tbl.add_row("Commit #", str(result.commit_idx))
    tbl.add_row("Bonf alpha", f"{result.bonferroni_alpha:.4f}")
    tbl.add_row("FDR alpha", f"{result.fdr_alpha:.4f}")
    tbl.add_row("Holdout equity", f"${result.final_equity:,.2f}")
    tbl.add_row("Equity %", f"{result.equity_pct:+.2f}%")
    tbl.add_row("PSR", f"{result.psr:.4f}")
    tbl.add_row("Max DD", f"{result.max_dd_pct:.2f}%")
    tbl.add_row("Profit Factor", f"{result.profit_factor:.2f}")
    tbl.add_row("Win Rate", f"{result.win_rate:.1f}%")
    tbl.add_row("N trades", str(result.n_trades))
    if result.success_criteria_text:
        tbl.add_row("Success criteria", result.success_criteria_text[:60] + ("..." if len(result.success_criteria_text) > 60 else ""))
    console.print(tbl)
    console.print()
    console.print(
        Panel(
            "[bold yellow]User interprets based on metrics. "
            "No auto-verdict is given -- see success_criteria_text above.[/]",
            border_style="yellow",
        )
    )
    console.print()
    console.rule("[bold red]END OF BLACK TEST REPORT[/]")


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _print_section_header(num: int, title: str, color: str = "cyan") -> None:
    console.print(f"[bold {color}][{num}] {title}[/]")