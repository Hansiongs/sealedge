"""quant_exp explore -- Run OOS exploration (Phase 0-3).

Loads data, runs WFA + SPA on the training set. Holdout stays sealed.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.rule import Rule

from quant_lib.cli._output import OutputManager
from quant_lib.experiments import get
from quant_lib.research.candidate import Candidate
from quant_lib.research.reporting import print_candidate_report
from quant_lib.research.session import ResearchSession
from quant_lib.core._logging import console
from quant_lib.utils.logging import setup_logging


def explore(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Experiment name"),
    no_spa: bool = typer.Option(False, "--no-spa", help="Skip SPA iterations."),
    report: Optional[str] = typer.Option(
        None,
        "--report",
        help=(
            "Generate HTML report to this path (within the run "
            "directory if relative). Use --no-plots to skip charts."
        ),
    ),
    no_plots: bool = typer.Option(
        False,
        "--no-plots",
        help="Skip chart generation in HTML report (text-only metrics).",
    ),
) -> None:
    """Run OOS exploration (Phase 0-3). Holdout stays sealed."""
    setup_logging(ctx.obj.get("verbose", 0) if ctx.obj else 0)

    try:
        exp = get(name)
    except KeyError:
        console.print(f"[red]Error:[/red] experiment '{name}' not found.")
        raise typer.Exit(code=1)

    train_s, train_e, hold_s, hold_e = exp.period.resolve()
    out = OutputManager(exp.name, mode="explore")

    console.print(Rule(f"[bold]Explore: {exp.name}[/]"))
    console.print(f"  Train:   {train_s} \u2192 {train_e}")
    console.print(f"  Holdout: {hold_s} \u2192 {hold_e} (sealed)")
    console.print(f"  Symbols: {', '.join(exp.universe.symbols)}")
    console.print()

    # Session with _skip_holdout_load=True: holdout data NOT fetched
    session = ResearchSession(
        training_period=(train_s, train_e),
        holdout_period=(hold_s, hold_e),
        symbols=exp.universe.symbols,
        cache_dir="./data_cache",
        _skip_holdout_load=True,
    )
    # NOTE (0.2.2): Pass strategy=exp.strategy so per-experiment StrategyConfig
    # overrides (PF weight, leverage, etc.) apply in CLI path. Previously
    # silently used default StrategyConfig() and ignored per-experiment
    # config.
    cand: Candidate = session.create_candidate(
        exp.hypothesis,
        strategy=exp.strategy,
    )

    try:
        # Phase 1: Universe
        console.print("[bold]Phase 1:[/] universe + features...")
        cand.run_universe(
            min_volume_usdt=exp.universe.min_volume_usdt,
            min_age_days=exp.universe.min_age_days,
        )
        console.print(f"  [green]Eligible:[/] {cand.eligible_symbols}")

        # Phase 2: WFA + SPA
        console.print()
        console.print("[bold]Phase 2:[/] WFA + SPA...")
        cand.run_edge_testing(n_spa_iters=0 if no_spa else 2000)
        console.print(f"  [green]OOS trades:[/] {cand.n_oos_trades}")
        console.print(f"  [green]Final equity:[/] ${cand.final_equity:,.2f}")
        console.print(f"  [green]SPA p-value:[/] {cand.spa_p_value:.4f}")

        # Phase 3: Narrowing
        console.print()
        console.print("[bold]Phase 3:[/] narrowing...")
        cand.run_narrowing()
        console.print(f"  [green]Narrowed:[/] {cand.narrowed_symbols}")
    except Exception as e:
        console.print(f"[red]Pipeline failed:[/red] {e}")
        out.save_metrics({"status": "failed", "error": str(e)})
        out.save_config(exp)
        out.link_latest()
        raise typer.Exit(code=1)

    # Print full report
    console.print()
    try:
        print_candidate_report(cand)
    except Exception as e:
        console.print(f"[yellow]Could not print full report: {e}[/yellow]")

    # Persist output
    out.save_metrics(
        {
            "status": "ok",
            "n_oos_trades": cand.n_oos_trades,
            "n_executed": cand.n_executed,
            "n_rejected": cand.n_rejected,
            "final_equity": cand.final_equity,
            "spa_p_value": cand.spa_p_value,
            "narrowed_symbols": cand.narrowed_symbols,
        }
    )
    out.save_config(exp)
    out.link_latest()

    # Optional HTML report
    if report is not None:
        _try_save_html_report(
            out=out,
            title=f"Explore: {exp.name}",
            cand=cand,
            session=session,
            report=report,
            no_plots=no_plots,
        )

    console.print()
    console.print(f"[green]\u2713[/green] Results saved: [bold]{out.path}[/bold]")
    console.print("[green]\u2713[/green] Holdout seal: INTACT (commit still possible)")


def _try_save_html_report(
    out: OutputManager,
    title: str,
    cand: Candidate,
    session: ResearchSession,
    report: str,
    no_plots: bool,
) -> None:
    """Build and save the HTML report. Silently skip charts if --no-plots.

    Errors during HTML report generation are logged but do not abort
    the run -- the text/metrics output has already been persisted.
    """
    from pathlib import Path

    from quant_lib.cli._report import build_explore_report

    chart_provider = _make_chart_provider(
        cand=cand,
        session=session,
        no_plots=no_plots,
    )
    sections = build_explore_report(cand, session, chart_provider)

    # Resolve the report target path. By default the report lives at
    # ``out.path / "report.html"``. A ``--report foo.html`` saves as
    # ``out.path / foo.html``. An absolute path is used as-is.
    if not report:
        report_target = out.path / "report.html"
    elif _looks_like_absolute(report):
        report_target = Path(report)
        report_target.parent.mkdir(parents=True, exist_ok=True)
    else:
        report_target = out.path / report

    try:
        out.save_html_report(
            title=title,
            sections=sections,
            output_name=(
                str(report_target)
                if _looks_like_absolute(report or "")
                else report_target.name
            ),
        )
        # If absolute path, copy from the run dir to the target.
        if report and _looks_like_absolute(report):
            run_report = out.path / "report.html"
            report_target.write_bytes(run_report.read_bytes())
        console.print(
            f"[green]\u2713[/green] HTML report saved: [bold]{report_target}[/bold]"
        )
    except Exception as e:
        console.print(f"[yellow]HTML report generation failed: {e}[/yellow]")


def _make_chart_provider(
    cand: Candidate,
    session: ResearchSession,
    no_plots: bool,
):
    """Build a chart_provider callable that returns base64 URIs or None.

    Lazy-imports plotting so matplotlib is an optional dep. If
    matplotlib is missing or ``--no-plots`` is set, every chart
    returns None (sections for unavailable charts are skipped).
    """
    if no_plots:
        return lambda _name: None

    try:
        from quant_lib.research import plotting  # noqa: F401
    except ImportError:
        console.print(
            "[yellow]matplotlib/seaborn not installed; skipping charts "
            "in HTML report (install with `pip install quant_lib[viz]`)[/yellow]"
        )
        return lambda _name: None

    # Resolve data needed for each chart
    daily_equity = cand.daily_equity or {}
    r_vals = [t.get("r_net", 0.0) for t in (cand.executed_trades or [])]

    # Per-symbol equity: per-symbol cumulative from executed_trades
    per_sym = _per_symbol_equity_from_trades(cand)

    def provider(name: str):
        try:
            if name == "equity_curve":
                return plotting.plot_equity_curve(daily_equity, session.initial_capital)
            if name == "drawdown_underwater":
                return plotting.plot_drawdown_underwater(daily_equity)
            if name == "trade_distribution":
                return plotting.plot_trade_distribution(r_vals)
            if name == "per_symbol_equity":
                return plotting.plot_per_symbol_equity(per_sym)
            if name == "wfa_progression":
                return plotting.plot_wfa_progression(cand.fold_params or {})
        except Exception as e:  # one chart failing should not abort
            console.print(f"[yellow]Chart {name!r} failed: {e}[/yellow]")
            return None
        return None

    return provider


def _per_symbol_equity_from_trades(cand: Candidate) -> dict:
    """Build per-symbol cumulative R-multiple series from executed trades.

    Returns ``{symbol: {date: cumulative_R}}`` suitable for
    ``plot_per_symbol_equity``. Uses exit_time as the trade point.
    """
    per_sym: dict = {}
    for t in cand.executed_trades or []:
        sym = t.get("symbol")
        exit_t = t.get("exit_time")
        r = float(t.get("r_net", 0.0))
        if sym is None or exit_t is None:
            continue
        per_sym.setdefault(sym, {})
        # Cumulative sum across the symbol's trades (sorted by exit_time)
        per_sym[sym][exit_t] = r
    # Sort and cumsum per symbol
    import pandas as pd

    out: dict = {}
    for sym, pts in per_sym.items():
        s = pd.Series(pts).sort_index().cumsum()
        out[sym] = {d: float(v) for d, v in s.items()}
    return out


def _looks_like_absolute(path_str: str) -> bool:
    """Cheap absolute-path detection: drive letter on Windows, leading /."""
    import os

    return os.path.isabs(path_str)
