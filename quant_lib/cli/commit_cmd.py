"""quant_exp commit -- break seal, run holdout once (irreversible).

Runs the full pipeline on the holdout. Breaks the seal.
"""

from __future__ import annotations

from typing import Optional

import os
import traceback

import typer
from rich.rule import Rule

from quant_lib.cli._output import OutputManager
from quant_lib.cli._utils import looks_like_absolute
from quant_lib.experiments import get
from quant_lib.research.candidate import Candidate
from quant_lib.research.commit import commit_to_holdout
from quant_lib.research.reporting import print_commit_report
from quant_lib.research.session import ResearchSession
from quant_lib.core._logging import console
from quant_lib.utils.logging import setup_logging


def commit(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Experiment name"),
    cache_dir: Optional[str] = typer.Option(
        None, "--cache-dir", help="Data cache directory (default ./data_cache)",
    ),
    seal_dir: Optional[str] = typer.Option(
        None, "--seal-dir", help="Holdout seal directory (default <cache_dir>/holdout_seals)",
    ),
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation prompt."),
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
    """Commit to holdout (irreversible). Breaks the seal."""
    setup_logging(ctx.obj.get("verbose", 0) if ctx.obj else 0)

    try:
        exp = get(name)
    except KeyError:
        console.print(f"[red]Error:[/red] experiment '{name}' not found.")
        raise typer.Exit(code=1)

    h = exp.hypothesis
    train_s, train_e, hold_s, hold_e = exp.period.resolve()
    success_criteria = h.success_criteria

    console.print(Rule(f"[bold red]COMMIT: {exp.name}[/bold red]", style="red"))
    console.print(
        "  [red]\u26a0 This will BREAK the holdout seal (irreversible).[/red]"
    )
    console.print(f"  Holdout:  {hold_s} \u2192 {hold_e}")
    console.print(f"  Strategy: {exp.strategy_type}")
    console.print(f"  Success:  {success_criteria}")
    console.print()

    if not yes:
        confirmed = typer.confirm("Proceed?")
        if not confirmed:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(code=0)

    out = OutputManager(exp.name, mode="commit")

    # Session: NO _skip_holdout_load. C-2 enforces hash verification.
    resolved_cache_dir = cache_dir or "./data_cache"
    if seal_dir:
        os.environ["QUANT_LIB_SEAL_DIR"] = seal_dir
    session = ResearchSession(
        training_period=(train_s, train_e),
        holdout_period=(hold_s, hold_e),
        symbols=exp.universe.symbols,
        cache_dir=resolved_cache_dir,
    )
    # NOTE (0.2.2): Pass strategy=exp.strategy so per-experiment StrategyConfig
    # overrides (PF weight, leverage, etc.) apply in CLI path. Previously
    # silently used default StrategyConfig() and ignored per-experiment
    # config.
    cand: Candidate = session.create_candidate(
        h,
        strategy=exp.strategy,
    )

    try:
        # Phases 1-3 (same as explore)
        cand.run_universe(
            min_volume_usdt=exp.universe.min_volume_usdt,
            min_age_days=exp.universe.min_age_days,
        )
        cand.run_edge_testing()
        cand.run_narrowing()
        cand.mark_ready()

        # Phase 4: BREAKS SEAL
        console.print()
        console.print("[bold]Phase 4:[/] commit to holdout (breaking seal)...")
        result = commit_to_holdout(
            cand,
            success_criteria_text=success_criteria,
        )
    except Exception as e:
        # Sprint 1 fix: persist the full traceback so real bugs
        # (typos, AttributeError, ImportError) are not silently
        # swallowed as "failed" status. The traceback is written
        # alongside the metrics JSON for post-mortem inspection.
        tb = traceback.format_exc()
        console.print(f"[red]Commit failed:[/red] {e}")
        console.print(f"[red]{tb}[/red]")
        out.save_metrics({"status": "failed", "error": str(e), "traceback": tb})
        out.save_config(exp)
        out.link_latest()
        raise typer.Exit(code=1)

    # Print full commit report
    console.print()
    try:
        print_commit_report(result, session=session)
    except Exception as e:
        console.print(f"[yellow]Could not print commit report: {e}[/yellow]")

    # Persist output
    out.save_metrics(
        {
            "status": "ok",
            "final_equity": result.final_equity,
            "equity_pct": result.equity_pct,
            "cagr_pct": result.cagr_pct,
            "max_dd_pct": result.max_dd_pct,
            "psr": result.psr,
            "profit_factor": result.profit_factor,
            "win_rate": result.win_rate,
            "n_trades": result.n_trades,
            "seal_broken": result.seal_broken,
            "seal_hash_before": result.seal_hash_before,
            "seal_hash_after": result.seal_hash_after,
        }
    )
    out.save_config(exp)
    out.link_latest()

    # Optional HTML report
    if report is not None:
        _try_save_html_report(
            out=out,
            title=f"Commit #{result.commit_idx}: {exp.name}",
            cand=cand,
            result=result,
            session=session,
            report=report,
            no_plots=no_plots,
        )

    console.print()
    console.print(f"[green]\u2713[/green] Results saved: [bold]{out.path}[/bold]")
    console.print(
        "[red]\u2713[/red] Seal [bold]BROKEN[/bold]. This holdout cannot be used again."
    )


def _try_save_html_report(
    out: OutputManager,
    title: str,
    cand: Candidate,
    result,
    session: ResearchSession,
    report: str,
    no_plots: bool,
) -> None:
    """Build and save the HTML report. Silently skip charts if --no-plots."""
    from pathlib import Path

    from quant_lib.cli._report import build_commit_report

    chart_provider = _make_chart_provider(
        cand=cand,
        result=result,
        session=session,
        no_plots=no_plots,
    )
    sections = build_commit_report(result, session, chart_provider)

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
    result,
    session: ResearchSession,
    no_plots: bool,
):
    """Build a chart_provider for the commit report.

    Uses the session's daily equity matrix to build a holdout-period
    equity series. The Candidate's `executed_trades` are not directly
    available in commit (commit_to_holdout returns its own trades),
    so we rely on the result's by_symbol_stats + reconstructed r_vals
    if available.
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

    # Sprint 2 fix 2.4 + Sprint 3 fix 3.6: the commit HTML report now
    # renders the REAL daily equity curve (not the synthetic 2-point
    # fake that Sprint 2 removed). ``result.daily_equity`` was added
    # to ``CommitResult`` in Sprint 3 fix 3.6; it carries the
    # pd.Timestamp -> equity map from ``commit_to_holdout``. ``None``
    # when no trades executed (all rejected) -- the chart provider
    # returns None and the report renders an honest "Chart not
    # available" placeholder. The trade_distribution chart still
    # works because it accepts an empty list gracefully.
    daily_equity = result.daily_equity or {}
    r_vals: list[float] = []

    def provider(name: str):
        try:
            if name == "equity_curve":
                return plotting.plot_equity_curve(daily_equity, session.initial_capital)
            if name == "drawdown_underwater":
                return plotting.plot_drawdown_underwater(daily_equity)
            if name == "trade_distribution":
                return plotting.plot_trade_distribution(r_vals)
        except Exception as e:
            console.print(f"[yellow]Chart {name!r} failed: {e}[/yellow]")
            return None
        return None

    return provider


# Sprint 2 fix removed: ``_build_equity_series_from_result``. The helper
# produced a misleading 2-point fake equity curve. Sprint 3 fix 3.6
# now plumbs the REAL daily equity through ``CommitResult.daily_equity``,
# so the chart provider above renders an honest chart (or honest
# "Chart not available" when no trades executed).


def _looks_like_absolute(path_str: str) -> bool:
    """Backward-compat alias for ``looks_like_absolute`` in cli/_utils.

    Sprint 2 fix 2.5: the implementation moved to ``cli/_utils.py`` to
    deduplicate the identical helper that was previously in both
    ``explore.py`` and ``commit_cmd.py``. This alias preserves
    backward compatibility for callers that imported the private name.
    """
    return looks_like_absolute(path_str)
