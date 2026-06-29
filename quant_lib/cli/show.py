"""quant_exp show -- Show details of an experiment."""
from __future__ import annotations

import typer
from rich.panel import Panel

from quant_lib.experiments import get
from quant_lib.core._logging import console


def show(name: str = typer.Argument(..., help="Experiment name")) -> None:
    """Show full details of an experiment."""
    try:
        exp = get(name)
    except KeyError:
        console.print(f"[red]Error:[/red] experiment '{name}' not found.")
        raise typer.Exit(code=1)

    h = exp.hypothesis
    p = exp.period
    u = exp.universe
    train_s, train_e, hold_s, hold_e = p.resolve()

    body_lines = [
        f"[bold]Name:[/]      {exp.name}",
        f"[bold]Strategy:[/]  {exp.strategy_type}",
        "",
        f"[bold]Mechanism:[/]  {h.mechanism}",
        f"[bold]Boundary:[/]   {h.boundary_conditions}",
        f"[bold]Success:[/]    {h.success_criteria}",
        f"[bold]Entry:[/]      {h.entry_logic}",
        f"[bold]Exit:[/]       {h.exit_logic}",
        "",
        "[bold]Period:[/]",
        f"  Train:   {train_s} \u2192 {train_e}",
        f"  Holdout: {hold_s} \u2192 {hold_e}",
        "",
        f"[bold]Universe:[/]   {', '.join(u.symbols)}",
        f"  min_volume_usdt: {u.min_volume_usdt:,.0f}",
        f"  min_age_days:    {u.min_age_days}",
    ]
    body = "\n".join(body_lines)
    console.print(Panel(body, title=f"[bold]{exp.name}[/]", border_style="cyan"))
