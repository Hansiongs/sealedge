"""quant_exp list -- List all registered experiments."""
from __future__ import annotations

from rich.table import Table

from quant_lib.experiments import all_experiments
from quant_lib.core._logging import console


def list_cmd() -> None:
    """List all registered experiments."""
    exps = all_experiments()
    if not exps:
        console.print("[yellow]No experiments registered.[/yellow]")
        console.print(
            "Add experiments by creating a file in "
            "[bold]quant_lib/experiments/[/bold]."
        )
        return

    tbl = Table(title="Registered Experiments", show_header=True)
    tbl.add_column("Name", style="bold cyan")
    tbl.add_column("Strategy")
    tbl.add_column("Train", justify="right")
    tbl.add_column("Holdout", justify="right")
    tbl.add_column("Symbols", justify="right")

    for exp in exps:
        train_s, train_e, hold_s, hold_e = exp.period.resolve()
        tbl.add_row(
            exp.name,
            exp.strategy_type,
            f"{train_s}\u2192{train_e}",
            f"{hold_s}\u2192{hold_e}",
            str(len(exp.universe.symbols)),
        )
    console.print(tbl)
