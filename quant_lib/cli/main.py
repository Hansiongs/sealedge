"""quant_exp CLI entry point.

Provides a Typer-based CLI with 5 subcommands:
- list: List registered experiments
- show: Show details of one experiment
- explore: Run OOS exploration (Phase 0-3)
- commit: Run final commit to holdout (Phase 4)
- status: Show holdout seal status and recent runs

NOTE (0.2.2): experiment auto-discovery is triggered by
``quant_lib.experiments.__init__`` (which imports built_in and calls
discover_experiments()). No need to import built_in here.
"""
from __future__ import annotations

import typer


app = typer.Typer(
    name="quant_exp",
    help="quant_exp: honest backtesting for crypto strategies.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        from quant_lib import __version__
        typer.echo(f"quant_exp {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    verbose: int = typer.Option(
        0, "-v", "--verbose", count=True, help="Increase verbosity (-v, -vv)."
    ),
) -> None:
    """quant_exp: honest backtesting for crypto strategies."""


# Register subcommands
from quant_lib.cli.list_cmd import list_cmd
from quant_lib.cli.show import show
from quant_lib.cli.explore import explore
from quant_lib.cli.commit_cmd import commit
from quant_lib.cli.status_cmd import status


app.command("list", help="List all registered experiments.")(list_cmd)
app.command("show", help="Show details of an experiment.")(show)
app.command(
    "explore",
    help="Run OOS exploration (Phase 0-3). Holdout stays sealed.",
)(explore)
app.command(
    "commit",
    help="Commit to holdout (Phase 4). Breaks holdout seal (irreversible).",
)(commit)
app.command("status", help="Show holdout seal status and recent runs.")(status)


if __name__ == "__main__":
    app()
