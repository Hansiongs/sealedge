"""quant_exp CLI entry point.

Provides a Typer-based CLI with 6 subcommands:
- init: Scaffold a new experiment
- list: List registered experiments
- show: Show details of one experiment
- explore: Run OOS exploration (Phase 0-3)
- commit: Run final commit to holdout (Phase 4)
- status: Show holdout seal status and recent runs
- migrate-seals: Re-sign holdout seals with the current HMAC secret

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


# Register subcommands. These imports are placed after `app` is defined
# because each subcommand module imports `app` from this module; importing
# them at the top would cause a circular import.
from quant_lib.cli.init_cmd import init  # noqa: E402
from quant_lib.cli.list_cmd import list_cmd  # noqa: E402
from quant_lib.cli.show import show  # noqa: E402
from quant_lib.cli.explore import explore  # noqa: E402
from quant_lib.cli.commit_cmd import commit  # noqa: E402
from quant_lib.cli.status_cmd import status  # noqa: E402
from quant_lib.cli.migrate_seals import migrate_seals_cmd  # noqa: E402


app.command("init", help="Scaffold a new experiment file and .env template.")(init)
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
app.command(
    "migrate-seals",
    help=(
        "Re-sign holdout seals with the current HMAC secret. "
        "Use after rotating QUANT_LIB_HMAC_SECRET or upgrading "
        "from a pre-0.3.0 install."
    ),
)(migrate_seals_cmd)


if __name__ == "__main__":
    app()
