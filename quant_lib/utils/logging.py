"""Structured logging setup for the CLI.

Provides a single ``setup_logging()`` function that configures Rich
console output and optionally a file handler.
"""
from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


# Module-level console for shared use
_console = Console()


def get_console() -> Console:
    """Return the module-level Rich Console instance."""
    return _console


def setup_logging(verbose: int = 0, log_file: Path | None = None) -> None:
    """Configure logging for the CLI.

    Parameters
    ----------
    verbose : int
        Verbosity level:
        - 0 = WARNING
        - 1 = INFO
        - 2 = DEBUG
    log_file : Path, optional
        If given, also write logs to this file (in addition to console).
    """
    level = [logging.WARNING, logging.INFO, logging.DEBUG][min(verbose, 2)]

    handlers: list[logging.Handler] = [
        RichHandler(
            console=_console,
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        ),
    ]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,  # Override any existing config
    )
