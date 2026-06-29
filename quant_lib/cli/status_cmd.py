"""quant_exp status -- Show holdout seal status and recent runs."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from rich.table import Table

from quant_lib.core._logging import console


# Seal directory discovery:
#   1. QUANT_LIB_SEAL_DIR env var (if set, takes precedence)
#   2. <cwd>/data_cache/holdout_seals (default for ad-hoc runs)
#
# ResearchSession writes seals to ``os.path.join(cache_dir, "holdout_seals")``
# unless ``seal_dir`` is passed explicitly. For a CLI user running
# ``quant_exp status``, the cache dir is typically ``./data_cache``, so
# this default matches. Users with non-default cache_dir should set
# QUANT_LIB_SEAL_DIR to point at the correct location.
_DEFAULT_SEAL_DIR = Path("data_cache/holdout_seals")
_SEAL_DIR = Path(os.environ.get("QUANT_LIB_SEAL_DIR", str(_DEFAULT_SEAL_DIR)))
_RESULTS_DIR = Path("results")

# Regex to parse OutputManager's dir_name format:
#   {YYYY-MM-DD}_{HHMMSS}_{experiment_name}_{mode}[_{git_short_hash}]
# experiment_name can contain underscores (validated by [a-z0-9_]+),
# so we cannot split by "_" with a fixed maxsplit. Use regex instead.
_RUN_NAME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<time>\d{6})_"
    r"(?P<name>.+?)_"
    r"(?P<mode>explore|commit)"
    r"(?:_(?P<git>[a-f0-9]+))?$"
)


def _parse_run_name(name: str) -> tuple[str, str, str] | None:
    """Parse OutputManager's run directory name into (ts, exp, mode).

    Returns None if the name doesn't match the expected format.
    """
    m = _RUN_NAME_RE.match(name)
    if m is None:
        return None
    return f"{m['date']}_{m['time']}", m["name"], m["mode"]


def status() -> None:
    """Show holdout seal status and recent runs."""
    if not _SEAL_DIR.exists():
        console.print("[yellow]No holdout seals found.[/yellow]")
        console.print(
            "Seals are created when a [bold]ResearchSession[/bold] is initialized."
        )
        return

    # Find most recent seal
    seals = sorted(_SEAL_DIR.glob("holdout_*.json"), reverse=True)
    if not seals:
        console.print("[yellow]No holdout seals found in[/yellow]", _SEAL_DIR)
        return

    try:
        latest = json.loads(seals[0].read_text())
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"[red]Failed to read seal file:[/red] {e}")
        return

    is_broken = latest.get("broken_at") is not None
    status_label = "[red]BROKEN[/red]" if is_broken else "[green]SEALED[/green]"

    tbl = Table(title="Holdout Seal", show_header=False)
    tbl.add_column(style="bold", no_wrap=True)
    tbl.add_column()
    tbl.add_row("Period", f"{latest.get('start', '?')} \u2192 {latest.get('end', '?')}")
    tbl.add_row("Status", status_label)
    tbl.add_row("Sealed at", str(latest.get("sealed_at", "?")))
    if is_broken:
        tbl.add_row("Broken at", str(latest.get("broken_at", "?")))
    data_hash = str(latest.get("data_hash", ""))
    if data_hash:
        tbl.add_row("Data hash", data_hash[:32] + ("..." if len(data_hash) > 32 else ""))
    tbl.add_row("File", str(seals[0].name))
    console.print(tbl)

    # Show recent runs
    if _RESULTS_DIR.exists():
        runs = sorted(
            [p for p in _RESULTS_DIR.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:5]
        if runs:
            console.print()
            rtbl = Table(title="Recent Runs (latest 5)", show_header=True)
            rtbl.add_column("Timestamp", style="bold")
            rtbl.add_column("Experiment")
            rtbl.add_column("Mode")
            for r in runs:
                parsed = _parse_run_name(r.name)
                if parsed is not None:
                    ts, name, mode = parsed
                    rtbl.add_row(ts, name, mode)
                else:
                    rtbl.add_row(r.name, "?", "?")
            console.print(rtbl)
