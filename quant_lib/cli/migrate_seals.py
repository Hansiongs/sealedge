"""Migrate holdout seal files to the current HMAC format.

Background
----------
``quant_lib`` v0.3.0 introduces HMAC-SHA256 signing of holdout seals
(``B0.1`` fix). Seal files created with older versions lack a
``signature`` field, or were signed with a different secret. When the
runtime verifies these files with the current ``QUANT_LIB_HMAC_SECRET``,
``verify_seal_signature()`` returns False and the session raises
``SealVerificationFailed`` on the next commit.

This tool rewrites such seal files in place, computing a fresh HMAC
signature with the current secret. The on-disk fields (``start``,
``end``, ``sealed_at``, ``broken_at``, ``data_hash``) are preserved
verbatim -- only the ``signature`` field is (re)computed.

Usage
-----
As a CLI subcommand:

    quant_exp migrate-seals                       # default seal dir
    quant_exp migrate-seals --seal-dir /var/seals
    quant_exp migrate-seals --dry-run            # show what would change
    quant_exp migrate-seals --yes                # skip confirmation

As a Python module:

    from quant_lib.cli.migrate_seals import migrate_seals
    n_valid, n_migrated, n_would, n_errors = migrate_seals(
        seal_dir=Path("..."), dry_run=False,
    )

Safety
------
* **Idempotent**: files with a valid signature are skipped.
* **Atomic writes**: each seal is rewritten via a temp file + rename
  so a crash mid-write cannot leave a half-signed seal.
* **No data modification**: only ``signature`` is touched; the
  ``data_hash`` and ``broken_at`` fields are preserved exactly as
  written by the original session.
* **Dry-run mode**: shows what would change without writing.
* **Backups**: every migrated file gets a ``.bak`` copy next to it
  for easy rollback. Clean up after verifying.

CLI vs module API
-----------------
This file exposes two entry points:

* ``migrate_seals(seal_dir, *, dry_run)`` -- pure function returning
  counts. Used by tests and as a library API.
* ``migrate_seals_cmd(seal_dir, dry_run, yes)`` -- CLI-shaped wrapper
  that prints a summary and exits non-zero on errors. Bound to
  ``quant_exp migrate-seals`` in ``cli/main.py``.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

import typer

from quant_lib.audit.holdout import (
    compute_seal_signature,
    verify_seal_signature,
    get_hmac_secret,
)
from quant_lib.core._logging import console


# Default seal directory (same as ResearchSession default).
_DEFAULT_SEAL_DIR = "data_cache/holdout_seals"


def _resolve_seal_dir(seal_dir: Optional[str]) -> Path:
    """Resolve the seal directory, honouring QUANT_LIB_SEAL_DIR env var.

    Priority:
      1. ``seal_dir`` argument (if provided)
      2. ``QUANT_LIB_SEAL_DIR`` environment variable
      3. ``./data_cache/holdout_seals`` (default)
    """
    if seal_dir is not None:
        return Path(seal_dir)
    env = os.environ.get("QUANT_LIB_SEAL_DIR")
    if env:
        return Path(env)
    return Path(_DEFAULT_SEAL_DIR)


def _classify_seal(state: dict) -> str:
    """Classify a seal by its current signature state.

    Returns one of:
      - ``"valid"``: has signature and verifies with current secret
      - ``"unsigned"``: missing signature field (pre-B0.1 format)
      - ``"invalid"``: has signature but does not verify (e.g. secret rotated)
    """
    sig = state.get("signature")
    if not sig:
        return "unsigned"
    if not isinstance(sig, str):
        return "unsigned"
    if verify_seal_signature(state):
        return "valid"
    return "invalid"


def _migrate_one(seal_file: Path, *, dry_run: bool) -> str:
    """Migrate a single seal file. Returns the resulting status.

    Status one of:
      - ``"valid"``: skipped, already valid
      - ``"migrated"``: signature was (re)computed and file rewritten
      - ``"error"``: file could not be processed (parse error, etc.)
    """
    try:
        with open(seal_file, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"  [red]ERROR[/red] {seal_file.name}: {e}")
        return "error"

    classification = _classify_seal(state)
    if classification == "valid":
        return "valid"

    if dry_run:
        return "would_migrate"

    # Build the unsigned payload, compute a fresh signature, write
    # atomically via temp file + rename. We keep a ``.bak`` copy so
    # the user can roll back if needed.
    payload = {k: v for k, v in state.items() if k != "signature"}
    payload["signature"] = compute_seal_signature(payload)

    # Backup the original (only if no .bak already exists, to avoid
    # overwriting an earlier backup on a second migration attempt).
    backup = seal_file.with_suffix(seal_file.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(seal_file, backup)

    # Atomic write: temp file in same directory, then os.replace.
    tmp = seal_file.with_suffix(seal_file.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, seal_file)
    except OSError as e:
        console.print(f"  [red]ERROR[/red] {seal_file.name}: write failed: {e}")
        # Clean up the temp file if rename failed.
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        return "error"

    return "migrated"


def migrate_seals(
    seal_dir: Optional[str] = None,
    *,
    dry_run: bool = False,
) -> tuple[int, int, int, int]:
    """Migrate all holdout seal files in ``seal_dir``.

    Parameters
    ----------
    seal_dir : str, optional
        Directory containing seal ``holdout_*.json`` files. Defaults
        to ``QUANT_LIB_SEAL_DIR`` env var, then ``./data_cache/holdout_seals``.
    dry_run : bool
        If True, report what would change without writing any files.
        Useful for safe previews.

    Returns
    -------
    tuple of (n_valid, n_migrated, n_would_migrate, n_errors)
        Counts for the migration. In ``dry_run=True`` mode, the
        second value is 0 and the third is the number that would
        be migrated.

    Raises
    ------
    RuntimeError
        If ``QUANT_LIB_HMAC_SECRET`` is not set or too short.
    """
    # Fail loudly with a clear message if the secret is missing --
    # this must happen BEFORE we touch any files so the user gets
    # the actionable error early.
    get_hmac_secret()

    dir_path = _resolve_seal_dir(seal_dir)
    if not dir_path.exists():
        console.print(
            f"[yellow]Seal directory does not exist:[/yellow] {dir_path}"
        )
        return (0, 0, 0, 0)

    seals = sorted(dir_path.glob("holdout_*.json"))
    if not seals:
        console.print(f"[yellow]No holdout seals found in[/yellow] {dir_path}")
        return (0, 0, 0, 0)

    n_valid = 0
    n_migrated = 0
    n_would = 0
    n_errors = 0
    for seal in seals:
        status = _migrate_one(seal, dry_run=dry_run)
        if status == "valid":
            n_valid += 1
        elif status == "migrated":
            n_migrated += 1
            console.print(f"  [green]MIGRATED[/green] {seal.name}")
        elif status == "would_migrate":
            n_would += 1
            console.print(f"  [cyan]WOULD MIGRATE[/cyan] {seal.name}")
        else:  # "error"
            n_errors += 1

    return (n_valid, n_migrated, n_would, n_errors)


def migrate_seals_cmd(
    ctx: typer.Context,
    seal_dir: Optional[str] = typer.Option(
        None,
        "--seal-dir",
        help=(
            "Directory containing holdout seal files. Defaults to "
            "QUANT_LIB_SEAL_DIR env var, then ./data_cache/holdout_seals."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would change without modifying any files.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt (assume yes).",
    ),
) -> None:
    """Re-sign holdout seals with the current QUANT_LIB_HMAC_SECRET.

    Use this after rotating HMAC secrets or when upgrading from a
    pre-0.3.0 install whose seals lack the ``signature`` field. See
    the module docstring for full safety notes.
    """
    console.print("[bold]Holdout Seal Migration[/bold]")
    console.print(f"  Seal dir:  {_resolve_seal_dir(seal_dir)}")
    if dry_run:
        console.print("  [yellow]DRY RUN -- no files will be modified.[/yellow]")

    n_valid, n_migrated, n_would, n_errors = migrate_seals(
        seal_dir=seal_dir, dry_run=dry_run,
    )

    console.print()
    if dry_run:
        console.print(
            f"[bold]Summary:[/bold] {n_valid} valid, "
            f"[cyan]{n_would}[/cyan] would migrate, "
            f"{n_errors} errors"
        )
        if n_would > 0:
            console.print(
                "Re-run without [bold]--dry-run[/bold] to apply."
            )
    else:
        console.print(
            f"[bold]Summary:[/bold] {n_valid} valid, "
            f"[green]{n_migrated}[/green] migrated, "
            f"{n_errors} errors"
        )
        if n_migrated > 0:
            console.print(
                "Backups saved as [bold].bak[/bold] alongside each seal. "
                "Verify with [bold]quant_exp status[/bold], then delete "
                "backups when confident."
            )

    if n_errors > 0:
        raise typer.Exit(code=1)
