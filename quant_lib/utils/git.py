"""Git utilities for run-time metadata injection.

Used by OutputManager to embed the current git commit hash in
metrics.json, ensuring paper artifacts are traceable.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def get_git_commit(short: bool = True, cwd: Path | None = None) -> str:
    """Get the current git commit hash.

    Parameters
    ----------
    short : bool
        If True, return the short (12-char) hash. Full hash is 40 chars.
    cwd : Path, optional
        Working directory for the git command. If None, uses current
        working directory.

    Returns
    -------
    str
        The commit hash, or ``"unknown"`` if not in a git repo or git
        is not available. Never raises.
    """
    try:
        args = ["git", "rev-parse"]
        if short:
            args.append("--short")
        args.append("HEAD")
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
            cwd=cwd,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        log.debug(f"git rev-parse failed: {type(e).__name__}: {e}")
        return "unknown"
