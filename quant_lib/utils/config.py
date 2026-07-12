"""
Optional env helpers: repo root, ``.env`` load, HMAC secret fallback.

Primary seal secret path is ``quant_lib.audit.holdout.get_hmac_secret``.
These wrappers are for tooling/CI outside the standard CLI flow.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Walk up from ``start`` (default: ``Path.cwd()``) until a
    directory containing ``pyproject.toml`` is found.

    Parameters
    ----------
    start : pathlib.Path, optional
        Starting directory. Defaults to ``Path.cwd()``.

    Returns
    -------
    pathlib.Path
        The directory containing ``pyproject.toml``, or ``start`` if
        no ``pyproject.toml`` is found in any parent.

    Notes
    -----
    Used by ``load_env_file`` to locate the ``.env`` file at the
    repo root (the conventional location for project secrets).
    """
    cwd = start or Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return cwd


def load_env_file(env_path: Optional[Path] = None) -> dict[str, str]:
    """Parse a simple ``.env`` file into a ``dict``.

    Handles ``KEY=VALUE`` lines; ignores blanks and lines starting
    with ``#``. Values are stripped of surrounding quotes
    (single or double). Does NOT modify ``os.environ``, return
    the dict and let the caller decide.

    Parameters
    ----------
    env_path : pathlib.Path, optional
        Path to the ``.env`` file. Defaults to
        ``<repo_root>/.env`` (where ``repo_root`` is the directory
        containing ``pyproject.toml``).

    Returns
    -------
    dict[str, str]
        Parsed key-value pairs. Empty dict if the file doesn't exist.

    Examples
    --------
    >>> env = load_env_file()
    >>> "QUANT_LIB_HMAC_SECRET" in env
    True
    """
    env_path = env_path or (find_repo_root() / ".env")
    if not env_path.exists():
        return {}
    env: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        env[key] = val
    return env


def get_hmac_secret_with_fallback() -> str:
    """Get the HMAC secret, trying ``os.environ`` first then ``.env``.

    Returns
    -------
    str
        The HMAC secret value.

    Raises
    ------
    RuntimeError
        If the secret is not found in either location.

    Notes
    -----
    Thin wrapper around ``quant_lib.audit.holdout.get_hmac_secret``
    that first attempts to populate ``os.environ`` from the ``.env``
    file. Use this when running outside the standard CLI (which
    loads the env file automatically).
    """
    if "QUANT_LIB_HMAC_SECRET" not in os.environ:
        env = load_env_file()
        if "QUANT_LIB_HMAC_SECRET" in env:
            os.environ["QUANT_LIB_HMAC_SECRET"] = env["QUANT_LIB_HMAC_SECRET"]
    # Delegate to the canonical accessor in audit.holdout
    from quant_lib.audit.holdout import get_hmac_secret
    return get_hmac_secret().decode("utf-8")


__all__ = [
    "find_repo_root",
    "load_env_file",
    "get_hmac_secret_with_fallback",
]
