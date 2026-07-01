"""Shared CLI utility functions.

Internal helpers used by multiple CLI subcommands. Sprint 2 fix 2.5:
extracted ``_looks_like_absolute`` from ``explore.py`` and
``commit_cmd.py`` to deduplicate the identical 4-line helper.
"""
from __future__ import annotations

import os


def looks_like_absolute(path_str: str) -> bool:
    """Cheap absolute-path detection.

    Returns True for paths that look absolute on either Windows
    (drive letter like ``C:\\foo``) or POSIX (leading ``/``).

    On Windows, ``os.path.isabs`` is the canonical check; on POSIX,
    ``/tmp/foo`` is absolute but ``C:\\foo`` is not. This function
    uses ``os.path.isabs`` which delegates to the platform, so it
    matches the user-OS's notion of "absolute".

    Used by ``--report`` path resolution in ``explore.py`` and
    ``commit_cmd.py``: a relative path is resolved under the run
    directory, an absolute path is used as-is.

    Parameters
    ----------
    path_str : str
        Path string to test.

    Returns
    -------
    bool
        True if ``path_str`` is absolute on the current OS.
    """
    return os.path.isabs(path_str)


__all__ = ["looks_like_absolute"]
