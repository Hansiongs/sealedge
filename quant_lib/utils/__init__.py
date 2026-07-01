"""
quant_lib.utils -- Shared utilities.

Phase 4 (v0.5.0): re-exports config helpers for convenience.
    >>> from quant_lib.utils import find_repo_root, load_env_file
"""
from quant_lib.utils.config import (
    find_repo_root,
    load_env_file,
    get_hmac_secret_with_fallback,
)

__all__ = [
    "find_repo_root",
    "load_env_file",
    "get_hmac_secret_with_fallback",
]
