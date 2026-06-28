"""
quant_lib entry point.

For the modern CLI (with list, show, explore, commit, status), use
the installed ``quant_exp`` command (see ``pyproject.toml``
``[project.scripts]``). This module delegates to it for backward
compatibility with ``python -m quant_lib <command>``.

Usage:
    python -m quant_lib --help
    python -m quant_lib list
    python -m quant_lib show vol_compression_v1
    python -m quant_lib explore vol_compression_v1
    python -m quant_lib commit vol_compression_v1
    python -m quant_lib status
"""
import sys

from quant_lib.cli.main import app


if __name__ == "__main__":
    sys.exit(app())
