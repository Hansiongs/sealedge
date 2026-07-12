"""
``python -m quant_lib`` entry point.

Delegates to the installed ``quant_exp`` CLI (``quant_lib.cli.main.app``).

Usage::

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
