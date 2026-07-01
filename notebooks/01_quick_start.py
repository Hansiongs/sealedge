"""Generate notebooks/01_quick_start.ipynb from this source.

Run this script once to (re)generate the .ipynb file:

    python notebooks/01_quick_start.py

The notebook format is well-defined JSON; this script emits the
canonical nbformat v4 structure so we don't need the `nbformat`
package as a build-time dependency.
"""
from __future__ import annotations
import json
from pathlib import Path


def make_cell(cell_type: str, source: str) -> dict:
    """Create an nbformat cell."""
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def make_md(source: str) -> dict:
    return make_cell("markdown", source)


def make_code(source: str) -> dict:
    return make_cell("code", source)


nb = {
    "cells": [
        make_md(
            "# 01 — Quick Start\n"
            "\n"
            "This notebook walks through the basic `quant_lib` workflow:\n"
            "list available experiments, run exploration, inspect output.\n"
            "\n"
            "**Prerequisites:** `pip install -e .` and a configured "
            "`.env` with `QUANT_LIB_HMAC_SECRET` (run `quant_exp init`)."
        ),
        make_md("## Setup"),
        make_code(
            "import quant_lib\n"
            "print(f\"quant_lib version: {quant_lib.__version__}\")"
        ),
        make_md("## List available experiments"),
        make_code(
            "from quant_lib.cli.list_cmd import list_cmd\n"
            "list_cmd()  # prints all registered experiments"
        ),
        make_md("## Run exploration (Phase 0-3)"),
        make_code(
            "# Run the exploration phase: fetch data, compute features,\n"
            "# walk-forward analysis, SPA p-value, per-symbol risk allocation.\n"
            "result = quant_lib.run_explore(\n"
            "    experiment_name=\"vol_compression_v1\",\n"
            "    cache_dir=\"./data_cache\",\n"
            ")\n"
            "# result is a Candidate in the \"narrowed\" stage.\n"
            "print(f\"Stage: {result.stage}\")\n"
            "print(f\"SPA p-value: {result.spa_p_value:.4f}\")\n"
            "print(f\"Eligible symbols: {result.eligible_symbols}\")"
        ),
        make_md("## Inspect result"),
        make_code(
            "# Per-symbol PSR (Probabilistic Sharpe Ratio)\n"
            "for sym, metrics in result.edge_metrics.items():\n"
            "    print(f\"{sym}: SR={metrics['sr']:.2f}, PSR={metrics['psr']:.4f}\")"
        ),
        make_md(
            "## Next steps\n"
            "\n"
            "- See `02_custom_experiment.ipynb` to register your own strategy.\n"
            "- See `03_interpreting_results.ipynb` for SPA/PSR/FDR deep-dive."
        ),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).parent / "01_quick_start.ipynb"
out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out_path}")
