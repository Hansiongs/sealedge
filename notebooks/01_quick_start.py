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
            "#\n"
            "# Post Hansen-literal SPA (claim #3 Blocker A fix):\n"
            "#  - ``result`` is an ``ExploreResult`` dataclass (Sprint 3 fix\n"
            "#    3.3). 8 fields: ``experiment``, ``n_oos_trades``, ``n_executed``,\n"
            "#    ``n_rejected``, ``final_equity``, ``spa_p_value``,\n"
            "#    ``spa_naive_p_value``, ``narrowed_symbols``. ``spa_p_value``\n"
            "#    now carries the Hansen-corrected p when WFA ``trial_r_nets``\n"
            "#    are available (NaN-safe fallback to legacy p); the legacy\n"
            "#    circular-permutation p is preserved in ``spa_naive_p_value``\n"
            "#    for transparency. There is no ``stage`` / ``eligible_symbols``\n"
            "#    attribute on ExploreResult (those are Candidate-only\n"
            "#    legacy fields).\n"
            "result = quant_lib.run_explore(\n"
            "    experiment_name=\"vol_compression_v1\",\n"
            "    cache_dir=\"./data_cache\",\n"
            ")\n"
            "print(f\"Experiment: {result.experiment}\")\n"
            "print(f\"Narrowed symbols: {result.narrowed_symbols}\")\n"
            "print(f\"SPA p-value (Hansen when trial_r_nets available, \"\n"
            "      f\"fallback to legacy): {result.spa_p_value:.4f}\")\n"
            "print(f\"Legacy SPA p-value (preserved for transparency): \"\n"
            "      f\"{result.spa_naive_p_value}\")"
        ),
        make_md("## Inspect result"),
        make_code(
            "# ``edge_metrics`` is a flat TOP-LEVEL dict (Sprint 3 fix 3.3\n"
            "# + Phase 6 wire-up), not the prior per-symbol nested shape.\n"
            "# Keys: ``n_oos_trades``, ``n_executed``, ``n_rejected``,\n"
            "# ``final_equity``, ``spa_p_value``, ``spa_naive_p_value``,\n"
            "# ``spa_joint_k_trials``, ``hansen_fallback``. Documented\n"
            "# in ``Candidate.run_edge_testing``.\n"
            "for key, value in sorted(result.edge_metrics.items()):\n"
            "    if isinstance(value, float):\n"
            "        print(f\"  {key}: {value:.4f}\")\n"
            "    else:\n"
            "        print(f\"  {key}: {value}\")"
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
