"""Generate notebooks/01_quick_start.ipynb from this source.

    python notebooks/01_quick_start.py

Emits nbformat v4 JSON without needing the ``nbformat`` package.
"""
from __future__ import annotations
import json
from pathlib import Path


def make_cell(cell_type: str, source: str) -> dict:
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
            "# 01 - Quick Start\n"
            "\n"
            "Basic `quant_lib` workflow: list experiments, run explore, "
            "inspect SPA metrics. Holdout seal stays closed.\n"
            "\n"
            "**Prerequisites:** `pip install -e .` and "
            "`QUANT_LIB_HMAC_SECRET` (`quant_exp init`)."
        ),
        make_md("## Setup"),
        make_code(
            "import quant_lib\n"
            "print(f\"quant_lib version: {quant_lib.__version__}\")"
        ),
        make_md("## List available experiments"),
        make_code(
            "from quant_lib.cli.list_cmd import list_cmd\n"
            "list_cmd()  # prints registered experiments"
        ),
        make_md("## Run explore (seal stays closed)"),
        make_code(
            "# Explore: data, WFA, SPA. Does not break the holdout seal.\n"
            "# ExploreResult fields: experiment, n_oos_trades, n_executed,\n"
            "# n_rejected, final_equity, spa_p_value, spa_naive_p_value,\n"
            "# narrowed_symbols. spa_p_value is Hansen-corrected when\n"
            "# trial_r_nets exist; spa_naive_p_value keeps the legacy p.\n"
            "result = quant_lib.run_explore(\n"
            "    experiment_name=\"vol_compression_v1\",\n"
            "    cache_dir=\"./data_cache\",\n"
            ")\n"
            "print(f\"Experiment: {result.experiment}\")\n"
            "print(f\"Narrowed symbols: {result.narrowed_symbols}\")\n"
            "print(f\"SPA p-value: {result.spa_p_value:.4f}\")\n"
            "print(f\"Legacy SPA p-value: {result.spa_naive_p_value}\")\n"
            "print(f\"OOS trades: {result.n_oos_trades}\")\n"
            "print(f\"Final equity (explore path): {result.final_equity:.2f}\")"
        ),
        make_md("## Inspect result"),
        make_code(
            "# Dict-style access still works for older snippets.\n"
            "for key in result.keys():\n"
            "    value = result[key]\n"
            "    if isinstance(value, float):\n"
            "        print(f\"  {key}: {value:.4f}\")\n"
            "    else:\n"
            "        print(f\"  {key}: {value}\")"
        ),
        make_md(
            "## Next steps\n"
            "\n"
            "- `02_custom_experiment.ipynb`: register your own strategy.\n"
            "- `03_interpreting_results.ipynb`: SPA / PSR / FDR notes.\n"
            "- Paper-grade sample is explore-only (SPA + trades); holdout "
            "PSR is on `run_commit` only."
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
