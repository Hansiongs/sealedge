"""Generate notebooks/02_custom_experiment.ipynb.

Run:    python notebooks/02_custom_experiment.py
"""
from __future__ import annotations
import json
from pathlib import Path


def make_cell(cell_type, source):
    return {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}


def make_md(s):
    return make_cell("markdown", s)


def make_code(s):
    return make_cell("code", s)


nb = {
    "cells": [
        make_md(
            "# 02 — Custom Experiment\n"
            "\n"
            "How to register a new strategy and run the full pipeline.\n"
            "\n"
            "Every experiment lives in `quant_lib/experiments/` as a "
            "small Python file that combines a `Hypothesis` with a "
            "`StrategyConfig` and calls `register()`."
        ),
        make_md("## 1. Define your hypothesis"),
        make_code(
            "from quant_lib.audit import for_vol_compression\n"
            "from quant_lib.experiments import (\n"
            "    PeriodConfig, UniverseConfig, StrategyConfig,\n"
            "    ExperimentConfig, register,\n"
            ")\n"
            "\n"
            "# The narrative fields are required for the experiment journal\n"
            "# and MUST be written BEFORE seeing any data. Use the factory\n"
            "# helpers (``for_vol_compression``, ``for_pullback_sniper``)\n"
            "# to get the correct ``strategy_type`` + default search space.\n"
            "hyp = for_vol_compression(\n"
            "    name=\"my_breakout_v1\",\n"
            "    mechanism=(\n"
            "        \"Volatility compression (vol_pct_rank < 0.15) \"\n"
            "        \"followed by volume-confirmed breakout of the 20-bar \"\n"
            "        \"high generates intraday momentum in liquid crypto \"\n"
            "        \"perpetuals.\"\n"
            "    ),\n"
            "    boundary_conditions=(\n"
            "        \"Fails in low-vol accumulation regimes where no \"\n"
            "        \"breakout follows compression. Fails on illiquid \"\n"
            "        \"pairs (< 50M USD daily volume).\"\n"
            "    ),\n"
            "    success_criteria=\"SPA p < 0.15, PF > 1.3, min 30 OOS trades\",\n"
            "    entry_logic=\"vol_pct_rank < 0.15 + close > HH_20 + rvol > 3.0\",\n"
            "    exit_logic=\"Trailing stop at ATR x 3.0, bailout at 36 bars\",\n"
            ")"
        ),
        make_md("## 2. Build the config"),
        make_code(
            "config = ExperimentConfig(\n"
            "    name=\"my_breakout_v1\",\n"
            "    strategy_type=\"vol_compression\",  # matches Hypothesis\n"
            "    hypothesis=hyp,\n"
            "    period=PeriodConfig(\n"
            "        train_start=\"2020-01-01\",\n"
            "        train_end=\"2024-12-31\",\n"
            "        # holdout_start/end auto-resolve to a post-training\n"
            "        # period of length ``holdout_months`` (default 6).\n"
            "    ),\n"
            "    universe=UniverseConfig(\n"
            "        symbols=[\"BTCUSDT\", \"ETHUSDT\", \"SOLUSDT\"],\n"
            "        min_volume_usdt=50_000_000,  # trailing 30d USD volume\n"
            "        min_age_days=365,\n"
            "    ),\n"
            "    strategy=StrategyConfig(\n"
            "        leverage=3.0,\n"
            "        pf_weight_clamp_floor=0.5,\n"
            "        pf_weight_clamp_ceiling=1.5,\n"
            "    ),\n"
            ")"
        ),
        make_md("## 3. Register and run"),
        make_code(
            "import quant_lib\n"
            "\n"
            "register(config)\n"
            "result = quant_lib.run_explore(\n"
            "    experiment_name=\"my_breakout_v1\",\n"
            "    cache_dir=\"./data_cache\",\n"
            ")"
        ),
        make_md(
            "## 4. Inspect & commit (Phase 4)\n"
            "\n"
            "If the SPA p-value is below your threshold and you want to "
            "test the strategy against the holdout:\n"
            "\n"
            "```python\n"
            "if result[\"spa_p_value\"] < 0.05:\n"
            "    commit_result = quant_lib.run_commit(\n"
            "        experiment_name=\"my_breakout_v1\",\n"
            "        cache_dir=\"./data_cache\",\n"
            "    )\n"
            "    print(f\"PSR: {commit_result.psr:.4f}\")\n"
            "    print(f\"Final equity: ${commit_result.final_equity:,.2f}\")\n"
            "```\n"
            "\n"
            "**WARNING:** `run_commit` is IRREVERSIBLE. The holdout seal "
            "is broken once you commit; subsequent runs cannot re-test "
            "against the same holdout."
        ),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "02_custom_experiment.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out}")
