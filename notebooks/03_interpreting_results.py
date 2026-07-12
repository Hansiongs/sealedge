"""Generate notebooks/03_interpreting_results.ipynb.

Run:    python notebooks/03_interpreting_results.py
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
            "# 03 - Interpreting Results\n"
            "\n"
            "SPA (explore path), PSR (Bailey & Lopez de Prado 2014; holdout "
            "on commit), FDR, and risk-allocation notes."
        ),
        make_md(
            "## SPA p-value (Superior Predictive Ability)\n"
            "\n"
            "SPA asks whether observed edge beats a resampling null. "
            "Explore reports portfolio SPA; paper-grade sample numbers "
            "come from that path with `n_spa_iters=2000`.\n"
            "\n"
            "**Interpretation tiers (context labels, not auto-verdicts):**\n"
            "- `p < 0.0025` → PROD\n"
            "- `0.0025 <= p < 0.025` → TRADE\n"
            "- `0.025 <= p < 0.075` → WATCH\n"
            "- `0.075 <= p < 0.15` → RESEARCH\n"
            "- `p >= 0.15` → NO EDGE\n"
            "\n"
            "**Caveats:**\n"
            "- Low p + tiny trade count is weak evidence.\n"
            "- SPA is not a live-trading pass.\n"
            "- Paper sample uses sealed explore only (no holdout commit)."
        ),
        make_md(
            "## PSR (Probabilistic Sharpe Ratio)\n"
            "\n"
            "PSR is probability that true Sharpe exceeds a benchmark "
            "(skew/kurtosis adjusted). The paper does **not** report "
            "holdout PSR for the explore-only sample; that metric appears "
            "after `run_commit`. The helper below is for learning the "
            "statistic on synthetic trades."
        ),
        make_code(
            "from quant_lib.core._testing import prob_sharpe_ratio, label_p_value\n"
            "import numpy as np\n"
            "\n"
            "rng = np.random.default_rng(42)\n"
            "trade_pnl = rng.normal(0.5, 1.0, 100)  # synthetic 100 trades\n"
            "sr, psr = prob_sharpe_ratio(trade_pnl, annualize=False)\n"
            "print(f\"Sharpe Ratio: {sr:.4f}\")\n"
            "print(f\"PSR: {psr:.4f}\")\n"
            "label, conf, interp = label_p_value(1 - psr, context=\"mean_r\")\n"
            "print(f\"Label: {label} ({conf})\")"
        ),
        make_md(
            "## FDR (Benjamini-Hochberg)\n"
            "\n"
            "When testing many symbols/hypotheses, FDR controls the expected "
            "fraction of false discoveries among rejections."
        ),
        make_code(
            "from quant_lib.core._testing import fdr_correction\n"
            "import numpy as np\n"
            "\n"
            "p_values = np.array([0.001, 0.05, 0.2, 0.4, 0.6])\n"
            "rejected, q_values = fdr_correction(p_values, alpha=0.05)\n"
            "print(\"Rejected:\", rejected)\n"
            "print(\"Adjusted p-values:\", q_values)\n"
            "\n"
            "rejected_loose, q_values_loose = fdr_correction(p_values, alpha=0.15)\n"
            "print(f\"At alpha=0.15: rejected {rejected_loose.sum()}/{len(p_values)}\")"
        ),
        make_md(
            "## Risk allocation weights\n"
            "\n"
            "Per-fold PF-weighted risk allocation:\n"
            "- Scale symbol size with recent PF\n"
            "- Clamp (default floor/ceiling) to avoid runaway concentration\n"
            "- Renormalize so total risk stays on target\n"
            "\n"
            "Live-style weights come from the last fold after explore."
        ),
        make_md(
            "## Next steps\n"
            "\n"
            "- `docs/methodology.md` for the statistical stack.\n"
            "- `scripts/reproduce.py` for paper-grade explore metrics.\n"
            "- Commit only when you intentionally break the seal."
        ),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "03_interpreting_results.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out}")
