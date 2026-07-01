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
            "# 03 — Interpreting Results\n"
            "\n"
            "Deep-dive into the statistical outputs:\n"
            "**SPA p-value**, **PSR (Probabilistic Sharpe Ratio)**, "
            "**FDR (False Discovery Rate)**, and **risk allocation weights**."
        ),
        make_md(
            "## SPA p-value (Superior Predictive Ability)\n"
            "\n"
            "Hansen (2005) SPA tests whether the observed strategy edge "
            "is real or just the best of many random alternatives.\n"
            "\n"
            "**Interpretation tiers (per-symbol SPA context):**\n"
            "- `p < 0.0025` → PROD (very strong evidence)\n"
            "- `0.0025 <= p < 0.025` → TRADE (strong evidence)\n"
            "- `0.025 <= p < 0.075` → WATCH (moderate evidence)\n"
            "- `0.075 <= p < 0.15` → RESEARCH (weak evidence)\n"
            "- `p >= 0.15` → NO EDGE\n"
            "\n"
            "**Caveats:**\n"
            "- SPA accounts for cross-symbol multiple comparisons.\n"
            "- Low p-value + small sample size = noise. Always inspect "
            "trade count (need >=30 for reliable inference).\n"
            "- SPA does NOT replace live trading validation."
        ),
        make_md("## PSR (Probabilistic Sharpe Ratio)"),
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
            "When testing MULTIPLE symbols, use FDR to control the false "
            "discovery rate (expected proportion of false positives)."
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
            "# Try a higher alpha for more rejections\n"
            "rejected_loose, q_values_loose = fdr_correction(p_values, alpha=0.15)\n"
            "print(f\"At alpha=0.15: rejected {rejected_loose.sum()}/{len(p_values)}\")"
        ),
        make_md(
            "## Risk allocation weights\n"
            "\n"
            "Per-fold PF (profit factor) weighted risk allocation:\n"
            "- Each symbol's position size scales with its recent PF\n"
            "- Clamped to `[0.5, 1.5]` to prevent runaway concentration\n"
            "- Total risk is preserved (renormalized across symbols)\n"
            "\n"
            "Inspect with `result.risk_weights` (per-fold) or "
            "`extract_final_fold_weights(result.risk_weights)` for the "
            "live-trading weights."
        ),
        make_md(
            "## Next steps\n"
            "\n"
            "- Review `docs/methodology.md` for full statistical theory.\n"
            "- Use `print_candidate_report(result)` for a formatted summary.\n"
            "- Use `plot_equity_curve(result.daily_equity)` to visualize."
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
