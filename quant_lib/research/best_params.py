"""Best-params selection across WFA folds.

Decision (Q1, 2025-06-25): per symbol, pick the fold with the highest
PSR (encoded as ``best_value`` in the fold dict) across all WFA folds.

Rationale
----------
This is consistent with live trading. In live, the trader runs Optuna
on all available history and uses the best params. The previous approach
("frozen from last fold") was defensible as "mimic live: optuna on
recent data" but inconsistent with the user's stated live workflow.

Anti-overfit stack (already in place, no changes here):
- PSR (probabilistic Sharpe, not raw SR)
- ESS (effective sample size, handles autocorrelation)
- FDR (Benjamini-Hochberg per-symbol correction)
- L2 reg (param_center + param_scale, pulls params to center)

No additional stability-weighting is applied here. The four anti-overfit
mechanisms above are sufficient.

Note
----
The previous function ``compute_frozen_params_best_last`` (last fold
approach) is preserved as a deprecation shim in
``quant_lib.core._wfa``. New code should use
:func:`pick_best_params_per_symbol` from this module.
"""
from __future__ import annotations

from typing import Any


# Safe defaults used when WFA produces no folds OR when a tunable
# key is missing from the best fold. Hand-picked "safe" values,
# NOT search_space midpoints. Backward compat with previous behavior.
_SAFE_DEFAULTS_VC: dict[str, Any] = {
    "vol_pct_thresh": 0.20,
    "pullback_bars": 5,
    "trail_atr": 3.0,
    "sl_mult": 1.5,
}
_SAFE_DEFAULTS_PB_EXTRA: dict[str, Any] = {
    "rsi_oversold": 30.0,
    "rsi_overbought": 70.0,
}

# Keys that are NOT frozen params (fold-specific metadata).
# Generic extraction skips these automatically.
_NON_FROZEN_KEYS: frozenset[str] = frozenset({
    "symbol", "fold", "total_folds",
    "is_start", "oos_start", "oos_end",
    "best_value",
})


def _get_safe_defaults(strategy_type: int) -> dict[str, Any]:
    """Return strategy-specific safe defaults (copy, not reference)."""
    if strategy_type == 1:
        return {**_SAFE_DEFAULTS_VC, **_SAFE_DEFAULTS_PB_EXTRA}
    return dict(_SAFE_DEFAULTS_VC)


def _extract_params_from_fold(fold: dict, defaults: dict[str, Any]) -> dict:
    """Extract numeric param values from a fold dict.

    Defensive: skips non-numeric values, metadata, and bools.
    Casts ``pullback_bars`` to int. Backfills missing keys from defaults.
    """
    params: dict[str, Any] = {}
    for key, val in fold.items():
        if key in _NON_FROZEN_KEYS:
            continue
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        if key == "pullback_bars":
            params[key] = int(round(float(val)))
        else:
            params[key] = float(val)
    # Backfill any missing keys from defaults
    for key, default_val in defaults.items():
        params.setdefault(key, default_val)
    return params


def pick_best_params_per_symbol(
    all_fold_params: dict[str, list[dict]],
    strategy_type: int = 0,
) -> dict[str, dict]:
    """For each symbol, pick the fold with the highest ``best_value``.

    Parameters
    ----------
    all_fold_params : dict
        ``{symbol: [fold_dict, ...]}`` from WFA.
        Each ``fold_dict`` should have ``best_value`` (float, the PSR *
        trade_weight from Optuna). If no fold has ``best_value`` for a
        given symbol, the last fold is used as fallback.
    strategy_type : int
        ``0`` = vol_compression (default), ``1`` = pullback_sniper.
        Determines which strategy-specific safe defaults are used when
        WFA produces no folds or when tunable keys are missing.

    Returns
    -------
    dict
        ``{symbol: {param_name: value, ...}}`` -- params from the BEST
        fold (highest ``best_value`` / PSR) for that symbol.

    Notes
    -----
    - Generic across strategies: takes all numeric keys from the best
      fold. Pullback_sniper RSI params (``rsi_oversold``,
      ``rsi_overbought``) are picked up automatically (no C-3 bug).
    - Defensive: skips non-numeric values, metadata, and bools.
    - ``pullback_bars`` is cast to int (rounded).
    - Tie-breaking: ``max()`` returns the first fold encountered with
      the maximum ``best_value`` (Python's stable max).
    """
    defaults = _get_safe_defaults(strategy_type)
    frozen: dict[str, dict] = {}

    for sym, folds in all_fold_params.items():
        if not folds:
            # No folds at all: use safe defaults
            frozen[sym] = dict(defaults)
            continue

        # Pick fold with highest best_value. Fall back to last fold if
        # no fold in the list has best_value (e.g., synthetic data).
        if any("best_value" in f for f in folds):
            best_fold = max(folds, key=lambda f: f.get("best_value", float("-inf")))
        else:
            best_fold = folds[-1]

        frozen[sym] = _extract_params_from_fold(best_fold, defaults)

    return frozen


__all__ = ["pick_best_params_per_symbol"]

