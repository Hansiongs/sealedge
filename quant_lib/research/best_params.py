"""Best-params selection across WFA folds.

Decision (R1, 2026-06-30): stability-gated best fold.

Selection rule:
  1. Find the fold with the highest ``best_value`` (PSR) -- the "best fold".
  2. Compute per-param CV (std/mean) across all folds for the param set
     of the best fold.
  3. If mean CV > ``cv_threshold`` (default 30%, aligned with
     ``print_param_stability`` UNSTABLE rating):
     - Fall back to **median per param** across all folds.
     - This is the "stability gate": best fold wins when its params
       are robust across folds; median is the safety net when the
       best fold is a lucky outlier.
  4. Otherwise, use the best fold's params (preserves PSR signal).

Rationale
---------
Pure best-per-fold has N-test selection bias across folds (pick the
max of N correlated PSR estimates). Pure median ignores the winning
signal entirely. Stability-gated selection preserves the best signal
when the best fold's params are *consistent* across folds, and
defaults to consensus when the best fold is a *lucky* outlier (high
PSR but param set varies wildly between folds).

Anti-overfit stack (already in place, upstream of selection):
- PSR (probabilistic Sharpe, not raw SR) in WalkForwardObjective
- ESS (effective sample size, Kish formula, handles autocorrelation)
- Time decay 15-month halflife on bar weights in WFA (recent bars
  dominate; reduces N-fold selection bias by weighting toward recent
  folds' PSR)
- Expanding window + warm-start Optuna (4 trials: 1 enqueue + 3 perturbed)
- L2 reg (param_center + param_scale, pulls params to center)
- FDR (Benjamini-Hochberg per-symbol correction in reporting)

This module adds the *final* safety net: stability gating at selection
time. The CV threshold is 30% by default, aligned with the existing
``print_param_stability`` UNSTABLE rating in
``quant_lib.core._metrics.print_param_stability`` (line 205-213).

Note
----
The previous function ``compute_frozen_params_best_last`` (last fold
approach) is preserved as a deprecation shim in
``quant_lib.core._wfa``. New code should use
:func:`pick_best_params_per_symbol` from this module.
"""
from __future__ import annotations

import numpy as np
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


def _compute_param_cv(
    param_name: str,
    folds: list[dict],
) -> float | None:
    """Compute coefficient of variation (std/mean) for one param across folds.

    Returns None if fewer than 2 valid values or if mean is ~zero.
    """
    vals = [f.get(param_name) for f in folds if f.get(param_name) is not None]
    if len(vals) < 2:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    mean_v = float(np.mean(arr))
    if abs(mean_v) < 1e-10:
        # Avoid div by zero; near-constant params (e.g., default center)
        # have effectively zero CV -- report as 0 not None.
        return 0.0
    std_v = float(np.std(arr, ddof=1))
    return std_v / abs(mean_v)


def pick_best_params_per_symbol(
    all_fold_params: dict[str, list[dict]],
    strategy_type: int = 0,
    cv_threshold: float = 0.30,
) -> dict[str, dict]:
    """For each symbol, pick the best fold if its params are stable,
    else fall back to median across folds.

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
    cv_threshold : float
        Maximum mean coefficient of variation (across best fold's params)
        allowed for "stable" selection. If mean CV > cv_threshold,
        the selection falls back to per-param median across folds.
        Default 0.30 (30%), aligned with the UNSTABLE rating in
        ``quant_lib.core._metrics.print_param_stability``.

    Returns
    -------
    dict
        ``{symbol: {param_name: value, ...}}`` -- params from the BEST
        fold (highest ``best_value`` / PSR) for that symbol, or median
        across folds if best fold's params are unstable.

    Notes
    -----
    - Generic across strategies: takes all numeric keys from the best
      fold. Pullback_sniper RSI params (``rsi_oversold``,
      ``rsi_overbought``) are picked up automatically.
    - Defensive: skips non-numeric values, metadata, and bools.
    - ``pullback_bars`` is cast to int (rounded).
    - Tie-breaking: ``max()`` returns the first fold encountered with
      the maximum ``best_value`` (Python's stable max).
    - Stability gate: per-param CV is computed across ALL folds (not
      just best fold's neighbours). Mean CV across the best fold's
      param set determines gate. CV aligned with
      ``print_param_stability`` thresholds (STABLE<15%, MODERATE<30%,
      UNSTABLE>=30%).
    - Fallback to median produces a phantom-param (median between
      discrete Optuna samples) but is preferred over best fold's
      outlier params when stability gate triggers.
    """
    from quant_lib.core._logging import log

    defaults = _get_safe_defaults(strategy_type)
    frozen: dict[str, dict] = {}

    for sym, folds in all_fold_params.items():
        if not folds:
            # No folds at all: use safe defaults
            frozen[sym] = dict(defaults)
            continue

        # Step 1: Find best fold. Fall back to last fold if no fold
        # in the list has best_value (e.g., synthetic data).
        if any("best_value" in f for f in folds):
            best_fold = max(folds, key=lambda f: f.get("best_value", float("-inf")))
        else:
            best_fold = folds[-1]

        # Step 2: Compute mean CV across the best fold's params.
        # Only consider params that exist in best_fold AND in >=2 folds.
        param_names = [
            k for k in best_fold
            if k not in _NON_FROZEN_KEYS
            and isinstance(best_fold.get(k), (int, float))
            and not isinstance(best_fold.get(k), bool)
        ]
        cvs: list[float] = []
        for p in param_names:
            cv = _compute_param_cv(p, folds)
            if cv is not None:
                cvs.append(cv)
        mean_cv = float(np.mean(cvs)) if cvs else 0.0

        # Step 3: Stability gate. If best fold's params are too
        # volatile across folds, fall back to per-param median.
        if mean_cv > cv_threshold and len(folds) >= 2:
            agg: dict[str, Any] = {}
            for p in param_names:
                vals = [
                    f.get(p) for f in folds
                    if f.get(p) is not None
                ]
                if vals:
                    # Cast to np.asarray to satisfy mypy's type inference
                    # (vals is list[Any | None] even after the filter).
                    agg[p] = float(np.median(np.asarray(vals, dtype=np.float64)))
                else:
                    agg[p] = defaults.get(p, np.nan)
            # Cast pullback_bars to int (rounded).
            if "pullback_bars" in agg:
                agg["pullback_bars"] = int(round(agg["pullback_bars"]))
            log.warning(
                f"[{sym}] Best fold params unstable "
                f"(mean CV={mean_cv:.0%} > {cv_threshold:.0%}); "
                f"falling back to median across {len(folds)} folds. "
                f"Best fold had best_value="
                f"{best_fold.get('best_value', 'n/a')}."
            )
            frozen[sym] = agg
        else:
            # Stable (or only one fold): use best fold as-is.
            frozen[sym] = _extract_params_from_fold(best_fold, defaults)
            if mean_cv > 0 and cvs:
                log.info(
                    f"[{sym}] Best fold params stable "
                    f"(mean CV={mean_cv:.0%}); using best fold "
                    f"(best_value={best_fold.get('best_value', 'n/a')})."
                )

    return frozen


__all__ = ["pick_best_params_per_symbol"]

