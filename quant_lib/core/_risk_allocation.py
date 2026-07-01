"""
Per-fold PF-weighted risk allocation (X1 scheme).

This module provides a self-contained, pure-Python implementation of
the per-fold decay-weighted Profit Factor (PF) risk rebalancing used
by the framework. It is the canonical risk-allocation path, called
from both ``Candidate.run_edge_testing`` (multi-fold WFA) and
``commit_to_holdout`` (no-op for holdout trades with no fold_key).

Design (X1 scheme):
  - For each fold, look at PAST folds' trades (decay-weighted)
  - Convert to clamped weight factor per symbol (in [floor, ceiling])
  - Rescale to preserve target total = baseline * (n_active_in_fold / n_total_symbols)
  - Apply by mutating ``t["risk_weight"]`` on each trade in current fold

Live-trading mirror: the framework re-allocates risk every 3 months
(one WFA fold), so the validation set itself validates the combined
edge of (strategy + meta-allocation).

No Numba, no heavy deps, leaf in the dependency DAG (imports only
from ``core/_config`` and ``core/_logging``).
"""
from __future__ import annotations

from collections import defaultdict

from quant_lib.core._config import DEFAULTS
from quant_lib.core._logging import log


# ─────────────────────────────────────────────────────────────────────
# Private helpers (extracted from prior single-file pipeline)
# ─────────────────────────────────────────────────────────────────────


def _compute_decay_weighted_pnl_loss(
    past_trades_by_fold: list[tuple[int, list[dict]]],
    halflife_folds: float,
) -> dict[str, tuple[float, float, int]]:
    """Compute decay-weighted (pnl, loss, n_trades) per symbol.

    Parameters
    ----------
    past_trades_by_fold : list of (n_folds_back, [trade_dict])
        Each tuple: (n_folds_back, [trades in that past fold]).
        n_folds_back >= 1 (1 = immediately previous fold).
        Trade dicts must have 'symbol' and 'r_net' fields.
    halflife_folds : float
        Decay halflife in folds. decay_weight = 0.5^(n/halflife).

    Returns
    -------
    dict
        {sym: (weighted_pnl, weighted_loss, n_trades)}.
        Symbols not in any past fold are absent.
    """
    sym_pnl: dict[str, float] = {}
    sym_loss: dict[str, float] = {}
    sym_count: dict[str, int] = {}
    for n_back, trades in past_trades_by_fold:
        decay_weight = 0.5 ** (n_back / halflife_folds)
        for t in trades:
            sym = t["symbol"]
            r = t["r_net"]
            sym_count[sym] = sym_count.get(sym, 0) + 1
            if r > 0:
                sym_pnl[sym] = sym_pnl.get(sym, 0.0) + r * decay_weight
            elif r < 0:
                sym_loss[sym] = sym_loss.get(sym, 0.0) + abs(r) * decay_weight
    out: dict[str, tuple[float, float, int]] = {}
    # Include ALL symbols that had any past trade, even if all r_net were 0
    # (those symbols should still be subject to the min_trades gate and
    # default to neutral factor, not silently excluded).
    for sym in sym_count:
        out[sym] = (
            sym_pnl.get(sym, 0.0),
            sym_loss.get(sym, 0.0),
            sym_count[sym],
        )
    return out


def _compute_clamped_factor(
    weighted_pnl: float,
    weighted_loss: float,
    n_past_trades: int,
    min_trades: int,
    clamp_floor: float,
    clamp_ceiling: float,
) -> float:
    """Convert decay-weighted pnl/loss to a clamped weight factor.

    Returns
    -------
    float
        Weight factor in [clamp_floor, clamp_ceiling], or 1.0 (neutral)
        when there is insufficient data or no signal.

    Rules
    -----
    - n_past_trades < min_trades: return 1.0 (insufficient data, neutral)
    - weighted_loss == 0 and weighted_pnl > 0: return clamp_ceiling
    - weighted_loss == 0 and weighted_pnl == 0: return 1.0 (no signal)
    - otherwise: return clamp(weighted_pnl / weighted_loss, floor, ceiling)
    """
    if n_past_trades < min_trades:
        return 1.0
    if weighted_loss > 0:
        pf = weighted_pnl / weighted_loss
        return max(clamp_floor, min(clamp_ceiling, pf))
    if weighted_pnl > 0:
        return clamp_ceiling
    return 1.0


def _rescale_factors_to_total(
    factors: dict[str, float],
    baseline_per_symbol: float,
    target_total: float,
) -> dict[str, float]:
    """Apply clamp*baseline, then rescale so sum equals target_total (X1).

    Parameters
    ----------
    factors : dict
        {sym: clamped_factor} for symbols to include in rescale.
    baseline_per_symbol : float
        Baseline weight per symbol (e.g., 0.01 for 1% per pair, 4 pairs).
    target_total : float
        Total weight to preserve (e.g., 0.04 for 4% total).

    Returns
    -------
    dict
        {sym: final_weight}. If sum of pre-rescale is 0 (all factors=0),
        returns empty dict (caller should handle as no weight).

        **Note on silent failure mode (v0.4.1 doc):** when this returns
        empty dict, the caller sets ``risk_weight=0.0`` for all trades
        in the fold (effectively skipping the fold). This is intentional
        behavior (no PF information → no allocation) but can be confusing
        if encountered unexpectedly. Investigate why all PF factors are
        zero (e.g., all trades in the fold lost money → PF=0 → factors=0).
    """
    if not factors or baseline_per_symbol <= 0:
        return {}
    pre_rescale = {sym: f * baseline_per_symbol for sym, f in factors.items()}
    sum_pre = sum(pre_rescale.values())
    if sum_pre <= 0:
        return {}
    rescale = target_total / sum_pre
    return {sym: w * rescale for sym, w in pre_rescale.items()}


# ─────────────────────────────────────────────────────────────────────
# Public orchestrator
# ─────────────────────────────────────────────────────────────────────


def apply_pf_weighted_risk_allocation(
    trades: list[dict],
    halflife_folds: float,
    clamp_floor: float,
    clamp_ceiling: float,
    min_trades: int,
    baseline_per_symbol: float,
    n_total_symbols: int,
    log_prefix: str = "PF Weight",
) -> dict[str, dict[str, float]]:
    """Per-fold PF-weighted risk allocation across OOS trades.

    For each unique fold (grouped by ``t["fold_key"]``):
      1. Build ``past_folds_data`` from all earlier folds (decay-weighted).
      2. Compute clamped factor per symbol via ``_compute_clamped_factor``.
      3. Rescale to preserve ``target_total = baseline * (n_active/n_total)``
         via ``_rescale_factors_to_total`` (X1 scheme: total risk preserved).
      4. Mutate each trade's ``risk_weight`` in current fold.

    No-op when no trade has a ``fold_key`` field (commit-path safety):
    holdout trades built by ``commit_to_holdout`` have no ``fold_key``,
    so the orchestrator returns an empty summary without touching
    any trades.

    Parameters
    ----------
    trades : list of dict
        OOS trades. Each must have ``symbol``, ``r_net``, ``fold_key``,
        and ``risk_weight``. ``fold_key`` is consumed (popped) after
        application; trades without ``fold_key`` are left untouched.
    halflife_folds : float
        Decay halflife in folds. ``decay_weight = 0.5^(n_back / halflife)``.
    clamp_floor, clamp_ceiling : float
        Weight factor bounds (e.g., 0.5 and 1.5).
    min_trades : int
        Minimum past trades required to apply a non-neutral factor.
        Below this, factor defaults to 1.0 (neutral).
    baseline_per_symbol : float
        Baseline weight per symbol (e.g., 0.01 = 1% per pair).
    n_total_symbols : int
        Total symbols in the universe. Used for per-fold budget
        scaling: ``target_total = baseline * n_total_symbols * (n_active/n_total)``.
    log_prefix : str, optional
        Prefix for the per-fold log line. Default "PF Weight".

    Returns
    -------
    dict
        ``{fold_key: {sym: final_weight}}`` summary. Empty dict when
        no trades have ``fold_key`` (commit-path no-op).
    """
    # Group trade indices by fold_key (skip trades without fold_key).
    trades_by_fold: dict[str, list[int]] = defaultdict(list)
    for t_idx, t in enumerate(trades):
        fk = t.get("fold_key")
        if fk is not None:
            trades_by_fold[fk].append(t_idx)

    if not trades_by_fold:
        return {}

    # Total baseline = baseline * n_total (6 default pairs * 0.01 historically).
    original_total = baseline_per_symbol * n_total_symbols

    summary: dict[str, dict[str, float]] = {}
    sorted_fold_keys = sorted(trades_by_fold.keys())

    for fold_idx, fk in enumerate(sorted_fold_keys):
        t_idxs = trades_by_fold[fk]

        # Symbols that had any trade in this fold.
        fold_syms: set[str] = {trades[tidx]["symbol"] for tidx in t_idxs}

        # Per-fold budget scale: how much of the original total is active.
        n_active = len(fold_syms)
        budget_scale = (n_active / n_total_symbols) if n_total_symbols > 0 else 1.0
        target_total_for_fold = original_total * budget_scale

        # === Past folds only (current fold excluded from its own PF) ===
        past_folds_data: list[tuple[int, list[dict]]] = []
        for n_back, past_fk in enumerate(sorted_fold_keys[:fold_idx], start=1):
            past_trade_dicts = [trades[tidx] for tidx in trades_by_fold[past_fk]]
            past_folds_data.append((n_back, past_trade_dicts))

        decay_pnl_loss = _compute_decay_weighted_pnl_loss(
            past_folds_data, halflife_folds
        )

        # === Clamped weight factors ===
        weight_factors: dict[str, float] = {}
        for sym in fold_syms:
            if sym not in decay_pnl_loss:
                # No past data for this symbol (e.g., first fold)
                weight_factors[sym] = 1.0
                continue
            w_pnl, w_loss, n_past = decay_pnl_loss[sym]
            weight_factors[sym] = _compute_clamped_factor(
                weighted_pnl=w_pnl,
                weighted_loss=w_loss,
                n_past_trades=n_past,
                min_trades=min_trades,
                clamp_floor=clamp_floor,
                clamp_ceiling=clamp_ceiling,
            )

        # === Rescale to preserve total (X1 scheme) ===
        final_weights = _rescale_factors_to_total(
            factors=weight_factors,
            baseline_per_symbol=baseline_per_symbol,
            target_total=target_total_for_fold,
        )

        # === Apply to trades in this fold ===
        for tidx in t_idxs:
            t = trades[tidx]
            sym = t["symbol"]
            if sym in final_weights:
                t["risk_weight"] = final_weights[sym]
            else:
                t["risk_weight"] = 0.0
            # Pop fold metadata so trades don't carry WFA-internal tags.
            t.pop("atr_inv", None)
            t.pop("fold_key", None)

        # === Per-fold log summary ===
        log_lines = [
            f"{sym}={final_weights.get(sym, 0.0):.4f} "
            f"(factor={weight_factors.get(sym, 1.0):.2f}x)"
            for sym in sorted(fold_syms)
        ]
        log.info(
            f"[{log_prefix}] Fold {fk}: {', '.join(log_lines)} "
            f"(budget_scale={budget_scale:.2f}, halflife={halflife_folds})"
        )

        summary[fk] = {sym: final_weights.get(sym, 0.0) for sym in fold_syms}

    return summary


# ─────────────────────────────────────────────────────────────────────
# Convenience: default baseline from DEFAULTS
# ─────────────────────────────────────────────────────────────────────


def default_baseline_per_symbol() -> float:
    """Return the default baseline risk weight per symbol from DEFAULTS.

    The default ``default_risk_per_pair`` (0.01 = 1% per pair) matches
    the historical ``STATIC["asset_risk_weights"]`` baseline.
    """
    return DEFAULTS["default_risk_per_pair"]


def extract_final_fold_weights(
    risk_summary: dict,
    eligible_symbols: list,
    default_weight: float,
) -> dict:
    """Extract the LAST fold's per-symbol weights as a complete mapping.

    The last fold's weights use the most prior data and represent
    the most informed meta-allocation decision. They are the
    canonical carry-over to the holdout (B0.4 contract: holdout
    must use the same per-symbol risk weights as the WFA).

    This helper is the bridge between ``apply_pf_weighted_risk_allocation``'s
    per-fold summary and the ``Candidate.risk_weights`` attribute that
    ``commit_to_holdout`` reads.

    Parameters
    ----------
    risk_summary : dict
        Output of ``apply_pf_weighted_risk_allocation``:
        ``{fold_key: {sym: final_weight}}``. May be empty (no folds).
    eligible_symbols : list
        All eligible symbols. The returned dict is keyed by these
        symbols, with missing entries filled by ``default_weight``.
    default_weight : float
        Fallback weight for symbols not present in the last fold
        (e.g., they did not trade in the final fold). Typically
        ``DEFAULTS["default_risk_per_pair"]`` (0.01 = 1% per pair).

    Returns
    -------
    dict
        ``{symbol: weight}`` for every symbol in ``eligible_symbols``.
        Empty dict if ``risk_summary`` is empty (no folds run).
    """
    if not risk_summary:
        return {}
    # Use the LAST fold (most recent, most informed). max() works
    # because fold keys are sortable period strings (e.g. "2024-Q1"
    # or "2024-03") -- the orchestrator's ``sorted_fold_keys`` uses
    # string sort which is correct for these formats.
    last_fold_key = max(risk_summary.keys())
    last_fold_weights = risk_summary[last_fold_key]
    return {
        sym: float(last_fold_weights.get(sym, default_weight))
        for sym in eligible_symbols
    }


def apply_pf_weighted_risk_allocation_is(
    oos_trades: list[dict],
    is_trades_per_fold: dict[str, list[dict]],
    halflife_folds: float,
    clamp_floor: float,
    clamp_ceiling: float,
    min_trades: int,
    baseline_per_symbol: float,
    n_total_symbols: int,
    log_prefix: str = "PF Weight (IS)",
) -> dict[str, dict[str, float]]:
    """IS-based PF-weighted risk allocation (Phase LOW-1).

    Decouples the meta-allocator from the strategy selector by using
    IN-SAMPLE trades for past-fold PF computation, instead of OOS
    trades (which the strategy selector has already used for SPA).

    For each unique OOS fold K (grouped by ``oos_trades[i].fold_key``):
      1. Build ``past_is_folds_data`` from IS folds BEFORE K
         (chronological order, current fold EXCLUDED to avoid data
         leak: the IS trades from fold K were generated by running
         the engine on the fold K training data with the winning
         Optuna params, so they're the "live" prediction for that
         fold, not the meta-input).
      2. Compute decay-weighted PF (pnl/loss) from those IS folds.
      3. Compute clamped factor per symbol via ``_compute_clamped_factor``.
      4. Rescale to preserve ``target_total = baseline * (n_active/n_total)``.
      5. Mutate each OOS trade's ``risk_weight`` in current fold.

    Parameters
    ----------
    oos_trades : list of dict
        OOS trades to weight. Each must have ``symbol``, ``r_net``,
        ``fold_key``, and ``risk_weight``. ``fold_key`` and ``atr_inv``
        are consumed (popped) after application; trades without
        ``fold_key`` are left untouched.
    is_trades_per_fold : dict
        ``{fold_key: [trade_dicts]}`` -- IN-SAMPLE trades per fold,
        as generated by ``run_wfa_per_symbol`` (Phase 2.5 IS-trade
        refactor). Each trade dict must have ``symbol`` and ``r_net``.
    halflife_folds : float
        Decay halflife in folds. ``decay_weight = 0.5^(n_back / halflife)``.
    clamp_floor, clamp_ceiling : float
        Weight factor bounds (e.g., 0.5 and 1.5).
    min_trades : int
        Minimum past IS trades required to apply a non-neutral factor.
    baseline_per_symbol : float
        Baseline weight per symbol (e.g., 0.01 = 1% per pair).
    n_total_symbols : int
        Total symbols in the universe.
    log_prefix : str, optional
        Prefix for the per-fold log line. Default "PF Weight (IS)".

    Returns
    -------
    dict
        ``{fold_key: {sym: final_weight}}`` summary. Empty dict when
        no OOS trades have ``fold_key`` (commit-path safety).

    Note
    ----
    This function is the canonical IS-decoupled risk-allocation path
    (replacing the OOS-based :func:`apply_pf_weighted_risk_allocation`).
    The OOS-based version is preserved for backward compat.

    Mutates ``oos_trades`` in place: sets ``risk_weight`` per trade
    and pops ``fold_key`` / ``atr_inv``. Callers should make a copy
    if they need the original fold metadata.
    """
    # Group OOS trades by fold_key (skip trades without fold_key).
    oos_by_fold: dict[str, list[int]] = defaultdict(list)
    for t_idx, t in enumerate(oos_trades):
        fk = t.get("fold_key")
        if fk is not None:
            oos_by_fold[fk].append(t_idx)

    if not oos_by_fold:
        return {}

    # Total baseline = baseline * n_total (6 default pairs * 0.01 historically).
    original_total = baseline_per_symbol * n_total_symbols

    summary: dict[str, dict[str, float]] = {}
    sorted_fold_keys = sorted(oos_by_fold.keys())

    for fold_idx, fk in enumerate(sorted_fold_keys):
        t_idxs = oos_by_fold[fk]

        # Symbols that had any OOS trade in this fold.
        fold_syms: set[str] = {oos_trades[tidx]["symbol"] for tidx in t_idxs}

        # Per-fold budget scale: how much of the original total is active.
        n_active = len(fold_syms)
        budget_scale = (n_active / n_total_symbols) if n_total_symbols > 0 else 1.0
        target_total_for_fold = original_total * budget_scale

        # === Past IS folds only (current fold EXCLUDED) ===
        # Chronological: folds before current. Current fold's IS trades
        # are EXCLUDED to avoid data leak -- those IS trades are the
        # "live prediction" for fold K generated by running the engine
        # on fold K training data with the winning Optuna params;
        # including them in past-fold PF would circularly influence
        # the weight assigned to fold K's trades.
        past_is_folds: list[tuple[int, list[dict]]] = []
        for n_back, past_fk in enumerate(sorted_fold_keys[:fold_idx], start=1):
            past_is_trades = is_trades_per_fold.get(past_fk, [])
            if past_is_trades:
                past_is_folds.append((n_back, past_is_trades))

        decay_pnl_loss = _compute_decay_weighted_pnl_loss(
            past_is_folds, halflife_folds
        )

        # === Clamped weight factors ===
        weight_factors: dict[str, float] = {}
        for sym in fold_syms:
            if sym not in decay_pnl_loss:
                # No past IS data for this symbol (e.g., first fold
                # or symbol didn't trade in past IS folds) -> neutral.
                weight_factors[sym] = 1.0
                continue
            w_pnl, w_loss, n_past = decay_pnl_loss[sym]
            weight_factors[sym] = _compute_clamped_factor(
                weighted_pnl=w_pnl,
                weighted_loss=w_loss,
                n_past_trades=n_past,
                min_trades=min_trades,
                clamp_floor=clamp_floor,
                clamp_ceiling=clamp_ceiling,
            )

        # === Rescale to preserve total (X1 scheme) ===
        final_weights = _rescale_factors_to_total(
            factors=weight_factors,
            baseline_per_symbol=baseline_per_symbol,
            target_total=target_total_for_fold,
        )

        # === Apply to OOS trades in this fold ===
        for tidx in t_idxs:
            t = oos_trades[tidx]
            sym = t["symbol"]
            if sym in final_weights:
                t["risk_weight"] = final_weights[sym]
            else:
                t["risk_weight"] = 0.0
            # Pop fold metadata so trades don't carry WFA-internal tags.
            t.pop("atr_inv", None)
            t.pop("fold_key", None)

        # === Per-fold log summary ===
        n_past_folds_used = sum(1 for _, trades in past_is_folds if trades)
        log_lines = [
            f"{sym}={final_weights.get(sym, 0.0):.4f} "
            f"(factor={weight_factors.get(sym, 1.0):.2f}x)"
            for sym in sorted(fold_syms)
        ]
        log.info(
            f"[{log_prefix}] Fold {fk}: {', '.join(log_lines)} "
            f"(budget_scale={budget_scale:.2f}, halflife={halflife_folds}, "
            f"n_past_is_folds={n_past_folds_used})"
        )

        summary[fk] = {sym: final_weights.get(sym, 0.0) for sym in fold_syms}

    return summary


__all__ = [
    "apply_pf_weighted_risk_allocation",
    "apply_pf_weighted_risk_allocation_is",
    "default_baseline_per_symbol",
    "extract_final_fold_weights",
]
