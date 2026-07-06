"""
Portfolio SPA -- Superior Predictive Ability test with circular permutation.

Extracted from Hans_Quant_Systems.py:
  - portfolio_spa (lines 1489-1844)
"""

import numpy as np
from numpy import ndarray
import pandas as pd

from quant_lib.core._config import STATIC, DEFAULTS
from quant_lib.core._logging import log, console
from quant_lib.core._portfolio import _trade_key, simulate_full_portfolio
from quant_lib.core._engine import simulate_trailing_stop_trade
from quant_lib.core._metrics import _stationary_block_bootstrap_resample

_AssetDataDict = dict[str, pd.DataFrame]


def _hansen_spa_p_value(
    observed_r_nets: ndarray | None,
    trial_r_nets: list[ndarray] | None,
    n_iters: int,
    rng_hansen,
    p_value_naive: float,
) -> tuple[float, dict]:
    """Hansen (2005) literal SPA null -- Superior Predictive Ability.

    Tests ``H0: best strategy is no better than the zero-mean benchmark``
    across the K Optuna IS trials, with the multiple-testing correction
    (cross-strategy max-statistic) that the legacy circular-permutation
    test lacked.

    Loss differential (benchmark = zero-mean null => L(benchmark) = 0,
    higher d_k = larger loss for strategy k):

        d_k = -r_net_k            (per-trade loss of strategy k)

    Per bootstrap iteration b, for EACH trial k Politis-Romano stationary
    block-resample d_k and compute Hansen Eq.6-style bootstrap statistic:

        T_raw^k_b = sqrt(n_k) * d_bar^k_b / std(d_k, ddof=1)

    Hansen Eq.7 recenter / nuisance-parameter discarding:

        A_bar_k       = mean_b(T_raw^k_b)
        A_bar_k_trunc = A_bar_k * 1{ A_bar_k >= 0 }
        T_acc^k_b     = T_raw^k_b - A_bar_k_trunc

    Cross-strategy max-statistic (Eq.8) -- the multiple-testing gate:

        T_null_max[b] = max_k T_acc^k_b

    Observed statistic over the OOS winner's r_nets:

        T_obs = sqrt(N) * mean(-r_obs) / std(-r_obs, ddof=1)

    Phipson-Smyth add-one:

        p_hansen = (1 + #{b: T_null_max[b] >= T_obs}) / (n_iters + 1)

    NaN-safe: if ``trial_r_nets`` is None/empty after filtering, or
    ``observed_r_nets`` has N<2, or any trial has std(d_k)<=0 (constant
    loss), return ``p_hansen = p_value_naive`` with ``fallback=True`` --
    the Hansen test cannot be computed, so we degrade to the legacy p. This
    is what keeps every legacy ``trial_r_nets=None`` caller byte-identical
    to the pre-Hansen path.

    Parameters
    ----------
    observed_r_nets, trial_r_nets : the OOS winner's r_nets and the K trials' IS
        PnL arrays (None sentinels filtered out by the caller are okay here;
        this helper filters again). Both are R-multiple or PnL arrays.
    n_iters : bootstrap iterations B (== the legacy permutation count).
    rng_hansen : a FRESH np.random.Generator (default_rng(rng_seed + 1));
        MUST NOT be ``rng_spa`` -- the legacy spy assert counts the
        simulate_* calls driven by ``rng_spa`` and reusing it would shift
        the legacy anchor distribution and break the spy under
        ``trial_r_nets=None``.
    p_value_naive : the legacy circular-permutation p_value; returned
        unchanged on any fallback path so 3-tuple callers are unaffected.

    Returns
    -------
    (p_hansen, stats) : p_hansen is the Hansen p-value (or p_value_naive
        on fallback); stats carries p_naive/p_hansen/T_obs/n_trials/
        n_obs/block_length/recenter_policy/fallback.
    """
    # Filter to real arrays.
    trials = [np.asarray(a, dtype=float).ravel() for a in (trial_r_nets or []) if a is not None]
    if len(trials) == 0:
        return p_value_naive, {
            "p_naive": p_value_naive, "p_hansen": p_value_naive,
            "T_obs": float("nan"), "n_trials": 0, "n_obs": 0,
            "block_length": 0, "recenter_policy": "hansen_literal",
            "fallback": True, "fallback_reason": "no_trial_r_nets",
        }
    r_obs = None
    if observed_r_nets is not None:
        r_obs = np.asarray(observed_r_nets, dtype=float).ravel()
    N = r_obs.shape[0] if r_obs is not None else 0
    if N < 2:
        return p_value_naive, {
            "p_naive": p_value_naive, "p_hansen": p_value_naive,
            "T_obs": float("nan"), "n_trials": len(trials), "n_obs": N,
            "block_length": 0, "recenter_policy": "hansen_literal",
            "fallback": True, "fallback_reason": "observed_n_less_than_2",
        }

    per_trial_std = []
    for d_neg_k in trials:
        # d_k = -loss_k benchmark null => d_k = -r_net_k. To gauge std==0:
        s = float(np.std(d_neg_k, ddof=1)) if len(d_neg_k) > 1 else 0.0
        per_trial_std.append(s)
    if any(s <= 0.0 for s in per_trial_std):
        return p_value_naive, {
            "p_naive": p_value_naive, "p_hansen": p_value_naive,
            "T_obs": float("nan"), "n_trials": len(trials), "n_obs": N,
            "block_length": 0, "recenter_policy": "hansen_literal",
            "fallback": True, "fallback_reason": "trial_std_zero",
        }

    # Build per-trial precomputed constants: n_k, block length p_k,
    # ddof-1 std of d_k. d_k = -trial_k (loss = -r_net).
    trial_consts = []
    for d_neg_k, s in zip(trials, per_trial_std):
        n_k = d_neg_k.shape[0]
        # Politis-Romano default expected block length p ~ n_k^(1/3),
        # honoring a STATIC override ceiling if set.
        if STATIC["spa_hansen_block_length_override"] > 0:
            p_k = int(STATIC["spa_hansen_block_length_override"])
        else:
            p_k = max(1, int(round(n_k ** (1.0 / 3.0))))
        # d_bar under the bootstrap will be drawn via resampling d_neg_k;
        # std is the FULL-series std (ddof=1) per Hansen Eq.6.
        trial_consts.append((d_neg_k, n_k, p_k, s))

    # Observed statistic over the OOS winner (loss = -r_obs).
    r_obs_std = float(np.std(r_obs, ddof=1))
    if r_obs_std <= 0.0:
        return p_value_naive, {
            "p_naive": p_value_naive, "p_hansen": p_value_naive,
            "T_obs": float("nan"), "n_trials": len(trials), "n_obs": N,
            "block_length": 0, "recenter_policy": "hansen_literal",
            "fallback": True, "fallback_reason": "observed_std_zero",
        }
    T_obs = float(np.sqrt(N) * np.mean(-r_obs) / r_obs_std)

    # Bootstrap: T_raw^k_b for k in trials, b in range(n_iters).
    # Shape (K, B). Use the Phase 1 stationary resample (kept numpy-only
    # -- the spy invariant: Hansen emits 0 simulate_* calls).
    K = len(trials)
    T_raw = np.empty((K, n_iters), dtype=np.float64)
    for k, (d_neg_k, n_k, p_k, s) in enumerate(trial_consts):
        sqrt_nk = np.sqrt(n_k)
        for b in range(n_iters):
            resampled = _stationary_block_bootstrap_resample(
                d_neg_k, rng_hansen, p=p_k, n_out=n_k
            )
            d_bar_b = resampled.mean()
            T_raw[k, b] = sqrt_nk * d_bar_b / s

    # Hansen Eq.7 recenter / discarding.
    A_bar = T_raw.mean(axis=1)              # (K,)
    A_bar_trunc = np.where(A_bar >= 0.0, A_bar, 0.0)
    T_acc = T_raw - A_bar_trunc[:, None]    # (K, B)

    # Eq.8 cross-strategy max-statistic.
    T_null_max = T_acc.max(axis=0)          # (B,)

    # Phipson-Smyth add-one.
    n_exceed = int(np.sum(T_null_max >= T_obs))
    p_hansen = (n_exceed + 1) / (n_iters + 1)

    # Witnessed block length (only meaningful when not overridden).
    p_witness = trial_consts[0][2]
    return p_hansen, {
        "p_naive": p_value_naive, "p_hansen": p_hansen,
        "T_obs": T_obs, "n_trials": K, "n_obs": N,
        "block_length": p_witness, "recenter_policy": "hansen_literal",
        "fallback": False,
    }


def _spa_finalize_return(
    equity: float,
    random_equities: ndarray,
    p_value: float,
    return_statistics: bool,
    stats: dict | None = None,
) -> tuple | tuple[float, ndarray, float] | tuple[float, ndarray, float, dict]:
    """Centralize the SPA return-shape contract.

    With ``return_statistics=False`` (legacy, the default) returns the strict
    3-tuple ``(equity, random_equities, p_value)`` every legacy caller pins
    (test_reproducibility.py, tools/stats.spa_test, the spy test). With
    ``return_statistics=True`` returns a 4-tuple whose 4th element is
    ``stats or {}`` (Hansen statistics; populated in Phase 5).

    Routing every return path (the 6 early-return guards + the final p-value
    return) through this helper guarantees ``return_statistics`` is honored
    uniformly with no copy-paste drift.

    Parameters
    ----------
    equity : float
        Observed (or fallback) final portfolio equity to be returned as
        the first element of the result tuple.
    random_equities : ndarray
        Per-iteration permuted portfolio equities (``len == n_iters``)
        used for the p-value computation; returned as the second tuple
        element.
    p_value : float
        SPA p-value (legacy circular-permutation, or NaN/1.0 sentinel
        on degenerate paths); returned as the third tuple element.
    return_statistics : bool
        When True, the 4-tuple shape is returned with the Hansen
        ``stats`` dict appended; when False, the strict legacy
        3-tuple is returned.
    stats : dict or None, optional
        Hansen statistics payload (populated in Phase 5); only emitted
        in the 4-tuple path. Defaults to None, which is normalized to
        ``{}`` when ``return_statistics`` is True.

    Returns
    -------
    tuple
        Either ``(equity, random_equities, p_value)`` when
        ``return_statistics`` is False, or
        ``(equity, random_equities, p_value, stats or {})`` when True.
    """
    if return_statistics:
        return equity, random_equities, p_value, stats if stats is not None else {}
    return equity, random_equities, p_value


def portfolio_spa(
    observed_trades: list[dict],
    asset_data: _AssetDataDict,
    daily_close_matrix: dict[str, dict],
    end_date: str,
    daily_hl_matrix: dict[str, dict] | None = None,
    n_iters: int = STATIC["spa_n_iters"],
    initial_capital: float = 1000.0,
    leverage: float = 3.0,
    mm_pct: float = 0.01,
    position_limit: int = 4,
    cb_hard_cooldown_hours: int = 24,
    fixed_cb_threshold: float = 0.15,
    rng_seed: int = 42,
    verbose: bool = False,
    liquidation_fee_pct: float = 0.005,
    fee_taker: float = 0.05,
    # NOTE (0.2.2): Was hardcoded 2.5 (stale, predates 0.2.0 default change).
    # Now mirrors DEFAULTS["stress_test_multiplier"] so direct spa_test() callers
    # get the same cost model as the WFA path (which passes DEFAULTS through).
    stress_mult: float = DEFAULTS["stress_test_multiplier"],
    weekend_penalty: float = DEFAULTS["weekend_liquidity_penalty"],
    asset_risk_weights: dict[str, float] | None = None,
    # Hansen-literal SPA (claim #3 Blocker A) — opt-in additions. All
    # three default to legacy behavior so every existing 3-tuple caller
    # (test_reproducibility.py, tools/stats.spa_test, the spy test) stays
    # byte-identical. Phase 5 fills the ``recenter_policy="hansen_literal"``
    # branch; Phase 3 only wires the return-shape contract.
    trial_r_nets: list[ndarray] | None = None,
    recenter_policy: str = "legacy",
    return_statistics: bool = False,
) -> tuple[float, ndarray, float] | tuple[float, ndarray, float, dict]:
    """Portfolio-level SPA (Superior Predictive Ability) test.

    Tests whether the observed strategy edge is genuine or random,
    using time-anchored circular permutation across all assets.

    Parameters
    ----------
    observed_trades : list of dict
        OOS trade records (formerly the ``pnl_array`` inputs) used as
        the observed sample; each dict must carry ``entry_time``,
        ``exit_time``, ``symbol``, ``trade_dir``, ``sl_mult``,
        ``trail_atr``, ``risk_weight``, ``r_net``. The Hansen-literal
        path operates numpy-only over ``r_net`` arrays.
    asset_data : dict
        Per-symbol OHLCV+ATR+funding DataFrames keyed by symbol;
        consumed by ``simulate_trailing_stop_trade`` for each
        permuted entry.
    daily_close_matrix : dict
        ``{sym: {date: close}}`` mapping used to pre-compute the
        correlation cache shared with ``simulate_full_portfolio``.
    end_date : str
        ISO date bounding the SPA anchor space; a ``>7d`` gap between
        ``end_date`` and the earliest asset data end triggers a warning.
    daily_hl_matrix : dict or None, optional
        ``{sym: {date: (high, low)}}`` matrix passed through to
        ``simulate_full_portfolio``; ``None`` is allowed.
    n_iters : int
        Number of bootstrap/permutation iterations ``B``; also the
        Hansen bootstrap iteration count when ``recenter_policy ==
        "hansen_literal"``.
    initial_capital : float
        Starting equity for both the observed and permuted portfolios;
        returned unchanged for empty-portfolio degenerate paths.
    leverage : float
        Portfolio leverage multiplier forwarded to the simulation.
    mm_pct : float
        Maintenance-margin percentage forwarded to ``simulate_full_portfolio``.
    position_limit : int
        Maximum number of concurrent positions in the portfolio simulator.
    cb_hard_cooldown_hours : int
        Circuit-breaker hard-cooldown window in hours; passed to
        ``simulate_full_portfolio``.
    fixed_cb_threshold : float
        Fixed circuit-breaker drawdown threshold; passed to
        ``simulate_full_portfolio``.
    rng_seed : int
        Seed for ``np.random.default_rng``; ``rng_seed + 1`` is used
        as a fresh generator for the Hansen-literal path so the legacy
        ``rng_spa`` distribution (and the spy invariant) is preserved.
    verbose : bool
        When True, prints ``SPA progress`` lines every ~10% of iterations.
    liquidation_fee_pct : float
        Liquidation fee percentage forwarded to ``simulate_full_portfolio``.
    fee_taker : float
        Taker fee rate forwarded to ``simulate_trailing_stop_trade``.
    stress_mult : float
        Stress-test cost multiplier; mirrors
        ``DEFAULTS["stress_test_multiplier"]`` so direct SPA callers
        share the WFA cost model.
    weekend_penalty : float
        Weekend liquidity penalty applied in ``simulate_trailing_stop_trade``;
        mirrors ``DEFAULTS["weekend_liquidity_penalty"]``.
    asset_risk_weights : dict or None, optional
        ``{sym: risk_weight}`` overrides forwarded to
        ``simulate_full_portfolio``; ``None`` disables per-asset CB
        weighting.
    trial_r_nets : list of ndarray or None, optional
        K Optuna IS-trial PnL arrays (None entries are filtered out)
        consumed by the Hansen-literal path; the cross-strategy
        max-statistic (Eq.8) is computed across these. ``None`` keeps
        the legacy 3-tuple contract byte-identical.
    recenter_policy : str
        ``"legacy"`` (default) for the original circular-permutation
        SPA, or ``"hansen_literal"`` to opt into the Politis-Romano
        stationary block bootstrap + Hansen Eq.7 recenter/discarding
        + Eq.8 cross-strategy max-statistic null. Only active when
        combined with non-empty ``trial_r_nets`` and
        ``return_statistics=True``.
    return_statistics : bool
        When False (default) returns the strict 3-tuple
        ``(equity, random_equities, p_value)`` every legacy caller
        pins; when True returns the 4-tuple with a Hansen
        statistics dict as the 4th element.

    Returns
    -------
    tuple
        ``(equity, random_equities, p_value)`` when ``return_statistics``
        is False, or ``(equity, random_equities, p_value, stats)`` when
        ``return_statistics`` is True. ``stats`` is the Hansen
        statistics dict on the opt-in path, or ``{}`` otherwise.
        ``p_value`` is the legacy circular-permutation p (preserved
        by the 3-tuple contract); the Hansen-corrected p lives in
        ``stats["p_hansen"]`` when populated.

    Notes
    -----
    Internally, this function constructs a per-call correlation cache
    (``_shared_corr_cache``) that is passed to every
    ``simulate_full_portfolio`` invocation. **Constraints:**

    - DO share within a single ``portfolio_spa`` call (the cache lives
      in the local scope; SPA iterations reuse correlation matrices).
    - DO NOT share across independent backtests — the cache is keyed by
      date and will leak stale entries across backtest ranges, producing
      wrong correlations. Each top-level backtest run gets a fresh cache.
    - DO NOT mutate the cache externally during a single backtest run.

    See ``core/_portfolio.py:simulate_full_portfolio`` for the consumer-
    side docstring of the same constraint.
    """
    aw = asset_risk_weights  # None is allowed; portfolio sim will skip per-asset CB

    if not observed_trades:
        return _spa_finalize_return(initial_capital, np.zeros(n_iters), 1.0, return_statistics)

    # Defensive filter -- trades MUST have sl_mult
    n_no_sl = sum(1 for t in observed_trades if t.get("sl_mult") is None)
    if n_no_sl > 0:
        observed_trades = [t for t in observed_trades if t.get("sl_mult") is not None]
        if not observed_trades:
            return _spa_finalize_return(initial_capital, np.zeros(n_iters), 1.0, return_statistics)
        log.warning(
            f"SPA: {n_no_sl}/{n_no_sl + len(observed_trades)} trades "
            f"missing sl_mult -- excluded from permutation."
        )

    rng_spa = np.random.default_rng(rng_seed)

    # Pre-compute correlation data ONCE for all 500+ SPA iterations
    _precomputed_sym_list = sorted(daily_close_matrix.keys())
    _precomputed_daily_returns = None
    if len(_precomputed_sym_list) >= 2:
        _ret_series_list = []
        for _sym in _precomputed_sym_list:
            _s = pd.Series(daily_close_matrix[_sym]).sort_index().pct_change().dropna()
            _ret_series_list.append(_s)
        _ret_df = pd.concat(_ret_series_list, axis=1, keys=_precomputed_sym_list).dropna()
        if len(_ret_df) > 30:
            _precomputed_daily_returns = _ret_df
        else:
            _precomputed_daily_returns = None
            _precomputed_sym_list = []
    else:
        _precomputed_sym_list = []

    # Cross-iteration correlation cache shared with simulate_full_portfolio.
    # Same convention as in _portfolio.py (dict | None). Annotation
    # needed because mypy can't infer dict literal in this scope.
    _shared_corr_cache: dict | None = {}

    # 1. Simulate observed trades for baseline equity
    observed_final_equity = simulate_full_portfolio(
        observed_trades,
        initial_capital,
        leverage,
        mm_pct,
        position_limit,
        cb_hard_cooldown_hours,
        fixed_cb_threshold,
        daily_close_matrix,
        aw,
        end_date=end_date,
        liquidation_fee_pct=liquidation_fee_pct,
        daily_hl_matrix=daily_hl_matrix,
        _precomputed_daily_returns=_precomputed_daily_returns,
        _precomputed_sym_list=_precomputed_sym_list,
        _shared_corr_cache=_shared_corr_cache,
    )[0]

    # Temporal anchoring: preserve cross-asset co-occurrence correlation
    first_entry = min(t["entry_time"] for t in observed_trades)
    relative_offsets = {}
    durations_h = {}
    trade_keys = []

    for t in observed_trades:
        tid = _trade_key(t)
        trade_keys.append(tid)
        relative_offsets[tid] = (t["entry_time"] - first_entry).total_seconds() / 3600.0
        dur = int((t["exit_time"] - t["entry_time"]).total_seconds() / 3600.0)
        durations_h[tid] = max(1, dur)

    # Valid time range for anchor
    global_start = max(v["time"].iloc[0] for v in asset_data.values())

    _data_end_date = pd.Timestamp(end_date)
    _data_lasts = {sym: v["time"].iloc[-1] for sym, v in asset_data.items()}
    _min_asset_end = min(_data_lasts.values())
    global_end = max(_data_end_date, _min_asset_end)
    _gap_days = (global_end - _min_asset_end).total_seconds() / 86400
    if _gap_days > 7:
        log.warning(
            f"SPA data gap: asset min end = {_min_asset_end.date()}, "
            f"END_DATE = {_data_end_date.date()} "
            f"(gap={_gap_days:.0f}d). "
            f"Some SPA iterations may have fewer trades for short-data assets."
        )
    global_start_np = np.datetime64(global_start)
    total_hours = (global_end - global_start) / np.timedelta64(1, "h")

    span_hours = max(relative_offsets.values()) + max(durations_h.values())
    max_anchor = max(0, total_hours - span_hours)

    random_equities = np.zeros(n_iters) if n_iters > 0 else np.array([])

    # Defensive guard (Phase 3.5 B1): if observed_final_equity is NaN
    # (e.g. from numerical issues in simulate_full_portfolio), the SPA
    # p-value comparison `random_equities >= NaN` would always be False,
    # giving n_exceed=0 -> p_value=1/(N+1) (misleadingly "significant").
    # Return NaN p-value explicitly so callers can detect the issue.
    if np.isnan(observed_final_equity):
        log.warning(
            "SPA: observed_final_equity is NaN (numerical issue in "
            "portfolio simulation). Returning NaN p-value."
        )
        return _spa_finalize_return(observed_final_equity, random_equities, float("nan"), return_statistics)

    anchor_ratio = span_hours / total_hours * 100 if total_hours > 0 else 0
    log.info(
        f"SPA anchor space: total_hours={total_hours:.0f}h, "
        f"span_hours={span_hours:.0f}h, max_anchor={max_anchor:.0f}h, "
        f"anchor_ratio={anchor_ratio:.1f}%"
    )

    # Degenerate anchor guard
    if total_hours > 0 and span_hours >= total_hours * 0.8:
        log.warning(
            f"SPA DEGENERATE: anchor_ratio={anchor_ratio:.0f}% "
            f"(span={span_hours:.0f}h / total={total_hours:.0f}h >= 80%). "
            f"Circular permutation creates near-identical null -> "
            f"p-value UNRELIABLE. Returning NaN."
        )
        return _spa_finalize_return(observed_final_equity, random_equities, float("nan"), return_statistics)

    times_hours_map = {
        sym: (asset_data[sym]["time"].values - global_start_np) / np.timedelta64(1, "h")
        for sym in asset_data
    }

    if total_hours <= 0:
        log.error("SPA: total_hours <= 0, no valid time range for permutation.")
        return _spa_finalize_return(observed_final_equity, random_equities, float("nan"), return_statistics)

    for it in range(n_iters):
        anchor_offset = rng_spa.uniform(0, total_hours)
        random_trades = []

        for i, t in enumerate(observed_trades):
            sym = t["symbol"]
            df_sym = asset_data[sym]
            tid = trade_keys[i]

            target_entry_hour = (anchor_offset + relative_offsets[tid]) % total_hours
            dur_h = durations_h[tid]

            idx = int(np.searchsorted(times_hours_map[sym], target_entry_hour))
            max_valid_idx = len(df_sym) - dur_h - 1
            if max_valid_idx < 0:
                continue
            if idx > max_valid_idx:
                idx = idx % (max_valid_idx + 1)

            sl_mult_val = t.get("sl_mult", 1.5)
            trail_atr_val = t.get("trail_atr", 3.0)
            direction = int(t.get("trade_dir", 1))

            rand_draw = float(rng_spa.random())

            exit_idx, exit_price, net_r, trend_mult = simulate_trailing_stop_trade(
                df_sym["high"].values,
                df_sym["low"].values,
                df_sym["close"].values,
                df_sym["atr"].values,
                df_sym["funding_rate"].values,
                df_sym["is_funding_hour"].values,
                df_sym["is_weekend"].values,
                df_sym["macro_trend"].values,
                idx,
                direction,
                sl_mult_val,
                trail_atr_val,
                DEFAULTS["bailout_bars"],
                fee_taker,
                weekend_penalty,
                stress_mult,
                rand_draw,
                DEFAULTS["trend_aligned_risk_mult"],
                DEFAULTS["trend_counter_risk_mult"],
            )

            if exit_idx < 0:
                continue

            entry_price = df_sym["close"].iloc[idx]
            atr_entry = df_sym["atr"].iloc[idx]
            sl_dist = atr_entry * sl_mult_val
            sl_pct = sl_dist / entry_price

            random_trades.append({
                "entry_time": df_sym["time"].iloc[idx],
                "exit_time": df_sym["time"].iloc[exit_idx],
                "symbol": sym,
                "trade_dir": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "sl_pct": sl_pct,
                "r_net": net_r,
                "risk_weight": t.get(
                    "risk_weight",
                    aw.get(sym, DEFAULTS["default_risk_per_pair"])
                    if aw
                    else DEFAULTS["default_risk_per_pair"],
                ),
                "trend_risk_mult": trend_mult,
            })

        if random_trades:
            eq, _, _, _ = simulate_full_portfolio(
                random_trades,
                initial_capital,
                leverage,
                mm_pct,
                position_limit,
                cb_hard_cooldown_hours,
                fixed_cb_threshold,
                daily_close_matrix,
                aw,
                end_date=end_date,
                liquidation_fee_pct=liquidation_fee_pct,
                daily_hl_matrix=daily_hl_matrix,
                _precomputed_daily_returns=_precomputed_daily_returns,
                _precomputed_sym_list=_precomputed_sym_list,
                _shared_corr_cache=_shared_corr_cache,
            )
            random_equities[it] = eq
        else:
            random_equities[it] = initial_capital

        if verbose and (it + 1) % max(1, n_iters // 10) == 0:
            pct = (it + 1) / n_iters * 100
            console.print(f"   SPA progress: {pct:.0f}% ({it+1}/{n_iters})")

    # Two coexisting SPA nulls (Phase 7) -- caller selects via kwargs:
    #   * LEGACY -- the default. Uniform time-anchored circular permutation
    #     of the observed trades (preserves cross-asset co-occurrence).
    #     This is what ``p_value`` below computes (Phipson & Smyth 2010
    #     add-one over ``n_exceed`` / ``n_iters + 1``).
    #   * HANSEN-LITERAL -- opt-in via ``recenter_policy="hansen_literal"``
    #     + ``trial_r_nets`` + ``return_statistics=True``. Politis-Romano
    #     stationary block bootstrap over the per-trial IS loss-
    #     differentials ``d_k = -r_net_k`` + Hansen (2005) Eq.7 recenter/
    #     discarding + Eq.8 cross-strategy max-statistic + Phipson &
    #     Smyth add-one (Hansen 2005). See ``_hansen_spa_p_value`` above.
    #     The Hansen path is NUMPY-ONLY on ``pnl_array``s (zero
    #     ``simulate_*`` calls) so the legacy spy invariant
    #     ``len(recorded_idx) == 2*n_iters`` holds by construction on
    #     BOTH paths (see ``tests/test_spa_validation.py``).
    # The incorrect "Davé 2008" label was removed; correct citation is
    # Phipson & Smyth (2010) for add-one correction in permutation tests.
    # Phase 3 (v0.4.1): detect when ALL SPA iterations failed to
    # produce trades (random_equities all equal initial_capital). In
    # that case, n_exceed would be 0 → p_value = 1/(N+1) would
    # misleadingly suggest significance. Return p_value=1.0 (cannot
    # reject null) instead, with a warning.
    if np.all(random_equities == initial_capital):
        log.warning(
            f"SPA: all {n_iters} iterations produced empty/zero equity "
            f"(random_equities all == initial_capital). Returning "
            f"p_value=1.0 (cannot reject null). Check upstream "
            f"simulate_trailing_stop_trade for trade generation."
        )
        return _spa_finalize_return(observed_final_equity, random_equities, 1.0, return_statistics)

    n_exceed = int(np.sum(random_equities >= observed_final_equity))
    p_value = (n_exceed + 1) / (n_iters + 1)

    # Hansen-literal SPA null (claim #3 Blocker A). Active only when the
    # caller opts in via ``recenter_policy="hansen_literal"`` AND supplies
    # ``trial_r_nets`` (the K IS PnL arrays) AND ``return_statistics=True``.
    # When inactive (the legacy default) ``p_value`` IS the legacy circular-
    # permutation p -- nobody downstream sees any change. When active, the
    # 3-tuple p_value stays the legacy p (legacy 3-tuple contract intact);
    # the Hansen p lives in ``hansen_stats["p_hansen"]`` for the 4-tuple
    # path (candidate.py picks it up). On any NaN-safe fallback the Hansen
    # p degrades to ``p_value`` (== p_naive) so opt-in callers never crash.
    hansen_stats: dict | None = None
    if (
        recenter_policy == "hansen_literal"
        and return_statistics
        and trial_r_nets is not None
        and len(trial_r_nets) > 0
    ):
        observed_r_nets = np.asarray(
            [float(t.get("r_net", float("nan"))) for t in observed_trades],
            dtype=float,
        )
        # FRESH generator — reusing rng_spa would shift the legacy anchor
        # distribution and break the spy test's ``2*n_iters`` invariant
        # under ``trial_r_nets=None`` (the spy asserts exact simulate_*
        # call counts driven by rng_spa).
        rng_hansen = np.random.default_rng(rng_seed + 1)
        _, hansen_stats = _hansen_spa_p_value(
            observed_r_nets, trial_r_nets, n_iters, rng_hansen, p_value
        )

    return _spa_finalize_return(
        observed_final_equity, random_equities, p_value, return_statistics,
        stats=hansen_stats,
    )
