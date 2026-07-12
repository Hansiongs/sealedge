"""
commit_to_holdout: one-shot holdout commit (irreversible).

Breaks the session seal after re-verifying the data hash. Uses frozen
params from the candidate (stability-gated best fold per symbol; see
``best_params.pick_best_params_per_symbol``). No re-optimization on
holdout. Full cost model (slippage, weekend, funding).

Returns metrics only (including holdout PSR). No auto-verdict; the
user reads ``success_criteria_text``. Paper-grade replication uses
explore only and does not call this path by default.
"""

from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import os

import numpy as np
import pandas as pd

from quant_lib.core._engine import fast_trade_loop, STRATEGY_PULLBACK_SNIPER
from quant_lib.core._config import STATIC, DEFAULTS, STRATEGY_FUNDING_RATE_CARRY
from quant_lib.core._metrics import build_daily_matrices
from quant_lib.core._testing import prob_sharpe_ratio, deflated_sharpe_ratio
from quant_lib.core._portfolio import simulate_full_portfolio
from quant_lib.research.exceptions import (
    CommitError,
    SealVerificationFailed,
    HoldoutAlreadyBroken,
)

if TYPE_CHECKING:
    from quant_lib.research.candidate import Candidate
    from quant_lib.research.session import ResearchSession


@dataclass
class CommitResult:
    """Result of a single holdout commit.

    All fields are informational only. No auto-verdict.
    User interprets based on `success_criteria_text` they provided.
    """
    candidate_name: str
    commit_idx: int
    holdout_period: tuple[str, str]
    timestamp: str

    # Equity metrics
    initial_capital: float
    final_equity: float
    equity_pct: float
    cagr_pct: float
    max_dd_pct: float

    # Trade metrics
    n_raw_trades: int
    n_executed_trades: int
    n_rejected: int
    reject_breakdown: dict

    # Statistical metrics
    n_trades: int
    win_rate: float
    avg_r: float
    median_r: float
    std_r: float
    best_r: float
    worst_r: float
    profit_factor: float
    avg_bars_held: float
    sharpe_r: float
    psr: float
    psr_ess: float
    skew: float
    kurtosis: float
    ess: float

    # FDR context
    bonferroni_alpha: float
    fdr_alpha: float

    # By-symbol breakdown
    by_symbol_stats: dict = field(default_factory=dict)

    # Sprint 3 fix 3.6: real daily equity curve from the holdout
    # portfolio simulation. Maps ``pd.Timestamp -> float``. ``None``
    # when no trades executed or when ``h_daily_eq`` was empty (e.g.,
    # all trades rejected). Replaces the synthetic 2-point fake that
    # the commit HTML report used to render (see commit_cmd.py
    # ``_make_chart_provider`` -- previously returned None for the
    # commit path, now uses this real series).
    daily_equity: Optional[dict] = None

    # Trend alignment impact
    with_trend_trades: int = 0
    with_trend_r_total: float = 0.0
    counter_trend_trades: int = 0
    counter_trend_r_total: float = 0.0

    # Regime stats (Phase 4.2): Bull/Bear profit factor & trade count.
    regime_bull_pf: float = float("nan")
    regime_bull_n: int = 0
    regime_bear_pf: float = float("nan")
    regime_bear_n: int = 0

    # Seal
    seal_hash_before: str = ""
    seal_hash_after: str = ""
    seal_broken: bool = False

    # User-provided criteria
    success_criteria_text: str = ""

    # Trade bootstrap (Phase 4.1): circular block bootstrap on trade
    # R-multiples. More appropriate than daily-return bootstrap for
    # strategies with sparse trades. NaN when n_trades < 5.
    trade_bootstrap_worst5_cagr: float = float("nan")
    trade_bootstrap_worst95_dd: float = float("nan")
    trade_bootstrap_worst5_dd: float = float("nan")
    trade_bootstrap_worst1_dd: float = float("nan")
    trade_bootstrap_block: int = 0

    # Multiple-testing adjustment (Phase 2.1: Bailey & López de Prado 2014).
    # The deflated PSR adjusts the single-trial PSR for the family of
    # Optuna trials that produced the winning params. A deflated_psr
    # > 0.95 means the strategy's edge is likely real, not just the best
    # of N trials under the null. NaN when n_trials < 2.
    deflated_psr: float = float("nan")
    n_trials_in_deflated: int = 0


def commit_to_holdout(
    candidate: "Candidate",
    success_criteria_text: str = "",
    verbose: bool = True,
) -> CommitResult:
    """Commit a candidate's frozen params to the holdout (single-shot).

    BLACK-BOX: Once called, the holdout seal is irreversibly broken.
    No re-optimization. No verdict (user interprets via success_criteria_text).

    Pre-commit guards (in order):
    1. Candidate must be in 'ready' stage (state machine invariant).
    2. Holdout seal must be intact (not broken, not tampered).
    3. No-peek data hash verification: recompute SHA256 of cached raw
       OHLCV (all columns) + BTC extended; must match the seal hash.
       (Phase 2.2: tampering with BTC pre-holdout history is now
       detected since BTC extended is in the seal.)
    4. ``min_train_months`` enforcement: the training period must be
       at least ``hypothesis.min_train_months`` months long. Raises
       ``CommitError`` otherwise. (Phase 3.7 E4: prevents bypassing
       the WFA minimum by going directly to commit.)

    Parameters
    ----------
    candidate : Candidate
        The candidate to commit (must be in 'ready' stage).
    success_criteria_text : str
        User-provided success criteria text (logged for audit, not
        evaluated by the framework).
    verbose : bool
        Whether to log progress (default True).

    Returns
    -------
    CommitResult
        Dataclass with all computed metrics, seal hashes (before/after),
        and audit fields.

    Raises
    ------
    NotReadyForCommit
        Candidate is not in 'ready' stage.
    HoldoutAlreadyBroken
        Holdout seal was already broken.
    SealVerificationFailed
        Data hash mismatch (data tampering detected).
    CommitError
        ``min_train_months`` not met, or other unrecoverable commit
        error.
    """
    session: "ResearchSession" = candidate.session
    hypothesis = candidate.hypothesis
    candidate.assert_ready()
    # Lock the candidate into the terminal 'ready' stage to prevent
    # further modifications (state machine invariant for irreversibility).
    if candidate.stage != "ready":
        candidate.mark_ready()

    # Check seal
    if not session.holdout_set.is_sealed():
        raise HoldoutAlreadyBroken(
            f"Holdout {session.holdout_period} already broken. "
            f"This holdout cannot be re-used. To evaluate a different "
            f"holdout window, create a new ResearchSession with a new "
            f"holdout_period (in experiments/base.py: PeriodConfig).",
        )

    if not session.holdout_set.verify():
        raise SealVerificationFailed(
            "Holdout seal verification failed -- data may have been "
            "tampered with since the session was created. "
            "Check that the underlying OHLCV files in the cache "
            "directory have not been modified, then create a new "
            "ResearchSession to retry.",
        )

    # ── C-2: Verify holdout data hash against seal (no-peek) ──
    # Re-compute hash from the cached holdout data loaded at session
    # creation. If the data has been modified between init and commit,
    # the hash will differ and we abort.
    from quant_lib.research.session import _compute_holdout_data_hash
    hold_start, hold_end = session.holdout_period
    narrowed_syms = candidate.narrowed_symbols
    frozen = candidate.frozen_params
    sp = hypothesis.merged_strategy_params()

    # ── Phase 3.7 E4: Enforce min_train_months ──
    # The hypothesis specifies the minimum training months required
    # (default 12). If the actual training period is shorter, refuse
    # to commit -- a short training period produces unreliable frozen
    # params and inflated holdout results.
    train_start, train_end = session.training_period
    n_train_months = (
        (pd.Timestamp(train_end) - pd.Timestamp(train_start)).days / 30.44
    )
    if n_train_months < hypothesis.min_train_months:
        raise CommitError(
            f"Training period ({n_train_months:.1f}mo) is shorter than "
            f"hypothesis min_train_months ({hypothesis.min_train_months}mo). "
            f"Refusing to commit -- re-run with longer training data."
        )

    # Re-compute hash from cached raw OHLCV (and BTC extended if any).
    # Phase 2.2: BTC extended is now part of the seal -- if tampered
    # with between init and commit, the hash mismatch is detected here.
    # Phase 2.3: funding data is also part of the seal. If tampered,
    # mismatch is detected.
    cached_data = session._holdout_data_for_hash
    cached_btc_extended = session._btc_extended_for_features
    cached_funding = getattr(session, "_holdout_funding_for_hash", None)
    recomputed_hash = _compute_holdout_data_hash(
        cached_data,
        btc_extended=cached_btc_extended,
        funding_data=cached_funding,
    )
    if recomputed_hash != session._holdout_hash:
        raise SealVerificationFailed(
            f"Holdout data has been modified between session creation and "
            f"commit (expected hash {session._holdout_hash[:16]}..., "
            f"got {recomputed_hash[:16]}...). "
            f"Possible tampering or data corruption. Commit aborted."
        )

    # ── Compute features from cached raw OHLCV (no re-fetch) ──
    # Phase 2.2: BTC extended data is now pre-loaded at session init
    # and stored in ``session._btc_extended_for_features``. We use
    # this cached data instead of re-fetching from the cache. This
    # ensures (a) hash consistency with the seal, and (b) no silent
    # data drift between init and commit.
    if cached_btc_extended is not None and len(cached_btc_extended) > 0:
        btc_holdout_ext = cached_btc_extended
    else:
        # No pre-loaded BTC extended (e.g., test bypass with
        # _skip_holdout_load or BTC not in symbols). Fall back to
        # the holdout-window BTC from cached_data if available;
        # otherwise empty (features will skip BTC context).
        btc_holdout_ext = cached_data.get(
            "BTCUSDT", pd.DataFrame()
        )

    from quant_lib.core._features import prepare_data_with_max_time
    holdout_features: dict[str, pd.DataFrame] = {}
    for sym in narrowed_syms:
        df_filtered = cached_data.get(sym, pd.DataFrame())
        if len(df_filtered) == 0:
            continue
        fund = session.cache.get_funding(sym, hold_start, hold_end)
        holdout_features[sym] = prepare_data_with_max_time(
            df_raw=df_filtered,
            df_btc_raw=btc_holdout_ext,
            df_funding_raw=fund,
            max_time=pd.Timestamp(hold_end),
            strategy_type=hypothesis.strategy_type,
            # Phase 2.2: Isolate macro_trend from holdout prices to
            # prevent state-persistence leakage. The trend signal
            # at the start of the holdout is the LAST value of the
            # pre-holdout EMA, forward-filled into the holdout.
            btc_holdout_start=pd.Timestamp(hold_start),
        )

    # ── Run frozen-params trade loop (no Optuna) ──
    all_holdout_trades: list[dict] = []
    for sym in narrowed_syms:
        df = holdout_features.get(sym)
        if df is None or len(df) == 0:
            continue
        sym_params = frozen.get(sym, {})
        critical_cols = [
            "open", "high", "low", "close",
            "hh_20", "ll_20", "ema_200",
            "vol_pct_rank", "rvol", "atr",
            "funding_rate", "macro_vol", "macro_trend",
            "is_weekend", "is_funding_hour",
        ]
        # Add pullback-specific cols if needed
        if hypothesis.strategy_type == STRATEGY_PULLBACK_SNIPER:
            critical_cols.extend(["rsi_14", "bullish_reversal", "bearish_reversal"])
        # Phase 2: funding_rate_carry requires funding_pct_rank to be non-NaN
        if hypothesis.strategy_type == STRATEGY_FUNDING_RATE_CARRY:
            critical_cols.extend(["funding_pct_rank"])
        df_clean = df.dropna(subset=critical_cols).copy()

        # Per-symbol seed offset prevents correlated cost noise across
        # assets (SPA assumes independent symbols). Using a stable hash
        # of the symbol name keeps commits reproducible across runs.
        _sym_offset = sum(ord(c) for c in sym)
        rng = np.random.default_rng(42 + _sym_offset)
        random_draws = rng.random(size=len(df_clean) * 2).astype(np.float64)

        rsi_14 = df_clean["rsi_14"].values if "rsi_14" in df_clean.columns else np.zeros(len(df_clean), dtype=np.float64)
        bullish_rev = df_clean["bullish_reversal"].values if "bullish_reversal" in df_clean.columns else np.zeros(len(df_clean), dtype=np.int32)
        bearish_rev = df_clean["bearish_reversal"].values if "bearish_reversal" in df_clean.columns else np.zeros(len(df_clean), dtype=np.int32)
        # Phase 2: funding_rate_carry feature. For vol_compression and
        # pullback_sniper, the strategy branch in fast_trade_loop never
        # reads this array, but the @njit signature requires it. Use 0.5
        # (neutral regime) as the safe default when the column is absent
        # or NaN.
        funding_pct_rank = (
            df_clean["funding_pct_rank"].values
            if "funding_pct_rank" in df_clean.columns
            else np.full(len(df_clean), 0.5, dtype=np.float64)
        )

        result = fast_trade_loop(
            df_clean["open"].values,
            df_clean["high"].values,
            df_clean["low"].values,
            df_clean["close"].values,
            df_clean["hh_20"].values,
            df_clean["ll_20"].values,
            df_clean["ema_200"].values,
            rsi_14,
            bullish_rev,
            bearish_rev,
            df_clean["vol_pct_rank"].values,
            df_clean["rvol"].values,
            df_clean["atr"].values,
            funding_pct_rank,
            df_clean["funding_rate"].values,
            df_clean["macro_vol"].values,
            df_clean["macro_trend"].values,
            df_clean["is_weekend"].values,
            df_clean["is_funding_hour"].values,
            hypothesis.strategy_type,
            sym_params.get("vol_pct_thresh", 0.20),
            DEFAULTS["fixed_rvol_thresh"],
            sym_params.get("pullback_bars", 5),
            sym_params.get("trail_atr", 3.0),
            sym_params.get("sl_mult", 1.5),
            DEFAULTS["bailout_bars"],
            0,
            STATIC["fee_taker"],
            1 if sp.get("use_rvol", True) else 0,
            1 if sp.get("use_ema", True) else 0,
            1 if sp.get("allow_long", True) else 0,
            1 if sp.get("allow_short", True) else 0,
            sym_params.get("rsi_oversold", 30.0),
            sym_params.get("rsi_overbought", 70.0),
            # Phase 2: funding_rate_carry thresholds (safe defaults;
            # commit.py currently exercises vol_compression and
            # pullback_sniper only -- funding_pct_rank is constant 0.5
            # above so these thresholds never trigger).
            0.90, 0.40, 0.60,
            DEFAULTS["weekend_liquidity_penalty"],
            DEFAULTS["stress_test_multiplier"],
            random_draws,
            DEFAULTS["trend_aligned_risk_mult"],
            DEFAULTS["trend_counter_risk_mult"],
        )

        pnl_array = result[0]
        idx_en = result[1]
        idx_ex = result[2]
        t_dir = result[3]
        t_entry_prices = result[6]
        t_exit_prices = result[7]
        t_sl_pcts = result[8]
        h_trend_mults = result[9]

        if len(pnl_array) > 0:
            en_times = df_clean["time"].iloc[idx_en].tolist()
            ex_times = df_clean["time"].iloc[idx_ex].tolist()
            _missing_risk_weight_syms: set[str] = set()
            for e_time, x_time, r, d, e_pr, x_pr, sl_pct_v, t_mult in zip(
                en_times, ex_times, pnl_array,
                t_dir[:len(pnl_array)],
                t_entry_prices,
                t_exit_prices,
                t_sl_pcts,
                h_trend_mults,
            ):
                # B0.4: use the candidate's PF-allocated weight for this
                # symbol. The previous code used 0.01 unconditionally
                # (via ``.get(sym, 0.01)``) because
                # ``candidate.risk_weights`` was an empty dict -- this
                # effectively discarded the entire PF allocation and
                # produced holdout PSR that was not representative of
                # the WFA edge.
                if sym in candidate.risk_weights:
                    _rw = float(candidate.risk_weights[sym])
                else:
                    _rw = DEFAULTS["default_risk_per_pair"]
                    _missing_risk_weight_syms.add(sym)
                all_holdout_trades.append({
                    "entry_time": e_time,
                    "exit_time": x_time,
                    "symbol": sym,
                    "r_net": r,
                    "entry_price": float(e_pr),
                    "exit_price": float(x_pr),
                    "trade_dir": d,
                    "sl_pct": sl_pct_v,
                    "sl_mult": sym_params.get("sl_mult", 1.5),
                    "trail_atr": sym_params.get("trail_atr", 3.0),
                    "risk_weight": _rw,
                    "trend_risk_mult": t_mult,
                })

            if _missing_risk_weight_syms:
                from quant_lib.core._logging import log as _hqs_log
                _hqs_log.warning(
                    f"Holdout commit: {len(_missing_risk_weight_syms)} symbol(s) "
                    f"missing from candidate.risk_weights: "
                    f"{sorted(_missing_risk_weight_syms)}. Using default "
                    f"{DEFAULTS['default_risk_per_pair']}. This may indicate "
                    f"run_edge_testing was not run or returned no folds."
                )

    # ── Portfolio simulation ──
    # Only use narrowed symbols that actually have features computed.
    # Symbols with no cached data are silently dropped (line 237).
    available_syms = [s for s in narrowed_syms if s in holdout_features]
    holdout_close, holdout_hl = build_daily_matrices(available_syms, holdout_features)
    # Pass the candidate's PF-allocated weights to the portfolio sim.
    # If the candidate has no risk_weights at all (no WFA folds),
    # pass None to let the simulator use its own default -- the
    # warning above was already emitted per-symbol.
    full_weights = candidate.risk_weights
    if full_weights:
        holdout_weights = {
            sym: full_weights[sym]
            for sym in available_syms
            if sym in full_weights
        }
    else:
        # No candidate weights (e.g., zero WFA folds). The per-trade
        # ``risk_weight`` above was already defaulted; we leave this
        # to the portfolio sim's own default.
        holdout_weights = None

    h_final_eq, h_daily_eq, h_executed, h_reject = simulate_full_portfolio(
        trades=all_holdout_trades,
        initial_cash=session.initial_capital,
        leverage=DEFAULTS["leverage"],
        mm_pct=STATIC["maintenance_margin_pct"],
        position_limit=DEFAULTS["global_position_limit"],
        cb_hard_cooldown_hours=DEFAULTS["cb_hard_cooldown_hours"],
        fixed_cb_threshold=DEFAULTS["fixed_cb_threshold"],
        daily_close_matrix=holdout_close,
        asset_risk_weights=holdout_weights,
        end_date=hold_end,
        liquidation_fee_pct=STATIC["liquidation_fee_pct"],
        daily_hl_matrix=holdout_hl,
    )

    # ── Compute SHA256 of actual holdout data BEFORE breaking seal ──
    # C-2 fix: use the cached raw OHLCV (all columns + BTC extended)
    # so the hash matches the one computed at session init. This
    # means that if data was modified between init and commit, the
    # commit_break call below will record the mismatch and the
    # verify() would also catch it (defense in depth).
    from quant_lib.research.session import _compute_holdout_data_hash
    seal_hash_after = _compute_holdout_data_hash(
        cached_data,
        btc_extended=cached_btc_extended,
        funding_data=cached_funding,  # Phase 2.3
    )

    # Use the public commit_break API (no private attribute access).
    # Returns (was_intact, hash_before, hash_after).
    was_intact, seal_hash_before, seal_hash_after = (
        session.holdout_set.commit_break(seal_hash_after)
    )
    if not was_intact:
        # Use the configured seal directory (env var or convention
        # default) instead of hardcoding the path. Phase 1.6 fix.
        seal_dir_hint = os.environ.get(
            "QUANT_LIB_SEAL_DIR",
            os.path.join(session.cache_dir, "holdout_seals"),
        )
        raise CommitError(
            "Holdout seal was already broken at the moment of "
            "commit_break. This indicates a race condition (another "
            "commit ran concurrently) or a state inconsistency. "
            f"Investigate {seal_dir_hint}/ for the file "
            "and re-create the session to retry.",
        )

    # ── Compute all metrics ──
    r_vals = np.array([t["r_net"] for t in h_executed]) if h_executed else np.array([])
    n_trades = len(r_vals)

    equity_change = h_final_eq - session.initial_capital
    equity_pct = (equity_change / session.initial_capital) * 100 if session.initial_capital > 0 else 0.0

    # CAGR
    if len(h_daily_eq) > 1:
        eq_series = pd.Series(h_daily_eq).sort_index()
        n_days = max(len(eq_series), 1)
        cagr = ((eq_series.iloc[-1] / eq_series.iloc[0]) ** (365.25 / n_days) - 1) * 100
        max_dd = ((eq_series - eq_series.cummax()) / eq_series.cummax()).min() * 100
    else:
        cagr = 0.0
        max_dd = 0.0

    # Trade-level stats
    if n_trades > 0:
        win_rate = (r_vals > 0).sum() / n_trades * 100
        avg_r = float(np.mean(r_vals))
        median_r = float(np.median(r_vals))
        std_r = float(np.std(r_vals))
        best_r = float(np.max(r_vals))
        worst_r = float(np.min(r_vals))
        gains = r_vals[r_vals > 0].sum()
        losses = abs(r_vals[r_vals < 0].sum())
        pf = gains / losses if losses > 0 else float("inf")
        sharpe = avg_r / std_r if std_r > 0 else 0.0
        # PSR + ESS (consolidated prob_sharpe_ratio in 0.2.2).
        # annualize=False because r_vals are R-multiples (per-trade), not
        # daily returns.
        _, psr_ess_val = prob_sharpe_ratio(r_vals, annualize=False)
        from scipy import stats as scipy_stats
        skew_val = float(scipy_stats.skew(r_vals)) if n_trades >= 3 else 0.0
        kurt_val = float(scipy_stats.kurtosis(r_vals, fisher=True)) if n_trades >= 3 else 0.0
        # Kish effective sample size: ESS = (sum w)^2 / sum(w^2).
        # For uniform weights w_i = 1/n, this simplifies to n.
        # Previously reported as n-1 (sample variance df), which was
        # inconsistent with the label "ess". Phase 1.2 fix: report
        # Kish-corrected ESS, which equals n for uniform weights.
        if n_trades > 0:
            w = np.ones(n_trades) / n_trades
            ess = 1.0 / float(np.dot(w, w))  # = n for uniform weights
        else:
            ess = 0.0
        # Bars held
        bars_list = []
        for t in h_executed:
            entry = t["entry_time"]
            exit_ = t["exit_time"]
            if hasattr(entry, "timestamp") and hasattr(exit_, "timestamp"):
                bars_list.append((exit_ - entry).total_seconds() / 3600)
            else:
                bars_list.append(0)
        avg_bars = float(np.mean(bars_list)) if bars_list else 0.0
    else:
        win_rate = avg_r = median_r = std_r = 0.0
        best_r = worst_r = 0.0
        pf = sharpe = 0.0
        # No evidence either way when zero trades -- NaN, not 0.5.
        # 0.5 would imply "neutral coin flip", which is misleading.
        psr_ess_val = float("nan")
        skew_val = kurt_val = 0.0
        ess = 0.0
        avg_bars = 0.0

    # ── Phase 2.1: Deflated PSR (Bailey & López de Prado 2014) ──
    # Adjusts the single-trial PSR for the family of Optuna trials
    # that produced the winning params. The "family" includes all
    # Optuna trials across symbols × folds during the WFA phase.
    # Real family size:
    #   n_symbols × n_folds × n_optuna_trials_per_fold
    # This is the missing piece between PSR (single-trial) and the
    # existing Bonferroni correction (which counts only n_commits).
    if n_trades >= 5 and psr_ess_val == psr_ess_val:  # n_trades > 0 and PSR not NaN
        # Approximate the per-trial Sharpe from the observed R-multiples.
        # Note: this is the same Sharpe used internally by PSR (skewness-
        # adjusted), not the raw per-trade Sharpe. We invert PSR via
        # the inverse normal CDF to get the "effective z-score" then
        # multiply by the SR standard deviation to recover the
        # effective SR. For a simpler approach, we use the observed
        # raw Sharpe (avg_r / std_r) as the input -- it is the
        # quantity Bailey & López de Prado 2014 explicitly use.
        # Use the raw per-trade SR as observed_sharpe input. The PSR
        # formula already adjusts for skew/kurt internally, but the
        # deflated PSR uses the observed SR directly per the
        # original Bailey & López de Prado paper.
        if n_trades > 1:
            # Realistic family of Optuna trials: symbols × folds ×
            # trials_per_fold. We compute this from the actual fold
            # data on the candidate.
            fold_counts = (
                sum(len(fps) for fps in candidate.fold_params.values())
                if hasattr(candidate, "fold_params") and candidate.fold_params
                else 0
            )
            # Each fold runs DEFAULTS["wfa_trials_per_fold"] trials.
            n_trials_deflated = max(
                fold_counts * DEFAULTS.get("wfa_trials_per_fold", 80),
                1,
            )
            # Map observed PSR back to an effective SR for the
            # deflated formula. We use the unadjusted per-trade
            # Sharpe (avg_r / std_r) as the input -- the PSR
            # already encoded skew/kurt, but the deflated formula
            # uses the SR as-is.
            observed_sharpe_for_deflated = sharpe  # = avg_r / std_r
            deflated_psr_val = deflated_sharpe_ratio(
                observed_sharpe=observed_sharpe_for_deflated,
                n_trials=n_trials_deflated,
                returns_skewness=skew_val,
                returns_excess_kurtosis=kurt_val,  # fisher=True is excess
                benchmark_sharpe=0.0,
                n_obs_per_trial=n_trades,
            )
            n_trials_in_deflated = n_trials_deflated
        else:
            deflated_psr_val = float("nan")
            n_trials_in_deflated = 0
    else:
        # Zero trades: no PSR to deflate.
        deflated_psr_val = float("nan")
        n_trials_in_deflated = 0

    # ── Phase 4.1: Trade bootstrap (on R-multiples, not daily returns) ──
    from quant_lib.core._metrics import run_trade_bootstrap
    if n_trades >= 5:
        tb = run_trade_bootstrap(r_vals, session.initial_capital)
        tb_worst5_cagr = tb["Worst5_CAGR"]
        tb_worst95_dd = tb["Worst95_DD"]
        tb_worst5_dd = tb["Worst5_DD"]
        tb_worst1_dd = tb["Worst1_DD"]
        tb_block = tb["Block"]
    else:
        tb_worst5_cagr = float("nan")
        tb_worst95_dd = float("nan")
        tb_worst5_dd = float("nan")
        tb_worst1_dd = float("nan")
        tb_block = 0

    # ── Phase 4.2: Regime stats (Bull/Bear PF) ──
    from quant_lib.core._metrics import compute_regime_stats
    if n_trades >= 3:
        regime_stats = compute_regime_stats(h_executed)
        regime_bull_pf, regime_bull_n = regime_stats["Bull"]
        regime_bear_pf, regime_bear_n = regime_stats["Bear"]
    else:
        regime_bull_pf = regime_bear_pf = float("nan")
        regime_bull_n = regime_bear_n = 0

    # By-symbol stats
    by_symbol = {}
    for sym in narrowed_syms:
        sym_trades = [t for t in h_executed if t["symbol"] == sym]
        if sym_trades:
            sym_r = np.array([t["r_net"] for t in sym_trades])
            sym_gains = sym_r[sym_r > 0].sum() if (sym_r > 0).any() else 0.0
            sym_losses = abs(sym_r[sym_r < 0].sum()) if (sym_r < 0).any() else 0.0
            by_symbol[sym] = {
                "n_trades": len(sym_trades),
                "win_rate": float((sym_r > 0).sum() / len(sym_r) * 100),
                "avg_r": float(np.mean(sym_r)),
                "profit_factor": sym_gains / sym_losses if sym_losses > 0 else float("inf"),
                "total_r": float(np.sum(sym_r)),
            }

    # Trend alignment impact
    with_trend_r_total = sum(
        t["r_net"] for t in h_executed
        if t.get("trend_risk_mult", 1.0) > 1.0
    )
    counter_trend_r_total = sum(
        t["r_net"] for t in h_executed
        if t.get("trend_risk_mult", 1.0) < 1.0
    )
    with_trend_count = sum(
        1 for t in h_executed
        if t.get("trend_risk_mult", 1.0) > 1.0
    )
    counter_trend_count = sum(
        1 for t in h_executed
        if t.get("trend_risk_mult", 1.0) < 1.0
    )

    # FDR context
    commit_idx = session.n_commits + 1
    bonf_alpha = session.adjusted_alpha_for_commit(commit_idx)
    fdr_alpha = session.fdr_alpha

    # ── Build result ──
    commit_result = CommitResult(
        candidate_name=hypothesis.name,
        commit_idx=commit_idx,
        holdout_period=session.holdout_period,
        timestamp=datetime.now(timezone.utc).isoformat()[:19],
        initial_capital=session.initial_capital,
        final_equity=h_final_eq,
        equity_pct=equity_pct,
        cagr_pct=cagr,
        max_dd_pct=max_dd,
        n_raw_trades=len(all_holdout_trades),
        n_executed_trades=len(h_executed),
        n_rejected=len(all_holdout_trades) - len(h_executed),
        reject_breakdown=dict(h_reject),
        n_trades=n_trades,
        win_rate=win_rate,
        avg_r=avg_r,
        median_r=median_r,
        std_r=std_r,
        best_r=best_r,
        worst_r=worst_r,
        profit_factor=pf,
        avg_bars_held=avg_bars,
        sharpe_r=sharpe,
        psr=psr_ess_val,
        psr_ess=psr_ess_val,
        skew=skew_val,
        kurtosis=kurt_val,
        ess=ess,
        deflated_psr=deflated_psr_val,
        n_trials_in_deflated=n_trials_in_deflated,
        # Phase 4.1: trade bootstrap fields
        trade_bootstrap_worst5_cagr=tb_worst5_cagr,
        trade_bootstrap_worst95_dd=tb_worst95_dd,
        trade_bootstrap_worst5_dd=tb_worst5_dd,
        trade_bootstrap_worst1_dd=tb_worst1_dd,
        trade_bootstrap_block=tb_block,
        # Phase 4.2: regime stats
        regime_bull_pf=regime_bull_pf,
        regime_bull_n=regime_bull_n,
        regime_bear_pf=regime_bear_pf,
        regime_bear_n=regime_bear_n,
        bonferroni_alpha=bonf_alpha,
        fdr_alpha=fdr_alpha,
        by_symbol_stats=by_symbol,
        # Sprint 3 fix 3.6: real daily equity from holdout sim.
        daily_equity=h_daily_eq if h_daily_eq else None,
        with_trend_trades=with_trend_count,
        with_trend_r_total=with_trend_r_total,
        counter_trend_trades=counter_trend_count,
        counter_trend_r_total=counter_trend_r_total,
        seal_hash_before=seal_hash_before,
        seal_hash_after=seal_hash_after,
        seal_broken=True,
        success_criteria_text=success_criteria_text,
    )

    # Record in session
    session.record_commit(
        candidate=candidate,
        final_equity=h_final_eq,
        equity_pct=equity_pct,
        n_trades=n_trades,
        psr=psr_ess_val,
        seal_hash=seal_hash_after,
        success_criteria_text=success_criteria_text,
    )

    # Log to journal
    session.journal.log_run(
        description=f"COMMIT #{commit_idx}: {hypothesis.name} | "
                    f"equity=${h_final_eq:,.2f} ({equity_pct:+.1f}%) | "
                    f"PSR={psr_ess_val:.3f} | "
                    f"n_trades={n_trades}",
        category="ablation",
        params_snapshot={
            "commit_idx": commit_idx,
            "final_equity": round(h_final_eq, 2),
            "equity_pct": round(equity_pct, 2),
            "psr": round(psr_ess_val, 4),
            "seal_hash": seal_hash_after[:16],
        },
    )

    if verbose:
        import logging
        log = logging.getLogger("rich")
        log.info(
            f"COMMIT #{commit_idx}: {hypothesis.name} | "
            f"equity=${h_final_eq:,.2f} ({equity_pct:+.1f}%) | "
            f"PSR={psr_ess_val:.3f} | "
            f"Bonf={bonf_alpha:.4f} | "
            f"n_trades={n_trades} | "
            f"seal BROKEN"
        )

    return commit_result
