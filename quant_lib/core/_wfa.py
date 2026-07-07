"""
Walk-Forward Analysis -- Optuna-based parameter optimization.

Extracted from Hans_Quant_Systems.py:
  - WalkForwardObjective (lines 1880-2051)
  - _get_purge_days (lines 2091-2110)
  - _adaptive_trials (lines 2113-2146)
  - run_wfa_per_symbol (lines 2180-2517)
"""

import numpy as np
from numpy import ndarray
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from scipy import stats as scipy_stats
from datetime import timedelta
from typing import Any

from quant_lib.core._config import STATIC, GLOBAL_SEED, DEFAULTS
from quant_lib.core._logging import console
from quant_lib.core._engine import fast_trade_loop

_FoldParamsList = list[dict[str, Any]]
_WFAResult = tuple[str, list[dict], _FoldParamsList]


class WalkForwardObjective:
    def __init__(
        self,
        df_prepped: pd.DataFrame,
        expected_trades_annual: int,
        use_rvol: bool,
        use_ema: bool,
        fold_seed: int,
        reg_lambda: float = 0.05,
        decay_halflife_months: int = 15,
        strategy_type: int = 0,
        allow_long: int = 1,
        allow_short: int = 1,
        search_space: dict | None = None,
    ):
        self.df_p: pd.DataFrame = df_prepped
        self.expected_trades: int = expected_trades_annual
        self.use_rvol: bool = use_rvol
        self.use_ema: bool = use_ema
        self.fold_seed: int = fold_seed
        self.reg_lambda: float = reg_lambda
        self.strategy_type: int = strategy_type
        self.allow_long: int = allow_long
        self.allow_short: int = allow_short
        self.search_space: dict = search_space or DEFAULTS["search_space"]

        # WFA parameter centers - extracted from hardcoded values to DEFAULTS config
        self.param_center: dict[str, float] = {
            "vol_pct_thresh": DEFAULTS["vol_thresh_center"],
            "pullback_bars": DEFAULTS["pullback_bars_center"],
            "trail_atr": DEFAULTS["trail_atr_center"],
            "sl_mult": DEFAULTS["sl_mult_center"],
            "rsi_oversold": DEFAULTS["rsi_oversold_center"],
            "rsi_overbought": DEFAULTS["rsi_overbought_center"],
        }
        self.param_scale: dict[str, float] = {
            "vol_pct_thresh": DEFAULTS["vol_thresh_scale"],
            "pullback_bars": DEFAULTS["pullback_bars_scale"],
            "trail_atr": DEFAULTS["trail_atr_scale"],
            "sl_mult": DEFAULTS["sl_mult_scale"],
            "rsi_oversold": DEFAULTS["rsi_oversold_scale"],
            "rsi_overbought": DEFAULTS["rsi_overbought_scale"],
        }

        rng = np.random.default_rng(fold_seed)
        self.random_draws: ndarray = rng.random(size=len(self.df_p) * 2).astype(np.float64)

        n_bars = len(df_prepped)
        # Phase 2.4 (v0.4.0): fail-fast on empty input. Previously the
        # ``self.bar_weights /= self.bar_weights.mean()`` line silently
        # produced an empty array (mean of empty is NaN; numpy propagates
        # the NaN into the division but does not raise). The downstream
        # ``__call__`` guard at len < 168 eventually returned -9999, but
        # the bar_weights state was corrupt and obscured the real bug.
        if n_bars == 0:
            raise ValueError(
                "WalkForwardObjective requires a non-empty df_prepped "
                f"(got {n_bars} bars). Check the upstream data pipeline."
            )
        if decay_halflife_months > 0:
            halflife_bars = decay_halflife_months * 30 * 24
            bar_ages = np.arange(n_bars - 1, -1, -1, dtype=np.float64)
            self.bar_weights = np.exp(-np.log(2) * bar_ages / halflife_bars)
            self.bar_weights /= self.bar_weights.mean()
        else:
            self.bar_weights = np.ones(n_bars, dtype=np.float64)

        # Hansen-literal SPA support (claim #3 Blocker A): collect the IS
        # PnL array of every Optuna trial in this fold so the SPA null can
        # resample per-trial loss-differentials (Hansen 2005 Eq.6-8). A
        # ``None`` sentinel marks trials that short-circuited an early
        # return (len df_p < 168, n_trades < 15, w_var <= 0) so the list
        # length always equals ``n_trials`` regardless of which branch the
        # trial took; the Hansen helper filters sentinels out. A fresh
        # ``WalkForwardObjective`` instance is built per fold (call site
        # ``run_wfa_per_symbol`` L577), so per-fold isolation is by
        # construction -- no cross-fold reset needed.
        self.trial_r_nets: list = []

    def __call__(self, trial: optuna.trial.Trial) -> float:
        if len(self.df_p) < 168:
            self.trial_r_nets.append(None)
            return -9999.0

        ss = self.search_space
        v_thresh = trial.suggest_float(
            "vol_pct_thresh", *ss["vol_pct_thresh"]
        )
        r_thresh = DEFAULTS["fixed_rvol_thresh"]
        pb_bars = trial.suggest_int(
            "pullback_bars", *ss["pullback_bars"]
        )
        t_atr = trial.suggest_float(
            "trail_atr", *ss["trail_atr"]
        )
        sl_mult = trial.suggest_float(
            "sl_mult", *ss["sl_mult"]
        )

        # Pullback sniper-specific params.
        # Defaults are derived from search_space midpoint when available,
        # falling back to the L2 center (which is the canonical midpoint).
        # This keeps search_space the single source of truth for param ranges.
        rsi_oversold = float(self.param_center["rsi_oversold"])
        rsi_overbought = float(self.param_center["rsi_overbought"])
        if self.strategy_type == 1:
            rsi_oversold_range = ss.get(
                "rsi_oversold",
                (self.param_center["rsi_oversold"] - self.param_scale["rsi_oversold"],
                 self.param_center["rsi_oversold"] + self.param_scale["rsi_oversold"]),
            )
            rsi_overbought_range = ss.get(
                "rsi_overbought",
                (self.param_center["rsi_overbought"] - self.param_scale["rsi_overbought"],
                 self.param_center["rsi_overbought"] + self.param_scale["rsi_overbought"]),
            )
            rsi_oversold = trial.suggest_float("rsi_oversold", *rsi_oversold_range)
            rsi_overbought = trial.suggest_float("rsi_overbought", *rsi_overbought_range)

        # Extract pullback-specific features from precomputed df
        rsi_14 = self.df_p["rsi_14"].values if "rsi_14" in self.df_p.columns else np.zeros(len(self.df_p), dtype=np.float64)
        bullish_rev = self.df_p["bullish_reversal"].values if "bullish_reversal" in self.df_p.columns else np.zeros(len(self.df_p), dtype=np.int32)
        bearish_rev = self.df_p["bearish_reversal"].values if "bearish_reversal" in self.df_p.columns else np.zeros(len(self.df_p), dtype=np.int32)

        result = fast_trade_loop(
            self.df_p["open"].values,
            self.df_p["high"].values,
            self.df_p["low"].values,
            self.df_p["close"].values,
            self.df_p["hh_20"].values,
            self.df_p["ll_20"].values,
            self.df_p["ema_200"].values,
            rsi_14,
            bullish_rev,
            bearish_rev,
            self.df_p["vol_pct_rank"].values,
            self.df_p["rvol"].values,
            self.df_p["atr"].values,
            # Phase 2: funding_pct_rank slot. WFA currently exercises
            # vol_compression and pullback_sniper; this slot is constant
            # 0.5 (neutral funding regime) so the funding-carry branch
            # never triggers until WFA is extended for that strategy.
            self.df_p["funding_pct_rank"].values if "funding_pct_rank" in self.df_p.columns else np.full(len(self.df_p), 0.5, dtype=np.float64),
            self.df_p["funding_rate"].values,
            self.df_p["macro_vol"].values,
            self.df_p["macro_trend"].values,
            self.df_p["is_weekend"].values,
            self.df_p["is_funding_hour"].values,
            self.strategy_type,
            v_thresh,
            r_thresh,
            pb_bars,
            t_atr,
            sl_mult,
            DEFAULTS["bailout_bars"],
            0,  # warmup_bars=0 because df_is has already been dropna'd
            STATIC["fee_taker"],
            self.use_rvol,
            self.use_ema,
            self.allow_long,
            self.allow_short,
            rsi_oversold,
            rsi_overbought,
            # Phase 2: funding_rate_carry thresholds (safe defaults).
            0.90, 0.40, 0.60,
            DEFAULTS["weekend_liquidity_penalty"],
            DEFAULTS["stress_test_multiplier"],
            self.random_draws,
            DEFAULTS["trend_aligned_risk_mult"],
            DEFAULTS["trend_counter_risk_mult"],
        )
        pnl_array = np.asarray(result[0], dtype=float)
        idx_entry = result[1]

        n_trades = len(pnl_array)
        if n_trades < 15:
            self.trial_r_nets.append(None)
            return -9999.0

        trade_w = self.bar_weights[idx_entry]
        trade_w = trade_w / trade_w.sum()

        w_mean = float(np.dot(pnl_array, trade_w))
        w_var = float(np.dot(trade_w, (pnl_array - w_mean) ** 2))
        if w_var <= 0:
            self.trial_r_nets.append(None)
            return -9999.0
        w_sr = w_mean / np.sqrt(w_var)

        ess = (trade_w.sum() ** 2) / np.dot(trade_w, trade_w)

        # ── PSR objective (replaces traditional SR) ──
        # PSR accounts for skewness and kurtosis, more valid for crypto
        # returns which are fat-tailed and asymmetric.
        # NOTE: Bailey's formula uses REGULAR kurtosis (γ₄ = excess + 3),
        # so the coefficient is (γ₄ - 1)/4 = (excess + 2)/4. prob_sharpe_ratio
        # in core/_testing.py uses the same conversion. See CHANGELOG v0.3.1
        # for the prior bug where (excess - 1)/4 was used.
        # Fallback (Sprint 1 fix): when Bailey's variance correction is
        # non-positive OR ESS < 2 OR n_trades < 10, the formula is
        # unreliable. Previously psr was set to 0.5 (neutral) which
        # allowed degenerate strategies to outcompete mediocre normal
        # ones in Optuna search. Now we fall back to the raw z-score
        # ``norm.cdf(w_sr)`` -- the asymptotic limit of Bailey's
        # formula when skew/kurt corrections go to zero. This preserves
        # the sign of SR (negative -> psr < 0.5) instead of defaulting
        # to neutral. If w_sr is NaN, the resulting psr is NaN and
        # Optuna rejects the trial (same behavior as before for the
        # truly degenerate case).
        if n_trades >= 10:
            skew = float(scipy_stats.skew(pnl_array))
            kurt = float(scipy_stats.kurtosis(pnl_array, fisher=True))
            var_corr = 1.0 - skew * w_sr + ((kurt + 2.0) / 4.0) * w_sr ** 2
            if var_corr <= 0.0 or ess < 2.0:
                # Bailey formula outside valid range -- use raw z-score
                # as the asymptotic degenerate limit (sign-preserving).
                psr = float(scipy_stats.norm.cdf(w_sr))
            else:
                psr_variance = var_corr / (ess - 1.0)
                psr = float(scipy_stats.norm.cdf(w_sr / np.sqrt(psr_variance)))
        else:
            # Sample too small for reliable skew/kurtosis estimation.
            # Use raw z-score (no skew/kurt correction) as conservative
            # sign-preserving fallback. If w_sr is NaN, psr is NaN and
            # Optuna rejects the trial.
            psr = float(scipy_stats.norm.cdf(w_sr))

        trade_weight = float(
            np.clip(np.log(ess + 1) / np.log(self.expected_trades + 1), 0.1, 1.0)
        )
        obj = psr * trade_weight

        if self.reg_lambda > 0:
            l2_terms = (
                ((v_thresh - self.param_center["vol_pct_thresh"]) / self.param_scale["vol_pct_thresh"]) ** 2
                + ((pb_bars - self.param_center["pullback_bars"]) / self.param_scale["pullback_bars"]) ** 2
                + ((t_atr - self.param_center["trail_atr"]) / self.param_scale["trail_atr"]) ** 2
                + ((sl_mult - self.param_center["sl_mult"]) / self.param_scale["sl_mult"]) ** 2
            )
            if self.strategy_type == 1:
                l2_terms += (
                    ((rsi_oversold - self.param_center["rsi_oversold"]) / self.param_scale["rsi_oversold"]) ** 2
                    + ((rsi_overbought - self.param_center["rsi_overbought"]) / self.param_scale["rsi_overbought"]) ** 2
                )
            l2_penalty = self.reg_lambda * l2_terms
            obj -= l2_penalty

        # Record this trial's IS PnL array for the Hansen-literal SPA null.
        # Reached only on the success path (past the 3 early-return guards
        # above all appended ``None`` sentinels). Downsampling not needed:
        # every entry is the per-trial candidate's IS PnL stream.
        self.trial_r_nets.append(pnl_array)
        return obj


def _get_purge_days(n_is_months: int) -> int:
    """
    Adaptive purge days -- shrinks as IS grows because contamination from
    boundary artifacts is naturally diluted by the larger data volume.
    """
    if n_is_months <= 0:
        # Defensive guard: empty/negative IS is an invalid input. Return
        # maximum purge (90 days) as a safe default. This branch should
        # not be reached in normal flow because run_wfa_per_symbol checks
        # `len(df_is) < 1000` and returns -9999 long before this is called.
        return 90
    if n_is_months <= 15:
        return 90
    elif n_is_months <= 30:
        return 60
    else:
        return 30


def _adaptive_trials(n_is_months: int, prev_best_params: dict | None) -> int:
    """
    Adaptive trial count based on IS size and warm-start availability.
    """
    if n_is_months <= 0:
        # Defensive guard: empty/negative IS. Return minimum 50 trials as
        # a safe default (callers should reject this case before reaching
        # here, but guard against division by zero / silly values).
        return 50
    base = DEFAULTS["wfa_trials_per_fold"]
    ada_prior = prev_best_params is not None

    if n_is_months < 18:
        return max(50, int(base * (0.70 if ada_prior else 0.85)))
    elif n_is_months < 30:
        return int(base * (0.60 if ada_prior else 0.75))
    else:
        return max(50, int(base * (0.50 if ada_prior else 0.70)))


def _run_engine_on_data(
    df,
    best_p: dict,
    strategy_type: int,
    fold_seed: int,
    use_rvol: int,
    use_ema: int,
    allow_long: int,
    allow_short: int,
    fold_key: str,
    atr_inv_fold: float,
    symbol: str,
) -> list[dict]:
    """Phase 2.5: run fast_trade_loop on a given DataFrame with
    the winning Optuna params, returning a list of trade dicts.

    Used to produce IS trades for the PF-weighted risk allocator
    (decoupling it from the OOS trades used by the strategy
    selector and SPA test). The structure of returned trade dicts
    matches the OOS trades: same keys, fold_key identifies the
    fold. Caller decides which subset of fields to use.

    Seed rationale (``fold_seed ^ 0xBEEF``):
        The base fold_seed is shared by three independent random
        streams (see ``run_wfa_per_symbol``):
            - ``fold_seed ^ 0xDEF``  → Optuna warm-start perturbation
            - ``fold_seed ^ 0xABCD`` → OOS random draws (cost noise)
            - ``fold_seed ^ 0xBEEF`` → IS random draws (cost noise)
        XOR masks are arbitrary constant bit-flips that produce
        statistically independent streams from the same base seed.
        All three streams are deterministic given ``fold_seed``,
        making the entire WFA pipeline reproducible across runs
        and machines. The three masks are non-overlapping high/
        mid/low bit patterns to avoid correlated sequences.
    """
    rng_is = np.random.default_rng(fold_seed ^ 0xBEEF)
    random_draws_is = rng_is.random(size=len(df) * 2).astype(np.float64)
    rsi_14_is = df["rsi_14"].values if "rsi_14" in df.columns else np.zeros(len(df), dtype=np.float64)
    bullish_rev_is = df["bullish_reversal"].values if "bullish_reversal" in df.columns else np.zeros(len(df), dtype=np.int32)
    bearish_rev_is = df["bearish_reversal"].values if "bearish_reversal" in df.columns else np.zeros(len(df), dtype=np.int32)

    (
        pnl, idx_en, idx_ex, t_dir,
        m_trend, cb_vol, en_pr, ex_pr, sl_pcts, trend_mults,
    ) = fast_trade_loop(
        df["open"].values,
        df["high"].values,
        df["low"].values,
        df["close"].values,
        df["hh_20"].values,
        df["ll_20"].values,
        df["ema_200"].values,
        rsi_14_is,
        bullish_rev_is,
        bearish_rev_is,
        df["vol_pct_rank"].values,
        df["rvol"].values,
        df["atr"].values,
        # Phase 2: funding_pct_rank slot (safe default).
        df["funding_pct_rank"].values if "funding_pct_rank" in df.columns else np.full(len(df), 0.5, dtype=np.float64),
        df["funding_rate"].values,
        df["macro_vol"].values,
        df["macro_trend"].values,
        df["is_weekend"].values,
        df["is_funding_hour"].values,
        strategy_type,
        best_p.get("vol_pct_thresh", 0.20),
        DEFAULTS["fixed_rvol_thresh"],
        best_p.get("pullback_bars", 5),
        best_p.get("trail_atr", 3.0),
        best_p.get("sl_mult", 1.5),
        DEFAULTS["bailout_bars"],
        0,
        STATIC["fee_taker"],
        use_rvol,
        use_ema,
        allow_long,
        allow_short,
        best_p.get("rsi_oversold", 30.0),
        best_p.get("rsi_overbought", 70.0),
        # Phase 2: funding_rate_carry thresholds (safe defaults).
        0.90, 0.40, 0.60,
        DEFAULTS["weekend_liquidity_penalty"],
        DEFAULTS["stress_test_multiplier"],
        random_draws_is,
        DEFAULTS["trend_aligned_risk_mult"],
        DEFAULTS["trend_counter_risk_mult"],
    )

    if len(pnl) == 0:
        return []
    en_times = df["time"].iloc[idx_en].tolist()
    ex_times = df["time"].iloc[idx_ex].tolist()
    trades = []
    for e_time, x_time, r, d, e_pr, x_pr, m_t, cb_v, sl_p, t_mult in zip(
        en_times, ex_times, pnl, t_dir, en_pr, ex_pr,
        m_trend, cb_vol, sl_pcts, trend_mults,
    ):
        trades.append({
            "entry_time": e_time,
            "exit_time": x_time,
            "symbol": symbol,
            "r_net": r,
            "entry_price": e_pr,
            "exit_price": x_pr,
            "trade_dir": d,
            "sl_pct": sl_p,
            "sl_mult": best_p.get("sl_mult", 1.5),
            "trail_atr": best_p.get("trail_atr", 3.0),
            "m_trend": m_t,
            "macro_vol": cb_v,
            "risk_weight": DEFAULTS["default_risk_per_pair"],
            "trend_risk_mult": t_mult,
            "atr_inv": atr_inv_fold,
            "fold_key": fold_key,
        })
    return trades


def run_wfa_per_symbol(
    symbol, precomputed_df, use_rvol, use_ema, verbose=True, reg_lambda=0.05,
    strategy_type=0, allow_long=1, allow_short=1, search_space=None,
    return_is_trades: bool = True,
):
    """Run walk-forward optimization for one symbol, collecting OOS trades.

    Phase 2.5: When ``return_is_trades=True`` (default), also returns
    a list of per-fold IS trades (one list of trade dicts per fold,
    in the same order as the OOS trades). These IS trades are produced
    by running fast_trade_loop on the IS data with the winning Optuna
    params. They are intended for the PF-weighted risk allocator only,
    so the meta-allocator is decoupled from the strategy selector
    (avoids double-use of OOS trades for both SPA and risk weighting).

    Returns
    -------
    tuple
        (local_trades, fold_params_list, is_trades_per_fold) when
        ``return_is_trades=True`` (default), else
        (local_trades, fold_params_list) for backward compat.

        - local_trades: list of OOS trade dicts
        - fold_params_list: list of fold metadata dicts
        - is_trades_per_fold: list of lists (one per fold) of IS trade
          dicts, or [] when ``return_is_trades=False``
    """
    local_trades = []
    fold_params_list = []
    is_trades_per_fold: list[list[dict]] = []  # Phase 2.5
    is_avg = DEFAULTS["default_expected_trades_per_year"]
    unique_months = precomputed_df["time"].dt.to_period("M").sort_values().unique()
    min_train_months = DEFAULTS["wfa_min_train_months"]
    _search_space = search_space or DEFAULTS["search_space"]
    test_months = DEFAULTS["wfa_test_months"]

    total_folds = len(
        range(min_train_months, len(unique_months) - test_months + 1, test_months)
    )
    fold_num = 0
    prev_best_params = None
    consecutive_failures = 0
    if verbose:
        console.print(f"\n[bold]{symbol}[/] - {total_folds} folds to process")

    for i in range(min_train_months, len(unique_months) - test_months + 1, test_months):
        fold_num += 1
        is_months = unique_months[0:i]
        oos_months = unique_months[i : i + test_months]

        # Compose a deterministic but unique seed per (symbol, IS-end-month).
        # Components:
        #   - GLOBAL_SEED: base seed (set in _config)
        #   - _sym_seed: stable symbol-specific offset (sum of char codes)
        #   - YYYYMM packed as year*100 + month: ensures each fold gets
        #     a fresh seed even if the same IS-month recurs for a different
        #     symbol later. XOR mixes bits; the packed-int avoids overlap
        #     between e.g. (year=2024, month=1) and (year=2023, month=13).
        _sym_seed = sum(ord(c) for c in symbol)
        is_year = is_months[-1].year
        is_month = is_months[-1].month
        fold_seed = GLOBAL_SEED ^ _sym_seed ^ (is_year * 100 + is_month)

        n_is_months = len(is_months)
        trials = _adaptive_trials(n_is_months, prev_best_params)
        purge_actual = _get_purge_days(n_is_months)
        train_end_purged = is_months[-1].end_time - timedelta(days=purge_actual)
        oos_end = oos_months[-1].end_time

        critical_cols = [
            "open", "high", "low", "close",
            "hh_20", "ll_20", "ema_200",
            "vol_pct_rank", "rvol", "atr",
            "macro_trend", "macro_vol",
        ]
        df_full = precomputed_df
        df_is = df_full[
            (df_full["time"] >= is_months[0].start_time)
            & (df_full["time"] <= train_end_purged)
        ].dropna(subset=critical_cols)

        df_oos = df_full[
            (df_full["time"] >= oos_months[0].start_time) & (df_full["time"] <= oos_end)
        ].dropna(subset=critical_cols)

        if len(df_is) < 1000 or len(df_oos) < 168:
            if verbose:
                console.print(
                    f"  [dim]fold {fold_num}/{total_folds}: skip (data too small)[/]"
                )
            continue

        # OOS contiguity check
        if len(df_oos) > 1:
            oos_diffs = df_oos["time"].diff().dropna()
            oos_max_gap = oos_diffs.max()
            if oos_max_gap > pd.Timedelta(hours=STATIC["max_allowed_gap_hours"]):
                if verbose:
                    console.print(
                        f"  [yellow]fold {fold_num}/{total_folds}: "
                        f"skip (OOS gap = {oos_max_gap.total_seconds()/3600:.0f}h "
                        f"> {STATIC['max_allowed_gap_hours']}h)[/]"
                    )
                continue

        # IS contiguity check
        if len(df_is) > 1:
            is_diffs = df_is["time"].diff().dropna()
            is_max_gap = is_diffs.max()
            if is_max_gap > pd.Timedelta(hours=STATIC["max_allowed_gap_hours"]):
                if verbose:
                    console.print(
                        f"  [yellow]fold {fold_num}/{total_folds}: "
                        f"skip (IS gap = {is_max_gap.total_seconds()/3600:.0f}h "
                        f"> {STATIC['max_allowed_gap_hours']}h)[/]"
                    )
                continue

        # Per-fold ATR% median for risk weights
        df_is_atr = df_is.dropna(subset=["atr", "close"])
        if len(df_is_atr) >= 50:
            atr_median_pct_fold = float(
                (df_is_atr["atr"] / df_is_atr["close"]).median() * 100
            )
        else:
            atr_median_pct_fold = 5.0
        atr_inv_fold = 1.0 / max(atr_median_pct_fold, 0.1)
        fold_key = str(oos_months[0])

        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=fold_seed)
        )

        # Warm-start
        if prev_best_params is not None:
            study.enqueue_trial(prev_best_params)
            rng_warm = np.random.default_rng(fold_seed ^ 0xDEF)
            ss = DEFAULTS["search_space"]
            for _ in range(3):
                perturbed = {
                    "vol_pct_thresh": float(
                        np.clip(
                            prev_best_params["vol_pct_thresh"] + rng_warm.uniform(-0.03, 0.03),
                            *ss["vol_pct_thresh"],
                        )
                    ),
                    "pullback_bars": int(
                        np.clip(
                            prev_best_params["pullback_bars"] + rng_warm.integers(-1, 2),
                            *ss["pullback_bars"],
                        )
                    ),
                    "trail_atr": float(
                        np.clip(
                            prev_best_params["trail_atr"] + rng_warm.uniform(-0.2, 0.2),
                            *ss["trail_atr"],
                        )
                    ),
                    "sl_mult": float(
                        np.clip(
                            prev_best_params["sl_mult"] + rng_warm.uniform(-0.1, 0.1),
                            *ss["sl_mult"],
                        )
                    ),
                }
                study.enqueue_trial(perturbed)

        wf_obj = WalkForwardObjective(
            df_is,
            is_avg,
            use_rvol,
            use_ema,
            fold_seed=fold_seed,
            reg_lambda=reg_lambda,
            decay_halflife_months=DEFAULTS["wfa_decay_halflife_months"],
            strategy_type=strategy_type,
            allow_long=allow_long,
            allow_short=allow_short,
            search_space=_search_space,
        )
        study.optimize(wf_obj, n_trials=trials, show_progress_bar=False)

        if study.best_value < -9000:
            if verbose:
                console.print(
                    f"  [red]fold {fold_num}/{total_folds}: all trials invalid, skip OOS[/]"
                )
            consecutive_failures += 1
            if consecutive_failures >= 2:
                if verbose:
                    console.print(
                        f"    -> {consecutive_failures} consecutive fails -> "
                        f"reset warm-start for next fold"
                    )
                prev_best_params = None
            continue

        consecutive_failures = 0
        prev_best_params = study.best_params

        best_p = study.best_params

        fold_params_list.append({
            "symbol": symbol,
            "fold": fold_num,
            "total_folds": total_folds,
            "is_start": is_months[0].start_time,
            "oos_start": oos_months[0].start_time,
            "oos_end": oos_end,
            "best_value": study.best_value,
            **best_p,
            # IS PnL array per Optuna trial in this fold (None sentinels for
            # trials that short-circuited an early return). Consumed by the
            # Hansen-literal SPA null (claim #3) via candidate aggregation.
            # Stored as ``.tolist()`` (JSON-friendly lists of float) so the
            # WFA reproducibility tests' ``params_a == params_b`` dict
            # equality on fold_params stays unambiguous -- comparing
            # numpy arrays raises "truth value of array is ambiguous".
            # The Hansen helper re-``np.asarray``s on read.
            "trial_r_nets": [
                arr.tolist() if arr is not None else None
                for arr in wf_obj.trial_r_nets
            ],
        })

        rng_oos = np.random.default_rng(fold_seed ^ 0xABCD)
        random_draws_oos = rng_oos.random(size=len(df_oos) * 2).astype(np.float64)

        # Extract pullback features if strategy is pullback_sniper
        rsi_14_oos = df_oos["rsi_14"].values if "rsi_14" in df_oos.columns else np.zeros(len(df_oos), dtype=np.float64)
        bullish_rev_oos = df_oos["bullish_reversal"].values if "bullish_reversal" in df_oos.columns else np.zeros(len(df_oos), dtype=np.int32)
        bearish_rev_oos = df_oos["bearish_reversal"].values if "bearish_reversal" in df_oos.columns else np.zeros(len(df_oos), dtype=np.int32)

        # Get best pullback_sniper params (from Optuna)
        rsi_oversold_oos = best_p.get("rsi_oversold", 30.0)
        rsi_overbought_oos = best_p.get("rsi_overbought", 70.0)

        (
            pnl_oos, idx_en, idx_ex, t_dir,
            m_trend, cb_vol, en_pr, ex_pr, sl_pcts, trend_mults,
        ) = fast_trade_loop(
            df_oos["open"].values,
            df_oos["high"].values,
            df_oos["low"].values,
            df_oos["close"].values,
            df_oos["hh_20"].values,
            df_oos["ll_20"].values,
            df_oos["ema_200"].values,
            rsi_14_oos,
            bullish_rev_oos,
            bearish_rev_oos,
            df_oos["vol_pct_rank"].values,
            df_oos["rvol"].values,
            df_oos["atr"].values,
            # Phase 2: funding_pct_rank slot (safe default).
            df_oos["funding_pct_rank"].values if "funding_pct_rank" in df_oos.columns else np.full(len(df_oos), 0.5, dtype=np.float64),
            df_oos["funding_rate"].values,
            df_oos["macro_vol"].values,
            df_oos["macro_trend"].values,
            df_oos["is_weekend"].values,
            df_oos["is_funding_hour"].values,
            strategy_type,
            best_p.get("vol_pct_thresh", 0.20),
            DEFAULTS["fixed_rvol_thresh"],
            best_p.get("pullback_bars", 5),
            best_p.get("trail_atr", 3.0),
            best_p.get("sl_mult", 1.5),
            DEFAULTS["bailout_bars"],
            0,
            STATIC["fee_taker"],
            use_rvol,
            use_ema,
            allow_long,
            allow_short,
            rsi_oversold_oos,
            rsi_overbought_oos,
            # Phase 2: funding_rate_carry thresholds (safe defaults).
            0.90, 0.40, 0.60,
            DEFAULTS["weekend_liquidity_penalty"],
            DEFAULTS["stress_test_multiplier"],
            random_draws_oos,
            DEFAULTS["trend_aligned_risk_mult"],
            DEFAULTS["trend_counter_risk_mult"],
        )

        if len(pnl_oos) > 0:
            en_times = df_oos["time"].iloc[idx_en].tolist()
            ex_times = df_oos["time"].iloc[idx_ex].tolist()
            for e_time, x_time, r, d, e_pr, x_pr, m_t, cb_v, sl_p, t_mult in zip(
                en_times, ex_times, pnl_oos, t_dir, en_pr, ex_pr,
                m_trend, cb_vol, sl_pcts, trend_mults,
            ):
                local_trades.append({
                    "entry_time": e_time,
                    "exit_time": x_time,
                    "symbol": symbol,
                    "r_net": r,
                    "entry_price": e_pr,
                    "exit_price": x_pr,
                    "trade_dir": d,
                    "sl_pct": sl_p,
                    "sl_mult": best_p.get("sl_mult", 1.5),
                    "trail_atr": best_p.get("trail_atr", 3.0),
                    "m_trend": m_t,
                    "macro_vol": cb_v,
                    "risk_weight": DEFAULTS["default_risk_per_pair"],
                    "trend_risk_mult": t_mult,
                    "atr_inv": atr_inv_fold,
                    "fold_key": fold_key,
                })

            # Phase 2.5 (Option A): also run the engine on the IS
            # data with the same winning Optuna params, so the
            # PF-weighted risk allocation can be computed on
            # IN-SAMPLE trades (which the meta-allocator hasn't
            # seen). This decouples the meta-allocator (IS-based)
            # from the strategy selector (OOS-based), avoiding
            # double-use of OOS trades for both SPA p-value and
            # risk weighting.
            if return_is_trades:
                is_trades_per_fold.append(_run_engine_on_data(
                    df=df_is,
                    best_p=best_p,
                    strategy_type=strategy_type,
                    fold_seed=fold_seed,
                    use_rvol=use_rvol,
                    use_ema=use_ema,
                    allow_long=allow_long,
                    allow_short=allow_short,
                    fold_key=fold_key,
                    atr_inv_fold=atr_inv_fold,
                    symbol=symbol,
                ))

            if verbose:
                console.print(
                    f"[bold green]✓ {symbol} fold {fold_num}/{total_folds}[/] "
                    f"WFA | IS:{is_months[0].start_time.strftime('%b %y')}"
                    f"-{is_months[-1].end_time.strftime('%b %y')} "
                    f"OOS:{oos_months[0].start_time.strftime('%b %y')}"
                    f"-{oos_months[-1].end_time.strftime('%b %y')} | "
                    f"{np.sum(pnl_oos):+.2f} R ({len(pnl_oos)} Trds)"
                )
        elif verbose:
            console.print(
                f"[bold green]✓ {symbol} fold {fold_num}/{total_folds}[/] "
                f"WFA | IS:{is_months[0].start_time.strftime('%b %y')}"
                f"-{is_months[-1].end_time.strftime('%b %y')} "
                f"OOS:{oos_months[0].start_time.strftime('%b %y')}"
                f"-{oos_months[-1].end_time.strftime('%b %y')} | no trades"
            )

    return local_trades, fold_params_list, is_trades_per_fold
