"""
Candidate: per-hypothesis state in a ResearchSession.

Each candidate represents one hypothesis attempt. It tracks the
state machine: hypothesis -> universe -> edge -> narrowed -> ready.

Per-hypothesis config (search_space, static_overrides, strategy_params)
is propagated through the candidate and used by the WFA engine.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional, Callable
import pandas as pd

from quant_lib.audit import Hypothesis
from quant_lib.core._features import prepare_data_with_max_time
from quant_lib.core._config import STATIC, DEFAULTS
from quant_lib.core._wfa import run_wfa_per_symbol
from quant_lib.research.best_params import pick_best_params_per_symbol
from quant_lib.core._metrics import build_daily_matrices
from quant_lib.core._spa import portfolio_spa
from quant_lib.core._portfolio import simulate_full_portfolio
from quant_lib.core._risk_allocation import (
    apply_pf_weighted_risk_allocation,
    extract_final_fold_weights,
)
from quant_lib.experiments.base import StrategyConfig
from quant_lib.research.exceptions import (
    CandidateError,
    InvalidStageTransition,
    NotReadyForCommit,
)

if TYPE_CHECKING:
    from quant_lib.research.session import ResearchSession


CandidateStage = Literal[
    "hypothesis", "universe", "edge", "narrowed", "ready"
]


@dataclass
class Candidate:
    """A single hypothesis attempt within a ResearchSession.

    State machine:
        hypothesis -> universe -> edge -> narrowed -> ready (for commit)

    Each stage produces data consumed by the next. Once `ready`,
    the candidate can be committed to the holdout via
    `commit_to_holdout(candidate)`.
    """

    hypothesis: Hypothesis
    session: "ResearchSession"

    # State
    stage: CandidateStage = "hypothesis"

    # Phase 1: Universe + Data
    eligible_symbols: list[str] = field(default_factory=list)
    precomputed_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    btc_data: Optional[pd.DataFrame] = None
    cache_hits: int = 0
    cache_misses: int = 0

    # Phase 2: Edge Testing
    all_oos_trades: list[dict] = field(default_factory=list)
    fold_params: dict[str, list[dict]] = field(default_factory=dict)
    executed_trades: list[dict] = field(default_factory=list)
    final_equity: float = 0.0
    daily_equity: dict = field(default_factory=dict)
    daily_close_matrix: dict = field(default_factory=dict)
    daily_hl_matrix: dict = field(default_factory=dict)
    risk_weights: dict = field(default_factory=dict)
    reject_reasons: dict = field(default_factory=dict)
    spa_p_value: float = 1.0
    edge_metrics: dict = field(default_factory=dict)

    # Phase 3: Narrowing
    narrowed_symbols: list[str] = field(default_factory=list)
    narrowing_rule: str = ""
    narrowing_context: str = ""

    # Frozen params (for commit)
    frozen_params: dict[str, dict] = field(default_factory=dict)

    # Per-experiment strategy config (PF-based risk allocation knobs)
    # Defaults to StrategyConfig() if not provided.
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    # Cached report (set after reporting.print_candidate_report)
    report: Optional[object] = None

    # ──────────────────────────────────────────────────────────────────
    # State machine methods
    # ──────────────────────────────────────────────────────────────────

    def _assert_stage_at_least(self, required: CandidateStage) -> None:
        """Raise if current stage is before required."""
        order = ["hypothesis", "universe", "edge", "narrowed", "ready"]
        cur_idx = order.index(self.stage)
        req_idx = order.index(required)
        if cur_idx < req_idx:
            raise InvalidStageTransition(
                f"Stage '{self.stage}' cannot access '{required}' outputs. "
                f"Run the appropriate phase first.",
                phase=self.hypothesis.name,
            )

    def _set_stage(self, new_stage: CandidateStage) -> None:
        """Transition to new stage with validation."""
        order = ["hypothesis", "universe", "edge", "narrowed", "ready"]
        cur_idx = order.index(self.stage)
        new_idx = order.index(new_stage)
        if new_idx < cur_idx:
            raise InvalidStageTransition(
                f"Cannot transition from '{self.stage}' to '{new_stage}' (backward).",
                phase=self.hypothesis.name,
            )
        if new_idx > cur_idx + 1:
            raise InvalidStageTransition(
                f"Cannot skip from '{self.stage}' to '{new_stage}'.",
                phase=self.hypothesis.name,
            )
        self.stage = new_stage

    # ──────────────────────────────────────────────────────────────────
    # Phase methods (white testing stages)
    # ──────────────────────────────────────────────────────────────────

    def run_universe(
        self,
        min_volume_usdt: float = 50_000_000.0,
        min_age_days: int = 180,
    ) -> None:
        """Phase 1: fetch data, select universe, compute features.

        - Loads klines for each candidate symbol (cache-first)
        - Selects eligible symbols (volume + age criteria)
        - Computes all features (leakage-aware, with strategy_type dispatch)
        - Caches per-asset data and precomputed features
        """
        if self.stage != "hypothesis":
            raise InvalidStageTransition(
                f"Cannot run_universe from stage '{self.stage}'",
                phase=self.hypothesis.name,
            )

        session = self.session
        train_start, train_end = session.training_period
        strategy_type = self.hypothesis.strategy_type

        # Fetch BTC extended data
        self.btc_data = session.cache.get_klines(
            "BTCUSDT", "1h", session.btc_data_start, train_end
        )

        # Fetch per-symbol raw data and apply universe filter on RAW (point-in-time)
        # The filter needs to see history before train_start, but the backtest
        # itself only uses data from train_start onwards. So we fetch the full
        # historical range (btc_data_start to train_end), filter, and only
        # then slice to the backtest range and compute features.
        raw_data: dict[str, pd.DataFrame] = {}
        for sym in session.symbols:
            sym_start = session.btc_data_start if sym == "BTCUSDT" else train_start
            raw_data[sym] = session.cache.get_klines(
                sym, "1h", session.btc_data_start, train_end
            )

        # Universe selection on raw data (full history available)
        start_dt = pd.Timestamp(train_start)
        self.eligible_symbols = []
        for sym in session.symbols:
            if sym not in raw_data:
                continue
            if not self._passes_universe_filter(
                raw_data[sym], start_dt, min_volume_usdt, min_age_days
            ):
                continue
            self.eligible_symbols.append(sym)

        if not self.eligible_symbols:
            raise CandidateError(
                f"No symbols passed universe selection for {self.hypothesis.name} "
                f"(min_volume_usdt={min_volume_usdt}, min_age_days={min_age_days})",
                phase=self.hypothesis.name,
            )

        # Compute features for eligible symbols only (slice to backtest range)
        for sym in self.eligible_symbols:
            sym_start = session.btc_data_start if sym == "BTCUSDT" else train_start
            df_raw_for_features = raw_data[sym][
                raw_data[sym]["time"] >= pd.Timestamp(sym_start)
            ]
            fund = session.cache.get_funding(sym, train_start, train_end)
            self.precomputed_data[sym] = prepare_data_with_max_time(
                df_raw=df_raw_for_features,
                df_btc_raw=self.btc_data,
                df_funding_raw=fund,
                max_time=pd.Timestamp(train_end),
                strategy_type=strategy_type,
            )

        self.cache_hits = session.cache._hits
        self.cache_misses = session.cache._misses

        self._set_stage("universe")

        session.journal.log_run(
            description=(
                f"Phase 1 done: {len(self.eligible_symbols)}/{len(session.symbols)} symbols "
                f"(min_vol=${min_volume_usdt:,.0f}, min_age={min_age_days}d)"
            ),
            category="ablation",
        )

    def _passes_universe_filter(
        self,
        df: pd.DataFrame,
        start_dt: pd.Timestamp,
        min_volume_usdt: float,
        min_age_days: int,
    ) -> bool:
        """Apply mechanical, point-in-time universe filter to one symbol.

        Two criteria (both must pass):
        1. Age: first available bar is at least min_age_days before start_dt.
        2. Volume: median daily volume (in USDT) over the 90 days before
           start_dt is at least min_volume_usdt.

        No strategy performance involved -- this is the Phase 1 selection
        stage. Point-in-time only: we look at data available at start_dt,
        not at the future.
        """
        if df is None or len(df) == 0:
            return False
        if "time" not in df.columns or "volume" not in df.columns or "close" not in df.columns:
            return False

        times = pd.to_datetime(df["time"])
        first_bar = times.min()
        if pd.isna(first_bar):
            return False
        age_at_start = (start_dt - first_bar).days
        if age_at_start < min_age_days:
            return False

        # Volume: rolling 24h sum, then multiply by ref close price
        # to get USDT, then take median over the 90-day lookback window.
        mask = (times >= start_dt - pd.Timedelta(days=90)) & (times <= start_dt)
        df_lookback = df.loc[mask, ["volume", "close"]]
        if len(df_lookback) < 24:
            return False
        daily_vol_units = df_lookback["volume"].rolling(24).sum().dropna()
        if len(daily_vol_units) == 0:
            return False
        ref_price = float(df_lookback["close"].iloc[-1])
        if ref_price <= 0:
            return False
        daily_vol_usdt = daily_vol_units * ref_price
        if float(daily_vol_usdt.median()) < min_volume_usdt:
            return False
        return True

    def run_edge_testing(
        self,
        n_spa_iters: int = 2000,
        use_rvol: bool = True,
        use_ema: bool = True,
    ) -> dict:
        """Phase 2: WFA per symbol, portfolio sim, SPA.

        Returns edge_metrics dict.
        """
        self._assert_stage_at_least("universe")

        if self.stage != "universe":
            raise InvalidStageTransition(
                f"Cannot run_edge_testing from stage '{self.stage}'",
                phase=self.hypothesis.name,
            )

        session = self.session
        train_start, train_end = session.training_period
        strategy_type = self.hypothesis.strategy_type
        sp = self.hypothesis.merged_strategy_params()
        search_space = self.hypothesis.merged_search_space()
        allow_long = 1 if sp.get("allow_long", True) else 0
        allow_short = 1 if sp.get("allow_short", True) else 0

        # WFA per symbol
        for sym in self.eligible_symbols:
            trades, fold_params = run_wfa_per_symbol(
                sym,
                self.precomputed_data[sym],
                use_rvol=use_rvol,
                use_ema=use_ema,
                reg_lambda=DEFAULTS["reg_lambda"],
                strategy_type=strategy_type,
                allow_long=allow_long,
                allow_short=allow_short,
                search_space=search_space,
                verbose=False,
            )
            self.all_oos_trades.extend(trades)
            if fold_params:
                self.fold_params[sym] = fold_params

        if not self.all_oos_trades:
            raise CandidateError(
                f"Zero OOS trades for {self.hypothesis.name}",
                phase=self.hypothesis.name,
            )

        # Compute frozen params (best across all WFA folds per symbol)
        # Q1 decision: pick fold with highest PSR per symbol, not last fold.
        # This matches live trading workflow: optuna on all history -> use best.
        self.frozen_params = pick_best_params_per_symbol(
            self.fold_params, strategy_type=strategy_type
        )

        # Per-fold PF-weighted risk allocation (X1 scheme).
        # Mutates t["risk_weight"] on each OOS trade; the portfolio sim
        # below reads per-trade risk_weight directly.
        #
        # B0.4 fix: also capture the per-fold summary and build
        # ``self.risk_weights`` from the LAST fold's allocation. The
        # last fold's weights use the most prior data and are the
        # canonical carry-over to the holdout. Previously this
        # attribute was reset to ``{}`` here, which caused every
        # holdout trade to silently use the 0.01 fallback in
        # ``commit_to_holdout`` regardless of the WFA allocation.
        risk_summary = apply_pf_weighted_risk_allocation(
            self.all_oos_trades,
            halflife_folds=self.strategy.pf_decay_halflife_folds,
            clamp_floor=self.strategy.pf_weight_clamp_floor,
            clamp_ceiling=self.strategy.pf_weight_clamp_ceiling,
            min_trades=self.strategy.pf_min_trades_for_weight,
            baseline_per_symbol=DEFAULTS["default_risk_per_pair"],
            n_total_symbols=len(self.eligible_symbols),
        )
        # Map every eligible symbol to a per-symbol weight, falling
        # back to the default for symbols not present in the last
        # fold (they didn't trade there). Empty dict if no folds ran
        # (very short training data) -- commit.py treats this as
        # "use defaults for all".
        self.risk_weights = extract_final_fold_weights(
            risk_summary=risk_summary,
            eligible_symbols=list(self.eligible_symbols),
            default_weight=DEFAULTS["default_risk_per_pair"],
        )

        # Daily matrices
        self.daily_close_matrix, self.daily_hl_matrix = build_daily_matrices(
            self.eligible_symbols, self.precomputed_data
        )

        # Portfolio simulation
        self.final_equity, self.daily_equity, self.executed_trades, self.reject_reasons = \
            simulate_full_portfolio(
                trades=self.all_oos_trades,
                initial_cash=session.initial_capital,
                leverage=DEFAULTS["leverage"],
                mm_pct=STATIC["maintenance_margin_pct"],
                position_limit=DEFAULTS["global_position_limit"],
                cb_hard_cooldown_hours=DEFAULTS["cb_hard_cooldown_hours"],
                fixed_cb_threshold=DEFAULTS["fixed_cb_threshold"],
                daily_close_matrix=self.daily_close_matrix,
                asset_risk_weights=self.risk_weights,
                end_date=train_end,
                liquidation_fee_pct=STATIC["liquidation_fee_pct"],
                daily_hl_matrix=self.daily_hl_matrix,
            )

        # SPA test
        asset_data = {}
        for sym in self.eligible_symbols:
            df = self.precomputed_data[sym].dropna(
                subset=["close", "atr", "funding_rate", "high", "low", "macro_trend"]
            )
            asset_data[sym] = df[
                ["time", "close", "high", "low", "atr",
                 "funding_rate", "is_weekend", "is_funding_hour", "macro_trend"]
            ]

        _, _, p_value = portfolio_spa(
            observed_trades=self.all_oos_trades,
            asset_data=asset_data,
            daily_close_matrix=self.daily_close_matrix,
            end_date=train_end,
            daily_hl_matrix=self.daily_hl_matrix,
            n_iters=n_spa_iters,
            initial_capital=session.initial_capital,
            leverage=DEFAULTS["leverage"],
            mm_pct=STATIC["maintenance_margin_pct"],
            position_limit=DEFAULTS["global_position_limit"],
            cb_hard_cooldown_hours=DEFAULTS["cb_hard_cooldown_hours"],
            fixed_cb_threshold=DEFAULTS["fixed_cb_threshold"],
            rng_seed=42,
            verbose=False,
            liquidation_fee_pct=STATIC["liquidation_fee_pct"],
            fee_taker=STATIC["fee_taker"],
            stress_mult=DEFAULTS["stress_test_multiplier"],
            weekend_penalty=DEFAULTS["weekend_liquidity_penalty"],
            asset_risk_weights=self.risk_weights,
        )

        self.spa_p_value = p_value
        self.edge_metrics = {
            "n_oos_trades": self.n_oos_trades,
            "n_executed": self.n_executed,
            "n_rejected": self.n_rejected,
            "final_equity": self.final_equity,
            "spa_p_value": p_value,
        }

        self._set_stage("edge")

        session.journal.log_run(
            description=f"Phase 2: {self.n_oos_trades} OOS trades, "
                        f"${self.final_equity:,.0f} equity, SPA p={p_value:.4f}",
            category="explore",
            params_snapshot={
                "n_oos": self.n_oos_trades,
                "n_exec": self.n_executed,
                "equity": round(self.final_equity, 2),
                "spa_p": round(p_value, 4),
            },
        )

        return self.edge_metrics

    def run_narrowing(
        self,
        rule: Optional[Callable] = None,
    ) -> None:
        """Phase 3: apply narrowing rule (context-aware from Phase 2).

        If no rule provided: keep full universe (broad-weak default).
        """
        self._assert_stage_at_least("edge")

        if self.stage != "edge":
            raise InvalidStageTransition(
                f"Cannot run_narrowing from stage '{self.stage}'",
                phase=self.hypothesis.name,
            )

        session = self.session

        if rule is None:
            # Default: keep all eligible symbols
            self.narrowed_symbols = list(self.eligible_symbols)
            self.narrowing_rule = "full_universe_default"
            self.narrowing_context = "Default: keep all eligible symbols"
        else:
            try:
                self.narrowed_symbols = rule(self.eligible_symbols, self.precomputed_data)
            except Exception as e:
                raise CandidateError(
                    f"Narrowing rule error: {e}",
                    phase=self.hypothesis.name,
                )
            if not self.narrowed_symbols:
                raise CandidateError(
                    "Narrowing resulted in empty universe.",
                    phase=self.hypothesis.name,
                )
            self.narrowing_rule = rule.__name__ if hasattr(rule, "__name__") else str(rule)
            self.narrowing_context = f"User rule: {self.narrowing_rule}"

        self._set_stage("narrowed")

        session.journal.log_run(
            description=f"Phase 3: {len(self.narrowed_symbols)}/{len(self.eligible_symbols)} symbols",
            category="ablation",
        )

    # ──────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────

    @property
    def is_ready_for_commit(self) -> bool:
        """Check if candidate has been marked ready for commit.

        The "ready" stage is only reached after explicit ``mark_ready()``,
        which validates that narrowing and frozen params are populated.
        This prevents accidental commits on candidates that haven't been
        finalized.
        """
        return (
            self.stage == "ready"
            and bool(self.narrowed_symbols)
            and bool(self.frozen_params)
        )

    def mark_ready(self) -> None:
        """Transition to the terminal 'ready' stage after validation.

        Idempotent: safe to call multiple times. Raises
        ``NotReadyForCommit`` if the candidate hasn't completed narrowing,
        has no frozen params, or hasn't met the minimum training months
        requirement.

        The ``min_train_months`` guard is enforced here (defense in depth)
        and again in ``commit_to_holdout`` (commit.py). A caller that
        creates a Candidate outside the normal flow (e.g. bypassing
        ``run_edge_testing``) is caught early rather than failing at
        commit time.
        """
        if self.stage == "ready":
            return
        if self.stage != "narrowed":
            raise InvalidStageTransition(
                f"Cannot mark_ready from stage '{self.stage}' "
                f"(must be 'narrowed').",
                phase=self.hypothesis.name,
            )
        if not self.narrowed_symbols:
            raise NotReadyForCommit(
                f"Candidate '{self.hypothesis.name}' has no narrowed_symbols.",
                phase=self.hypothesis.name,
            )
        if not self.frozen_params:
            raise NotReadyForCommit(
                f"Candidate '{self.hypothesis.name}' has no frozen_params.",
                phase=self.hypothesis.name,
            )
        # Phase 3.7 E4 / Phase 4: min_train_months enforcement.
        # The hypothesis specifies the minimum training months required.
        # If the actual training period is shorter, refuse to mark ready.
        # This duplicates the check in commit_to_holdout (commit.py:189-203)
        # as defense-in-depth: callers that go directly to mark_ready()
        # without going through the full commit path are still protected.
        train_start, train_end = self.session.training_period
        n_train_months = (
            (pd.Timestamp(train_end) - pd.Timestamp(train_start)).days / 30.44
        )
        if n_train_months < self.hypothesis.min_train_months:
            raise NotReadyForCommit(
                f"Candidate '{self.hypothesis.name}': training period "
                f"({n_train_months:.1f}mo) is shorter than hypothesis "
                f"min_train_months ({self.hypothesis.min_train_months}mo). "
                f"Adjust the training period in the experiment config or "
                f"relax min_train_months in the hypothesis.",
            )
        self._set_stage("ready")

    @property
    def n_oos_trades(self) -> int:
        return len(self.all_oos_trades)

    @property
    def n_executed(self) -> int:
        return len(self.executed_trades)

    @property
    def n_rejected(self) -> int:
        return self.n_oos_trades - self.n_executed

    @property
    def equity_change_pct(self) -> float:
        if self.final_equity > 0:
            return (self.final_equity - self.session.initial_capital) / self.session.initial_capital * 100
        return 0.0

    def assert_ready(self) -> None:
        """Raise NotReadyForCommit if not ready.

        Note: this only validates preconditions (narrowed + has data).
        To actually commit, call :meth:`mark_ready` first to transition
        into the terminal 'ready' stage.
        """
        if not self.narrowed_symbols:
            raise NotReadyForCommit(
                f"Candidate '{self.hypothesis.name}' not ready: "
                f"narrowed_symbols is empty.",
                phase=self.hypothesis.name,
            )
        if not self.frozen_params:
            raise NotReadyForCommit(
                f"Candidate '{self.hypothesis.name}' not ready: "
                f"frozen_params is empty.",
                phase=self.hypothesis.name,
            )
        if self.stage != "ready":
            raise NotReadyForCommit(
                f"Candidate '{self.hypothesis.name}' not ready: "
                f"stage={self.stage} (must be 'ready' -- call mark_ready() first).",
                phase=self.hypothesis.name,
            )

    # ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Candidate(name={self.hypothesis.name}, "
            f"strategy={self.hypothesis.strategy_name}, "
            f"stage={self.stage}, "
            f"trades={self.n_oos_trades}/{self.n_executed})"
        )
