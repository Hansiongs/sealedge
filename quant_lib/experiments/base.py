"""Base classes for experiments.

An experiment combines:
- A formal Hypothesis (mechanism, boundary, success criteria)
- A PeriodConfig (train/holdout dates)
- A UniverseConfig (symbols, filters)
- A StrategyConfig (per-experiment strategy/portfolio overrides)

All dataclasses are frozen (immutable) to enforce validation at
construction time and prevent accidental mutation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

import pandas as pd

from quant_lib.audit import Hypothesis


# Type alias for strategy name (string is user-facing; int is engine-internal)
StrategyType = Literal["vol_compression", "pullback_sniper"]


# Mapping from string name to int (matches core/_engine.py constants)
STRATEGY_NAME_TO_INT: dict[str, int] = {
    "vol_compression": 0,
    "pullback_sniper": 1,
}
STRATEGY_INT_TO_NAME: dict[int, str] = {v: k for k, v in STRATEGY_NAME_TO_INT.items()}


# Pattern for valid experiment names (lowercase, digits, underscores only)
_NAME_RE = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class PeriodConfig:
    """Train/holdout period configuration.

    If holdout_start/end are None, they are auto-resolved as a
    POST-training period of length ``holdout_months`` (default 6):
    ``[train_end + 1 day, train_end + 1 day + holdout_months]``.

    The post-training convention enforces the no-peek guarantee: the
    holdout must be data that the WFA never saw, not an embargo slice
    of training data. WFA purging handles IS<->OOS contamination via
    ``_get_purge_days`` separately.

    Parameters
    ----------
    train_start, train_end : str
        Training period (YYYY-MM-DD). Inclusive on both ends.
    holdout_start, holdout_end : str, optional
        Holdout period. If None, auto-resolved from train_end + holdout_months.
    holdout_months : int
        Months of post-training holdout when auto-resolving. Default 6.
        Ignored if holdout_start/end are both explicit.
    """
    train_start: str
    train_end: str
    holdout_start: Optional[str] = None
    holdout_end: Optional[str] = None
    holdout_months: int = 6

    def __post_init__(self) -> None:
        # Validate holdout_months bounds. The frozen=True dataclass pattern
        # does NOT allow assignment in __post_init__ without object.__setattr__.
        if not isinstance(self.holdout_months, int) or self.holdout_months < 1:
            raise ValueError(
                f"holdout_months must be a positive integer (>= 1), "
                f"got {self.holdout_months!r}"
            )

    def resolve(self) -> tuple[str, str, str, str]:
        """Resolve holdout_start/end if None (post-training convention).

        Returns
        -------
        tuple of (train_start, train_end, holdout_start, holdout_end)
        """
        if self.holdout_start is None or self.holdout_end is None:
            # POST-training: [train_end + 1d, train_end + N months + 1d]
            # +1 day avoids same-day boundary ambiguity and ensures
            # hold_start is strictly AFTER train_end (validation invariant
            # in ResearchSession).
            train_end_dt = pd.Timestamp(self.train_end)
            hold_start_dt = train_end_dt + pd.Timedelta(days=1)
            hold_end_dt = hold_start_dt + pd.DateOffset(months=self.holdout_months)
            return (
                self.train_start,
                self.train_end,
                hold_start_dt.strftime("%Y-%m-%d"),
                hold_end_dt.strftime("%Y-%m-%d"),
            )
        return (
            self.train_start,
            self.train_end,
            self.holdout_start,
            self.holdout_end,
        )


@dataclass(frozen=True)
class UniverseConfig:
    """Symbol universe configuration for an experiment.

    Two criteria (both must pass) for universe eligibility (point-in-time):
    - Listing age >= ``min_age_days`` before ``train_start``
    - Trailing median daily volume >= ``min_volume_usdt``
    """
    symbols: list[str]
    min_volume_usdt: float = 50_000_000.0
    min_age_days: int = 180


@dataclass(frozen=True)
class StrategyConfig:
    """Per-experiment strategy/portfolio overrides.

    These keys are moved from the global STATIC dict in
    ``core/_config.py``, so each experiment can customize without
    affecting others. Defaults match the previous STATIC values
    for a 1:1 migration.
    """
    # Capital
    initial_capital: float = 1000.0
    leverage: float = 3.0

    # Position limits
    global_position_limit: int = 4

    # Engine parameters
    bailout_bars: int = 36
    weekend_liquidity_penalty: float = 2.0
    stress_test_multiplier: float = 2.0
    fixed_rvol_thresh: float = 2.5
    cb_hard_cooldown_hours: int = 24
    fixed_cb_threshold: float = 0.15

    # Regularization
    reg_lambda: float = 0.05

    # Trend alignment
    trend_aligned_risk_mult: float = 1.5
    trend_counter_risk_mult: float = 0.5

    # WFA
    wfa_purge_days: int = 90
    wfa_min_train_months: int = 12
    wfa_decay_halflife_months: int = 15
    wfa_test_months: int = 3
    wfa_trials_per_fold: int = 80

    # PF-based risk allocation
    pf_weight_clamp_floor: float = 0.5
    pf_weight_clamp_ceiling: float = 1.5
    pf_decay_halflife_folds: int = 2
    pf_min_trades_for_weight: int = 10

    # Per-symbol expected trades/year (optional; auto-compute if None)
    expected_trades_per_year: Optional[dict[str, int]] = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration.

    Combines a formal Hypothesis with period, universe, and strategy
    configuration. Use ``register()`` to add to the global registry.
    """
    name: str
    strategy_type: StrategyType
    hypothesis: Hypothesis
    period: PeriodConfig
    universe: UniverseConfig
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    def __post_init__(self) -> None:
        # Validate name
        if not _NAME_RE.match(self.name):
            raise ValueError(
                f"Experiment name '{self.name}' must match [a-z0-9_]+ "
                f"(lowercase letters, digits, underscores only)"
            )

        # Validate name matches hypothesis name
        if self.hypothesis.name != self.name:
            raise ValueError(
                f"Experiment name '{self.name}' must match "
                f"hypothesis.name '{self.hypothesis.name}'"
            )

        # Validate strategy_type matches hypothesis
        expected_strategy_name = STRATEGY_INT_TO_NAME.get(
            self.hypothesis.strategy_type
        )
        if expected_strategy_name is None:
            raise ValueError(
                f"Hypothesis '{self.name}' has unknown strategy_type "
                f"{self.hypothesis.strategy_type} (expected 0=vol_compression "
                f"or 1=pullback_sniper)"
            )
        if expected_strategy_name != self.strategy_type:
            raise ValueError(
                f"Experiment '{self.name}' strategy_type='{self.strategy_type}' "
                f"does not match hypothesis strategy_type "
                f"'{expected_strategy_name}' (int={self.hypothesis.strategy_type})"
            )

    @property
    def strategy_type_int(self) -> int:
        """Strategy type as int (for engine use)."""
        return STRATEGY_NAME_TO_INT[self.strategy_type]
