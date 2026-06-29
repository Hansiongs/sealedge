"""
Phase 0: Formal Hypothesis Definition.

Framework principle: the hypothesis must be written BEFORE seeing any data,
and must contain mechanism, boundary conditions, and success criteria.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Optional


class StrategyType(IntEnum):
    """Type-safe identifier for the strategy variant a hypothesis uses.

    Using this enum instead of raw integers (0/1) gives:

    * **Readability**: ``StrategyType.PULLBACK_SNIPER`` vs the magic
      number ``1``.
    * **Type checking**: mypy catches accidental swaps between
      the two strategies.
    * **Self-documentation**: the int values are tied to the
      engine's contract via the enum members, so a rename surfaces
      all call sites immediately.

    The int values are part of the engine's public ABI (``fast_trade_loop``
    takes ``strategy_type: int``) so they MUST stay stable. To add a
    new strategy, append a new member here AND add a matching
    ``STRATEGY_*`` constant in ``core/_engine.py`` AND update
    ``STRATEGY_NAME_TO_INT`` in ``experiments/base.py``.

    Examples
    --------
    >>> from quant_lib.audit import Hypothesis, StrategyType
    >>> h = Hypothesis(name="v1", mechanism="...", boundary_conditions="...",
    ...                success_criteria="...", strategy_type=StrategyType.PULLBACK_SNIPER)
    >>> h.strategy_type
    1
    >>> h.strategy_name
    'pullback_sniper'
    """

    VOL_COMPRESSION = 0
    PULLBACK_SNIPER = 1


# Backwards-compat aliases. Existing code (and engine.py / __init__.py
# re-exports) uses these module-level int constants. The values are
# explicitly typed as ``int`` so the engine's Numba-compiled function,
# which expects a raw int, accepts them as-is without conversion.
STRATEGY_VOL_COMPRESSION: int = int(StrategyType.VOL_COMPRESSION)
STRATEGY_PULLBACK_SNIPER: int = int(StrategyType.PULLBACK_SNIPER)


@dataclass(frozen=True)
class Hypothesis:
    """Formal strategy hypothesis -- written before touching any data.

    Parameters
    ----------
    name : str
        Short identifier for this hypothesis (e.g. "vol_breakout_v1").
    mechanism : str
        Logical explanation of why this strategy should have an edge.
        Must be mechanism-based, NOT "because the backtest looks good".
    boundary_conditions : str
        Conditions under which this hypothesis is expected to fail.
    success_criteria : str
        Pre-defined success metrics (e.g. "SPA p < 0.15, PF > 1.5").
        These must be set BEFORE seeing results.
    entry_logic : str
        Description of entry conditions.
    exit_logic : str
        Description of exit conditions.
    universe_rules : str, optional
        Criteria for universe selection (Phase 1).
    timestamp : datetime, optional
        Auto-set to UTC time of creation.
    git_commit : str, optional
        Git commit hash at hypothesis creation time. Should be populated
        by the caller via ``git rev-parse HEAD``.
    strategy_type : int or StrategyType, optional
        Strategy identifier. Accepts either the ``StrategyType`` enum
        value (``StrategyType.VOL_COMPRESSION`` /
        ``StrategyType.PULLBACK_SNIPER``) or the legacy int constant
        (``STRATEGY_VOL_COMPRESSION = 0`` /
        ``STRATEGY_PULLBACK_SNIPER = 1``). Stored as an int internally
        so the value is JSON-serializable and matches the engine's
        raw-int parameter signature.
    search_space : dict, optional
        Per-hypothesis Optuna search space. If None, uses STATIC defaults.
    static_overrides : dict, optional
        Per-hypothesis STATIC config overrides (e.g., custom bailout_bars).
    strategy_params : dict, optional
        Strategy-specific params (allow_long, allow_short, rsi_period, etc).
    min_train_months : int, optional
        Minimum training months for WFA. Default 12.

    Example
    -------
    >>> hyp = Hypothesis(
    ...     name="vol_compression_breakout",
    ...     mechanism="Volatility compression followed by volume breakout "
    ...               "generates intraday momentum in liquid crypto futures",
    ...     boundary_conditions="Fails in strong trend regimes without "
    ...                        "pullback (2021 bull run)",
    ...     success_criteria="SPA p < 0.15, PF > 1.3, min 30 trades",
    ...     entry_logic="vol_pct_rank < 0.20 + close breaks HH_20 + rvol > 2.5",
    ...     exit_logic="Trailing stop at ATR × 3.0, bailout at 36 bars",
    ...     strategy_type=0,
    ...     search_space={"vol_pct_thresh": (0.10, 0.40), ...},
    ... )
    """

    name: str
    mechanism: str
    boundary_conditions: str
    success_criteria: str
    entry_logic: str
    exit_logic: str
    universe_rules: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    git_commit: Optional[str] = None
    # Default is the int form (0) for JSON-serializable round-trips;
    # the StrategyType enum above lets callers pass either form.
    strategy_type: int = 0
    search_space: Optional[dict] = None
    static_overrides: Optional[dict] = None
    strategy_params: Optional[dict] = None
    min_train_months: int = 12

    def summary(self) -> str:
        """Return a one-line summary string."""
        return (
            f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}] "
            f"{self.name}: {self.mechanism[:80]}..."
        )

    def validate(self) -> list:
        """Check hypothesis completeness. Returns list of missing fields."""
        missing = []
        for field_name in ["mechanism", "boundary_conditions", "success_criteria"]:
            if not getattr(self, field_name):
                missing.append(field_name)
        return missing

    def to_dict(self) -> dict:
        """Serialize to dict for logging/storage."""
        return {
            "name": self.name,
            "mechanism": self.mechanism,
            "boundary_conditions": self.boundary_conditions,
            "success_criteria": self.success_criteria,
            "entry_logic": self.entry_logic,
            "exit_logic": self.exit_logic,
            "universe_rules": self.universe_rules,
            "timestamp": self.timestamp.isoformat(),
            "git_commit": self.git_commit,
            "strategy_type": self.strategy_type,
            "search_space": self.search_space,
            "static_overrides": self.static_overrides,
            "strategy_params": self.strategy_params,
            "min_train_months": self.min_train_months,
        }

    @property
    def strategy_name(self) -> str:
        """Human-readable strategy name."""
        # Compare against the IntEnum's underlying int value so
        # ``strategy_type`` can be either a raw int (legacy call
        # sites) or a ``StrategyType`` member (new call sites).
        if self.strategy_type == StrategyType.VOL_COMPRESSION:
            return "vol_compression_breakout"
        elif self.strategy_type == StrategyType.PULLBACK_SNIPER:
            return "pullback_sniper"
        return f"unknown_{self.strategy_type}"

    def merged_static_overrides(self) -> dict:
        """Get static_overrides or empty dict."""
        return dict(self.static_overrides or {})

    def merged_search_space(self) -> dict:
        """Get search_space or empty dict."""
        return dict(self.search_space or {})

    def merged_strategy_params(self) -> dict:
        """Get strategy_params with defaults."""
        defaults = {"allow_long": True, "allow_short": True}
        merged = {**defaults, **(self.strategy_params or {})}
        return merged


# Strategy type constants (STRATEGY_VOL_COMPRESSION / STRATEGY_PULLBACK_SNIPER)
# are defined near the top of this module as IntEnum-backed aliases for
# backwards compatibility. See ``StrategyType`` above.

# Default search spaces per strategy
DEFAULT_VOL_COMPRESSION_SEARCH_SPACE = {
    "vol_pct_thresh": (0.10, 0.40),
    "pullback_bars": (3, 8),
    "trail_atr": (1.5, 5.0),
    "sl_mult": (1.0, 3.0),
}

DEFAULT_PULLBACK_SNIPER_SEARCH_SPACE = {
    "vol_pct_thresh": (0.10, 0.40),  # unused but kept for compat
    "pullback_bars": (3, 8),         # unused but kept for compat
    "trail_atr": (1.5, 5.0),
    "sl_mult": (1.0, 3.0),
    "rsi_oversold": (25, 35),
    "rsi_overbought": (65, 75),
}


def for_vol_compression(
    name: str,
    mechanism: str,
    boundary_conditions: str,
    success_criteria: str,
    entry_logic: str = "vol_pct_rank < thresh + close breaks HH_20/LL_20 + pullback",
    exit_logic: str = "Trailing stop at ATR x trail_atr, bailout at 36 bars",
    universe_rules: str | None = None,
    search_space: dict | None = None,
    static_overrides: dict | None = None,
    strategy_params: dict | None = None,
    min_train_months: int = 12,
) -> Hypothesis:
    """Factory: Hypothesis for vol_compression_breakout strategy."""
    return Hypothesis(
        name=name,
        mechanism=mechanism,
        boundary_conditions=boundary_conditions,
        success_criteria=success_criteria,
        entry_logic=entry_logic,
        exit_logic=exit_logic,
        universe_rules=universe_rules,
        strategy_type=STRATEGY_VOL_COMPRESSION,
        search_space=search_space or DEFAULT_VOL_COMPRESSION_SEARCH_SPACE,
        static_overrides=static_overrides,
        strategy_params=strategy_params,
        min_train_months=min_train_months,
    )


def for_pullback_sniper(
    name: str,
    mechanism: str,
    boundary_conditions: str,
    success_criteria: str,
    entry_logic: str = "RSI < oversold + bullish_reversal candle (or mirror for short)",
    exit_logic: str = "Trailing stop OR TP at hh_20/ll_20 OR bailout at 36 bars",
    universe_rules: str | None = None,
    search_space: dict | None = None,
    static_overrides: dict | None = None,
    strategy_params: dict | None = None,
    min_train_months: int = 12,
) -> Hypothesis:
    """Factory: Hypothesis for pullback_sniper strategy."""
    return Hypothesis(
        name=name,
        mechanism=mechanism,
        boundary_conditions=boundary_conditions,
        success_criteria=success_criteria,
        entry_logic=entry_logic,
        exit_logic=exit_logic,
        universe_rules=universe_rules,
        strategy_type=STRATEGY_PULLBACK_SNIPER,
        search_space=search_space or DEFAULT_PULLBACK_SNIPER_SEARCH_SPACE,
        static_overrides=static_overrides,
        strategy_params=strategy_params,
        min_train_months=min_train_months,
    )
