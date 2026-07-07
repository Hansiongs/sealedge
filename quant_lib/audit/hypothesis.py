"""
Phase 0: Formal Hypothesis Definition.

Framework principle: the hypothesis must be written BEFORE seeing any data,
and must contain mechanism, boundary conditions, and success criteria.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Optional

# Sprint 2 fix: import the single-source-of-truth strategy type
# constants from ``core/_config.py`` instead of redeclaring. The
# int values match the ``StrategyType`` enum below and are part of
# the engine's public ABI (``fast_trade_loop`` takes ``strategy_type:
# int``). ``audit -> core._config`` is a safe direction: ``_config``
# is a leaf module with no quant_lib imports of its own.
from quant_lib.core._config import (  # noqa: I001, E402
    STRATEGY_VOL_COMPRESSION,
    STRATEGY_PULLBACK_SNIPER,
    STRATEGY_FUNDING_RATE_CARRY,
)


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
    FUNDING_RATE_CARRY = 2


# Note: ``STRATEGY_VOL_COMPRESSION`` and ``STRATEGY_PULLBACK_SNIPER``
# are imported from ``core/_config.py`` at the top of this module.
# They are NOT redeclared here -- doing so would shadow the import
# and defeat the Sprint 2 single-source-of-truth fix.


@dataclass(frozen=True)
class Hypothesis:
    """Formal strategy hypothesis -- written before touching any data.

    The Hypothesis is the contract between the researcher's claim ("I
    believe X causes Y in market Z") and the framework's verification
    pipeline (WFA + SPA + FDR). Writing it BEFORE running experiments
    is the framework's primary defense against hindsight bias.

    Parameters
    ----------
    name : str
        Short identifier (e.g. ``"vol_breakout_v1"``). Must match
        ``[a-z0-9_]+`` regex. Used as filename, registry key, and
        CLI experiment name.
    mechanism : str
        **Required narrative field.** Logical explanation of why this
        strategy should have an edge. Must be mechanism-based
        ("X causes Y in market Z because..."), NOT "the backtest
        looks good" or "I saw a pattern". The mechanism is what
        you defend in the experiment journal.
    boundary_conditions : str
        **Required narrative field.** Conditions under which the
        hypothesis is expected to FAIL. Examples: "fails in strong
        trend regimes without pullback", "fails on illiquid pairs
        < 50M USD daily volume", "fails during high funding-rate
        environments (perpetual basis blowout)".
    success_criteria : str
        **Required narrative field.** Pre-defined success metrics,
        set BEFORE seeing results. Must be specific. Examples:
        ``"SPA p < 0.15, PF > 1.3, ≥ 30 OOS trades"``,
        ``"PSR > 0.95 per symbol, total trades ≥ 50"``.
    entry_logic : str
        **Required narrative field.** Human-readable description of
        entry conditions. The actual numeric logic is in the engine;
        this field is for the journal / reviewer.
    exit_logic : str
        **Required narrative field.** Same as ``entry_logic`` but for
        exit conditions (trailing stop, bailout, TP bracket).
    universe_rules : str, optional
        Criteria for universe selection (Phase 1). Example:
        ``"Top 20 by 30-day USD volume, exclude stablecoins, age ≥ 180d"``.
    timestamp : datetime, optional
        Auto-set to UTC time of creation. Preserved through
        serialization (used by ExperimentLog).
    git_commit : str, optional
        Git commit hash at hypothesis creation time. Should be
        populated by the caller via ``git rev-parse HEAD`` for
        reproducibility auditing.
    strategy_type : int or StrategyType, optional
        Strategy identifier. Accepts ``StrategyType.VOL_COMPRESSION``
        (0) or ``StrategyType.PULLBACK_SNIPER`` (1). Stored as int
        internally so the value is JSON-serializable and matches
        the engine's raw-int parameter signature.
    search_space : dict, optional
        Per-hypothesis Optuna search space (parameter_name →
        ``(low, high)`` tuple). If None, the framework uses
        ``DEFAULT_VOL_COMPRESSION_SEARCH_SPACE`` or
        ``DEFAULT_PULLBACK_SNIPER_SEARCH_SPACE`` based on
        ``strategy_type``.
    static_overrides : dict, optional
        Per-hypothesis STATIC config overrides (e.g.,
        ``{"fee_taker": 0.001}`` for VIP-tier fee model).
    strategy_params : dict, optional
        Strategy-specific params. Recognized keys:
        - ``allow_long`` (bool, default True)
        - ``allow_short`` (bool, default True)
        - ``rsi_period`` (int, default 14, pullback_sniper only)
    min_train_months : int, optional
        Minimum training months for WFA. Default 12.

    Examples
    --------
    **vol_compression breakout** (using factory helper):

    >>> from quant_lib.audit import for_vol_compression
    >>> hyp = for_vol_compression(
    ...     name="btc_breakout_v2",
    ...     mechanism=(
    ...         "Volatility compression followed by volume-confirmed "
    ...         "breakout of the 20-bar high generates intraday momentum "
    ...         "in liquid crypto perpetuals. Compression represents "
    ...         "absorption of supply; the breakout is the release."
    ...     ),
    ...     boundary_conditions=(
    ...         "Fails in low-vol accumulation regimes where no breakout "
    ...         "follows compression. Fails on illiquid pairs (< 50M USD "
    ...         "daily volume) where the breakout is noise. Fails during "
    ...         "extreme funding environments where basis blowouts distort "
    ...         "the price action."
    ...     ),
    ...     success_criteria="SPA p < 0.10, PF > 1.4, ≥ 40 OOS trades",
    ...     entry_logic=(
    ...         "vol_pct_rank < 0.15 + close > HH_20 + rvol > 3.0 + "
    ...         "EMA200 confirmation"
    ...     ),
    ...     exit_logic=(
    ...         "Trailing stop at ATR × 4.0, bailout at 48 bars, no TP "
    ...         "bracket (vol compression rides the full move)"
    ...     ),
    ...     universe_rules="BTC, ETH, SOL, AVAX, MATIC — age ≥ 365d, "
    ...                    "30d volume ≥ 100M USD",
    ...     search_space={
    ...         "vol_pct_thresh": (0.10, 0.25),  # tighter than default
    ...         "trail_atr": (2.0, 5.0),
    ...         "sl_mult": (1.5, 2.5),
    ...         "pullback_bars": (3, 6),
    ...     },
    ...     min_train_months=24,  # 2 years of training data
    ... )

    **pullback_sniper** (RSI + reversal):

    >>> from quant_lib.audit import for_pullback_sniper
    >>> hyp = for_pullback_sniper(
    ...     name="rsi_reversal_v1",
    ...     mechanism=(
    ...         "RSI oversold/overbought extremes combined with "
    ...         "bullish/bearish reversal candles predict short-term "
    ...         "mean-reversion in crypto perpetuals. The RSI extreme "
    ...         "indicates exhaustion; the reversal candle confirms "
    ...         "the turning point."
    ...     ),
    ...     boundary_conditions=(
    ...         "Fails in strong trends where RSI stays extreme for "
    ...         "extended periods (no reversion). Fails on illiquid "
    ...         "pairs where the reversal candle is unreliable. "
    ...         "Fails during high-impact news events where RSI "
    ...         "extremes are not exhaustion but continuation."
    ...     ),
    ...     success_criteria="SPA p < 0.15, PF > 1.3, ≥ 50 OOS trades",
    ...     entry_logic=(
    ...         "RSI < 25 + bullish_reversal candle (or mirror for short: "
    ...         "RSI > 75 + bearish_reversal)"
    ...     ),
    ...     exit_logic=(
    ...         "TP bracket at HH_20/LL_20 (mean-reversion target) OR "
    ...         "trailing stop at ATR × 3.0 OR bailout at 36 bars"
    ...     ),
    ...     universe_rules="BTC, ETH, SOL — age ≥ 365d",
    ...     search_space={
    ...         "rsi_oversold": (20, 30),
    ...         "rsi_overbought": (70, 80),
    ...         "trail_atr": (2.0, 4.0),
    ...         "sl_mult": (1.5, 2.5),
    ...     },
    ...     strategy_params={"rsi_period": 14, "allow_short": True},
    ... )

    **Custom hypothesis** (advanced — direct Hypothesis constructor):

    >>> from quant_lib.audit import Hypothesis, StrategyType
    >>> hyp = Hypothesis(
    ...     name="hybrid_v1",
    ...     mechanism="Hybrid: pullback setup with vol confirmation",
    ...     boundary_conditions="Fails when both signals disagree",
    ...     success_criteria="SPA p < 0.10, PF > 1.5",
    ...     entry_logic="pullback_bars + vol_pct_rank < 0.20",
    ...     exit_logic="ATR × 3.0 trail, bailout 36 bars",
    ...     strategy_type=StrategyType.PULLBACK_SNIPER,
    ...     search_space={"vol_pct_thresh": (0.10, 0.25)},
    ...     strategy_params={"allow_long": True, "allow_short": False},
    ... )

    **Register + run** (full pipeline):

    >>> from quant_lib.experiments.base import (
    ...     PeriodConfig, UniverseConfig, StrategyConfig, ExperimentConfig,
    ... )
    >>> from quant_lib.experiments import register
    >>> register(ExperimentConfig(
    ...     name=hyp.name,
    ...     strategy_type=hyp.strategy_name,
    ...     hypothesis=hyp,
    ...     period=PeriodConfig(
    ...         train_start="2020-01-01",
    ...         train_end="2024-12-31",
    ...         holdout_months=6,  # auto-resolve holdout
    ...     ),
    ...     universe=UniverseConfig(
    ...         symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    ...         min_volume_usdt=100_000_000,
    ...         min_age_days=365,
    ...     ),
    ...     strategy=StrategyConfig(
    ...         initial_capital=1000.0,
    ...         leverage=3.0,
    ...         wfa_trials_per_fold=80,  # default
    ...     ),
    ... ))

    Notes
    -----
    **Required narrative fields** (``mechanism``, ``boundary_conditions``,
    ``success_criteria``, ``entry_logic``, ``exit_logic``) are validated by
    ``Hypothesis.validate()`` and the framework refuses to register an
    experiment with missing fields. This is by design -- the sealed
    holdout discipline requires explicit hypotheses, not ad-hoc
    parameter sweeps.

    **Common pitfalls**:

    1. **Vague mechanism** -- "this looks profitable in the backtest"
       is not a mechanism. The mechanism must explain WHY in terms
       of market microstructure, behavioral finance, or empirical
       regularities.

    2. **Success criteria set AFTER seeing results** -- the framework
       cannot prevent this socially, but the journal will note the
       timestamp. If ``timestamp`` is very close to the first commit
       timestamp, the journal marks the hypothesis as low-confidence.

    3. **Search space too narrow** -- if the default Optuna search
       ranges are tighter than your hypothesis warrants, override
       them. E.g., a "loose compression" hypothesis should have
       ``vol_pct_thresh: (0.05, 0.40)``, not the default (0.10, 0.40).

    4. **Strategy type mismatch** -- ``strategy_type=0`` (vol_compression)
       with search_space containing ``rsi_oversold`` will silently
       ignore the RSI params (the vol_compression engine doesn't
       use them). Verify the strategy_type matches your signal.

    See Also
    --------
    for_vol_compression : Factory for vol_compression_breakout strategy.
    for_pullback_sniper : Factory for pullback_sniper (RSI + reversal).
    quant_lib.experiments.base.ExperimentConfig : Top-level config
        combining Hypothesis with period/universe/strategy settings.
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
        elif self.strategy_type == StrategyType.FUNDING_RATE_CARRY:
            return "funding_rate_carry"
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

DEFAULT_FUNDING_RATE_CARRY_SEARCH_SPACE = {
    "vol_pct_thresh": (0.10, 0.40),   # unused but kept for compat
    "pullback_bars": (3, 8),          # unused but kept for compat
    "trail_atr": (1.5, 5.0),
    "sl_mult": (1.0, 3.0),
    "rsi_oversold": (25, 35),         # unused but kept for compat
    "rsi_overbought": (65, 75),       # unused but kept for compat
    "funding_entry_pct": (0.85, 0.95),  # entry threshold (e.g., 0.90 = top 10th pctile)
    "funding_exit_low": (0.30, 0.50),   # neutral zone lower bound
    "funding_exit_high": (0.50, 0.70),  # neutral zone upper bound
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
    """Factory: Hypothesis for ``vol_compression_breakout`` strategy.

    This is the canonical hypothesis factory for the vol_compression
    strategy (engine constant ``STRATEGY_VOL_COMPRESSION = 0``). The
    factory sets ``strategy_type=0`` and applies
    ``DEFAULT_VOL_COMPRESSION_SEARCH_SPACE`` if no search_space is
    provided. All other fields are required narrative content.

    Parameters
    ----------
    name : str
        Short identifier. Must match ``[a-z0-9_]+``. Used as the
        registry key, filename, and CLI experiment name.
    mechanism : str
        Why the strategy should have an edge. See ``Hypothesis``
        docstring for guidance and pitfalls.
    boundary_conditions : str
        Conditions under which the hypothesis is expected to fail.
    success_criteria : str
        Pre-defined success metrics (SPA p-value, PF, min trades, etc.).
    entry_logic : str, default
        ``"vol_pct_rank < thresh + close breaks HH_20/LL_20 + pullback"``.
        Override only if your strategy differs meaningfully from the
        default; the string is for journal/reviewer documentation.
    exit_logic : str, default
        ``"Trailing stop at ATR x trail_atr, bailout at 36 bars"``.
    universe_rules : str, optional
        Universe selection criteria (see Hypothesis docstring).
    search_space : dict, optional
        Optuna search space (parameter_name → ``(low, high)`` tuple).
        If None, uses ``DEFAULT_VOL_COMPRESSION_SEARCH_SPACE``:

        .. code-block:: python

           {
               "vol_pct_thresh": (0.10, 0.40),  # compression threshold
               "pullback_bars": (3, 8),         # pullback wait window
               "trail_atr": (1.5, 5.0),         # trailing stop ATR mult
               "sl_mult": (1.0, 3.0),           # initial SL ATR mult
           }

        Override ranges that are too narrow for your hypothesis.

    static_overrides : dict, optional
        Per-hypothesis STATIC config overrides. Example:
        ``{"fee_taker": 0.001}`` for VIP-tier fee model.
    strategy_params : dict, optional
        Strategy-specific params. Recognized keys:
        - ``allow_long`` (bool, default True)
        - ``allow_short`` (bool, default True)
    min_train_months : int, default 12
        Minimum training months for WFA. Increase to 24+ for more
        robust WFA estimates on slower timeframes.

    Returns
    -------
    Hypothesis
        Frozen dataclass instance with ``strategy_type=0``.

    Examples
    --------
    **Minimal usage** (uses all defaults):

    >>> from quant_lib.audit import for_vol_compression
    >>> hyp = for_vol_compression(
    ...     name="btc_breakout_v1",
    ...     mechanism="Vol compression + breakout in liquid crypto",
    ...     boundary_conditions="Fails in low-vol accumulation regimes",
    ...     success_criteria="SPA p < 0.15, PF > 1.3",
    ... )

    **Production usage** (custom search space + universe rules):

    >>> hyp = for_vol_compression(
    ...     name="btc_breakout_v2",
    ...     mechanism=(
    ...         "Volatility compression (vol_pct_rank < 0.15) followed by "
    ...         "volume-confirmed breakout of the 20-bar high generates "
    ...         "intraday momentum in liquid crypto perpetuals."
    ...     ),
    ...     boundary_conditions=(
    ...         "Fails on illiquid pairs (< 50M USD daily volume). "
    ...         "Fails during extreme funding environments where basis "
    ...         "blowouts distort price action."
    ...     ),
    ...     success_criteria="SPA p < 0.10, PF > 1.4, ≥ 40 OOS trades",
    ...     universe_rules="BTC, ETH, SOL — age ≥ 365d, 30d vol ≥ 100M USD",
    ...     search_space={
    ...         "vol_pct_thresh": (0.10, 0.25),  # tighter than default
    ...         "trail_atr": (2.0, 5.0),
    ...         "sl_mult": (1.5, 2.5),
    ...     },
    ...     min_train_months=24,
    ... )

    **Long-only variant** (disable shorts):

    >>> hyp = for_vol_compression(
    ...     name="btc_long_only",
    ...     mechanism="...",
    ...     boundary_conditions="...",
    ...     success_criteria="...",
    ...     strategy_params={"allow_long": True, "allow_short": False},
    ... )

    Notes
    -----
    **Search space guidance**:

    - ``vol_pct_thresh``: lower = stricter compression (fewer trades,
      higher per-trade quality). Default (0.10, 0.40) covers typical
      crypto regimes. For "tight compression" strategies, try (0.05, 0.20).
    - ``pullback_bars``: number of bars to wait for pullback after setup.
      Higher = more patience, fewer but cleaner entries.
    - ``trail_atr``: trailing stop ATR multiplier. Higher = wider stop,
      more room for volatility but more drawdown per trade.
    - ``sl_mult``: initial stop loss ATR multiplier. Lower = tighter
      stop, higher win rate but smaller winners.

    **Common pitfalls**:

    1. Setting ``vol_pct_thresh: (0.05, 0.10)`` -- too tight, will
       generate almost no trades on most pairs.
    2. Forgetting ``allow_short: False`` for long-only strategies.
       By default both directions are enabled.
    3. Setting ``min_train_months < 12`` -- WFA needs at least 1 year
       for stable PF estimates.

    See Also
    --------
    for_pullback_sniper : RSI + reversal factory.
    Hypothesis : Underlying dataclass.
    """
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
    """Factory: Hypothesis for ``pullback_sniper`` (RSI + reversal) strategy.

    This is the canonical hypothesis factory for the pullback_sniper
    strategy (engine constant ``STRATEGY_PULLBACK_SNIPER = 1``). The
    factory sets ``strategy_type=1`` and applies
    ``DEFAULT_PULLBACK_SNIPER_SEARCH_SPACE`` if no search_space is
    provided.

    Pullback sniper differs from vol_compression in three ways:
    1. Uses RSI extremes (not vol compression) as the primary signal
    2. Uses reversal candles (bullish_reversal / bearish_reversal) for confirmation
    3. Adds a TP bracket at HH_20/LL_20 (mean-reversion target), in
       addition to the trailing stop

    Parameters
    ----------
    name : str
        Short identifier. Must match ``[a-z0-9_]+``.
    mechanism : str
        Why the strategy should have an edge. The default mechanism
        is "RSI oversold/overbought extremes + reversal candles predict
        short-term mean-reversion". Override for custom mechanisms.
    boundary_conditions : str
        Conditions under which the hypothesis is expected to fail.
        Examples:
        - "fails in strong trends where RSI stays extreme for extended periods"
        - "fails on illiquid pairs where reversal candles are unreliable"
        - "fails during high-impact news events where RSI extremes are
          continuation, not exhaustion"
    success_criteria : str
        Pre-defined success metrics.
    entry_logic : str, default
        ``"RSI < oversold + bullish_reversal candle (or mirror for short)"``.
    exit_logic : str, default
        ``"Trailing stop OR TP at hh_20/ll_20 OR bailout at 36 bars"``.
        The TP bracket (HH_20/LL_20) is the mean-reversion target.
    universe_rules : str, optional
        Universe selection criteria.
    search_space : dict, optional
        Optuna search space. If None, uses
        ``DEFAULT_PULLBACK_SNIPER_SEARCH_SPACE``:

        .. code-block:: python

           {
               "vol_pct_thresh": (0.10, 0.40),  # unused but kept for compat
               "pullback_bars": (3, 8),         # unused but kept for compat
               "trail_atr": (1.5, 5.0),
               "sl_mult": (1.0, 3.0),
               "rsi_oversold": (25, 35),        # primary signal
               "rsi_overbought": (65, 75),      # primary signal
           }

        The vol_pct_thresh and pullback_bars keys are unused by the
        pullback_sniper engine but kept for registry compatibility.
        Focus tuning on ``rsi_oversold``, ``rsi_overbought``,
        ``trail_atr``, and ``sl_mult``.

    static_overrides : dict, optional
        Per-hypothesis STATIC config overrides.
    strategy_params : dict, optional
        Strategy-specific params. Recognized keys:
        - ``allow_long`` (bool, default True)
        - ``allow_short`` (bool, default True)
        - ``rsi_period`` (int, default 14) — Wilder smoothing period
          for the RSI calculation. Most users should keep the default.
    min_train_months : int, default 12

    Returns
    -------
    Hypothesis
        Frozen dataclass instance with ``strategy_type=1``.

    Examples
    --------
    **Minimal usage** (uses all defaults):

    >>> from quant_lib.audit import for_pullback_sniper
    >>> hyp = for_pullback_sniper(
    ...     name="rsi_reversal_v1",
    ...     mechanism="RSI extremes + reversal candles = mean-reversion",
    ...     boundary_conditions="Fails in strong trends (RSI stays extreme)",
    ...     success_criteria="SPA p < 0.15, PF > 1.3",
    ... )

    **Production usage** (conservative RSI thresholds + long-only):

    >>> hyp = for_pullback_sniper(
    ...     name="rsi_reversal_v2",
    ...     mechanism=(
    ...         "RSI < 25 indicates exhaustion (overbought/oversold "
    ...         "extreme); bullish_reversal candle confirms the turning "
    ...         "point. Together they predict short-term mean-reversion "
    ...         "in crypto perpetuals."
    ...     ),
    ...     boundary_conditions=(
    ...         "Fails in strong trends where RSI stays extreme for "
    ...         "extended periods (no reversion). Fails on illiquid "
    ...         "pairs where reversal candles are unreliable."
    ...     ),
    ...     success_criteria="SPA p < 0.10, PF > 1.4, ≥ 50 OOS trades",
    ...     universe_rules="BTC, ETH, SOL — age ≥ 365d",
    ...     search_space={
    ...         "rsi_oversold": (20, 30),     # stricter than default
    ...         "rsi_overbought": (70, 80),  # stricter than default
    ...         "trail_atr": (2.0, 4.0),
    ...         "sl_mult": (1.5, 2.5),
    ...     },
    ...     strategy_params={
    ...         "allow_long": True,
    ...         "allow_short": False,  # long-only
    ...         "rsi_period": 14,      # Wilder's default
    ...     },
    ...     min_train_months=24,
    ... )

    **RSI period override** (faster-reacting RSI):

    >>> hyp = for_pullback_sniper(
    ...     name="rsi_fast_v1",
    ...     mechanism="Fast RSI (period=7) catches short-term extremes",
    ...     boundary_conditions="Fails in choppy markets (RSI whipsaws)",
    ...     success_criteria="...",
    ...     strategy_params={"rsi_period": 7},
    ... )

    Notes
    -----
    **Search space guidance**:

    - ``rsi_oversold`` (default 25-35): lower = stricter oversold,
      fewer but cleaner entries. < 20 may miss valid setups; > 35
      generates noise trades.
    - ``rsi_overbought`` (default 65-75): symmetric to oversold.
      Keep rsi_overbought - 100 = -rsi_oversold for symmetric signals.
    - ``trail_atr``: trailing stop. For mean-reversion, narrower than
      vol_compression (default 1.5-5.0) is often better.
    - ``sl_mult``: initial SL. Wider for mean-reversion (the thesis
      is the move reverts; the SL protects against continued trend).

    **Common pitfalls**:

    1. Setting ``rsi_oversold: (35, 45)`` -- not extreme enough,
       generates false signals on every minor pullback.
    2. Forgetting to disable shorts for "long-only reversal" -- by
       default both directions are enabled, which doubles the trade
       count and may dilute the reversal signal quality.
    3. Setting ``rsi_period < 7`` -- RSI becomes noisy, false
       reversal signals dominate.
    4. Using this strategy on < 50 trades per year assets -- RSI
       signals require sufficient sample size for reliable PSR.

    See Also
    --------
    for_vol_compression : Vol compression + breakout factory.
    Hypothesis : Underlying dataclass.
    """
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


def for_funding_rate_carry(
    name: str,
    mechanism: str,
    boundary_conditions: str,
    success_criteria: str,
    entry_logic: str = (
        "funding_pct_rank > entry_thresh -> SHORT; "
        "funding_pct_rank < (1 - entry_thresh) -> LONG"
    ),
    exit_logic: str = (
        "Funding returns to neutral zone [exit_low, exit_high] OR "
        "trailing stop OR bailout"
    ),
    universe_rules: str | None = None,
    search_space: dict | None = None,
    static_overrides: dict | None = None,
    strategy_params: dict | None = None,
    min_train_months: int = 12,
) -> Hypothesis:
    """Factory: Hypothesis for ``funding_rate_carry`` (perp funding carry) strategy.

    This is the canonical hypothesis factory for the funding_rate_carry
    strategy (engine constant ``STRATEGY_FUNDING_RATE_CARRY = 2``). The
    factory sets ``strategy_type=2`` and applies
    ``DEFAULT_FUNDING_RATE_CARRY_SEARCH_SPACE`` if no search_space is
    provided.

    Funding rate carry is a fundamentally different strategy class from
    vol_compression (momentum) and pullback_sniper (mean-reversion):

    - Uses 30-day rolling percentile rank of perp funding rates
    - Shorts when funding is elevated (P90+: longs paying shorts)
    - Longs when funding is depressed (P10-: shorts paying longs)
    - Exits when funding returns to neutral zone [P30-P70]
    - Trade thesis: funding rates mean-revert to neutral; capturing
      the carry while waiting for the reversion is the edge.

    Parameters
    ----------
    name : str
        Short identifier. Must match ``[a-z0-9_]+``.
    mechanism : str
        Why the strategy should have an edge. The default mechanism
        is "perp funding rates mean-revert to neutral; capturing
        carry during the reversion is the edge". Override for custom
        mechanisms.
    boundary_conditions : str
        Conditions under which the hypothesis is expected to fail.
        Examples:
        - "fails during structural funding regimes (e.g., sustained
          bull market with persistently positive funding)"
        - "fails on illiquid pairs where funding rate signal is noisy"
        - "fails during market stress when funding spikes are not
          mean-reverting but persistent"
    success_criteria : str
        Pre-defined success metrics.
    entry_logic : str, default
        ``"funding_pct_rank > entry_thresh -> SHORT; funding_pct_rank
        < (1 - entry_thresh) -> LONG"``. Override only if your
        strategy differs meaningfully.
    exit_logic : str, default
        ``"Funding returns to neutral zone [exit_low, exit_high] OR
        trailing stop OR bailout"``.
    universe_rules : str, optional
        Universe selection criteria.
    search_space : dict, optional
        Optuna search space. If None, uses
        ``DEFAULT_FUNDING_RATE_CARRY_SEARCH_SPACE``:

        .. code-block:: python

           {
               "vol_pct_thresh": (0.10, 0.40),     # unused but kept for compat
               "pullback_bars": (3, 8),            # unused but kept for compat
               "trail_atr": (1.5, 5.0),
               "sl_mult": (1.0, 3.0),
               "rsi_oversold": (25, 35),           # unused but kept for compat
               "rsi_overbought": (65, 75),         # unused but kept for compat
               "funding_entry_pct": (0.85, 0.95),  # primary entry threshold
               "funding_exit_low": (0.30, 0.50),   # neutral zone lower bound
               "funding_exit_high": (0.50, 0.70),  # neutral zone upper bound
           }

        Focus tuning on ``funding_entry_pct``, ``funding_exit_low``,
        ``funding_exit_high``, ``trail_atr``, and ``sl_mult``.

    static_overrides : dict, optional
        Per-hypothesis STATIC config overrides.
    strategy_params : dict, optional
        Strategy-specific params. Recognized keys:
        - ``allow_long`` (bool, default True)
        - ``allow_short`` (bool, default True)
    min_train_months : int, default 12

    Returns
    -------
    Hypothesis
        Frozen dataclass instance with ``strategy_type=2``.

    Examples
    --------
    **Minimal usage** (uses all defaults):

    >>> from quant_lib.audit import for_funding_rate_carry
    >>> hyp = for_funding_rate_carry(
    ...     name="btc_funding_carry_v1",
    ...     mechanism="Perp funding rates mean-revert; capture carry",
    ...     boundary_conditions="Fails in persistent bull regimes",
    ...     success_criteria="SPA p < 0.15, PF > 1.2",
    ... )

    **Production usage** (tighter entry threshold, long-only):

    >>> hyp = for_funding_rate_carry(
    ...     name="btc_funding_carry_v2",
    ...     mechanism=(
    ...         "Funding rates in BTC perps show strong mean-reversion: "
    ...         "when the 8h funding rate exceeds the 95th percentile of "
    ...         "the trailing 30-day distribution, the implicit short "
    ...         "carry (received from longs) is reliably harvested over "
    ...         "the subsequent 3-7 days as funding normalizes."
    ...     ),
    ...     boundary_conditions=(
    ...         "Fails during structural bull regimes (2021 Q1, 2024 Q1) "
    ...         where funding stays elevated for weeks. Fails on illiquid "
    ...         "altcoins where funding is sparse."
    ...     ),
    ...     success_criteria="SPA p < 0.10, PF > 1.3, >= 30 OOS trades",
    ...     universe_rules="BTC, ETH, SOL -- age >= 365d, 30d volume >= 100M USD",
    ...     search_space={
    ...         "funding_entry_pct": (0.90, 0.95),  # stricter than default
    ...         "funding_exit_low": (0.35, 0.45),
    ...         "funding_exit_high": (0.55, 0.65),
    ...         "trail_atr": (2.0, 4.0),
    ...         "sl_mult": (1.5, 2.5),
    ...     },
    ...     strategy_params={"allow_long": True, "allow_short": False},
    ...     min_train_months=24,
    ... )

    Notes
    -----
    **Search space guidance**:

    - ``funding_entry_pct``: higher = stricter entry (fewer trades,
      higher per-trade quality). Default (0.85, 0.95) covers typical
      crypto regimes. For "tight carry" strategies, try (0.92, 0.97).
    - ``funding_exit_low`` / ``funding_exit_high``: define the neutral
      zone. Symmetric around 0.5 (e.g., 0.35/0.65) is typical; asymmetric
      zones (e.g., 0.40/0.60) bias toward longer hold periods.
    - ``trail_atr``: trailing stop. Funding carry trades can be volatile
      around funding events; wider stops (3-5 ATR) are typical.
    - ``sl_mult``: initial SL. Funding-driven mean-reversion is
      typically slower than price-driven, so wider SLs (1.5-2.5 ATR)
      are common.

    **Common pitfalls**:

    1. Setting ``funding_entry_pct: (0.50, 0.60)`` -- not extreme enough,
       generates noise entries on minor funding fluctuations.
    2. Forgetting to disable shorts for "long-only carry" -- by default
       both directions are enabled, which can double trade count and
       dilute the carry signal quality.
    3. Setting ``funding_exit_low > funding_exit_high`` -- invalid zone
       definition; will cause infinite hold. Validate at registration.
    4. Using this strategy on < 30 trades per year assets -- funding
       signals require sufficient sample size for reliable PSR.

    See Also
    --------
    for_vol_compression : Vol compression + breakout factory.
    for_pullback_sniper : RSI + reversal factory.
    Hypothesis : Underlying dataclass.
    """
    return Hypothesis(
        name=name,
        mechanism=mechanism,
        boundary_conditions=boundary_conditions,
        success_criteria=success_criteria,
        entry_logic=entry_logic,
        exit_logic=exit_logic,
        universe_rules=universe_rules,
        strategy_type=STRATEGY_FUNDING_RATE_CARRY,
        search_space=search_space or DEFAULT_FUNDING_RATE_CARRY_SEARCH_SPACE,
        static_overrides=static_overrides,
        strategy_params=strategy_params,
        min_train_months=min_train_months,
    )
