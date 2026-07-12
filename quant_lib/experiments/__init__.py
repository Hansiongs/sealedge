"""Experiment registry and config types.

Define experiments as modules under ``quant_lib/experiments/`` and
``register(...)`` them. Package import auto-discovers non-framework
modules. Example::

    from quant_lib.audit import for_vol_compression
    from quant_lib.experiments import PeriodConfig, UniverseConfig, from_hypothesis, register

    register(from_hypothesis(
        name="my_strategy",
        hypothesis=for_vol_compression(...),
        period=PeriodConfig(train_start="2021-07-01", train_end="2025-12-31"),
        universe=UniverseConfig(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"]),
    ))
"""
from __future__ import annotations


# Base classes
from .base import (
    ExperimentConfig,
    PeriodConfig,
    UniverseConfig,
    StrategyConfig,
    StrategyType,
    STRATEGY_NAME_TO_INT,
    STRATEGY_INT_TO_NAME,
)

# Registry
from .registry import register, get, all_experiments, exists, clear, count

# Re-export Hypothesis for convenience
from quant_lib.audit import Hypothesis


def from_hypothesis(
    name: str,
    hypothesis: Hypothesis,
    period: PeriodConfig,
    universe: UniverseConfig,
    strategy: StrategyConfig | None = None,
) -> ExperimentConfig:
    """Construct ExperimentConfig from a Hypothesis.

    Convenience for users who already use the Hypothesis factory pattern
    (``for_vol_compression()``, ``for_pullback_sniper()``).

    The ``strategy_type`` is derived from ``hypothesis.strategy_type``.

    Parameters
    ----------
    name : str
        Experiment name (must match ``hypothesis.name``).
    hypothesis : Hypothesis
        The formal hypothesis (from ``quant_lib.audit``).
    period : PeriodConfig
        Train/holdout period.
    universe : UniverseConfig
        Symbol universe.
    strategy : StrategyConfig, optional
        Per-experiment strategy overrides. Defaults to ``StrategyConfig()``
        with all defaults.

    Returns
    -------
    ExperimentConfig
    """
    strategy_name = STRATEGY_INT_TO_NAME[hypothesis.strategy_type]
    return ExperimentConfig(
        name=name,
        strategy_type=strategy_name,
        hypothesis=hypothesis,
        period=period,
        universe=universe,
        strategy=strategy if strategy is not None else StrategyConfig(),
    )


# Auto-discover experiments on import. This populates the registry with
# any experiments found in this package. Idempotent: subsequent imports
# are no-ops (cached in sys.modules and the _DISCOVERED flag in built_in).
# These imports are placed after `from_hypothesis` so that any user
# experiment file importing `from_hypothesis` (or the registry helpers)
# before its own `@register` decorator can rely on them being defined.
from . import built_in  # noqa: E402
from .built_in import discover_experiments  # noqa: E402

# Trigger initial discovery.
discover_experiments()


__all__ = [
    # Base
    "ExperimentConfig",
    "PeriodConfig",
    "UniverseConfig",
    "StrategyConfig",
    "StrategyType",
    "STRATEGY_NAME_TO_INT",
    "STRATEGY_INT_TO_NAME",
    # Registry
    "register",
    "get",
    "all_experiments",
    "exists",
    "clear",
    "count",
    # Discovery
    "discover_experiments",
    "built_in",
    # Helpers
    "from_hypothesis",
    "Hypothesis",
]
