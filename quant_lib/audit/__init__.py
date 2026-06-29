"""
quant_lib.audit -- Integrity layer for honest backtesting.

Provides tools for:
- Phase 0: Formal hypothesis definition with timestamp
- Principle 2: Experiment counter with Bonferroni tracking
- Phase 4: Sealed holdout set management

Usage:
    from quant_lib.audit import Hypothesis, ExperimentLog, HoldoutSet
"""

from quant_lib.audit.hypothesis import (
    Hypothesis,
    StrategyType,
    STRATEGY_VOL_COMPRESSION,
    STRATEGY_PULLBACK_SNIPER,
    DEFAULT_VOL_COMPRESSION_SEARCH_SPACE,
    DEFAULT_PULLBACK_SNIPER_SEARCH_SPACE,
    for_vol_compression,
    for_pullback_sniper,
)
from quant_lib.audit.journal import ExperimentLog
from quant_lib.audit.holdout import HoldoutSet

__all__ = [
    "Hypothesis",
    "ExperimentLog",
    "HoldoutSet",
    "StrategyType",
    "STRATEGY_VOL_COMPRESSION",
    "STRATEGY_PULLBACK_SNIPER",
    "DEFAULT_VOL_COMPRESSION_SEARCH_SPACE",
    "DEFAULT_PULLBACK_SNIPER_SEARCH_SPACE",
    "for_vol_compression",
    "for_pullback_sniper",
]
