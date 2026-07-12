"""
quant_lib.audit -- integrity primitives for the research path.

- Hypothesis definition (timestamped)
- Experiment counter (Bonferroni / FDR bookkeeping)
- Sealed holdout set

Usage::

    from quant_lib.audit import Hypothesis, ExperimentLog, HoldoutSet
"""

from quant_lib.audit.hypothesis import (
    Hypothesis,
    StrategyType,
    STRATEGY_VOL_COMPRESSION,
    STRATEGY_PULLBACK_SNIPER,
    STRATEGY_FUNDING_RATE_CARRY,
    DEFAULT_VOL_COMPRESSION_SEARCH_SPACE,
    DEFAULT_PULLBACK_SNIPER_SEARCH_SPACE,
    DEFAULT_FUNDING_RATE_CARRY_SEARCH_SPACE,
    for_vol_compression,
    for_pullback_sniper,
    for_funding_rate_carry,
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
    "STRATEGY_FUNDING_RATE_CARRY",
    "DEFAULT_VOL_COMPRESSION_SEARCH_SPACE",
    "DEFAULT_PULLBACK_SNIPER_SEARCH_SPACE",
    "DEFAULT_FUNDING_RATE_CARRY_SEARCH_SPACE",
    "for_vol_compression",
    "for_pullback_sniper",
    "for_funding_rate_carry",
]
