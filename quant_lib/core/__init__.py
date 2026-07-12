"""Private implementation for sealedge (not a public import surface).

JIT / performance code underpins ``quant_lib.tools`` and
``quant_lib.research``. Do not import ``quant_lib.core`` in user code.

Modules
-------
_engine, _features, _portfolio, _metrics, _spa, _wfa,
_risk_allocation, _config, _data, _logging, _utils

Prefer ``quant_lib.tools`` or ``quant_lib.research``. ``StrategyConfig``
and ``Candidate`` are defined for users under ``research``, even though
execution may call into ``core``.
"""
