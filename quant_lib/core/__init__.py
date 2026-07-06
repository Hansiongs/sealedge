"""Private implementation modules for sealedge.

This subpackage contains JIT-compiled and performance-critical code that
underpins the public API. Importing from ``quant_lib.core`` directly is
**not** part of the supported public surface — users should access
functionality through ``quant_lib.tools`` (white-box composable API) or
``quant_lib.research`` (high-level session/workflow).

Modules
-------
_engine : Numba-accelerated backtest engine and trade simulation.
_features : Feature engineering with leakage-aware shifts.
_portfolio : Portfolio-level aggregation and equity-curve construction.
_metrics : Sharpe / PSR / Deflated-SR / SPA statistics.
_spa : Statistical significance testing (SPA, Phipson-Smyth permutation,
       Hansen-literal stationary block bootstrap).
_wfa : Walk-forward analysis orchestrator with deterministic RNG.
_risk_allocation : Per-fold PF-weighted risk rebalancing.
_config : Centralized static constants and default parameters.
_data : Data loading (cache-first; network-aware).
_logging : Logging utilities.
_utils : Internal shared helpers.

Notes
-----
These modules are kept private (prefixed with ``_`` and not re-exported
in ``quant_lib.__init__``) so that refactoring of internal performance
layers does not break downstream code. If you find yourself importing
from ``quant_lib.core`` directly, prefer the equivalent public API in
``quant_lib.tools`` or ``quant_lib.research``.

The class names ``StrategyConfig`` and ``Candidate`` (re-exported in
``quant_lib.research.candidate``) flow through ``core`` for execution,
but their definition and user-facing interface live in ``research``.
"""
