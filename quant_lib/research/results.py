"""Result dataclasses for the public Python API.

Sprint 3 fix 3.3: ``ExploreResult`` was a plain ``dict`` returned from
``run_explore``, while ``CommitResult`` was a proper dataclass. The
asymmetry forced callers to use ``result["spa_p_value"]`` for explore
but ``result.spa_p_value`` for commit. This module defines a typed
``ExploreResult`` that:

1. Supports attribute access (``result.spa_p_value``) for type safety.
2. Supports dict-style access (``result["spa_p_value"]``) for backward
   compatibility with the prior dict-based API and existing tests.
3. Supports ``.keys()``, ``.values()``, ``.items()``, ``len()``, and
   ``in`` so existing code that iterates the dict keeps working.

This is a structural backward-compatible upgrade: existing
``result["spa_p_value"]`` access still works; new code can use
``result.spa_p_value`` for type safety.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, asdict
from typing import Iterator, Any


@dataclass
class ExploreResult:
    """Result of a single ``run_explore`` call.

    Fields are the same ones the prior dict-returning ``run_explore``
    exposed. Use attribute access (``r.spa_p_value``) for type safety;
    dict-style access (``r["spa_p_value"]``) is preserved for backward
    compatibility.

    Examples
    --------
    Attribute access (preferred):

    >>> r = ExploreResult(experiment="v1", n_oos_trades=10, ...)
    >>> r.spa_p_value
    0.123

    Dict-style access (backward compat):

    >>> r["spa_p_value"]
    0.123
    >>> "spa_p_value" in r
    True
    >>> list(r.keys())
    ['experiment', 'n_oos_trades', 'n_executed', 'n_rejected',
     'final_equity', 'spa_p_value', 'narrowed_symbols']

    Iteration and unpacking:

    >>> for key, value in r.items():
    ...     print(f"{key} = {value}")
    >>> len(r)
    7
    """

    experiment: str
    n_oos_trades: int
    n_executed: int
    n_rejected: int
    final_equity: float
    spa_p_value: float
    narrowed_symbols: list[str]

    # ------------------------------------------------------------------
    # Dict-style backward compat (Sprint 3 fix 3.3)
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """Dict-style field access. Raises ``KeyError`` for unknown keys."""
        if not isinstance(key, str):
            raise TypeError(
                f"ExploreResult keys must be str, got {type(key).__name__}"
            )
        if key in {f.name for f in fields(self)}:
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in {f.name for f in fields(self)} if isinstance(key, str) else False

    def __iter__(self) -> Iterator[str]:
        """Iterate field names in declaration order."""
        return iter(f.name for f in fields(self))

    def __len__(self) -> int:
        return len(fields(self))

    def keys(self):
        return [f.name for f in fields(self)]

    def values(self):
        return [getattr(self, f.name) for f in fields(self)]

    def items(self):
        return [(f.name, getattr(self, f.name)) for f in fields(self)]

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style ``.get(key, default)`` for safe access."""
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict:
        """Return a plain ``dict`` representation (for JSON serialization)."""
        return asdict(self)


__all__ = ["ExploreResult"]
