"""Result dataclasses for the public Python API.

``ExploreResult`` is the typed return of ``run_explore`` (attribute and
dict-style access). ``CommitResult`` lives in ``commit.py``.

Explore surfaces SPA (and related trade/equity fields). Holdout PSR is
on the commit path, not on explore, matching the sealed explore/commit
split used in the manuscript sample.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, asdict
from typing import Iterator, Any


@dataclass
class ExploreResult:
    """Result of one ``run_explore`` call (seal stays closed).

    Supports attribute access (``r.spa_p_value``) and dict-style access
    (``r["spa_p_value"]``) for older callers. SPA fields are explore
    metrics; holdout PSR is produced by ``run_commit`` / ``commit_to_holdout``.
    """

    experiment: str
    n_oos_trades: int
    n_executed: int
    n_rejected: int
    final_equity: float
    spa_p_value: float
    narrowed_symbols: list[str]
    # Legacy circular-permutation SPA p. When Hansen-literal path is
    # active, spa_p_value is the Hansen-corrected p; this field keeps the
    # legacy statistic. Optional default for older constructors/stubs.
    spa_naive_p_value: float | None = None

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
