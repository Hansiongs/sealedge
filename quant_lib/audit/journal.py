"""
Experiment counter and decision journal.

Each tested hypothesis increments the counter for multiple-testing
bookkeeping (Bonferroni / FDR context). Fixing a bug is not the same
as a new trial that chases better metrics.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Literal


@dataclass
class JournalEntry:
    """Single decision entry in the experiment journal.

    Parameters
    ----------
    type : str
        "run" for a strategy test, "modify" for code change.
    description : str
        What was done.
    category : str
        "bugfix" -- technical correction (not snooping).
        "improve" -- parameter/strategy change (counts toward experiment).
        "explore" -- exploratory test (counts toward experiment).
        "ablation" -- controlled ablation study (counts but discounted).
    hypothesis_name : str
        Which hypothesis this entry belongs to.
    params_snapshot : dict, optional
        Parameters used in this run.
    """

    type: Literal["run", "modify"]
    description: str
    category: Literal["bugfix", "improve", "explore", "ablation"]
    hypothesis_name: str
    params_snapshot: Optional[dict] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ExperimentLog:
    """Decision journal and experiment counter.

    Tracks:
    1. Number of experiments per hypothesis (for Bonferroni adjustment).
    2. Distinction between bugfix and improve.
    3. Full audit trail for every decision.

    Parameters
    ----------
    hypothesis_name : str
        Name of the hypothesis being tested.
    initial_alpha : float
        Base significance level (default 0.05).

    Example
    -------
    >>> log = ExperimentLog("vol_breakout_v1")
    >>> log.log_run("Initial WFA run", category="explore")
    >>> log.log_modify("Changed trail_atr from 3.0 to 2.5", category="improve")
    >>> log.adjusted_alpha()
    0.025  # Bonferroni correction: 0.05 / 2
    """

    def __init__(self, hypothesis_name: str, initial_alpha: float = 0.05,
                 journal_path: Optional[str] = None):
        self.hypothesis_name = hypothesis_name
        self.initial_alpha = initial_alpha
        self.entries: list[JournalEntry] = []
        self.journal_path = journal_path

        # Load existing entries if journal file exists (P3-C2 fix)
        if journal_path and os.path.exists(journal_path):
            try:
                with open(journal_path, "r") as f:
                    data = json.load(f)
                self.initial_alpha = data.get("initial_alpha", initial_alpha)
                for ed in data.get("entries", []):
                    self.entries.append(JournalEntry(
                        type=ed["type"],
                        description=ed["description"],
                        category=ed["category"],
                        hypothesis_name=ed["hypothesis_name"],
                        params_snapshot=ed.get("params_snapshot"),
                        timestamp=datetime.fromisoformat(ed["timestamp"]),
                    ))
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                # NOTE (0.2.2): Was silent pass. Now log warning so audit-trail
                # corruption is visible. The framework's value proposition
                # is "no look-ahead", silently losing the audit trail
                # defeats that purpose. Start fresh after warning.
                import logging
                logging.getLogger("rich").warning(
                    f"Journal '{self.hypothesis_name}' file at {journal_path} "
                    f"is corrupt or unreadable ({type(e).__name__}: {e}). "
                    f"Starting fresh, previous entries lost."
                )

    def log_run(
        self,
        description: str,
        category: Literal["bugfix", "improve", "explore", "ablation"] = "explore",
        params_snapshot: Optional[dict] = None,
    ) -> JournalEntry:
        """Log a test/run execution."""
        entry = JournalEntry(
            type="run",
            description=description,
            category=category,
            hypothesis_name=self.hypothesis_name,
            params_snapshot=params_snapshot,
        )
        self.entries.append(entry)
        self._save_to_disk()
        return entry

    def log_modify(
        self,
        description: str,
        category: Literal["bugfix", "improve"] = "improve",
    ) -> JournalEntry:
        """Log a code or parameter modification."""
        entry = JournalEntry(
            type="modify",
            description=description,
            category=category,
            hypothesis_name=self.hypothesis_name,
        )
        self.entries.append(entry)
        self._save_to_disk()
        return entry

    @property
    def n_experiments(self) -> int:
        """Total number of experiment-counting entries (excludes bugfix)."""
        return sum(
            1 for e in self.entries if e.category in ("improve", "explore")
        )

    @property
    def n_bugfixes(self) -> int:
        """Total number of bugfix entries (not counted as experiments)."""
        return sum(1 for e in self.entries if e.category == "bugfix")

    @property
    def n_ablations(self) -> int:
        """Total number of ablation studies."""
        return sum(1 for e in self.entries if e.category == "ablation")

    def adjusted_alpha(self, discount_ablations: bool = True) -> float:
        """Bonferroni-adjusted significance level.

        Ablations can be partially discounted (count as 0.5) since they
        are controlled comparisons of known components.

        Parameters
        ----------
        discount_ablations : bool
            If True, count ablations as 0.5 experiments instead of 1.

        Returns
        -------
        float
            Adjusted alpha level for the next test.
        """
        # ``n_tests`` is annotated float from the start because the
        # half-weight ablation contribution (``* 0.5``) deliberately
        # makes the count fractional. The Bonferroni-adjusted alpha
        # ``initial_alpha / (n_tests + 1)`` then uses the same
        # formula Bailey & Lopez de Prado describe in the
        # deflated-SR literature. Without the float annotation,
        # mypy flags the int+float assignment as incompatible.
        n_tests: float = self.n_experiments
        if discount_ablations:
            # NOTE (0.2.2): Was subtract (bug, ablations are disjoint from
            # n_experiments, so subtracting undercounted tests and made
            # adjusted_alpha too lenient). Now adds ablations at half weight,
            # which is the documented "discount" intent: ablations are
            # controlled comparisons, so they count but with less penalty
            # than full explore/improve experiments.
            n_tests = n_tests + self.n_ablations * 0.5

        if n_tests <= 0:
            return self.initial_alpha

        return self.initial_alpha / (n_tests + 1)

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            f"ExperimentLog: {self.hypothesis_name}",
            f"  Runs logged    : {len(self.entries)}",
            f"  Experiments    : {self.n_experiments} (counting toward FDR)",
            f"  Bugfixes       : {self.n_bugfixes} (not counted)",
            f"  Ablations      : {self.n_ablations}",
            f"  Base alpha     : {self.initial_alpha}",
            f"  Adjusted alpha : {self.adjusted_alpha():.4f}",
        ]
        # Last 5 entries
        if self.entries:
            lines.append("  Recent entries:")
            for e in self.entries[-5:]:
                stamp = e.timestamp.strftime("%H:%M:%S")
                lines.append(
                    f"    [{stamp}] [{e.category:>8}] {e.description[:100]}"
                )
        return "\n".join(lines)

    def to_dict_list(self) -> list:
        """Serialize all entries to list of dicts."""
        return [
            {
                "type": e.type,
                "description": e.description,
                "category": e.category,
                "hypothesis_name": e.hypothesis_name,
                "params_snapshot": e.params_snapshot,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in self.entries
        ]

    def _save_to_disk(self) -> None:
        """Persist journal to file if journal_path is set (P3-C2 fix).

        Phase 3.1: Atomic write via tempfile + os.replace. A crash
        mid-write leaves the original file intact (extra safety
        for audit trail integrity, the framework's central value
        proposition prevents silent loss of decision history).
        Previously used direct ``open()`` which could produce a
        half-written JSON file on crash, losing the audit trail.
        """
        if not self.journal_path:
            return
        os.makedirs(os.path.dirname(self.journal_path) or ".", exist_ok=True)
        data = {
            "hypothesis_name": self.hypothesis_name,
            "initial_alpha": self.initial_alpha,
            "entries": self.to_dict_list(),
        }
        dir_name = os.path.dirname(self.journal_path) or "."
        fd, tmp_path = tempfile.mkstemp(
            dir=dir_name, prefix=".journal_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.journal_path)
        except Exception:
            # Cleanup temp file on any failure so we don't leave .journal_*.tmp
            # files littered around after a crash.
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
