"""
quant_lib.research -- White-box research session + black-box commit.

Provides:
    ResearchSession : iterative hypothesis exploration (white-box)
    Candidate       : per-hypothesis state machine
    commit_to_holdout : single-shot holdout (black-box, irreversible)

Usage:
    from quant_lib.audit import for_vol_compression, for_pullback_sniper
    from quant_lib.research import ResearchSession, commit_to_holdout

    session = ResearchSession(
        training_period=("2020-01-01", "2024-12-31"),
        holdout_period=("2025-01-01", "2025-06-30"),
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    )

    # White testing (iterative)
    hyp = for_vol_compression("vol_v1", ...)
    cand = session.create_candidate(hyp)
    cand.run_universe()
    cand.run_edge_testing()
    cand.run_narrowing()

    # Black box (single-shot)
    result = commit_to_holdout(cand, success_criteria_text="...")
"""

from quant_lib.research.session import (
    ResearchSession,
    SessionCommitRecord,
    DEFAULT_FDR_ALPHA,
)
from quant_lib.research.candidate import Candidate, CandidateStage
from quant_lib.research.commit import commit_to_holdout, CommitResult
from quant_lib.research.reporting import print_candidate_report, print_commit_report
from quant_lib.research.exceptions import (
    ResearchError,
    SessionError,
    CandidateError,
    CommitError,
    NotReadyForCommit,
    HoldoutAlreadyBroken,
    InvalidPeriod,
)
from quant_lib.research.cache import DataCache

__all__ = [
    "ResearchSession",
    "SessionCommitRecord",
    "Candidate",
    "CandidateStage",
    "commit_to_holdout",
    "CommitResult",
    "print_candidate_report",
    "print_commit_report",
    "DataCache",
    "DEFAULT_FDR_ALPHA",
    "ResearchError",
    "SessionError",
    "CandidateError",
    "CommitError",
    "NotReadyForCommit",
    "HoldoutAlreadyBroken",
    "InvalidPeriod",
]
