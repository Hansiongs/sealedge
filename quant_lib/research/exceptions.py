"""
Research module exceptions.

Hierarchy:
    ResearchError (base)
    ├── SessionError
    │   ├── SessionNotInitialized
    │   ├── HoldoutAlreadyBroken
    │   ├── InvalidPeriod
    │   └── TooManyCandidates
    ├── CandidateError
    │   ├── NotReadyForCommit
    │   ├── InvalidStageTransition
    │   └── MissingFrozenParams
    └── CommitError
        ├── CommitBlocked
        └── SealVerificationFailed
"""


class ResearchError(Exception):
    """Base exception for all research module errors."""

    def __init__(self, message: str, phase: str = ""):
        self.phase = phase
        super().__init__(f"[{phase}] {message}" if phase else message)


class SessionError(ResearchError):
    """Session-level errors."""
    pass


class SessionNotInitialized(SessionError):
    """Session was not properly initialized."""
    pass


class HoldoutAlreadyBroken(SessionError):
    """Holdout seal was already broken; cannot commit again."""
    pass


class InvalidPeriod(SessionError):
    """Training/holdout period configuration is invalid."""
    pass


class TooManyCandidates(SessionError):
    """Too many candidates in one session (FDR correction would be too strict)."""
    pass


class CandidateError(ResearchError):
    """Candidate-level errors."""
    pass


class NotReadyForCommit(CandidateError):
    """Candidate has not completed all required stages (universe, edge, narrowing)."""
    pass


class InvalidStageTransition(CandidateError):
    """Attempted to run a stage out of order (e.g., edge_testing before universe)."""
    pass


class MissingFrozenParams(CandidateError):
    """Candidate has no frozen params (WFA produced no folds)."""
    pass


class CommitError(ResearchError):
    """Commit-level errors."""
    pass


class CommitBlocked(CommitError):
    """Commit was blocked (e.g., too many commits on same holdout)."""
    pass


class SealVerificationFailed(CommitError):
    """Holdout seal verification failed (data tampered or hash mismatch)."""
    pass
