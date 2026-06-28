"""Tests for research module exceptions.

Validates the exception hierarchy and the ``phase`` formatting
implemented by ``ResearchError.__init__``. The full subtree:

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
import pytest

from quant_lib.research.exceptions import (
    ResearchError,
    SessionError,
    CandidateError,
    CommitError,
    SessionNotInitialized,
    HoldoutAlreadyBroken,
    InvalidPeriod,
    TooManyCandidates,
    NotReadyForCommit,
    InvalidStageTransition,
    MissingFrozenParams,
    CommitBlocked,
    SealVerificationFailed,
)


# Hierarchy: every leaf must be a subclass of ResearchError.
ALL_EXCEPTIONS = [
    SessionError,
    CandidateError,
    CommitError,
    SessionNotInitialized,
    HoldoutAlreadyBroken,
    InvalidPeriod,
    TooManyCandidates,
    NotReadyForCommit,
    InvalidStageTransition,
    MissingFrozenParams,
    CommitBlocked,
    SealVerificationFailed,
]


@pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS)
def test_all_exceptions_inherit_from_research_error(exc_cls):
    assert issubclass(exc_cls, ResearchError)


@pytest.mark.parametrize("exc_cls", [
    SessionError, SessionNotInitialized, HoldoutAlreadyBroken,
    InvalidPeriod, TooManyCandidates,
])
def test_session_subtree_inherits_from_session_error(exc_cls):
    assert issubclass(exc_cls, SessionError)


@pytest.mark.parametrize("exc_cls", [
    CandidateError, NotReadyForCommit, InvalidStageTransition,
    MissingFrozenParams,
])
def test_candidate_subtree_inherits_from_candidate_error(exc_cls):
    assert issubclass(exc_cls, CandidateError)


@pytest.mark.parametrize("exc_cls", [
    CommitError, CommitBlocked, SealVerificationFailed,
])
def test_commit_subtree_inherits_from_commit_error(exc_cls):
    assert issubclass(exc_cls, CommitError)


def test_research_error_phase_format():
    """``phase=`` argument produces ``[{phase}] {message}`` formatting."""
    e = ResearchError("test", phase="PHASE1")
    assert e.phase == "PHASE1"
    assert str(e) == "[PHASE1] test"


def test_research_error_no_phase():
    """Without phase, message is rendered as-is (no prefix)."""
    e = ResearchError("test")
    assert e.phase == ""
    assert str(e) == "test"


def test_research_error_empty_phase_same_as_no_phase():
    """Empty-string phase and omitted phase must produce identical output."""
    e_empty = ResearchError("test", phase="")
    e_omitted = ResearchError("test")
    assert str(e_empty) == str(e_omitted)
    assert e_empty.phase == e_omitted.phase == ""


def test_research_error_can_be_raised_and_caught():
    """The base class must be catchable as ``ResearchError``."""
    with pytest.raises(ResearchError) as exc_info:
        raise SessionError("oops", phase="S1")
    assert exc_info.value.phase == "S1"
    assert "[S1] oops" in str(exc_info.value)


@pytest.mark.parametrize("exc_cls,parent_cls", [
    (SessionError, ResearchError),
    (CandidateError, ResearchError),
    (CommitError, ResearchError),
    (SessionNotInitialized, SessionError),
    (HoldoutAlreadyBroken, SessionError),
    (InvalidPeriod, SessionError),
    (TooManyCandidates, SessionError),
    (NotReadyForCommit, CandidateError),
    (InvalidStageTransition, CandidateError),
    (MissingFrozenParams, CandidateError),
    (CommitBlocked, CommitError),
    (SealVerificationFailed, CommitError),
])
def test_each_exception_caught_by_immediate_parent(exc_cls, parent_cls):
    """Each leaf exception must be catchable as its immediate parent."""
    try:
        raise exc_cls("boom", phase="X")
    except parent_cls as caught:
        assert caught.phase == "X"
