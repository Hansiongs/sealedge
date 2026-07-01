"""Shared pipeline helpers used by both the Python API and CLI.

Sprint 3 fix 3.4: before Sprint 3, ``run_explore`` (in
``quant_lib/__init__.py``) and ``quant_exp explore`` (in
``quant_lib/cli/explore.py``) had near-identical session/candidate
construction logic. Drift between the two entry points was a real
risk (e.g., one of them could forget the
``strategy=exp.strategy`` argument, silently losing per-experiment
overrides).

This module defines ``build_explore_candidate`` -- a single source
of truth for the explore-pipeline boilerplate (resolve period,
build session, create candidate). Both ``run_explore`` and the CLI
``explore`` command call it.
"""
from __future__ import annotations

from quant_lib.audit import Hypothesis
from quant_lib.experiments.base import ExperimentConfig


def build_explore_candidate(
    experiment_name: str,
    cache_dir: str,
):
    """Build a ResearchSession + Candidate ready to run phases 1-3.

    The helper does NOT run any phases -- the caller decides which
    phases to run (explore stops at "narrowed", commit continues to
    "ready"). This keeps the helper trivially testable.

    Parameters
    ----------
    experiment_name : str
        Name of a registered experiment (see ``quant_exp list``).
    cache_dir : str
        Directory for cached data.

    Returns
    -------
    tuple of (Candidate, ExperimentConfig)
        The created candidate (stage="hypothesis", not yet executed)
        and the experiment config it was built from. Callers needing
        session-level data should access ``cand.session`` (the
        Candidate dataclass holds the session as a field).

    Raises
    ------
    KeyError
        If ``experiment_name`` is not registered. The CLI converts
        this to a friendly error message; the Python API lets it
        propagate.
    """
    from quant_lib.experiments import get
    from quant_lib.research.session import ResearchSession

    exp = get(experiment_name)
    train_s, train_e, hold_s, hold_e = exp.period.resolve()

    session = ResearchSession(
        training_period=(train_s, train_e),
        holdout_period=(hold_s, hold_e),
        symbols=exp.universe.symbols,
        cache_dir=cache_dir,
        _skip_holdout_load=True,
    )
    # NOTE (0.2.2): Pass strategy=exp.strategy so per-experiment
    # StrategyConfig overrides (PF weight, leverage, etc.) apply.
    # Previously the CLI path silently used default StrategyConfig()
    # and ignored per-experiment config. With this single helper,
    # the Python API and CLI both get per-experiment overrides by
    # construction.
    cand = session.create_candidate(exp.hypothesis, strategy=exp.strategy)
    return cand, exp


__all__ = ["build_explore_candidate"]
