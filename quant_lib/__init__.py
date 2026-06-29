"""
quant_lib -- Quantitative Strategy Testing Library

Modular toolkit for honest backtesting with audit trail.
Based on the research session framework: Hypothesis -> Universe -> Edge -> Narrow -> Holdout.

Modules:
    tools/    : White-box public API  (composable, you control the flow)
    audit/    : Integrity layer       (hypothesis journal, experiment counter, holdout seal)
    core/     : Private implementation (do not import directly)
    research/ : ResearchSession (white-box iterative) + commit_to_holdout (black-box)
    experiments/ : Experiment registry (user-defined strategies, auto-discovery)
    cli/      : quant_exp CLI (list, show, explore, commit, status)
    utils/    : Shared utilities (config, git, logging)
"""

# Eager import to ensure submodules are available
from quant_lib import tools, audit, core, research
# Re-export CommitResult so that ``run_commit``'s return annotation
# resolves at type-check time without a string forward reference.
from quant_lib.research import CommitResult  # noqa: F401

# Version is set in pyproject.toml and accessible via importlib.metadata
# but we keep a hardcoded fallback for direct access
__version__: str = "0.3.0"


# === High-level Python API ===
# For notebook/interactive use. For CLI, see `quant_exp` (entry point
# in pyproject.toml [project.scripts]).

def run_explore(
    experiment_name: str,
    cache_dir: str = "./data_cache",
    n_spa_iters: int = 2000,
) -> dict:
    """Run Phase 0-3 (exploration) on an experiment.

    Loads data, runs WFA + SPA on the training set. Holdout stays sealed.

    Parameters
    ----------
    experiment_name : str
        Name of a registered experiment (see ``quant_exp list``).
    cache_dir : str
        Directory for cached data.
    n_spa_iters : int
        Number of SPA permutation iterations (0 to skip SPA).

    Returns
    -------
    dict
        Metrics from the exploration run: n_oos_trades, n_executed,
        n_rejected, final_equity, spa_p_value, narrowed_symbols.
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
    cand = session.create_candidate(exp.hypothesis, strategy=exp.strategy)
    cand.run_universe(
        min_volume_usdt=exp.universe.min_volume_usdt,
        min_age_days=exp.universe.min_age_days,
    )
    cand.run_edge_testing(n_spa_iters=n_spa_iters)
    cand.run_narrowing()

    return {
        "experiment": exp.name,
        "n_oos_trades": cand.n_oos_trades,
        "n_executed": cand.n_executed,
        "n_rejected": cand.n_rejected,
        "final_equity": cand.final_equity,
        "spa_p_value": cand.spa_p_value,
        "narrowed_symbols": cand.narrowed_symbols,
    }


def run_commit(
    experiment_name: str,
    cache_dir: str = "./data_cache",
) -> CommitResult:
    """Run Phase 4 (commit) on an experiment.

    IRREVERSIBLE. Breaks the holdout seal. This holdout cannot be used
    again after a successful commit.

    Parameters
    ----------
    experiment_name : str
        Name of a registered experiment.
    cache_dir : str
        Directory for cached data.

    Returns
    -------
    CommitResult
        Full commit result with equity metrics, trade stats, PSR, etc.
        See :class:`quant_lib.research.CommitResult` for fields.
    """
    from quant_lib.experiments import get
    from quant_lib.research.commit import commit_to_holdout
    from quant_lib.research.session import ResearchSession

    exp = get(experiment_name)
    h = exp.hypothesis
    train_s, train_e, hold_s, hold_e = exp.period.resolve()

    session = ResearchSession(
        training_period=(train_s, train_e),
        holdout_period=(hold_s, hold_e),
        symbols=exp.universe.symbols,
        cache_dir=cache_dir,
    )
    cand = session.create_candidate(h, strategy=exp.strategy)
    cand.run_universe(
        min_volume_usdt=exp.universe.min_volume_usdt,
        min_age_days=exp.universe.min_age_days,
    )
    cand.run_edge_testing()
    cand.run_narrowing()
    cand.mark_ready()

    return commit_to_holdout(cand, success_criteria_text=h.success_criteria)


__all__ = [
    "tools",
    "audit",
    "core",
    "research",
    "run_explore",
    "run_commit",
]
