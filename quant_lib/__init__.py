"""
quant_lib -- importable package for sealedge.

Sealed-holdout research path for quantitative backtests:
Hypothesis -> Universe -> Edge -> Narrow -> Holdout.

Modules:
    tools/       : Public composable API (you own the flow)
    audit/       : Hypothesis journal, experiment counter, holdout seal
    core/        : Private implementation (do not import)
    research/    : ResearchSession + commit_to_holdout
    experiments/ : Experiment registry (auto-discovery)
    cli/         : quant_exp CLI
    utils/       : Shared config / git / logging helpers
"""

# Sprint 1 fix: avoid eager import of heavy submodules (tools, core,
# research). Importing ``quant_lib`` only to access ``__version__`` or
# a single submodule should not require loading the Numba engine and
# pulling in pandas/numpy/numba at the module level. We re-export the
# submodules at module-level attribute access via ``__getattr__`` so
# ``quant_lib.core``, ``quant_lib.research``, ``quant_lib.tools``,
# ``quant_lib.audit`` all still work for ``isinstance`` checks,
# debugger inspection, and IDE autocomplete. Type checkers and
# doc-build tools see the explicit module references in ``__all__``.

__version__: str = "0.5.1"

# Forward references for typing
_LAZY_MODULES = ("tools", "audit", "core", "research")


def __getattr__(name: str):
    """PEP 562 lazy submodule import.

    Lazy-load ``tools``, ``audit``, ``core``, ``research`` on first access
    so ``import quant_lib`` stays cheap when you only need ``__version__``
    or ``run_commit``. Also resolves ``ExploreResult`` / ``CommitResult``.

    Parameters
    ----------
    name : str
        Attribute name being accessed; one of the lazy submodule names
        (``tools``, ``audit``, ``core``, ``research``) or the lazily
        resolved dataclass names (``ExploreResult``, ``CommitResult``).
    """
    if name in _LAZY_MODULES:
        import importlib
        mod = importlib.import_module(f"quant_lib.{name}")
        globals()[name] = mod
        return mod
    if name == "CommitResult":
        from quant_lib.research import CommitResult
        globals()["CommitResult"] = CommitResult
        return CommitResult
    if name == "ExploreResult":
        from quant_lib.research import ExploreResult
        globals()["ExploreResult"] = ExploreResult
        return ExploreResult
    raise AttributeError(f"module 'quant_lib' has no attribute {name!r}")


# Sprint 3 fix 3.3: ``__all__`` is at the bottom of the file (after
# ``run_commit`` and after ``ExploreResult`` is referenced via the
# lazy ``__getattr__`` above). Type-checkers (mypy) read ``__all__``
# here for static analysis -- this also exposes ExploreResult for
# IDE autocomplete without forcing the research submodule to load.


# === High-level Python API ===
# For notebook/interactive use. For CLI, see `quant_exp` (entry point
# in pyproject.toml [project.scripts]).

def _build_candidate_for_explore(experiment_name, cache_dir):
    """Thin wrapper around the shared pipeline helper.

    Sprint 3 fix 3.4: the real implementation lives in
    ``quant_lib.research._pipeline.build_explore_candidate`` and is
    used by both ``run_explore`` (Python API) and the CLI's
    ``quant_exp explore`` command. This wrapper exists so the public
    API surface (``from quant_lib import _build_candidate_for_explore``)
    is unchanged from before Sprint 3.

    Parameters
    ----------
    experiment_name : str
        Name of a registered experiment (see ``quant_exp list``);
        forwarded to ``build_explore_candidate``.
    cache_dir : str
        Directory for cached data; forwarded to
        ``build_explore_candidate``.

    Returns
    -------
    Candidate
        The Candidate constructed for the explore phase; identical
        object returned by ``build_explore_candidate``.
    """
    from quant_lib.research._pipeline import build_explore_candidate
    return build_explore_candidate(experiment_name, cache_dir)


def run_explore(
    experiment_name: str,
    cache_dir: str = "./data_cache",
    n_spa_iters: int = 2000,
) -> "ExploreResult":
    """Run explore (phases 0-3). Holdout seal stays closed.

    Loads data, runs WFA + SPA on the training/OOS window. Does not
    break the seal and does not return holdout PSR (that is ``run_commit``).

    Parameters
    ----------
    experiment_name : str
        Registered experiment name (``quant_exp list``).
    cache_dir : str
        Data cache directory.
    n_spa_iters : int
        SPA iterations (default 2000, paper-grade; 0 skips SPA).

    Returns
    -------
    ExploreResult
        Explore metrics (SPA, trades, equity, narrowed symbols).
        Dict-style access still works for older callers.
    """
    from quant_lib.research._pipeline import build_explore_candidate
    from quant_lib.research import ExploreResult

    cand, exp = build_explore_candidate(experiment_name, cache_dir)
    cand.run_universe(
        min_volume_usdt=exp.universe.min_volume_usdt,
        min_age_days=exp.universe.min_age_days,
    )
    cand.run_edge_testing(n_spa_iters=n_spa_iters)
    cand.run_narrowing()

    return ExploreResult(
        experiment=exp.name,
        n_oos_trades=cand.n_oos_trades,
        n_executed=cand.n_executed,
        n_rejected=cand.n_rejected,
        final_equity=cand.final_equity,
        spa_p_value=cand.spa_p_value,
        # spa_naive_p_value: legacy SPA p when Hansen path sets spa_p_value.
        # getattr: MockCandidate/StubCandidate stubs may only set spa_p_value.
        spa_naive_p_value=getattr(cand, "spa_naive_p_value", None),
        narrowed_symbols=cand.narrowed_symbols,
    )


def run_commit(
    experiment_name: str,
    cache_dir: str = "./data_cache",
) -> "CommitResult":
    """Run holdout commit (phase 4). Irreversible.

    Breaks the seal after hash re-verification. Not used by the default
    paper-grade ``scripts/reproduce.py`` path (explore only).

    Returns
    -------
    CommitResult
        Holdout equity/trade metrics and holdout PSR, among other fields.
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
    "ExploreResult",
]
