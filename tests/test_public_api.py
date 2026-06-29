"""Direct unit tests for ``quant_lib.__init__`` (public API surface)."""
from __future__ import annotations


import pytest

import quant_lib
from quant_lib import run_commit, run_explore


# ═══════════════════════════════════════════════════════════════════════
# Version
# ═══════════════════════════════════════════════════════════════════════


class TestVersion:
    """The package exposes a ``__version__`` string."""

    def test_version_is_string(self):
        assert isinstance(quant_lib.__version__, str)

    def test_version_is_non_empty(self):
        assert quant_lib.__version__ != ""

    def test_version_format(self):
        """Version follows a major.minor.patch[-pre] format."""
        import re
        assert re.match(r"^\d+\.\d+\.\d+", quant_lib.__version__), (
            f"Unexpected version format: {quant_lib.__version__}"
        )


# ═══════════════════════════════════════════════════════════════════════
# __all__
# ═══════════════════════════════════════════════════════════════════════


class TestPublicAPI:
    """The package's public surface is exported via ``__all__``."""

    def test_all_contains_expected_entries(self):
        for name in ("tools", "audit", "core", "research",
                     "run_explore", "run_commit"):
            assert name in quant_lib.__all__, (
                f"{name} missing from __all__"
            )

    def test_all_entries_are_importable(self):
        """Each name in ``__all__`` is actually accessible."""
        for name in quant_lib.__all__:
            assert hasattr(quant_lib, name), f"{name} not accessible"

    def test_eager_submodule_imports(self):
        """Submodules are eagerly imported at package load."""
        for mod in ("tools", "audit", "core", "research"):
            assert hasattr(quant_lib, mod), f"{mod} not loaded eagerly"


# ═══════════════════════════════════════════════════════════════════════
# run_explore
# ═══════════════════════════════════════════════════════════════════════


class TestRunExplore:
    """``run_explore`` is the high-level Phase 0-3 entry point."""

    def test_unknown_experiment_raises_key_error(self):
        """Unknown experiment name raises KeyError (from registry)."""
        with pytest.raises(KeyError):
            run_explore("nonexistent_xyz_experiment")

    def test_runs_for_registered_experiment(self, tmp_path, monkeypatch):
        """A registered experiment can be run end-to-end (smoke test)."""
        # Build a minimal registered experiment
        from quant_lib.audit import for_vol_compression
        from quant_lib.experiments import (
            ExperimentConfig, PeriodConfig, StrategyConfig,
            UniverseConfig, clear, register,
        )
        clear()
        h = for_vol_compression(
            name="test_exp_smoke",
            mechanism="m", boundary_conditions="b",
            success_criteria="c", entry_logic="e", exit_logic="x",
        )
        cfg = ExperimentConfig(
            name="test_exp_smoke",
            strategy_type="vol_compression",
            hypothesis=h,
            period=PeriodConfig(train_start="2020-01-01", train_end="2024-12-31"),
            universe=UniverseConfig(symbols=["BTCUSDT"]),
            strategy=StrategyConfig(),
        )
        register(cfg)

        # Mock the heavy pipeline to avoid network/disk

        # Build a mock session+candidate that returns dummy attributes
        class MockCandidate:
            n_oos_trades = 5
            n_executed = 4
            n_rejected = 1
            final_equity = 1100.0
            spa_p_value = 0.05
            narrowed_symbols = ["BTCUSDT"]
            def run_universe(self, **kw): pass
            def run_edge_testing(self, **kw): pass
            def run_narrowing(self): pass

        class MockSession:
            def create_candidate(self, h, **kw):
                return MockCandidate()

        def mock_session_init(*args, **kwargs):
            return MockSession()

        monkeypatch.setattr(
            "quant_lib.research.session.ResearchSession.__init__",
            lambda self, *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "quant_lib.research.session.ResearchSession.create_candidate",
            lambda self, h, **kw: MockCandidate(),
        )

        result = run_explore("test_exp_smoke", cache_dir=str(tmp_path))

        assert isinstance(result, dict)
        assert result["experiment"] == "test_exp_smoke"
        assert result["n_oos_trades"] == 5
        assert result["narrowed_symbols"] == ["BTCUSDT"]
        clear()


# ═══════════════════════════════════════════════════════════════════════
# run_commit
# ═══════════════════════════════════════════════════════════════════════


class TestRunCommit:
    """``run_commit`` is the high-level Phase 4 entry point."""

    def test_unknown_experiment_raises_key_error(self):
        """Unknown experiment name raises KeyError (from registry)."""
        with pytest.raises(KeyError):
            run_commit("nonexistent_xyz_experiment")

    def test_signature_has_cache_dir_param(self):
        """``run_commit`` accepts a ``cache_dir`` keyword argument."""
        import inspect
        sig = inspect.signature(run_commit)
        assert "cache_dir" in sig.parameters
        assert "experiment_name" in sig.parameters

    def test_signature_has_default_cache_dir(self):
        """``cache_dir`` has a default value (the user can omit it)."""
        import inspect
        sig = inspect.signature(run_commit)
        assert sig.parameters["cache_dir"].default is not inspect.Parameter.empty


# ═══════════════════════════════════════════════════════════════════════
# Docstrings (sanity)
# ═══════════════════════════════════════════════════════════════════════


class TestDocstrings:
    """Public functions have informative docstrings."""

    def test_run_explore_has_docstring(self):
        assert run_explore.__doc__ is not None
        assert "experiment" in run_explore.__doc__.lower()

    def test_run_commit_has_docstring(self):
        assert run_commit.__doc__ is not None
        assert "irreversible" in run_commit.__doc__.lower() or "phase 4" in run_commit.__doc__.lower()


# ═══════════════════════════════════════════════════════════════════════
# End-to-end (mocked) run_explore
# ═══════════════════════════════════════════════════════════════════════


class TestRunExploreEndToEnd:
    """``run_explore`` end-to-end with mocked data layer."""

    def test_run_explore_returns_dict_with_all_keys(self, monkeypatch, tmp_path):
        """run_explore returns dict with all documented keys."""
        from quant_lib.experiments import (
            ExperimentConfig, PeriodConfig, StrategyConfig,
            UniverseConfig, clear, register,
        )
        from quant_lib.audit import for_vol_compression

        clear()
        h = for_vol_compression(
            "e2e_exp", "m", "b", "c", entry_logic="e", exit_logic="x",
        )
        cfg = ExperimentConfig(
            name="e2e_exp", strategy_type="vol_compression", hypothesis=h,
            period=PeriodConfig(
                train_start="2020-01-01", train_end="2020-02-01",
                holdout_start="2020-02-01", holdout_end="2020-03-01",
            ),
            universe=UniverseConfig(symbols=["BTCUSDT"]),
            strategy=StrategyConfig(),
        )
        register(cfg)

        class _MockCand:
            n_oos_trades = 10
            n_executed = 8
            n_rejected = 2
            final_equity = 1100.0
            spa_p_value = 0.04
            narrowed_symbols = ["BTCUSDT"]
            def run_universe(self, **kw): pass
            def run_edge_testing(self, **kw): pass
            def run_narrowing(self): pass

        monkeypatch.setattr(
            "quant_lib.research.session.ResearchSession.__init__",
            lambda self, *a, **kw: None,
        )
        monkeypatch.setattr(
            "quant_lib.research.session.ResearchSession.create_candidate",
            lambda self, h, **kw: _MockCand(),
        )

        result = run_explore("e2e_exp", cache_dir=str(tmp_path), n_spa_iters=0)
        assert result["experiment"] == "e2e_exp"
        assert result["n_oos_trades"] == 10
        assert result["n_executed"] == 8
        assert result["n_rejected"] == 2
        assert result["final_equity"] == 1100.0
        assert result["spa_p_value"] == 0.04
        assert result["narrowed_symbols"] == ["BTCUSDT"]
        clear()


# ═══════════════════════════════════════════════════════════════════════
# End-to-end (mocked) run_commit
# ═══════════════════════════════════════════════════════════════════════


class TestRunCommitEndToEnd:
    """``run_commit`` end-to-end with mocked data layer."""

    def test_run_commit_returns_commit_result_with_fields(self, monkeypatch, tmp_path):
        """run_commit executes full pipeline and returns CommitResult fields."""
        from quant_lib.experiments import (
            ExperimentConfig, PeriodConfig, StrategyConfig,
            UniverseConfig, clear, register,
        )
        from quant_lib.audit import for_vol_compression
        from quant_lib.research.session import ResearchSession
        from quant_lib.research import commit as commit_mod

        clear()
        h = for_vol_compression("commit_e2e", "m", "b", "c")
        cfg = ExperimentConfig(
            name="commit_e2e", strategy_type="vol_compression", hypothesis=h,
            period=PeriodConfig(
                train_start="2020-01-01", train_end="2020-02-01",
                holdout_start="2020-02-01", holdout_end="2020-03-01",
            ),
            universe=UniverseConfig(symbols=["BTCUSDT"]),
            strategy=StrategyConfig(),
        )
        register(cfg)

        class _MockResult:
            candidate_name = "commit_e2e"
            final_equity = 1050.0
            equity_pct = 0.05
            cagr_pct = 0.5
            max_dd_pct = 0.1
            psr = 0.95
            profit_factor = 1.5
            win_rate = 0.6
            n_trades = 10
            commit_idx = 1
            holdout_period = ("2020-02-01", "2020-03-01")
            initial_capital = 1000.0
            n_raw_trades = 10
            n_executed_trades = 8
            n_rejected = 2
            reject_breakdown = {}
            avg_r = 0.1
            median_r = 0.05
            std_r = 0.5
            best_r = 1.0
            worst_r = -0.5
            avg_bars_held = 5.0
            sharpe_r = 1.2
            psr_ess = 0.9
            skew = 0.1
            kurtosis = 3.0
            ess = 8
            bonferroni_alpha = 0.05
            fdr_alpha = 0.1
            by_symbol_stats = {}
            with_trend_trades = 5
            with_trend_r_total = 1.0
            counter_trend_trades = 3
            counter_trend_r_total = 0.5
            seal_hash_before = "0" * 64
            seal_hash_after = "1" * 64
            seal_broken = True
            success_criteria_text = "PSR > 0.9"
            timestamp = "2024-01-01T00:00:00Z"

        monkeypatch.setattr(
            commit_mod, "commit_to_holdout", lambda *a, **kw: _MockResult()
        )
        monkeypatch.setattr(
            ResearchSession, "__init__", lambda self, *a, **kw: None,
        )
        monkeypatch.setattr(
            ResearchSession, "create_candidate",
            lambda self, h, **kw: type("C", (), {
                "run_universe": lambda *a, **kw: None,
                "run_edge_testing": lambda *a, **kw: None,
                "run_narrowing": lambda *a, **kw: None,
                "mark_ready": lambda *a, **kw: None,
            })(),
        )

        result = run_commit("commit_e2e", cache_dir=str(tmp_path))
        assert result.candidate_name == "commit_e2e"
        assert result.seal_broken is True
        assert result.psr == 0.95
        assert result.final_equity == 1050.0
        clear()
