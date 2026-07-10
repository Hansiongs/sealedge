"""Tests for the high-level Python API (run_explore, run_commit)."""
import pytest

from quant_lib import run_commit, run_explore
from quant_lib.experiments import clear
from quant_lib.experiments.built_in import reset as reset_discovery


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear registry, reset discovery, and re-discover built-in experiments.

    Re-discovers so the registry is populated for other tests that
    depend on `all_experiments()` (e.g., test_cli.py smoke tests).
    """
    from quant_lib.experiments import discover_experiments
    clear()
    reset_discovery()
    discover_experiments()
    yield
    clear()
    reset_discovery()
    discover_experiments()


# ════════════════════════════════════════════════════════════════════════
# Imports
# ════════════════════════════════════════════════════════════════════════


class TestImports:
    def test_run_explore_importable(self):
        from quant_lib import run_explore
        assert callable(run_explore)

    def test_run_commit_importable(self):
        from quant_lib import run_commit
        assert callable(run_commit)

    def test_exports_in_all(self):
        import quant_lib
        assert "run_explore" in quant_lib.__all__
        assert "run_commit" in quant_lib.__all__


# ════════════════════════════════════════════════════════════════════════
# Signature & Docs
# ════════════════════════════════════════════════════════════════════════


class TestSignature:
    def test_run_explore_signature(self):
        """run_explore takes name, cache_dir, n_spa_iters."""
        import inspect
        sig = inspect.signature(run_explore)
        params = list(sig.parameters.keys())
        assert "experiment_name" in params
        assert "cache_dir" in params
        assert "n_spa_iters" in params

    def test_run_commit_signature(self):
        """run_commit takes name, cache_dir."""
        import inspect
        sig = inspect.signature(run_commit)
        params = list(sig.parameters.keys())
        assert "experiment_name" in params
        assert "cache_dir" in params


# ════════════════════════════════════════════════════════════════════════
# Error handling (without actually running the heavy pipeline)
# ════════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    def test_run_explore_unknown_experiment(self):
        """run_explore with unknown name raises KeyError."""
        with pytest.raises(KeyError):
            run_explore("definitely_not_a_real_experiment_xyz")

    def test_run_commit_unknown_experiment(self):
        """run_commit with unknown name raises KeyError."""
        with pytest.raises(KeyError):
            run_commit("definitely_not_a_real_experiment_xyz")

    def test_run_explore_existing_experiment_is_registered(self):
        """Paper strategies are registered so run_explore can dispatch by name.

        Does not call run_explore end-to-end (tens of minutes / data-heavy).
        Full pipeline coverage: scripts/reproduce.py + output_paper_grade/.
        """
        from quant_lib.experiments import exists

        assert exists("vol_compression_v1")
        assert exists("pullback_sniper_rsi")
        assert exists("funding_rate_carry")
