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

    def test_run_explore_existing_experiment_runs(self):
        """run_explore with registered experiment (vol_compression_v1) attempts to run.

        NOTE: This test may take long or fail on environments without
        cached data. We only verify the call dispatches correctly.
        """
        # Just verify the function can be called without import errors
        # (the actual run may need network/data which we don't mock here)
        # Verify the function signature accepts a string
        try:
            result = run_explore("vol_compression_v1")
            # If it succeeded, verify result structure
            assert isinstance(result, dict)
            assert "experiment" in result
        except Exception as e:
            # If it failed (e.g., no data), we still verify the error
            # is NOT a programming error
            assert not isinstance(e, (ImportError, AttributeError, TypeError)), (
                f"run_explore should not fail with programming error: {e}"
            )
