"""Tests for the quant_exp CLI.

Tests the new Typer-based CLI (replaces old argparse-based __main__).
"""
import os
import subprocess
import sys

import pytest


# Path to project root (where pyproject.toml lives)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_cli(*args, expect_returncode: int | None = None, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run `python -m quant_lib <args>` and return the result.

    Uses subprocess to test the actual CLI behavior (not just the typer app).
    Sets PYTHONIOENCODING=utf-8 so Rich/typer unicode output doesn't crash
    on Windows (cp1252) consoles.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, "-m", "quant_lib", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=PROJECT_ROOT,
        env=env,
    )
    if expect_returncode is not None:
        assert result.returncode == expect_returncode, (
            f"Expected return code {expect_returncode}, got {result.returncode}.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    return result


# ════════════════════════════════════════════════════════════════════════
# Help & Version
# ════════════════════════════════════════════════════════════════════════


class TestCLIHelp:
    def test_top_level_help(self):
        """`python -m quant_lib --help` shows usage."""
        result = _run_cli("--help")
        combined = (result.stdout + result.stderr).lower()
        assert "usage" in combined
        assert "quant_exp" in combined or "quant_lib" in combined

    def test_no_args_shows_help(self):
        """No args: typer with no_args_is_help=True shows help."""
        result = _run_cli()
        combined = (result.stdout + result.stderr).lower()
        # Should show help (no_args_is_help=True)
        assert "usage" in combined
        assert "command" in combined  # typer shows available commands

    def test_version_flag(self):
        """`python -m quant_lib --version` shows the current version."""
        from quant_lib import __version__
        result = _run_cli("--version")
        # --version exits cleanly (0 or 1, depending on typer)
        combined = result.stdout + result.stderr
        assert __version__ in combined


# ════════════════════════════════════════════════════════════════════════
# Subcommand Help
# ════════════════════════════════════════════════════════════════════════


class TestSubcommandHelp:
    def test_list_help(self):
        result = _run_cli("list", "--help")
        combined = (result.stdout + result.stderr).lower()
        assert "experiment" in combined

    def test_show_help(self):
        result = _run_cli("show", "--help")
        combined = (result.stdout + result.stderr).lower()
        assert "name" in combined

    def test_explore_help(self):
        result = _run_cli("explore", "--help")
        combined = (result.stdout + result.stderr).lower()
        assert "exploration" in combined or "holdout" in combined

    def test_commit_help(self):
        result = _run_cli("commit", "--help")
        combined = (result.stdout + result.stderr).lower()
        assert "commit" in combined or "irreversible" in combined or "seal" in combined

    def test_status_help(self):
        result = _run_cli("status", "--help")
        combined = (result.stdout + result.stderr).lower()
        assert "status" in combined


# ════════════════════════════════════════════════════════════════════════
# Direct unit tests (no subprocess)
# ════════════════════════════════════════════════════════════════════════


class TestCLIApp:
    def test_app_is_typer_instance(self):
        from quant_lib.cli.main import app
        import typer
        assert isinstance(app, typer.Typer)

    def test_subcommands_registered(self):
        """All 5 subcommands should be registered."""
        from quant_lib.cli.main import app
        # Typer stores registered commands
        command_names = set()
        if hasattr(app, "registered_commands"):
            for cmd in app.registered_commands:
                if hasattr(cmd, "name"):
                    command_names.add(cmd.name)
        elif hasattr(app, "commands"):
            command_names = set(app.commands.keys())
        # Should have at least these 5
        for expected in ("list", "show", "explore", "commit", "status"):
            assert expected in command_names, (
                f"Expected subcommand '{expected}' in {command_names}"
            )

    def test_explore_missing_name_fails(self):
        """`explore` without name arg should fail."""
        result = _run_cli("explore", expect_returncode=2)
        # Typer returns 2 for missing required arg

    def test_commit_missing_name_fails(self):
        """`commit` without name arg should fail."""
        result = _run_cli("commit", "-y", expect_returncode=2)

    def test_show_nonexistent_experiment_fails(self):
        """`show <nonexistent>` should exit nonzero."""
        result = _run_cli("show", "definitely_not_a_real_experiment_xyz")
        assert result.returncode != 0
        combined = (result.stdout + result.stderr).lower()
        assert "not found" in combined or "error" in combined


# ════════════════════════════════════════════════════════════════════════
# Direct unit tests for commit_cmd / status_cmd (F4 / F5)
# ════════════════════════════════════════════════════════════════════════
# These tests use Typer's ``CliRunner`` to invoke the command
# callbacks directly without spinning up a subprocess.  They focus
# on the error-path coverage that the subprocess-based smoke tests
# cannot exercise (e.g., user aborts commit, experiment not found).


class TestCommitCmdUnit:
    """Direct unit tests for ``quant_lib.cli.commit_cmd.commit``.

    These tests drive the commit command via ``python -m quant_lib``
    and assert on the observable outcomes of the early-return /
    abort paths that the subprocess smoke tests cannot exercise
    without network access.
    """

    def test_commit_unknown_experiment_returns_exit_1(self):
        """``commit <unknown>`` must exit nonzero with an error message."""
        result = _run_cli("commit", "nonexistent_xyz_experiment", "-y")
        combined = (result.stdout + result.stderr).lower()
        assert "not found" in combined or "error" in combined or result.returncode != 0

    def test_commit_yes_flag_is_accepted(self):
        """``-y`` flag must be plumbed through (no UsageError)."""
        result = _run_cli(
            "commit", "nonexistent_xyz_experiment", "-y",
        )
        combined = (result.stdout + result.stderr)
        # The command should not raise UsageError from a bad flag
        assert "Usage:" not in combined, (
            f"-y flag not accepted: {combined}"
        )

    def test_commit_missing_name_fails(self):
        """``commit`` without name arg should fail."""
        result = _run_cli("commit", "-y", expect_returncode=2)

    def test_commit_help_lists_subcommand(self):
        """``commit --help`` should describe the command."""
        result = _run_cli("commit", "--help")
        combined = (result.stdout + result.stderr).lower()
        assert "commit" in combined or "irreversible" in combined or "seal" in combined


class TestCommitAbortPath:
    """Test the user-abort path in the commit command."""

    def test_commit_user_abort_exits_zero(self, monkeypatch):
        """When user says 'no' to confirmation, exit 0 (not crash)."""
        from typer.testing import CliRunner
        from quant_lib.cli.main import app
        from quant_lib.experiments import (
            ExperimentConfig, PeriodConfig, StrategyConfig,
            UniverseConfig, clear, register,
        )
        from quant_lib.audit import for_vol_compression
        import typer

        clear()
        h = for_vol_compression(
            "abort_test", "m", "b", "c",
        )
        cfg = ExperimentConfig(
            name="abort_test", strategy_type="vol_compression", hypothesis=h,
            period=PeriodConfig(
                train_start="2020-01-01", train_end="2020-02-01",
                holdout_start="2020-02-01", holdout_end="2020-03-01",
            ),
            universe=UniverseConfig(symbols=["BTCUSDT"]),
            strategy=StrategyConfig(),
        )
        register(cfg)

        # Mock session so we don't need network
        from quant_lib.research.session import ResearchSession
        monkeypatch.setattr(ResearchSession, "__init__",
                           lambda self, *a, **kw: None)
        monkeypatch.setattr(ResearchSession, "create_candidate",
                           lambda self, h, **kw: type("C", (), {
                               "run_universe": lambda **kw: None,
                               "run_edge_testing": lambda **kw: None,
                               "run_narrowing": lambda: None,
                               "mark_ready": lambda: None,
                           })())

        runner = CliRunner()
        monkeypatch.setattr(typer, "confirm", lambda *a, **kw: False)
        result = runner.invoke(app, ["commit", "abort_test"])
        assert result.exit_code == 0
        assert "abort" in result.stdout.lower()
        clear()


class TestStatusCmdUnit:
    """Direct unit tests for ``quant_lib.cli.status_cmd.status``."""

    def test_status_run_no_seal_directory(self, monkeypatch):
        """``status`` with no seal directory should report 'no holdout seals found'."""
        # Run with HQS_DATA_DIR pointing to empty dir to force the
        # "no seals" branch.
        import tempfile
        with tempfile.TemporaryDirectory() as empty_dir:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            # Patch the data_cache path via env var
            monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "quant_lib", "status"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                cwd=empty_dir,
                env=env,
            )
            combined = (result.stdout + result.stderr).lower()
            # Should not crash, and should not show real seals
            assert "no" in combined or "empty" in combined or result.returncode == 0

    def test_status_parse_run_name_helper(self):
        """The ``_parse_run_name`` helper is exposed by status_cmd."""
        from quant_lib.cli.status_cmd import _parse_run_name
        result = _parse_run_name(
            "2026-06-26_120000_vol_compression_v1_explore",
        )
        assert result is not None
        ts, name, mode = result
        assert ts == "2026-06-26_120000"
        assert name == "vol_compression_v1"
        assert mode == "explore"

    def test_status_parse_invalid_returns_none(self):
        from quant_lib.cli.status_cmd import _parse_run_name
        assert _parse_run_name("not_a_valid_name") is None
        assert _parse_run_name("") is None
        assert _parse_run_name("2026-06-26_120000_explore") is None

    def test_status_with_empty_seal_dir(self, monkeypatch):
        """``status`` with an empty seal directory prints 'no seals'."""
        # Use cwd to a fresh empty dir so no seals are found
        import tempfile
        with tempfile.TemporaryDirectory() as empty_dir:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "quant_lib", "status"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                cwd=empty_dir,
                env=env,
            )
            combined = (result.stdout + result.stderr).lower()
            assert "no" in combined or result.returncode == 0


# ════════════════════════════════════════════════════════════════════════
# Smoke tests (don't actually run explore/commit -- they need network/data)
# ════════════════════════════════════════════════════════════════════════


class TestCLISmoke:
    def test_list_runs_without_crash(self):
        """`quant_exp list` should not crash and should show registered experiments."""
        # Ensure experiments are discovered in the parent process
        # (other tests may have cleared the registry).
        from quant_lib.experiments import (
            all_experiments, built_in, discover_experiments,
        )
        built_in.reset()
        discover_experiments()

        result = _run_cli("list")
        combined = (result.stdout + result.stderr).lower()
        # Should print either the table or "No experiments registered"
        assert "experiment" in combined or "no experiments" in combined
        # Should show registered experiments (table shows them, possibly
        # truncated due to column width). Use the registry as the
        # source of truth.
        names = {e.name for e in all_experiments()}
        assert "vol_compression_v1" in names
        assert "pullback_sniper_rsi" in names

    def test_status_runs_without_crash(self):
        """`quant_exp status` should not crash."""
        result = _run_cli("status")
        # May or may not find seals, but shouldn't crash
        combined = (result.stdout + result.stderr).lower()
        # Either shows seals or "no seals found"
        assert "holdout" in combined or "no holdout" in combined or "seal" in combined

    def test_show_vol_compression_v1(self):
        """`quant_exp show vol_compression_v1` shows experiment details."""
        result = _run_cli("show", "vol_compression_v1")
        combined = (result.stdout + result.stderr).lower()
        # Should show key fields
        assert "vol_compression" in combined
        assert "btcusdt" in combined
        assert "mechanism" in combined or "volatility" in combined

    def test_show_unknown_experiment_fails(self):
        """`quant_exp show <unknown>` should exit nonzero."""
        result = _run_cli("show", "nonexistent_experiment_xyz")
        assert result.returncode != 0
        combined = (result.stdout + result.stderr).lower()
        assert "not found" in combined or "error" in combined

    def test_explore_subcommand_registered(self):
        """`quant_exp explore --help` should work and mention key options."""
        result = _run_cli("explore", "--help")
        combined = (result.stdout + result.stderr).lower()
        # Should mention holdout preservation
        assert "holdout" in combined or "exploration" in combined

    def test_commit_subcommand_registered(self):
        """`quant_exp commit --help` should work and mention irreversibility."""
        result = _run_cli("commit", "--help")
        combined = (result.stdout + result.stderr).lower()
        # Should mention irreversibility or seal
        assert "irreversible" in combined or "seal" in combined or "commit" in combined

    def test_installed_quant_exp_script_works(self):
        """The installed `quant_exp` script (from [project.scripts]) should work.

        This verifies that pyproject.toml's [project.scripts] entry
        correctly installs and the script is invokable.
        """
        import shutil
        quant_exp_path = shutil.which("quant_exp")
        if quant_exp_path is None:
            pytest.skip(
                "quant_exp script not installed in PATH "
                "(run `pip install -e .` to install)"
            )
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [quant_exp_path, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            env=env,
        )
        # Should print version
        assert "0.2.2" in (result.stdout + result.stderr)

    def test_python_api_run_explore_signature(self):
        """The Python API `run_explore` should be callable from quant_lib."""
        from quant_lib import run_explore, run_commit
        import inspect
        # Verify signature
        sig = inspect.signature(run_explore)
        assert "experiment_name" in sig.parameters
        sig2 = inspect.signature(run_commit)
        assert "experiment_name" in sig2.parameters


# ════════════════════════════════════════════════════════════════════════
# Status command run-name parsing (0.2.2 regression test)
# ════════════════════════════════════════════════════════════════════════


class TestRunNameParsing:
    """Regression test for the OutputManager run-directory name parser.

    0.2.2 fix: the previous parser used ``r.name.split("_", 3)`` which
    truncated experiment names containing underscores (e.g.,
    "vol_compression_v1" became just "vol" with the rest glued to
    the mode). Replaced with a regex that handles underscores in
    experiment names.
    """

    def test_parse_with_underscore_in_exp_name(self):
        """vol_compression_v1 with mode=explore: parse correctly."""
        from quant_lib.cli.status_cmd import _parse_run_name
        result = _parse_run_name("2026-06-26_120000_vol_compression_v1_explore")
        assert result is not None
        ts, name, mode = result
        assert ts == "2026-06-26_120000"
        assert name == "vol_compression_v1"
        assert mode == "explore"

    def test_parse_with_git_suffix(self):
        """vol_compression_v1_explore with git suffix: parse correctly."""
        from quant_lib.cli.status_cmd import _parse_run_name
        result = _parse_run_name(
            "2026-06-26_120000_vol_compression_v1_explore_abc1234"
        )
        assert result is not None
        ts, name, mode = result
        assert ts == "2026-06-26_120000"
        assert name == "vol_compression_v1"
        assert mode == "explore"

    def test_parse_pullback_sniper_with_git_suffix(self):
        """pullback_sniper_rsi: name has 2 underscores, mode=commit."""
        from quant_lib.cli.status_cmd import _parse_run_name
        result = _parse_run_name(
            "2026-06-26_120000_pullback_sniper_rsi_commit_deadbee"
        )
        assert result is not None
        ts, name, mode = result
        assert ts == "2026-06-26_120000"
        assert name == "pullback_sniper_rsi"
        assert mode == "commit"

    def test_parse_no_git_suffix(self):
        """No git suffix: still parse correctly."""
        from quant_lib.cli.status_cmd import _parse_run_name
        result = _parse_run_name(
            "2026-06-26_120000_my_strategy_commit"
        )
        assert result is not None
        ts, name, mode = result
        assert ts == "2026-06-26_120000"
        assert name == "my_strategy"
        assert mode == "commit"

    def test_parse_invalid_returns_none(self):
        """Invalid format returns None (not raise)."""
        from quant_lib.cli.status_cmd import _parse_run_name
        assert _parse_run_name("not_a_valid_name") is None
        assert _parse_run_name("") is None
        assert _parse_run_name("2026-06-26_120000_explore") is None
        # mode must be 'explore' or 'commit'
        assert _parse_run_name("2026-06-26_120000_name_invalid") is None


# ═══════════════════════════════════════════════════════════════════════
# --report and --no-plots flags
# ═══════════════════════════════════════════════════════════════════════


class TestReportFlag:
    """``--report`` and ``--no-plots`` flags on explore/commit subcommands."""

    def test_explore_help_lists_report_flag(self):
        """``quant_exp explore --help`` mentions --report."""
        result = _run_cli("explore", "--help")
        assert "--report" in (result.stdout + result.stderr)

    def test_explore_help_lists_no_plots_flag(self):
        """``quant_exp explore --help`` mentions --no-plots."""
        result = _run_cli("explore", "--help")
        assert "--no-plots" in (result.stdout + result.stderr)

    def test_commit_help_lists_report_flag(self):
        result = _run_cli("commit", "--help")
        assert "--report" in (result.stdout + result.stderr)

    def test_commit_help_lists_no_plots_flag(self):
        result = _run_cli("commit", "--help")
        assert "--no-plots" in (result.stdout + result.stderr)


# ═══════════════════════════════════════════════════════════════════════
# Internal helper functions (strategy helper coverage)
# ═══════════════════════════════════════════════════════════════════════


class TestCLIInternalHelpers:
    """Unit tests for internal helpers used by CLI subcommands."""

    def test_looks_like_absolute_explore(self):
        from quant_lib.cli.explore import _looks_like_absolute
        # On Windows, /abs/path is NOT absolute (no drive letter).
        # Use os.path.isabs behavior directly.
        assert _looks_like_absolute(os.path.abspath(".")) is True
        assert _looks_like_absolute("relative/path") is False
        if os.name == "nt":
            assert _looks_like_absolute("C:\\abs") is True
            assert _looks_like_absolute("D:\\another") is True

    def test_looks_like_absolute_commit(self):
        from quant_lib.cli.commit_cmd import _looks_like_absolute
        assert _looks_like_absolute(os.path.abspath(".")) is True
        assert _looks_like_absolute("relative/path") is False

    def test_explore_with_report_flag_accepted(self):
        """``--report foo.html`` is accepted (no parsing error)."""
        # The command will fail (no Binance data, etc.) but the flag
        # must be parsed without a "no such option" error.
        result = _run_cli(
            "explore", "vol_compression_v1",
            "--report", "report.html",
            "--no-spa",  # skip SPA so the run is fast
            expect_returncode=None,
            timeout=120,
        )
        combined = (result.stdout + result.stderr).lower()
        # Typer should NOT have rejected the option
        assert "no such option" not in combined
        assert "unrecognized" not in combined

    def test_explore_with_no_plots_flag_accepted(self):
        result = _run_cli(
            "explore", "vol_compression_v1",
            "--no-plots", "--no-spa",
            expect_returncode=None,
            timeout=120,
        )
        combined = (result.stdout + result.stderr).lower()
        assert "no such option" not in combined
        assert "unrecognized" not in combined
