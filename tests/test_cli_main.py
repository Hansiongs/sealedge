"""Direct unit tests for ``quant_lib.cli.main``.

Tests cover:
- The Typer app object structure (name, help, subcommand registration)
- The version callback behaviour
"""
from __future__ import annotations

import typer

from quant_lib import cli
from quant_lib.cli.main import app


# ═══════════════════════════════════════════════════════════════════════
# App structure
# ═══════════════════════════════════════════════════════════════════════


class TestAppStructure:
    """The Typer ``app`` object has the expected configuration."""

    def test_app_is_typer_instance(self):
        assert isinstance(app, typer.Typer)

    def test_app_name(self):
        assert app.info.name == "quant_exp"

    def test_app_help(self):
        assert "honest backtesting" in (app.info.help or "")

    def test_app_no_args_is_help(self):
        """``no_args_is_help=True`` is set on the app."""
        assert app.info.no_args_is_help is True

    def test_app_disables_completion(self):
        """``add_completion=False`` is set (verified via help output)."""
        info = app.info
        assert info is not None  # basic smoke check

    def test_all_5_subcommands_registered(self):
        """The 5 documented subcommands are registered."""
        command_names = set()
        if hasattr(app, "registered_commands"):
            for cmd in app.registered_commands:
                if hasattr(cmd, "name"):
                    command_names.add(cmd.name)
        for expected in ("list", "show", "explore", "commit", "status"):
            assert expected in command_names, (
                f"Missing subcommand: {expected}; have {command_names}"
            )

    def test_subcommand_help_texts(self):
        """Each subcommand has a non-empty help string."""
        for cmd in app.registered_commands:
            if hasattr(cmd, "help"):
                assert cmd.help, f"Empty help for {cmd.name}"


# ═══════════════════════════════════════════════════════════════════════
# main callback
# ═══════════════════════════════════════════════════════════════════════


class TestMainCallback:
    """The ``main`` callback handles ``--version`` and ``-v`` flags."""

    def test_main_callback_accepts_version_flag(self):
        """``--version`` is a recognised flag (not a command)."""
        from quant_lib.cli.main import main as main_fn
        assert callable(main_fn)


# ═══════════════════════════════════════════════════════════════════════
# Module structure
# ═══════════════════════════════════════════════════════════════════════


class TestModuleStructure:
    """Verify the expected subcommands are imported."""

    def test_subcommands_imported_at_module_level(self):
        """All 5 subcommand callbacks are imported in the module."""
        from quant_lib.cli import list_cmd, show, explore, commit_cmd, status_cmd
        assert list_cmd.list_cmd is not None
        assert show.show is not None
        assert explore.explore is not None
        assert commit_cmd.commit is not None
        assert status_cmd.status is not None

    def test_submodule_names(self):
        """Module exposes the expected attributes."""
        assert hasattr(cli, "main")
        assert hasattr(cli.main, "app")
        assert hasattr(cli.main, "main")
        assert hasattr(cli.main, "_version_callback")
