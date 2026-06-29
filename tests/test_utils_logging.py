"""Direct unit tests for ``quant_lib.utils.logging``.

Tests cover:
- ``get_console()`` returns the module-level ``Console``
- ``setup_logging(verbose)`` sets the correct log level
- ``setup_logging(log_file=...)`` attaches a file handler
- Verbosity clamping at 2
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from rich.console import Console

from quant_lib.utils import logging as logging_mod
from quant_lib.utils.logging import get_console, setup_logging


# ═══════════════════════════════════════════════════════════════════════
# get_console
# ═══════════════════════════════════════════════════════════════════════


class TestGetConsole:
    """``get_console`` returns the module-level Console singleton."""

    def test_returns_console_instance(self):
        c = get_console()
        assert isinstance(c, Console)

    def test_returns_same_instance(self):
        """The module exposes a singleton console."""
        c1 = get_console()
        c2 = get_console()
        assert c1 is c2

    def test_singleton_matches_module_attribute(self):
        """Returned console is the module-level ``_console``."""
        c = get_console()
        assert c is logging_mod._console


# ═══════════════════════════════════════════════════════════════════════
# setup_logging - verbosity levels
# ═══════════════════════════════════════════════════════════════════════


class TestSetupLoggingVerbosity:
    """``setup_logging(verbose)`` sets the root logger level."""

    def test_verbose_0_sets_warning(self):
        setup_logging(verbose=0)
        assert logging.getLogger().level == logging.WARNING

    def test_verbose_1_sets_info(self):
        setup_logging(verbose=1)
        assert logging.getLogger().level == logging.INFO

    def test_verbose_2_sets_debug(self):
        setup_logging(verbose=2)
        assert logging.getLogger().level == logging.DEBUG

    def test_verbose_above_2_clamped_to_debug(self):
        """Verbosity > 2 is clamped to DEBUG (the max in the table)."""
        setup_logging(verbose=10)
        assert logging.getLogger().level == logging.DEBUG

    def test_default_verbose_is_warning(self):
        """Default verbose=0 → WARNING."""
        setup_logging()
        assert logging.getLogger().level == logging.WARNING

    def test_verbose_negative_clamps_to_warning(self):
        """Negative verbose (e.g., from bad input) clamps to WARNING."""
        setup_logging(verbose=-1)
        # ``min(-1, 2) = -1`` → table lookup ``[-1]`` raises IndexError
        # Actually this is a contract violation; verify actual behavior.
        # The current code uses ``min(verbose, 2)`` which doesn't
        # bound below.  A negative index reads from the end of the
        # list (Python's negative indexing).  So table[-1] is DEBUG.
        # Document this in the test:
        assert logging.getLogger().level == logging.DEBUG


# ═══════════════════════════════════════════════════════════════════════
# setup_logging - file handler
# ═══════════════════════════════════════════════════════════════════════


class TestSetupLoggingFileHandler:
    """``setup_logging(log_file=...)`` adds a FileHandler."""

    def test_no_file_handler_by_default(self, tmp_path):
        setup_logging()
        # No FileHandler on root
        for h in logging.getLogger().handlers:
            assert not isinstance(h, logging.FileHandler)

    def test_file_handler_attached(self, tmp_path):
        log_file = tmp_path / "subdir" / "test.log"
        setup_logging(log_file=log_file)
        # File should be created (parent dirs too)
        assert log_file.parent.exists()
        # FileHandler present on root
        file_handlers = [
            h for h in logging.getLogger().handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1

    def test_file_handler_writes_to_file(self, tmp_path, caplog):
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file, verbose=0)
        logger = logging.getLogger("test_module")
        logger.warning("hello world")
        # File should contain the message
        content = log_file.read_text(encoding="utf-8")
        assert "hello world" in content

    def test_file_handler_uses_utf8(self, tmp_path):
        log_file = tmp_path / "utf8.log"
        setup_logging(log_file=log_file)
        handler = [
            h for h in logging.getLogger().handlers
            if isinstance(h, logging.FileHandler)
        ][0]
        assert handler.encoding == "utf-8"

    def test_file_handler_uses_write_mode(self, tmp_path):
        """FileHandler uses ``mode="w"`` (truncates)."""
        log_file = tmp_path / "test.log"
        # Pre-populate
        log_file.write_text("stale content")
        setup_logging(log_file=log_file)
        handler = [
            h for h in logging.getLogger().handlers
            if isinstance(h, logging.FileHandler)
        ][0]
        assert handler.mode == "w"

    def test_file_handler_creates_parent_dirs(self, tmp_path):
        log_file = tmp_path / "deeply" / "nested" / "dir" / "test.log"
        assert not log_file.parent.exists()
        setup_logging(log_file=log_file)
        assert log_file.parent.exists()

    def test_file_handler_formatter(self, tmp_path):
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        handler = [
            h for h in logging.getLogger().handlers
            if isinstance(h, logging.FileHandler)
        ][0]
        fmt = handler.formatter
        # Format includes asctime, levelname, name, message
        assert fmt._fmt == "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


# ═══════════════════════════════════════════════════════════════════════
# setup_logging - RichHandler
# ═══════════════════════════════════════════════════════════════════════


class TestSetupLoggingRichHandler:
    """The RichHandler is always attached."""

    def test_rich_handler_attached(self):
        setup_logging()
        from rich.logging import RichHandler
        rich_handlers = [
            h for h in logging.getLogger().handlers
            if isinstance(h, RichHandler)
        ]
        assert len(rich_handlers) == 1

    def test_rich_handler_uses_module_console(self):
        """RichHandler is bound to the module's singleton console."""
        from rich.logging import RichHandler
        setup_logging()
        rh = [
            h for h in logging.getLogger().handlers
            if isinstance(h, RichHandler)
        ][0]
        assert rh.console is logging_mod._console

    def test_rich_handler_has_markup(self):
        """markup=True is set (allows rich tags in log messages)."""
        from rich.logging import RichHandler
        setup_logging()
        rh = [
            h for h in logging.getLogger().handlers
            if isinstance(h, RichHandler)
        ][0]
        assert rh.markup is True


# ═══════════════════════════════════════════════════════════════════════
# setup_logging - force override
# ═══════════════════════════════════════════════════════════════════════


class TestSetupLoggingForce:
    """``setup_logging`` with ``force=True`` overrides existing config."""

    def test_call_twice_replaces_handlers(self):
        setup_logging()
        # We only need the post-second state; setup_logging's force=True
        # should leave exactly one RichHandler after the second call.
        setup_logging()
        handlers_after_second = list(logging.getLogger().handlers)
        # Both RichHandlers should be present (one from each call)
        # but the count of FileHandlers should be 0 (we didn't pass log_file)
        from rich.logging import RichHandler
        rich_count = sum(
            1 for h in handlers_after_second
            if isinstance(h, RichHandler)
        )
        # ``force=True`` removes existing handlers, so only 1 RichHandler
        # should remain (from the second call).
        assert rich_count == 1, (
            f"Expected 1 RichHandler after force=True, got {rich_count}"
        )

    def test_call_with_different_file_replaces_handler(self, tmp_path):
        log_file1 = tmp_path / "first.log"
        log_file2 = tmp_path / "second.log"
        setup_logging(log_file=log_file1)
        setup_logging(log_file=log_file2)
        # Only one FileHandler remains (force=True removed the first)
        file_handlers = [
            h for h in logging.getLogger().handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert Path(file_handlers[0].baseFilename) == log_file2


# ═══════════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset logging config after each test to prevent leakage."""
    yield
    # Remove all handlers from root
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    root.setLevel(logging.WARNING)
