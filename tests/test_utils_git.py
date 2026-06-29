"""Direct unit tests for ``quant_lib.utils.git.get_git_commit``.

Tests cover the happy path (real git output), failure modes
(CalledProcessError, TimeoutExpired, FileNotFoundError, OSError),
and the ``short`` / ``cwd`` parameters.
"""
from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from quant_lib.utils.git import get_git_commit


# ═══════════════════════════════════════════════════════════════════════
# Happy path
# ═══════════════════════════════════════════════════════════════════════


class TestGetGitCommitSuccess:
    """``get_git_commit`` returns the stripped stdout on success."""

    def test_short_commit_returned(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="abc1234\n", stderr="",
            )
            result = get_git_commit(short=True)
        assert result == "abc1234"

    def test_long_commit_returned(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="abc1234567890abcdef0123456789abcdef01234\n",
                stderr="",
            )
            result = get_git_commit(short=False)
        assert result == "abc1234567890abcdef0123456789abcdef01234"

    def test_strips_whitespace(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="  abc1234  \n", stderr="",
            )
            result = get_git_commit()
        assert result == "abc1234"

    def test_handles_empty_output(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr="",
            )
            result = get_git_commit()
        assert result == ""


# ═══════════════════════════════════════════════════════════════════════
# Command construction
# ═══════════════════════════════════════════════════════════════════════


class TestCommandConstruction:
    """Verify the right git command is constructed."""

    def test_short_flag_passed(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="abc1234", stderr="",
            )
            get_git_commit(short=True)
        call_args = mock_run.call_args
        cmd = call_args.args[0]
        assert "git" in cmd
        assert "rev-parse" in cmd
        assert "--short" in cmd
        assert "HEAD" in cmd

    def test_no_short_flag_when_long(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="abc", stderr="",
            )
            get_git_commit(short=False)
        call_args = mock_run.call_args
        cmd = call_args.args[0]
        assert "--short" not in cmd

    def test_cwd_passed_through(self, tmp_path):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="abc1234", stderr="",
            )
            get_git_commit(cwd=tmp_path)
        call_args = mock_run.call_args
        assert call_args.kwargs.get("cwd") == tmp_path

    def test_default_cwd_is_none(self):
        """When ``cwd`` is not provided, subprocess uses parent's cwd."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="abc", stderr="",
            )
            get_git_commit()
        call_args = mock_run.call_args
        # cwd defaults to None (subprocess interprets as parent's cwd)
        assert call_args.kwargs.get("cwd") is None

    def test_capture_output_passed(self):
        """stdout/stderr must be captured (not printed to terminal)."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="abc", stderr="",
            )
            get_git_commit()
        call_args = mock_run.call_args
        assert call_args.kwargs.get("capture_output") is True
        assert call_args.kwargs.get("text") is True

    def test_timeout_passed(self):
        """A timeout must be set to prevent hanging on network git."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="abc", stderr="",
            )
            get_git_commit()
        call_args = mock_run.call_args
        assert "timeout" in call_args.kwargs
        assert call_args.kwargs["timeout"] >= 1

    def test_check_true_passed(self):
        """``check=True`` makes non-zero exit raise CalledProcessError."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="abc", stderr="",
            )
            get_git_commit()
        call_args = mock_run.call_args
        assert call_args.kwargs.get("check") is True


# ═══════════════════════════════════════════════════════════════════════
# Failure paths
# ═══════════════════════════════════════════════════════════════════════


class TestGetGitCommitFailures:
    """All failure modes return ``"unknown"`` without raising."""

    def test_called_process_error_returns_unknown(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=128, cmd=["git", "rev-parse"],
            )
            result = get_git_commit()
        assert result == "unknown"

    def test_timeout_expired_returns_unknown(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd=["git", "rev-parse"], timeout=2,
            )
            result = get_git_commit()
        assert result == "unknown"

    def test_file_not_found_returns_unknown(self):
        """git binary not installed."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            result = get_git_commit()
        assert result == "unknown"

    def test_oserror_returns_unknown(self):
        """Generic OSError (e.g., permission denied) is handled."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = OSError("permission denied")
            result = get_git_commit()
        assert result == "unknown"

    def test_unexpected_exception_propagates(self):
        """Contract: only the 4 documented exception types are caught.

        A non-documented exception type (e.g., ``RuntimeError``)
        propagates to the caller.  This is a contract test, not a
        bug — it documents the framework's behaviour.
        """
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = RuntimeError("totally unexpected")
            with pytest.raises(RuntimeError, match="totally unexpected"):
                get_git_commit()

    def test_logs_debug_on_failure(self, caplog):
        """Failures are logged at DEBUG level (not WARNING)."""
        import logging
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=128, cmd=["git", "rev-parse"],
            )
            with caplog.at_level(logging.DEBUG, logger="quant_lib.utils.git"):
                result = get_git_commit()
        assert result == "unknown"
        # A debug log was emitted
        debug_msgs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("git rev-parse failed" in r.message for r in debug_msgs)


# ═══════════════════════════════════════════════════════════════════════
# Contract guarantees
# ═══════════════════════════════════════════════════════════════════════


class TestContract:
    """Documented contract: never raises, always returns a string."""

    def test_returns_string_on_success(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="abc", stderr="",
            )
            result = get_git_commit()
        assert isinstance(result, str)

    def test_returns_string_on_documented_failures(self):
        """For the four documented failure types, return type is str."""
        for exc in (
            subprocess.CalledProcessError(128, ["git"]),
            subprocess.TimeoutExpired(["git"], 2),
            FileNotFoundError("git"),
            OSError("perm"),
        ):
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = exc
                result = get_git_commit()
            assert isinstance(result, str)
            assert result == "unknown"
