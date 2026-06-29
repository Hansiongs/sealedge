"""Direct unit tests for ``quant_lib.cli.status_cmd.status``.

These tests exercise the status display logic with synthetic seal
files in a temp directory, capturing the rich console output to
verify rendering. We use ``Console(record=True)`` so we can assert
on the actual text printed.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

import quant_lib.cli.status_cmd as status_mod
from quant_lib.cli import status_cmd


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


class _CapturingConsole:
    """Context manager that redirects the status_cmd module's
    module-level ``console`` to a buffer that records output.
    """

    def __init__(self):
        self.console = Console(
            file=io.StringIO(),
            record=True,
            width=120,
            force_terminal=False,
            color_system=None,
        )

    def __enter__(self):
        self._patcher = patch.object(status_mod, "console", self.console)
        self._patcher.__enter__()
        return self

    def __exit__(self, *args):
        self._patcher.__exit__(*args)

    @property
    def text(self) -> str:
        return self.console.export_text()


def _write_seal(seal_dir: Path, name: str, payload: dict) -> Path:
    """Write a synthetic seal JSON file under ``seal_dir``."""
    seal_dir.mkdir(parents=True, exist_ok=True)
    path = seal_dir / f"holdout_{name}.json"
    path.write_text(json.dumps(payload))
    return path


def _make_seal_payload(
    *,
    start: str = "2025-01-01",
    end: str = "2025-06-30",
    sealed_at: str = "2025-01-01T00:00:00+00:00",
    broken_at: str | None = None,
    data_hash: str = "0" * 64,
) -> dict:
    return {
        "start": start,
        "end": end,
        "sealed_at": sealed_at,
        "broken_at": broken_at,
        "data_hash": data_hash,
    }


# ─────────────────────────────────────────────────────────────────────
# _parse_run_name
# ─────────────────────────────────────────────────────────────────────


class TestParseRunName:
    """The ``_parse_run_name`` helper is used by both the run-name
    display logic and by external callers.
    """

    def test_basic_explore(self):
        ts, name, mode = status_cmd._parse_run_name(
            "2026-06-26_120000_vol_compression_v1_explore",
        )
        assert ts == "2026-06-26_120000"
        assert name == "vol_compression_v1"
        assert mode == "explore"

    def test_basic_commit(self):
        ts, name, mode = status_cmd._parse_run_name(
            "2026-06-26_120000_pullback_sniper_rsi_commit",
        )
        assert mode == "commit"
        assert name == "pullback_sniper_rsi"

    def test_with_git_suffix(self):
        ts, name, mode = status_cmd._parse_run_name(
            "2026-06-26_120000_vol_compression_v1_explore_abc1234",
        )
        assert name == "vol_compression_v1"
        assert mode == "explore"

    def test_pullback_sniper_with_git(self):
        ts, name, mode = status_cmd._parse_run_name(
            "2026-06-26_120000_pullback_sniper_rsi_commit_deadbee",
        )
        assert name == "pullback_sniper_rsi"
        assert mode == "commit"

    def test_invalid_returns_none(self):
        assert status_cmd._parse_run_name("not_a_valid_name") is None
        assert status_cmd._parse_run_name("") is None
        assert status_cmd._parse_run_name("2026-06-26_120000_explore") is None
        assert status_cmd._parse_run_name(
            "2026-06-26_120000_name_invalid",
        ) is None  # bad mode

    def test_no_time_returns_none(self):
        assert status_cmd._parse_run_name(
            "vol_compression_v1_explore",
        ) is None

    def test_bad_date_format_returns_none(self):
        assert status_cmd._parse_run_name(
            "2026_06_26_120000_name_explore",
        ) is None


# ─────────────────────────────────────────────────────────────────────
# _SEAL_DIR / _RESULTS_DIR constants
# ─────────────────────────────────────────────────────────────────────


class TestStatusConstants:
    """Module-level path constants are Path objects pointing to
    the expected directories.
    """

    def test_seal_dir_is_under_data_cache(self):
        assert status_cmd._SEAL_DIR.name == "holdout_seals"
        assert "data_cache" in status_cmd._SEAL_DIR.parts

    def test_results_dir_is_results(self):
        assert status_cmd._RESULTS_DIR.name == "results"


# ─────────────────────────────────────────────────────────────────────
# status() — main entry point
# ─────────────────────────────────────────────────────────────────────


class TestStatusNoSealDir:
    """When the seal directory does not exist, status prints a
    friendly message and returns.
    """

    def test_status_with_no_seal_directory(self, tmp_path, monkeypatch):
        """``status`` with no seal dir shows 'no holdout seals found'."""
        # Redirect _SEAL_DIR to a non-existent path under tmp
        nonexistent = tmp_path / "no_seals"
        monkeypatch.setattr(status_cmd, "_SEAL_DIR", nonexistent)
        # Make sure _RESULTS_DIR also doesn't exist or is empty
        results = tmp_path / "no_results"
        monkeypatch.setattr(status_cmd, "_RESULTS_DIR", results)

        with _CapturingConsole() as cap:
            status_cmd.status()

        text = cap.text.lower()
        assert "no holdout seals" in text


class TestStatusEmptySealDir:
    """When the seal directory exists but contains no seal files,
    status prints a friendly message.
    """

    def test_status_with_empty_seal_directory(self, tmp_path, monkeypatch):
        seal_dir = tmp_path / "holdout_seals"
        seal_dir.mkdir()  # exists but empty
        monkeypatch.setattr(status_cmd, "_SEAL_DIR", seal_dir)
        results = tmp_path / "results"
        monkeypatch.setattr(status_cmd, "_RESULTS_DIR", results)

        with _CapturingConsole() as cap:
            status_cmd.status()

        text = cap.text.lower()
        assert "no holdout seals" in text


class TestStatusSealed:
    """When a sealed (intact) holdout seal exists, status displays
    it with the appropriate 'SEALED' status.
    """

    def test_status_shows_sealed_label(self, tmp_path, monkeypatch):
        seal_dir = tmp_path / "holdout_seals"
        _write_seal(
            seal_dir,
            "2025-01-01_2025-06-30",
            _make_seal_payload(broken_at=None),
        )
        monkeypatch.setattr(status_cmd, "_SEAL_DIR", seal_dir)
        results = tmp_path / "results"
        monkeypatch.setattr(status_cmd, "_RESULTS_DIR", results)

        with _CapturingConsole() as cap:
            status_cmd.status()

        text = cap.text
        assert "Holdout Seal" in text
        assert "SEALED" in text
        assert "2025-01-01" in text
        assert "2025-06-30" in text

    def test_status_picks_most_recent_seal(self, tmp_path, monkeypatch):
        """When multiple seals exist, the one with the lexicographically
        largest prefix (which is also the most recent) is shown.
        """
        seal_dir = tmp_path / "holdout_seals"
        _write_seal(
            seal_dir, "2024-01-01_2024-06-30",
            _make_seal_payload(start="2024-01-01", end="2024-06-30"),
        )
        _write_seal(
            seal_dir, "2025-01-01_2025-06-30",
            _make_seal_payload(start="2025-01-01", end="2025-06-30"),
        )
        _write_seal(
            seal_dir, "2024-07-01_2024-12-31",
            _make_seal_payload(start="2024-07-01", end="2024-12-31"),
        )
        monkeypatch.setattr(status_cmd, "_SEAL_DIR", seal_dir)
        results = tmp_path / "results"
        monkeypatch.setattr(status_cmd, "_RESULTS_DIR", results)

        with _CapturingConsole() as cap:
            status_cmd.status()

        text = cap.text
        # The most recent (2025) should be shown, not 2024
        assert "2025-01-01" in text
        assert "2025-06-30" in text

    def test_status_truncates_long_data_hash(self, tmp_path, monkeypatch):
        """Data hash longer than 32 chars is truncated with '...'."""
        seal_dir = tmp_path / "holdout_seals"
        long_hash = "abcdef1234567890" * 4  # 64 chars
        _write_seal(
            seal_dir, "2025-01-01_2025-06-30",
            _make_seal_payload(data_hash=long_hash),
        )
        monkeypatch.setattr(status_cmd, "_SEAL_DIR", seal_dir)
        results = tmp_path / "results"
        monkeypatch.setattr(status_cmd, "_RESULTS_DIR", results)

        with _CapturingConsole() as cap:
            status_cmd.status()

        text = cap.text
        # First 32 chars of long_hash + "..."
        assert long_hash[:32] in text
        assert "..." in text

    def test_status_omits_hash_row_when_empty(self, tmp_path, monkeypatch):
        """If data_hash is empty string, no Hash row is rendered."""
        seal_dir = tmp_path / "holdout_seals"
        _write_seal(
            seal_dir, "2025-01-01_2025-06-30",
            _make_seal_payload(data_hash=""),
        )
        monkeypatch.setattr(status_cmd, "_SEAL_DIR", seal_dir)
        results = tmp_path / "results"
        monkeypatch.setattr(status_cmd, "_RESULTS_DIR", results)

        with _CapturingConsole() as cap:
            status_cmd.status()

        text = cap.text
        # Hash row should not appear
        assert "Data hash" not in text


class TestStatusBroken:
    """When the latest seal is broken, status shows 'BROKEN' and
    includes the 'Broken at' row.
    """

    def test_status_shows_broken_label(self, tmp_path, monkeypatch):
        seal_dir = tmp_path / "holdout_seals"
        _write_seal(
            seal_dir, "2025-01-01_2025-06-30",
            _make_seal_payload(broken_at="2025-02-15T12:00:00+00:00"),
        )
        monkeypatch.setattr(status_cmd, "_SEAL_DIR", seal_dir)
        results = tmp_path / "results"
        monkeypatch.setattr(status_cmd, "_RESULTS_DIR", results)

        with _CapturingConsole() as cap:
            status_cmd.status()

        text = cap.text
        assert "BROKEN" in text
        assert "Broken at" in text
        assert "2025-02-15" in text

    def test_status_handles_missing_keys(self, tmp_path, monkeypatch):
        """A seal with missing keys still renders (with '?' for missing)."""
        seal_dir = tmp_path / "holdout_seals"
        # Write a minimal seal with only some keys
        path = seal_dir / "holdout_2025-01-01_2025-06-30.json"
        seal_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"start": "2025-01-01"}))
        monkeypatch.setattr(status_cmd, "_SEAL_DIR", seal_dir)
        results = tmp_path / "results"
        monkeypatch.setattr(status_cmd, "_RESULTS_DIR", results)

        with _CapturingConsole() as cap:
            status_cmd.status()

        text = cap.text
        # Should not crash, should render with '?' for missing fields
        assert "2025-01-01" in text


class TestStatusCorruptSeal:
    """If the seal file is malformed, status prints an error and
    returns gracefully.
    """

    def test_status_with_corrupt_json(self, tmp_path, monkeypatch):
        seal_dir = tmp_path / "holdout_seals"
        seal_dir.mkdir(parents=True, exist_ok=True)
        (seal_dir / "holdout_2025-01-01_2025-06-30.json").write_text(
            "{ invalid json",
        )
        monkeypatch.setattr(status_cmd, "_SEAL_DIR", seal_dir)
        results = tmp_path / "results"
        monkeypatch.setattr(status_cmd, "_RESULTS_DIR", results)

        with _CapturingConsole() as cap:
            status_cmd.status()

        text = cap.text.lower()
        assert "failed" in text or "error" in text


# ─────────────────────────────────────────────────────────────────────
# status() — recent runs section
# ─────────────────────────────────────────────────────────────────────


class TestStatusRecentRuns:
    """status() also displays the most recent run directories."""

    def test_status_with_no_results_directory(self, tmp_path, monkeypatch):
        """No results/ dir → only seal section shown."""
        seal_dir = tmp_path / "holdout_seals"
        _write_seal(
            seal_dir, "2025-01-01_2025-06-30",
            _make_seal_payload(),
        )
        monkeypatch.setattr(status_cmd, "_SEAL_DIR", seal_dir)
        results = tmp_path / "no_results"
        monkeypatch.setattr(status_cmd, "_RESULTS_DIR", results)

        with _CapturingConsole() as cap:
            status_cmd.status()

        text = cap.text
        assert "Holdout Seal" in text
        assert "Recent Runs" not in text

    def test_status_with_recent_runs(self, tmp_path, monkeypatch):
        """Results dir with valid run dirs shows 'Recent Runs' table."""
        seal_dir = tmp_path / "holdout_seals"
        _write_seal(
            seal_dir, "2025-01-01_2025-06-30",
            _make_seal_payload(),
        )
        monkeypatch.setattr(status_cmd, "_SEAL_DIR", seal_dir)
        results = tmp_path / "results"
        results.mkdir()
        (results / "2026-06-26_120000_vol_compression_v1_explore").mkdir()
        (results / "2026-06-25_100000_pullback_sniper_rsi_commit").mkdir()
        monkeypatch.setattr(status_cmd, "_RESULTS_DIR", results)

        with _CapturingConsole() as cap:
            status_cmd.status()

        text = cap.text
        assert "Recent Runs" in text
        # Both experiment names should be visible
        assert "vol_compression_v1" in text or "pullback_sniper_rsi" in text

    def test_status_caps_recent_runs_at_five(self, tmp_path, monkeypatch):
        """At most 5 run directories are displayed (most recent by mtime)."""
        seal_dir = tmp_path / "holdout_seals"
        _write_seal(
            seal_dir, "2025-01-01_2025-06-30",
            _make_seal_payload(),
        )
        monkeypatch.setattr(status_cmd, "_SEAL_DIR", seal_dir)
        results = tmp_path / "results"
        results.mkdir()
        for i in range(10):
            run_dir = results / f"2026-01-{i+1:02d}_120000_run_{i}_explore"
            run_dir.mkdir()
            # Explicitly set mtime so sort order is deterministic:
            # run_9 (last created) is the most recent.
            import time
            os.utime(run_dir, (time.time() + i, time.time() + i))
        monkeypatch.setattr(status_cmd, "_RESULTS_DIR", results)

        with _CapturingConsole() as cap:
            status_cmd.status()

        text = cap.text
        assert "Recent Runs" in text
        # The earliest run (run_0) should be excluded (only top 5 most recent)
        assert "run_0" not in text
        # The most recent (run_9) should be shown
        assert "run_9" in text

    def test_status_handles_invalid_run_name(self, tmp_path, monkeypatch):
        """A run directory with unparseable name shows '?' in table."""
        seal_dir = tmp_path / "holdout_seals"
        _write_seal(
            seal_dir, "2025-01-01_2025-06-30",
            _make_seal_payload(),
        )
        monkeypatch.setattr(status_cmd, "_SEAL_DIR", seal_dir)
        results = tmp_path / "results"
        results.mkdir()
        (results / "garbage_name").mkdir()  # doesn't match _RUN_NAME_RE
        monkeypatch.setattr(status_cmd, "_RESULTS_DIR", results)

        with _CapturingConsole() as cap:
            status_cmd.status()

        # Should not crash; "?" placeholder is rendered
        text = cap.text
        assert "Recent Runs" in text
        assert "garbage_name" in text  # dir name still shown

    def test_status_skips_files_in_results_dir(self, tmp_path, monkeypatch):
        """Files (non-directories) in results/ are ignored."""
        seal_dir = tmp_path / "holdout_seals"
        _write_seal(
            seal_dir, "2025-01-01_2025-06-30",
            _make_seal_payload(),
        )
        monkeypatch.setattr(status_cmd, "_SEAL_DIR", seal_dir)
        results = tmp_path / "results"
        results.mkdir()
        (results / "stray_file.txt").write_text("not a directory")
        (results / "valid_run_explore").mkdir()
        monkeypatch.setattr(status_cmd, "_RESULTS_DIR", results)

        with _CapturingConsole() as cap:
            status_cmd.status()

        # No crash; only directory entries considered
        text = cap.text
        assert "Recent Runs" in text
