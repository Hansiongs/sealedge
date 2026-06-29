"""Tests for the holdout-seal migration tool.

Covers ``quant_lib.cli.migrate_seals`` (the library API) and the
``quant_exp migrate-seals`` CLI subcommand (via the test runner).

The migration tool rewrites holdout seal files in place to (re)compute
the HMAC signature with the current ``QUANT_LIB_HMAC_SECRET``. This
is needed when:

1. Rotating the HMAC secret.
2. Upgrading from ``quant_lib`` v0.2.x or earlier (pre-B0.1) where
   seals were not HMAC-signed at all.
3. Recovering from a previous migration with the wrong secret.

These tests verify the tool is idempotent, atomic, and safe.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quant_lib.audit.holdout import (
    compute_seal_signature,
    verify_seal_signature,
    _reset_hmac_secret_cache,
)
from quant_lib.cli.migrate_seals import (
    _classify_seal,
    _resolve_seal_dir,
    migrate_seals,
)


# Each test that needs a fresh seal needs a unique secret to avoid
# cached-state bleed. We use a constant 64-char string.
_TEST_SECRET = "x" * 64


@pytest.fixture(autouse=True)
def _set_hmac_secret(monkeypatch):
    """Set QUANT_LIB_HMAC_SECRET for every test in this module.

    The autouse fixture also resets the cache so changing the env
    var (in tests that rotate the secret) takes effect.
    """
    from quant_lib.audit.holdout import _reset_hmac_secret_cache
    monkeypatch.setenv("QUANT_LIB_HMAC_SECRET", _TEST_SECRET)
    _reset_hmac_secret_cache()
    yield
    _reset_hmac_secret_cache()


def _write_seal(
    seal_path: Path,
    *,
    data_hash: str = "a" * 64,
    include_signature: bool = True,
    signature: str | None = None,
) -> dict:
    """Helper: write a seal-like dict to ``seal_path`` and return it.

    If ``include_signature`` is True and ``signature`` is None, a
    fresh signature is computed with the current secret. Pass
    ``signature="bad"`` to write a deliberately-invalid signature.
    """
    state = {
        "start": "2025-01-01",
        "end": "2025-06-30",
        "sealed_at": "2025-01-01T00:00:00+00:00",
        "broken_at": None,
        "data_hash": data_hash,
    }
    if include_signature:
        if signature is None:
            state["signature"] = compute_seal_signature(state)
        else:
            state["signature"] = signature
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    seal_path.write_text(json.dumps(state, indent=2))
    return state


class TestClassifySeal:
    """Unit tests for the internal ``_classify_seal`` helper."""

    def test_valid_signature_classifies_as_valid(self, tmp_path):
        state = _write_seal(tmp_path / "holdout_2025-01-01_2025-06-30.json")
        assert _classify_seal(state) == "valid"

    def test_missing_signature_classifies_as_unsigned(self):
        state = {"start": "2025-01-01", "end": "2025-06-30"}
        assert _classify_seal(state) == "unsigned"

    def test_empty_signature_classifies_as_unsigned(self):
        state = {
            "start": "2025-01-01",
            "end": "2025-06-30",
            "signature": "",
        }
        assert _classify_seal(state) == "unsigned"

    def test_wrong_signature_classifies_as_invalid(self, tmp_path):
        state = _write_seal(
            tmp_path / "holdout_2025-01-01_2025-06-30.json",
            signature="deadbeef" * 8,
        )
        assert _classify_seal(state) == "invalid"


class TestResolveSealDir:
    """Unit tests for the seal-dir resolution priority chain."""

    def test_argument_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("QUANT_LIB_SEAL_DIR", "/env/path")
        assert _resolve_seal_dir("/arg/path") == Path("/arg/path")

    def test_env_wins_over_default(self, monkeypatch):
        monkeypatch.setenv("QUANT_LIB_SEAL_DIR", "/env/path")
        assert _resolve_seal_dir(None) == Path("/env/path")

    def test_default_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv("QUANT_LIB_SEAL_DIR", raising=False)
        assert _resolve_seal_dir(None) == Path("data_cache/holdout_seals")


class TestMigrateSealsLibraryAPI:
    """End-to-end tests for the ``migrate_seals`` library function."""

    def test_no_directory_returns_zeros(self, tmp_path):
        n_valid, n_migrated, n_would, n_errors = migrate_seals(
            seal_dir=str(tmp_path / "does_not_exist"),
        )
        assert (n_valid, n_migrated, n_would, n_errors) == (0, 0, 0, 0)

    def test_empty_directory_returns_zeros(self, tmp_path):
        n_valid, n_migrated, n_would, n_errors = migrate_seals(
            seal_dir=str(tmp_path),
        )
        assert (n_valid, n_migrated, n_would, n_errors) == (0, 0, 0, 0)

    def test_valid_seals_are_skipped(self, tmp_path):
        _write_seal(tmp_path / "holdout_2025-01-01_2025-06-30.json")
        n_valid, n_migrated, n_would, n_errors = migrate_seals(
            seal_dir=str(tmp_path),
        )
        assert n_valid == 1
        assert n_migrated == 0
        assert n_would == 0
        assert n_errors == 0

    def test_unsigned_seals_are_migrated(self, tmp_path):
        # Pre-B0.1 format: no signature field at all.
        _write_seal(
            tmp_path / "holdout_2025-01-01_2025-06-30.json",
            include_signature=False,
        )
        n_valid, n_migrated, n_would, n_errors = migrate_seals(
            seal_dir=str(tmp_path),
        )
        assert n_valid == 0
        assert n_migrated == 1
        assert n_errors == 0
        # Verify the rewritten file now has a valid signature.
        state = json.loads(
            (tmp_path / "holdout_2025-01-01_2025-06-30.json").read_text()
        )
        assert verify_seal_signature(state)
        # And a backup was created.
        assert (tmp_path / "holdout_2025-01-01_2025-06-30.json.bak").exists()

    def test_invalid_signature_is_migrated(self, tmp_path):
        # Seal signed with a different secret.
        _write_seal(
            tmp_path / "holdout_2025-01-01_2025-06-30.json",
            signature="badbadbad" * 8,
        )
        n_valid, n_migrated, n_would, n_errors = migrate_seals(
            seal_dir=str(tmp_path),
        )
        assert n_migrated == 1
        assert n_errors == 0
        state = json.loads(
            (tmp_path / "holdout_2025-01-01_2025-06-30.json").read_text()
        )
        assert verify_seal_signature(state)

    def test_dry_run_does_not_modify_files(self, tmp_path):
        _write_seal(
            tmp_path / "holdout_2025-01-01_2025-06-30.json",
            include_signature=False,
        )
        # Snapshot the original content.
        original = (
            tmp_path / "holdout_2025-01-01_2025-06-30.json"
        ).read_text()
        n_valid, n_migrated, n_would, n_errors = migrate_seals(
            seal_dir=str(tmp_path), dry_run=True,
        )
        # File is untouched.
        assert (
            tmp_path / "holdout_2025-01-01_2025-06-30.json"
        ).read_text() == original
        # But the count shows it would have been migrated.
        assert n_would == 1
        assert n_migrated == 0
        # No backup file was created either.
        assert not (
            tmp_path / "holdout_2025-01-01_2025-06-30.json.bak"
        ).exists()

    def test_idempotent_on_second_run(self, tmp_path):
        """A second run after migration should skip the now-valid seal."""
        seal = tmp_path / "holdout_2025-01-01_2025-06-30.json"
        _write_seal(seal, include_signature=False)
        # First run: migrate.
        migrate_seals(seal_dir=str(tmp_path))
        # Second run: skip.
        n_valid, n_migrated, _, n_errors = migrate_seals(
            seal_dir=str(tmp_path),
        )
        assert n_valid == 1
        assert n_migrated == 0
        assert n_errors == 0

    def test_preserves_data_fields(self, tmp_path):
        """Migration must not touch start/end/sealed_at/data_hash."""
        seal = tmp_path / "holdout_2025-01-01_2025-06-30.json"
        original = _write_seal(
            seal,
            data_hash="0" * 64,
            include_signature=False,
        )
        original_broken = original.get("broken_at")
        original_sealed = original.get("sealed_at")
        original_hash = original.get("data_hash")
        migrate_seals(seal_dir=str(tmp_path))
        new_state = json.loads(seal.read_text())
        assert new_state["start"] == "2025-01-01"
        assert new_state["end"] == "2025-06-30"
        assert new_state["sealed_at"] == original_sealed
        assert new_state["broken_at"] == original_broken
        assert new_state["data_hash"] == original_hash
        # And the signature is now valid.
        assert verify_seal_signature(new_state)

    def test_mixed_directory_reports_each_correctly(self, tmp_path):
        """A directory with a mix of valid, unsigned, and broken
        seals should be classified independently per file."""
        _write_seal(tmp_path / "holdout_2025-01-01_2025-06-30.json")
        _write_seal(
            tmp_path / "holdout_2025-07-01_2025-12-31.json",
            include_signature=False,
        )
        _write_seal(
            tmp_path / "holdout_2026-01-01_2026-06-30.json",
            signature="badbad" * 10,
        )
        n_valid, n_migrated, _, n_errors = migrate_seals(
            seal_dir=str(tmp_path),
        )
        assert n_valid == 1
        assert n_migrated == 2
        assert n_errors == 0

    def test_corrupt_json_counts_as_error_not_crash(self, tmp_path):
        """A malformed JSON file is counted as an error but does not
        crash the whole run; other valid files are still processed.
        """
        _write_seal(tmp_path / "holdout_2025-01-01_2025-06-30.json")
        (tmp_path / "holdout_2025-07-01_2025-12-31.json").write_text(
            "{this is not valid json"
        )
        n_valid, n_migrated, _, n_errors = migrate_seals(
            seal_dir=str(tmp_path),
        )
        assert n_valid == 1
        assert n_errors == 1
        assert n_migrated == 0

    def test_missing_secret_raises(self, tmp_path):
        """If ``QUANT_LIB_HMAC_SECRET`` is not set, the tool fails
        loudly with a clear error before touching any files.

        We patch the conftest autouse away for this test by setting
        the env var to an empty string (which the framework treats
        the same as unset). The cache is reset so the change is
        picked up on the next call.
        """
        import os as _os
        _saved = _os.environ.get("QUANT_LIB_HMAC_SECRET")
        # ``_os.environ[k] = ""`` is treated as "not set" by
        # get_hmac_secret (it checks ``if not raw``). This avoids
        # racing with the conftest autouse monkeypatch which only
        # tracks the variable's value via setenv/delenv, not direct
        # os.environ mutation.
        _os.environ["QUANT_LIB_HMAC_SECRET"] = ""
        _reset_hmac_secret_cache()
        try:
            # Write the seal file directly (bypassing _write_seal,
            # which would call compute_seal_signature and raise
            # before we even get to the assertion). Pre-B0.1 seals
            # have no signature field, which is what we want here.
            (tmp_path / "holdout_2025-01-01_2025-06-30.json").write_text(
                json.dumps(
                    {
                        "start": "2025-01-01",
                        "end": "2025-06-30",
                        "sealed_at": "2025-01-01T00:00:00+00:00",
                        "broken_at": None,
                        "data_hash": "a" * 64,
                    },
                    indent=2,
                )
            )
            with pytest.raises(
                RuntimeError,
                match="HMAC seal secret is not configured",
            ):
                migrate_seals(seal_dir=str(tmp_path))
            # File is untouched (migrate_seals should not have read it).
            assert (
                tmp_path / "holdout_2025-01-01_2025-06-30.json"
            ).exists()
        finally:
            if _saved is not None:
                _os.environ["QUANT_LIB_HMAC_SECRET"] = _saved
            else:
                _os.environ.pop("QUANT_LIB_HMAC_SECRET", None)
            _reset_hmac_secret_cache()

    def test_short_secret_raises(self, tmp_path):
        """Secrets shorter than the minimum length are rejected."""
        import os as _os
        _saved = _os.environ.get("QUANT_LIB_HMAC_SECRET")
        _os.environ["QUANT_LIB_HMAC_SECRET"] = "short"
        _reset_hmac_secret_cache()
        try:
            with pytest.raises(RuntimeError, match="too short"):
                migrate_seals(seal_dir=str(tmp_path))
        finally:
            if _saved is not None:
                _os.environ["QUANT_LIB_HMAC_SECRET"] = _saved
            else:
                _os.environ.pop("QUANT_LIB_HMAC_SECRET", None)
            _reset_hmac_secret_cache()


class TestMigrateSealsCLI:
    """Tests for the ``quant_exp migrate-seals`` CLI subcommand."""

    def test_help_shows_options(self):
        """``quant_exp migrate-seals --help`` should list the options."""
        # Invoke the CLI via python -m so we don't depend on a
        # `quant_exp` console-script install in this test env.
        result = subprocess.run(
            [sys.executable, "-m", "quant_lib.cli.main", "migrate-seals", "--help"],
            capture_output=True, text=True, env={**os.environ},
        )
        # Typer exits 0 on --help.
        assert result.returncode == 0, result.stderr
        assert "--seal-dir" in result.stdout
        assert "--dry-run" in result.stdout
        assert "--yes" in result.stdout

    def test_cli_dry_run_on_real_seal_dir(self, tmp_path, monkeypatch):
        """``migrate-seals --dry-run`` on a directory with an
        unsigned seal should report 'WOULD MIGRATE' without
        modifying the file.
        """
        monkeypatch.setenv("QUANT_LIB_HMAC_SECRET", _TEST_SECRET)
        _reset_hmac_secret_cache()
        seal = tmp_path / "holdout_2025-01-01_2025-06-30.json"
        _write_seal(seal, include_signature=False)
        original = seal.read_text()
        result = subprocess.run(
            [
                sys.executable, "-m", "quant_lib.cli.main", "migrate-seals",
                "--seal-dir", str(tmp_path),
                "--dry-run",
            ],
            capture_output=True, text=True, env={**os.environ},
        )
        assert result.returncode == 0, result.stderr
        assert "WOULD MIGRATE" in result.stdout
        # File untouched.
        assert seal.read_text() == original
        # No backup created.
        assert not seal.with_suffix(seal.suffix + ".bak").exists()

    def test_cli_actually_migrates_when_not_dry_run(self, tmp_path, monkeypatch):
        """Without --dry-run, the file is rewritten with a valid signature."""
        monkeypatch.setenv("QUANT_LIB_HMAC_SECRET", _TEST_SECRET)
        _reset_hmac_secret_cache()
        seal = tmp_path / "holdout_2025-01-01_2025-06-30.json"
        _write_seal(seal, include_signature=False)
        result = subprocess.run(
            [
                sys.executable, "-m", "quant_lib.cli.main", "migrate-seals",
                "--seal-dir", str(tmp_path),
                "--yes",
            ],
            capture_output=True, text=True, env={**os.environ},
        )
        assert result.returncode == 0, result.stderr
        assert "MIGRATED" in result.stdout
        new_state = json.loads(seal.read_text())
        assert verify_seal_signature(new_state)
        # Backup exists.
        assert seal.with_suffix(seal.suffix + ".bak").exists()
