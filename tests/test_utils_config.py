"""Tests for ``quant_lib.utils.config`` (Phase 4 — v0.5.0)."""

import os
import tempfile
from pathlib import Path

import pytest

from quant_lib.utils.config import (
    find_repo_root,
    load_env_file,
    get_hmac_secret_with_fallback,
)


class TestFindRepoRoot:
    """find_repo_root: locate the project root via pyproject.toml."""

    def test_finds_repo_root_from_cwd(self):
        """From cwd, find_repo_root must locate the project root."""
        # We are running tests from the project root, so this should
        # return the project root (parent of quant_lib/).
        root = find_repo_root()
        assert (root / "pyproject.toml").exists()

    def test_returns_start_when_no_pyproject_toml(self, tmp_path):
        """If no pyproject.toml is found in any parent, return start."""
        # Create a deep temp directory with no pyproject.toml
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        result = find_repo_root(start=deep)
        assert result == deep


class TestLoadEnvFile:
    """load_env_file: parse .env into a dict."""

    def test_loads_simple_key_value(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY1=value1\nKEY2=value2\n")
        result = load_env_file(env_path=env_file)
        assert result == {"KEY1": "value1", "KEY2": "value2"}

    def test_ignores_comments_and_blanks(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# This is a comment\n"
            "\n"
            "KEY1=value1\n"
            "  # Indented comment\n"
            "KEY2=value2\n"
        )
        result = load_env_file(env_path=env_file)
        assert result == {"KEY1": "value1", "KEY2": "value2"}

    def test_strips_quotes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            'KEY_DOUBLE="double-quoted"\n'
            "KEY_SINGLE='single-quoted'\n"
            "KEY_NONE=no-quotes\n"
        )
        result = load_env_file(env_path=env_file)
        assert result["KEY_DOUBLE"] == "double-quoted"
        assert result["KEY_SINGLE"] == "single-quoted"
        assert result["KEY_NONE"] == "no-quotes"

    def test_missing_file_returns_empty_dict(self, tmp_path):
        env_file = tmp_path / "nonexistent.env"
        result = load_env_file(env_path=env_file)
        assert result == {}

    def test_handles_equals_in_value(self, tmp_path):
        """Values with '=' should preserve the part after first '='."""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=value=with=equals\n")
        result = load_env_file(env_path=env_file)
        # str.partition('=') returns ('KEY', '=', 'value=with=equals')
        assert result["KEY"] == "value=with=equals"

    def test_handles_crlf_line_endings(self, tmp_path):
        """Windows-style CRLF line endings must be parsed correctly."""
        env_file = tmp_path / ".env"
        # Write with explicit CRLF (Python text mode converts to LF on
        # most platforms, so write bytes directly)
        env_file.write_bytes(b"KEY1=value1\r\nKEY2=value2\r\n# comment\r\n")
        result = load_env_file(env_path=env_file)
        assert result == {"KEY1": "value1", "KEY2": "value2"}

    def test_handles_empty_value(self, tmp_path):
        """KEY= (no value) should yield empty string."""
        env_file = tmp_path / ".env"
        env_file.write_text("EMPTY_KEY=\nANOTHER=nonempty\n")
        result = load_env_file(env_path=env_file)
        assert result["EMPTY_KEY"] == ""
        assert result["ANOTHER"] == "nonempty"

    def test_handles_spaces_around_equals(self, tmp_path):
        """KEY = value (spaces around =) must be stripped correctly."""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY1 = value with spaces\nKEY2=value2\n")
        result = load_env_file(env_path=env_file)
        assert result["KEY1"] == "value with spaces"
        assert result["KEY2"] == "value2"

    def test_handles_unicode_values(self, tmp_path):
        """Unicode values must be preserved."""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=héllo_wörld_🚀\n", encoding="utf-8")
        result = load_env_file(env_path=env_file)
        assert result["KEY"] == "héllo_wörld_🚀"


class TestGetHmacSecretWithFallback:
    """get_hmac_secret_with_fallback: env var → .env file."""

    def test_raises_when_secret_not_set_anywhere(self, tmp_path, monkeypatch):
        """If neither env var nor .env has the secret, raise RuntimeError."""
        # Clear env var
        monkeypatch.delenv("QUANT_LIB_HMAC_SECRET", raising=False)
        # Point load_env_file to an empty dir
        monkeypatch.setattr(
            "quant_lib.utils.config.find_repo_root",
            lambda: tmp_path,
        )
        # Clear the cached secret in holdout.py (if any)
        from quant_lib.audit import holdout
        if hasattr(holdout.get_hmac_secret, "_cached"):
            delattr(holdout.get_hmac_secret, "_cached")
        with pytest.raises(RuntimeError, match="HMAC"):
            get_hmac_secret_with_fallback()

    def test_loads_from_env_var(self, monkeypatch):
        """When env var is set, return it directly."""
        monkeypatch.setenv("QUANT_LIB_HMAC_SECRET", "x" * 32)
        # Clear cache
        from quant_lib.audit import holdout
        if hasattr(holdout.get_hmac_secret, "_cached"):
            delattr(holdout.get_hmac_secret, "_cached")
        result = get_hmac_secret_with_fallback()
        assert result == "x" * 32

    def test_loads_from_env_file_when_env_var_missing(self, tmp_path, monkeypatch):
        """When env var is missing but .env has it, load from .env."""
        monkeypatch.delenv("QUANT_LIB_HMAC_SECRET", raising=False)
        env_file = tmp_path / ".env"
        # HMAC secret must be >= 32 chars per holdout.py validation
        secret_in_env = "x" * 32
        env_file.write_text(f"QUANT_LIB_HMAC_SECRET={secret_in_env}\n")
        monkeypatch.setattr(
            "quant_lib.utils.config.find_repo_root",
            lambda: tmp_path,
        )
        # Clear cache
        from quant_lib.audit import holdout
        if hasattr(holdout.get_hmac_secret, "_cached"):
            delattr(holdout.get_hmac_secret, "_cached")
        result = get_hmac_secret_with_fallback()
        assert result == secret_in_env
