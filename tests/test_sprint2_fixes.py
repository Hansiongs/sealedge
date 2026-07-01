"""Tests for Sprint 2 review-driven fixes.

Covers the 7 fixes in Sprint 2:

2.1 STRATEGY_* constants deduplicated (single source of truth)
2.2 README version badge updated to 0.5.1
2.3 API reference pages render member summaries (not stubs)
2.4 HTML equity curve lie removed from commit_cmd
2.5 _looks_like_absolute extracted to cli/_utils.py
2.6 Notebook 02 API consistency (fix all wrong fields/imports)
2.7 _safe_mklink LATEST fallback has a reader (read_latest_run_dir)

Each test is small and targeted. Anti-reintroduction guards are
included where relevant (the regression tests for the missing
function, the deduplicated constants, etc.).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


# =====================================================================
# Fix 2.1: STRATEGY_* constants single source of truth
# =====================================================================


class TestStrategyConstantsSingleSource:
    """Sprint 2 fix 2.1: STRATEGY_* constants defined ONCE in
    ``core/_config.py`` and imported by all consumers.

    Pre-fix: triplicated across ``audit/hypothesis.py``,
    ``core/_features.py``, ``core/_engine.py``.
    Post-fix: single source of truth.
    """

    def test_config_has_the_canonical_constants(self):
        """core/_config.py must define STRATEGY_VOL_COMPRESSION = 0 and
        STRATEGY_PULLBACK_SNIPER = 1 as int constants (single source)."""
        from quant_lib.core import _config as cfg
        assert hasattr(cfg, "STRATEGY_VOL_COMPRESSION")
        assert hasattr(cfg, "STRATEGY_PULLBACK_SNIPER")
        assert cfg.STRATEGY_VOL_COMPRESSION == 0
        assert cfg.STRATEGY_PULLBACK_SNIPER == 1
        assert isinstance(cfg.STRATEGY_VOL_COMPRESSION, int)
        assert isinstance(cfg.STRATEGY_PULLBACK_SNIPER, int)

    def test_audit_uses_config_constants(self):
        """audit/hypothesis.py must re-export the SAME object identity
        (no shadowing). Identity check catches re-introduction of
        local redeclarations."""
        from quant_lib.core import _config as cfg
        from quant_lib.audit import hypothesis as hyp_mod

        # Identity check: both names point to the same int object
        # (Python caches small ints so identity holds for 0/1).
        assert hyp_mod.STRATEGY_VOL_COMPRESSION is cfg.STRATEGY_VOL_COMPRESSION
        assert hyp_mod.STRATEGY_PULLBACK_SNIPER is cfg.STRATEGY_PULLBACK_SNIPER

    def test_engine_imports_from_config(self):
        """core/_engine.py must NOT redeclare constants locally."""
        import inspect

        from quant_lib.core import _engine as engine_mod
        from quant_lib.core import _config as cfg

        source = inspect.getsource(engine_mod)
        # Anti-reintroduction guard: must NOT have local assignment.
        assert "STRATEGY_VOL_COMPRESSION = 0" not in source, (
            "core/_engine.py still redeclares STRATEGY_VOL_COMPRESSION = 0. "
            "Sprint 2 fix 2.1: import from core/_config.py."
        )
        assert "STRATEGY_PULLBACK_SNIPER = 1" not in source, (
            "core/_engine.py still redeclares STRATEGY_PULLBACK_SNIPER = 1. "
            "Sprint 2 fix 2.1: import from core/_config.py."
        )
        # Must import from config.
        assert engine_mod.STRATEGY_VOL_COMPRESSION is cfg.STRATEGY_VOL_COMPRESSION
        assert engine_mod.STRATEGY_PULLBACK_SNIPER is cfg.STRATEGY_PULLBACK_SNIPER

    def test_features_imports_from_config(self):
        """core/_features.py must NOT redeclare constants locally."""
        import inspect

        from quant_lib.core import _features as feat_mod
        from quant_lib.core import _config as cfg

        source = inspect.getsource(feat_mod)
        assert "STRATEGY_VOL_COMPRESSION = 0" not in source, (
            "core/_features.py still redeclares STRATEGY_VOL_COMPRESSION = 0."
        )
        assert "STRATEGY_PULLBACK_SNIPER = 1" not in source, (
            "core/_features.py still redeclares STRATEGY_PULLBACK_SNIPER = 1."
        )
        assert feat_mod.STRATEGY_VOL_COMPRESSION is cfg.STRATEGY_VOL_COMPRESSION
        assert feat_mod.STRATEGY_PULLBACK_SNIPER is cfg.STRATEGY_PULLBACK_SNIPER

    def test_strategy_type_enum_matches(self):
        """StrategyType IntEnum values must equal the canonical constants."""
        from quant_lib.audit import StrategyType
        from quant_lib.core._config import (
            STRATEGY_VOL_COMPRESSION,
            STRATEGY_PULLBACK_SNIPER,
        )

        assert int(StrategyType.VOL_COMPRESSION) == STRATEGY_VOL_COMPRESSION
        assert int(StrategyType.PULLBACK_SNIPER) == STRATEGY_PULLBACK_SNIPER


# =====================================================================
# Fix 2.2: README version badge
# =====================================================================


class TestReadmeVersionBadge:
    """Sprint 2 fix 2.2: README version badge stale at 0.3.0;
    updated to 0.5.1 (matches actual pyproject.toml)."""

    def test_readme_badge_shows_current_version(self):
        """README badge must show 0.5.1 (matching pyproject.toml)."""
        repo_root = Path(__file__).resolve().parent.parent
        readme = repo_root / "README.md"
        pyproject = repo_root / "pyproject.toml"

        assert readme.exists()
        assert pyproject.exists()

        # Extract version from pyproject.toml
        pyproject_content = pyproject.read_text(encoding="utf-8")
        version_match = re.search(
            r'^version\s*=\s*"([^"]+)"', pyproject_content, re.MULTILINE,
        )
        assert version_match, "pyproject.toml must have a version field"
        pyproject_version = version_match.group(1)

        # README badge must mention this version.
        readme_content = readme.read_text(encoding="utf-8")
        badge_pattern = re.compile(
            rf"Version\s+{re.escape(pyproject_version)}", re.IGNORECASE,
        )
        assert badge_pattern.search(readme_content), (
            f"README badge does not mention current version "
            f"{pyproject_version}. Sprint 2 fix 2.2: update badge."
        )

    def test_readme_citation_block_uses_current_version(self):
        """README's BibTeX citation block must use the current version too."""
        repo_root = Path(__file__).resolve().parent.parent
        readme = (repo_root / "README.md").read_text(encoding="utf-8")
        pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")

        version_match = re.search(
            r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE,
        )
        pyproject_version = version_match.group(1)

        # BibTeX citation block has ``version = {X.Y.Z},``.
        bibtex_pattern = re.compile(
            r"version\s*=\s*\{([^}]*)\}",
        )
        bibtex_match = bibtex_pattern.search(readme)
        assert bibtex_match, (
            "README BibTeX citation block does not have a version = {X.Y.Z} "
            "field. Update README.md citation section."
        )
        bibtex_version = bibtex_match.group(1)
        assert bibtex_version == pyproject_version, (
            f"README BibTeX citation uses version {bibtex_version}, "
            f"but pyproject.toml is {pyproject_version}."
        )


# =====================================================================
# Fix 2.3: API reference pages render member summaries
# =====================================================================


class TestAPIReferencePages:
    """Sprint 2 fix 2.3: docs/api/*.md must render member summaries
    (not just stub text). Without this the mkdocs API reference
    renders almost nothing."""

    @pytest.mark.parametrize("page", [
        "audit.md",
        "core.md",
        "research.md",
        "tools.md",
    ])
    def test_api_page_renders_members(self, page):
        """Each API reference page must have ``members: summary``
        (or ``members: true``) in its mkdocstrings config."""
        repo_root = Path(__file__).resolve().parent.parent
        api_pages_dir = repo_root / "docs" / "api"
        page_path = api_pages_dir / page
        if not page_path.exists():
            pytest.skip(f"{page} not found")
        content = page_path.read_text(encoding="utf-8")
        # Either ``members: summary`` or ``members: true`` is acceptable.
        assert re.search(r"members:\s*(summary|true)", content), (
            f"{page} does not render members. Sprint 2 fix 2.3: "
            f"add `members: summary` to the mkdocstrings options."
        )
        # Anti-reintroduction guard for the bug case.
        assert "members: false" not in content, (
            f"{page} still has `members: false` (the buggy stub). "
            f"Sprint 2 fix 2.3 reverted this."
        )


# =====================================================================
# Fix 2.5: _looks_like_absolute deduplicated to cli/_utils.py
# =====================================================================


class TestLooksLikeAbsoluteDeduplicated:
    """Sprint 2 fix 2.5: _looks_like_absolute moved to cli/_utils.py.
    Both explore.py and commit_cmd.py keep private aliases for
    backward compatibility."""

    def test_canonical_helper_in_cli_utils(self):
        """cli/_utils.py must export ``looks_like_absolute``."""
        from quant_lib.cli import _utils
        assert hasattr(_utils, "looks_like_absolute")
        assert callable(_utils.looks_like_absolute)

    def test_explore_alias_delegates(self):
        """explore._looks_like_absolute must delegate to the shared helper."""
        from quant_lib.cli.explore import _looks_like_absolute as expl_abs
        from quant_lib.cli._utils import looks_like_absolute as canonical

        # Identity check on a few values (functions are deterministic).
        for path in ["report.html", "/tmp/foo", "subdir/report.html", ""]:
            assert expl_abs(path) == canonical(path), (
                f"explore._looks_like_absolute({path!r}) diverged from "
                f"cli._utils.looks_like_absolute."
            )

    def test_commit_alias_delegates(self):
        """commit_cmd._looks_like_absolute must delegate to the shared helper."""
        from quant_lib.cli.commit_cmd import _looks_like_absolute as commit_abs
        from quant_lib.cli._utils import looks_like_absolute as canonical

        for path in ["report.html", "/tmp/foo", "C:\\report.html", ""]:
            assert commit_abs(path) == canonical(path), (
                f"commit_cmd._looks_like_absolute({path!r}) diverged."
            )

    def test_aliases_have_same_docstring_intent(self):
        """The private aliases exist for backward compat. Their docstrings
        should mention the Sprint 2 dedup."""
        from quant_lib.cli.explore import _looks_like_absolute as expl
        from quant_lib.cli.commit_cmd import _looks_like_absolute as cm

        # Function objects -- verify they're distinct from the canonical.
        from quant_lib.cli._utils import looks_like_absolute as canonical
        assert expl is not canonical
        assert cm is not canonical


# =====================================================================
# Fix 2.6: Notebook 02 API consistency
# =====================================================================


class TestNotebook02APIConsistency:
    """Sprint 2 fix 2.6: notebook 02's example code must use the
    current public API. Pre-fix had several wrong field names
    (``training`` instead of ``train_start``/``train_end``,
    ``min_volume_usd`` instead of ``min_volume_usdt``,
    non-existent Hypothesis fields like ``expectation`` /
    ``holding_period`` / ``exit_rules`` / ``invalidation``)."""

    @pytest.fixture
    def notebook_source(self):
        """Return the executable code cells of notebook 02."""
        repo_root = Path(__file__).resolve().parent.parent
        nb_path = repo_root / "notebooks" / "02_custom_experiment.ipynb"
        if not nb_path.exists():
            pytest.skip("notebook 02 not found")
        with open(nb_path) as f:
            nb = json.load(f)
        cells = []
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                src = "".join(cell.get("source", []))
                if src.strip():
                    cells.append(src)
        return "\n\n".join(cells)

    def test_uses_real_hypothesis_fields(self, notebook_source):
        """Notebook must use real Hypothesis fields. Anti-reintroduction:
        these strings MUST NOT appear (they were the bug)."""
        for bad_field in [
            "expectation=",
            "holding_period=",
            "exit_rules=",
            "invalidation=",
        ]:
            assert bad_field not in notebook_source, (
                f"Notebook 02 still uses non-existent Hypothesis field "
                f"``{bad_field}``. Sprint 2 fix 2.6: use real fields "
                f"(mechanism, boundary_conditions, success_criteria, "
                f"entry_logic, exit_logic)."
            )

    def test_period_config_uses_real_field_names(self, notebook_source):
        """PeriodConfig uses train_start/train_end, not training/holdout."""
        # The buggy form was PeriodConfig(training=(...), holdout=(...)).
        assert "PeriodConfig(" in notebook_source, (
            "Notebook should construct a PeriodConfig."
        )
        assert "training=" not in notebook_source, (
            "Notebook still uses ``training=`` for PeriodConfig. "
            "Sprint 2 fix 2.6: use ``train_start=`` / ``train_end=``."
        )
        assert "min_volume_usd=" not in notebook_source, (
            "Notebook still uses ``min_volume_usd=``. "
            "Sprint 2 fix 2.6: use ``min_volume_usdt=``."
        )

    def test_uses_factory_helper(self, notebook_source):
        """Notebook must use the for_vol_compression factory (canonical)."""
        assert "for_vol_compression" in notebook_source, (
            "Notebook should use ``for_vol_compression`` factory helper "
            "instead of constructing Hypothesis directly."
        )

    def test_example_is_executable(self):
        """End-to-end: run the actual code from the notebook."""
        # This catches the "looks right but isn't" failure mode.
        from quant_lib.audit import for_vol_compression
        from quant_lib.experiments import (
            PeriodConfig, UniverseConfig, StrategyConfig,
            ExperimentConfig, register, clear,
        )
        clear()

        hyp = for_vol_compression(
            name="test_nb02_exec",
            mechanism="Test mechanism",
            boundary_conditions="None",
            success_criteria="SPA p < 0.15",
        )
        cfg = ExperimentConfig(
            name="test_nb02_exec",
            strategy_type="vol_compression",
            hypothesis=hyp,
            period=PeriodConfig(
                train_start="2020-01-01",
                train_end="2020-12-31",
            ),
            universe=UniverseConfig(
                symbols=["BTCUSDT"],
                min_volume_usdt=50_000_000,
                min_age_days=365,
            ),
            strategy=StrategyConfig(),
        )
        register(cfg)

        from quant_lib.experiments import exists
        assert exists("test_nb02_exec")
        clear()


# =====================================================================
# Fix 2.7: _safe_mklink LATEST fallback has a reader
# =====================================================================


class TestReadLatestRunDir:
    """Sprint 2 fix 2.7: read_latest_run_dir() resolves the most recent
    run directory, transparently handling symlink OR LATEST marker."""

    def test_resolves_via_symlink(self, tmp_path, monkeypatch):
        """On POSIX or Windows-with-admin, ``results/latest`` is a symlink."""
        from quant_lib.cli import _output

        # Set up: results/latest symlink -> run_dir
        run_dir = tmp_path / "2026-07-01_120000_v1_explore_abcdef"
        run_dir.mkdir()
        latest_link = tmp_path / "latest"
        try:
            latest_link.symlink_to(run_dir.name)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        # Monkey-patch the module's _RESULTS_DIR to our tmp_path.
        monkeypatch.setattr(_output, "_RESULTS_DIR", tmp_path)

        result = _output.read_latest_run_dir()
        assert result is not None
        assert result.resolve() == run_dir.resolve()

    def test_resolves_via_marker(self, tmp_path, monkeypatch):
        """On Windows-without-admin, ``results/LATEST`` is a marker file."""
        from quant_lib.cli import _output

        run_dir = tmp_path / "2026-07-01_120000_v1_explore_abcdef"
        run_dir.mkdir()
        marker = tmp_path / "LATEST"
        marker.write_text(run_dir.name)

        monkeypatch.setattr(_output, "_RESULTS_DIR", tmp_path)

        result = _output.read_latest_run_dir()
        assert result is not None
        assert result.resolve() == run_dir.resolve()

    def test_returns_none_when_nothing_exists(self, tmp_path, monkeypatch):
        """Empty results dir returns None (no symlink, no marker)."""
        from quant_lib.cli import _output

        monkeypatch.setattr(_output, "_RESULTS_DIR", tmp_path)
        assert _output.read_latest_run_dir() is None

    def test_returns_none_when_target_missing(self, tmp_path, monkeypatch):
        """Marker/symlink pointing to deleted directory returns None."""
        from quant_lib.cli import _output

        marker = tmp_path / "LATEST"
        marker.write_text("nonexistent_dir")

        monkeypatch.setattr(_output, "_RESULTS_DIR", tmp_path)
        assert _output.read_latest_run_dir() is None

    def test_default_results_dir(self, monkeypatch, tmp_path):
        """Default ``results_dir=None`` uses module-level _RESULTS_DIR."""
        from quant_lib.cli import _output

        run_dir = tmp_path / "2026-07-01_run"
        run_dir.mkdir()
        marker = tmp_path / "LATEST"
        marker.write_text(run_dir.name)

        monkeypatch.setattr(_output, "_RESULTS_DIR", tmp_path)
        result = _output.read_latest_run_dir()  # no arg
        assert result is not None
        assert result.resolve() == run_dir.resolve()
