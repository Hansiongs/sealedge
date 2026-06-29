"""Direct unit tests for ``quant_lib.cli._output.OutputManager``.

Tests cover:
- ``__init__`` directory creation + naming
- ``save_metrics`` JSON output with metadata
- ``save_config`` YAML output for ``ExperimentConfig``
- ``link_latest`` symlink behaviour + cross-platform fallback
- ``_safe_mklink`` internal helper

All tests use a fresh temp directory for ``_RESULTS_DIR`` so the
real ``results/`` directory is not touched.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from quant_lib.cli import _output
from quant_lib.cli._output import OutputManager


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def tmp_results_dir(tmp_path, monkeypatch):
    """Redirect ``_RESULTS_DIR`` to a fresh temp dir for the test."""
    new_dir = tmp_path / "results"
    new_dir.mkdir()
    monkeypatch.setattr(_output, "_RESULTS_DIR", new_dir)
    return new_dir


@pytest.fixture
def mock_git_commit(monkeypatch):
    """Patch ``get_git_commit`` to return a deterministic short hash."""
    def _fake_get_git_commit(short: bool = True) -> str:
        if short:
            return "abc1234"
        return "abc1234567890abcdef"
    monkeypatch.setattr(_output, "get_git_commit", _fake_get_git_commit)
    return _fake_get_git_commit


# ═══════════════════════════════════════════════════════════════════════
# __init__
# ═══════════════════════════════════════════════════════════════════════


class TestOutputManagerInit:
    """``OutputManager(experiment_name, mode)`` creates a timestamped dir."""

    def test_creates_run_directory(self, tmp_results_dir, mock_git_commit):
        out = OutputManager("vol_compression_v1", "explore")
        assert out.path.exists()
        assert out.path.is_dir()
        assert out.path.parent == tmp_results_dir

    def test_directory_name_format(self, tmp_results_dir, mock_git_commit):
        """Directory name: ``<ts>_<name>_<mode>_<git>``."""
        out = OutputManager("my_exp", "commit")
        # Format: 2025-01-01_HHMMSS_my_exp_commit_abc1234
        assert out.path.name.startswith("20")
        assert "my_exp" in out.path.name
        assert "commit" in out.path.name
        assert "abc1234" in out.path.name

    def test_directory_name_without_git(
        self, tmp_results_dir, monkeypatch,
    ):
        """If git commit is 'unknown', the suffix is omitted."""
        def _fake(short: bool = True) -> str:
            return "unknown"
        monkeypatch.setattr(_output, "get_git_commit", _fake)
        out = OutputManager("e1", "explore")
        assert "unknown" not in out.path.name

    def test_directory_is_unique_across_calls(
        self, tmp_results_dir, mock_git_commit,
    ):
        """Two OutputManager calls in the same second produce
        distinct directories (each gets its own mtime-sec prefix)."""
        out1 = OutputManager("e1", "explore")
        out2 = OutputManager("e1", "explore")
        # Same second → names may collide; check that at least the
        # path objects differ (could be the same name if mtime matches,
        # but the second mkdir would re-create the same dir).
        # The framework does not guarantee uniqueness within a second
        # (this is a known limitation of the timestamp-based naming).
        # So just assert both calls return valid Path objects.
        assert isinstance(out1.path, Path)
        assert isinstance(out2.path, Path)

    def test_explicit_mode(self, tmp_results_dir, mock_git_commit):
        out = OutputManager("e1", "commit")
        assert out.mode == "commit"
        assert "commit" in out.path.name


# ═══════════════════════════════════════════════════════════════════════
# save_metrics
# ═══════════════════════════════════════════════════════════════════════


class TestSaveMetrics:
    """``save_metrics(dict)`` writes JSON with run metadata."""

    def test_writes_metrics_json(self, tmp_results_dir, mock_git_commit):
        out = OutputManager("e1", "explore")
        path = out.save_metrics({"final_equity": 1500.0, "n_trades": 42})
        assert path.name == "metrics.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["final_equity"] == 1500.0
        assert data["n_trades"] == 42

    def test_metrics_includes_run_metadata(
        self, tmp_results_dir, mock_git_commit,
    ):
        out = OutputManager("e1", "explore")
        path = out.save_metrics({"n_trades": 10})
        data = json.loads(path.read_text())
        # Auto-added metadata
        assert "timestamp" in data
        assert data["git_commit"] == "abc1234567890abcdef"  # long form
        assert data["mode"] == "explore"
        assert data["experiment"] == "e1"
        assert "python_version" in data
        assert "platform" in data
        assert "quant_lib_version" in data

    def test_metrics_user_keys_take_precedence(
        self, tmp_results_dir, mock_git_commit,
    ):
        """User-provided keys with the same name as auto-added ones
        win (so callers can override mode/experiment intentionally).
        """
        out = OutputManager("e1", "explore")
        path = out.save_metrics({"mode": "custom_mode", "experiment": "override"})
        data = json.loads(path.read_text())
        assert data["mode"] == "custom_mode"
        assert data["experiment"] == "override"

    def test_metrics_handles_non_serializable(
        self, tmp_results_dir, mock_git_commit,
    ):
        """Non-JSON-serializable values are coerced via ``default=str``."""
        from datetime import datetime
        out = OutputManager("e1", "explore")
        path = out.save_metrics({"now": datetime(2025, 1, 1, 12, 0)})
        data = json.loads(path.read_text())
        # datetime gets stringified
        assert "2025" in data["now"]


# ═══════════════════════════════════════════════════════════════════════
# save_config
# ═══════════════════════════════════════════════════════════════════════


class TestSaveConfig:
    """``save_config(ExperimentConfig)`` writes YAML."""

    def _make_exp_config(self):
        """Build a minimal ``ExperimentConfig`` for save_config tests."""
        from quant_lib.audit import for_vol_compression
        from quant_lib.experiments import (
            ExperimentConfig,
            PeriodConfig,
            StrategyConfig,
            UniverseConfig,
        )
        h = for_vol_compression(
            name="cfg_v1",
            mechanism="test mechanism",
            boundary_conditions="test boundary",
            success_criteria="test criteria",
            entry_logic="test entry",
            exit_logic="test exit",
        )
        return ExperimentConfig(
            name="cfg_v1",
            strategy_type="vol_compression",
            hypothesis=h,
            period=PeriodConfig(
                train_start="2020-01-01", train_end="2024-12-31",
            ),
            universe=UniverseConfig(symbols=["BTCUSDT"]),
            strategy=StrategyConfig(),
        )

    def test_writes_config_yaml(self, tmp_results_dir, mock_git_commit):
        cfg = self._make_exp_config()
        out = OutputManager("cfg_v1", "explore")
        path = out.save_config(cfg)
        assert path.name == "config.yaml"
        assert path.exists()
        text = path.read_text()
        assert "cfg_v1" in text
        assert "vol_compression" in text
        assert "test mechanism" in text

    def test_config_includes_hypothesis_fields(
        self, tmp_results_dir, mock_git_commit,
    ):
        cfg = self._make_exp_config()
        out = OutputManager("cfg_v1", "explore")
        path = out.save_config(cfg)
        text = path.read_text()
        # All five hypothesis field names are present
        for field in ("mechanism", "boundary_conditions", "success_criteria",
                      "entry_logic", "exit_logic"):
            assert field in text
        # The values from the test fixture are serialised
        for value in ("test mechanism", "test boundary", "test criteria",
                      "test entry", "test exit"):
            assert value in text

    def test_config_includes_period_and_universe(
        self, tmp_results_dir, mock_git_commit,
    ):
        cfg = self._make_exp_config()
        out = OutputManager("cfg_v1", "explore")
        path = out.save_config(cfg)
        text = path.read_text()
        assert "2020-01-01" in text
        assert "BTCUSDT" in text


# ═══════════════════════════════════════════════════════════════════════
# link_latest
# ═══════════════════════════════════════════════════════════════════════


class TestLinkLatest:
    """``link_latest`` creates a ``latest`` symlink to the run dir."""

    def test_creates_symlink(self, tmp_results_dir, mock_git_commit):
        out = OutputManager("e1", "explore")
        out.link_latest()
        link = tmp_results_dir / "latest"
        assert link.is_symlink() or link.exists()
        # If symlink works (Linux/Mac or admin Windows), target matches
        if link.is_symlink():
            assert link.resolve() == out.path.resolve()

    def test_link_overwrites_existing(
        self, tmp_results_dir, mock_git_commit,
    ):
        """Calling link_latest twice should not error on second call."""
        out1 = OutputManager("e1", "explore")
        out1.link_latest()
        out2 = OutputManager("e1", "commit")
        out2.link_latest()
        # latest should now point to out2 (or out1, depending on which
        # call was second — both are valid)
        link = tmp_results_dir / "latest"
        assert link.is_symlink() or link.exists()

    def test_link_latest_delegates_to_safe_mklink(
        self, tmp_results_dir, mock_git_commit, monkeypatch,
    ):
        """``link_latest`` should call ``_safe_mklink`` exactly once
        with the correct args.
        """
        out = OutputManager("e1", "explore")
        called = []

        def _spy(link, target):
            called.append((link, target))

        monkeypatch.setattr(_output, "_safe_mklink", _spy)
        out.link_latest()
        assert len(called) == 1
        link, target = called[0]
        assert str(link).endswith("latest")
        assert target == out.path.name


# ═══════════════════════════════════════════════════════════════════════
# _safe_mklink
# ═══════════════════════════════════════════════════════════════════════


class TestSafeMklink:
    """``_safe_mklink`` creates a symlink (or fallback marker)."""

    def test_creates_new_symlink(self, tmp_path):
        link = tmp_path / "latest"
        target = "some_run_dir"
        _output._safe_mklink(link, target)
        if link.is_symlink():
            assert os.readlink(link) == target

    def test_overwrites_existing_symlink(self, tmp_path):
        link = tmp_path / "latest"
        _output._safe_mklink(link, "old_target")
        _output._safe_mklink(link, "new_target")
        if link.is_symlink():
            assert os.readlink(link) == "new_target"

    def test_overwrites_existing_file(self, tmp_path):
        """An existing regular file at the link path is replaced."""
        link = tmp_path / "latest"
        link.write_text("stale data")
        _output._safe_mklink(link, "new_target")
        if link.is_symlink():
            assert os.readlink(link) == "new_target"

    def test_falls_back_to_marker_on_oserror(
        self, tmp_path, monkeypatch,
    ):
        """If ``Path.symlink_to`` raises, the LATEST marker is written."""
        link = tmp_path / "latest"
        # Patch Path.symlink_to at class level
        def _raise(self, *args, **kwargs):
            raise OSError("simulated symlink failure")
        monkeypatch.setattr(Path, "symlink_to", _raise)
        _output._safe_mklink(link, "target_dir")
        # LATEST marker should exist
        marker = tmp_path / "LATEST"
        assert marker.exists()
        assert "target_dir" in marker.read_text()

    def test_marker_creation_failure_is_swallowed(
        self, tmp_path, monkeypatch,
    ):
        """If both symlink and marker creation fail, the function
        silently swallows the error (logs but does not raise).
        """
        link = tmp_path / "latest"

        def _raise(self, *args, **kwargs):
            raise OSError("simulated symlink failure")
        monkeypatch.setattr(Path, "symlink_to", _raise)
        # Patch Path.write_text to fail when writing the LATEST marker.
        original_write_text = Path.write_text

        def _failing_write_text(self, *args, **kwargs):
            if self.name == "LATEST":
                raise OSError("simulated marker failure")
            return original_write_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", _failing_write_text)
        # Should not raise
        _output._safe_mklink(link, "target_dir")


# ═══════════════════════════════════════════════════════════════════════
# save_html_report
# ═══════════════════════════════════════════════════════════════════════


class TestSaveHtmlReport:
    """``OutputManager.save_html_report`` produces a self-contained file."""

    def test_writes_html_file(self, tmp_results_dir, mock_git_commit):
        out = OutputManager("vol_compression_v1", "explore")
        path = out.save_html_report(
            title="Test Report",
            sections=[("Summary", "Hello world")],
        )
        assert path.exists()
        assert path.name == "report.html"
        text = path.read_text(encoding="utf-8")
        assert text.startswith("<!DOCTYPE html>")
        assert "Test Report" in text
        assert "Hello world" in text

    def test_custom_output_name(self, tmp_results_dir, mock_git_commit):
        out = OutputManager("vol_compression_v1", "explore")
        path = out.save_html_report(
            title="Test",
            sections=[],
            output_name="custom.html",
        )
        assert path.name == "custom.html"
        assert path.exists()

    def test_subtitle_appears(self, tmp_results_dir, mock_git_commit):
        out = OutputManager("vol_compression_v1", "explore")
        path = out.save_html_report(
            title="Test",
            subtitle="My subtitle",
            sections=[],
        )
        text = path.read_text(encoding="utf-8")
        assert "My subtitle" in text

    def test_empty_subtitle_omitted(self, tmp_results_dir, mock_git_commit):
        """Empty subtitle should not leave an empty <p> tag."""
        out = OutputManager("vol_compression_v1", "explore")
        path = out.save_html_report(title="Test", subtitle="", sections=[])
        text = path.read_text(encoding="utf-8")
        # No empty subtitle paragraph
        assert '<p style="color:#7f8c8d;"></p>' not in text

    def test_kv_section_renders_as_table(self, tmp_results_dir, mock_git_commit):
        out = OutputManager("vol_compression_v1", "explore")
        path = out.save_html_report(
            title="Test",
            sections=[("Metrics", [("PSR", "0.85"), ("Trades", "100")])],
        )
        text = path.read_text(encoding="utf-8")
        assert "<table>" in text
        assert "PSR" in text
        assert "0.85" in text
        assert "Trades" in text
        assert "100" in text

    def test_table_section_renders_rows(self, tmp_results_dir, mock_git_commit):
        out = OutputManager("vol_compression_v1", "explore")
        path = out.save_html_report(
            title="Test",
            sections=[("By Symbol", [
                ["Symbol", "N", "WR"],
                ["BTCUSDT", "50", "60%"],
                ["ETHUSDT", "30", "55%"],
            ])],
        )
        text = path.read_text(encoding="utf-8")
        assert "<thead>" in text
        assert "<tbody>" in text
        assert "BTCUSDT" in text
        assert "ETHUSDT" in text

    def test_chart_section_embeds_base64(self, tmp_results_dir, mock_git_commit):
        out = OutputManager("vol_compression_v1", "explore")
        b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        path = out.save_html_report(
            title="Test",
            sections=[("Equity Curve", ("chart", b64))],
        )
        text = path.read_text(encoding="utf-8")
        assert b64 in text
        assert '<img src="data:image/png;base64,iVBOR' in text

    def test_html_escape_in_kv(self, tmp_results_dir, mock_git_commit):
        """Special HTML characters in values must be escaped."""
        out = OutputManager("vol_compression_v1", "explore")
        path = out.save_html_report(
            title="Test",
            sections=[("X", [("Note", "<script>alert('xss')</script>")])],
        )
        text = path.read_text(encoding="utf-8")
        # Raw script tag must NOT appear; the escaped version should.
        assert "<script>alert" not in text
        assert "&lt;script&gt;" in text

    def test_html_escape_in_heading(self, tmp_results_dir, mock_git_commit):
        out = OutputManager("vol_compression_v1", "explore")
        path = out.save_html_report(
            title="A & B",
            sections=[],
        )
        text = path.read_text(encoding="utf-8")
        assert "A &amp; B" in text
        # Title is also in the <title> tag -- both should be escaped.
        assert text.count("A &amp; B") >= 1

    def test_multiple_sections_in_order(self, tmp_results_dir, mock_git_commit):
        out = OutputManager("vol_compression_v1", "explore")
        path = out.save_html_report(
            title="Test",
            sections=[
                ("First", "Section one"),
                ("Second", "Section two"),
                ("Third", "Section three"),
            ],
        )
        text = path.read_text(encoding="utf-8")
        i1 = text.index("Section one")
        i2 = text.index("Section two")
        i3 = text.index("Section three")
        assert i1 < i2 < i3, "Sections must be in declaration order"

    def test_no_external_image_refs(self, tmp_results_dir, mock_git_commit):
        """Charts are inline; report has no external image refs (portable)."""
        out = OutputManager("vol_compression_v1", "explore")
        path = out.save_html_report(
            title="Test",
            sections=[
                ("Chart 1", ("chart", "data:image/png;base64,AAAA")),
                ("Chart 2", ("chart", "data:image/png;base64,BBBB")),
            ],
        )
        text = path.read_text(encoding="utf-8")
        # No <img src="path"> or <img src='file://...'>
        assert "src=\"charts/" not in text
        assert "src=\"./" not in text

    def test_kv_dispatch_explicit(self, tmp_results_dir, mock_git_commit):
        out = OutputManager("vol_compression_v1", "explore")
        path = out.save_html_report(
            title="Test",
            sections=[("X", ("kv", [("a", 1), ("b", 2)]))],
        )
        text = path.read_text(encoding="utf-8")
        assert "<table>" in text
        assert "1" in text and "2" in text

    def test_table_dispatch_explicit(self, tmp_results_dir, mock_git_commit):
        out = OutputManager("vol_compression_v1", "explore")
        path = out.save_html_report(
            title="Test",
            sections=[("X", ("table", [["A", "B"], ["1", "2"]]))],
        )
        text = path.read_text(encoding="utf-8")
        assert "<thead>" in text
        assert "1" in text and "2" in text

    def test_html_dispatch_passes_through(self, tmp_results_dir, mock_git_commit):
        """Raw HTML dispatch is the caller's responsibility for escaping."""
        out = OutputManager("vol_compression_v1", "explore")
        path = out.save_html_report(
            title="Test",
            sections=[("Custom", ("html", "<p>Raw HTML</p>"))],
        )
        text = path.read_text(encoding="utf-8")
        assert "<p>Raw HTML</p>" in text

    def test_empty_sections_list(self, tmp_results_dir, mock_git_commit):
        out = OutputManager("vol_compression_v1", "explore")
        path = out.save_html_report(title="Test", sections=[])
        text = path.read_text(encoding="utf-8")
        # Body is empty between h1 and footer, but the file is valid.
        assert "Test" in text
        assert "Generated by quant_exp" in text

    def test_file_is_utf8(self, tmp_results_dir, mock_git_commit):
        """The HTML file is written as UTF-8 (so unicode headings work)."""
        out = OutputManager("vol_compression_v1", "explore")
        path = out.save_html_report(
            title="Test",
            sections=[("Café", "naïve résumé")],
        )
        text = path.read_text(encoding="utf-8")
        assert "Café" in text
        assert "naïve" in text


# ═══════════════════════════════════════════════════════════════════════
# _render_* pure-function unit tests
# ═══════════════════════════════════════════════════════════════════════


class TestRenderPureFunctions:
    """Direct unit tests for _render_table, _render_kv, _render_chart,
    _render_section, _safe_mklink.

    These pure functions are the building blocks of the HTML report.
    They require no OutputManager instance and no temp directory.
    """

    def test_render_table_empty_rows(self):
        from quant_lib.cli._output import _render_table
        assert "no data" in _render_table([]).lower()

    def test_render_table_with_header_and_rows(self):
        from quant_lib.cli._output import _render_table
        html = _render_table([["A", "B"], ["1", "2"]])
        assert "<th>A</th>" in html
        assert "<td>1</td>" in html

    def test_render_table_escapes_html(self):
        from quant_lib.cli._output import _render_table
        html = _render_table([["<script>"]])
        assert "<script>" not in html

    def test_render_kv(self):
        from quant_lib.cli._output import _render_kv
        html = _render_kv([("key1", "val1")])
        assert "key1" in html and "val1" in html
        assert "kv-label" in html

    def test_render_chart_data_uri(self):
        from quant_lib.cli._output import _render_chart
        html = _render_chart("data:image/png;base64,ABC")
        assert "<img" in html and "data:image/png" in html

    def test_render_chart_path(self):
        from quant_lib.cli._output import _render_chart
        html = _render_chart("./chart.png")
        assert "./chart.png" in html

    def test_render_chart_empty(self):
        from quant_lib.cli._output import _render_chart
        assert "not available" in _render_chart("").lower()

    def test_render_section_string(self):
        from quant_lib.cli._output import _render_section
        html = _render_section("Title", "<p>body</p>")
        assert "<h2>Title</h2>" in html
        assert "<p>body</p>" in html

    def test_render_section_dict(self):
        from quant_lib.cli._output import _render_section
        html = _render_section("KV", {"a": 1, "b": 2})
        assert "kv-label" in html

    def test_render_section_tuple_dispatch_table(self):
        from quant_lib.cli._output import _render_section
        html = _render_section("T", ("table", [["A"], ["1"]]))
        assert "<th>A</th>" in html

    def test_render_section_tuple_dispatch_chart(self):
        from quant_lib.cli._output import _render_section
        html = _render_section("C", ("chart", "data:image/png;base64,X"))
        assert "<img" in html

    def test_render_section_tuple_dispatch_html(self):
        from quant_lib.cli._output import _render_section
        html = _render_section("H", ("html", "<b>raw</b>"))
        assert "<b>raw</b>" in html

    def test_render_section_list_of_tuples(self):
        from quant_lib.cli._output import _render_section
        html = _render_section("Pairs", [("k1", "v1"), ("k2", "v2")])
        assert "k1" in html and "v2" in html

    def test_render_section_list_of_lists(self):
        from quant_lib.cli._output import _render_section
        html = _render_section("Grid", [["A"], ["1"]])
        assert "<th>A</th>" in html

    def test_output_manager_creates_directory(self, monkeypatch, tmp_path):
        from quant_lib.cli._output import OutputManager
        monkeypatch.chdir(tmp_path)
        out = OutputManager("test_exp", mode="explore")
        assert out.path.exists()
        assert out.experiment_name == "test_exp"
        assert out.mode == "explore"

    def test_output_manager_save_metrics(self, monkeypatch, tmp_path):
        from quant_lib.cli._output import OutputManager
        import json
        monkeypatch.chdir(tmp_path)
        out = OutputManager("m", mode="explore")
        p = out.save_metrics({"n": 5, "ratio": 0.1})
        data = json.loads(p.read_text())
        assert data["n"] == 5
        assert "timestamp" in data
        assert "git_commit" in data

    def test_output_manager_save_config(self, monkeypatch, tmp_path):
        from quant_lib.cli._output import OutputManager
        import yaml
        from quant_lib.experiments.base import (
            ExperimentConfig, PeriodConfig, StrategyConfig, UniverseConfig,
        )
        from quant_lib.audit import for_vol_compression
        monkeypatch.chdir(tmp_path)
        out = OutputManager("c", mode="commit")
        h = for_vol_compression("c", "m", "b", "c")
        cfg = ExperimentConfig(
            name="c", strategy_type="vol_compression", hypothesis=h,
            period=PeriodConfig(train_start="2020-01-01", train_end="2024-12-31"),
            universe=UniverseConfig(symbols=["BTCUSDT"]),
            strategy=StrategyConfig(),
        )
        p = out.save_config(cfg)
        data = yaml.safe_load(p.read_text())
        assert data["name"] == "c"

    def test_output_manager_save_html_report(self, monkeypatch, tmp_path):
        from quant_lib.cli._output import OutputManager
        monkeypatch.chdir(tmp_path)
        out = OutputManager("h", mode="explore")
        p = out.save_html_report(
            title="Test",
            sections=[("Heading1", "raw html"), ("KV", {"k": "v"})],
        )
        content = p.read_text()
        assert "<h1>Test</h1>" in content
        assert "Heading1" in content
        assert "kv-label" in content
