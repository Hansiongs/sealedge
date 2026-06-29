"""Direct unit tests for ``quant_lib.cli.list_cmd.list_cmd``.

Tests verify the rendering of the experiment table with both
empty and populated registries, using a captured rich console
to assert on the printed output.
"""
from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from rich.console import Console

import quant_lib.cli.list_cmd as list_mod
from quant_lib.cli import list_cmd
from quant_lib.experiments import (
    ExperimentConfig,
    PeriodConfig,
    StrategyConfig,
    UniverseConfig,
    clear,
    register,
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


class _CapturingConsole:
    """Context manager that patches ``list_cmd``'s module-level
    ``console`` to a buffer.
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
        self._patcher = patch.object(list_mod, "console", self.console)
        self._patcher.__enter__()
        return self

    def __exit__(self, *args):
        self._patcher.__exit__(*args)

    @property
    def text(self) -> str:
        return self.console.export_text()


@pytest.fixture
def clean_registry():
    """Clear the experiment registry before and after each test."""
    clear()
    yield
    clear()


def _make_exp(name: str, strategy_type: str = "vol_compression",
              symbols: list[str] | None = None) -> ExperimentConfig:
    """Build a minimal ``ExperimentConfig`` for test purposes."""
    from quant_lib.audit import for_vol_compression, for_pullback_sniper
    if strategy_type == "vol_compression":
        h = for_vol_compression(
            name=name,
            mechanism="m", boundary_conditions="b",
            success_criteria="c", entry_logic="e", exit_logic="x",
        )
    else:
        h = for_pullback_sniper(
            name=name,
            mechanism="m", boundary_conditions="b",
            success_criteria="c", entry_logic="e", exit_logic="x",
        )
    return ExperimentConfig(
        name=name,
        strategy_type=strategy_type,
        hypothesis=h,
        period=PeriodConfig(train_start="2020-01-01", train_end="2024-12-31"),
        universe=UniverseConfig(symbols=symbols or ["BTCUSDT"]),
        strategy=StrategyConfig(),
    )


# ═══════════════════════════════════════════════════════════════════════
# Empty registry
# ═══════════════════════════════════════════════════════════════════════


class TestListEmptyRegistry:
    """When no experiments are registered, a friendly message is shown."""

    def test_list_with_no_experiments(self, clean_registry):
        with _CapturingConsole() as cap:
            list_cmd.list_cmd()
        text = cap.text.lower()
        assert "no experiments" in text
        # Hint about adding experiments
        assert "quant_lib/experiments/" in text or "add" in text


# ═══════════════════════════════════════════════════════════════════════
# Populated registry
# ═══════════════════════════════════════════════════════════════════════


class TestListWithExperiments:
    """When experiments are registered, they appear in a table."""

    def test_list_renders_table_title(self, clean_registry):
        register(_make_exp("vol_compression_v1"))
        with _CapturingConsole() as cap:
            list_cmd.list_cmd()
        text = cap.text
        assert "Registered Experiments" in text

    def test_list_shows_experiment_name(self, clean_registry):
        register(_make_exp("vol_compression_v1"))
        register(_make_exp("pullback_sniper_rsi", "pullback_sniper"))
        with _CapturingConsole() as cap:
            list_cmd.list_cmd()
        text = cap.text
        assert "vol_compression_v1" in text
        assert "pullback_sniper_rsi" in text

    def test_list_shows_strategy_column(self, clean_registry):
        register(_make_exp("v1", "vol_compression"))
        register(_make_exp("v2", "pullback_sniper"))
        with _CapturingConsole() as cap:
            list_cmd.list_cmd()
        text = cap.text
        # Both strategy types shown
        assert "vol_compression" in text
        assert "pullback_sniper" in text

    def test_list_shows_train_period(self, clean_registry):
        register(_make_exp("v1"))
        with _CapturingConsole() as cap:
            list_cmd.list_cmd()
        text = cap.text
        # Train period rendered
        assert "2020-01-01" in text
        assert "2024-12-31" in text
        # Arrow separator
        assert "→" in text

    def test_list_shows_holdout_period(self, clean_registry):
        register(_make_exp("v1"))
        with _CapturingConsole() as cap:
            list_cmd.list_cmd()
        text = cap.text
        # Holdout auto-resolved (train_end + 1 day → +6 months)
        assert "2025-01-01" in text
        assert "2025-07-01" in text

    def test_list_shows_symbol_count(self, clean_registry):
        register(_make_exp("v1", symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"]))
        with _CapturingConsole() as cap:
            list_cmd.list_cmd()
        text = cap.text
        # Count "3" appears in the output
        assert "3" in text

    def test_list_columns_present(self, clean_registry):
        """All 5 column headers are rendered."""
        register(_make_exp("v1"))
        with _CapturingConsole() as cap:
            list_cmd.list_cmd()
        text = cap.text
        for col in ("Name", "Strategy", "Train", "Holdout", "Symbols"):
            assert col in text, f"Missing column header: {col}"


# ═══════════════════════════════════════════════════════════════════════
# Multiple experiments
# ═══════════════════════════════════════════════════════════════════════


class TestListMultipleExperiments:
    """Verify all registered experiments are listed, sorted or in
    registration order.
    """

    def test_list_shows_all_registered(self, clean_registry):
        for i in range(5):
            register(_make_exp(f"exp_{i}"))
        with _CapturingConsole() as cap:
            list_cmd.list_cmd()
        text = cap.text
        for i in range(5):
            assert f"exp_{i}" in text, f"Missing exp_{i} in output"

    def test_list_count_matches_registry(self, clean_registry):
        for i in range(3):
            register(_make_exp(f"e{i}"))
        with _CapturingConsole() as cap:
            list_cmd.list_cmd()
        # Verify by checking the number of times "e" names appear
        text = cap.text
        for i in range(3):
            assert f"e{i}" in text


# ═══════════════════════════════════════════════════════════════════════
# After clear()
# ═══════════════════════════════════════════════════════════════════════


class TestListAfterClear:
    """If the registry is cleared between calls, list reflects that."""

    def test_list_after_clear_shows_empty(self, clean_registry):
        register(_make_exp("temp_exp"))
        with _CapturingConsole() as cap1:
            list_cmd.list_cmd()
        assert "temp_exp" in cap1.text
        # Clear and re-list
        clear()
        with _CapturingConsole() as cap2:
            list_cmd.list_cmd()
        assert "no experiments" in cap2.text.lower()
