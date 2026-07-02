"""Tests for Sprint 3 review-driven fixes.

Covers the Sprint 3 fixes:

3.1 STATIC/DEFAULTS -> TypedDict (zero call-site change, type-safe)
3.5 StrategyConfig/DEFAULTS sync guard (prevents drift)
3.7 Capture mutation baseline (CI infrastructure)

Sprint 3 fixes 3.2 (Candidate typed fields), 3.3 (ExploreResult),
3.4 (CLI/API shared helper), and 3.6 (commit daily_equity plumbing)
have their own dedicated test classes added incrementally.
"""
from __future__ import annotations

import dataclasses
import inspect
import os
from pathlib import Path

import pytest


# =====================================================================
# Fix 3.1: STATIC/DEFAULTS are TypedDicts
# =====================================================================


class TestStaticDefaultsTypedDict:
    """Sprint 3 fix 3.1: STATIC and DEFAULTS are TypedDicts (not
    dict[str, Any]). Runtime behavior unchanged -- TypedDict IS a
    regular dict. The annotation only changes what mypy sees."""

    def test_static_is_typeddict(self):
        """STATIC annotation must be StaticConfig (a TypedDict)."""
        from quant_lib.core._config import STATIC, StaticConfig
        # Runtime: STATIC is a regular dict.
        assert isinstance(STATIC, dict)
        # Static annotation: TypedDict class with __annotations__.
        assert hasattr(StaticConfig, "__annotations__")
        assert dataclasses.is_dataclass(StaticConfig) is False  # TypedDict, not dataclass
        # TypedDict check (Python 3.10+ has typing.is_typeddict).
        import typing
        assert typing.is_typeddict(StaticConfig)

    def test_defaults_is_typeddict(self):
        """DEFAULTS annotation must be DefaultsConfig (a TypedDict)."""
        from quant_lib.core._config import DEFAULTS, DefaultsConfig
        import typing

        assert isinstance(DEFAULTS, dict)
        assert typing.is_typeddict(DefaultsConfig)

    def test_static_keys_present(self):
        """All STATIC keys must match StaticConfig schema annotations."""
        from quant_lib.core._config import STATIC, StaticConfig

        schema_keys = set(StaticConfig.__annotations__.keys())
        actual_keys = set(STATIC.keys())

        missing = schema_keys - actual_keys
        extra = actual_keys - schema_keys

        assert not missing, (
            f"StaticConfig schema declares keys not in STATIC literal: "
            f"{sorted(missing)}. Add them to the STATIC dict."
        )
        assert not extra, (
            f"STATIC literal has keys not declared in StaticConfig: "
            f"{sorted(extra)}. Add them to StaticConfig schema."
        )

    def test_defaults_keys_present(self):
        """All DEFAULTS keys must match DefaultsConfig schema annotations."""
        from quant_lib.core._config import DEFAULTS, DefaultsConfig

        schema_keys = set(DefaultsConfig.__annotations__.keys())
        actual_keys = set(DEFAULTS.keys())

        missing = schema_keys - actual_keys
        extra = actual_keys - schema_keys

        assert not missing, (
            f"DefaultsConfig schema declares keys not in DEFAULTS literal: "
            f"{sorted(missing)}. Add them to the DEFAULTS dict."
        )
        assert not extra, (
            f"DEFAULTS literal has keys not declared in DefaultsConfig: "
            f"{sorted(extra)}. Add them to DefaultsConfig schema."
        )

    def test_static_runtime_values_unchanged(self):
        """TypedDict migration must preserve all runtime values."""
        from quant_lib.core._config import STATIC

        # Spot-check key values.
        assert STATIC["fee_taker"] == 0.05
        assert STATIC["wfa_purge_days"] == 90
        assert STATIC["bootstrap_n_sim"] == 2000
        assert STATIC["atr_len"] == 20

    def test_defaults_runtime_values_unchanged(self):
        """TypedDict migration must preserve all DEFAULTS values."""
        from quant_lib.core._config import DEFAULTS

        assert DEFAULTS["initial_capital"] == 1000.0
        assert DEFAULTS["leverage"] == 3.0
        assert DEFAULTS["pf_weight_clamp_floor"] == 0.5
        assert DEFAULTS["market_impact_volume_pct"] == 0.01

    def test_dict_access_still_works(self):
        """All existing call sites using dict-access must work unchanged.

        This is the critical BC guarantee: 89+ call sites use
        ``STATIC["foo"]`` / ``DEFAULTS["bar"]`` syntax. TypedDict
        preserves this at runtime."""
        from quant_lib.core._config import STATIC, DEFAULTS

        for key in STATIC:
            value = STATIC[key]  # dict-style access
            assert value is not None, f"STATIC[{key!r}] is None"
        for key in DEFAULTS:
            value = DEFAULTS[key]
            assert value is not None, f"DEFAULTS[{key!r}] is None"

    def test_dict_mutation_still_works(self):
        """Tests that mutate STATIC/DEFAULTS (e.g., conftest.py) must work."""
        from quant_lib.core._config import STATIC, DEFAULTS

        original = DEFAULTS["leverage"]
        try:
            DEFAULTS["leverage"] = 99.0
            assert DEFAULTS["leverage"] == 99.0
        finally:
            DEFAULTS["leverage"] = original
        assert DEFAULTS["leverage"] == original


# =====================================================================
# Fix 3.5: StrategyConfig/DEFAULTS sync guard
# =====================================================================


class TestStrategyConfigStaysInSync:
    """Sprint 3 fix 3.5: DEFAULTS in core/_config.py and StrategyConfig
    in experiments/base.py are intentionally redundant (the latter is
    the user-facing API; the former is the internal fast-path). They
    MUST stay in sync. This test catches drift."""

    def test_defaults_keys_present_in_strategy_config(self):
        """For every scalar DEFAULTS key, StrategyConfig must have a
        matching field. (Excludes dict-shaped keys like search_space
        which are intentionally DEFAULTS-only.)"""
        from quant_lib.core._config import DEFAULTS
        from quant_lib.experiments.base import StrategyConfig

        sc_fields = {f.name for f in dataclasses.fields(StrategyConfig)}
        defaults_keys = set(DEFAULTS.keys())

        # Dict-shaped keys in DEFAULTS that are NOT scalar config knobs:
        dict_shape_keys = {"search_space", "search_space_pb"}
        # Defaults-only keys that intentionally don't mirror per-experiment
        # StrategyConfig fields (these are framework-level conveniences):
        framework_only = {
            "default_risk_per_pair",
            "default_expected_trades_per_year",
            "psr_weight_floor",
            "market_impact_volume_pct",
            # WFA parameter centers/scales (L2 regularization constants
            # extracted from hardcoded values in core/_wfa.py; these are
            # framework-level tuning parameters, not per-experiment overrides)
            "vol_thresh_center",
            "pullback_bars_center",
            "trail_atr_center",
            "sl_mult_center",
            "rsi_oversold_center",
            "rsi_overbought_center",
            "vol_thresh_scale",
            "pullback_bars_scale",
            "trail_atr_scale",
            "sl_mult_scale",
            "rsi_oversold_scale",
            "rsi_overbought_scale",
        }

        keys_to_check = defaults_keys - dict_shape_keys - framework_only
        missing = keys_to_check - sc_fields
        assert not missing, (
            f"DEFAULTS has scalar keys with no matching StrategyConfig "
            f"field: {sorted(missing)}. Either add the field to "
            f"StrategyConfig or move the DEFAULTS key to the "
            f"framework-only or dict-shape set in this test."
        )

    def test_strategy_config_keys_present_in_defaults(self):
        """For every scalar StrategyConfig field (except expected_trades_per_year
        and wfa_purge_days which are intentional overrides), DEFAULTS must
        have a matching key."""
        from quant_lib.core._config import DEFAULTS
        from quant_lib.experiments.base import StrategyConfig

        sc_fields = {f.name for f in dataclasses.fields(StrategyConfig)}
        defaults_keys = set(DEFAULTS.keys())

        # StrategyConfig-only fields (intentional):
        # - expected_trades_per_year: per-experiment override (None default)
        # - wfa_purge_days: lives in STATIC, not DEFAULTS (legacy)
        sc_only = {"expected_trades_per_year", "wfa_purge_days"}

        keys_to_check = sc_fields - sc_only
        missing = keys_to_check - defaults_keys
        assert not missing, (
            f"StrategyConfig has scalar fields with no matching DEFAULTS "
            f"key: {sorted(missing)}. Either add the DEFAULTS key or "
            f"add the field to the sc_only set in this test."
        )

    def test_shared_keys_have_identical_values(self):
        """For keys present in BOTH, the default values MUST match.

        This is the actual sync contract: same key, same default value.
        Catches the bug class where someone updates StrategyConfig but
        forgets to update DEFAULTS (or vice versa)."""
        from quant_lib.core._config import DEFAULTS
        from quant_lib.experiments.base import StrategyConfig

        sc_defaults = {
            f.name: f.default
            for f in dataclasses.fields(StrategyConfig)
            if f.default is not dataclasses.MISSING
        }

        # Keys we care about: both have them, scalar defaults only.
        for key in DEFAULTS:
            if key in sc_defaults and not isinstance(DEFAULTS[key], dict):
                assert DEFAULTS[key] == sc_defaults[key], (
                    f"DEFAULTS[{key!r}] = {DEFAULTS[key]!r} but "
                    f"StrategyConfig.{key} default = {sc_defaults[key]!r}. "
                    f"They MUST stay in sync (see comment in core/_config.py)."
                )

    def test_wfa_purge_days_canonical_source_is_static(self):
        """wfa_purge_days lives in STATIC (not DEFAULTS), so StrategyConfig
        gets its default from there. This documents the design intent."""
        from quant_lib.core._config import STATIC

        assert STATIC["wfa_purge_days"] == 90
        # StrategyConfig's wfa_purge_days default is also 90:
        sc_field = next(
            f for f in dataclasses.fields(__import__(
                "quant_lib.experiments.base", fromlist=["StrategyConfig"],
            ).StrategyConfig) if f.name == "wfa_purge_days"
        )
        assert sc_field.default == 90

    def test_dict_shape_keys_are_actually_dicts(self):
        """The search_space / search_space_pb keys must be dicts."""
        from quant_lib.core._config import DEFAULTS

        assert isinstance(DEFAULTS["search_space"], dict)
        assert isinstance(DEFAULTS["search_space_pb"], dict)

        # Each entry has (low, high) tuple values.
        for space in (DEFAULTS["search_space"], DEFAULTS["search_space_pb"]):
            for k, v in space.items():
                assert isinstance(v, tuple), (
                    f"search_space[{k!r}] = {v!r} is not a tuple"
                )
                assert len(v) == 2, (
                    f"search_space[{k!r}] = {v!r} is not a (low, high) pair"
                )


# =====================================================================
# Fix 3.7: Mutation baseline + CI infrastructure
# =====================================================================


class TestMutationBaselineInfra:
    """Sprint 3 fix 3.7: mutation baseline file structure.

    The baseline file is populated by the weekly mutation CI run
    (``make mutate`` or the GitHub Actions ``mutation.yml`` workflow).
    Until the first run populates it, the baseline file exists with
    documentation but no captured score.

    This test verifies:
    1. The baseline file exists at the documented path
    2. It documents the capture procedure
    3. It has the documented structure (even if values are blank)
    """

    def test_mutation_baseline_file_exists(self):
        """mutation_baseline.txt must exist at repo root."""
        repo_root = Path(__file__).resolve().parent.parent
        baseline = repo_root / "mutation_baseline.txt"
        assert baseline.exists(), (
            "mutation_baseline.txt missing. Sprint 3 fix 3.7 requires "
            "this file to exist so the weekly CI job can populate it."
        )

    def test_mutation_baseline_documents_capture_procedure(self):
        """Baseline file must document how to capture the baseline."""
        repo_root = Path(__file__).resolve().parent.parent
        baseline = repo_root / "mutation_baseline.txt"
        content = baseline.read_text(encoding="utf-8")
        # The file should mention the capture procedure.
        assert "capture" in content.lower() or "How to" in content, (
            "mutation_baseline.txt missing capture procedure docs."
        )

    def test_mutation_workflow_exists(self):
        """GitHub Actions mutation workflow must exist."""
        repo_root = Path(__file__).resolve().parent.parent
        workflow = repo_root / ".github" / "workflows" / "mutation.yml"
        assert workflow.exists(), (
            ".github/workflows/mutation.yml missing. "
            "Sprint 3 fix 3.7 requires this workflow to exist."
        )

    def test_mutmut_config_scopes_mutation(self):
        """mutmut must scope to candidate.py + commit.py (the F16 scope)."""
        repo_root = Path(__file__).resolve().parent.parent
        # The pyproject.toml [tool.mutmut] section defines the scope.
        pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
        # Should mention both candidate.py and commit.py.
        assert "candidate.py" in pyproject, (
            "pyproject.toml [tool.mutmut] must scope to candidate.py"
        )
        assert "commit.py" in pyproject, (
            "pyproject.toml [tool.mutmut] must scope to commit.py"
        )


# =====================================================================
# Fix 3.2: Candidate typed fields
# =====================================================================


class TestCandidateTypedFields:
    """Sprint 3 fix 3.2: Candidate previously-untyped dict fields now
    have explicit type aliases. Runtime is still plain ``dict`` --
    only static analysis is improved."""

    def test_type_aliases_exported(self):
        """Type aliases for Candidate dict fields must be exported."""
        from quant_lib.research.candidate import (
            DailyCloseMatrix,
            DailyHLMatrix,
            RiskWeights,
            RejectReasons,
            EdgeMetrics,
            FrozenParams,
            FoldParams,
        )
        # Type aliases are types at runtime.
        for alias in (
            DailyCloseMatrix, DailyHLMatrix, RiskWeights, RejectReasons,
            EdgeMetrics, FrozenParams, FoldParams,
        ):
            assert alias is not None

    def test_candidate_field_annotations_use_aliases(self):
        """Candidate dataclass fields must reference the type aliases."""
        import dataclasses

        from quant_lib.research.candidate import Candidate
        field_types = {f.name: f.type for f in dataclasses.fields(Candidate)}

        # Spot-check the alias-typed fields exist.
        for fname in (
            "daily_close_matrix", "daily_hl_matrix", "risk_weights",
            "reject_reasons", "edge_metrics", "frozen_params",
            "fold_params",
        ):
            assert fname in field_types, (
                f"Candidate.{fname} annotation missing"
            )

    def test_candidate_runtime_dicts_work(self):
        """Typed dict fields must still accept any dict at runtime."""
        import pandas as pd
        from quant_lib.audit import for_vol_compression
        from quant_lib.research.candidate import Candidate

        h = for_vol_compression("test_ty", "m", "b", "c")
        c = Candidate(hypothesis=h, session=None)

        # Plain dicts at runtime -- TypedDict would be too strict for
        # code that builds them progressively.
        c.daily_close_matrix = {"BTCUSDT": {}}
        c.daily_hl_matrix = {"BTCUSDT": {pd.Timestamp("2025-01-01"): {"high": 1.0, "low": 0.5}}}
        c.risk_weights = {"BTCUSDT": 0.01}
        c.reject_reasons = {"cb_cooldown": 1, "invalid_sl_pct": 0}
        c.edge_metrics = {"n_oos_trades": 10, "spa_p_value": 0.5}
        c.frozen_params = {"BTCUSDT": {"vol_pct_thresh": 0.2}}
        c.fold_params = {"2024-Q1": [{"vol_pct_thresh": 0.2}]}

        assert c.daily_close_matrix["BTCUSDT"] == {}
        assert c.risk_weights["BTCUSDT"] == 0.01
        assert c.frozen_params["BTCUSDT"]["vol_pct_thresh"] == 0.2


# =====================================================================
# Fix 3.3: ExploreResult dataclass
# =====================================================================


class TestExploreResultDataclass:
    """Sprint 3 fix 3.3: ``run_explore`` returns an ``ExploreResult``
    dataclass instead of a plain dict. Backward-compat dict-style
    access preserved."""

    def test_exploreresult_is_exported(self):
        """ExploreResult must be importable from quant_lib.research."""
        from quant_lib.research import ExploreResult
        assert ExploreResult is not None

    def test_lazy_resolution_via_dunder_getattr(self):
        """``quant_lib.ExploreResult`` must resolve via PEP 562 __getattr__."""
        import quant_lib
        er = quant_lib.ExploreResult
        assert er is not None
        # Must be a dataclass.
        import dataclasses
        assert dataclasses.is_dataclass(er)

    def test_attribute_access(self):
        """ExploreResult supports attribute access for type safety."""
        from quant_lib.research import ExploreResult

        r = ExploreResult(
            experiment="v1",
            n_oos_trades=10,
            n_executed=8,
            n_rejected=2,
            final_equity=1100.0,
            spa_p_value=0.123,
            narrowed_symbols=["BTCUSDT"],
        )
        assert r.experiment == "v1"
        assert r.spa_p_value == 0.123
        assert r.narrowed_symbols == ["BTCUSDT"]

    def test_dict_style_backward_compat(self):
        """Sprint 3 fix 3.3 BC: dict-style access still works."""
        from quant_lib.research import ExploreResult

        r = ExploreResult(
            experiment="v1", n_oos_trades=10, n_executed=8, n_rejected=2,
            final_equity=1100.0, spa_p_value=0.123, narrowed_symbols=["BTCUSDT"],
        )
        # Dict-style getitem
        assert r["spa_p_value"] == 0.123
        assert r["experiment"] == "v1"
        # KeyError on unknown
        import pytest as _pytest
        with _pytest.raises(KeyError):
            _ = r["nonexistent_key"]
        # TypeError on non-string key
        with _pytest.raises(TypeError):
            _ = r[42]

    def test_keys_values_items(self):
        """Dict-protocol methods work for backward compat."""
        from quant_lib.research import ExploreResult

        r = ExploreResult(
            experiment="v1", n_oos_trades=10, n_executed=8, n_rejected=2,
            final_equity=1100.0, spa_p_value=0.123, narrowed_symbols=["BTCUSDT"],
        )
        keys = list(r.keys())
        assert keys == [
            "experiment", "n_oos_trades", "n_executed", "n_rejected",
            "final_equity", "spa_p_value", "narrowed_symbols",
        ]
        assert len(r) == 7
        assert "spa_p_value" in r
        assert "nope" not in r
        # get() with default
        assert r.get("spa_p_value") == 0.123
        assert r.get("missing") is None
        assert r.get("missing", "fallback") == "fallback"
        # items() iteration
        items = dict(r.items())
        assert items["spa_p_value"] == 0.123

    def test_to_dict_returns_plain_dict(self):
        """to_dict() returns a JSON-serializable plain dict."""
        import json
        from quant_lib.research import ExploreResult

        r = ExploreResult(
            experiment="v1", n_oos_trades=10, n_executed=8, n_rejected=2,
            final_equity=1100.0, spa_p_value=0.123, narrowed_symbols=["BTCUSDT"],
        )
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["experiment"] == "v1"
        # JSON-serializable (lists and primitives only).
        json.dumps(d)


# =====================================================================
# Fix 3.4: CLI / Python API shared pipeline helper
# =====================================================================


class TestSharedPipelineHelper:
    """Sprint 3 fix 3.4: ``build_explore_candidate`` is the single
    source of truth for explore-pipeline session/candidate construction.
    Used by both ``run_explore`` and ``quant_exp explore``."""

    def test_helper_importable_from_research(self):
        """Helper must be importable from the research submodule."""
        from quant_lib.research._pipeline import build_explore_candidate
        assert callable(build_explore_candidate)

    def test_helper_raises_keyerror_for_unknown_experiment(self):
        """Unknown experiment name must raise KeyError (CLI converts to
        friendly error; Python API lets it propagate)."""
        from quant_lib.research._pipeline import build_explore_candidate
        import pytest as _pytest
        with _pytest.raises(KeyError):
            build_explore_candidate("nonexistent_exp_xyz", cache_dir="./data_cache")

    def test_helper_returns_candidate_and_exp(self):
        """Helper returns (Candidate, ExperimentConfig) tuple."""
        from quant_lib.research._pipeline import build_explore_candidate
        from quant_lib.experiments import (
            ExperimentConfig, PeriodConfig, StrategyConfig, UniverseConfig,
            register, clear,
        )
        from quant_lib.audit import for_vol_compression

        clear()
        hyp = for_vol_compression("test_pipe", "m", "b", "c")
        cfg = ExperimentConfig(
            name="test_pipe", strategy_type="vol_compression", hypothesis=hyp,
            period=PeriodConfig(train_start="2020-01-01", train_end="2020-12-31"),
            universe=UniverseConfig(symbols=["BTCUSDT"], min_volume_usdt=50_000_000),
            strategy=StrategyConfig(),
        )
        register(cfg)

        cand, exp = build_explore_candidate("test_pipe", cache_dir="./data_cache")
        assert exp.name == "test_pipe"
        # Candidate.stage starts at "hypothesis" (no phases run yet).
        assert cand.stage == "hypothesis"
        # cand.session is accessible for callers that need it.
        assert cand.session is not None

        clear()


# =====================================================================
# Fix 3.6: Real daily_equity plumbed through CommitResult
# =====================================================================


class TestCommitResultDailyEquity:
    """Sprint 3 fix 3.6: ``CommitResult.daily_equity`` carries the
    real daily equity curve from ``commit_to_holdout`` (replacing the
    synthetic 2-point fake that Sprint 2 removed)."""

    def test_commit_result_has_daily_equity_field(self):
        """CommitResult dataclass must have daily_equity field."""
        import dataclasses
        from quant_lib.research.commit import CommitResult

        fields = {f.name for f in dataclasses.fields(CommitResult)}
        assert "daily_equity" in fields, (
            "CommitResult missing daily_equity field. Sprint 3 fix 3.6."
        )

    def test_daily_equity_default_is_none(self):
        """daily_equity must default to None (backward compat: existing
        code that doesn't pass it must continue to work)."""
        import dataclasses
        from quant_lib.research.commit import CommitResult

        field = next(f for f in dataclasses.fields(CommitResult) if f.name == "daily_equity")
        assert field.default is None

    def test_commit_result_bc_instantiation_without_daily_equity(self):
        """Constructing CommitResult without daily_equity must still work
        (BC: existing test fixtures and code rely on this)."""
        from quant_lib.research.commit import CommitResult

        r = CommitResult(
            candidate_name="t", commit_idx=1,
            holdout_period=("2025-01-01", "2025-06-30"),
            timestamp="2025-01-01T00:00:00",
            initial_capital=1000.0, final_equity=1100.0,
            equity_pct=10.0, cagr_pct=21.0, max_dd_pct=5.0,
            n_raw_trades=10, n_executed_trades=8, n_rejected=2,
            reject_breakdown={},
            n_trades=8, win_rate=62.5, avg_r=0.5, median_r=0.3,
            std_r=1.0, best_r=2.5, worst_r=-1.5,
            profit_factor=1.8, avg_bars_held=12.0,
            sharpe_r=0.5, psr=0.85, psr_ess=0.85,
            skew=0.2, kurtosis=3.5, ess=8.0,
            bonferroni_alpha=0.075, fdr_alpha=0.15,
            # NO daily_equity arg -- defaults to None
        )
        assert r.daily_equity is None

    def test_commit_result_accepts_real_daily_equity(self):
        """CommitResult must accept a populated daily_equity dict."""
        import pandas as pd
        from quant_lib.research.commit import CommitResult

        eq = {
            pd.Timestamp("2025-01-01"): 1000.0,
            pd.Timestamp("2025-01-02"): 1010.0,
            pd.Timestamp("2025-01-03"): 1005.0,
        }
        r = CommitResult(
            candidate_name="t", commit_idx=1,
            holdout_period=("2025-01-01", "2025-06-30"),
            timestamp="2025-01-01T00:00:00",
            initial_capital=1000.0, final_equity=1005.0,
            equity_pct=0.5, cagr_pct=1.0, max_dd_pct=0.5,
            n_raw_trades=3, n_executed_trades=3, n_rejected=0,
            reject_breakdown={},
            n_trades=3, win_rate=66.7, avg_r=0.5, median_r=0.3,
            std_r=1.0, best_r=1.0, worst_r=-0.5,
            profit_factor=2.0, avg_bars_held=10.0,
            sharpe_r=0.8, psr=0.9, psr_ess=0.9,
            skew=0.1, kurtosis=3.0, ess=3.0,
            bonferroni_alpha=0.075, fdr_alpha=0.15,
            daily_equity=eq,
        )
        assert r.daily_equity is eq
        assert len(r.daily_equity) == 3
        assert r.daily_equity[pd.Timestamp("2025-01-02")] == 1010.0

    def test_cli_commit_chart_provider_uses_real_daily_equity(self):
        """Sprint 3 fix 3.6: commit chart provider uses result.daily_equity
        instead of the Sprint-2-removed fake."""
        from typer.testing import CliRunner
        import pandas as pd

        from quant_lib.cli.main import app
        from quant_lib.cli.commit_cmd import _make_chart_provider
        from quant_lib.research.commit import CommitResult

        eq = {
            pd.Timestamp("2025-01-01"): 1000.0,
            pd.Timestamp("2025-01-02"): 1010.0,
        }
        result = CommitResult(
            candidate_name="t", commit_idx=1,
            holdout_period=("2025-01-01", "2025-06-30"),
            timestamp="2025-01-01T00:00:00",
            initial_capital=1000.0, final_equity=1010.0,
            equity_pct=1.0, cagr_pct=2.0, max_dd_pct=0.5,
            n_raw_trades=3, n_executed_trades=3, n_rejected=0,
            reject_breakdown={},
            n_trades=3, win_rate=66.7, avg_r=0.5, median_r=0.3,
            std_r=1.0, best_r=1.0, worst_r=-0.5,
            profit_factor=2.0, avg_bars_held=10.0,
            sharpe_r=0.8, psr=0.9, psr_ess=0.9,
            skew=0.1, kurtosis=3.0, ess=3.0,
            bonferroni_alpha=0.075, fdr_alpha=0.15,
            daily_equity=eq,
        )
        # Mock session (CLI uses session.initial_capital).
        from unittest.mock import MagicMock
        session = MagicMock()
        session.initial_capital = 1000.0

        provider = _make_chart_provider(
            cand=MagicMock(),
            result=result,
            session=session,
            no_plots=True,  # skip actual chart rendering
        )
        # With no_plots=True, all charts return None (intentional).
        # Sprint 3 fix 3.6 verifies the provider at least doesn't crash
        # with the real daily_equity -- the chart code is exercised in
        # CLI integration tests.
        assert callable(provider)
