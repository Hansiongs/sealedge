"""Tests for the experiment registry and configuration.

These tests cover:
- PeriodConfig (auto-resolve holdout)
- UniverseConfig (basic)
- StrategyConfig (defaults match previous STATIC)
- ExperimentConfig (validation)
- Registry (register, get, all, exists, clear, count)
- from_hypothesis helper
- Auto-discovery (verifies vol_compression_v1 and pullback_sniper_rsi are loaded)
"""
import pandas as pd
import pytest

from quant_lib.audit import for_pullback_sniper, for_vol_compression
from quant_lib.experiments import (
    ExperimentConfig,
    PeriodConfig,
    STRATEGY_INT_TO_NAME,
    STRATEGY_NAME_TO_INT,
    StrategyConfig,
    UniverseConfig,
    all_experiments,
    built_in,
    clear,
    count,
    discover_experiments,
    exists,
    from_hypothesis,
    get,
    register,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear registry before/after each test.

    This ensures each test starts with a clean registry. Tests that
    need the auto-discovered experiments (e.g., TestAutoDiscovery) call
    ``discover_experiments()`` explicitly to populate the registry.
    """
    clear()
    yield
    clear()


# ════════════════════════════════════════════════════════════════════════
# PeriodConfig
# ════════════════════════════════════════════════════════════════════════


class TestPeriodConfig:
    def test_resolve_auto_holdout_post_training_default_6mo(self):
        """Default holdout_months=6 → POST-training period of 6 months.

        Convention: [train_end + 1 day, train_end + 6 months + 1 day].
        This enforces the no-peek guarantee: holdout must be data the
        WFA never saw, not an embargo slice of training data.
        """
        p = PeriodConfig(train_start="2020-01-01", train_end="2024-12-31")
        ts, te, hs, he = p.resolve()
        assert ts == "2020-01-01"
        assert te == "2024-12-31"
        # 2024-12-31 + 1 day = 2025-01-01; +6 months = 2025-07-01
        assert hs == "2025-01-01"
        assert he == "2025-07-01"

    def test_resolve_explicit_holdout(self):
        p = PeriodConfig(
            train_start="2020-01-01",
            train_end="2024-12-31",
            holdout_start="2025-01-01",
            holdout_end="2025-06-30",
        )
        _, _, hs, he = p.resolve()
        assert hs == "2025-01-01"
        assert he == "2025-06-30"

    def test_resolve_partial_explicit_falls_back_to_auto(self):
        """If either holdout_start or holdout_end is None, fall back to auto."""
        p = PeriodConfig(
            train_start="2020-01-01",
            train_end="2024-12-31",
            holdout_start="2025-01-01",
            holdout_end=None,
        )
        _, _, hs, he = p.resolve()
        # Falls back to post-training auto with default 6mo
        assert hs == "2025-01-01"
        assert he == "2025-07-01"


class TestPeriodConfigHoldoutMonths:
    """Configurable holdout_months parameter (Phase 1.1 fix)."""

    def test_default_holdout_months_is_6(self):
        """Default holdout_months must be 6 (backward-compat with prior default)."""
        cfg = PeriodConfig(train_start="2020-01-01", train_end="2024-12-31")
        assert cfg.holdout_months == 6

    def test_custom_holdout_months_3(self):
        cfg = PeriodConfig(
            train_start="2020-01-01", train_end="2024-12-31",
            holdout_months=3,
        )
        _, _, h_s, h_e = cfg.resolve()
        # 2024-12-31 + 1d = 2025-01-01; +3mo = 2025-04-01
        assert h_s == "2025-01-01"
        assert h_e == "2025-04-01"

    def test_custom_holdout_months_1(self):
        cfg = PeriodConfig(
            train_start="2020-01-01", train_end="2024-12-31",
            holdout_months=1,
        )
        _, _, h_s, h_e = cfg.resolve()
        # 2024-12-31 + 1d = 2025-01-01; +1mo = 2025-02-01
        assert h_s == "2025-01-01"
        assert h_e == "2025-02-01"

    def test_holdout_months_zero_raises(self):
        """holdout_months=0 must be rejected (no holdout = no test = invalid)."""
        with pytest.raises(ValueError, match="holdout_months"):
            PeriodConfig(
                train_start="2020-01-01", train_end="2024-12-31",
                holdout_months=0,
            )

    def test_holdout_months_negative_raises(self):
        with pytest.raises(ValueError, match="holdout_months"):
            PeriodConfig(
                train_start="2020-01-01", train_end="2024-12-31",
                holdout_months=-3,
            )

    def test_holdout_months_non_integer_raises(self):
        with pytest.raises(ValueError, match="holdout_months"):
            PeriodConfig(
                train_start="2020-01-01", train_end="2024-12-31",
                holdout_months=1.5,  # type: ignore[arg-type]
            )

    def test_explicit_holdout_overrides_months(self):
        """If holdout_start/end are both explicit, holdout_months is ignored."""
        cfg = PeriodConfig(
            train_start="2020-01-01", train_end="2024-12-31",
            holdout_start="2025-06-01", holdout_end="2025-09-01",
            holdout_months=6,  # should be ignored
        )
        _, _, h_s, h_e = cfg.resolve()
        assert h_s == "2025-06-01"
        assert h_e == "2025-09-01"

    def test_post_training_never_overlaps_train(self):
        """For any train_end, auto-resolved holdout start must be strictly > train_end.

        Critical invariant: enforces no-peek guarantee at the framework
        level. If this fails, ResearchSession validation will reject the
        period, but the framework should never GENERATE invalid periods.
        """
        for train_end in ["2020-01-31", "2024-02-29", "2024-12-31", "2025-06-15"]:
            cfg = PeriodConfig(train_start="2020-01-01", train_end=train_end)
            _, _, h_s, _ = cfg.resolve()
            assert pd.Timestamp(h_s) > pd.Timestamp(train_end), (
                f"Holdout start {h_s} must be strictly > train_end {train_end}"
            )

    def test_post_training_preserves_total_length(self):
        """For a 6-month holdout, hold_end - hold_start should be ~180 days.

        Edge case: month-end arithmetic via DateOffset.
        """
        cfg = PeriodConfig(
            train_start="2020-01-01", train_end="2024-12-31",
            holdout_months=6,
        )
        _, _, h_s, h_e = cfg.resolve()
        days = (pd.Timestamp(h_e) - pd.Timestamp(h_s)).days
        # 6 months is 181-184 days depending on start month
        assert 180 <= days <= 185, f"6-month holdout should be ~180 days, got {days}"


# ════════════════════════════════════════════════════════════════════════
# StrategyConfig
# ════════════════════════════════════════════════════════════════════════


class TestStrategyConfig:
    def test_defaults_match_previous_static(self):
        """Defaults should match the previous STATIC values for 1:1 migration."""
        s = StrategyConfig()
        assert s.initial_capital == 1000.0
        assert s.leverage == 3.0
        assert s.global_position_limit == 4
        assert s.bailout_bars == 36
        assert s.weekend_liquidity_penalty == 2.0
        assert s.stress_test_multiplier == 2.0
        assert s.cb_hard_cooldown_hours == 24
        assert s.fixed_cb_threshold == 0.15
        assert s.fixed_rvol_thresh == 2.5
        assert s.reg_lambda == 0.05
        assert s.wfa_purge_days == 90
        assert s.wfa_min_train_months == 12
        assert s.wfa_test_months == 3
        assert s.wfa_trials_per_fold == 80
        assert s.pf_weight_clamp_floor == 0.5
        assert s.pf_weight_clamp_ceiling == 1.5
        assert s.pf_decay_halflife_folds == 2
        assert s.pf_min_trades_for_weight == 10
        assert s.expected_trades_per_year is None

    def test_overrides(self):
        s = StrategyConfig(leverage=2.0, stress_test_multiplier=1.5)
        assert s.leverage == 2.0
        assert s.stress_test_multiplier == 1.5
        # Other fields stay default
        assert s.bailout_bars == 36

    def test_expected_trades_dict(self):
        s = StrategyConfig(expected_trades_per_year={"BTCUSDT": 50})
        assert s.expected_trades_per_year == {"BTCUSDT": 50}


# ════════════════════════════════════════════════════════════════════════
# ExperimentConfig validation
# ════════════════════════════════════════════════════════════════════════


def _hyp(name="test_exp", strategy_type=0):
    """Helper to build a Hypothesis for testing."""
    if strategy_type == 0:
        return for_vol_compression(
            name=name,
            mechanism="m",
            boundary_conditions="b",
            success_criteria="c",
        )
    else:
        return for_pullback_sniper(
            name=name,
            mechanism="m",
            boundary_conditions="b",
            success_criteria="c",
        )


class TestExperimentConfig:
    def test_valid_vol_compression(self):
        h = _hyp()
        cfg = ExperimentConfig(
            name="test_exp",
            strategy_type="vol_compression",
            hypothesis=h,
            period=PeriodConfig(train_start="2020-01-01", train_end="2024-12-31"),
            universe=UniverseConfig(symbols=["BTCUSDT"]),
        )
        assert cfg.name == "test_exp"
        assert cfg.strategy_type_int == 0
        assert isinstance(cfg.strategy, StrategyConfig)

    def test_valid_pullback_sniper(self):
        h = _hyp(strategy_type=1)
        cfg = ExperimentConfig(
            name="test_exp",
            strategy_type="pullback_sniper",
            hypothesis=h,
            period=PeriodConfig(train_start="2020-01-01", train_end="2024-12-31"),
            universe=UniverseConfig(symbols=["BTCUSDT"]),
        )
        assert cfg.strategy_type_int == 1

    def test_invalid_name_uppercase_raises(self):
        h = _hyp()
        with pytest.raises(ValueError, match="must match"):
            ExperimentConfig(
                name="InvalidName",
                strategy_type="vol_compression",
                hypothesis=h,
                period=PeriodConfig(train_start="2020-01-01", train_end="2024-12-31"),
                universe=UniverseConfig(symbols=["BTCUSDT"]),
            )

    def test_invalid_name_with_spaces_raises(self):
        h = _hyp()
        with pytest.raises(ValueError, match="must match"):
            ExperimentConfig(
                name="with spaces",
                strategy_type="vol_compression",
                hypothesis=h,
                period=PeriodConfig(train_start="2020-01-01", train_end="2024-12-31"),
                universe=UniverseConfig(symbols=["BTCUSDT"]),
            )

    def test_invalid_name_with_dash_raises(self):
        h = _hyp()
        with pytest.raises(ValueError, match="must match"):
            ExperimentConfig(
                name="with-dash",
                strategy_type="vol_compression",
                hypothesis=h,
                period=PeriodConfig(train_start="2020-01-01", train_end="2024-12-31"),
                universe=UniverseConfig(symbols=["BTCUSDT"]),
            )

    def test_name_mismatch_with_hypothesis_raises(self):
        h = _hyp(name="other")
        with pytest.raises(ValueError, match="must match hypothesis.name"):
            ExperimentConfig(
                name="test_exp",
                strategy_type="vol_compression",
                hypothesis=h,
                period=PeriodConfig(train_start="2020-01-01", train_end="2024-12-31"),
                universe=UniverseConfig(symbols=["BTCUSDT"]),
            )

    def test_strategy_type_mismatch_raises(self):
        h = _hyp(strategy_type=0)  # vol_compression
        with pytest.raises(ValueError, match="does not match"):
            ExperimentConfig(
                name="test_exp",
                strategy_type="pullback_sniper",  # wrong
                hypothesis=h,
                period=PeriodConfig(train_start="2020-01-01", train_end="2024-12-31"),
                universe=UniverseConfig(symbols=["BTCUSDT"]),
            )

    def test_frozen_dataclass(self):
        """ExperimentConfig is frozen -- attributes cannot be reassigned."""
        h = _hyp()
        cfg = ExperimentConfig(
            name="test_exp",
            strategy_type="vol_compression",
            hypothesis=h,
            period=PeriodConfig(train_start="2020-01-01", train_end="2024-12-31"),
            universe=UniverseConfig(symbols=["BTCUSDT"]),
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.name = "other_name"  # type: ignore


# ════════════════════════════════════════════════════════════════════════
# Registry
# ════════════════════════════════════════════════════════════════════════


def _make_config(name, strategy_type=0):
    h = _hyp(name=name, strategy_type=strategy_type)
    return ExperimentConfig(
        name=name,
        strategy_type=STRATEGY_INT_TO_NAME[strategy_type],
        hypothesis=h,
        period=PeriodConfig(train_start="2020-01-01", train_end="2024-12-31"),
        universe=UniverseConfig(symbols=["BTCUSDT"]),
    )


class TestRegistry:
    def test_register_and_get(self):
        cfg = _make_config("test1")
        result = register(cfg)
        assert result is cfg
        assert get("test1") is cfg
        assert exists("test1")
        assert count() == 1

    def test_get_missing_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get("nonexistent_experiment")

    def test_get_missing_lists_available(self):
        register(_make_config("alpha"))
        register(_make_config("beta"))
        with pytest.raises(KeyError, match="alpha") as exc_info:
            get("gamma")
        # Error message should mention available experiments
        assert "alpha" in str(exc_info.value)
        assert "beta" in str(exc_info.value)

    def test_all_experiments_sorted(self):
        register(_make_config("zebra"))
        register(_make_config("alpha"))
        register(_make_config("middle"))
        names = [c.name for c in all_experiments()]
        assert names == ["alpha", "middle", "zebra"]

    def test_register_overwrites_with_warning(self, caplog):
        register(_make_config("dup"))
        with caplog.at_level("WARNING"):
            register(_make_config("dup"))
        assert count() == 1
        assert "already registered" in caplog.text

    def test_clear(self):
        register(_make_config("a"))
        register(_make_config("b"))
        assert count() == 2
        clear()
        assert count() == 0
        assert all_experiments() == []
        assert not exists("a")
        assert not exists("b")

    def test_register_decorator_pattern(self):
        """@register works as a decorator on a function returning a config."""
        @register
        def make_exp():
            return _make_config("decorated_exp")

        assert exists("decorated_exp")
        assert get("decorated_exp").name == "decorated_exp"


# ════════════════════════════════════════════════════════════════════════
# from_hypothesis helper
# ════════════════════════════════════════════════════════════════════════


class TestFromHypothesis:
    def test_vol_compression(self):
        h = for_vol_compression(
            name="test_vc",
            mechanism="m",
            boundary_conditions="b",
            success_criteria="c",
        )
        cfg = from_hypothesis(
            name="test_vc",
            hypothesis=h,
            period=PeriodConfig(train_start="2020-01-01", train_end="2024-12-31"),
            universe=UniverseConfig(symbols=["BTCUSDT"]),
        )
        assert cfg.strategy_type == "vol_compression"
        assert cfg.strategy_type_int == 0
        assert cfg.hypothesis is h

    def test_pullback_sniper(self):
        h = for_pullback_sniper(
            name="test_pb",
            mechanism="m",
            boundary_conditions="b",
            success_criteria="c",
        )
        cfg = from_hypothesis(
            name="test_pb",
            hypothesis=h,
            period=PeriodConfig(train_start="2020-01-01", train_end="2024-12-31"),
            universe=UniverseConfig(symbols=["BTCUSDT"]),
        )
        assert cfg.strategy_type == "pullback_sniper"
        assert cfg.strategy_type_int == 1

    def test_default_strategy(self):
        h = for_vol_compression(
            name="test_default",
            mechanism="m",
            boundary_conditions="b",
            success_criteria="c",
        )
        cfg = from_hypothesis(
            name="test_default",
            hypothesis=h,
            period=PeriodConfig(train_start="2020-01-01", train_end="2024-12-31"),
            universe=UniverseConfig(symbols=["BTCUSDT"]),
        )
        assert isinstance(cfg.strategy, StrategyConfig)
        assert cfg.strategy.leverage == 3.0  # default

    def test_custom_strategy(self):
        h = for_vol_compression(
            name="test_custom",
            mechanism="m",
            boundary_conditions="b",
            success_criteria="c",
        )
        custom = StrategyConfig(leverage=2.5, stress_test_multiplier=1.5)
        cfg = from_hypothesis(
            name="test_custom",
            hypothesis=h,
            period=PeriodConfig(train_start="2020-01-01", train_end="2024-12-31"),
            universe=UniverseConfig(symbols=["BTCUSDT"]),
            strategy=custom,
        )
        assert cfg.strategy.leverage == 2.5
        assert cfg.strategy.stress_test_multiplier == 1.5


# ════════════════════════════════════════════════════════════════════════
# Auto-discovery
# ════════════════════════════════════════════════════════════════════════


class TestAutoDiscovery:
    @pytest.fixture(autouse=True)
    def _ensure_discovered(self):
        """Auto-discover before each test in this class.

        The module-level fixture clears the registry. We need to also
        reset the discovery state (so it's not a no-op) and re-discover.
        """
        built_in.reset()
        discover_experiments()
        yield

    def test_built_in_experiments_are_loaded(self):
        """After import of quant_lib.experiments, the 2 built-in
        experiments should be in the registry."""
        assert exists("vol_compression_v1")
        assert exists("pullback_sniper_rsi")
        # Phase 2: funding_rate_carry added -- 3 strategies total.
        assert count() == 3

    def test_vol_compression_v1_details(self):
        cfg = get("vol_compression_v1")
        assert cfg.strategy_type == "vol_compression"
        assert cfg.strategy_type_int == 0
        assert "Volatility compression" in cfg.hypothesis.mechanism
        assert cfg.period.train_start == "2020-01-01"
        assert cfg.period.train_end == "2025-12-31"
        assert "BTCUSDT" in cfg.universe.symbols
        assert "ETHUSDT" in cfg.universe.symbols
        assert "SOLUSDT" in cfg.universe.symbols

    def test_pullback_sniper_rsi_details(self):
        cfg = get("pullback_sniper_rsi")
        assert cfg.strategy_type == "pullback_sniper"
        assert cfg.strategy_type_int == 1
        assert "RSI" in cfg.hypothesis.mechanism
        assert cfg.period.train_start == "2020-01-01"
        assert "BTCUSDT" in cfg.universe.symbols

    def test_auto_resolve_holdout(self):
        """vol_compression_v1 has no explicit holdout, should auto-resolve POST-training."""
        cfg = get("vol_compression_v1")
        ts, te, hs, he = cfg.period.resolve()
        assert ts == "2020-01-01"
        assert te == "2025-12-31"
        # Post-training convention: [train_end + 1d, train_end + 6mo + 1d]
        # 2025-12-31 + 1d = 2026-01-01; +6mo = 2026-07-01
        assert hs == "2026-01-01"
        assert he == "2026-07-01"

    def test_discover_is_idempotent(self):
        """Calling discover twice should not double-register."""
        first_count = count()
        discover_experiments()  # no-op (already discovered)
        discover_experiments()  # no-op
        assert count() == first_count

    def test_reset_allows_rediscovery(self):
        clear()
        assert count() == 0
        built_in.reset()
        discover_experiments()
        # Phase 2: funding_rate_carry added -- 3 strategies total.
        assert count() == 3


# ════════════════════════════════════════════════════════════════════════
# Strategy type mapping sanity check
# ════════════════════════════════════════════════════════════════════════


class TestStrategyTypeMapping:
    def test_name_to_int_bidirectional(self):
        for name, int_val in STRATEGY_NAME_TO_INT.items():
            assert STRATEGY_INT_TO_NAME[int_val] == name

    def test_vol_compression_is_0(self):
        assert STRATEGY_NAME_TO_INT["vol_compression"] == 0

    def test_pullback_sniper_is_1(self):
        assert STRATEGY_NAME_TO_INT["pullback_sniper"] == 1
