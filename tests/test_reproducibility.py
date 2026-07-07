"""Reproducibility tests: same input + same seed = same output.

These tests verify that the framework's core computations are
deterministic. This is critical for paper defense (reproducibility is
a scientific requirement) and for debugging (non-deterministic bugs
are the worst kind).

Components tested:
- simulate_full_portfolio (pure function, no RNG)
- run_bootstrap (uses GLOBAL_SEED+12345, deterministic)
- portfolio_spa (uses np.random.default_rng(seed), deterministic)
- pick_best_params_per_symbol (pure function)
- apply_pf_weighted_risk_allocation (pure function, no RNG)
"""
import sys

import numpy as np
import pandas as pd
import pytest

from quant_lib.core._metrics import run_bootstrap
from quant_lib.core._portfolio import simulate_full_portfolio
from quant_lib.core._risk_allocation import apply_pf_weighted_risk_allocation
from quant_lib.core._spa import portfolio_spa
from quant_lib.experiments.base import (
    PeriodConfig,
    StrategyConfig,
)
from quant_lib.research.best_params import pick_best_params_per_symbol


# ════════════════════════════════════════════════════════════════════════
# Pure functions (no RNG involved) -- these are trivially deterministic
# ════════════════════════════════════════════════════════════════════════


class TestPureFunctionDeterminism:
    """Test functions that should be 100% deterministic (no RNG)."""

    def test_pick_best_params_per_symbol_deterministic(self):
        """pick_best_params_per_symbol is a pure function."""
        folds = [
            {"best_value": 0.5, "vol_pct_thresh": 0.10, "trail_atr": 2.0,
             "sl_mult": 1.0, "pullback_bars": 3},
            {"best_value": 0.9, "vol_pct_thresh": 0.25, "trail_atr": 3.5,
             "sl_mult": 1.8, "pullback_bars": 6},
        ]
        r1 = pick_best_params_per_symbol({"BTCUSDT": folds}, strategy_type=0)
        r2 = pick_best_params_per_symbol({"BTCUSDT": folds}, strategy_type=0)
        assert r1 == r2

    def test_period_config_resolve_deterministic(self):
        """PeriodConfig.resolve() is deterministic."""
        p = PeriodConfig(train_start="2020-01-01", train_end="2025-12-31")
        assert p.resolve() == p.resolve()

    def test_apply_pf_weighted_risk_allocation_deterministic(self):
        """apply_pf_weighted_risk_allocation is a pure function (no RNG)."""
        trades1 = [
            {"symbol": "BTC", "r_net": 1.0, "fold_key": "F1", "risk_weight": 0.01},
            {"symbol": "ETH", "r_net": -0.5, "fold_key": "F1", "risk_weight": 0.01},
            {"symbol": "BTC", "r_net": 0.5, "fold_key": "F2", "risk_weight": 0.01},
        ]
        trades2 = [
            {"symbol": "BTC", "r_net": 1.0, "fold_key": "F1", "risk_weight": 0.01},
            {"symbol": "ETH", "r_net": -0.5, "fold_key": "F1", "risk_weight": 0.01},
            {"symbol": "BTC", "r_net": 0.5, "fold_key": "F2", "risk_weight": 0.01},
        ]
        r1 = apply_pf_weighted_risk_allocation(
            trades1, halflife_folds=2, clamp_floor=0.5, clamp_ceiling=1.5,
            min_trades=1, baseline_per_symbol=0.01, n_total_symbols=2,
        )
        r2 = apply_pf_weighted_risk_allocation(
            trades2, halflife_folds=2, clamp_floor=0.5, clamp_ceiling=1.5,
            min_trades=1, baseline_per_symbol=0.01, n_total_symbols=2,
        )
        assert r1 == r2
        # Trade risk_weights are also deterministic.
        assert trades1[0]["risk_weight"] == trades2[0]["risk_weight"]
        assert trades1[1]["risk_weight"] == trades2[1]["risk_weight"]
        assert trades1[2]["risk_weight"] == trades2[2]["risk_weight"]

    def test_strategy_config_defaults_deterministic(self):
        """StrategyConfig() with no overrides is deterministic."""
        s1 = StrategyConfig()
        s2 = StrategyConfig()
        assert s1 == s2


# ════════════════════════════════════════════════════════════════════════
# simulate_full_portfolio -- depends on portfolio sim internals
# ════════════════════════════════════════════════════════════════════════


class TestPortfolioSimulationDeterminism:
    """simulate_full_portfolio should be deterministic given same inputs."""

    @pytest.fixture
    def simple_simulation_inputs(self):
        """Minimal inputs for portfolio simulation."""
        # Deterministic seed + trade count kept as documentation of
        # the simulated scope (3 trades, simple structure).
        _rng = np.random.default_rng(42)
        _n = 100
        # 3 trades, simple structure
        trades = [
            {
                "entry_time": pd.Timestamp("2024-01-15"),
                "exit_time": pd.Timestamp("2024-01-20"),
                "symbol": "BTCUSDT",
                "r_net": 0.5,
                "entry_price": 100.0,
                "exit_price": 105.0,
                "trade_dir": 1,
                "sl_pct": 0.02,
                "risk_weight": 0.01,
                "trend_risk_mult": 1.0,
            },
            {
                "entry_time": pd.Timestamp("2024-02-10"),
                "exit_time": pd.Timestamp("2024-02-15"),
                "symbol": "ETHUSDT",
                "r_net": -0.3,
                "entry_price": 100.0,
                "exit_price": 97.0,
                "trade_dir": -1,
                "sl_pct": 0.02,
                "risk_weight": 0.01,
                "trend_risk_mult": 1.0,
            },
        ]
        # Daily prices
        dates = pd.date_range("2024-01-01", "2024-03-31", freq="D")
        daily_close = {
            "BTCUSDT": {d: 100.0 + i * 0.1 for i, d in enumerate(dates)},
            "ETHUSDT": {d: 100.0 + i * 0.05 for i, d in enumerate(dates)},
        }
        daily_hl = {
            "BTCUSDT": {d: {"high": 101.0, "low": 99.0} for d in dates},
            "ETHUSDT": {d: {"high": 100.5, "low": 99.5} for d in dates},
        }
        return trades, daily_close, daily_hl

    def test_simulate_portfolio_deterministic(self, simple_simulation_inputs):
        """Same inputs -> same outputs (twice)."""
        trades, daily_close, daily_hl = simple_simulation_inputs
        asset_risk_weights = {"BTCUSDT": 0.01, "ETHUSDT": 0.01}

        eq1, daily1, exec1, rej1 = simulate_full_portfolio(
            trades,
            initial_cash=1000.0,
            leverage=3.0,
            mm_pct=0.01,
            position_limit=4,
            cb_hard_cooldown_hours=24,
            fixed_cb_threshold=0.15,
            daily_close_matrix=daily_close,
            asset_risk_weights=asset_risk_weights,
            end_date="2024-03-31",
            daily_hl_matrix=daily_hl,
        )
        eq2, daily2, exec2, rej2 = simulate_full_portfolio(
            trades,
            initial_cash=1000.0,
            leverage=3.0,
            mm_pct=0.01,
            position_limit=4,
            cb_hard_cooldown_hours=24,
            fixed_cb_threshold=0.15,
            daily_close_matrix=daily_close,
            asset_risk_weights=asset_risk_weights,
            end_date="2024-03-31",
            daily_hl_matrix=daily_hl,
        )

        assert eq1 == eq2
        assert daily1 == daily2
        assert len(exec1) == len(exec2)
        assert rej1 == rej2


# ════════════════════════════════════════════════════════════════════════
# run_bootstrap -- uses GLOBAL_SEED+12345 internally
# ════════════════════════════════════════════════════════════════════════


class TestBootstrapDeterminism:
    """run_bootstrap uses a fixed internal seed, should be deterministic."""

    @pytest.fixture
    def bootstrap_inputs(self):
        """Minimal inputs for bootstrap."""
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        daily_ret = pd.Series(rng.normal(0.001, 0.02, n), index=dates)
        eq_series = pd.Series(
            1000 * (1 + daily_ret).cumprod(),
            index=dates,
        )
        return daily_ret, eq_series, -0.15

    def test_bootstrap_deterministic(self, bootstrap_inputs):
        """Same bootstrap inputs -> same bootstrap output."""
        daily_ret, eq_series, max_dd = bootstrap_inputs

        bs1 = run_bootstrap(daily_ret, eq_series, max_dd, 1000.0)
        bs2 = run_bootstrap(daily_ret, eq_series, max_dd, 1000.0)

        # All keys should be identical
        assert set(bs1.keys()) == set(bs2.keys())
        for key in bs1:
            assert bs1[key] == bs2[key], (
                f"Bootstrap output differs for {key}: {bs1[key]} vs {bs2[key]}"
            )


# ════════════════════════════════════════════════════════════════════════
# portfolio_spa -- uses np.random.default_rng(seed) internally
# ════════════════════════════════════════════════════════════════════════


class TestSPADeterminism:
    """portfolio_spa uses internal RNG seeded with rng_seed parameter.

    Same seed -> same p-value (Phipson & Smyth 2010 add-one corrected).
    """

    @pytest.fixture
    def spa_inputs(self):
        """Minimal inputs for SPA."""
        rng = np.random.default_rng(42)
        # 10 trades
        trades = []
        for i in range(10):
            trades.append({
                "entry_time": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i * 5),
                "exit_time": pd.Timestamp("2024-01-05") + pd.Timedelta(days=i * 5),
                "symbol": "BTCUSDT",
                "r_net": float(rng.normal(0.05, 0.2)),
                "entry_price": 100.0,
                "exit_price": 102.0,
                "trade_dir": 1,
                "sl_pct": 0.02,
                "sl_mult": 1.5,
                "trail_atr": 3.0,
                "risk_weight": 0.01,
            })
        # Asset data
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        asset_data = {
            "BTCUSDT": pd.DataFrame({
                "time": dates,
                "close": 100 + np.cumsum(rng.normal(0, 1, n)),
                "high": 102 + np.cumsum(rng.normal(0, 1, n)),
                "low": 98 + np.cumsum(rng.normal(0, 1, n)),
                "atr": rng.uniform(1, 3, n),
                "funding_rate": rng.normal(0, 0.001, n),
                "is_weekend": np.zeros(n, dtype=int),
                "is_funding_hour": np.zeros(n, dtype=int),
                "macro_trend": np.ones(n, dtype=int),
            })
        }
        daily_close = {
            "BTCUSDT": {d: float(100 + i) for i, d in enumerate(dates)},
        }
        daily_hl = {
            "BTCUSDT": {d: {"high": 102.0, "low": 98.0} for d in dates},
        }
        return trades, asset_data, daily_close, daily_hl

    def test_spa_same_seed_same_pvalue(self, spa_inputs):
        """SPA with same seed produces same p-value (twice)."""
        trades, asset_data, daily_close, daily_hl = spa_inputs
        asset_risk_weights = {"BTCUSDT": 0.01}

        # Run twice with same seed
        obs1, _, p1 = portfolio_spa(
            trades, asset_data, daily_close, "2023-12-31",
            daily_hl_matrix=daily_hl,
            n_iters=100,  # small for test speed
            initial_capital=1000.0,
            leverage=3.0,
            mm_pct=0.01,
            position_limit=4,
            cb_hard_cooldown_hours=24,
            fixed_cb_threshold=0.15,
            rng_seed=42,
            verbose=False,
            asset_risk_weights=asset_risk_weights,
        )
        obs2, _, p2 = portfolio_spa(
            trades, asset_data, daily_close, "2023-12-31",
            daily_hl_matrix=daily_hl,
            n_iters=100,
            initial_capital=1000.0,
            leverage=3.0,
            mm_pct=0.01,
            position_limit=4,
            cb_hard_cooldown_hours=24,
            fixed_cb_threshold=0.15,
            rng_seed=42,
            verbose=False,
            asset_risk_weights=asset_risk_weights,
        )

        assert obs1 == obs2
        assert p1 == p2

    def test_spa_accepts_different_seeds(self, spa_inputs):
        """SPA can be called with different seeds without erroring.

        Note: With degenerate input, both seeds may give the same
        p-value (e.g., p=1.0 for no-edge strategies). We just verify
        that the framework accepts different seeds and returns valid
        p-values in [0, 1].
        """
        trades, asset_data, daily_close, daily_hl = spa_inputs
        asset_risk_weights = {"BTCUSDT": 0.01}

        _, _, p1 = portfolio_spa(
            trades, asset_data, daily_close, "2023-12-31",
            daily_hl_matrix=daily_hl,
            n_iters=100, initial_capital=1000.0, leverage=3.0,
            mm_pct=0.01, position_limit=4, cb_hard_cooldown_hours=24,
            fixed_cb_threshold=0.15, rng_seed=42, verbose=False,
            asset_risk_weights=asset_risk_weights,
        )
        _, _, p2 = portfolio_spa(
            trades, asset_data, daily_close, "2023-12-31",
            daily_hl_matrix=daily_hl,
            n_iters=100, initial_capital=1000.0, leverage=3.0,
            mm_pct=0.01, position_limit=4, cb_hard_cooldown_hours=24,
            fixed_cb_threshold=0.15, rng_seed=999, verbose=False,
            asset_risk_weights=asset_risk_weights,
        )

        # Both p-values must be in valid range [0, 1]
        assert 0.0 <= p1 <= 1.0
        assert 0.0 <= p2 <= 1.0


# ════════════════════════════════════════════════════════════════════════
# Experiment registry determinism
# ════════════════════════════════════════════════════════════════════════


class TestRegistryDeterminism:
    """discover_experiments() should be idempotent and deterministic."""

    def test_double_discover_same_result(self):
        """Calling discover twice gives same registry."""
        from quant_lib.experiments import (
            all_experiments, clear, built_in, discover_experiments,
        )
        clear()
        built_in.reset()
        discover_experiments()
        first = sorted(e.name for e in all_experiments())
        discover_experiments()  # second time, idempotent
        second = sorted(e.name for e in all_experiments())
        assert first == second

    def test_reload_gives_same_experiments(self):
        """importlib.reload of experiment modules doesn't change registry."""
        from quant_lib.experiments import (
            all_experiments, clear, built_in, discover_experiments,
        )
        clear()
        built_in.reset()
        discover_experiments()
        first_names = sorted(e.name for e in all_experiments())

        # Manually trigger reload
        import importlib
        for mod_name in [
            "quant_lib.experiments.vol_compression_v1",
            "quant_lib.experiments.pullback_sniper_rsi",
        ]:
            if mod_name in sys.modules:
                importlib.reload(sys.modules[mod_name])

        second_names = sorted(e.name for e in all_experiments())
        assert first_names == second_names
