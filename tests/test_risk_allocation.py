"""Tests for the per-fold PF-weighted risk allocation module.

Covers:
  - Private helpers (_compute_decay_weighted_pnl_loss, _compute_clamped_factor,
    _rescale_factors_to_total) -- extracted from prior single-file pipeline.
  - Public orchestrator apply_pf_weighted_risk_allocation: cold start,
    full multi-fold sequencing, no-fold_key no-op, total preserved,
    log summary, integration with Candidate.
"""
import logging

import pytest

from quant_lib.core._risk_allocation import (
    _compute_decay_weighted_pnl_loss,
    _compute_clamped_factor,
    _rescale_factors_to_total,
    apply_pf_weighted_risk_allocation,
    default_baseline_per_symbol,
)


# ─────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────


class TestDecayWeightedPnLLoss:
    def test_empty_input_returns_empty(self):
        result = _compute_decay_weighted_pnl_loss([], halflife_folds=2)
        assert result == {}

    def test_single_fold_single_symbol(self):
        past = [(1, [{"symbol": "BTCUSDT", "r_net": 0.5}])]
        result = _compute_decay_weighted_pnl_loss(past, halflife_folds=2)
        # decay_weight = 0.5^(1/2) ≈ 0.7071
        assert "BTCUSDT" in result
        w_pnl, w_loss, n = result["BTCUSDT"]
        assert n == 1
        assert w_pnl == pytest.approx(0.5 * 0.7071, rel=1e-3)
        assert w_loss == 0.0

    def test_decay_diminishes_with_distance(self):
        """Farther folds have less weight (decay)."""
        past_close = [(1, [{"symbol": "BTCUSDT", "r_net": 1.0}])]
        past_far = [(4, [{"symbol": "BTCUSDT", "r_net": 1.0}])]
        r_close = _compute_decay_weighted_pnl_loss(past_close, halflife_folds=2)
        r_far = _compute_decay_weighted_pnl_loss(past_far, halflife_folds=2)
        assert r_close["BTCUSDT"][0] > r_far["BTCUSDT"][0]

    def test_separates_pnl_and_loss(self):
        past = [
            (1, [
                {"symbol": "BTCUSDT", "r_net": 1.0},
                {"symbol": "BTCUSDT", "r_net": -0.5},
            ]),
        ]
        result = _compute_decay_weighted_pnl_loss(past, halflife_folds=2)
        w_pnl, w_loss, n = result["BTCUSDT"]
        assert n == 2
        assert w_pnl > 0  # positive trade
        assert w_loss > 0  # negative trade

    def test_multi_symbol(self):
        past = [
            (1, [
                {"symbol": "BTCUSDT", "r_net": 0.5},
                {"symbol": "ETHUSDT", "r_net": -0.3},
            ]),
        ]
        result = _compute_decay_weighted_pnl_loss(past, halflife_folds=2)
        assert "BTCUSDT" in result
        assert "ETHUSDT" in result
        assert result["BTCUSDT"][2] == 1
        assert result["ETHUSDT"][2] == 1

    def test_includes_zero_r_net_symbols(self):
        """Symbols with only zero r_net should still be in result."""
        past = [(1, [{"symbol": "BTCUSDT", "r_net": 0.0}])]
        result = _compute_decay_weighted_pnl_loss(past, halflife_folds=2)
        assert "BTCUSDT" in result
        w_pnl, w_loss, n = result["BTCUSDT"]
        assert n == 1
        assert w_pnl == 0.0
        assert w_loss == 0.0

    def test_higher_halflife_slower_decay(self):
        past = [(3, [{"symbol": "BTCUSDT", "r_net": 1.0}])]
        r_low_halflife = _compute_decay_weighted_pnl_loss(past, halflife_folds=1)
        r_high_halflife = _compute_decay_weighted_pnl_loss(past, halflife_folds=10)
        # Higher halflife -> less decay -> larger w_pnl
        assert r_high_halflife["BTCUSDT"][0] > r_low_halflife["BTCUSDT"][0]


class TestClampedFactor:
    def test_neutral_when_insufficient_trades(self):
        result = _compute_clamped_factor(
            weighted_pnl=1.0, weighted_loss=0.5,
            n_past_trades=5, min_trades=10,
            clamp_floor=0.5, clamp_ceiling=1.5,
        )
        assert result == 1.0

    def test_neutral_when_zero_pnl_zero_loss(self):
        result = _compute_clamped_factor(
            weighted_pnl=0.0, weighted_loss=0.0,
            n_past_trades=20, min_trades=10,
            clamp_floor=0.5, clamp_ceiling=1.5,
        )
        assert result == 1.0

    def test_ceiling_when_zero_loss_positive_pnl(self):
        result = _compute_clamped_factor(
            weighted_pnl=1.0, weighted_loss=0.0,
            n_past_trades=20, min_trades=10,
            clamp_floor=0.5, clamp_ceiling=1.5,
        )
        assert result == 1.5

    def test_clamp_below_floor(self):
        # pf = 0.1, clamp_floor = 0.5 -> 0.5
        result = _compute_clamped_factor(
            weighted_pnl=0.1, weighted_loss=1.0,
            n_past_trades=20, min_trades=10,
            clamp_floor=0.5, clamp_ceiling=1.5,
        )
        assert result == 0.5

    def test_clamp_above_ceiling(self):
        # pf = 2.5, clamp_ceiling = 1.5 -> 1.5
        result = _compute_clamped_factor(
            weighted_pnl=2.5, weighted_loss=1.0,
            n_past_trades=20, min_trades=10,
            clamp_floor=0.5, clamp_ceiling=1.5,
        )
        assert result == 1.5

    def test_pf_within_bounds(self):
        # pf = 1.2, within [0.5, 1.5] -> 1.2
        result = _compute_clamped_factor(
            weighted_pnl=1.2, weighted_loss=1.0,
            n_past_trades=20, min_trades=10,
            clamp_floor=0.5, clamp_ceiling=1.5,
        )
        assert result == pytest.approx(1.2, rel=1e-9)

    def test_min_trades_boundary_inclusive(self):
        """n_past_trades == min_trades is the threshold for applying PF."""
        result = _compute_clamped_factor(
            weighted_pnl=1.0, weighted_loss=1.0,
            n_past_trades=10, min_trades=10,
            clamp_floor=0.5, clamp_ceiling=1.5,
        )
        assert result == pytest.approx(1.0, rel=1e-9)

    def test_min_trades_boundary_below(self):
        result = _compute_clamped_factor(
            weighted_pnl=1.0, weighted_loss=1.0,
            n_past_trades=9, min_trades=10,
            clamp_floor=0.5, clamp_ceiling=1.5,
        )
        assert result == 1.0


class TestRescaleFactorsToTotal:
    def test_empty_factors_returns_empty(self):
        result = _rescale_factors_to_total({}, baseline_per_symbol=0.01, target_total=0.04)
        assert result == {}

    def test_zero_baseline_returns_empty(self):
        result = _rescale_factors_to_total(
            {"BTCUSDT": 1.0}, baseline_per_symbol=0.0, target_total=0.04
        )
        assert result == {}

    def test_preserves_target_total(self):
        factors = {"BTCUSDT": 1.5, "ETHUSDT": 0.5}
        result = _rescale_factors_to_total(
            factors, baseline_per_symbol=0.01, target_total=0.04
        )
        # pre_rescale = {BTC: 0.015, ETH: 0.005}, sum = 0.02
        # rescale = 0.04 / 0.02 = 2.0
        # final = {BTC: 0.03, ETH: 0.01}
        assert result["BTCUSDT"] == pytest.approx(0.03, rel=1e-9)
        assert result["ETHUSDT"] == pytest.approx(0.01, rel=1e-9)
        assert sum(result.values()) == pytest.approx(0.04, rel=1e-9)

    def test_all_factors_one_preserves_baseline(self):
        """When all factors = 1.0, final = baseline_per_symbol each, target preserved."""
        factors = {"BTCUSDT": 1.0, "ETHUSDT": 1.0}
        result = _rescale_factors_to_total(
            factors, baseline_per_symbol=0.01, target_total=0.02
        )
        # pre_rescale = {BTC: 0.01, ETH: 0.01}, sum = 0.02
        # rescale = 1.0
        # final = {BTC: 0.01, ETH: 0.01}
        assert result["BTCUSDT"] == pytest.approx(0.01, rel=1e-9)
        assert result["ETHUSDT"] == pytest.approx(0.01, rel=1e-9)

    def test_all_zero_factors_returns_empty(self):
        factors = {"BTCUSDT": 0.0, "ETHUSDT": 0.0}
        result = _rescale_factors_to_total(
            factors, baseline_per_symbol=0.01, target_total=0.04
        )
        assert result == {}


# ─────────────────────────────────────────────────────────────────────
# Public orchestrator: apply_pf_weighted_risk_allocation
# ─────────────────────────────────────────────────────────────────────


def _make_trade(symbol, r_net, fold_key, risk_weight=0.01):
    return {
        "symbol": symbol,
        "r_net": r_net,
        "fold_key": fold_key,
        "risk_weight": risk_weight,
    }


class TestApplyPFWeightedRiskAllocation:
    def test_no_fold_key_is_noop(self):
        """Commit-path: trades without fold_key must be left untouched."""
        trades = [
            _make_trade("BTCUSDT", 0.5, None),
            _make_trade("ETHUSDT", -0.3, None),
        ]
        result = apply_pf_weighted_risk_allocation(
            trades=trades,
            halflife_folds=2,
            clamp_floor=0.5,
            clamp_ceiling=1.5,
            min_trades=10,
            baseline_per_symbol=0.01,
            n_total_symbols=4,
        )
        # No-op: no summary, no mutations
        assert result == {}
        for t in trades:
            assert t["risk_weight"] == 0.01  # unchanged
            assert t["fold_key"] is None  # unchanged
            assert "atr_inv" not in t

    def test_cold_start_first_fold_neutral(self):
        """First fold has no past -> all factors = 1.0."""
        trades = [_make_trade("BTCUSDT", 0.5, "2020-01")]
        result = apply_pf_weighted_risk_allocation(
            trades=trades,
            halflife_folds=2,
            clamp_floor=0.5,
            clamp_ceiling=1.5,
            min_trades=10,
            baseline_per_symbol=0.01,
            n_total_symbols=2,
        )
        # 1 fold, 1 active sym: budget_scale = 1/2 = 0.5, target = 0.01 * 2 * 0.5 = 0.01
        # factor = 1.0 (no past data), so final = 0.005 each * rescale
        # pre_rescale = {BTC: 0.005}, sum = 0.005, rescale = 0.01 / 0.005 = 2.0
        # final[BTC] = 0.01
        assert "2020-01" in result
        assert result["2020-01"]["BTCUSDT"] == pytest.approx(0.01, rel=1e-6)
        assert trades[0]["risk_weight"] == pytest.approx(0.01, rel=1e-6)
        # fold_key consumed
        assert "fold_key" not in trades[0]

    def test_two_folds_uses_past_data(self):
        """Second fold should have past data from first fold."""
        trades = [
            _make_trade("BTCUSDT", 1.0, "2020-Q1"),  # fold 1, winning
            _make_trade("ETHUSDT", -1.0, "2020-Q1"),  # fold 1, losing
            _make_trade("BTCUSDT", 0.5, "2020-Q2"),  # fold 2
            _make_trade("ETHUSDT", -0.5, "2020-Q2"),  # fold 2
        ]
        result = apply_pf_weighted_risk_allocation(
            trades=trades,
            halflife_folds=2,
            clamp_floor=0.5,
            clamp_ceiling=1.5,
            min_trades=1,  # only 1 trade per symbol per fold in this test
            baseline_per_symbol=0.01,
            n_total_symbols=2,
        )
        # 2 folds, both have 2 active syms: budget_scale = 1.0, target = 0.02
        # Fold 1: no past -> all factor = 1.0 -> final = 0.01 each, total = 0.02
        # Fold 2: BTC had +1.0 past (positive pnl, no loss) -> ceiling 1.5
        #         ETH had -1.0 past (no pnl, positive loss) -> pf = 0/1 = 0 -> floor 0.5
        # pre_rescale fold2 = {BTC: 0.015, ETH: 0.005}, sum = 0.02
        # rescale = 1.0, final = {BTC: 0.015, ETH: 0.005}
        assert "2020-Q1" in result
        assert "2020-Q2" in result
        # Fold 1: both 0.01
        assert result["2020-Q1"]["BTCUSDT"] == pytest.approx(0.01, rel=1e-6)
        assert result["2020-Q1"]["ETHUSDT"] == pytest.approx(0.01, rel=1e-6)
        # Fold 2: BTC 0.015 (ceiling), ETH 0.005 (floor)
        assert result["2020-Q2"]["BTCUSDT"] == pytest.approx(0.015, rel=1e-6)
        assert result["2020-Q2"]["ETHUSDT"] == pytest.approx(0.005, rel=1e-6)

    def test_total_preserved_per_fold(self):
        """For each fold, sum of final_weights == target_total_for_fold."""
        trades = [
            _make_trade("BTCUSDT", 0.5, "2020-01"),
            _make_trade("ETHUSDT", 0.3, "2020-01"),
            _make_trade("BTCUSDT", 0.7, "2020-02"),
            _make_trade("SOLUSDT", 0.4, "2020-02"),
        ]
        result = apply_pf_weighted_risk_allocation(
            trades=trades,
            halflife_folds=2,
            clamp_floor=0.5,
            clamp_ceiling=1.5,
            min_trades=10,
            baseline_per_symbol=0.01,
            n_total_symbols=3,
        )
        for fk, weights in result.items():
            total = sum(weights.values())
            n_active = len(weights)
            expected = 0.01 * 3 * (n_active / 3)  # baseline * n_total * budget_scale
            assert total == pytest.approx(expected, rel=1e-6), (
                f"Fold {fk}: total={total}, expected={expected}"
            )

    def test_per_trade_mutation(self):
        """Trades in each fold should have their risk_weight mutated."""
        trades = [
            _make_trade("BTCUSDT", 1.0, "F1", risk_weight=0.01),
            _make_trade("BTCUSDT", 0.5, "F2", risk_weight=0.01),
        ]
        apply_pf_weighted_risk_allocation(
            trades=trades,
            halflife_folds=2,
            clamp_floor=0.5,
            clamp_ceiling=1.5,
            min_trades=1,  # only 1 past trade per symbol
            baseline_per_symbol=0.01,
            n_total_symbols=1,
        )
        # F1: no past -> neutral factor (1.0) -> final = baseline = 0.01
        # F2: BTC has only 1 symbol; rescale preserves target_total = 0.01
        #      (factor 1.5 gets rescaled to maintain total)
        # In both folds, with n_active=1, the rescale nullifies the factor.
        # This is the X1 "preserve total" property at single-symbol edge case.
        assert trades[0]["risk_weight"] == pytest.approx(0.01, rel=1e-6)
        assert trades[1]["risk_weight"] == pytest.approx(0.01, rel=1e-6)
        # fold_key consumed
        assert "fold_key" not in trades[0]
        assert "fold_key" not in trades[1]

    def test_atr_inv_consumed(self):
        """The atr_inv field (set by WFA) should be popped after allocation."""
        trades = [
            {**_make_trade("BTCUSDT", 0.5, "F1"), "atr_inv": 0.05},
        ]
        apply_pf_weighted_risk_allocation(
            trades=trades,
            halflife_folds=2,
            clamp_floor=0.5,
            clamp_ceiling=1.5,
            min_trades=10,
            baseline_per_symbol=0.01,
            n_total_symbols=1,
        )
        assert "atr_inv" not in trades[0]

    def test_budget_scale_partial_universe(self):
        """When fewer symbols are active, target_total scales down."""
        trades = [
            _make_trade("BTCUSDT", 0.5, "F1"),  # only BTC
            # ETH and SOL absent
        ]
        result = apply_pf_weighted_risk_allocation(
            trades=trades,
            halflife_folds=2,
            clamp_floor=0.5,
            clamp_ceiling=1.5,
            min_trades=10,
            baseline_per_symbol=0.01,
            n_total_symbols=3,  # 3 total but only 1 active
        )
        # 1 active of 3: budget_scale = 1/3, target = 0.01 * 3 * 1/3 = 0.01
        # Factor = 1.0 (no past), so pre_rescale[BTC] = 0.01
        # rescale = 0.01 / 0.01 = 1.0, final[BTC] = 0.01
        assert result["F1"]["BTCUSDT"] == pytest.approx(0.01, rel=1e-6)

    def test_log_summary_emitted(self, caplog):
        """Orchestrator should log per-fold summary."""
        trades = [
            _make_trade("BTCUSDT", 0.5, "F1"),
        ]
        with caplog.at_level(logging.INFO, logger="rich"):
            apply_pf_weighted_risk_allocation(
                trades=trades,
                halflife_folds=2,
                clamp_floor=0.5,
                clamp_ceiling=1.5,
                min_trades=10,
                baseline_per_symbol=0.01,
                n_total_symbols=1,
            )
        # At least one log line should mention "PF Weight" and "F1"
        assert any("PF Weight" in r.message and "F1" in r.message for r in caplog.records)

    def test_returns_summary_per_fold(self):
        """Return value is {fold_key: {sym: final_weight}}."""
        trades = [
            _make_trade("BTCUSDT", 0.5, "F1"),
            _make_trade("ETHUSDT", 0.3, "F2"),
        ]
        result = apply_pf_weighted_risk_allocation(
            trades=trades,
            halflife_folds=2,
            clamp_floor=0.5,
            clamp_ceiling=1.5,
            min_trades=10,
            baseline_per_symbol=0.01,
            n_total_symbols=2,
        )
        assert set(result.keys()) == {"F1", "F2"}
        for fk, weights in result.items():
            assert isinstance(weights, dict)
            assert all(isinstance(v, float) for v in weights.values())


class TestDefaultBaseline:
    def test_returns_positive_value(self):
        baseline = default_baseline_per_symbol()
        assert baseline > 0
        assert baseline < 0.1  # sanity bound
