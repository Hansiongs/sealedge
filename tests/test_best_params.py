"""Tests for the new best-params selection (Q1).

These tests verify that ``pick_best_params_per_symbol`` correctly:
- Picks the fold with highest ``best_value`` (PSR) per symbol
- Falls back to last fold when ``best_value`` is missing
- Uses strategy-specific safe defaults when no folds
- Handles tie-breaking, missing keys, and non-numeric values
- Is generic across strategies (vol_compression, pullback_sniper)
"""

from quant_lib.research.best_params import pick_best_params_per_symbol


# ════════════════════════════════════════════════════════════════════════
# Best-Value Selection (the core Q1 behavior)
# ════════════════════════════════════════════════════════════════════════


class TestPickBestParamsByValue:
    def test_picks_highest_best_value(self):
        """Symbol with multiple folds: pick the one with highest best_value."""
        folds = [
            {"best_value": 0.5, "vol_pct_thresh": 0.10,
             "trail_atr": 2.0, "sl_mult": 1.0, "pullback_bars": 3},
            {"best_value": 0.9, "vol_pct_thresh": 0.25,  # WINNER
             "trail_atr": 3.5, "sl_mult": 1.8, "pullback_bars": 6},
            {"best_value": 0.7, "vol_pct_thresh": 0.20,
             "trail_atr": 3.0, "sl_mult": 1.5, "pullback_bars": 5},
        ]
        result = pick_best_params_per_symbol(
            {"BTCUSDT": folds}, strategy_type=0
        )
        assert result["BTCUSDT"]["vol_pct_thresh"] == 0.25
        assert result["BTCUSDT"]["trail_atr"] == 3.5
        assert result["BTCUSDT"]["sl_mult"] == 1.8
        assert result["BTCUSDT"]["pullback_bars"] == 6

    def test_first_fold_can_win(self):
        """Highest best_value in fold 0, not last fold."""
        folds = [
            {"best_value": 0.95, "vol_pct_thresh": 0.30,  # FIRST FOLD WINS
             "trail_atr": 4.0, "sl_mult": 2.0, "pullback_bars": 7},
            {"best_value": 0.5, "vol_pct_thresh": 0.20,
             "trail_atr": 3.0, "sl_mult": 1.5, "pullback_bars": 5},
        ]
        result = pick_best_params_per_symbol(
            {"BTCUSDT": folds}, strategy_type=0
        )
        assert result["BTCUSDT"]["vol_pct_thresh"] == 0.30
        assert result["BTCUSDT"]["trail_atr"] == 4.0

    def test_middle_fold_can_win(self):
        """Highest best_value in the middle fold, not first or last."""
        folds = [
            {"best_value": 0.5, "vol_pct_thresh": 0.10,
             "trail_atr": 2.0, "sl_mult": 1.0, "pullback_bars": 3},
            {"best_value": 0.9, "vol_pct_thresh": 0.25,  # MIDDLE WINS
             "trail_atr": 3.5, "sl_mult": 1.8, "pullback_bars": 6},
            {"best_value": 0.7, "vol_pct_thresh": 0.20,
             "trail_atr": 3.0, "sl_mult": 1.5, "pullback_bars": 5},
        ]
        result = pick_best_params_per_symbol(
            {"BTCUSDT": folds}, strategy_type=0
        )
        assert result["BTCUSDT"]["vol_pct_thresh"] == 0.25

    def test_tie_breaking_first_wins(self):
        """When multiple folds have the same best_value, first one wins.

        Python's max() returns the first occurrence on ties.
        """
        folds = [
            {"best_value": 0.8, "vol_pct_thresh": 0.10,  # tied
             "trail_atr": 2.0, "sl_mult": 1.0, "pullback_bars": 3},
            {"best_value": 0.8, "vol_pct_thresh": 0.30,  # tied
             "trail_atr": 4.0, "sl_mult": 2.0, "pullback_bars": 7},
        ]
        result = pick_best_params_per_symbol(
            {"BTCUSDT": folds}, strategy_type=0
        )
        # First fold (with vol=0.10) wins on tie
        assert result["BTCUSDT"]["vol_pct_thresh"] == 0.10

    def test_differs_from_last_fold_approach(self):
        """Sanity check: best-by-PSR != last fold when best is in middle."""
        folds = [
            {"best_value": 0.5, "vol_pct_thresh": 0.10,
             "trail_atr": 2.0, "sl_mult": 1.0, "pullback_bars": 3},
            {"best_value": 0.9, "vol_pct_thresh": 0.25,  # WINNER
             "trail_atr": 3.5, "sl_mult": 1.8, "pullback_bars": 6},
            {"best_value": 0.6, "vol_pct_thresh": 0.20,  # last fold
             "trail_atr": 3.0, "sl_mult": 1.5, "pullback_bars": 5},
        ]
        result = pick_best_params_per_symbol(
            {"BTCUSDT": folds}, strategy_type=0
        )
        # Uses middle fold params, not last fold
        assert result["BTCUSDT"]["vol_pct_thresh"] == 0.25  # middle
        assert result["BTCUSDT"]["vol_pct_thresh"] != 0.20  # not last

    def test_negative_best_value_picked_correctly(self):
        """PSR can be negative. Lowest negative is worse than highest negative."""
        folds = [
            {"best_value": -0.3, "vol_pct_thresh": 0.10,
             "trail_atr": 2.0, "sl_mult": 1.0, "pullback_bars": 3},
            {"best_value": -0.1, "vol_pct_thresh": 0.25,  # LEAST BAD
             "trail_atr": 3.5, "sl_mult": 1.8, "pullback_bars": 6},
        ]
        result = pick_best_params_per_symbol(
            {"BTCUSDT": folds}, strategy_type=0
        )
        assert result["BTCUSDT"]["vol_pct_thresh"] == 0.25


# ════════════════════════════════════════════════════════════════════════
# Fallbacks and Edge Cases
# ════════════════════════════════════════════════════════════════════════


class TestFallbacks:
    def test_no_symbols_returns_empty(self):
        result = pick_best_params_per_symbol({}, strategy_type=0)
        assert result == {}

    def test_symbol_with_no_folds_uses_vol_compression_defaults(self):
        result = pick_best_params_per_symbol(
            {"BTCUSDT": []}, strategy_type=0
        )
        assert result["BTCUSDT"]["vol_pct_thresh"] == 0.20
        assert result["BTCUSDT"]["pullback_bars"] == 5
        assert result["BTCUSDT"]["trail_atr"] == 3.0
        assert result["BTCUSDT"]["sl_mult"] == 1.5

    def test_symbol_with_no_folds_uses_pullback_sniper_defaults(self):
        result = pick_best_params_per_symbol(
            {"BTCUSDT": []}, strategy_type=1
        )
        # Base 4 keys
        assert result["BTCUSDT"]["vol_pct_thresh"] == 0.20
        assert result["BTCUSDT"]["pullback_bars"] == 5
        assert result["BTCUSDT"]["trail_atr"] == 3.0
        assert result["BTCUSDT"]["sl_mult"] == 1.5
        # Plus RSI defaults
        assert result["BTCUSDT"]["rsi_oversold"] == 30.0
        assert result["BTCUSDT"]["rsi_overbought"] == 70.0

    def test_vol_compression_defaults_exclude_rsi(self):
        """strategy_type=0 should not include RSI in defaults."""
        result = pick_best_params_per_symbol(
            {"BTCUSDT": []}, strategy_type=0
        )
        assert "rsi_oversold" not in result["BTCUSDT"]
        assert "rsi_overbought" not in result["BTCUSDT"]

    def test_fallback_to_last_when_no_best_value(self):
        """If no fold has best_value, fall back to last fold."""
        folds = [
            {"vol_pct_thresh": 0.10, "trail_atr": 2.0,
             "sl_mult": 1.0, "pullback_bars": 3},  # no best_value
            {"vol_pct_thresh": 0.30, "trail_atr": 4.0,  # last, no best_value
             "sl_mult": 2.5, "pullback_bars": 7},
        ]
        result = pick_best_params_per_symbol(
            {"BTCUSDT": folds}, strategy_type=0
        )
        # Falls back to last fold
        assert result["BTCUSDT"]["vol_pct_thresh"] == 0.30
        assert result["BTCUSDT"]["trail_atr"] == 4.0

    def test_mixed_folds_with_and_without_best_value(self):
        """Some folds have best_value, some don't. Use the best best_value."""
        folds = [
            {"vol_pct_thresh": 0.10, "trail_atr": 2.0,  # no best_value
             "sl_mult": 1.0, "pullback_bars": 3},
            {"best_value": 0.85, "vol_pct_thresh": 0.25,  # has best_value
             "trail_atr": 3.5, "sl_mult": 1.8, "pullback_bars": 6},
            {"best_value": 0.5, "vol_pct_thresh": 0.20,  # has best_value
             "trail_atr": 3.0, "sl_mult": 1.5, "pullback_bars": 5},
        ]
        result = pick_best_params_per_symbol(
            {"BTCUSDT": folds}, strategy_type=0
        )
        # Fold with best_value=0.85 wins
        assert result["BTCUSDT"]["vol_pct_thresh"] == 0.25


# ════════════════════════════════════════════════════════════════════════
# Multi-Symbol
# ════════════════════════════════════════════════════════════════════════


class TestMultiSymbol:
    def test_each_symbol_uses_own_best(self):
        """Per-symbol best, not global best."""
        btc_folds = [
            {"best_value": 0.5, "vol_pct_thresh": 0.10,
             "trail_atr": 2.0, "sl_mult": 1.0, "pullback_bars": 3},
            {"best_value": 0.6, "vol_pct_thresh": 0.20,
             "trail_atr": 3.0, "sl_mult": 1.5, "pullback_bars": 5},
        ]
        eth_folds = [
            {"best_value": 0.9, "vol_pct_thresh": 0.30,  # ETH best
             "trail_atr": 4.0, "sl_mult": 2.0, "pullback_bars": 7},
            {"best_value": 0.4, "vol_pct_thresh": 0.15,
             "trail_atr": 2.5, "sl_mult": 1.2, "pullback_bars": 4},
        ]
        result = pick_best_params_per_symbol(
            {"BTCUSDT": btc_folds, "ETHUSDT": eth_folds}, strategy_type=0
        )
        # BTC's best (0.6, 0.20)
        assert result["BTCUSDT"]["vol_pct_thresh"] == 0.20
        assert result["BTCUSDT"]["trail_atr"] == 3.0
        # ETH's best (0.9, 0.30)
        assert result["ETHUSDT"]["vol_pct_thresh"] == 0.30
        assert result["ETHUSDT"]["trail_atr"] == 4.0

    def test_symbol_not_in_fold_params_uses_defaults(self):
        """Symbol in `symbols` list but not in dict -- wait, the new
        function takes the dict keys directly, not a separate symbols
        list. Verify it iterates dict keys."""
        result = pick_best_params_per_symbol(
            {"BTCUSDT": [], "ETHUSDT": []}, strategy_type=0
        )
        assert "BTCUSDT" in result
        assert "ETHUSDT" in result
        # Both have defaults
        assert result["BTCUSDT"]["vol_pct_thresh"] == 0.20
        assert result["ETHUSDT"]["vol_pct_thresh"] == 0.20

    def test_one_symbol_with_folds_one_without(self):
        """Mixed: one symbol has folds, other doesn't."""
        folds_with = [
            {"best_value": 0.9, "vol_pct_thresh": 0.30,
             "trail_atr": 4.0, "sl_mult": 2.0, "pullback_bars": 7},
        ]
        result = pick_best_params_per_symbol(
            {"BTCUSDT": folds_with, "ETHUSDT": []}, strategy_type=0
        )
        # BTC uses fold params
        assert result["BTCUSDT"]["vol_pct_thresh"] == 0.30
        # ETH uses defaults
        assert result["ETHUSDT"]["vol_pct_thresh"] == 0.20


# ════════════════════════════════════════════════════════════════════════
# Generic extraction & strategy-specific behavior
# ════════════════════════════════════════════════════════════════════════


class TestGenericExtraction:
    def test_pullback_sniper_includes_rsi(self):
        """Regression test for C-3: pullback_sniper RSI params
        must be included in frozen params."""
        folds = [
            {"best_value": 0.85,
             "vol_pct_thresh": 0.20, "pullback_bars": 5,
             "trail_atr": 3.0, "sl_mult": 1.5,
             "rsi_oversold": 28.0, "rsi_overbought": 72.0},
        ]
        result = pick_best_params_per_symbol(
            {"BTCUSDT": folds}, strategy_type=1
        )
        assert result["BTCUSDT"]["rsi_oversold"] == 28.0
        assert result["BTCUSDT"]["rsi_overbought"] == 72.0
        # Base 4 keys still preserved
        assert result["BTCUSDT"]["vol_pct_thresh"] == 0.20
        assert result["BTCUSDT"]["trail_atr"] == 3.0

    def test_pullback_sniper_backfill_rsi_defaults(self):
        """If best fold for pullback_sniper is missing RSI keys,
        safe defaults are used."""
        folds = [
            {"best_value": 0.85, "vol_pct_thresh": 0.20,
             "pullback_bars": 5, "trail_atr": 3.0, "sl_mult": 1.5},
        ]
        result = pick_best_params_per_symbol(
            {"BTCUSDT": folds}, strategy_type=1
        )
        # Backfilled from pullback_sniper safe defaults
        assert result["BTCUSDT"]["rsi_oversold"] == 30.0
        assert result["BTCUSDT"]["rsi_overbought"] == 70.0

    def test_extra_keys_preserved_generic(self):
        """If fold has extra numeric keys (e.g., future param), preserve them."""
        folds = [
            {"best_value": 0.85, "vol_pct_thresh": 0.20,
             "pullback_bars": 5, "trail_atr": 3.0, "sl_mult": 1.5,
             "future_param_v2": 42.0},
        ]
        result = pick_best_params_per_symbol(
            {"BTCUSDT": folds}, strategy_type=0
        )
        assert result["BTCUSDT"]["future_param_v2"] == 42.0

    def test_metadata_excluded(self):
        """Fold metadata (best_value, fold, dates) must NOT appear in frozen."""
        folds = [
            {
                "best_value": 0.85,
                "fold": 5, "total_folds": 10,
                "is_start": "2024-01-01", "oos_start": "2024-04-01",
                "oos_end": "2024-07-01",
                "symbol": "BTCUSDT",  # duplicate of key in all_fold_params
                "vol_pct_thresh": 0.20, "pullback_bars": 5,
                "trail_atr": 3.0, "sl_mult": 1.5,
            }
        ]
        result = pick_best_params_per_symbol(
            {"BTCUSDT": folds}, strategy_type=0
        )
        # Metadata must be filtered
        for meta_key in (
            "best_value", "fold", "total_folds",
            "is_start", "oos_start", "oos_end", "symbol",
        ):
            assert meta_key not in result["BTCUSDT"], (
                f"Metadata key '{meta_key}' must not be in frozen_params"
            )
        # Real params preserved
        assert result["BTCUSDT"]["vol_pct_thresh"] == 0.20
        assert result["BTCUSDT"]["trail_atr"] == 3.0

    def test_non_numeric_values_skipped(self):
        """Defensive: strings, lists, None in fold are skipped."""
        folds = [
            {"best_value": 0.85, "vol_pct_thresh": 0.20,
             "pullback_bars": 5, "trail_atr": 3.0, "sl_mult": 1.5,
             "weird_string": "not_a_number",
             "weird_list": [1, 2, 3],
             "weird_none": None,
             "weird_bool": True,
             },
        ]
        result = pick_best_params_per_symbol(
            {"BTCUSDT": folds}, strategy_type=0
        )
        # Real numeric keys preserved
        assert result["BTCUSDT"]["vol_pct_thresh"] == 0.20
        # Non-numeric values NOT in frozen
        assert "weird_string" not in result["BTCUSDT"]
        assert "weird_list" not in result["BTCUSDT"]
        assert "weird_none" not in result["BTCUSDT"]
        assert "weird_bool" not in result["BTCUSDT"]


# ════════════════════════════════════════════════════════════════════════
# pullback_bars int conversion
# ════════════════════════════════════════════════════════════════════════


class TestPullbackBarsInt:
    def test_pullback_bars_is_int(self):
        """pullback_bars should be cast to int (rounded)."""
        folds = [
            {"best_value": 0.85, "vol_pct_thresh": 0.20,
             "pullback_bars": 5.7, "trail_atr": 3.0, "sl_mult": 1.5},
        ]
        result = pick_best_params_per_symbol(
            {"BTCUSDT": folds}, strategy_type=0
        )
        assert isinstance(result["BTCUSDT"]["pullback_bars"], int)
        assert result["BTCUSDT"]["pullback_bars"] == 6  # round(5.7) = 6

    def test_pullback_bars_rounds_down(self):
        """round(5.4) = 5."""
        folds = [
            {"best_value": 0.85, "vol_pct_thresh": 0.20,
             "pullback_bars": 5.4, "trail_atr": 3.0, "sl_mult": 1.5},
        ]
        result = pick_best_params_per_symbol(
            {"BTCUSDT": folds}, strategy_type=0
        )
        assert result["BTCUSDT"]["pullback_bars"] == 5

    def test_pullback_bars_int_unchanged(self):
        """If already int, stays int."""
        folds = [
            {"best_value": 0.85, "vol_pct_thresh": 0.20,
             "pullback_bars": 7, "trail_atr": 3.0, "sl_mult": 1.5},
        ]
        result = pick_best_params_per_symbol(
            {"BTCUSDT": folds}, strategy_type=0
        )
        assert isinstance(result["BTCUSDT"]["pullback_bars"], int)
        assert result["BTCUSDT"]["pullback_bars"] == 7
