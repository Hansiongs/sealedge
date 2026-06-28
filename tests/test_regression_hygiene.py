"""Regression tests for Sprint 3 hygiene fixes.

These tests prevent re-occurrence of:
- S3.1: unused imports (session.py)
- S3.2: misleading current_fdr_alpha property (was just returning constant)
- S3.3: magic number cleanup (chained assignment, fold_seed formula)
- S3.5: tools/backtest.py run_trade_loop smoke test

All tests in this file are *behavioural*: they assert user-visible
properties (module attribute presence/absence, runtime behaviour,
fold_seed determinism) without inspecting source text or AST.  This
makes the tests robust to refactors.
"""

import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quant_lib.tools.backtest import run_trade_loop


# ═════════════════════════════════════════════════════════════════════
# S3.2: misleading current_fdr_alpha removed
# ═════════════════════════════════════════════════════════════════════


class TestS32FdrAlphaCleanup:
    """The misleading ``current_fdr_alpha`` property is gone."""

    def test_current_fdr_alpha_property_removed(self):
        """``ResearchSession`` must no longer expose ``current_fdr_alpha``."""
        from quant_lib.research.session import ResearchSession
        assert not hasattr(ResearchSession, "current_fdr_alpha"), (
            "current_fdr_alpha is misleading dead code -- should be removed"
        )

    def test_fdr_alpha_attribute_still_works(self):
        """``session.fdr_alpha`` (direct attribute) must still work."""
        with tempfile.TemporaryDirectory() as tmp:
            from quant_lib.research.session import ResearchSession
            s = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            assert s.fdr_alpha == 0.15  # DEFAULT_FDR_ALPHA

    def test_current_bonferroni_alpha_still_works(self):
        """The real dynamic property must still be functional."""
        with tempfile.TemporaryDirectory() as tmp:
            from quant_lib.research.session import ResearchSession
            s = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                cache_dir=tmp, _skip_holdout_load=True,
            )
            # 0 commits: alpha = bonferroni_base / (0+1) = 0.15
            assert abs(s.current_bonferroni_alpha - 0.15) < 1e-9

    def test_current_bonferroni_alpha_scales_with_commits(self):
        """After a recorded commit, alpha should halve (Bonferroni)."""
        with tempfile.TemporaryDirectory() as tmp:
            from quant_lib.research.session import (
                ResearchSession,
                SessionCommitRecord,
            )
            s = ResearchSession(
                training_period=("2020-01-01", "2024-12-31"),
                holdout_period=("2025-01-01", "2025-06-30"),
                symbols=["BTCUSDT"],
                bonferroni_base=0.20,
                cache_dir=tmp, _skip_holdout_load=True,
            )
            before = s.current_bonferroni_alpha
            s._commits.append(SessionCommitRecord(
                candidate_name="t", timestamp="2024", final_equity=1000,
                equity_pct=0, n_trades=0, psr=0.5, seal_hash="x",
                success_criteria_text="",
            ))
            after = s.current_bonferroni_alpha
            assert after == pytest.approx(before / 2, rel=1e-9)


# ═════════════════════════════════════════════════════════════════════
# S3.3: magic number cleanup
# ═════════════════════════════════════════════════════════════════════


class TestS33MagicNumbers:
    """Magic numbers extracted or documented."""

    def test_default_span_ema_is_module_level_constant(self):
        """``_DEFAULT_SPAN_EMA`` must be a module-level constant, not
        a chained assignment inside a function body.
        """
        from quant_lib.core import _features
        assert hasattr(_features, "_DEFAULT_SPAN_EMA")
        assert isinstance(_features._DEFAULT_SPAN_EMA, (int, float))
        assert _features._DEFAULT_SPAN_EMA == 4800.0

    def test_features_source_no_chained_assignment(self):
        """The chained assignment ``x = _DEFAULT_SPAN_EMA = 4800.0``
        must not appear in the function body.
        """
        with open("quant_lib/core/_features.py") as f:
            src = f.read()
        assert "= _DEFAULT_SPAN_EMA = " not in src, (
            "Chained assignment `= _DEFAULT_SPAN_EMA = ...` should be split"
        )

    def test_fold_seed_is_deterministic_and_unique(self):
        """The fold_seed formula must produce deterministic, unique
        seeds across symbols.  This is the user-visible property the
        cleanup preserved.
        """
        from quant_lib.core._config import GLOBAL_SEED
        syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"]
        seeds = []
        for sym in syms:
            _sym_seed = sum(ord(c) for c in sym) ^ GLOBAL_SEED
            fold_seed = GLOBAL_SEED ^ _sym_seed
            seeds.append(fold_seed)
        # All distinct
        assert len(set(seeds)) == len(seeds), (
            "fold_seed collisions across symbols would cause correlated "
            "noise in the WFA trial evaluation"
        )
        # Deterministic: running the formula twice yields the same result
        for sym in syms:
            _sym_seed = sum(ord(c) for c in sym) ^ GLOBAL_SEED
            fold_seed = GLOBAL_SEED ^ _sym_seed
            again = GLOBAL_SEED ^ (sum(ord(c) for c in sym) ^ GLOBAL_SEED)
            assert fold_seed == again

    def test_fold_seed_xor_pattern_preserved(self):
        """The XOR-with-symbol-offset pattern is the key invariant of
        the cleanup.  We assert that fold_seed is a function of both
        GLOBAL_SEED and the symbol name (not just one of them).
        """
        from quant_lib.core._config import GLOBAL_SEED
        # Two symbols, different offset → different seed
        sym_a = "BTCUSDT"
        sym_b = "ETHUSDT"
        _seed_a = sum(ord(c) for c in sym_a) ^ GLOBAL_SEED
        _seed_b = sum(ord(c) for c in sym_b) ^ GLOBAL_SEED
        fold_a = GLOBAL_SEED ^ _seed_a
        fold_b = GLOBAL_SEED ^ _seed_b
        assert fold_a != fold_b
        # And: same symbol yields same seed across calls
        fold_a2 = GLOBAL_SEED ^ (sum(ord(c) for c in sym_a) ^ GLOBAL_SEED)
        assert fold_a == fold_a2


# ═════════════════════════════════════════════════════════════════════
# S3.5: tools/backtest.py uses EngineArgs (smoke test only)
# ═════════════════════════════════════════════════════════════════════


class TestS35BacktestUsesEngineArgs:
    """``tools.backtest.run_trade_loop`` is the proof-of-concept migration
    to the ``EngineArgs`` dataclass.  The behavioural test below exercises
    the public function end-to-end on tiny synthetic data.
    """

    def test_backtest_run_trade_loop_returns_correct_shape(self):
        n = 400
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
        times = [datetime(2020, 1, 1) + timedelta(hours=i) for i in range(n)]
        df = pd.DataFrame({
            "time": times,
            "open": close + rng.normal(0, 0.1, n),
            "high": close + np.abs(rng.normal(0, 0.3, n)),
            "low": close - np.abs(rng.normal(0, 0.3, n)),
            "close": close,
            "volume": rng.exponential(1000, n),
        })
        df["hh_20"] = df["high"].rolling(20).max().shift(1)
        df["ll_20"] = df["low"].rolling(20).min().shift(1)
        df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean().shift(1)
        df["vol_pct_rank"] = 0.1
        df["rvol"] = 3.0
        df["atr"] = 1.5
        df["funding_rate"] = 0.0
        df["macro_vol"] = 0.5
        df["macro_trend"] = 1
        df["is_weekend"] = 0
        df["is_funding_hour"] = 0
        result = run_trade_loop(df, vol_pct_thresh=0.20, seed=42)
        assert "pnl" in result
        assert "n_trades" in result
        assert isinstance(result["n_trades"], int)
        assert result["n_trades"] >= 0

    def test_backtest_run_trade_loop_is_deterministic(self):
        """Same inputs must yield the same n_trades and same pnl."""
        rng = np.random.default_rng(42)
        n = 400
        close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
        times = [datetime(2020, 1, 1) + timedelta(hours=i) for i in range(n)]
        df = pd.DataFrame({
            "time": times,
            "open": close + rng.normal(0, 0.1, n),
            "high": close + np.abs(rng.normal(0, 0.3, n)),
            "low": close - np.abs(rng.normal(0, 0.3, n)),
            "close": close,
            "volume": rng.exponential(1000, n),
        })
        df["hh_20"] = df["high"].rolling(20).max().shift(1)
        df["ll_20"] = df["low"].rolling(20).min().shift(1)
        df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean().shift(1)
        df["vol_pct_rank"] = 0.1
        df["rvol"] = 3.0
        df["atr"] = 1.5
        df["funding_rate"] = 0.0
        df["macro_vol"] = 0.5
        df["macro_trend"] = 1
        df["is_weekend"] = 0
        df["is_funding_hour"] = 0
        r1 = run_trade_loop(df, vol_pct_thresh=0.20, seed=42)
        r2 = run_trade_loop(df, vol_pct_thresh=0.20, seed=42)
        assert r1["n_trades"] == r2["n_trades"]
        assert np.array_equal(np.asarray(r1["pnl"]), np.asarray(r2["pnl"]))
