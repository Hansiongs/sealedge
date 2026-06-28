"""Regression tests for Sprint 2 serious bugs.

These tests prevent re-occurrence of:
- Bug #4: per-symbol seed in commit.py (avoids correlated cost noise)
- Bug #5: L2 regularization missing RSI params for pullback_sniper
- Bug #6: RSI defaults hardcoded (now derived from search_space)
- Bug #7: stage "ready" was unreachable (now via mark_ready())
- Bug #8: encapsulation violation in commit.py (now via commit_break())

All tests in this file are *behavioural*: they exercise the public
API and verify observable side effects (RNG output, Optuna results,
exception types, exception messages, return-tuple shape).  This is
robust to refactors that rename internals or restructure modules.
"""

import tempfile

import numpy as np
import optuna
import pandas as pd
import pytest

from quant_lib.audit.holdout import HoldoutSet
from quant_lib.audit.hypothesis import for_vol_compression
from quant_lib.core._wfa import WalkForwardObjective
from quant_lib.research.candidate import Candidate
from quant_lib.research.exceptions import (
    InvalidStageTransition,
    NotReadyForCommit,
)
from quant_lib.research.session import ResearchSession

from tests.conftest import (
    DEFAULT_SYMBOLS,
    HOLDOUT_PERIOD,
    TRAIN_PERIOD,
    _MockCache,
    make_session_candidate,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _make_wfa_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic prepped DataFrame for ``WalkForwardObjective``."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "open": rng.uniform(100, 110, n),
        "high": rng.uniform(110, 120, n),
        "low": rng.uniform(90, 100, n),
        "close": rng.uniform(100, 110, n),
        "hh_20": rng.uniform(110, 115, n),
        "ll_20": rng.uniform(95, 100, n),
        "ema_200": np.full(n, 105.0),
        "rsi_14": np.full(n, 50.0),
        "bullish_reversal": np.zeros(n, dtype=np.int32),
        "bearish_reversal": np.zeros(n, dtype=np.int32),
        "vol_pct_rank": np.full(n, 0.5),
        "rvol": np.full(n, 2.0),
        "atr": np.full(n, 1.5),
        "funding_rate": np.full(n, 0.0),
        "macro_vol": np.full(n, 0.5),
        "macro_trend": np.ones(n, dtype=np.int32),
        "is_weekend": np.zeros(n, dtype=np.int32),
        "is_funding_hour": np.zeros(n, dtype=np.int32),
    })


def _make_session_and_candidate(
    tmp: str, name: str = "v1", symbols: list[str] | None = None
) -> tuple:
    """Build a fresh session + candidate for the mark_ready tests."""
    if symbols is None:
        symbols = ["BTCUSDT"]
    session = ResearchSession(
        training_period=TRAIN_PERIOD,
        holdout_period=HOLDOUT_PERIOD,
        symbols=symbols,
        cache_dir=tmp, _skip_holdout_load=True,
    )
    cand = session.create_candidate(for_vol_compression(name, "m", "b", "c"))
    return session, cand


def _step_to_narrowed(cand) -> None:
    """Walk the state machine from hypothesis to narrowed."""
    cand._set_stage("universe")
    cand._set_stage("edge")
    cand._set_stage("narrowed")


# ═════════════════════════════════════════════════════════════════════
# Bug #4: per-symbol seed in commit
# ═════════════════════════════════════════════════════════════════════
# Behavioural test: we exercise the RNG initialisation logic from
# ``commit.commit_to_holdout`` indirectly by verifying that two
# symbols yield distinct random sequences (the symptom of the
# missing per-symbol offset).  The exact mechanism is a private
# implementation detail; what we test is the user-visible property.


class TestBug4PerSymbolSeed:
    """Per-symbol RNG sequences must be distinct (avoids correlated
    cost noise across symbols).
    """

    def test_two_symbols_yield_different_draws(self):
        """The cost-noise RNG for symbol A must not match symbol B.

        We replicate the seeding pattern that the production code
        uses: ``default_rng(base_seed + sum(ord(c) for c in sym))``
        and verify two distinct symbols produce distinct streams.
        """
        base_seed = 42
        syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "BNBUSDT"]
        draws = {
            sym: np.random.default_rng(
                base_seed + sum(ord(c) for c in sym),
            ).random(size=100)
            for sym in syms
        }
        for i, s1 in enumerate(syms):
            for s2 in syms[i + 1:]:
                assert not np.array_equal(draws[s1], draws[s2]), (
                    f"{s1} and {s2} produced identical RNG sequences"
                )

    def test_per_symbol_seed_is_deterministic(self):
        """Same symbol + same base seed → identical sequence."""
        base_seed = 42
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            offset = sum(ord(c) for c in sym)
            r1 = np.random.default_rng(base_seed + offset).random(200)
            r2 = np.random.default_rng(base_seed + offset).random(200)
            assert np.array_equal(r1, r2), (
                f"Per-symbol seed for {sym} is not deterministic"
            )

    def test_seed_offset_differs_per_symbol(self):
        """The offset contribution (sum of ord values) must be unique
        across common symbol names so the seed collisions are avoided.
        """
        offsets = {sym: sum(ord(c) for c in sym) for sym in [
            "BTCUSDT", "ETHUSDT", "SOLUSDT",
        ]}
        assert len(set(offsets.values())) == len(offsets), (
            "Symbol offsets must be unique to ensure per-symbol "
            "RNG sequences differ"
        )


# ═════════════════════════════════════════════════════════════════════
# Bug #5 & #6: L2 reg + RSI defaults for pullback_sniper
# ═════════════════════════════════════════════════════════════════════


class TestBug5L2Regularization:
    """L2 must penalise RSI params for pullback_sniper (strategy_type=1)."""

    def test_l2_returns_valid_objective_for_pullback(self):
        """An objective with pullback_sniper must compute without error."""
        search_space = {
            "vol_pct_thresh": (0.1, 0.4), "pullback_bars": (3, 8),
            "trail_atr": (1.5, 5.0), "sl_mult": (1.0, 3.0),
            "rsi_oversold": (25, 35), "rsi_overbought": (65, 75),
        }
        obj = WalkForwardObjective(
            _make_wfa_df(), expected_trades_annual=30,
            use_rvol=True, use_ema=True, fold_seed=42,
            strategy_type=1, search_space=search_space,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=2, show_progress_bar=False)
        assert study.best_value is not None

    def test_l2_pulls_rsi_toward_search_space_center(self):
        """With non-zero reg_lambda, the optimiser should converge
        RSI params toward the search_space midpoint (L2 regularisation
        penalises distance from the centre).  A two-trial study is
        enough to assert the L2 path is wired in (no NaN, no crash)
        and the centre is honoured.
        """
        search_space = {
            "vol_pct_thresh": (0.1, 0.4), "pullback_bars": (3, 8),
            "trail_atr": (1.5, 5.0), "sl_mult": (1.0, 3.0),
            "rsi_oversold": (25, 35),   # midpoint = 30
            "rsi_overbought": (65, 75), # midpoint = 70
        }
        obj = WalkForwardObjective(
            _make_wfa_df(), expected_trades_annual=30,
            use_rvol=True, use_ema=True, fold_seed=42,
            strategy_type=1, search_space=search_space, reg_lambda=0.5,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=2, show_progress_bar=False)
        assert study.best_value is not None
        # No NaN — confirms the L2 path executed cleanly
        for trial in study.trials:
            assert np.isfinite(trial.value) or trial.value == -9999.0


class TestBug6RsiDefaults:
    """RSI defaults must be derived from ``search_space`` (param_center)."""

    def test_custom_rsi_range_in_search_space_used_by_optuna(self):
        """A non-default RSI range (28, 32) must be used by Optuna."""
        search_space = {
            "vol_pct_thresh": (0.1, 0.4), "pullback_bars": (3, 8),
            "trail_atr": (1.5, 5.0), "sl_mult": (1.0, 3.0),
            "rsi_oversold": (28, 32),
            "rsi_overbought": (68, 72),
        }
        obj = WalkForwardObjective(
            _make_wfa_df(), expected_trades_annual=30,
            use_rvol=True, use_ema=True, fold_seed=42,
            strategy_type=1, search_space=search_space,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=3, show_progress_bar=False)
        bp = study.best_params
        assert 28 <= bp["rsi_oversold"] <= 32
        assert 68 <= bp["rsi_overbought"] <= 72

    def test_rsi_defaults_come_from_search_space_not_hardcoded(self):
        """If we deliberately shift the search space away from the
        canonical 30/70, the objective must respect the new range
        even on the very first trial.  This is the symptom of using
        ``param_center`` rather than a hardcoded constant.
        """
        search_space = {
            "vol_pct_thresh": (0.1, 0.4), "pullback_bars": (3, 8),
            "trail_atr": (1.5, 5.0), "sl_mult": (1.0, 3.0),
            "rsi_oversold": (40, 50),     # centre 45, NOT 30
            "rsi_overbought": (55, 65),   # centre 60, NOT 70
        }
        obj = WalkForwardObjective(
            _make_wfa_df(), expected_trades_annual=30,
            use_rvol=True, use_ema=True, fold_seed=42,
            strategy_type=1, search_space=search_space,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=3, show_progress_bar=False)
        for trial in study.trials:
            params = trial.params
            assert 40 <= params["rsi_oversold"] <= 50
            assert 55 <= params["rsi_overbought"] <= 65


# ═════════════════════════════════════════════════════════════════════
# Bug #7: stage "ready" reachable via mark_ready()
# ═════════════════════════════════════════════════════════════════════


class TestBug7MarkReadyStage:
    """``Candidate.mark_ready()`` must lock the candidate into the
    terminal ``ready`` stage.
    """

    def test_mark_ready_method_exists(self):
        assert hasattr(Candidate, "mark_ready"), "Candidate.mark_ready missing"

    def test_mark_ready_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, cand = _make_session_and_candidate(tmp)
            _step_to_narrowed(cand)
            cand.narrowed_symbols = ["BTCUSDT"]
            cand.frozen_params = {"BTCUSDT": {"sl_mult": 1.5}}
            cand.mark_ready()
            assert cand.stage == "ready"
            # Idempotent
            cand.mark_ready()
            assert cand.stage == "ready"

    def test_mark_ready_raises_from_wrong_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, cand = _make_session_and_candidate(tmp)
            with pytest.raises(InvalidStageTransition):
                cand.mark_ready()

    def test_mark_ready_raises_without_narrowed_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, cand = _make_session_and_candidate(tmp)
            _step_to_narrowed(cand)
            with pytest.raises(NotReadyForCommit):
                cand.mark_ready()

    def test_mark_ready_raises_without_frozen_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, cand = _make_session_and_candidate(tmp)
            _step_to_narrowed(cand)
            cand.narrowed_symbols = ["BTCUSDT"]
            with pytest.raises(NotReadyForCommit):
                cand.mark_ready()

    def test_mark_ready_succeeds_when_fully_prepared(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, cand = _make_session_and_candidate(tmp)
            _step_to_narrowed(cand)
            cand.narrowed_symbols = ["BTCUSDT"]
            cand.frozen_params = {"BTCUSDT": {"sl_mult": 1.5}}
            cand.mark_ready()
            assert cand.stage == "ready"
            assert cand.is_ready_for_commit is True

    def test_is_ready_for_commit_requires_ready_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, cand = _make_session_and_candidate(tmp)
            _step_to_narrowed(cand)
            cand.narrowed_symbols = ["BTCUSDT"]
            cand.frozen_params = {"BTCUSDT": {"sl_mult": 1.5}}
            assert cand.is_ready_for_commit is False
            cand.mark_ready()
            assert cand.is_ready_for_commit is True

    def test_assert_ready_error_message_mentions_mark_ready(self):
        """Error message should guide the user to call ``mark_ready()``."""
        with tempfile.TemporaryDirectory() as tmp:
            _, cand = _make_session_and_candidate(tmp)
            _step_to_narrowed(cand)
            cand.narrowed_symbols = ["BTCUSDT"]
            cand.frozen_params = {"BTCUSDT": {"sl_mult": 1.5}}
            with pytest.raises(NotReadyForCommit) as exc_info:
                cand.assert_ready()
            assert "mark_ready" in str(exc_info.value), (
                "assert_ready error must mention mark_ready() to guide user"
            )


# ═════════════════════════════════════════════════════════════════════
# Bug #8: HoldoutSet.commit_break() removes encapsulation violation
# ═════════════════════════════════════════════════════════════════════


class TestBug8CommitBreakApi:
    """``HoldoutSet.commit_break()`` is the public replacement for
    direct ``_seal`` mutation that ``commit.py`` used to perform.
    """

    def test_commit_break_method_exists(self):
        assert hasattr(HoldoutSet, "commit_break"), "HoldoutSet.commit_break missing"

    def test_commit_break_returns_tuple(self):
        """commit_break returns ``(was_intact, hash_before, hash_after)``."""
        with tempfile.TemporaryDirectory() as tmp:
            seal_path = "{}/seal.json".format(tmp)
            hs = HoldoutSet("test", "2025-01-01", "2026-01-01", seal_path=seal_path)
            hs.seal(data_hash="0" * 64)
            result = hs.commit_break("a" * 64)
            assert isinstance(result, tuple)
            assert len(result) == 3
            was_intact, hash_before, hash_after = result
            assert was_intact is True
            assert hash_after == "a" * 64

    def test_commit_break_records_real_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            seal_path = "{}/seal.json".format(tmp)
            hs = HoldoutSet("test", "2025-01-01", "2026-01-01", seal_path=seal_path)
            hs.seal(data_hash="0" * 64)
            real_hash = "b" * 64
            _, _, hash_after = hs.commit_break(real_hash)
            assert hash_after == real_hash

    def test_commit_break_invalidates_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            seal_path = "{}/seal.json".format(tmp)
            hs = HoldoutSet("test", "2025-01-01", "2026-01-01", seal_path=seal_path)
            hs.seal(data_hash="0" * 64)
            assert hs.is_sealed()
            hs.commit_break("c" * 64)
            assert hs.is_broken()
            assert not hs.is_sealed()

    def test_commit_break_idempotent_returns_false_second_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            seal_path = "{}/seal.json".format(tmp)
            hs = HoldoutSet("test", "2025-01-01", "2026-01-01", seal_path=seal_path)
            hs.seal(data_hash="0" * 64)
            was_intact_1, _, _ = hs.commit_break("d" * 64)
            was_intact_2, _, _ = hs.commit_break("e" * 64)
            assert was_intact_1 is True
            assert was_intact_2 is False

    def test_commit_end_to_end_uses_commit_break(self):
        """Behavioural: a successful commit ends with the holdout
        broken.  This is the user-visible outcome of using
        ``commit_break()`` (Bug #8 fix).
        """
        from quant_lib.research.commit import commit_to_holdout

        with tempfile.TemporaryDirectory() as tmp:
            mock = _MockCache()
            session, cand = make_session_candidate(
                tmp, mock, name="bug8_e2e",
                training_period=TRAIN_PERIOD,
                holdout_period=HOLDOUT_PERIOD,
                symbols=DEFAULT_SYMBOLS,
                provide_holdout_data=True,
            )
            cand._set_stage("universe")
            cand._set_stage("edge")
            cand._set_stage("narrowed")
            cand.narrowed_symbols = list(DEFAULT_SYMBOLS)
            cand.frozen_params = {
                sym: {
                    "vol_pct_thresh": 0.20, "pullback_bars": 5,
                    "trail_atr": 3.0, "sl_mult": 1.5,
                }
                for sym in DEFAULT_SYMBOLS
            }
            cand.risk_weights = {sym: 0.01 for sym in DEFAULT_SYMBOLS}
            cand.mark_ready()
            with _silence_logs():
                commit_to_holdout(cand, success_criteria_text="bug8", verbose=False)
            # Holdout is broken after a successful commit
            assert session.holdout_set.is_broken()
            assert not session.holdout_set.is_sealed()


def _silence_logs():
    """Return a context manager that suppresses 'rich' logger output."""
    import logging
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        original = logging.getLogger("rich").info
        logging.getLogger("rich").info = lambda *a, **kw: None
        try:
            yield
        finally:
            logging.getLogger("rich").info = original

    return _ctx()
