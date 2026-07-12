"""
ResearchSession: sealed-holdout research container.

One session owns one holdout seal, many candidates, and the explore
phases (universe, edge, narrowing). Commit is a separate one-shot call
that re-verifies the seal hash then breaks it.

No auto-verdict: metrics and success_criteria text are for the user.

At session creation the holdout (and feature-affecting inputs) are
hashed; ``commit_to_holdout`` re-hashes and aborts on mismatch.
"""

import hashlib
import os
from copy import deepcopy
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import pandas as pd

from quant_lib.audit import Hypothesis, HoldoutSet
from quant_lib.audit.journal import ExperimentLog as _ExperimentLog
from quant_lib.research.exceptions import (
    SessionError,
    InvalidPeriod,
)
from quant_lib.research.cache import DataCache

if TYPE_CHECKING:
    from quant_lib.experiments.base import StrategyConfig
    from quant_lib.research.candidate import Candidate


# FDR alpha (Benjamini-Hochberg). Crypto-adjusted, less strict than Bonferroni.
DEFAULT_FDR_ALPHA = 0.15

# Min candidates before raising warning
MAX_CANDIDATES_WARN = 50

# Length of a hex SHA256 digest (64 chars).
_HASH_LEN = 64


# Columns hashed for holdout no-peek enforcement. We hash all OHLCV
# columns (not just close) so that tampering with high/low/open/volume
# is also detected. Strategy decisions can depend on any of these
# columns (e.g., stop-loss uses high/low; vol_pct_rank uses volume).
_HOLDOUT_HASH_COLUMNS = ("time", "open", "high", "low", "close", "volume")


def _compute_holdout_data_hash(
    holdout_data: dict[str, pd.DataFrame],
    btc_extended: Optional[pd.DataFrame] = None,
    funding_data: Optional[dict[str, pd.DataFrame]] = None,
) -> str:
    """Compute SHA256 of holdout raw OHLCV + funding per symbol.

    The hash covers ALL OHLCV columns (open, high, low, close, volume)
    so that any tampering with the underlying price/volume data is
    detected. Strategy signals (vol_pct_rank, stops, etc.) can depend
    on any of these columns; hashing only close would allow silent
    tampering of high/low/volume that could bias results.

    If ``btc_extended`` is provided, its data is also hashed. The BTC
    extended range (covering ``btc_data_start`` to ``hold_end``) is used
    to compute EMA-warmup features at commit time. Tampering with this
    pre-holdout BTC data would silently change EMA features and trade
    signals, so it must be in the seal.

    If ``funding_data`` is provided (Phase 2.3: BC break), each
    symbol's funding rate series is also hashed. Funding data affects
    trade PnL through the funding_impact_pct term in the engine.
    Tampering with funding data between session creation and commit
    would silently change trade outcomes, so it must be in the seal.

    Symbols are sorted to make ordering deterministic.

    Parameters
    ----------
    holdout_data : dict
        {sym: DataFrame with OHLCV columns}.
    btc_extended : pd.DataFrame, optional
        BTC data covering ``btc_data_start`` to ``hold_end`` (includes
        the holdout window). If None, only ``holdout_data`` is hashed.
    funding_data : dict, optional
        Phase 2.3: {sym: DataFrame with funding_rate column}.
        If None, funding is not hashed (backward compat for tests
        that don't pass funding).

    Returns
    -------
    str
        64-char hex SHA256 digest.
    """
    hasher = hashlib.sha256()
    # Hash holdout window for all symbols
    for sym in sorted(holdout_data.keys()):
        df = holdout_data[sym]
        if df is None or len(df) == 0:
            continue
        # Select only the canonical columns that exist; missing cols
        # are silently skipped (allows partial data shapes during testing).
        cols = [c for c in _HOLDOUT_HASH_COLUMNS if c in df.columns]
        if not cols:
            continue
        hasher.update(
            df[cols].to_csv(index=False).encode("utf-8")
        )
    # Hash BTC extended range (covers pre-holdout data for EMA warmup).
    # Pre-Phase-2.2 this was NOT hashed, so tampering with the BTC
    # pre-holdout history was undetectable and could bias EMA-based
    # features at commit time.
    if btc_extended is not None and len(btc_extended) > 0:
        cols = [c for c in _HOLDOUT_HASH_COLUMNS if c in btc_extended.columns]
        if cols:
            hasher.update(
                btc_extended[cols].to_csv(index=False).encode("utf-8")
            )
    # Phase 2.3: Hash funding data per symbol. This is a BC break
    # for users with existing seals (the new hash will differ from
    # the old one, so they must re-create sessions).
    if funding_data is not None:
        for sym in sorted(funding_data.keys()):
            df = funding_data[sym]
            if df is None or len(df) == 0:
                continue
            cols = [c for c in ["time", "funding_rate"] if c in df.columns]
            if not cols:
                continue
            hasher.update(
                df[cols].to_csv(index=False).encode("utf-8")
            )
    return hasher.hexdigest()


@dataclass
class SessionCommitRecord:
    """Record of a single commit (for audit trail).

    Attributes
    ----------
    candidate_name : str
        Name of the candidate that was committed.
    timestamp : str
        ISO 8601 timestamp (truncated to seconds) of when the commit
        was recorded.
    final_equity : float
        Final equity value at the end of the holdout phase.
    equity_pct : float
        Equity expressed as a percentage of initial capital.
    n_trades : int
        Number of trades executed during the holdout phase.
    psr : float
        Probabilistic Sharpe Ratio of the holdout commit.
    seal_hash : str
        SHA256 hex digest identifying the holdout seal under which
        the commit was recorded.
    success_criteria_text : str
        Free-form text describing the success criteria used to evaluate
        this commit (recorded verbatim from the hypothesis).
    """
    candidate_name: str
    timestamp: str
    final_equity: float
    equity_pct: float
    n_trades: int
    psr: float
    seal_hash: str
    success_criteria_text: str


class ResearchSession:
    """White-box research session for iterative hypothesis exploration.

    Parameters
    ----------
    training_period : tuple of str
        (start_date, end_date) for training (e.g., "2020-01-01", "2024-12-31").
    holdout_period : tuple of str
        (start_date, end_date) for holdout. Must be AFTER training_period.
        Sealed at session start (one-shot, irreversible).
    btc_data_start : str
        Extended BTC history start for EMA 4800 warmup.
    symbols : list of str
        Candidate symbols for universe construction.
    universe_kwargs : dict, optional
        Keyword args for universe selection.
    bonferroni_base : float
        Base alpha for multiple testing correction. Default 0.15 (crypto).
    fdr_alpha : float
        Alpha for FDR (Benjamini-Hochberg) across multiple commits. Default 0.15.
    cache_dir : str
        Directory for cached data. Default "data_cache".
    cache_ttl_days : int
        Cache TTL in days. Default 7.
    min_train_months : int
        Minimum training months enforced. Default 12.
    seal_dir : str, optional
        Directory where the HMAC holdout seal file is persisted.
        Default ``os.path.join(cache_dir, "holdout_seals")``. Override
        this if you want seals stored outside the cache tree (e.g. on
        a separate volume for backup isolation, or in a per-experiment
        subdirectory). The directory is created if it does not exist.
    _holdout_data : dict, optional (INTERNAL/TESTING)
        Pre-loaded holdout OHLCV data as ``{sym: DataFrame}``. When
        provided, the session skips the data-load step and uses this
        data directly (still computes hash from it). Intended for
        tests that want to bypass the network/cache layer.
        Production code should NOT pass this.
    _btc_extended : pd.DataFrame, optional (INTERNAL/TESTING)
        Pre-loaded BTC extended data covering ``btc_data_start`` to
        ``hold_end``. Used together with ``_holdout_data`` when tests
        want to verify the no-peek check for the BTC extended
        range (Phase 2.2). Ignored if ``_holdout_data`` is None.
        Production code should NOT pass this.
    _holdout_hash : str, optional (INTERNAL/TESTING)
        Pre-computed holdout hash. When provided together with
        ``_holdout_data``, the session uses this hash instead of
        computing it. Must be a 64-char hex SHA256. Tests that don't
        care about no-peek enforcement can use a deterministic fake
        (e.g., "0" * 64). Production code should NOT pass this.
    _skip_holdout_load : bool, optional (INTERNAL/TESTING)
        If True, skip the holdout data-load + hash step entirely.
        Use a deterministic fake hash ("0" * 64) and empty cached
        data. This is the most ergonomic test bypass: tests that
        don't care about the holdout seal mechanism can pass this
        flag and avoid network/cache entirely. Production code
        should NOT pass this.
    _holdout_funding : dict, optional (INTERNAL/TESTING)
        Pre-loaded holdout funding data as ``{sym: DataFrame}``.
        Used together with ``_holdout_data`` when tests want to
        control the funding content (the production path auto-
        fetches via ``self.cache.get_funding``). Ignored if
        ``_holdout_data`` is None. Production code should NOT pass
        this. Phase 2.3.
    """

    def __init__(
        self,
        training_period: tuple[str, str],
        holdout_period: tuple[str, str],
        symbols: list[str],
        btc_data_start: str = "2019-06-01",
        universe_kwargs: Optional[dict] = None,
        bonferroni_base: float = 0.15,
        fdr_alpha: float = DEFAULT_FDR_ALPHA,
        cache_dir: str = "data_cache",
        cache_ttl_days: int = 7,
        min_train_months: int = 12,
        initial_capital: float = 1000.0,
        seal_dir: Optional[str] = None,
        _holdout_data: Optional[dict[str, pd.DataFrame]] = None,
        _btc_extended: Optional[pd.DataFrame] = None,
        _holdout_hash: Optional[str] = None,
        _skip_holdout_load: bool = False,
        _holdout_funding: Optional[dict[str, pd.DataFrame]] = None,
    ):
        # Validate periods
        train_start, train_end = training_period
        hold_start, hold_end = holdout_period
        if pd.Timestamp(hold_start) <= pd.Timestamp(train_end):
            raise InvalidPeriod(
                f"Holdout start ({hold_start}) must be AFTER training end ({train_end}).",
            )

        self.training_period = training_period
        self.holdout_period = holdout_period
        self.btc_data_start = btc_data_start
        self.symbols = list(symbols)
        self.universe_kwargs = universe_kwargs or {}
        self.bonferroni_base = bonferroni_base
        self.fdr_alpha = fdr_alpha
        self.cache_dir = cache_dir
        self.cache_ttl_days = cache_ttl_days
        self.min_train_months = min_train_months
        self.initial_capital = initial_capital
        # Resolve seal directory in priority order:
        #   1. ``seal_dir`` argument (explicit override)
        #   2. ``QUANT_LIB_SEAL_DIR`` env var (for tooling/CLI config)
        #   3. ``<cache_dir>/holdout_seals`` (convention default)
        # Using a single source of truth avoids the previous
        # "data_cache/holdout_seals" hardcoded path which broke for
        # users running from a different working dir.
        if seal_dir is not None:
            self.seal_dir = seal_dir
        else:
            env_seal_dir = os.environ.get("QUANT_LIB_SEAL_DIR")
            if env_seal_dir:
                self.seal_dir = env_seal_dir
            else:
                self.seal_dir = os.path.join(cache_dir, "holdout_seals")

        # State
        self.candidates: list["Candidate"] = []
        self._commits: list[SessionCommitRecord] = []

        # Cache
        self.cache = DataCache(cache_dir=cache_dir, ttl_days=cache_ttl_days)

        # === C-2: Pre-commit holdout data hash (no-peek enforcement) ===
        # Load holdout raw OHLCV for each symbol + extended BTC. The
        # hash is computed at init and verified at commit. Any tampering
        # with the data between init and commit will be caught.
        #
        # Phase 2.2: We also preserve BTC extended data (covering
        # ``btc_data_start`` to ``hold_end``) for both:
        #   1. Hashing at init (tampering with pre-holdout BTC history
        #      is now detected)
        #   2. Reuse at commit (no re-fetch needed; the cached data is
        #      the authoritative version, ensuring hash consistency)
        if _skip_holdout_load:
            # Internal/testing bypass: skip data load, use fake hash
            # (deterministic, NOT the SHA256 of empty data).
            self._holdout_data_for_hash: dict[str, pd.DataFrame] = {}
            self._btc_extended_for_features: Optional[pd.DataFrame] = None
            self._holdout_hash: str = "0" * 64
        elif _holdout_data is not None:
            # Internal/testing path: caller pre-loaded holdout window.
            # If _btc_extended was also provided, use it; otherwise None
            # (tests that don't need BTC extended don't need it).
            # Phase 3.8 E5: deepcopy ensures full isolation from caller
            # mutation (e.g., a test that mutates a DataFrame in place
            # would otherwise see its mutation reflected in the session).
            # The perf cost is acceptable since this only happens once
            # at session init.
            self._holdout_data_for_hash = deepcopy(_holdout_data)
            self._btc_extended_for_features = (
                deepcopy(_btc_extended) if _btc_extended is not None else None
            )
        else:
            # Production path: load from cache. Returns both the holdout
            # window (per symbol) and the BTC extended range (for EMA).
            (
                self._holdout_data_for_hash,
                self._btc_extended_for_features,
            ) = self._load_holdout_ohlcv(
                list(symbols), hold_start, hold_end, btc_data_start
            )

        # Phase 2.3: Pre-load funding data so it can be hashed into the
        # seal. Funding affects trade PnL (via funding_impact_pct in
        # the engine), so tampering with funding between session
        # creation and commit would silently change trade outcomes.
        # This is a BC break for users with existing seals (the new
        # hash will differ from the old one).
        self._holdout_funding_for_hash: dict[str, pd.DataFrame] = {}
        if not _skip_holdout_load:
            # Test path: use the pre-loaded funding if provided.
            if _holdout_funding is not None:
                for sym, fund_df in _holdout_funding.items():
                    if fund_df is not None and len(fund_df) > 0:
                        self._holdout_funding_for_hash[sym] = fund_df.copy()
            else:
                # Production path: fetch from cache.
                for sym in symbols:
                    try:
                        fund_df = self.cache.get_funding(
                            sym, hold_start, hold_end
                        )
                        if fund_df is not None and len(fund_df) > 0:
                            # Defensive copy to insulate from cache mutations
                            self._holdout_funding_for_hash[sym] = fund_df.copy()
                    except Exception as e:
                        # Surface clearly; do not silently swallow
                        raise SessionError(
                            f"Failed to load holdout funding for {sym}: {e}"
                        ) from e

        # Compute or use provided hash (skip if _skip_holdout_load
        # already set a deterministic fake).
        if not _skip_holdout_load:
            if _holdout_hash is not None:
                if len(_holdout_hash) != _HASH_LEN:
                    raise ValueError(
                        f"_holdout_hash must be a {_HASH_LEN}-char hex SHA256, "
                        f"got len={len(_holdout_hash)}"
                    )
                self._holdout_hash = _holdout_hash
            else:
                self._holdout_hash = _compute_holdout_data_hash(
                    self._holdout_data_for_hash,
                    btc_extended=self._btc_extended_for_features,
                    funding_data=self._holdout_funding_for_hash,
                )

        # Seal holdout with real hash (file persistence)
        os.makedirs(self.seal_dir, exist_ok=True)
        seal_path = os.path.join(
            self.seal_dir,
            f"holdout_{hold_start}_{hold_end}.json",
        )
        self.holdout_set = HoldoutSet(
            name=f"holdout_{hold_start}_{hold_end}",
            start=hold_start,
            end=hold_end,
            seal_path=seal_path,
        )
        self.holdout_set.seal(data_hash=self._holdout_hash)

        # Journal
        self.journal = _ExperimentLog(
            hypothesis_name=f"session_{hold_start}_{hold_end}",
            initial_alpha=bonferroni_base,
            journal_path=os.path.join(
                self.seal_dir,
                f"journal_session_{hold_start}_{hold_end}.json",
            ),
        )
        self.journal.log_run(
            description=(
                f"ResearchSession created: {len(symbols)} candidate symbols, "
                f"train={train_start}->{train_end}, holdout={hold_start}->{hold_end}, "
                f"holdout_hash={self._holdout_hash[:16]}..."
            ),
            category="ablation",
        )

    def _load_holdout_ohlcv(
        self,
        symbols: list[str],
        hold_start: str,
        hold_end: str,
        btc_data_start: str,
    ) -> tuple[dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
        """Load holdout raw OHLCV for each symbol (C-2 internal helper).

        Phase 2.2: Returns BOTH the per-symbol holdout window AND the
        BTC extended range (covers ``btc_data_start`` to ``hold_end``
        for EMA warmup). Both are included in the no-peek hash.

        Each non-BTC symbol's data is sliced to the holdout period
        (>= hold_start). BTC is loaded with its full extended range
        (btc_data_start -> hold_end); both the full range and the
        holdout window are returned. The holdout window is what trade
        decisions are based on; the full range is needed for EMA
        feature warmup.

        Returns
        -------
        tuple of (holdout_data, btc_extended)
            holdout_data : dict
                {sym: DataFrame with holdout window only}.
                For BTC, this is the same as ``btc_extended`` filtered
                to ``>= hold_start``.
            btc_extended : pd.DataFrame or None
                BTC data from ``btc_data_start`` to ``hold_end`` (full
                range). None if BTC is not in ``symbols``.
        """
        out: dict[str, pd.DataFrame] = {}
        btc_extended: Optional[pd.DataFrame] = None
        for sym in symbols:
            # BTC gets the extended range for EMA warmup; other symbols
            # only need the holdout window.
            sym_start = btc_data_start if sym == "BTCUSDT" else hold_start
            try:
                df_raw = self.cache.get_klines(
                    sym, "1h", sym_start, hold_end
                )
            except Exception as e:
                # Surface clearly; do not silently swallow
                raise SessionError(
                    f"Failed to load holdout data for {sym}: {e}"
                ) from e
            if sym == "BTCUSDT":
                # Preserve full extended range (for EMA warmup + hash).
                # Make a defensive copy so the cache's internal data
                # is not aliased.
                btc_extended = df_raw.copy()
                # Holdout window for BTC
                df_filtered = df_raw[
                    df_raw["time"] >= pd.Timestamp(hold_start)
                ].copy()
            else:
                df_filtered = df_raw
            out[sym] = df_filtered
        return out, btc_extended

    # ──────────────────────────────────────────────────────────────────
    # Candidate lifecycle
    # ──────────────────────────────────────────────────────────────────

    def create_candidate(
        self,
        hypothesis: Hypothesis,
        strategy: "StrategyConfig | None" = None,
    ) -> "Candidate":
        """Create a new candidate for a hypothesis.

        Auto-logs to journal (counted as experiment) and registry.
        Validates hypothesis completeness.

        Parameters
        ----------
        hypothesis : Hypothesis
            The formal hypothesis to test.
        strategy : StrategyConfig, optional
            Per-experiment strategy/portfolio overrides. When provided,
            the candidate uses these knobs (e.g., ``pf_weight_clamp_floor``)
            for the per-fold PF-weighted risk allocation. When ``None``,
            defaults to ``StrategyConfig()`` with framework defaults.
        """
        missing = hypothesis.validate()
        if missing:
            raise SessionError(
                f"Hypothesis '{hypothesis.name}' missing: {missing}",
            )

        if len(self.candidates) >= MAX_CANDIDATES_WARN:
            import warnings
            warnings.warn(
                f"Session has {len(self.candidates)} candidates. "
                f"FDR adjustment will be very strict. Consider a new session.",
            )

        from quant_lib.research.candidate import Candidate
        from quant_lib.experiments.base import StrategyConfig

        candidate = Candidate(
            hypothesis=hypothesis,
            session=self,
            strategy=strategy if strategy is not None else StrategyConfig(),
        )
        self.candidates.append(candidate)

        # Auto-log to journal
        self.journal.log_run(
            description=f"Candidate created: {hypothesis.name} "
                        f"(strategy={hypothesis.strategy_name})",
            category="explore",
            params_snapshot={
                "strategy_type": hypothesis.strategy_type,
                "min_train_months": hypothesis.min_train_months,
            },
        )
        return candidate

    # ──────────────────────────────────────────────────────────────────
    # Multi-commit adjustment
    # ──────────────────────────────────────────────────────────────────

    @property
    def n_commits(self) -> int:
        return len(self._commits)

    @property
    def current_bonferroni_alpha(self) -> float:
        """Bonferroni-adjusted alpha for the NEXT commit.

        Computed as ``bonferroni_base / (n_commits + 1)``, i.e. 1-indexed
        (the next commit is treated as the (n+1)-th test). Use this
        when committing to decide the rejection threshold.

        Returns
        -------
        float
            The Bonferroni-adjusted alpha for the next commit, equal
            to ``bonferroni_base / max(n_commits + 1, 1)``.
        """
        return self.bonferroni_base / max(self.n_commits + 1, 1)

    def adjusted_alpha_for_commit(self, commit_idx: int) -> float:
        """Bonferroni alpha for a specific commit (1-indexed).

        Parameters
        ----------
        commit_idx : int
            1-indexed commit position used to compute
            ``bonferroni_base / max(commit_idx, 1)``.

        Returns
        -------
        float
            Bonferroni-adjusted alpha for the given commit index.
        """
        return self.bonferroni_base / max(commit_idx, 1)

    def record_commit(
        self,
        candidate: "Candidate",
        final_equity: float,
        equity_pct: float,
        n_trades: int,
        psr: float,
        seal_hash: str,
        success_criteria_text: str,
    ) -> None:
        """Record a commit (called by commit_to_holdout).

        Parameters
        ----------
        candidate : Candidate
            The committed candidate; its ``hypothesis.name`` is used
            to label the ``SessionCommitRecord``.
        final_equity : float
            Final equity value at commit time.
        equity_pct : float
            Equity as a percentage of initial capital.
        n_trades : int
            Number of trades executed during the holdout phase.
        psr : float
            Probabilistic Sharpe Ratio of the commit.
        seal_hash : str
            SHA256 hex digest identifying the holdout seal.
        success_criteria_text : str
            Free-form success criteria text, recorded verbatim on the
            ``SessionCommitRecord``.

        Returns
        -------
        None
        """
        record = SessionCommitRecord(
            candidate_name=candidate.hypothesis.name,
            timestamp=datetime.now(timezone.utc).isoformat()[:19],
            final_equity=final_equity,
            equity_pct=equity_pct,
            n_trades=n_trades,
            psr=psr,
            seal_hash=seal_hash,
            success_criteria_text=success_criteria_text,
        )
        self._commits.append(record)

    # ──────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable session state.

        Returns
        -------
        str
            Single-line ``ResearchSession(...)`` summary covering the
            training/holdout period, candidate and commit counts, the
            current Bonferroni and FDR alphas, and the holdout state
            (``SEALED``/``BROKEN``).
        """
        return (
            f"ResearchSession("
            f"train={self.training_period[0]}->{self.training_period[1]}, "
            f"holdout={self.holdout_period[0]}->{self.holdout_period[1]}, "
            f"candidates={len(self.candidates)}, "
            f"commits={self.n_commits}, "
            f"bonf_alpha={self.current_bonferroni_alpha:.4f}, "
            f"fdr_alpha={self.fdr_alpha:.4f}, "
            f"holdout={'SEALED' if self.holdout_set.is_sealed() else 'BROKEN'})"
        )

    def __repr__(self) -> str:
        return self.summary()
