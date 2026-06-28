"""HTML report builders for `quant_exp` runs.

Builds the structured list of ``(heading, content)`` sections that
``OutputManager.save_html_report`` consumes. Keeping the build logic
in a dedicated module (rather than inline in the CLI) makes it
unit-testable without subprocess calls.

Both ``build_explore_report`` and ``build_commit_report`` accept
an optional ``chart_provider`` callable that maps chart names to
base64 data URIs. The CLI injects the real charting functions;
tests inject pre-built stubs.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence


ChartProvider = Callable[[str], Optional[str]]


def _safe_pct(num: float, denom: float) -> str:
    """Format ``num / denom`` as a percentage string.

    Returns ``"N/A"`` if ``denom`` is zero. The result is a string
    formatted as ``"+12.34%"`` (signed) or ``"N/A"``.
    """
    if not denom:
        return "N/A"
    return f"{(num / denom) * 100:+.2f}%"


def _fmt_usd(v: float) -> str:
    """Format a USD value with thousands separator and 2 decimals."""
    return f"${v:,.2f}"


def build_explore_report(
    candidate: Any,
    session: Any,
    chart_provider: ChartProvider,
) -> list:
    """Build a list of report sections for an explore run.

    The ``candidate`` and ``session`` objects are duck-typed: we only
    access attributes that ``Candidate`` and ``ResearchSession``
    expose (``daily_equity``, ``executed_trades``, ``spa_p_value``,
    etc.). Tests can pass lightweight stubs.

    Parameters
    ----------
    candidate : Candidate-like
        Source of edge metrics and equity data.
    session : ResearchSession-like
        Source of journal / FDR context.
    chart_provider : callable
        ``chart_provider(name) -> base64_data_uri_or_None``. Receives
        a chart name and returns either a data URI string or ``None``
        when the chart is unavailable (e.g., ``--no-plots``).

    Returns
    -------
    list of (heading, content) tuples
        Ready to pass to ``OutputManager.save_html_report``.
    """
    sections: list = []

    # 1. Hypothesis
    h = candidate.hypothesis
    sections.append((
        "Hypothesis",
        [
            ("Name", h.name),
            ("Strategy", h.strategy_name),
            ("Mechanism", h.mechanism),
            ("Boundary", h.boundary_conditions),
            ("Success criteria", h.success_criteria),
            ("Entry logic", h.entry_logic),
            ("Exit logic", h.exit_logic),
            ("Min train months", str(h.min_train_months)),
        ],
    ))

    # 2. Universe + period
    train_s, train_e = session.training_period
    hold_s, hold_e = session.holdout_period
    sections.append((
        "Period & Universe",
        [
            ("Training", f"{train_s} -> {train_e}"),
            ("Holdout (sealed)", f"{hold_s} -> {hold_e}"),
            ("Symbols (candidates)", str(len(session.symbols))),
            ("Symbols (eligible)", f"{len(candidate.eligible_symbols)}/{len(session.symbols)}"),
            ("Symbols (narrowed)", f"{len(candidate.narrowed_symbols)}/{len(candidate.eligible_symbols)}"),
            ("Cache hits", str(candidate.cache_hits)),
            ("Cache misses", str(candidate.cache_misses)),
        ],
    ))

    # 3. Portfolio simulation
    n_oos = candidate.n_oos_trades
    n_exec = candidate.n_executed
    n_rej = candidate.n_rejected
    rej = candidate.reject_reasons or {}
    sections.append((
        "Portfolio Simulation",
        [
            ("Initial capital", _fmt_usd(session.initial_capital)),
            ("Final equity", _fmt_usd(candidate.final_equity)),
            ("Equity change", _safe_pct(
                candidate.final_equity - session.initial_capital,
                session.initial_capital,
            )),
            ("OOS trades", str(n_oos)),
            ("Executed", str(n_exec)),
            ("Rejected", str(n_rej)),
            ("  CB cooldown", str(rej.get("cb_cooldown", 0))),
            ("  Position limit", str(rej.get("position_limit", 0))),
            ("  Margin insufficient", str(rej.get("margin_insufficient", 0))),
        ],
    ))

    # 4. SPA test
    sections.append((
        "SPA Test",
        [
            ("p-value", f"{candidate.spa_p_value:.4f}"),
        ],
    ))

    # 5. Charts: equity curve + drawdown (Phase 1 charts)
    eq_chart = chart_provider("equity_curve")
    if eq_chart:
        sections.append(("Equity Curve", ("chart", eq_chart)))
    dd_chart = chart_provider("drawdown_underwater")
    if dd_chart:
        sections.append(("Drawdown (Underwater)", ("chart", dd_chart)))

    # 6. Phase 2 charts: trade distribution, per-symbol, WFA progression
    r_vals = [t.get("r_net", 0.0) for t in candidate.executed_trades]
    if r_vals:
        td_chart = chart_provider("trade_distribution")
        if td_chart:
            sections.append(("Trade R-Multiple Distribution", ("chart", td_chart)))

    # Per-symbol equity
    per_sym_chart = chart_provider("per_symbol_equity")
    if per_sym_chart:
        sections.append(("Per-Symbol Equity", ("chart", per_sym_chart)))

    # WFA progression
    wfa_chart = chart_provider("wfa_progression")
    if wfa_chart:
        sections.append(("WFA Fold Progression", ("chart", wfa_chart)))

    # 7. FDR / Bonferroni context
    j = session.journal
    sections.append((
        "Journal & FDR Context",
        [
            ("Hypothesis name", j.hypothesis_name),
            ("Entries (total)", str(len(j.entries))),
            ("Experiments (counts toward FDR)", str(j.n_experiments)),
            ("Bugfixes (not counted)", str(j.n_bugfixes)),
            ("Ablations", str(j.n_ablations)),
            ("Base alpha", f"{j.initial_alpha:.4f}"),
            ("Adjusted alpha", f"{j.adjusted_alpha():.4f}"),
            ("Holdout status",
             "SEALED" if session.holdout_set.is_sealed() else "BROKEN"),
        ],
    ))

    return sections


def build_commit_report(
    result: Any,
    session: Any,
    chart_provider: ChartProvider,
) -> list:
    """Build a list of report sections for a commit run.

    Parameters
    ----------
    result : CommitResult-like
        Source of holdout metrics, trade stats, seal hashes.
    session : ResearchSession-like
        Source of journal / period context.
    chart_provider : callable
        ``chart_provider(name) -> base64_data_uri_or_None``.

    Returns
    -------
    list of (heading, content) tuples
    """
    sections: list = []

    # 1. Pre-commit verification
    sections.append((
        "Pre-Commit Verification",
        [
            ("Candidate", result.candidate_name),
            ("Commit index", str(result.commit_idx)),
            ("Holdout period", f"{result.holdout_period[0]} -> {result.holdout_period[1]}"),
            ("Initial capital", _fmt_usd(result.initial_capital)),
            ("Bonferroni alpha", f"{result.bonferroni_alpha:.4f}"),
            ("FDR alpha", f"{result.fdr_alpha:.4f}"),
            ("Seal status", "BROKEN" if result.seal_broken else "SEALED"),
            ("Timestamp", result.timestamp),
        ],
    ))

    if result.success_criteria_text:
        sections.append((
            "Success Criteria (user-defined)",
            result.success_criteria_text,
        ))

    # 2. Holdout simulation
    rej = result.reject_breakdown or {}
    rej_rows = [["Reason", "Count"]]
    for k, v in rej.items():
        rej_rows.append([k, str(v)])
    if len(rej_rows) == 1:
        rej_rows.append(["(no rejections)", "0"])
    sections.append((
        "Holdout Simulation",
        [
            ("Initial capital", _fmt_usd(result.initial_capital)),
            ("Final equity", _fmt_usd(result.final_equity)),
            ("Raw trades", str(result.n_raw_trades)),
            ("Executed", str(result.n_executed_trades)),
            ("Rejected", str(result.n_rejected)),
        ],
    ))
    sections.append(("Rejection Breakdown", ("table", rej_rows)))

    # 3. Per-trade stats
    sections.append((
        "Per-Trade Stats",
        [
            ("N trades", str(result.n_trades)),
            ("Win rate", f"{result.win_rate:.1f}%"),
            ("Avg R", f"{result.avg_r:+.3f}"),
            ("Median R", f"{result.median_r:+.3f}"),
            ("Std R", f"{result.std_r:.3f}"),
            ("Best trade", f"{result.best_r:+.2f} R"),
            ("Worst trade", f"{result.worst_r:+.2f} R"),
            ("Profit factor", f"{result.profit_factor:.2f}"),
            ("Avg bars held", f"{result.avg_bars_held:.1f}"),
        ],
    ))

    # 4. By-symbol stats
    if result.by_symbol_stats:
        sym_rows = [["Symbol", "N", "WR", "Avg R", "PF", "Total R"]]
        for sym, stats in sorted(result.by_symbol_stats.items()):
            sym_rows.append([
                sym,
                str(stats["n_trades"]),
                f"{stats['win_rate']:.0f}%",
                f"{stats['avg_r']:+.3f}",
                f"{stats['profit_factor']:.2f}",
                f"{stats['total_r']:+.1f}",
            ])
        sections.append(("By Symbol", ("table", sym_rows)))

    # 5. Equity curve stats
    sections.append((
        "Equity Curve Stats",
        [
            ("Initial", _fmt_usd(result.initial_capital)),
            ("Final", _fmt_usd(result.final_equity)),
            ("Equity %", f"{result.equity_pct:+.2f}%"),
            ("CAGR (annualized)", f"{result.cagr_pct:+.2f}%"),
            ("Max DD", f"{result.max_dd_pct:.2f}%"),
            ("Sharpe (R)", f"{result.sharpe_r:.3f}"),
        ],
    ))

    # 6. PSR + ESS
    sections.append((
        "PSR + ESS (Holdout)",
        [
            ("N trades", str(result.n_trades)),
            ("SR (R-multiple)", f"{result.sharpe_r:.4f}"),
            ("Skewness", f"{result.skew:.4f}"),
            ("Kurtosis", f"{result.kurtosis:.4f}"),
            ("ESS", f"{result.ess:.1f}"),
            ("PSR", f"{result.psr:.4f}"),
        ],
    ))

    # 7. Risk metrics
    sections.append((
        "Risk Metrics (Trend Alignment)",
        [
            ("With-trend trades (1.5x)", str(result.with_trend_trades)),
            ("With-trend total R", f"{result.with_trend_r_total:+.1f}"),
            ("Counter-trend trades (0.5x)", str(result.counter_trend_trades)),
            ("Counter-trend total R", f"{result.counter_trend_r_total:+.1f}"),
        ],
    ))

    # 8. Seal & audit
    sections.append((
        "Seal & Audit",
        [
            ("Seal hash (before)", (result.seal_hash_before or "")[:64]),
            ("Seal hash (after)", (result.seal_hash_after or "")[:64]),
            ("Seal status", "BROKEN" if result.seal_broken else "SEALED"),
        ],
    ))

    # 9. Charts (commit has daily_equity from session)
    eq_chart = chart_provider("equity_curve")
    if eq_chart:
        sections.append(("Equity Curve (Holdout)", ("chart", eq_chart)))
    dd_chart = chart_provider("drawdown_underwater")
    if dd_chart:
        sections.append(("Drawdown (Underwater)", ("chart", dd_chart)))
    td_chart = chart_provider("trade_distribution")
    if td_chart:
        sections.append(("Trade R-Multiple Distribution", ("chart", td_chart)))

    # 10. Final summary
    sections.append((
        "Final Summary (informational only -- no verdict)",
        [
            ("Hypothesis", result.candidate_name),
            ("Commit #", str(result.commit_idx)),
            ("Bonf alpha", f"{result.bonferroni_alpha:.4f}"),
            ("FDR alpha", f"{result.fdr_alpha:.4f}"),
            ("Holdout equity", _fmt_usd(result.final_equity)),
            ("Equity %", f"{result.equity_pct:+.2f}%"),
            ("PSR", f"{result.psr:.4f}"),
            ("Max DD", f"{result.max_dd_pct:.2f}%"),
            ("Profit Factor", f"{result.profit_factor:.2f}"),
            ("Win Rate", f"{result.win_rate:.1f}%"),
            ("N trades", str(result.n_trades)),
        ],
    ))

    return sections


__all__ = [
    "build_explore_report",
    "build_commit_report",
    "ChartProvider",
]
