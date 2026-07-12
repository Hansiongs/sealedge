"""Optional charts for research output (matplotlib + seaborn).

Each plot either saves a PNG (``output_path``) or returns a base64 PNG
data URI for HTML. Uses the Agg backend for headless/CI.

Optional deps: ``matplotlib>=3.5``, ``seaborn>=0.12``. Install with::

    pip install quant_lib[viz]

If the dependencies are missing, importing plotting.py raises
``ImportError`` with a clear install hint. Callers should guard
imports accordingly.
"""
from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Any, Optional

# IMPORTANT: matplotlib backend MUST be set before pyplot is imported.
# "Agg" is non-interactive and works in headless / CI environments.
try:
    import matplotlib
    matplotlib.use("Agg")  # noqa: E402
    import matplotlib.pyplot as plt  # noqa: E402
    import numpy as np  # noqa: E402
    import pandas as pd  # noqa: E402
    import seaborn as sns  # noqa: E402
    _MATPLOTLIB_AVAILABLE = True
except ImportError as _exc:  # pragma: no cover - exercised only without deps
    _MATPLOTLIB_AVAILABLE = False
    _IMPORT_ERROR = _exc

    def _missing_dep(*args, **kwargs):  # pragma: no cover
        raise ImportError(
            "quant_lib.research.plotting requires matplotlib and seaborn. "
            "Install with: pip install quant_lib[viz]"
        ) from _IMPORT_ERROR

    # Stub names so static analysis doesn't break; runtime calls raise.
    plt = _missing_dep  # type: ignore
    np = _missing_dep  # type: ignore
    pd = _missing_dep  # type: ignore
    sns = _missing_dep  # type: ignore


log = logging.getLogger(__name__)


# Apply seaborn theme once at module import. whitegrid + deep palette
# is the framework's default for all research charts.
if _MATPLOTLIB_AVAILABLE:
    sns.set_theme(style="whitegrid", palette="deep")


def _fig_to_base64(fig: Any) -> str:
    """Convert a matplotlib figure to a base64 data URI for HTML embedding.

    The figure is closed after encoding to avoid memory leaks.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to encode.

    Returns
    -------
    str
        Data URI string of the form ``data:image/png;base64,<...>``.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _normalize_daily_equity(daily_equity: Any) -> "pd.Series":
    """Convert various daily_equity representations to a sorted Series.

    Accepts dict, pandas Series, or DataFrame with a 'time' column.
    Empty or all-NaN input is allowed; returned Series may be empty.
    """
    if isinstance(daily_equity, pd.Series):
        return daily_equity.sort_index()
    if isinstance(daily_equity, pd.DataFrame):
        if "time" in daily_equity.columns:
            value_col = "equity" if "equity" in daily_equity.columns else daily_equity.columns[0]
            if value_col == "time":
                value_col = daily_equity.columns[1] if len(daily_equity.columns) > 1 else value_col
            s = pd.Series(
                daily_equity[value_col].values,
                index=pd.to_datetime(daily_equity["time"]),
            )
            return s.sort_index()
        return daily_equity.iloc[:, 0].sort_index()
    # dict-like
    s = pd.Series(daily_equity)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def plot_equity_curve(
    daily_equity: Any,
    initial_capital: float,
    output_path: Optional[str] = None,
) -> str:
    """Plot cumulative equity curve with initial capital reference line.

    Green/red shading indicates periods above/below initial capital.
    Most useful as a sanity check and headline figure for paper submission.

    Parameters
    ----------
    daily_equity : dict, pandas.Series, or DataFrame
        Time-indexed equity values (in USD). Dict keys are parsed as
        datetimes; DataFrames must contain a 'time' column.
    initial_capital : float
        Reference initial capital for the horizontal line and shading.
    output_path : str, optional
        If provided, save PNG to this path and return the path.
        Otherwise, return a base64 data URI for HTML embedding.

    Returns
    -------
    str
        Output file path (if ``output_path`` was given) or base64 data URI.

    Raises
    ------
    ImportError
        If matplotlib/seaborn are not installed.
    """
    if not _MATPLOTLIB_AVAILABLE:
        raise _IMPORT_ERROR
    eq = _normalize_daily_equity(daily_equity)
    if eq.empty:
        # Render an empty figure with a notice rather than crashing.
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No equity data available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Equity Curve")
        if output_path:
            fig.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            return output_path
        return _fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eq.index, eq.values, linewidth=1.5, color="steelblue", label="Equity")
    ax.axhline(
        initial_capital, color="gray", linestyle="--", alpha=0.6,
        label="Initial capital",
    )
    above = eq.values >= initial_capital
    ax.fill_between(
        eq.index, initial_capital, eq.values,
        where=above, alpha=0.15, color="green", interpolate=True,
    )
    ax.fill_between(
        eq.index, initial_capital, eq.values,
        where=~above, alpha=0.15, color="red", interpolate=True,
    )
    final_eq = float(eq.iloc[-1])
    final_pct = (final_eq - initial_capital) / initial_capital * 100 if initial_capital else 0
    ax.set_title(f"Equity Curve (final: ${final_eq:,.2f}, {final_pct:+.2f}%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (USD)")
    ax.legend(loc="best")
    fig.autofmt_xdate()

    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return output_path
    return _fig_to_base64(fig)


def plot_drawdown_underwater(
    daily_equity: Any,
    output_path: Optional[str] = None,
) -> str:
    """Plot underwater (drawdown-from-peak) chart.

    Pairs naturally with ``plot_equity_curve`` as a complementary
    headline figure. Drawdown is shown as a percentage of peak equity
    (negative values, inverted y-axis).

    Parameters
    ----------
    daily_equity : dict, pandas.Series, or DataFrame
        Time-indexed equity values.
    output_path : str, optional
        If provided, save PNG to this path. Otherwise return base64 data URI.

    Returns
    -------
    str
        Output file path or base64 data URI.

    Raises
    ------
    ImportError
        If matplotlib/seaborn are not installed.
    """
    if not _MATPLOTLIB_AVAILABLE:
        raise _IMPORT_ERROR
    eq = _normalize_daily_equity(daily_equity)
    if eq.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No equity data available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Drawdown (Underwater)")
        if output_path:
            fig.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            return output_path
        return _fig_to_base64(fig)

    running_max = eq.cummax()
    drawdown_pct = (eq - running_max) / running_max * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(
        eq.index, 0, drawdown_pct,
        color="indianred", alpha=0.7, interpolate=True,
    )
    ax.plot(eq.index, drawdown_pct, color="darkred", linewidth=1)
    max_dd = float(drawdown_pct.min())
    ax.set_title(f"Drawdown (Underwater) -- max: {max_dd:.2f}%")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.invert_yaxis()
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    fig.autofmt_xdate()

    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return output_path
    return _fig_to_base64(fig)


def _to_finite_array(values: Any) -> "np.ndarray":
    """Convert input to a 1D numpy array of finite floats.

    Drops NaN/Inf values. Returns an empty array if no finite values.
    Accepts list, tuple, numpy array, or pandas Series.
    """
    arr = np.asarray(list(values) if isinstance(values, (list, tuple)) else values,
                     dtype=np.float64)
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    return finite


def plot_trade_distribution(
    r_vals: Any,
    output_path: Optional[str] = None,
    bins: int = 40,
) -> str:
    """Plot R-multiple distribution (histogram + KDE + mean/median markers).

    Shows the tail-risk profile of executed trades. Most useful after
    Phase 2 (edge testing) or after a commit, when the R-multiple
    distribution is the primary edge evidence.

    KDE is suppressed when ``n < 5`` or ``std == 0`` (KDE requires
    variance). The function still renders a histogram in those cases.

    Parameters
    ----------
    r_vals : list, tuple, numpy array, or pandas Series
        Per-trade R-multiples (net R after costs).
    output_path : str, optional
        If provided, save PNG to this path. Otherwise return base64 data URI.
    bins : int
        Number of histogram bins. Default 40.

    Returns
    -------
    str
        Output file path or base64 data URI.

    Raises
    ------
    ImportError
        If matplotlib/seaborn are not installed.
    """
    if not _MATPLOTLIB_AVAILABLE:
        raise _IMPORT_ERROR
    finite = _to_finite_array(r_vals)
    if finite.size == 0:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, "No trade data available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Trade R-Multiple Distribution")
        if output_path:
            fig.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            return output_path
        return _fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    mean_r = float(finite.mean())
    median_r = float(np.median(finite))
    std_r = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    skew = float(((finite - mean_r) ** 3).mean() / (std_r ** 3)) if std_r > 0 else 0.0

    n = int(finite.size)
    ax.hist(finite, bins=min(bins, max(10, n // 2 or 10)),
            color="steelblue", alpha=0.7, edgecolor="white", density=True)

    if n >= 5 and std_r > 0:
        try:
            sns.kdeplot(finite, ax=ax, color="navy", linewidth=1.5, warn_singular=False)
        except Exception:  # pragma: no cover - seaborn internal
            pass

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Zero")
    ax.axvline(mean_r, color="red", linestyle="-", linewidth=1.2,
               label=f"Mean ({mean_r:+.3f})")
    ax.axvline(median_r, color="green", linestyle="-", linewidth=1.2,
               label=f"Median ({median_r:+.3f})")

    pos_count = int((finite > 0).sum())
    neg_count = int((finite < 0).sum())
    ax.set_title(
        f"Trade R-Multiple Distribution "
        f"(n={n}, mean={mean_r:+.3f}, std={std_r:.3f}, skew={skew:+.2f})"
    )
    ax.set_xlabel("R-multiple (net)")
    ax.set_ylabel("Density")
    ax.legend(loc="best", fontsize=9)
    info = f"Positive: {pos_count} | Negative: {neg_count} | Win rate: {pos_count / n * 100:.1f}%"
    ax.text(0.02, 0.95, info, transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.7))

    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return output_path
    return _fig_to_base64(fig)


def plot_spa_null(
    random_equities: Any,
    observed_equity: float,
    p_value: float,
    output_path: Optional[str] = None,
) -> str:
    """Plot SPA null distribution with observed marker and p-value annotation.

    The Superior Predictive Ability (SPA) test produces a null
    distribution of portfolio final equities under circular permutation
    of trade entries. The observed equity is compared against this
    null; a low p-value means the observed is significantly better
    than the null.

    Parameters
    ----------
    random_equities : array-like
        Final equity values from SPA permutation iterations.
    observed_equity : float
        The actual (non-permuted) final equity from the backtest.
    p_value : float
        SPA p-value. Shown in the chart title. If NaN, the title says so.
    output_path : str, optional
        If provided, save PNG to this path. Otherwise return base64 data URI.

    Returns
    -------
    str
        Output file path or base64 data URI.

    Raises
    ------
    ImportError
        If matplotlib/seaborn are not installed.
    """
    if not _MATPLOTLIB_AVAILABLE:
        raise _IMPORT_ERROR
    finite = _to_finite_array(random_equities)
    fig, ax = plt.subplots(figsize=(9, 5))

    if finite.size == 0 or not np.isfinite(observed_equity):
        ax.text(0.5, 0.5,
                "SPA null distribution unavailable\n(insufficient data or NaN p-value)",
                ha="center", va="center", transform=ax.transAxes)
        p_str = "NaN" if (p_value is None or not np.isfinite(p_value)) else f"{p_value:.4f}"
        ax.set_title(f"SPA Test -- p-value: {p_str}")
        if output_path:
            fig.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            return output_path
        return _fig_to_base64(fig)

    n_iters = int(finite.size)
    mean_null = float(finite.mean())
    ax.hist(finite, bins=min(50, max(10, n_iters // 20)),
            color="lightgray", alpha=0.85, edgecolor="white",
            label=f"Null (n={n_iters})")

    if mean_null > 0:
        ax.axvline(mean_null, color="gray", linestyle="--", linewidth=1.0,
                   label=f"Null mean ({mean_null:,.2f})")

    is_significant = np.isfinite(p_value) and p_value < 0.05
    obs_color = "darkgreen" if is_significant else "firebrick"
    obs_label = "Observed (significant)" if is_significant else "Observed"
    ax.axvline(observed_equity, color=obs_color, linestyle="-", linewidth=2.0,
               label=f"{obs_label} ({observed_equity:,.2f})")

    p_str = f"{p_value:.4f}" if np.isfinite(p_value) else "NaN"
    ax.set_title(f"SPA Test -- p-value: {p_str}  (observed vs {n_iters} null permutations)")
    ax.set_xlabel("Final equity (USD)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="best", fontsize=9)

    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return output_path
    return _fig_to_base64(fig)


def _series_from_equity_input(equity: Any) -> "pd.Series":
    """Convert a per-symbol equity representation to a sorted Series."""
    return _normalize_daily_equity(equity)


def plot_per_symbol_equity(
    per_symbol_equity: dict,
    output_path: Optional[str] = None,
) -> str:
    """Plot per-symbol cumulative equity curves on a single axis.

    Each symbol is one line. Reveals whether the portfolio edge is
    driven by one or two symbols (concentrated) or spread across the
    universe (broad).

    Parameters
    ----------
    per_symbol_equity : dict
        ``{symbol: daily_equity}`` where each value can be a dict,
        Series, or DataFrame (same format as ``plot_equity_curve``).
    output_path : str, optional
        If provided, save PNG to this path. Otherwise return base64 data URI.

    Returns
    -------
    str
        Output file path or base64 data URI.

    Raises
    ------
    ImportError
        If matplotlib/seaborn are not installed.
    """
    if not _MATPLOTLIB_AVAILABLE:
        raise _IMPORT_ERROR
    fig, ax = plt.subplots(figsize=(10, 5))

    if not per_symbol_equity:
        ax.text(0.5, 0.5, "No per-symbol equity data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Per-Symbol Cumulative Equity")
        if output_path:
            fig.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            return output_path
        return _fig_to_base64(fig)

    palette = sns.color_palette("deep", n_colors=len(per_symbol_equity))
    final_values = {}
    for (sym, equity), color in zip(sorted(per_symbol_equity.items()), palette):
        s = _series_from_equity_input(equity)
        if s.empty:
            continue
        ax.plot(s.index, s.values, linewidth=1.2, color=color, label=sym)
        final_values[sym] = float(s.iloc[-1])

    if final_values:
        best = max(final_values, key=final_values.get)
        worst = min(final_values, key=final_values.get)
        ax.set_title(
            f"Per-Symbol Cumulative Equity "
            f"(best: {best} ${final_values[best]:,.0f}, "
            f"worst: {worst} ${final_values[worst]:,.0f})"
        )
    else:
        ax.set_title("Per-Symbol Cumulative Equity (no data)")

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL (USD)")
    if len(per_symbol_equity) <= 12:
        ax.legend(loc="best", fontsize=9)
    else:
        ax.legend(loc="best", fontsize=8, ncol=2)
    fig.autofmt_xdate()

    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return output_path
    return _fig_to_base64(fig)


def _extract_best_values(fold_params: dict) -> dict:
    """Extract per-symbol best_value sequences from WFA fold params.

    Accepts a ``{sym: [fold_dict, ...]}`` mapping (the structure held in
    ``Candidate.fold_params``). Each fold_dict should expose a numeric
    ``best_value`` (typically the PSR-weighted objective from Optuna).
    """
    out: dict[str, list[float]] = {}
    for sym, folds in fold_params.items():
        vals = []
        for fp in folds or []:
            bv = fp.get("best_value") if isinstance(fp, dict) else None
            if bv is None or not np.isfinite(bv):
                continue
            vals.append(float(bv))
        if vals:
            out[sym] = vals
    return out


def plot_wfa_progression(
    fold_params: Any,
    output_path: Optional[str] = None,
) -> str:
    """Plot WFA fold progression: best_value per fold per symbol.

    Each symbol is a line; the x-axis is fold index, y-axis is the
    best Optuna objective (PSR-weighted). Reveals whether optimization
    is converging (flat/oscillating) or diverging (drift up or down).

    Parameters
    ----------
    fold_params : dict
        ``{symbol: [fold_dict, ...]}`` (the structure held in
        ``Candidate.fold_params``). Each fold_dict may have keys
        ``best_value`` (numeric), ``fold`` (int), ``oos_start`` /
        ``oos_end`` (datetimes) -- only ``best_value`` is required.
    output_path : str, optional
        If provided, save PNG to this path. Otherwise return base64 data URI.

    Returns
    -------
    str
        Output file path or base64 data URI.

    Raises
    ------
    ImportError
        If matplotlib/seaborn are not installed.
    """
    if not _MATPLOTLIB_AVAILABLE:
        raise _IMPORT_ERROR
    fig, ax = plt.subplots(figsize=(10, 5))

    per_sym = _extract_best_values(fold_params)
    if not per_sym:
        ax.text(0.5, 0.5, "No WFA fold data available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("WFA Fold Progression (best_value)")
        if output_path:
            fig.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            return output_path
        return _fig_to_base64(fig)

    palette = sns.color_palette("deep", n_colors=len(per_sym))
    for (sym, vals), color in zip(sorted(per_sym.items()), palette):
        x = list(range(1, len(vals) + 1))
        ax.plot(x, vals, marker="o", linewidth=1.2, color=color,
                markersize=5, label=sym)

    ax.set_title(f"WFA Fold Progression (best_value across {len(per_sym)} symbols)")
    ax.set_xlabel("Fold index")
    ax.set_ylabel("Best value (PSR-weighted objective)")
    if len(per_sym) <= 12:
        ax.legend(loc="best", fontsize=9)
    else:
        ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return output_path
    return _fig_to_base64(fig)


__all__ = [
    "plot_equity_curve",
    "plot_drawdown_underwater",
    "plot_trade_distribution",
    "plot_spa_null",
    "plot_per_symbol_equity",
    "plot_wfa_progression",
]
