#!/usr/bin/env python
"""Self-contained reproduction script for the sealedge JSS submission.

This script reproduces the **explore-phase** results (WFA + SPA) for
all three strategies registered in the sealedge framework:

1. ``vol_compression_v1``  (volatility-compression momentum breakout)
2. ``pullback_sniper_rsi`` (RSI-extreme mean-reversion)
3. ``funding_rate_carry``  (perp funding-rate carry)

JSS submission context
----------------------
Per JSS guidelines ("rough guideline, one hour on a regular PC"), this
script is designed to be:

* **Self-contained** -- one Python entry-point, no manual intervention
* **Deterministic**  -- RNG seeded explicitly via ``GLOBAL_SEED = 42`` from
  ``quant_lib.core._config``; per-symbol offsets are stable across runs
* **Cross-checkable** -- emits ``results.json`` (machine-readable metrics)
  and ``results.md`` (human-readable summary) for reviewer verification
* **Non-destructive** -- runs ``run_explore`` only (Phase 0-3 of the
  pipeline). Holdout seal is NOT broken. Reviewer can re-run on their
  own machine and confirm identical numbers.

Paper-grade default is ``n_spa_iters=2000`` (no reduced-iter path).
For a shorter smoke test, run a single strategy first; SPA precision is
unchanged.

Usage
-----
::

    python scripts/reproduce.py                 # full pipeline (~53 min measured; 1-2 h cold)
    python scripts/reproduce.py --output-dir /tmp/jss-rep
    python scripts/reproduce.py --strategies vol_compression_v1  # single-strategy smoke

Output
------
* ``<output-dir>/results.json`` -- per-strategy metrics, platform metadata,
  git commit hash, seed values
* ``<output-dir>/results.md``   -- human-readable summary table

Exit code: 0 on success (all strategies ran), non-zero on first failure.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

# HMAC seal secret. ``run_explore`` writes a fresh holdout seal per
# strategy; it requires ``QUANT_LIB_HMAC_SECRET`` to be set to >= 32 chars.
# For reproducibility the exact value does not matter (each run creates
# its own throwaway seal), but the env var MUST be present or
# ``run_explore`` raises ``RuntimeError`` at the first ``HoldoutSet``
# construction. Set a deterministic default before any ``quant_lib``
# import if the reviewer hasn't provided one.
if not os.environ.get("QUANT_LIB_HMAC_SECRET"):
    os.environ["QUANT_LIB_HMAC_SECRET"] = "sealedge-jss-reproduction-32chars-min"

# Make ``quant_lib`` importable when running from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from quant_lib import __version__, run_explore  # noqa: E402
from quant_lib.core._config import GLOBAL_SEED  # noqa: E402
from quant_lib.experiments import (  # noqa: E402
    all_experiments,
    discover_experiments,
)

# Strategies covered by the paper. Order matches the paper's discussion
# order: momentum first, mean-reversion second, fundamental/carry third.
DEFAULT_STRATEGIES: list[str] = [
    "vol_compression_v1",
    "pullback_sniper_rsi",
    "funding_rate_carry",
]

DEFAULT_OUTPUT_DIR = _REPO_ROOT / "replication" / "output"
DEFAULT_N_SPA_ITERS = 2000  # paper-grade only; no reduced-iter script


def _check_data_coverage(
    strategies: list[str],
    cache_dir: str,
) -> list[str]:
    """Pre-flight check: verify cached data covers each strategy's needs.

    Universe filters need (i) a volume lookback before ``train_start``
    (90 days) and (ii) enough history for ``min_age_days`` (paper: 180).
    Funding-rate experiments also need ``{sym}_FUNDING_MASTER.csv``.
    Fail loud before ``run_explore``; do not fetch network data here.

    Returns a list of human-readable error messages (empty if all OK).
    """
    import pandas as pd

    errors: list[str] = []
    from quant_lib.experiments import get

    def _csv_time_min(path: Path):
        try:
            df = pd.read_csv(path, usecols=["time"])
        except (ValueError, KeyError, OSError) as e:
            errors.append(f"cached CSV unreadable ({path}): {e}")
            return None
        ts = pd.to_datetime(df["time"], errors="coerce").dropna()
        if ts.empty:
            errors.append(f"cached CSV has no parseable timestamps ({path})")
            return None
        return ts.min()

    for name in strategies:
        try:
            exp = get(name)
        except KeyError:
            continue
        train_start = exp.period.train_start
        lookback_end = pd.Timestamp(train_start)
        min_age = int(getattr(exp.universe, "min_age_days", 0) or 0)
        hist_days = max(90, min_age)
        lookback_start = lookback_end - pd.Timedelta(days=hist_days)
        needs_funding = "funding" in name.lower()

        for sym in exp.universe.symbols:
            csv_path = Path(cache_dir) / f"{sym}_1h_MASTER.csv"
            if not csv_path.exists():
                errors.append(
                    f"[{name}] symbol {sym}: 1h master missing ({csv_path}).\n"
                    f"    Script does NOT fetch. Pre-cache with:\n"
                    f"    from quant_lib.tools.data import fetch_klines\n"
                    f"    fetch_klines({sym!r}, '1h', '2020-01-01', '2025-12-31')"
                )
                continue
            cached_min = _csv_time_min(csv_path)
            if cached_min is None:
                continue
            if getattr(cached_min, "tzinfo", None) is not None:
                cached_min = cached_min.tz_localize(None)
            lb = lookback_start
            if getattr(lb, "tzinfo", None) is not None:
                lb = lb.tz_localize(None)
            if cached_min > lb:
                days_short = (cached_min - lb).days
                errors.append(
                    f"[{name}] symbol {sym}: 1h data starts {cached_min.date()}, "
                    f"need history from {lb.date()} or earlier "
                    f"(train_start={lookback_end.date()}, "
                    f"hist_days={hist_days} for volume lookback / min_age).\n"
                    f"    Currently ~{days_short} days short. Pre-cache earlier "
                    f"bars via quant_lib.tools.data.fetch_klines; do not change "
                    f"train_start if you want paper-grade numbers."
                )

            if needs_funding:
                fund_path = Path(cache_dir) / f"{sym}_FUNDING_MASTER.csv"
                if not fund_path.exists():
                    errors.append(
                        f"[{name}] symbol {sym}: funding master missing "
                        f"({fund_path}).\n"
                        f"    Pre-cache with:\n"
                        f"    from quant_lib.tools.data import fetch_funding\n"
                        f"    fetch_funding({sym!r}, '2021-01-01', '2025-12-31')"
                    )
                    continue
                fmin = _csv_time_min(fund_path)
                if fmin is None:
                    continue
                if getattr(fmin, "tzinfo", None) is not None:
                    fmin = fmin.tz_localize(None)
                if fmin > lookback_end:
                    errors.append(
                        f"[{name}] symbol {sym}: funding data starts "
                        f"{fmin.date()}, after train_start {lookback_end.date()}."
                    )
    return errors


def _capture_metadata() -> dict[str, Any]:
    """Capture platform / package metadata for reviewer cross-check.

    These fields are deterministic for a given machine + repo state;
    they help reviewers confirm the script ran in the expected
    environment.
    """
    # Git commit hash (best-effort; may be empty in uncommitted state).
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    # Key dependency versions -- reviewer can compare against their
    # own ``pip freeze`` to verify reproducibility.
    deps: dict[str, str] = {}
    for pkg in ("numpy", "pandas", "numba", "scipy", "optuna"):
        try:
            version = __import__(pkg).__version__
            deps[pkg] = version
        except (ImportError, AttributeError):
            deps[pkg] = "not installed"

    return {
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "quant_lib_version": __version__,
        "global_seed": GLOBAL_SEED,
        "git_commit": commit,
        "dependency_versions": deps,
    }


def _explore_result_to_dict(result: Any) -> dict[str, Any]:
    """Convert an ExploreResult to a JSON-serialisable dict.

    ExploreResult is a dataclass with both attribute and dict-style access
    (backward-compat). We normalise to plain dict via ``dataclasses.asdict``
    where possible to keep all fields. ``edge_metrics`` sub-dict carries
    ``spa_p_value``, ``spa_naive_p_value``, ``hansen_fallback``, etc.; we
    promote ``hansen_fallback`` and the two SPA fields to the top level so
    reviewers can grep ``results.json`` for ``"hansen_fallback"`` and see
    the path taken by each strategy without walking nested structures.
    """
    import dataclasses
    if dataclasses.is_dataclass(result):
        # ``dataclasses.asdict`` is typed for instances, not types.
        # ``type: ignore[arg-type]`` because mypy sees the union and
        # can't narrow without a runtime isinstance check (we already
        # called is_dataclass above, which is sufficient).
        d = dataclasses.asdict(result)  # type: ignore[arg-type]
    else:
        # Legacy dict-style support.
        d = dict(result)
    # Filter out non-serialisable values (e.g., numpy arrays) to avoid
    # ``TypeError: Object of type ndarray is not JSON serialisable``.
    out: dict[str, Any] = {}
    for k, v in d.items():
        try:
            json.dumps(v, default=str)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = repr(v)
    # Promote Hansen diagnostics from edge_metrics to top level.
    em = out.get("edge_metrics")
    if isinstance(em, dict):
        for key in ("spa_p_value", "spa_naive_p_value", "hansen_fallback"):
            if key in em and key not in out:
                out[key] = em[key]
    return out


def _run_strategy(
    name: str,
    n_spa_iters: int,
    cache_dir: str,
) -> dict[str, Any]:
    """Run ``run_explore`` for one strategy and capture the result.

    Catches exceptions so that one failing strategy does not abort the
    entire reproduction script (Phase 2 requirement: each strategy
    must produce an output file for reviewer cross-check).
    """
    t_start = dt.datetime.now(dt.timezone.utc)
    try:
        result = run_explore(
            experiment_name=name,
            cache_dir=cache_dir,
            n_spa_iters=n_spa_iters,
        )
        elapsed = (dt.datetime.now(dt.timezone.utc) - t_start).total_seconds()
        return {
            "status": "success",
            "elapsed_seconds": round(elapsed, 2),
            "metrics": _explore_result_to_dict(result),
        }
    except Exception as exc:  # noqa: BLE001 (intentional: catch-all for review)
        elapsed = (dt.datetime.now(dt.timezone.utc) - t_start).total_seconds()
        return {
            "status": "failed",
            "elapsed_seconds": round(elapsed, 2),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def _write_markdown(
    output: dict[str, Any],
    out_path: Path,
) -> None:
    """Render a human-readable Markdown summary for reviewer cross-check.

    The Markdown shows the same numbers as ``results.json`` but in a
    table format that lets a reviewer skim the headline metrics
    (SPA p-value, n OOS trades, final equity) without opening the JSON.
    Explore does not emit holdout PSR; that metric is on the commit path.
    """
    meta = output["metadata"]
    strategies = output["strategies"]

    lines: list[str] = []
    lines.append("# sealedge reproduction -- results summary\n")
    lines.append(f"Captured: `{meta['captured_at_utc']}`  ")
    lines.append(f"Git commit: `{meta['git_commit']}`  ")
    lines.append(f"Global seed: **{meta['global_seed']}**  ")
    lines.append(f"quant_lib version: `{meta['quant_lib_version']}`  ")
    lines.append(
        f"Python: `{meta['python_version']}` on `{meta['system']}` "
        f"({meta['machine']})\n"
    )
    lines.append("## Strategy results\n")
    lines.append(
        "| Strategy | Status | SPA p-value | "
        "n OOS trades | Final equity | Elapsed (s) |"
    )
    lines.append(
        "|----------|--------|-------------|"
        "-------------|--------------|-------------|"
    )
    for name, result in strategies.items():
        if result["status"] != "success":
            lines.append(
                f"| `{name}` | ❌ {result['status']} |, |, |, "
                f"| {result.get('elapsed_seconds', '?')} |"
            )
            continue
        m = result["metrics"]
        spa = m.get("spa_p_value", m.get("spa_naive_p_value", ", "))
        n_oos = m.get("n_oos_trades", m.get("n_trades", ", "))
        equity = m.get("final_equity", ", ")
        elapsed = result.get("elapsed_seconds", ", ")
        # Format numerics (3 d.p. floats) for readability.
        def fmt(v: Any) -> str:
            if isinstance(v, float):
                if v != v:  # NaN check
                    return "nan"
                return f"{v:.4f}"
            return str(v)
        lines.append(
            f"| `{name}` | ✅ success | {fmt(spa)} | "
            f"{n_oos} | {fmt(equity)} | {elapsed} |"
        )
    lines.append("")
    lines.append("## Reviewer notes\n")
    lines.append(
        "* This script runs **explore** phase only. The holdout seal "
        "is NOT broken. To verify holdout performance, use "
        "`python -m quant_lib run_commit <strategy>` separately.\n"
    )
    lines.append(
        "* All numbers in `results.json` are deterministic given the "
        f"global seed ({meta['global_seed']}) and dependency versions.\n"
    )
    lines.append(
        "* Paper-grade default is `n_spa_iters=2000`. For a shorter smoke "
        "test with the same SPA precision, run a single strategy: "
        "`python scripts/reproduce.py --strategies vol_compression_v1` "
        "(or `make reproduce-one EXP=vol_compression_v1`).\n"
    )

    out_path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce sealedge explore-phase results for JSS.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_STRATEGIES,
        help=(
            "Strategies to run. Default: "
            f"{' '.join(DEFAULT_STRATEGIES)}. Use a subset for smoke test."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--n-spa-iters",
        type=int,
        default=DEFAULT_N_SPA_ITERS,
        help=f"SPA permutation iterations (default: {DEFAULT_N_SPA_ITERS})",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(_REPO_ROOT / "data_cache"),
        help="Data cache directory (default: ./data_cache)",
    )
    args = parser.parse_args(argv)

    # Ensure all experiments are registered before we try to run them.
    discover_experiments()
    available = {e.name for e in all_experiments()}
    missing = [s for s in args.strategies if s not in available]
    if missing:
        print(
            f"ERROR: strategies not registered: {missing}. "
            f"Available: {sorted(available)}",
            file=sys.stderr,
        )
        return 1

    # Pre-flight: verify cached data covers each strategy's needs.
    # If the cached data starts AFTER the date where the universe
    # filter's 90-day lookback window starts, every symbol will fail
    # the filter and ``run_explore`` raises ``CandidateError``. We
    # catch that here (BEFORE spending an hour running) and emit a
    # clear, actionable error so the reviewer knows exactly what to fix.
    coverage_errors = _check_data_coverage(args.strategies, args.cache_dir)
    if coverage_errors:
        print(
            "ERROR: cached data does not satisfy the universe-filter\n"
            "requirements for one or more strategies. The script refuses\n"
            "to run because every strategy would produce zero trades.\n",
            file=sys.stderr,
        )
        for err in coverage_errors:
            print(f"  - {err}\n", file=sys.stderr)
        return 1

    # Prepare output directory.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    md_path = args.output_dir / "results.md"

    # Run all strategies.
    output: dict[str, Any] = {
        "metadata": _capture_metadata(),
        "config": {
            "strategies": args.strategies,
            "n_spa_iters": args.n_spa_iters,
            "cache_dir": args.cache_dir,
        },
        "strategies": {},
    }

    print(
        f"==> Reproducing {len(args.strategies)} strategies with "
        f"n_spa_iters={args.n_spa_iters}, seed={GLOBAL_SEED}..."
    )
    for name in args.strategies:
        print(f"  -> {name}...")
        result = _run_strategy(name, args.n_spa_iters, args.cache_dir)
        output["strategies"][name] = result
        if result["status"] == "success":
            print(f"     ✅ {result['elapsed_seconds']}s")
        else:
            print(
                f"     ❌ {result['error_type']}: {result['error_message']}",
                file=sys.stderr,
            )

    # Write outputs.
    json_path.write_text(json.dumps(output, indent=2, default=str))
    _write_markdown(output, md_path)
    print("\n==> Results written to:")
    print(f"     {json_path}")
    print(f"     {md_path}")

    # Exit non-zero if any strategy failed (still emits output files
    # so reviewer can inspect what went wrong).
    if any(s["status"] != "success" for s in output["strategies"].values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
