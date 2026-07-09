#!/usr/bin/env python
"""Self-contained reproduction script for the sealedge JSS submission.

This script reproduces the **explore-phase** results (WFA + SPA + PSR) for
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

The ``fast`` variant in ``scripts/reproduce_fast.py`` reduces ``n_spa_iters``
to 500 (vs 2000 default) for faster iteration at the cost of slightly
noisier p-values.

Usage
-----
::

    python scripts/reproduce.py                 # full reproduction (~1h target)
    python scripts/reproduce.py --output-dir /tmp/jss-rep
    python scripts/reproduce.py --strategies vol_compression_v1 pullback_sniper_rsi

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
DEFAULT_N_SPA_ITERS = 2000  # paper-grade. Fast script uses 500.


def _check_data_coverage(
    strategies: list[str],
    cache_dir: str,
) -> list[str]:
    """Pre-flight check: verify cached data covers each strategy's needs.

    The framework's universe filter requires a 90-day volume lookback
    BEFORE each strategy's ``train_start``. The cached ``data_cache/``
    CSV files have specific start dates -- if those dates are too
    recent, every symbol fails the filter and the script produces
    zero trades. This function catches that case BEFORE spending
    an hour running ``run_explore``, and tells the reviewer exactly
    which data is missing and where to get it.

    Returns a list of human-readable error messages (empty if all OK).
    """
    import pandas as pd

    errors: list[str] = []
    # Load each experiment's required period.
    from quant_lib.experiments import get
    for name in strategies:
        try:
            exp = get(name)
        except KeyError:
            # Already caught by the discovery check in main(); skip.
            continue
        train_start = exp.period.train_start
        lookback_end = pd.Timestamp(train_start)
        lookback_start = lookback_end - pd.Timedelta(days=90)

        # Check each symbol's cached data.
        for sym in exp.universe.symbols:
            csv_path = (
                Path(cache_dir) / f"{sym}_1h_MASTER.csv"
            )
            if not csv_path.exists():
                errors.append(
                    f"[{name}] symbol {sym}: cached data file missing "
                    f"({csv_path}).\n"
                    f"    The script does NOT fetch from the network -- "
                    f"pre-cache manually via\n"
                    f"    ``quant_lib.tools.data.prefetch_master_csv({sym!r}, ...)``\n"
                    f"    before running this script."
                )
                continue
            try:
                df = pd.read_csv(csv_path, usecols=["time"])
            except (ValueError, KeyError) as e:
                errors.append(
                    f"[{name}] symbol {sym}: cached CSV unreadable: {e}"
                )
                continue
            df["time"] = pd.to_datetime(df["time"], errors="coerce").dropna()
            if df.empty:
                errors.append(
                    f"[{name}] symbol {sym}: cached CSV has no parseable "
                    f"timestamps"
                )
                continue
            cached_min = df["time"].min()
            # The 90-day lookback window before train_start needs at
            # least 24 hourly bars (= 1 day minimum, but realistically
            # the universe filter wants the full 90 days).
            cached_lookback_coverage = (cached_min <= lookback_start)
            if not cached_lookback_coverage:
                # Compute how many days short we are.
                days_short = (cached_min - lookback_start).days
                errors.append(
                    f"[{name}] symbol {sym}: cached data starts "
                    f"{cached_min.date()}, but the universe filter needs "
                    f"90-day volume lookback ending at {lookback_end.date()}.\n"
                    f"    Need data from {lookback_start.date()} or earlier; "
                    f"currently {days_short} days short.\n"
                    f"    Fix: change ``train_start`` in "
                    f"``quant_lib/experiments/{name}.py`` to a later date "
                    f"(recommended: at least 3 months after cached data "
                    f"starts), OR pre-cache additional data via the "
                    f"data-fetch utilities in ``quant_lib/core/_data.py``."
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
    where possible to keep all fields.
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
    (PSR, SPA p-value, n OOS trades, final equity) without opening
    the JSON.
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
        "| Strategy | Status | PSR | SPA p-value | "
        "n OOS trades | Final equity | Elapsed (s) |"
    )
    lines.append(
        "|----------|--------|-----|-------------|"
        "-------------|--------------|-------------|"
    )
    for name, result in strategies.items():
        if result["status"] != "success":
            lines.append(
                f"| `{name}` | ❌ {result['status']} | — | — | — | — "
                f"| {result.get('elapsed_seconds', '?')} |"
            )
            continue
        m = result["metrics"]
        psr = m.get("psr", "—")
        spa = m.get("spa_p_value", m.get("spa_naive_p_value", "—"))
        n_oos = m.get("n_oos_trades", m.get("n_trades", "—"))
        equity = m.get("final_equity", "—")
        elapsed = result.get("elapsed_seconds", "—")
        # Format numerics (3 d.p. floats) for readability.
        def fmt(v: Any) -> str:
            if isinstance(v, float):
                if v != v:  # NaN check
                    return "nan"
                return f"{v:.4f}"
            return str(v)
        lines.append(
            f"| `{name}` | ✅ success | {fmt(psr)} | {fmt(spa)} | "
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
        "* For fast iteration (sacrificing SPA precision), use "
        "`scripts/reproduce_fast.py` which reduces `n_spa_iters` to 500.\n"
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
