#!/usr/bin/env python
"""Fast reproduction script (sub-1-hour target) for sealedge.

This is the speed-optimized variant of ``scripts/reproduce.py`` for
reviewers who need a quick sanity check rather than paper-grade
metrics.

Speed vs accuracy trade-off
----------------------------
* ``n_spa_iters=500`` (vs 2000 default) -- SPA p-values are noisier
  (Monte Carlo standard error ~sqrt(p*(1-p)/n) ~0.022 at p=0.5),
  but order-of-magnitude conclusions ("significant or not") are
  stable. Critical-region estimates (p < 0.05) may shift between
  0.03 and 0.07 between runs -- that's the noise floor.
* No holdout commit (``run_explore`` only). The holdout seal is
  preserved for the full reproduction script.
* All other settings match the full script: same RNG seed, same
  strategies, same output format.

Output: ``<output-dir>/results.json`` + ``results.md`` (same format
as ``reproduce.py``, so a reviewer can diff the two).

Exit code: 0 on success, non-zero on first failure.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# Same HMAC + sys.path bootstrap as ``reproduce.py``.
if not __import__("os").environ.get("QUANT_LIB_HMAC_SECRET"):
    __import__("os").environ["QUANT_LIB_HMAC_SECRET"] = (
        "sealedge-jss-reproduction-32chars-min"
    )

# Import the sibling script directly so we don't depend on a package
# layout (scripts/ is not a Python package). ``importlib.util`` gives us
# the same module namespace as ``import reproduce`` would if scripts/
# were on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
_spec = importlib.util.spec_from_file_location(
    "reproduce", _REPO_ROOT / "scripts" / "reproduce.py"
)
reproduce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reproduce)
_reproduce = reproduce

# Override defaults for the fast variant.
DEFAULT_N_SPA_ITERS = 500  # vs 2000 in reproduce.py
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "replication" / "output_fast"


def main(argv: list[str] | None = None) -> int:
    """Delegate to ``reproduce.main()`` with fast defaults applied."""
    parser = argparse.ArgumentParser(
        description="Fast reproduction (~sub-1-hour) for sealedge JSS.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=_reproduce.DEFAULT_STRATEGIES,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--n-spa-iters",
        type=int,
        default=DEFAULT_N_SPA_ITERS,
    )
    parser.add_argument(
        "--cache-dir",
        default=str(_REPO_ROOT / "data_cache"),
    )
    args = parser.parse_args(argv)

    # Delegate to the canonical implementation. We synthesise the
    # ``argv`` it expects.
    delegate_argv = [
        "--strategies", *args.strategies,
        "--output-dir", str(args.output_dir),
        "--n-spa-iters", str(args.n_spa_iters),
        "--cache-dir", args.cache_dir,
    ]
    return _reproduce.main(delegate_argv)


if __name__ == "__main__":
    raise SystemExit(main())
