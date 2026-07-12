#!/usr/bin/env python
"""Paper-grade seal micro-demo (Claim 1) for the sealedge JSS package.

Runs a **synthetic** holdout seal lifecycle against ``quant_lib.audit.holdout``
only.  It does **not** open a registered-experiment seal, does **not** call
``run_explore`` / ``run_commit``, and does **not** touch ``data_cache/`` market
files used by the SPA sample.

Steps (asserted, then written to JSON/MD):

1. Seal with a fixed hex digest under a temp seal path.
2. ``is_sealed()`` and ``verify()`` succeed.
3. Tamper the on-disk ``data_hash`` without rewriting a valid HMAC.
4. ``verify()`` fails (tamper detected).
5. Fresh seal + matching ``commit_break`` succeeds once (``was_intact``).
6. Second ``commit_break`` fails; ``seal()`` after break raises RuntimeError.

Usage
-----
::

    python scripts/reproduce_seal_demo.py
    make reproduce-seal

Output (default ``replication/output_seal_demo/``):

* ``results.json`` -- machine-readable step log + environment metadata
* ``results.md``   -- short reviewer table

Exit code 0 only if every expected assertion holds.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_DEMO_SECRET = "sealedge-jss-seal-demo-32chars-min"
if not os.environ.get("QUANT_LIB_HMAC_SECRET"):
    os.environ["QUANT_LIB_HMAC_SECRET"] = _DEMO_SECRET

from quant_lib import __version__  # noqa: E402
from quant_lib.audit.holdout import (  # noqa: E402
    HoldoutSet,
    _reset_hmac_secret_cache,
)

_FAKE_HASH = "a" * 64
_TAMPER_HASH = "b" * 64
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "replication" / "output_seal_demo"


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _run_demo(seal_dir: Path) -> dict[str, Any]:
    """Execute the seal lifecycle; return structured results."""
    _reset_hmac_secret_cache()
    steps: list[dict[str, Any]] = []
    seal_path = seal_dir / "holdout_demo_2025-01-01_2025-06-30.json"

    # --- Steps 1–2: seal + verify ---
    hs = HoldoutSet(
        "paper_seal_demo",
        "2025-01-01",
        "2025-06-30",
        seal_path=str(seal_path),
    )
    hs.seal(data_hash=_FAKE_HASH)
    sealed_ok = hs.is_sealed()
    verify_ok = hs.verify()
    steps.append(
        {
            "id": 1,
            "name": "seal_and_verify",
            "is_sealed": sealed_ok,
            "verify": verify_ok,
            "data_hash_prefix": _FAKE_HASH[:12] + "...",
            "pass": bool(sealed_ok and verify_ok),
        }
    )
    if not (sealed_ok and verify_ok):
        return {"ok": False, "steps": steps, "error": "seal_and_verify failed"}

    # --- Steps 3–4: tamper on-disk hash (invalidates HMAC / field check) ---
    payload = json.loads(seal_path.read_text(encoding="utf-8"))
    payload["data_hash"] = _TAMPER_HASH
    seal_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    verify_after_tamper = hs.verify()
    steps.append(
        {
            "id": 2,
            "name": "tamper_detected",
            "verify_after_tamper": verify_after_tamper,
            "pass": verify_after_tamper is False,
        }
    )
    if verify_after_tamper is not False:
        return {"ok": False, "steps": steps, "error": "tamper not detected"}

    # --- Step 5: clean seal + one-shot break ---
    seal_path.unlink(missing_ok=True)
    hs3 = HoldoutSet(
        "paper_seal_demo",
        "2025-01-01",
        "2025-06-30",
        seal_path=str(seal_path),
    )
    hs3.seal(data_hash=_FAKE_HASH)
    was_intact, hash_before, hash_after = hs3.commit_break(
        _FAKE_HASH, description="paper seal micro-demo"
    )
    break_ok = (
        was_intact is True
        and hash_after == _FAKE_HASH
        and hs3.is_broken()
        and not hs3.is_sealed()
    )
    steps.append(
        {
            "id": 3,
            "name": "commit_break_once",
            "was_intact": was_intact,
            "is_broken": hs3.is_broken(),
            "is_sealed": hs3.is_sealed(),
            "hash_before_prefix": (hash_before[:12] + "...") if hash_before else "",
            "hash_after_prefix": (hash_after[:12] + "...") if hash_after else "",
            "pass": break_ok,
        }
    )
    if not break_ok:
        return {"ok": False, "steps": steps, "error": "commit_break_once failed"}

    # --- Step 6: second break fails; reseal raises ---
    was_intact_2, _, _ = hs3.commit_break(_FAKE_HASH, description="second break")
    reseal_raised = False
    try:
        hs3.seal(data_hash=_FAKE_HASH)
    except RuntimeError:
        reseal_raised = True
    oneshot_ok = (was_intact_2 is False) and reseal_raised
    steps.append(
        {
            "id": 4,
            "name": "one_shot_invariants",
            "second_break_was_intact": was_intact_2,
            "reseal_raises_runtime_error": reseal_raised,
            "pass": oneshot_ok,
        }
    )
    if not oneshot_ok:
        return {"ok": False, "steps": steps, "error": "one_shot_invariants failed"}

    return {
        "ok": True,
        "steps": steps,
        "error": None,
        "seal_path_basename": seal_path.name,
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Seal micro-demo (Claim 1)",
        "",
        "Synthetic HMAC holdout lifecycle. Does **not** break registered",
        "experiment seals or run SPA explore.",
        "",
        f"- package version: `{payload['metadata']['quant_lib_version']}`",
        f"- global ok: **{payload['ok']}**",
        f"- captured_at_utc: `{payload['metadata']['captured_at_utc']}`",
        "",
        "| Step | Name | Pass |",
        "|------|------|------|",
    ]
    for s in payload["steps"]:
        lines.append(f"| {s['id']} | `{s['name']}` | {s.get('pass')} |")
    lines.extend(
        [
            "",
            "Expected: all steps `True`. Re-run:",
            "",
            "```bash",
            "python scripts/reproduce_seal_demo.py",
            "# or: make reproduce-seal",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for results.json / results.md",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sealedge_seal_demo_") as tmp:
        demo = _run_demo(Path(tmp))

    payload: dict[str, Any] = {
        "metadata": {
            "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "quant_lib_version": __version__,
            "git_commit": _git_commit(),
            "demo_kind": "synthetic_hmac_holdout",
            "touches_registered_experiments": False,
            "touches_market_cache": False,
        },
        "ok": demo["ok"],
        "error": demo.get("error"),
        "steps": demo["steps"],
        "seal_path_basename": demo.get("seal_path_basename"),
    }

    json_path = out_dir / "results.json"
    md_path = out_dir / "results.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_md(md_path, payload)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"ok={payload['ok']}")
    if not payload["ok"]:
        print(f"error={payload['error']}", file=sys.stderr)
        for s in payload["steps"]:
            print(
                f"  step {s['id']} {s['name']}: pass={s.get('pass')}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
