# sealedge

**Objective**: a research-grade software framework for sealed-holdout
backtesting of crypto futures strategies. The framework is the
deliverable; it must stand on its own as reproducible research
software (well-tested, well-documented, citation-ready, installable).

A software paper submission is planned for the **Journal of
Statistical Software (JSS)**. If fully rejected, fallback is
**e-Informatica Software Engineering Journal**.

## Project state (as of this revision)

- Name: **sealedge** (PyPI: `sealedge`, package: `quant_lib` pending rename)
- Objective: **research software** (locked).
- Target journal: **JSS** (primary), **e-Informatica** (fallback).
- License plan: GPL-3.0-or-later for JSS; revert to MIT if fallback.
- Stage: pre-flight (editor inquiry, experiments, tool comparison).
- Affiliation for any paper draft: BINUS Online Learning,
  Bina Nusantara University, Jakarta, Indonesia.

## Core paper claims

The framework must defend these four claims in review:

1. **Sealed-holdout discipline** via HMAC-SHA256 (data hash + state
   integrity — see `quant_lib/audit/holdout.py`).
2. **PSR / Deflated-SR** for multiple-testing correction.
3. **SPA (White Reality Check)** for statistical significance — two
   coexisting nulls: legacy uniform time-anchored permutation
   (regression-tested 3-tuple contract, default) and opt-in
   Hansen-literal (Politis–Romano stationary block bootstrap,
   Hansen 2005 Eq.7 recenter + Eq.8 cross-strategy max-stat — the
   actual multiple-testing correction that the legacy path lacked).
   See `docs/methodology.md §6` for the exact-null spec, four
   paper-disclosed divergences, and three user-accepted caveats.
4. **Deterministic walk-forward analysis** with reproducible RNG.

**Reviewer operating principle**: optimize for "does this claim
hold?", not "is this code maximally clean?". Cleanup happens after
paper ships.

## Architecture snapshot

- `quant_lib/audit/` — HMAC seal, holdout discipline, journal
- `quant_lib/research/` — session, candidate (state machine), commit
- `quant_lib/core/` — engine (Numba-accelerated), WFA, risk allocation
- `quant_lib/tools/` — universe filtering, stats, features, portfolio
- `quant_lib/cli/` — Typer-based CLI (`quant_exp` command)
- `tests/conftest.py` — Shared fixtures (`_MockCache`, factories,
  autouse isolation). Source of truth for cross-test setup.
- `tests/test_invariants.py` — Property-based tests (hypothesis).
  Defends statistical claims (#2, #3).
- `tests/test_regression_b0_*.py` — Regression tests for tracked bugs.

## Review priority ranking

When you find a defect, rank it before reporting:

| Severity   | Definition                                                              | Action                  |
|------------|-------------------------------------------------------------------------|-------------------------|
| **Blocker** | Breaks a paper claim (HMAC, PSR, SPA, holdout) or causes data corruption | Fix before any ship     |
| **Major**   | Real bug, reproducible, doesn't break a paper claim                     | Fix this sprint         |
| **Minor**   | Defect under specific config/timing; hard to trigger                    | Fix when convenient     |
| **Nit**     | Style, dead code, comment rot, naming                                   | Skip unless clusters    |

**Default assumption**: if you cannot name a failure scenario with
concrete inputs → it's a Nit. Do not surface Nits unless they
cluster (>3 in same area signals a systematic issue, not noise).

## Definition of done

The framework meets the **research-software bar** when all four
conditions hold:

1. ✅ All Blocker and Major findings are resolved (or filed as
   tracked issues with explicit deferral rationale).
2. ✅ `make test` passes serially and with `-n auto` (parallel).
3. ✅ Clean-environment install + test from a fresh venv works
   end-to-end (the "reviewer test" — could a stranger install and
   run this without help?).
4. ✅ Each of the four core claims has at least one passing
   regression test that documents the claim's behavior.

When the framework hits this bar, a paper draft can be produced by
mapping each claim to a paper section. Until then, paper drafting
is premature.

**Minor and Nit findings do NOT block ship.** Track them in issues
if needed; do not keep re-reviewing the codebase for them.

## Out-of-scope (do not flag)

These are deliberate trade-offs for paper-submission scope discipline.
Do not surface findings in these zones unless the user explicitly
asks:

- **Refactoring suggestions** that don't fix a defect (e.g.,
  "could be cleaner if split into 3 functions").
- **Adding new tests** beyond what's needed for the four paper claims.
- **Performance optimization** unless the user asks.
- **Reformatting** that doesn't change behavior (`ruff format` handles
  this mechanically).
- **Documentation improvements** to README/CHANGELOG beyond noting
  the change that was just made.
- **Comment style** (grammar, tone) — only flag if the comment is
  factually wrong.
- **Backwards-compat aliases** in test fixtures — they're deliberate;
  ask before suggesting removal.
- **Test-file dead code** — only flag if it's actively misleading or
  shadows a real symbol.

## Working with this codebase

- Tests live in `tests/`. The single source of truth for cross-test
  fixtures is `tests/conftest.py`.
- Production code lives in `quant_lib/`. Public API is exposed via
  `quant_lib/__init__.py`.
- Configuration constants live in `quant_lib/core/_config.py`
  (`STATIC`, `DEFAULTS`). Tests should patch via
  `tests.conftest.patch_statics` rather than `monkeypatch.setattr`
  to preserve restoration semantics.
- Holdout periods are centralized in `tests/conftest.py` as
  `TRAIN_PERIOD`, `HOLDOUT_PERIOD`, etc. Tests should import these
  names rather than embedding date literals.
- For holdout-period override on a single test, use the
  `@pytest.mark.holdout_period_for_isolation(HOLDOUT_PERIOD_ALT)`
  marker (registered in `pyproject.toml`).

## Quick commands

```bash
make test            # serial
make test-parallel   # xdist (-n auto)
make test-cov        # coverage report
make lint            # ruff + mypy
make mutate          # mutation testing (Linux/WSL only)
make reproduce EXP=vol_compression_v1  # full pipeline
```
