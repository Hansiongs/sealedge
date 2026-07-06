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
- Stage: pre-flight (OJS submission preparation, experiments, tool comparison).
- **Pre-submission inquiry is NOT endorsed by JSS.** `editor@jstatsoft.org` exists for resource-limited external file cases, not pre-inquiry. Use the OJS step-by-step submission guide instead.
- **JSS one-shot rule**: rejected manuscripts cannot be resubmitted. Plan must be solid before submit.
- **Revision deadlines**: 6 months to return revisions, 1 year to bring conditionally accepted paper to final form — else considered withdrawn.
- **JSS frozen publication**: once published, the JSS copy of source code + article is immutable. Post-publication maintenance (bug fixes, new features) must live outside JSS via GitHub/PyPI. The submitted version is a stable snapshot.
- Affiliation for any paper draft: BINUS Online Learning,
  Bina Nusantara University, Jakarta, Indonesia.
- **Submission URL**: https://www.jstatsoft.org/submission (OJS, wajib login). 5 steps: Start → Upload Submission → Enter Metadata → Confirmation → Next Steps. 5 mandatory checklist items: PDF in JSS style, source code, replication materials, GPL-2/3/compatible license, privacy policy. Plus CCAL copyright acknowledgment.

## JSS submission enablers (pre-requisites to actually submit)

These are NOT code quality tasks; they're readiness gates tied to the
JSS mission and authors page. They must be done before Fase 4 (OJS
submission). See `roadmap_paper.md` for the full phased plan.

- **License migration**: MIT → GPL-3.0-or-later (currently MIT in `LICENSE` and `pyproject.toml`; both must change). Revert to MIT only if JSS rejects and we fall back to e-Informatica.
- **PyPI publication**: `sealedge` (rename from `quant_lib`) must be on PyPI, not just GitHub. Currently `pyproject.toml` name is `quant_lib` — rename required.
- **Formatted help files (Python)**: JSS requires "formatted help files" for environments with library systems. Python equivalent: NumPy/Google-style docstrings + Sphinx autodoc (or mkdocs). Not optional. Existing `mkdocs.yml` is a starting point but needs API reference completeness.
- **Repo size audit**: total repo + replication bundle must be < 50 MB (JSS upload limit). Dataset fixtures (e.g., `tests/fixtures/btcusdt_1h_2024_jan.csv`) and built site output (`site/`) count. Externalize large data if needed.
- **Replication script**: one standalone, commented script that reproduces all manuscript results. Target "rough guideline, one hour on a regular PC". If it runs longer, supply a separate "fast" script that reproduces similar results in reasonable time.
- **Output file**: results from running the replication script, for cross-check by reviewers. Equivalent of R's `knitr::spin` "code.html" — for Python use log file + `pip freeze` + Python version + platform info.
- **RNG seed**: explicitly initialized in the replication script.
- **Platform dependencies**: documented in submission.

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

**Per-claim software-innovation test (JSS mission gate)**: JSS does
NOT publish methodological innovations in computational statistics
— it publishes software contributions. Before submission, each of
the four claims must be defensible as "substantial software
innovation wrapping an established method", not novel statistics.

- Claim 1 (HMAC sealed-holdout): software engineering wrapping of
  cryptographic primitive + standard holdout methodology → safe.
- Claim 2 (PSR/Deflated-SR): implementation of established method
  (Bailey & Lopez de Prado 2014) → safe.
- Claim 3 (SPA + Phipson & Smyth 2010): implementation of
  established method → safe.
- Claim 4 (Deterministic WFA): **needs extra scrutiny**. Must be
  framed as an engineering choice (deterministic computation of an
  established WFA algorithm), not a novel statistical method.

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
