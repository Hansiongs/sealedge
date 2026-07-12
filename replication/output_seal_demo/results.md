# Seal micro-demo (Claim 1)

Synthetic HMAC holdout lifecycle. Does **not** break registered
experiment seals or run SPA explore.

- package version: `0.5.1`
- global ok: **True**
- captured_at_utc: `2026-07-11T18:41:19.054882+00:00`

| Step | Name | Pass |
|------|------|------|
| 1 | `seal_and_verify` | True |
| 2 | `tamper_detected` | True |
| 3 | `commit_break_once` | True |
| 4 | `one_shot_invariants` | True |

Expected: all steps `True`. Re-run:

```bash
python scripts/reproduce_seal_demo.py
# or: make reproduce-seal
```
