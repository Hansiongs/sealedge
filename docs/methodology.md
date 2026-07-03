# Methodology

This document is the methodology section for any paper that uses
`quant_lib`. It documents the statistical methods, design choices,
and justifications so that reviewers can audit the framework.

**Source code references** are in the form `path/to/file.py:line` so
that any claim can be traced to a specific line of code.

## 1. Holdout Seal (C-2 fix)

The framework enforces a **cryptographically-sealed holdout** to
prevent data snooping. The mechanism is implemented in
`quant_lib/audit/holdout.py:76` and `quant_lib/research/session.py:188`.

### Mechanism

1. **At session creation** (`ResearchSession.__init__`):
   - Load raw OHLCV (all columns: open, high, low, close, volume)
     for each symbol in the holdout period (`session.py:202`)
   - Load BTC extended data (covering `btc_data_start` to `hold_end`)
     for EMA warmup at commit time (`session.py:251-292`)
   - Compute SHA256 of both the per-symbol holdout window AND the
     BTC extended range (`session.py:54-92`)
   - Seal the `HoldoutSet` with this hash (`session.py:248`)

2. **At commit** (`commit_to_holdout`):
   - Re-compute the SHA256 from the cached data (including BTC
     extended) (`commit.py:144-160`)
   - If hash mismatch → raise `SealVerificationFailed` (data
     tampered) (`commit.py:161-166`)
   - Use the cached BTC extended data for EMA features (no
     re-fetch, ensuring hash consistency) (`commit.py:170-187`)
   - Break the seal (irreversible; can only be used once)
     (`commit.py:356`)

### What is sealed

As of 0.2.3, the seal hash covers:
- **All OHLCV columns** (open, high, low, close, volume) for every
  symbol in the holdout window. Pre-0.2.3 only `(time, close)` were
  hashed, allowing silent tampering of high/low/volume that could
  bias strategies using those columns (e.g., SL uses high/low;
  vol_pct_rank uses volume).
- **BTC extended range** (from `btc_data_start` to `hold_end`).
  This range is used at commit time to compute EMA-warmup features
  (`btc_ema_4800` has a 69-day half-life, so it needs ~200 days of
  pre-holdout history to be ~95% converged). Pre-0.2.3 the BTC
  extended range was NOT hashed, so tampering with pre-holdout BTC
  data would silently change EMA features and trade signals.

### Why this matters

Without a sealed holdout, an analyst can:
- See the holdout data
- Tune the strategy based on what "works" on the holdout
- Report inflated results

The sealed holdout guarantees that the holdout data and all
features-affecting inputs are **never modified** between sealing
and breaking, and that any modification is detected.

### References

- `audit/holdout.py` — `HoldoutSet`, `HoldoutSeal` classes
- `research/session.py:54-92` — `_compute_holdout_data_hash` (SHA256, all OHLCV + BTC extended)
- `research/session.py:251-292` — `_load_holdout_ohlcv` (preserves BTC extended)
- `research/commit.py:144-166` — seal verification at commit time

## 2. Purge Days

`wfa_purge_days` (`core/_config.py:73`) removes a gap between
training and OOS to prevent parameter contamination at the
boundary.

### Why it's needed

In live trading, there is no boundary between "IS" and "OOS" — the
strategy uses all available history continuously. In backtest with
WFA, there is an artificial seam at the IS-OOS boundary.

Features with long lookback windows (e.g., `vol_pct_rank` uses a
720-bar rolling window = 30 days at 1H, defined in
`core/_features.py:96`) cause **feature overlap** at this boundary:

```
Day 365 (LAST IS):  vol_pct_rank = data[335..365]   (all IS)
Day 366 (FIRST OOS): vol_pct_rank = data[336..366]   (365 IS + 1 OOS)
```

Without purge, the last IS bar's features and the first OOS bar's
features are nearly identical. Optuna can exploit this by selecting
parameters that "fit" the boundary, and the same params will appear
to "fit" the OOS start (artificially inflating results).

### Implementation

Adaptive purge (`core/_wfa.py:228`):
- IS ≤ 15 months → 90 days purge
- IS ≤ 30 months → 60 days purge
- IS > 30 months → 30 days purge (minimum, covers `vol_pct_rank`)

**Rationale:** Makin kecil IS, makin besar efek contamination. The
adaptive scheme balances safety vs. data loss.

### Known limitation

`btc_ema_4800` (span 4800, ~200 days) is not fully converged after
90-day purge (~87% real-data weight, `core/_config.py:23-25`).
`macro_trend` is binary (1/-1) and tolerates residual contamination
well, so this is a documented limitation rather than a bug.

## 3. Stress Multiplier

`stress_test_multiplier` (default 2.0×, `core/_config.py:84`)
scales slippage noise to model "what if actual slippage is worse
than assumed?"

### How it works

- `random_stress` ~ Uniform[1.0, stress_mult) per trade
  (`core/_engine.py:237-239`)
- `exit_slip` = `base_exit_slip` × `random_stress` × `weekend_penalty`
  (`core/_engine.py:242`)

**Effective multiplier range:**
- Base × 1.0× (best case) = 1.0× base
- Base × 2.0× (worst case) × 1.0 (weekday) = 2.0× base
- Base × 2.0× (worst case) × 2.0 (weekend) = 4.0× base (worst case)

### Why 2.0× (not 2.5×)

Previous default was 2.5× (`core/_config.py:84` in older code).
After audit, 2.0× is preferred:
- Liquid crypto pairs (BTC, ETH): 2.0× is very defensive
- Weekend penalty (2.0×) is applied separately
- Combined worst case 4.0× (vs. 5.0× before) is more realistic
- The user can override via `StrategyConfig.stress_test_multiplier`

## 4. Best-Params Selection (Q1, 2025-06-25)

At commit time, the framework picks the **best params per symbol
across all WFA folds** (highest PSR), rather than "frozen params
from the last fold."

### Rationale

This is consistent with **live trading workflow**. In live trading,
the trader runs Optuna on all available history and uses the best
params. The previous "frozen from last fold" approach was defensible
as "mimic live: optuna on most recent data" but was inconsistent
with the user's stated live workflow.

### Implementation

`quant_lib/research/best_params.py:pick_best_params_per_symbol`:
- For each symbol, find the fold with the highest `best_value`
  (PSR × trade_weight from Optuna)
- Use that fold's params, with strategy-specific safe defaults
  backfilling any missing keys

```python
# Pseudocode
for sym, folds in all_fold_params.items():
    best_fold = max(folds, key=lambda f: f.get("best_value", -inf))
    frozen[sym] = _extract_params_from_fold(best_fold, defaults)
```

### Anti-overfit stack (no additional stability needed)

The framework already has **4 layers of anti-overfit**:
1. **PSR** (Probabilistic Sharpe Ratio) — accounts for skew/kurtosis,
   not raw SR (`core/_testing.py:18-40`)
2. **ESS** (Effective Sample Size) — handles autocorrelation
   (`core/_testing.py:191-202`)
3. **FDR** (Benjamini-Hochberg) — per-symbol correction
   (`core/_testing.py:43-88`)
4. **L2 regularization** — pulls params to center
   (`core/_wfa.py:208-220`)

Adding stability-weighting (e.g., "mean PSR across folds") would be
**double-jealous** and over-engineered. The user explicitly
rejected this design.

### Backward compat

`compute_frozen_params_best_last` is preserved as a deprecation
shim in `core/_wfa.py:583`. Behavior unchanged (last fold). New
code should use `pick_best_params_per_symbol`.

## 5. Risk Allocation (PF decay + clamp + rescale)

Beyond strategy edge, `quant_lib` applies **adaptive risk
allocation** across symbols as part of the trading system. This is
intentional: crypto markets are 24/7, highly volatile, with frequent
gap events and funding costs — risk management is as critical as
entry-timing edge.

### Mechanism

Implemented in `core/_risk_allocation.py`:

1. **Decay-weighted PF** per symbol from past folds
   (`core/_risk_allocation.py:_compute_decay_weighted_pnl_loss`):
   ```
   decay_weight = 0.5 ^ (n_folds_back / halflife_folds)
   weighted_pnl[sym] = Σ(r_net * decay_weight) for r_net > 0
   weighted_loss[sym] = Σ(|r_net| * decay_weight) for r_net < 0
   ```

2. **Clamp** to [floor, ceiling] per symbol
   (`core/_risk_allocation.py:_compute_clamped_factor`):
   ```
   factor[sym] = clip(weighted_pnl / weighted_loss, floor, ceiling)
   ```

3. **Rescale** to preserve total risk
   (`core/_risk_allocation.py:_rescale_factors_to_total`):
   ```
   pre_rescale[sym] = factor[sym] * baseline_per_symbol
   rescale = target_total / sum(pre_rescale)
   final[sym] = pre_rescale[sym] * rescale
   ```

### Parameters (in `StrategyConfig`)

- `pf_decay_halflife_folds` (default 2 = 6 months)
- `pf_weight_clamp_floor` (default 0.5)
- `pf_weight_clamp_ceiling` (default 1.5)
- `pf_min_trades_for_weight` (default 10)

### Why not "pure" OOS?

Risk allocation uses **past fold performance** to size future
positions. This is **deliberate** and mirrors live trading behavior.
The alternative (frozen weights from training) is available by
setting `pf_weight_clamp_floor = pf_weight_clamp_ceiling = 1.0`.

### Cold-start handling

The first fold has no past folds → all factors are 1.0 (neutral).
This is a critical property: new symbols start with equal weight,
and the system adapts over time.

## 6. Statistical Tests

### PSR (Probabilistic Sharpe Ratio)

`core/_testing.py:17-145`. Adjusts the Sharpe ratio for skewness
and kurtosis using the Bailey & Lopez de Prado (2012) formula:

```
PSR = Φ((SR - SR_benchmark) / σ_SR)
σ_SR = sqrt(Var_correction / (n_eff - 1))
Var_correction = 1 - skew·SR + ((kurt_excess + 2)/4) · SR²
```

Where `kurt_excess` is **excess kurtosis** (kurtosis - 3, 0 for
normal data). This is the convention used in Bailey's PSR paper.
Note: Bailey's formula uses REGULAR kurtosis (γ₄ = excess + 3),
so the coefficient is `(γ₄ - 1)/4 = (excess + 2)/4`. The code in
`core/_wfa.py` and `core/_testing.py` uses this conversion (Sprint 1
fix to documentation; the code was correct in v0.3.1+).

#### Conventions (as of 0.2.3)

- **Kurtosis convention**: `fisher=True` (excess kurtosis). The
  pre-0.2.3 code used `fisher=False` (regular kurtosis, 3 for
  normal), which overestimated variance by `+3/4 * SR²` offset and
  understated PSR.
- **Weighted PSR**: when `trade_weights` is provided, the function
  computes **weighted** mean, weighted variance, and weighted SR
  (matching `core/_wfa.py:179-200` inline formula). The Kish ESS
  is used as `n_eff` in the variance denominator.
- **Asymptotic regime**: Bailey's formula is an asymptotic expansion
  valid for moderate SR (|SR| < ~2 for near-normal data). For higher
  SR, the variance correction can go negative; the function clips
  it to 1e-8 and returns a high PSR (close to 1.0) rather than NaN.
- **Annualize flag**: multiplies SR and benchmark by `sqrt(365.25)`.
  Note: PSR is NOT scale-invariant under annualization in Bailey's
  formula (the constant "1" in Var_correction does not scale), so
  PSR values will differ between `annualize=True` and `annualize=False`.
  The flag only affects the SR/benchmark scaling, not the PSR
  invariance.
- **Edge cases**: returns NaN for insufficient data (< 10 returns),
  zero/negative variance, ESS < 2 in weighted mode, or extreme SR
  (> 1e6, usually from near-constant input).

Reference: Bailey & Lopez de Prado (2012), "The Sharpe Ratio
Efficient Frontier"; Bailey & Lopez de Prado (2014), "The
Deflated Sharpe Ratio."

### SPA (Superior Predictive Ability)

`core/_spa.py:20-265`. Tests whether the observed strategy edge
is genuine or random, using **time-anchored circular permutation**
of observed trades.

> **Note on null distribution**: this is **not** a proper Hansen
> (2005) SPA stationary-bootstrap null. The null here is a uniform
> time-anchored permutation of the observed trades, which preserves
> cross-asset co-occurrence structure (all trades share a single
> random anchor offset per iteration, so relative timing between
> assets is unchanged). The implementation explicitly disclaims
> Hansen (2005) conformance in `core/_spa.py:297-303`.

- Bootstrap the null distribution by permuting trade entry times
- Compare observed equity against the null
- Compute Phipson & Smyth (2010) corrected p-value (the standard
  add-one correction for permutation tests):
  ```
  p = (n_exceed + 1) / (n_iters + 1)
  ```

References: Hansen (2005), "A Test for Superior Predictive Ability"
(provides the test-statistic framework); Phipson & Smyth (2010),
"Permutation P-values Should Never Be Zero" (provides the add-one
correction applied here). Note: a prior reference list entry cited
"Davé & Seal (2008)" for this correction; that paper does not
address the add-one — the correct citation is Phipson & Smyth (2010).

### FDR (Benjamini-Hochberg)

`core/_testing.py:43-88`. Controls False Discovery Rate for
multi-symbol testing.

```
adjusted[i] = min(1.0, p[i] * n / rank[i])  # for sorted p-values
adjusted = min_accumulate(adjusted[::-1])[::-1]  # enforce monotonicity
```

Reference: Benjamini & Hochberg (1995).

### Bonferroni

1-indexed: `α_adjusted = α_base / (n_commits + 1)`
(`audit/journal.py:146-169`). Use for the next-commit threshold.

## 7. Trend Alignment Risk Multiplier

`core/_config.py:73-75` (also in `StrategyConfig`):
- `trend_aligned_risk_mult = 1.5` (with-trend entry)
- `trend_counter_risk_mult = 0.5` (counter-trend entry)

`core/_engine.py:307-310, 346-349`: Scales position size by 1.5×
when entry is with-trend (long in bull, short in bear) and 0.5×
when counter-trend.

## 8. Cost Model

`core/_engine.py:213-265`. Per-trade cost components:
- `fee_taker × 2` (entry + exit, default 0.05% each = 0.10% total)
- `current_entry_slip` (per entry, scaled by ATR%)
- `exit_slip` (per exit, scaled by ATR% × random_stress × weekend_penalty)
- `funding_impact` (accumulated funding rate during trade)

Total cost is clamped at 5.0 R (`core/_engine.py:254`).

## 9. Engine Implementation Notes

`core/_engine.py:fast_trade_loop` is a **Numba @njit** compiled
function. Numba is opaque to coverage tools — line coverage is
not measurable. The tests verify behavior, not line coverage.

Two strategy types supported via `strategy_type` int (0=vol_compression,
1=pullback_sniper) in `core/_engine.py:21-22`.

## 10. References

- Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe
  Ratio: Correcting for Selection Bias, Backtest Overfitting, and
  Non-Normality." *Journal of Portfolio Management* 40(5).
- Hansen, P. R. (2005). "A Test for Superior Predictive Ability."
  *Journal of Business & Economic Statistics* 23(4).
- Phipson, B. & Smyth, G. K. (2010). "Permutation P-values Should
  Never Be Zero: Calculating Exact P-values When Permutations Are
  Randomly Drawn." *Statistical Applications in Genetics and
  Molecular Biology* 9(1), Article 39.
  (Note: the prior entry for "Davé & Seal (2008)" has been retired;
  it did not address the add-one correction used here.)
- Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False
  Discovery Rate: A Practical and Powerful Approach to Multiple
  Testing." *Journal of the Royal Statistical Society, Series B*
  57(1).
- López de Prado, M. (2018). *Advances in Financial Machine
  Learning*. Wiley.
- Optuna: Akiba, T. et al. (2019). "Optuna: A Next-generation
  Hyperparameter Optimization Framework." *KDD*.

## 11. Experiment Reproducibility

Same config + same seed → same output. Verified by
`tests/test_reproducibility.py`:

- `pick_best_params_per_symbol` (pure function)
- `simulate_full_portfolio` (pure function)
- `apply_pf_weighted_risk_allocation` (pure function, no RNG)
- `run_bootstrap` (uses `GLOBAL_SEED + 12345` internally)
- `portfolio_spa` (uses `np.random.default_rng(seed)`)
- `discover_experiments` (idempotent)

All 10 reproducibility tests pass. The framework is deterministic
given the same configuration.

## 12. Versioning

This methodology applies to **`quant_lib` v0.5.1** (released
2026-07-01; matches `pyproject.toml:7`). Major methodology changes
will be documented in `CHANGELOG.md` with version bumps.

**Drift detection**: when bumping `pyproject.toml` `version`, also
update the version pin in this section and re-validate that every
formula reference (`core/_testing.py:…`, `core/_wfa.py:…`,
`core/_spa.py:…`) still matches the current source — line numbers
shift as the code evolves.
