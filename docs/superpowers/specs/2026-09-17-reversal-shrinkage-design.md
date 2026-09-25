# Self-weighting reversal correction (uncertainty-driven λ)

Date: 2026-09-17
Branch: `v1-C3`
Status: design, awaiting final review before implementation

## Problem

The reversal correction (`headline_estimator="reversal"` on `PowerModelMethod`, added
2026-09-17, currently uncommitted on `v1-C3`) fixes F2: reference turbines (true uplift 0) read a
positive mean because the counterfactual model has an out-of-sample sum bias that inflates the
forward ratio. Fitting the reverse direction (train the upgraded rows, predict the baseline rows)
and reporting `sqrt((1+r_fwd)/(1+r_rev)) − 1` cancels a shrinkage common to both directions and
takes the real T13 AeroUp wholefarm reference mean from +1.019 to +0.009 pp.

But the synthetic prepost sweep showed a **crossover at ~6 months**: the full correction is a clear
win at post-period ≥ 6 months and **harmful below** it. Below ~6 months the reverse fit trains on a
short post-period Q, is data-starved, and over-corrects strongly negative (1-month `cp_plus_10`
reads −2.16 pp bias) while doubling spread.

The branch is currently **unguarded** — it applies the full correction at every Q. We need to gate
it. The requirement (user, 2026-09-17): do **not** hand-fit an arbitrary `λ(Q-length)` fade curve
from a handful of anecdotes. The method should judge its own chance of help vs harm from measurable
properties of the data and set the shrinkage accordingly. Simple measures of Q (date span, row
count) may sit on top as guardrail bounds, but they are not the mechanism.

## What "trust" means here

The correction's premise is that the reverse model's out-of-sample sum bias equals the forward
model's (`b_rev == b_fwd`), so the shared bias cancels. Reversal **helps** when that holds and
**harms** when Q is too starved for the reverse fit's bias to resemble the forward direction's. So
the quantity to self-measure is: how confident are we that the reverse (Q-trained) reading is
precise enough to trust its bias estimate. That is a property of the data, not the calendar.

## Design

### A. Parameterization — log-space bias-subtraction shrinkage

Work in log space. With `f = log(1+r_fwd)`, `g = log(1+r_rev)`, and `θ = log(1+u)` the target:

```
f = θ + β      (forward: shared bias β = log(1+b) inflates the ratio)
g = −θ + β     (reverse: true ratio is 1/(1+u), same bias)
```

The two-equation solution is the full reversal, `θ = (f−g)/2`, `β̂ = (f+g)/2`. Introduce a shrinkage
`λ ∈ [0,1]` on how much of the estimated shared bias to subtract:

```
θ̂(λ) = f − λ·β̂ = (1 − λ/2)·f − (λ/2)·g
report u = exp(θ̂(λ)) − 1
```

- λ = 0 → forward **exactly** (`θ̂ = f`). This is the correct floor. Note: shrinking `r_rev → 0`
  instead (the earlier "shrink r_rev toward 0" phrasing) is wrong — it collapses to
  `sqrt(1+r_fwd) − 1 ≈ r_fwd/2`, halving a real uplift. Use the log-space bias-subtraction form.
- λ = 1 → the current full reversal.

`β̂` is the quantity `_implied_shrinkage` already reports.

### B. λ is computed, not fit

λ minimizes the mean squared error of the reported estimate:

```
MSE(λ) = (1−λ)²·β²  +  Var(θ̂(λ))
Var(θ̂(λ)) = (1−λ/2)²·σ_f²  +  (λ/2)²·σ_g²  −  2·(1−λ/2)·(λ/2)·σ_fg
```

Setting dMSE/dλ = 0 gives the closed form:

```
λ* = (2β² + σ_f² + σ_fg) / (2β² + ½σ_f² + ½σ_g² + σ_fg),  clamped to [0, 1]
```

Use the **positive-part James–Stein** plug-in for the unknown `β²`:

```
β² → ( β̂²  −  Var(β̂) )₊ ,   Var(β̂) = (σ_f² + σ_g² + 2σ_fg)/4
```

so an apparent bias within noise of zero is not "corrected".

Limits (all emergent, none imposed):

- reverse fit useless, σ_g → ∞ → **λ → 0** (report forward — the right call for a tiny Q)
- both fits precise and β real → **λ → 1** (full correction)
- apparent bias within noise of zero → the **bias-driven** part of λ vanishes (James–Stein sets
  β² → 0), so we do not chase a mirage; any λ that remains is then pure variance reduction (below)
  and is itself small when σ_g is large
- genuine β ≈ 0 with a precise reverse fit → λ can still be large, because `−g` is a second
  unbiased estimate of θ, so the reversal doubles as variance reduction, not only bias removal.
  This is safe: with β ≈ 0 the correction is unbiased at any λ, and short-Q noise (large σ_g) pulls
  λ back down through the same formula.

This is computed per campaign **and** per turbine.

### C. σ estimation and the effort lever

A new `effort` level on `PowerModelMethod` (low / high; medium reserved for later) selects how
`σ_f, σ_g, σ_fg` are estimated. This is the v1 successor to v0's bootstrap-loops override — a named
level, not a raw loop count — and matches how analysts work: run cheap while reasoning through
config (excluded turbines, manual north change-points), then one authoritative full-effort final
run.

- **low** → σ from cross-fit **fold spread**, reusing the folds already computed for the F2
  diagnostic. Roughly no extra refits. For the iterate-and-reason phase.
- **high** → σ from a **time-block bootstrap** (blocks respect autocorrelation), reusing
  toggle_specialist's calibrated block machinery. Each replicate refits both directions and
  recomputes `f` and `g` jointly, so `σ_fg` comes for free. The authoritative final run.

Effort feeds through to λ and therefore the headline. Per the decision below, the legs are not
required to agree: the high-effort run is the number of record and the headline is allowed to
sharpen with effort.

### D. Headline uncertainty (enabled, not yet a P95)

The same σ inputs yield a headline sampling variance directly (the `Var(θ̂(λ))` above). The **valid**
way to report it at high effort is to have each block-bootstrap replicate recompute the entire
λ-shrunk headline and record that final `u`; the percentile spread of those per-replicate headlines
is an honest sampling interval because it captures the uncertainty in λ and β̂ themselves, not just
in f and g. The analytic `Var(θ̂(λ))` formula treats λ as known and understates, so it is a
diagnostic only.

Scope: this is the **random/sampling** component of a headline uncertainty. A full P95 in the v0
sense also carries systematic terms (residual method bias when λ<1, long-term wind-resource
representativeness, model misspecification beyond what reversal removes). So this machinery is the
natural seed for the Phase-2 uncertainty workstream, not a calibrated P95.

Phase-1 handling: **compute** the headline band and expose it as a diagnostic/optional field in the
run output; do **not** put it on the leaderboard or call it a calibrated P95. Low-effort produces
only an indicative band (fold-spread is interpolation variance and misses extrapolation and
λ-uncertainty).

### E. Scope boundaries and decided defaults

1. Shrinkage applies to the **scalar prepost headline** (`method.py:570-573`). The per-bin
   conditional path (`_conditional_by_bin`) **keeps full reversal for now** — it is a diagnostic;
   the same treatment can be added later. (Decided default, 2026-09-17.)
2. Toggle is untouched — it already keeps the forward ratio.
3. β is estimated **per campaign** from `(f+g)/2`. No informative "bias is ~+0.4 % positive" prior
   from the F2 work is baked in; that would fit to this dataset. (Decided default, 2026-09-17;
   informative prior noted only as a possible future refinement.)
4. Simple-Q guardrails (minimum date span, minimum row count) are **sanity clamps only** — the σ
   signal does the real work.
5. Final-run-authoritative: effort is allowed to move the P50 headline via λ; the high-effort run
   is authoritative. fold-σ is not required to be a calibrated stand-in for bootstrap-σ. (Decided,
   2026-09-17.)

## Validation

Re-run last session's `study_power_model_compare` prepost sweep (7 profiles × 1/2/3/6/12-month
post-periods) with computed-λ reversal, and confirm:

- references (truth 0) read ~0 pp mean at **every** length (the F2 target)
- §8 injected uplifts recovered with |bias| ≤ forward at every length
- the **~6-month crossover disappears**: short Q self-shrinks toward forward (large σ_g → small λ)
  instead of over-correcting to about −2 pp

Plus the real T13 AeroUp campaign: reference mean ≈ 0, headline stable across effort within its
band. Expectation: computed λ traces out the empirical crossover on its own.

## Build order (TDD)

1. Pure functions with unit tests: the log-space shrinkage combine, and `λ*` (limits, monotonicity
   in σ_g, the James–Stein zero-bias case).
2. Fold-spread σ estimator.
3. Wire `effort`/λ into `PowerModelMethod.estimate`; extend `TestReversalCorrection`.
4. Block-bootstrap σ for high effort, recording the per-replicate headline for the band.
5. Run the validation sweep; tune only the guardrail clamps if needed (not a λ curve).
6. Re-record the benchmark baseline JSON (the new feature moves it).

Record the outcome as a CF finding in `docs/v1/findings_campaigns.md` (and the private
`c3-findings.md`), not as a tracked spec.

## Open questions for review

- Block size / replicate count for the high-effort bootstrap (start from toggle_specialist's
  calibrated 6-hour blocks; confirm it transfers to the prepost energy contrast).
- Whether the per-turbine λ should ever be pooled/regularized across a farm's references, or stay
  strictly independent per turbine.
