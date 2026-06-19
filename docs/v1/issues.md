# wind-up v1 — first issues (drafts)

Drafts of the first concrete issues, to be refined here and then created as GitHub
issues on `resgroup/wind-up`. These cover **Phase 1** only (see
[roadmap.md](roadmap.md)): the public evaluation harness, the v0 baseline, the
minimal data contract, and the first new candidate method — all judged on **P50
accuracy and precision only**.

Suggested order of execution: #1 → #2 → #3 → #4 → #5, with #4 (the data contract)
informing #3 and #5.

---

## Issue 1 — Synthetic upgrade-dataset generator (WS1)

**Goal:** generate realistic SCADA-like datasets with a *known* injected uplift, to
serve as ground truth for evaluating uplift methods.

**Scope**
- Select real SCADA from stable, no-upgrade periods/turbines (start with Hill of
  Towie open data; design for other open wind farms).
- Inject known uplift *profiles*:
  - constant Cp change; this increases or decreases power in region 2 but does not change rated power
  - wind-speed-dependent Cp change;
  - condition-dependent (e.g. direction/wake-dependent — wake-steering shape, or stability dependent as TuneUp may be) Cp change.
  - rated power change
- Preserve realistic structure (autocorrelation, reference turbines, conditions)
  so methods can't trivially detect the injection.
- Emit the dataset plus a machine-readable record of the true injected uplift
  (overall and per-condition) for scoring.

**Done when:** a documented function/CLI produces ≥1 synthetic dataset per profile
from open data, with the ground-truth uplift recorded alongside.

**Open questions:** where the generator lives (wind-up subpackage vs companion
public repo — current lean: in-repo but standalone); how condition-dependent
profiles are parameterised.

---

## Issue 2 — P50 evaluation harness & scoring (WS1)

**Goal:** score any uplift method's P50 estimate against the known injected truth,
including a short-campaign robustness sweep.

**Scope**
- Accuracy (bias) and precision (spread) metrics: recovered uplift vs injected,
  overall and (where applicable) per condition.
- Short-campaign sweep: re-score as a function of campaign length / data volume to
  quantify how accuracy and precision degrade with less data (serves G2).
- A simple results format / leaderboard so methods can be compared side by side.
- **P50 only** — no uncertainty/P95 scoring in this phase.

**Done when:** given a method (conforming to the Issue 4 contract) and a synthetic
dataset, the harness emits comparable accuracy/precision numbers and a
campaign-length curve.

---

## Issue 3 — Wire the v0 binned method as the baseline (WS2)

**Goal:** run the existing v0 binned power-curve uplift method through the harness
to establish the bar every new method must beat.

**Scope**
- Adapt the current pre/post power-performance pipeline to consume the Issue 4
  data contract and emit a P50 estimate the harness can score.
- Record baseline accuracy/precision and the short-campaign curve for each
  synthetic profile.

**Done when:** baseline P50 scores exist for all synthetic profiles and are the
reference point in the leaderboard.

---

## Issue 4 — Assessment data contract + pluggable method interface (WS3)

**Goal:** the minimal foundation that lets multiple methods and the harness share
one interface — without a big refactor.

**Scope**
- Define a standardized **per test-reference, pre/post conditioned dataset** (a
  documented schema) carrying **treatment-invariant features only** (reference /
  met-mast / LiDAR / ERA5 derived; never the test turbine's own SCADA wind speed —
  see design note §3).
- Define a thin method interface: input = the contract; output = a P50 uplift
  estimate (overall and, where supported, per condition).
- Add a minimal `assessment_method` selector so a method can be chosen via config,
  reusing existing inputs / test-ref pairing / result objects.

**Done when:** the v0 baseline (Issue 3) and a stub method both run through the
same contract + selector, and the harness (Issue 2) consumes their outputs.

**Note:** keep this deliberately thin — just enough to avoid baking in assumptions
about the winning method.

---

## Issue 5 — First candidate: cross-fit R-learner (P50) (WS2)

**Goal:** implement the design-note R-learner as the first new method behind the
Issue 4 interface and score it against the baseline.

**Scope**
- Cross-fit R-learner producing a P50 uplift, per the design note
  (LightGBM outcome [L2/Huber] + propensity nuisances; effect model on residuals).
- **Treatment-invariant reference-only features** — enforced; include a regression
  test on a deliberately treatment-corrupted nacelle wind speed proving the
  reference-only rule removes the post-treatment bias (design note §8). Reference
  turbines affected by the wakes of upgraded turbines may need to be excluded in
  certain wind directions (very relevant for wake steering).
- Score on the harness across all synthetic profiles and the short-campaign sweep;
  compare to the v0 baseline.
- Uncertainty/P95 explicitly **out of scope** here (Phase 3 / WS4).

**Done when:** the R-learner runs through the contract, the bias-guard test passes,
and its P50 accuracy/precision vs the baseline is recorded in the leaderboard.

---

## Not in the first wave (tracked for later phases)

- Uncertainty / P95 model: block bootstrap, conformal OOD, density-ratio long-term
  weighting (WS4, Phase 3).
- Further candidate methods: DSWE / `funGP`, Astolfi multivariate-linear (WS2,
  Phase 2).
- Conditional/heterogeneous uplift reporting & SHAP story (G3, Phase 2).
- Report content generation and I/O / step-independence maturation (WS5, Phase 4).
