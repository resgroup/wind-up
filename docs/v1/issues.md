# wind-up v1 — first issues (drafts)

Drafts of the first concrete issues, to be refined here and then created as GitHub
issues on `resgroup/wind-up`. These cover **Phase 1** only (see
[roadmap.md](roadmap.md)): the public evaluation harness, the v0 baseline, the
minimal data contract, and the first new candidate method — all judged on **P50
accuracy and precision only**.

Suggested order of execution: #1 → #2 → #3 → #4 → #5. (Issue 4 was originally a data
contract + method-selector issue; it has been re-scoped to a naive energy-ratio method —
see Issue 4 below for why.)

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

## Issue 4 — Naive energy-ratio method (WS3)

**Goal:** a second, deliberately simple, fully independent method that validates the
harness is not implicitly tuned to v0 — and proves the existing thin method seam is
genuinely pluggable.

**Why this, not the original "data contract" issue:** the thin
`MethodInput`/`MethodOutput` seam from Issues 2–3 already *is* the shared, method-
agnostic contract; the drafted "per test-reference conditioned dataset" was over-fit to
v0 (an R-learner fits once per test turbine over all references at once), and the
`assessment_method` production selector only earns its place once there is a winner to
promote. The durable kernel of the old issue — a treatment-invariant reference-only
feature builder + the §8 bias-guard test (design note §3/§8) — folds into Issue 5.

**The method.** For a set of rows let `ρ = Σ test_power / Σ reference_total_power` over
*complete-case* timestamps (test turbine **and every** reference finite). Estimate
`uplift = ρ(treated) / ρ(baseline) − 1`. It never reads the test turbine's own wind
speed (design note §3), shares no code with v0, and has no wind_up dependency. It makes
no covariate-shift correction by design, so it is the "don't condition at all" floor:
biased on prepost, near-unbiased on toggle (interleaved on/off share a wind
distribution).

**Scope**
- `NaiveRatioMethod` behind the existing `Method` seam; prepost **and** toggle.
- Rich per-run diagnostics (a data-stats CSV per `all`/`baseline`/`upgraded` segment, a
  headline-results CSV, optional plots) so a human can confirm the right data was
  received and interpreted; the headline uplift is re-derivable from the stats CSV.
- Add toggle support to `V0BinnedMethod` (wiring wind_up's native toggle assessment) so
  v0 can be scored on toggle campaigns too.
- Add the naive method to the existing prepost driver; add a new toggle example driver
  (3% Cp increase, 20-min-on/20-min-off) scoring naive + v0 + oracle.

**Done when:** `naive_ratio` is scored alongside `v0_binned` and the oracle on the
synthetic profiles for both prepost and toggle, the per-run diagnostics are written, and
its accuracy/precision appears in the leaderboard.

---

## Issue 5 — First candidate: cross-fit R-learner (P50) (WS2)

**Goal:** implement the design-note R-learner as the first new method and score it against the baselines (naive and v0).

**Scope**
- Cross-fit R-learner producing a P50 uplift, per the design note
  (LightGBM outcome [L2/Huber] + propensity nuisances; effect model on residuals).
- Be sure to provide rich csvs and diagnostic plots; look at Naive for inspiration and grow from there depending on the method details
- There should be no restrictions / limitations on what data is provided (v0 dependency removed from harness in previous PR). To start with just provide the same data columns that v0 gets to prove the method is superior; even more data columns (eg all Min and Max fields) can be provided later
- Provide ERA5 data to the method as well. It will need to be upsampled to 10min timebase. Use a wind speed correlation sweep to sync with the SCADA (see existing wind-up code for inspiration)
- **Treatment-invariant reference-only features** — enforced; include a regression
  test on a deliberately treatment-corrupted nacelle wind speed proving the
  reference-only rule removes the post-treatment bias (design note §8). Reference
  turbines affected by the wakes of upgraded turbines may need to be excluded in
  certain wind directions (very relevant for wake steering).
- Another known feature to avoid is voltage (at the turbine's external connection); if the turbines are wired in series then the voltage drop across the test turbine will be approximately proportional to its active power, so if the method can see the voltage at the two turbines either side of the test turbine then estimating power is trivial and the power estimate is not using information about the weather. The giveaway that this issue is happening is feature importance; make sure feature importance diagnostic plots and log messages are emitted.
- Score on the harness across all synthetic profiles and the short-campaign sweep;
  compare to naive first (fast), then the v0 baseline after naive is beaten.
- Uncertainty/P95 explicitly **out of scope** here (Phase 3 / WS4) but keep it in mind.
- Reporting uplift by condition (eg uplift by wind speed, direction, etc) out of scope for now but keep it in find

**Done when:** the R-learner runs through a prepost and toggle study and its P50 accuracy/precision vs the baseline is recorded in the leaderboard and similar or better than v0.

---

## Not in the first wave (tracked for later phases)

- Uncertainty / P95 model: block bootstrap, conformal OOD, density-ratio long-term
  weighting (WS4, Phase 3).
- Further candidate methods: DSWE / `funGP`, Astolfi multivariate-linear (WS2,
  Phase 2).
- Conditional/heterogeneous uplift reporting & SHAP story (G3, Phase 2).
- Report content generation and I/O / step-independence maturation (WS5, Phase 4).
- **Method-controlled baseline horizon / recency weighting (R-learner).** Today the R-learner
  pools *all* pre-upgrade baseline it is given (e.g. 24 months) into the nuisance fits with equal
  weight, and uses no timestamp features, so it cannot down-weight stale data. Since the goal is
  usually the upgrade's *future* benefit, the most recent baseline is the most representative of
  the farm's future state, and far-past data (turbine ageing, sensor drift, soiling, controller
  changes) is less so. Add a method-owned horizon control — a `max_baseline_months` cap and/or an
  exponential recency weight on the fit (via LightGBM `sample_weight`) — so the method, not the
  harness, decides how far back to trust. Keep it compatible with the no-timestamp-feature rule
  (weighting/selection by recency is fine; calendar features are not) and the block bootstrap.
  Pairs naturally with the long-term ERA5 extrapolation work (representativeness weighting, WS4).
