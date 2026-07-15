# wind-up v1 — first issues (drafts)

Drafts of the concrete issues, to be refined here and then created as GitHub
issues on `resgroup/wind-up`. Issues 1–8 cover **Phase 1** (see
[roadmap.md](roadmap.md)): the public evaluation harness, the v0 baseline, the
minimal data contract, and the first new candidate methods — all judged on **P50
accuracy and precision only**. Issues 9+ are the **second wave**, drafted after
Issue 8 shipped: improving `power_model` (the current best method) on overall and
conditional P50, then extending the measurement to AEP uplift and starting the
uncertainty (Phase 3 / WS4) work.

Suggested order of execution: #1 → #2 → #3 → #4 → #5 (done), then
#9 → #10 → #11 → #12 → #13 → #14 → #15 (done) → #16 → #17 → #18 → #19. (#9–#12 are
independent input-data and model trials sharing one evaluation protocol — many ideas,
each tested one by one; #14 is small and independent, so it can be pulled earlier — and
#15's adaptive rule wants #14's hardened conditional scoring in place; #16/#17 are the
`power_model` hygiene + consistency follow-ups that fall out of #15 and are best done
before the AEP/uncertainty work builds on the conditional path; #19 builds on #18's AEP
machinery, so it goes last.) (Issue 4 was originally a data contract + method-selector
issue; it has been re-scoped to a naive energy-ratio method — see Issue 4 below for why.)

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
- Reporting uplift by condition (eg uplift by wind speed, direction, etc) out of scope for now but keep it in mind

**Done when:** the R-learner runs through a prepost and toggle study and its P50 accuracy/precision vs the baseline is recorded in the leaderboard and similar or better than v0.

---

## Issue 6 — Clip power_model predicted power to a sane range (power_model)

**Goal:** stop the boosted counterfactual over/under-shooting the physically plausible
range at the extremes — a small precision gain, especially in the fragile tail bins. Do
this first: it is low-risk and serves as the simple test case for Issue 7.

**Scope**
- In `PowerModelMethod._fit_predict` (`benchmarking/baselines/power_model/method.py`),
  clip every model prediction (the upgraded counterfactual *and* the baseline-holdout
  prediction) to `[lower, upper]` with `lower = min(0, min(y_train))` and
  `upper = max(rated_power_kw, max(y_train))`, where `y_train` is the fitted-on baseline
  outcome. (Tree boosting sums trees, so predictions can slightly exceed the training
  `y` range; the clip binds only at the extremes — expect a small effect.)
- Add an optional `rated_power_kw: float | None = None` config field; when `None` the
  upper bound is just `max(y_train)`. HoT rated = 2300 kW.
- Keep it a pure post-prediction transform so overall and conditional both use clipped
  predictions consistently.

**Done when:** predictions are bounded; existing recovery/placebo tests still pass; a
unit test on the clip helper confirms out-of-range predictions are pulled to the bounds
and in-range ones are untouched.

---

## Issue 7 — Confirm the improvement-evaluation workflow surfaces what's needed (power_model)

**Goal:** before the larger bias-correction work, prove that
`study_power_model_compare.py` gives the information needed to judge a power_model change
— using the Issue 6 clip as the first, simple test case.

**Scope**
- Run `benchmarking/baselines/study_power_model_compare.py` (power_model vs frozen
  v0/naive, 3-method plots) on the clipped power_model.
- Confirm it reports, side-by-side and legibly, both **overall** P50 error and the
  **per-condition** (ws & TI) recovered-vs-truth curves for the `ti_dependent_cp` /
  `ws_dependent_cp` hard cases — enough to see whether a change helped, hurt, or was
  neutral, overall and per bin.
- If a needed view is missing (e.g. a corrected-vs-current conditional overlay against
  truth, or a per-bin error table), add the minimal reporting to the study/inspect
  scripts so Issue 8 can be evaluated the same way.

**Done when:** a single command produces the overall + per-condition comparison for the
clip change, and the improvement is readable from it without ad-hoc digging.

---

## Issue 8 — Cross-prediction bias-cancellation for shrinkage-driven conditional bias (power_model)

**Goal:** remove the counterfactual model's **per-condition** (shrinkage) bias — the F5 root
cause — by cancelling it between two symmetric train/predict directions on weather-matched
data. The overall P50 is left as the single full-window fit (its whole-window shrinkage
integrates to ≈0, so it is already the cleanest headline); the two-direction correction is
spent on the per-condition decomposition, which is then re-leveled so its energy aggregation
equals that headline (F8).

**Scope**
- **Matching-variable analysis (one-off, do first).** On Hill of Towie, run a feature-
  importance analysis to choose which ERA5 variables to match on (likely wind speed +
  direction, possibly more). Record the chosen set + rationale in `docs/v1/findings.md`
  and hard-code it as the default matching set. ERA5 is preferred for its full coverage
  and temporal stability; the set may later be tuned per wind farm.
- **ERA5 coarsened-exact matching (CEM).** New utility: bin baseline vs upgraded rows on
  the chosen (synced) ERA5 variables; within each cell subsample the larger side to the
  smaller side's count (seeded); drop one-sided cells. Yields equal-count, weather-matched
  baseline/upgraded sets. The matching axis (ERA5) is distinct from the reporting/binning
  axis (test-turbine ws/TI, kept as today so bins match ground truth).
- **Two directions + geometric combine.** Forward: train on matched baseline, predict
  matched upgraded → `r_fwd` (overall and per bin via `energy_ratio_by_bin`). Reverse:
  train on matched upgraded, predict matched baseline → `r_rev`. Combine
  `uplift = sqrt((1+r_fwd)/(1+r_rev)) − 1` (exact under a common per-bin multiplicative
  shrinkage); also emit implied bias `1/sqrt((1+r_fwd)(1+r_rev))` as a diagnostic. Guard
  non-positive `(1+r)` and empty/sparse bins.
- **Default, not opt-in.** The matched two-direction cross-prediction is the **sole**
  conditional-uplift method and is **on by default** via `conditional_uplift: bool = True` on
  `PowerModelMethod` (an A/B `bias_correct` flag was used during development, then removed once
  the approach won). It requires ERA5 (the matching axis) and runs **last** — nothing else
  depends on it — so `conditional_uplift=False` skips the expensive cross-prediction and returns
  only the overall P50. Per-run outputs: conditional CSVs in a `conditional/` subfolder, the
  implied-shrinkage plot in `plots/7_conditional_uplift/`.
- Reuse the existing outcome model factory (`make_outcome_model`) and `CONDITION_BINS`;
  diagnostics carry the implied-shrinkage `1/sqrt((1+r_fwd)(1+r_rev))` and the CEM balance, and
  `study_power_model_compare.py` overlays the committed benchmark vs the current run vs truth per
  covered `(profile, condition)` (a benchmark regression view).

**Done when:** `study_power_model_compare.py` (Issue 7) shows the `ti_dependent_cp` /
`ws_dependent_cp` conditional curves materially flatter toward truth with the overall P50
unchanged; a regression test recovers a known flat-zero placebo per-condition uplift to a small
absolute per-bin bias (the shrinkage tilt cancelled); findings.md updated. *Shipped 2026-07-03:
overall P50 bit-identical, conditional score roughly halved (prepost mean |bias| 18.2→6.3 pp,
toggle 13.0→4.1 pp) and the benchmark regenerated under the new default.*

---

## Issue 9 — ERA5 feature engineering: derived quantities + hub-height interpolation (WS2)

**Goal:** use ERA5 better. Today the model consumes the raw Open-Meteo columns (plus
direction sin/cos); derive the physically meaningful, §3-legal quantities that actually
drive turbine power and its scatter — including a turbulence proxy, which the model
currently lacks entirely (the F5 root cause: the §3 rule denies it the test turbine's own
ws/TI, so its residuals correlate with TI). This issue also establishes the shared
**candidate-by-candidate evaluation protocol** that Issues 10–11 reuse.

**Evaluation protocol (shared by Issues 9–11).** There are many ideas for improving the
input data available to the model; they must be tested **one by one** (or in the smallest
sensible groups) so each one's effect is known. Per candidate: add it, A/B via
`study_power_model_compare.py` against the committed benchmark, and accept only what earns
its place. Gates: **overall P50 no worse** — the placebo prepost bias is the sharpest read
(a feature that imports temporal drift shows up there); **conditional** mean |bias| /
`implied_shrinkage` improved or neutral; **feature importance sane** (the F2 lesson: a
channel can act as a calendar proxy; the Issue 5 voltage lesson: importance diagnostics
are the giveaway). Record every accepted **and rejected** candidate with evidence in
`findings.md`; regenerate the benchmark JSON whenever a default changes.

**Scope (the ERA5 candidates)**
- **Hub-height wind speed.** New optional `hub_height_m` config (Hill of Towie = 59 m).
  Compute the local shear exponent `alpha = ln(ws_100m/ws_10m)/ln(10)` per row and
  interpolate with the shear power law: `ws_hh = ws_100m · (hh/100)^alpha`.
- **Gust turbulence proxy:** `wind_gusts_10m / wind_speed_10m` — a unitless, TI-like
  quantity (guard the calm-wind denominator). Do a comparison to real (SCADA) TI and play around with the calculation if it the ERA5 derived TI is not realistic.
- **Shear exponent** `alpha` as a feature in its own right (F6 already validated it as a
  real independent signal and folded out the 10 m/100 m collinearity), **vertical veer**
  (`wind_direction_100m − wind_direction_10m`, wrapped to ±180°), **air density** (from
  temperature + surface pressure + humidity), and a **stability indicator** if the
  available fields allow.
- Build as a **shared ERA5-derivation utility** so every method *and* the CEM matching
  step reuse it (the shear derivation currently lives only in
  `inspect_era5_matching_importance.py`).
- Optionally revisit `matching_vars` afterwards (e.g. the gust proxy or `alpha` in place
  of raw gusts, per F6's deferral) — a separate benchmark regeneration if taken.

**Done when:** the derivation utility exists with unit tests (known-input checks); every
candidate has an accept/reject verdict recorded in `findings.md`; the benchmark is
regenerated under the accepted default set with overall P50 and conditional no worse.

---

## Issue 10 — Time features: campaign-relative drift, season, solar position (WS2)

**Goal:** today the timestamp is dropped before modelling, so the model cannot know about
anything that varies with time but not weather. Trial explicitly-constructed time features
— the headline motivation is **reference turbines changing over time**, a major bias risk
and one of the main challenges the method must deal with: a drift feature gives the model
a chance to absorb reference change instead of attributing it to the upgrade. It is also possible to ERA5 to drift over time vs the site due to ERA5 input data changes over time and site exposure changes over time (eg forestry growth, new neighbour wind farm, etc.)

**Scope (one by one, per the Issue 9 protocol)**
- **`time_since_campaign_start`** continuous value in unit of days (negative in the baseline). Lets the model track
  slow reference change relative to the campaign.
- **Time of year:** continuous value measuring the Julian day offset to a meteorologically meaningful anchor (say
  June 21), split into sin/cos components — tells the model roughly what season it is,
  potentially useful beyond the instantaneous weather data.
- **Time of day, as solar altitude + azimuth** computed from an optional lat/long config
  and the UTC timestamp (a physically meaningful encoding of diurnal cycle; calculation
  code exists in other projects and can be ported).
- **Known caveats to test against, not reasons to skip.** In prepost,
  `time_since_campaign_start` is treatment-collinear, and trees cannot extrapolate — for
  upgraded rows the feature exceeds every training value, so predictions clamp at the
  boundary leaves: it can encode baseline-internal drift but freezes it at the changeover.
  Season features against a < 12-month baseline are partial calendar proxies. Whether each
  helps or imports bias is exactly what the placebo gate decides — run it at multiple
  campaign lengths in **both** modes before accepting.
- The R-learner's "no timestamp features" rule (later-work list) was about shuffled
  cross-fitting; power_model has no cross-fitting, so trialling these is legitimate — but
  a time feature dominating the importance ranking is a red flag.

**Done when:** each time feature has an accept/reject verdict with placebo evidence in
both modes in `findings.md`; the benchmark is regenerated if any are accepted.

---

## Issue 11 — Reference power statistics features (max / min / SD) (WS2)

**Goal:** give the counterfactual a local variability/turbulence signal through the most
**calibration-stable channel** — the reference power signal. Bring in each reference's
active-power **max, min and SD** companion fields (present in the Hill of Towie open
data; usually available from any wind farm). Within-period power SD in particular is a
§3-legal turbulence proxy, sited at the farm rather than at ERA5's grid scale.

**Why reference power and not reference wind speed:** reference nacelle wind speed /
wind-speed SD were considered and **rejected**. A reference anemometer is at high risk of
changing calibration over time; in a prepost campaign that drift would be read as uplift
(a toggle campaign might tolerate it, but prepost and toggle share one code path, so the
risky feature stays out of both). Reference *power* is more likely to be stable — and
since references are usually the same turbine type as the test turbine, exposed to the
same performance-degradation tendencies, it leads to a fairer expectation of what the
test turbine could have produced.

**Scope**
- Wire the max/min/SD active-power fields through `build_reference_features` (same
  `"<tag> @ <turbine>"` naming; NaN-tolerant as today; `check_reference_only` still
  applies). Column names configurable like the existing `active_power_col`.
- A/B per the Issue 9 protocol: one field (or the trio) at a time, placebo gate,
  importance watch — power max/min/SD could in principle carry a drift signature too
  (e.g. a reference derate changes its max), which is precisely what the placebo run
  detects.
- Confirm the harness/synthetic path carries these columns unmodified for reference
  turbines (only the test turbine's signals are injected).

**Done when:** verdict per field recorded in `findings.md`; benchmark regenerated if
accepted; the rejected reference-anemometer alternative and its rationale are noted in
the findings entry.

---

## Issue 12 — Outcome-model fundamentals: objective, hyperparameters, calibration slope, alternative learners (WS2)

**Goal:** revisit the basics of the counterfactual model itself. Today it is a LightGBM
regressor with the L2 objective and fixed design-note hyperparameters (600 trees,
lr 0.03, 63 leaves, `min_child_samples=200`, subsample/colsample 0.8) — never tuned, no
early stopping, and identical whether the fit has ~13k training rows (3-month toggle) or
~250k (2-year prepost baseline).

**Two framing principles (record them in findings.md so they aren't relitigated):**
- **The objective must target the conditional mean.** The estimand is an energy ratio and
  energy is a sum of conditional means, so L2 stays the default (design note §2). Power
  conditional on features is skewed (near cut-in and around rated), so median-type
  objectives — MAE, Huber in its robust regime, quantile-0.5 — estimate the median and
  would bias the energy sum. The F5 shrinkage is a *regularisation* artefact, not a loss
  artefact; changing objective does not fix it. Legitimate within the mean family:
  **Tweedie** or variance-weighted L2 for the strong heteroscedasticity of power
  (an efficiency candidate, not a bias fix).
- **Tune on uplift metrics, never on prediction RMSE.** More regularisation can improve
  held-out RMSE while *worsening* shrinkage — the two objectives disagree exactly where
  it matters. Yardsticks: placebo bias on the harness, per-bin residual flatness and
  predicted-vs-actual **calibration slope** (target ≈ 1) on a time-blocked held-out
  baseline, and replicate spread. Guard against overfitting the benchmark: tune on the
  held-out-baseline proxies, confirm on the harness, ideally on turbines/windows not used
  for tuning.

**Scope (per the Issue 9 protocol, one candidate at a time)**
- **Data-size-adaptive capacity:** early stopping on a time-blocked validation split;
  scale `min_child_samples` / leaves with training size (the 3-month and 24-month fits
  differ ~10× in rows but share one capacity today).
- **`linear_tree=True`:** piecewise-linear leaves reduce the flat-leaf compression at the
  edges of the feature distribution — a direct attack on the F5 shrinkage (and it
  softens the Issue 10 boundary-clamping caveat, since linear leaves extrapolate).
- **Post-hoc calibration-slope correction:** fit actual-vs-predicted on time-blocked
  held-out baseline (linear, or isotonic), apply to the counterfactual predictions — the
  cheap de-shrinking cousin of Issue 13's residual calibration; measure how much each
  contributes when combined.
- **Seed ensembling:** average K seeds (subsample/colsample noise) for variance
  reduction; measure the replicate-spread gain vs run-time cost.
- **Alternative learners behind a model-factory seam** (make the outcome model injectable
  on `PowerModelMethod` rather than hard-coded to `make_outcome_model`): CatBoost /
  XGBoost as same-family sanity checks; a small tabular NN or TabPFN for the short-
  campaign regime; and a deliberately low-variance structured baseline (GAM / linear on
  hub-height ws + direction features from Issue 9) — lightly regularised so nearly
  shrinkage-free, valuable as a cross-check on the tree models' conditional bias even if
  its overall accuracy is worse.
- **Quantile objectives are out of scope for the point estimate** (median ≠ mean under
  skew → biased energy). Quantile models return in Issue 19 / WS4 for conformal OOD
  filtering and diagnostics — not as the uplift estimator.

**Done when:** a findings entry records the verdict per candidate against the uplift
yardsticks; defaults change only where those improve; the benchmark is regenerated if
defaults change.

---

## Issue 13 — Calibrate out the structural headline bias; revisit toggle's campaign-only training (WS2)

**Goal:** remove the persistent overall-P50 bias: prepost ≈ **−0.4 pp on every profile**
(F3/F4) and toggle ≈ +0.4 pp at 3 months. The flatness across profiles says it is a model
artefact, not effect-dependent — but the two modes get there by different mechanisms, and
this issue addresses both. **Prepost:** the model's conditional bias `b(x)` integrates to
≈ 0 over the *training* (baseline) covariate mix, but the headline evaluates it over the
*upgraded* window's mix — any weather/seasonal shift between the windows converts
conditional bias into headline bias. Estimate that term on untreated data and subtract it
(this is F5's implication #1, baseline-residual calibration, never actioned — aimed at
the headline rather than the bins, which Issue 8 already fixed). **Toggle:** under
`toggle_campaign_only=True` the train (off) and predict (on) rows interleave through the
same weeks, so train/predict covariate shift is minimal by construction — the 3-month
bias is more plausibly small-sample shrinkage from the tiny fit (~6–7k off rows vs ~250k
for a 2-year prepost baseline). The candidate fix is more training data: the pre-campaign
baseline that `toggle_campaign_only` currently drops, made safe by exactly this issue's
calibration plus Issue 10's time features.

**Scope**
- **Time-blocked baseline cross-validation.** Replace the random 20% holdout in
  `_holdout_fit` with contiguous time blocks (out-of-fold prediction for every baseline
  row). The current shuffled split leaks autocorrelation (holdout rows sit minutes from
  training rows), so its residuals are optimistic — fine as a display, unusable as a
  calibration basis. This also makes the step-5 fit-quality diagnostics honest.
- **The calibration.** From the out-of-fold baseline residuals, estimate the mean residual
  per ERA5 cell (reuse the CEM coarsening); weight cells by the *upgraded* window's
  occupancy to get the expected headline bias under the upgraded mix; subtract it from
  `Σcounterfactual` (guard cells unseen in baseline — fall back to the global mean
  residual). For toggle, the campaign's off rows are untreated data under (almost
  exactly) the on rows' covariate mix, so their out-of-fold residuals estimate the
  headline bias more directly than the ERA5-cell reweighting prepost needs.
- **Toggle training-window revisit.** Today `toggle_campaign_only=True` drops every
  pre-campaign row, so a 3-month toggle fits on the campaign's off rows only. Trial
  including the pre-campaign baseline as a 2×2 on the placebo sweep — {campaign-only,
  all-data} × {calibration off, on} — with the Issue 10 time features in place.
  `time_since_campaign_start` is *not* treatment-collinear in toggle (on and off
  interleave), so the model can learn baseline→campaign drift and interpolate within the
  campaign, largely voiding the Issue 10 boundary-clamping caveat. Win condition: the
  extra rows cut shrinkage/spread without importing drift bias. Decide the
  `toggle_campaign_only` default from that evidence (`naive_ratio` keeps campaign-only
  regardless — there the restriction *is* the method's distribution matching).
- **Time-decay weights are a contingency, not a deliverable.** If the placebo with
  pre-campaign data shows drift-driven bias that the time features and the calibration do
  not absorb, trial exponential time-decay sample weights before rejecting pre-campaign
  data outright; otherwise the age-of-data feature carries the recency signal and keeps
  full effective sample size.
- **A/B behind a flag** (the Issue 8 playbook): placebo-centred evaluation across the
  campaign sweep in **both modes** — the win condition is placebo bias → ≈ 0 **without**
  a spread cost at short campaigns (the F7 failure mode to avoid).
- **Sequencing with Issues 9–12:** run after (or with) the feature and model trials —
  better features and a better-calibrated model shrink `b(x)` and therefore the
  correction (Issue 12's calibration-slope fix is the closest cousin); report how much
  bias remains for this calibration once those land.
- Keep the Issue 8 conditional path consistent: the re-level target becomes the
  calibrated headline.

**Done when:** pooled prepost |bias| materially reduced (target ≲ 0.1–0.2 pp on the
placebo) at no spread cost at any campaign length; the 3-month toggle bias explained, and
either materially reduced or the campaign-only default re-confirmed with placebo
evidence; a recorded decision on the `toggle_campaign_only` default; findings entry
quantifying both mechanisms; benchmark regenerated if the calibration or the toggle
default changes.

---

## Issue 14 — Harden the conditional decomposition: count floor, imputation, balance, re-level coverage (WS2)

**Goal:** fix the three known remaining defects of the two-direction conditional
estimate, so every reported per-bin number is either trustworthy or an explicitly
flagged, physics-informed fallback — and keep the scoring honest about the difference.

**Motivation strengthened by Issues 12–13 (F14–F16):** three further independent hits on
the un-floored sparse tail. (a) F15's residual calibration was partly sunk by sparse-cell
noise (~13 rows/cell on a 3-month toggle). (b) F16: merely *sample-weighting* the matched
direction fits flipped degenerate extreme-TI bins to three-digit per-bin uplift (+1039 %
in one replicate) at some decay half-lives and not others — replicate chance, i.e. the
tail bins sit on a knife edge; the time-decay weights had to be confined to the headline
fit as a workaround, and the floor should make the conditional path robust enough to lift
that restriction. (c) Throughout the F14–F16 A/B screens the conditional *mean* deltas
were dominated by the same few exploding tail cells, forcing verdicts to be read from
overall rows and cell tallies — the floor is also what makes the conditional benchmark
signal itself trustworthy.

**Scope**
- **Per-reporting-bin matched-count floor** (the F7/F9 follow-up): below a minimum
  matched count per side in a reporting bin, replace the overshooting two-direction
  combine with the imputed value (next bullet) and set `covered=False` in the
  conditional CSV — a flagged fallback, never a bare NaN, so every bin stays scoreable
  and abstention can't game the leaderboard (`summarize_errors` drops non-finite errors,
  so NaN-ing hard bins would otherwise improve the conditional score for free). The
  floor is on *effective* per-side counts — Kish ESS `(Σw)²/Σw²` if the balance
  reweighting below is adopted, raw counts otherwise — with the threshold chosen on
  placebo/benchmark evidence across many bins, not tuned to the one known bad TI bin.
  Kills the sparse-extreme swings (e.g. TI (0.45,0.50] −77 → +93 pp).
- **Physics-informed imputation for floored ws bins.** Uplift vs wind speed has a known
  shape for most upgrades (Cp-maximising from cut-in to rated, rated power thereafter):
  fill uncovered low-ws bins from the closest covered bin above (bfill on ascending ws),
  then fill everything above the last covered bin with **0 uplift** (at rated, baseline
  and upgraded both hit rated power). Two documented caveats: the 0-at-rated prior is
  wrong for uprating/power-boost upgrades (keep the imputer a documented, replaceable
  default and let benchmark profiles that violate it expose it), and when coverage stops
  well below rated the 0-fill is conservative for the in-between bins. TI has no such
  ordering physics — impute floored TI bins at the overall uplift.
- **Re-level with imputed bins pinned.** `_relevel_conditional` computes the aggregation
  over covered (finite-shape) bins only, but the target `1+overall` includes the energy
  of *uncovered* bins — so covered bins absorb the uncovered bins' MWh and λ is tilted
  when coverage is imperfect. Fix: hold imputed bins fixed at their imputed uplift
  (pinned, not λ-rescaled) and solve λ over the measured bins only, so measured +
  imputed together satisfy the headline identity exactly (guard the degenerate case of
  no measured bins → overall-only). With the floor creating more imputed bins, this
  matters more than today.
- **Per-reporting-bin balance.** The shrinkage-cancellation premise is *per bin*, but CEM
  equalises the ERA5-cell mix only globally — within one ws/TI reporting bin the forward
  and reverse row sets can still have different weather mixes (partly because the bin
  axes are the test turbine's post-treatment nacelle signals, so the same nominal bin
  samples different weather pre vs post). Post-stratify within each reporting bin to the
  **intersection** of the two directions' ERA5-cell supports (no new model fits needed):
  honest but thins sparse bins, which is fine — the ESS floor then catches them instead
  of a heavily-reweighted number sneaking through. Measure whether it moves the per-bin
  bias before adopting.
- **Coverage in the scoring.** Surface coverage next to conditional accuracy in the
  leaderboard/comparison outputs (e.g. fraction of upgraded energy in `covered` bins),
  and compare A/B variants on commonly-covered bins as well as overall — imputed bins
  are scored like any other (the flag distinguishes them), so the imputation prior is
  itself benchmarked.
**Scope addition (2026-07-05) — make `study_power_model_compare.py` measure the out-of-the-box
method on the full campaign range.** The committed benchmark must reflect performance with **no
specialist tuning**: today the driver passes four accepted-by-findings behaviour settings that are
not `PowerModelMethod` class defaults, and the campaign grid stops at 3 months.
- **Promote the accepted behaviour defaults onto the class** so a bare `PowerModelMethod`
  behaves like the benchmarked method: `TUNED_MODEL_PARAMS` (`min_child_samples=50`, F14) as the
  `model_params` default and `availability_feature=False` (F13) directly; make
  `CURATED_ERA5_EXCLUDE` defaultable by softening the exclusion to drop-if-present (it currently
  raises on non-Open-Meteo frames, which is why F13 left it driver-level). `reference_stat_cols`
  stays driver-level — it names a source-specific SCADA tag, i.e. data-schema description, not
  tuning; document that distinction where the drivers configure it. After promotion the driver
  should pass **only** schema description (columns, rated power, ERA5 frame, stat-col names).
- **Extend the scored campaign grid to 1/2/3/6/12 months in both modes** (toggle drops 9): relax
  the alignment guard to check truth equality on *intersecting* cases only (still failing loudly
  on any truth mismatch); **compute `naive_ratio` fresh** for campaign lengths the frozen
  reference run does not cover (it is cheap — v0 stays reference-only and is simply absent at
  1–2 months, where comparing to naive suffices); regenerate the committed benchmark JSON on the
  new grid. The F16 short-campaign regime evidence (`inspect_short_campaigns.py`) then becomes
  part of every future A/B automatically — which Issue 15's adaptive rule needs.
- **Sequencing:** the three fixes interact (balance changes effective counts → floor;
  floor changes coverage → re-level), so land and measure them incrementally in the
  order re-level fix (a pure bug) → floor + imputation → balance, per the Issue 9
  protocol. Unit tests for the floor, the imputation, and the reweighting; A/B via
  `study_power_model_compare.py` as usual.

**Done when:** the sparse-extreme bins are no longer the "worse than benchmark" set
(fixed, or flagged-and-imputed by the floor); conditional |bias| no worse on
commonly-covered bins; coverage visible in the leaderboard; the decomposition —
measured plus pinned imputed bins — still energy-aggregates exactly to the headline.
Additionally (the 2026-07-05 scope addition): a bare `PowerModelMethod` given only data-schema
config behaves identically to the benchmarked configuration; `study_power_model_compare.py`
scores 1/2/3/6/12-month campaigns in both modes with naive computed fresh where the reference
lacks it; the benchmark JSON is regenerated on the new grid.

*Shipped 2026-07-08 (findings F17): count floor (`_MIN_BIN_MATCHED_COUNT=50`) + physics imputation +
corrected pinned-imputed re-level — overall P50 bit-identical, conditional score −3.7 pp prepost /
−2.6 pp toggle, sparse-extreme "worse" bins eliminated (0 worse). Coverage kept method-internal (per-run
CSV), not surfaced to the leaderboard (user decision). Per-bin balance **deferred** — floor cleared the
done-when, so adopt-only-if-it-helps left it unbuilt (follow-up). Defaults promoted to the class; grid
1–12 months both modes with fresh naive; benchmark regenerated. Note: the frozen 30-June reference dir is
stale (its v0 needs regenerating on current code — see F17).*

---

## Issue 15 — The headline estimator configures itself: regime-adaptive training window and estimator (WS2)

**Goal:** the method must "just work" on any campaign with **no manual configuration** —
a user should not need to know when to flip `toggle_campaign_only`, pick a
`toggle_estimator` or tune a decay half-life. F16 measured why this matters: the best
configuration is regime-dependent, with a crossover near ~3 months.

**The measured regime map (F16, `inspect_short_campaigns.py` + the benchmark grid):**
- Toggle training window: campaign-only wins at 1–2 months (the campaign is <5 % of the
  all-data training set, so drift dominates; 1-month score 0.334 vs 0.895), all-data wins
  at ≥3 months (shrinkage dominates; 3-month 0.294 vs 0.359).
- `double_ratio` (the model-normalised naive comparison): toggle placebo headline bias
  ≈ 0 at every 3–12-month length, but *worse* than the default at 1–2 months — its
  `rho_off` is currently measured over all off rows (two years of eras) while the ON
  window is a sliver, so the cancellation subtracts the wrong era's miscalibration.
- Decay half-life: ~90 days is best-or-near-best at 1/2/3 months in both modes; ≥1 year
  is the safe long-campaign default (548 d accepted in F16).

**Scope (in preference order — a dominating single configuration beats selection logic)**
- **Era-local `rho_off` first.** Fix `double_ratio` itself: measure `rho_off` on
  campaign-window off rows only (out-of-fold), predicted by the same all-data-trained
  fold models. If that removes the 1–2-month failure while keeping the ≥3-month bias ≈ 0,
  one estimator may dominate everywhere and become the toggle default outright — no
  selection machinery needed. Test at 1, 2, 3, 6, 12 months in both framings.
- **Adaptive training window / half-life as fallback.** If no single configuration
  dominates: derive a blended weight from the campaign's own observables
  — campaign length, campaign-row count vs pre-campaign-row count — never from the
  benchmark truth. Prefer a smooth blend (e.g. weight the campaign-only and all-data
  estimates by a logistic in log(campaign rows)) over a hard threshold, so estimates
  don't jump discontinuously at a boundary; hard-validate whatever rule is chosen at the
  boundary lengths (2–4 months).
- **Guard against benchmark overfitting** (the Issue 12 framing rule): the rule's
  parameters must be justified by the mechanism (drift-vs-shrinkage trade-off), tuned at
  most coarsely, and confirmed on profiles/turbines not used to pick them.
- **Evaluation surface:** extend the scored campaign grid to include 1- and 2-month
  campaigns for the adaptive-rule validation (either promote them into the committed
  benchmark — needs fresh naive/v0 reference runs — or keep `inspect_short_campaigns.py`
  as the reproducible driver for that regime, citing it in the findings entry).
- Remove the user-facing knobs this makes redundant (or demote them to expert overrides
  with the automatic behaviour as the default).

**Done when:** the out-of-the-box configuration is best-or-tied (within the materiality
band) at every campaign length 1–12 months in both modes on the placebo and standard
profiles; no scenario requires the user to choose a training window, estimator or
half-life; the findings entry records the mechanism-based justification for the rule (or
the dominance of the single configuration); benchmark regenerated.

---

## Issue 16 — Prune the historic opt-ins: orient `power_model` to the winning configuration (power_model)

**Goal:** the accumulated development knobs on `PowerModelMethod` are almost all opt-ins that
**lost their A/B and are off in the shipped default**. Remove the dead ones so the code presents
only the successful approach — smaller surface, less overfit temptation, easier to reason about.
This is code hygiene, not a methodology change: the default configuration is unchanged, so the
committed benchmark must come out **bit-identical** (the acceptance test). Also finishes the
Issue 15 knob cleanup.

**The removal set (each off/absent in the default; findings verdict in brackets)**
- `calibrate_slope` [F14 — toggle overcorrects; OOF-vs-final shrinkage gap is structural],
  `calibrate_residuals` [F15 — OOF residuals don't transfer to a refit model in level or shape].
- `early_stopping` [F14 — neutral even in its target 3mo-toggle regime, +15% runtime],
  `n_seed_ensemble` [F14 — spread is weather-sampling not seed noise, +75% runtime].
- `toggle_estimator="double_ratio"` **and** the temporary `rho_off_scope` knob [F16/F19 — never
  beats the counterfactual default on score; boxed between "nothing to correct" (all-data →
  `rho_off`≈1) and "too noisy to correct" (campaign-only)]. Removing `double_ratio` makes
  `toggle_estimator` a single-value knob → drop it entirely; the counterfactual is the sole
  headline estimator.
- `time_features` (+ `latitude`, `longitude`) [F11 — all rejected: prepost drift import, calendar
  proxy], `era5_derivations` (+ `hub_height_m`) [F10 — all rejected *as model features*].

**Keep (with rationale, so it isn't re-litigated)**
- The **injectable `model_factory` seam** (roadmap Phase-2 learners) *may* stay, but drop the dead
  `hgb`/`linear` registry entries and the calibration/early-stop/seed plumbing in `fitting.py`,
  keeping only the shared `time_block_folds`. Decide seam-stays-vs-goes on whether any driver needs
  it; if nothing uses it, remove it too.
- **`era5_derived.py` and `time_features.py` stay as utilities** — CEM matching / the inspection
  scripts use the derivations even though the *feature knobs* are removed. Only the model-feature
  wiring and the method-surface knobs go.
- Settled defaults stay: `era5_exclude=CURATED_ERA5_EXCLUDE` (F13), `availability_feature=False`
  (F13), `reference_stat_cols` (schema), `matching_vars`/`matching_bin_edges` (F6), the adaptive
  time-decay default and its `time_decay_half_life_days` expert override (F20).

**`toggle_campaign_only` — confirm then likely remove.** The adaptive half-life (F20) is a *soft*
campaign-only at short campaigns, so `toggle_campaign_only=True` is probably now redundant for
`power_model`. One cheap A/B (adaptive vs adaptive+campaign-only, toggle 1–3mo, both placebo and a
recovery profile) settles it; remove the knob if redundant, keep with fresh evidence if not.
(`naive_ratio` keeps its own campaign-only restriction — a different method, untouched.)

**Scope**
- Delete the dead knobs, their config-validation branches, their `_config_params` entries, their
  plumbing, and their unit tests; thin `fitting.py`; update any driver/inspection script that set a
  removed knob (e.g. `inspect_short_campaigns.py`'s `double_ratio*` arms).
- Re-run `study_power_model_compare.py` on the default config and assert the benchmark is
  bit-identical (no `--update-baseline` needed — nothing should change).

**Done when:** the removed knobs are gone with their tests; `poe all-fast` green; a default-config
run of `study_power_model_compare.py` reproduces the committed benchmark bit-identically; a short
findings entry records the pruned set and the `toggle_campaign_only` verdict.

---

## Issue 17 — Make the conditional estimator consistent with the headline (power_model)

**Goal:** the per-bin conditional uplift and the overall headline currently use **different data and
weighting**, an asymmetry that grew up historically (F8/F15/F16) and is now partly obsolete. Reconcile
them — either unify the data/weighting path, or keep the asymmetry only where fresh evidence still
demands it and document why.

**The current asymmetry (`method.py:_estimate_conditional`)**
| aspect | headline (overall) | conditional (per-bin) |
| --- | --- | --- |
| training data | **all** baseline | campaign-window rows **only** (F15) |
| recency weighting | **adaptive half-life** (F20) | **none** (F16) |
| estimator | single full-window fit | two-direction CEM-matched cross-prediction, re-levelled to the headline |

The re-level (F8/F14) keeps the *aggregate* consistent; the *inputs* diverge.

**Why this is an investigation, not a one-liner**
- **The missing half-life is downstream of the campaign-only matching.** The conditional already
  excludes pre-campaign rows, so a decay weight has little to act on — adding the half-life is only
  meaningful *if* the conditional also starts using pre-campaign data like the headline.
- **F15 localised the all-data damage to the *matching*, not the fits** (matching pre-campaign rows
  against campaign on-rows reads reference/era drift as per-bin uplift). So naive unification
  (feed it all-data + decay) is known to fail; genuine unification needs *weighted or restricted*
  matching — design work.
- **But the narrow version is now unblocked:** Issue 14 deferred lifting the "weights are
  headline-fit-only" restriction until the per-bin count floor existed ("the floor should make the
  conditional path robust enough to lift that restriction"); the floor shipped in **F17**. So
  re-trialling recency weighting on the conditional path is ready.

**Scope (per the Issue 9 A/B protocol, one change at a time)**
- **Re-trial recency weighting on the conditional fits** now the F17 count floor is in place — the
  specific F14/F16-flagged item. Watch the sparse extreme-TI/ws tail bins that destabilised before
  the floor.
- **Trial unifying the training window**: let the conditional use all-data with the adaptive half-life
  *and* an appropriately weighted/restricted CEM match, vs today's campaign-only match. The win
  condition is the F15 drift bias staying cancelled while the extra data cuts per-bin spread.
- **Preserve the exact energy-aggregation identity** (measured + pinned-imputed bins aggregate to the
  headline) throughout — it is the one consistency property already correct and must not regress.
- Evaluate on the conditional benchmark (`conditional_benchmark_comparison`) across profiles and
  campaign lengths; gate on conditional |bias| no worse (ideally better) and overall P50 unchanged.

**Done when:** either the conditional and headline share one data/weighting path (with the energy
identity intact and conditional |bias| no worse), **or** each surviving asymmetry is deliberately
documented with fresh post-floor evidence for why it must remain; benchmark regenerated if any default
changes; findings entry records the verdict.
Campaign curve plots for the winning approach are re-generated for user inspection.

---

## Issue 18 — AEP uplift: long-term extrapolation design + harness scoring (WS1/WS4)

**Goal:** turn the campaign conditional uplift into an **AEP uplift** — the long-term
annual energy benefit, PR #100's third metric — and score it in the harness against known
ground truth. The core idea: choose the condition axes from the nature of the upgrade
(wind speed only for most upgrades; add wind direction etc. for complex upgrades like
wake steering), estimate the **long-term MWh distribution** over those condition bins,
and take the energy-weighted sum of the conditional uplift:
`AEP uplift = Σ_b E_b^LT · (1+u_b) / Σ_b E_b^LT − 1`.

**Scope**
- **Upgrade-nature-driven condition axes.** A config on the method/driver naming the
  binning axes for extrapolation (default `("ws",)`; wake steering `("ws", "wd")`, …).
  Generalise the Issue 8 conditional machinery to produce `u_b` on those axes (today it
  emits ws and TI *marginals*; direction and joint axes need the same treatment).
- **Long-term energy distribution `E_b^LT`.** Two candidate sources, decided in design:
  (a) long on-site SCADA history binned on measured conditions (simple, but needs years
  of history and stationary sensors); (b) long-term ERA5 (10–20 y) driven through the
  campaign-learned relation between ERA5 and the test turbine's conditions/energy — e.g.
  the baseline counterfactual model evaluated over the long-term ERA5 record, or an
  ERA5-cell occupancy map × per-cell mean energy from the campaign. (b) is preferred
  (works for any site, matches the WS4 density-ratio framing).
- **The axis-consistency question (design decision).** Conditional uplift is binned on
  test-measured ws/TI, but the long-term record is ERA5 — either learn the ERA5→test-axis
  transfer on the campaign overlap, or run the AEP path end-to-end on the ERA5 axis
  (matching, uplift bins and long-term weights all on the same treatment-invariant axis).
  Sketch both in a short design note section before coding; the ERA5-axis route avoids a
  post-treatment reporting axis entirely and is likely cleaner for AEP.
- **Coverage fallback.** Long-term bins with no measured `u_b` (or floored by Issue 14)
  take Issue 14's imputed values (bfill-then-0 on the ws axis — the 0-at-rated prior
  matters even more here, since long-term energy concentrates in high-ws bins that an
  overall-uplift fallback would over-credit); report the coverage fraction (share of
  long-term energy in bins with a measured uplift) as a headline diagnostic.
- **Harness truth + scoring.** The generator knows the true uplift function, so the true
  AEP uplift per replicate is the injected profile applied over the **full multi-year
  dataset** (not just the campaign window); score `estimate − truth` with the same
  bias/spread/score metrics and campaign-length sweep. Include a naive extrapolation
  baseline (`AEP uplift = campaign overall uplift`) — the bar to beat, which only a
  condition-dependent profile can separate from the real thing.
- **A direction-dependent (wake-steering-shaped) profile** in the generator, so the
  "bin by direction" path is actually exercised (listed in Issue 1, never implemented).

**Done when:** the design decision (axis strategy) is recorded; the harness emits AEP
truth and scores; `power_model` produces an AEP estimate on at least the ws axis and
beats the naive extrapolation baseline on condition-dependent profiles.

---

## Issue 19 — Uncertainty: campaign & AEP P50 uncertainty with harness coverage scoring (WS4)

> **Partly delivered ahead of this issue, for `toggle_specialist` only (2026-07-15).** A
> toggle-first slice built the machinery this issue describes: the additive seam extension
> (`MethodOutput.sigma_overall` / `uncertainty_diagnostics`), the circular block bootstrap
> (`benchmarking/baselines/block_bootstrap.py`), harness calibration scoring
> (`benchmarking/harness/calibration.py`), and a 64-replicate coverage study
> (`study_toggle_specialist_uncertainty.py`). `toggle_specialist` now reports a **non-optional**
> 1σ on the headline and every power bin. **Read F28/F29 in `findings.md` before planning the
> rest** — they change three assumptions written below:
>
> - **"block length ≥ the residual autocorrelation scale, likely ~1 day" is wrong for a fast
>   toggle.** The *paired* on/off residual decorrelates in ~1-3h (autocorr 0.003 at 48h), so a
>   day-scale block captures nothing and biases σ **low** via circular-overlap collapse. There is
>   no σ-vs-L plateau to read; block length must be picked by coverage against truth.
> - **The scored quantity should be 1σ coverage (~68.3%), not only P95.** A one-sided P95 check
>   is a weak instrument: it tests one tail at 95%, so it needs ~4x the ensemble to resolve the
>   same miscalibration, and it cannot distinguish a scale error from a shape error. The measured
>   errors are **platykurtic** (kurtosis -0.35..-0.76), which means P95 ≠ P50 − 1.645σ even when σ
>   is correctly scaled — so the normality shortcut below must be checked, not assumed.
> - **"Validate first on the placebo" buys little.** The three profiles' errors correlate
>   0.977-0.995, so the placebo is not an independent test; replicates are the only real evidence
>   axis, and the ensemble must be sized on them (n=4 produced a phantom -0.7 pp bias that
>   vanished at n=64).
>
> Still open here: the *cheap vs full* bootstrap comparison, `power_model` uncertainty, AEP σ, and
> P95 itself.

**Goal:** start Phase 3 — report an uncertainty (σ / P95) on the campaign overall P50 and
the AEP P50, and verify it in the harness per PR #100: the campaign-uplift P95 should sit
below the true uplift ~95% of the time. (P50 accuracy work stays the priority; this issue
establishes the machinery and a first honest number, not the final uncertainty model.)

**Scope**
- **Method-side estimator: circular block bootstrap** over time blocks of the baseline
  and upgraded rows (block length ≥ the residual autocorrelation scale, likely ~1 day).
  Two variants to compare on a small grid: *cheap* (fix the fitted model, resample the
  (actual, counterfactual) pairs → distribution of the energy ratio) and *full* (refit the
  model per resample — captures model-fit noise, ~50–100× the cost). Emit σ and empirical
  quantiles; P95 = P50 − 1.645σ under normality or the empirical quantile, whichever the
  data supports.
- **Seam extension (additive).** Optional uncertainty fields on `MethodOutput`
  (e.g. `sigma_overall`, `p95_overall`, per-bin σ) — `None` for methods that don't emit
  them, no breaking change to existing methods.
- **Harness scoring.** Coverage = fraction of (replicate × campaign) cases with
  `truth ≥ P95` (target 0.95), plus a calibration read at 2–3 more quantiles using the
  replicate ensemble, plus mean interval width — so a method cannot win coverage by
  inflating σ. Validate first on the placebo (truth 0), where miscalibration is most
  visible.
- **Why not quantile regression for P95:** per-row predictive quantiles describe
  single-timestamp scatter and do not aggregate into a quantile of the campaign *ratio*
  without independence assumptions the autocorrelated 10-min data violates — hence the
  block bootstrap. Quantile models stay on the WS4 list for conformal OOD filtering and
  diagnostics.
- **AEP P95.** Combine the campaign sampling uncertainty (bootstrap above) with the
  long-term-distribution uncertainty (resample ERA5 *years* to get the inter-annual
  variability of `E_b^LT`). Document explicitly what is in and out of scope of the
  reported number (e.g. model-form error and sensor drift are out, for now).
- Cross-check the block bootstrap design against v0's existing bootstrap (`wind_up`
  uncertainty machinery) — same idea, method-appropriate implementation; no v0 import.

**Done when:** `power_model` emits σ/P95 for campaign and AEP uplift; the harness reports
coverage and interval width per profile × campaign length; observed coverage is within an
agreed tolerance of nominal on the placebo and the standard profiles.

---

## Not in the first wave (tracked for later phases)

- Uncertainty / P95 model beyond Issue 19's first cut: conformal OOD filtering,
  density-ratio long-term weighting in place of hard CEM subsampling (WS4, Phase 3).
  Density-ratio weighting also reclaims the short-campaign matched-set size the CEM
  subsample throws away (F7 follow-up).
- Further candidate methods: DSWE / `funGP`, Astolfi multivariate-linear (WS2,
  Phase 2).
- Conditional/heterogeneous uplift reporting & SHAP story beyond ws/TI/direction
  bins (G3, Phase 2).
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
- ~~**Derived ERA5 atmospheric features.**~~ Promoted into **Issue 9** (air density, shear
  exponent, veer, stability indicators as a shared ERA5-derivation utility).
