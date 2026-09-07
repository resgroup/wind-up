# wind-up v1 — campaign findings log

Empirical findings from the realistic whole-farm campaigns tranche
([issues_campaigns.md](issues_campaigns.md)). Newest first. Each entry records what was
observed, the evidence, the root cause, and what it implies for the method design or the
issues list.

Entries are numbered **CF*n*** — a separate series from [findings.md](findings.md)'s
**F*n***, which belongs to the earlier, back-burnered effort.

Keep entries reproducible: name the driver and the exact configuration, not just conclusions.

---

## CF13 — The reference screen works where it has data and references, and nowhere else: it needs a campaign of **150+ days** and it finds Hill of Towie's genuinely bad turbine (**T17, +4.7%**) on a 19-reference pool. Two constraints, both measured, decide when it may run at all

*2026-09-04. R3, Phase B. Drivers: `benchmarking.campaigns.screen_calibration` (clean placebo
pools, one screening pass at an infinite floor, so every deviation is the farm's own noise),
`benchmarking.campaigns.reference_fixture` (injected changes) and
`benchmarking.campaigns.screen_roadtest` (four real reference sets). The screen estimates each
candidate reference as if it were a test turbine against the others and rules out clear outliers
from the pool median, worst one per pass; a ruled-out reference is made power-free rather than
removed.*

**The screen found a real bad reference nobody was looking for.** On clean Hill of Towie data with
nothing injected, **T17 reads +4.67%** against a 19-reference pool whose median is +0.26% and whose
next-worst member is 1.49 pp out. It reproduces independently on an unrelated 5-reference pool
(+4.74%). Checked directly against the SCADA, T17's median power ratio to T09/T12/T14/T16 steps
from **0.735 across 2017 to 0.804 across 2018**, with two collapsed months in 2017 (Aug 0.44, Nov
0.49 against a ~0.75 norm). Its low absolute ratio is siting; the *step* is the finding. T17 is not
in the benchmark turbine set (`DEFAULT_WTG_NUMBERS = [1, 3, 4, 7]`).

**Constraint 1 — the campaign must be long enough, and this is not a floor problem.** Running the
frozen benchmark sweep with the screen on removed references it should not have. The removals were
not marginal: one read T04 at **-5.902%, 8.02 pp from its pool median** against a 2.5 pp floor,
with T01 at +2.12% and T03 at +4.24% — a 10 pp spread across three references on a placebo. That
case held ~4167 upgraded records, about 29 days. The worst clean deviation by campaign length,
across four treatment-start windows:

| campaign | 1 mo | 2 mo | 3 mo | 6 mo | 12 mo |
|---|---|---|---|---|---|
| worst deviation | **2.87 pp** | **2.47 pp** | 1.51 pp | 1.43 pp | 0.95 pp |

At 1-2 months the clean spread reaches the floor, so **no floor separates a bad reference from a
good one there**. So the gate is a minimum-data rule rather than a different floor: a campaign too
short to screen is not screened at all, rather than screened badly.

These four windows were not enough to set the number. A 90-day gate still false-positived on
3-month campaigns in the sweep, which samples far more windows and found 2.9-3.25 pp there against
the 1.51 pp these four suggested. **The shipped value is `screen_min_campaign_days = 150`**, set by
the sweep -- see the benchmark section below. The lesson is that a hand-picked window sample is a
weaker calibration instrument than the sweep itself.

**Constraint 2 — the floor is set by clean spread, not by sensitivity.** Clean placebo pools spread
up to 1.19 pp in prepost with nothing injected, so `screen_floor = 0.025` leaves about a factor of
two. An earlier 1 pp placeholder fired on healthy turbines in three separate places, and on the
clean fixture cell that false positive moved the estimate **0.88 pp** — the screen made a good
campaign worse. At 2.5 pp the clean cells are unchanged to five decimal places.

**Two bad references halve the signal.** One reference at -4% reads 3.48 pp from the median; two at
-4% read 1.81 pp, because they drag the median toward themselves. So the majority-vote design
weakens as more references go bad, quantifiably, and detecting two together takes roughly double
the magnitude of detecting one.

**Toggle is not screened.** Two contrasts were measured and both fail. The campaign's own toggle
schedule is blind by construction — a reference is not the thing being toggled, so its change sits
in the on- and off-blocks alike (injected arms read 0.73-1.27 pp against a 0.77 pp clean baseline,
and the ranking named the wrong turbine). A mid-campaign prepost split sees a change but cannot
attribute it: each side holds ~3 months of a 6-month test in different seasons, clean noise rises
to 2.0-3.9 pp, and the five-reference injected arm read *below* its own clean baseline. Since
ruling out a good reference costs more than leaving a mild bad one in, the screen does not run in
toggle. **`power_model` remains exposed there anyway** — its fit reaches into the pre-campaign
baseline under `adaptive_time_decay`, which straddles a boundary change and reintroduces a bias the
toggle contrast would otherwise cancel. That is a fit-window exposure; closing it re-opens the
`toggle_campaign_only` knob Issue 16 pruned, so it is left open deliberately.

**Cost.** Screening plus the post-screen reference-uplift pass adds ~2N model fits per estimate. On
the 21-turbine farm that is ~30 minutes per test turbine, and it made the frozen benchmark sweep
roughly 12x slower. **Settled: `report_reference_uplifts` defaults to True and every sweep driver
passes False.** A sweep scores estimators rather than reporting campaigns, and the report is an
output, not an input -- turning it off moves no estimate. The screen itself still runs in a sweep,
since that does change estimates.

**The remediation A/B: the screen earns its place, the waking boolean does not.** Movement from
each pool's clean cell on the prepost fixture arms, truth zero, so smaller is better [pp]:

| remediation | T15 +5% | T15 -4% | two at -8% | mean | worst |
|---|---|---|---|---|---|
| no screen | 1.185 | 1.137 | 3.198 | 1.840 | 3.198 |
| drop | 0.376 | 0.276 | 1.332 | 0.661 | 1.332 |
| **direction** | 0.314 | 0.288 | 1.272 | **0.625** | **1.272** |
| direction + waking | **0.197** | 0.398 | 1.408 | 0.668 | 1.408 |

Screening cuts mean error from **1.84 pp to 0.63 pp**, a 66% reduction, and recovers up to 3.2 pp
on the two-bad case. Every arm leaves the clean cells byte-identical (+0.343% / +0.453%), so
nothing is paid for on a healthy campaign and the choice of remediation cannot move the frozen
benchmarks, which screen nobody.

**Keeping the ruled-out reference's direction beats dropping it outright** on mean and worst case:
its wake geometry is uncorrupted by a performance change and worth retaining, which is the point of
making a bad reference power-free rather than removing it.

**The waking boolean is kept, and the A/B is why it *can* be.** It is marginally behind on this
fixture -- best on one arm, worst on the other two -- but the three remediations sit within
**0.04 pp** of each other (0.625 / 0.661 / 0.668) across three arms on one farm and one test
turbine. That is not evidence it costs anything; it is evidence the fixture cannot separate them.
Against that, it carries information the fixture does not exercise: a reference that stops
producing entirely still tells the model it has stopped waking its neighbours, which neither
`drop` nor direction-only can express once the power channel is gone. The decision was taken on
that basis -- no measurable cost, a clear physical argument -- and the other two options were
pruned rather than left as dead knobs.

**The road test: the calibration transfers to farms it never saw.** Four reference sets, placebo
campaigns, truth exactly zero, screen on:

| farm | references | test uplift | reference overall | screened |
|---|---|---|---|---|
| Hill of Towie, whole farm | 19 | +0.44% / +0.42% | +0.357% | **T17** |
| Hill of Towie west (T01-T15) | 13 | +0.46% / +0.39% | +0.385% | none |
| Kelmarsh | 4 | -0.02% / +0.30% | +0.254% | none |
| Penmanshiel | 12 | -0.23% / +0.10% | -0.257% | none |

The floor was calibrated on Hill of Towie alone, and it fires on the one farm with a genuinely bad
turbine while staying silent on three that do not, including two farms it had never seen. The
reference overall uplift reads -0.26% to +0.39% throughout, so the sanity check behaves everywhere.
`hot_west` is the control: it excludes T17 and screens nobody while giving essentially the same
estimate as the whole farm.

**The frozen benchmarks are unchanged.** Zero references ruled out across the sweep, zero MOVED
verdicts, and every **overall** mean-over-profiles delta zero in both modes at every campaign
length. The largest movement anywhere is **0.013 pp**, in the degenerate 2-month wind-speed
conditional cell whose own score is 14.5 pp -- 0.09% relative, inside the sweep's +/-0.1 pp neutral
band. It is deterministic (identical across two runs), it predates the PR-140 review fixes, and it
is not the positional feature renaming, which was checked directly and gives bit-identical
predictions. It is consistent with the known LightGBM thread-order reproduction floor, which
earlier work put at ~0.7 pp cross-machine. Worth stating precisely rather than rounding to "zero":
the headline P50 cells are exactly unchanged, the degenerate conditional tail is not quite. Getting
there took two goes: at a 90-day gate the sweep still false-positived on 3-month campaigns (a
reference read -2.9%, 3.1 pp from its pool median, and ruling it out moved prepost overall score
**+0.103 pp and spread +0.126 pp the wrong way**). The same three-reference pool is clean at 6 and
12 months, where the screen runs and correctly finds nothing, so campaign length is the driver
rather than pool size, and the gate went to 150 days. The sweep is the better calibration
instrument than a hand-picked set of windows: four sampled windows had put 3-month campaigns at
1.51 pp worst, and the sweep's wider sampling found 2.9-3.25 pp.

**The power-minimum artefact, closed.** `ReferenceCpChange` originally moved a reference's mean
power while leaving `active_power_min` — an unconditional model feature — at its pre-change value,
an impossible channel mismatch the screen might have keyed on instead of the Cp shift. It now
scales positive minima with the mean and leaves negative ones alone (a negative minimum is
parasitic draw, which a Cp change does not scale). Re-running the fixture with it fixed leaves the
**screened** results identical to three decimal places (+0.197 / +0.398 / +1.408 pp) and the same
turbines detected, while the unscreened bite strengthens ~4% (-1.185 -> -1.233, +1.137 -> +1.181,
+3.198 -> +3.312) now that the minimum carries the change too. There is a structural reason the
screened numbers cannot move: a ruled-out reference loses its power columns, mean and minimum
alike, so the consistency of a channel that is no longer in the matrix cannot matter. The artefact
could only ever have affected detection, and detection is unchanged.

**Four bugs the runs caught that the unit tests did not.** The `waking` feature was bool, which
collapses to object dtype once reindexed and the outcome model rejects (now tri-state float, with
a missing record left unknown rather than asserted not-waking). The screen kept screening a
two-reference pool after a drop, where deviation-from-median is degenerate and it flagged one
arbitrarily. The floor placeholder fired on healthy turbines. And the reference-uplift pass
recursed through its own clones once it gained its own flag — the clones had been stopped by the
screen's flag by accident — turning a 39-minute road test into one still running after nine hours.
Each was found by a real run, not a fixture: the toy pools are too small for the recursion to bite
and the sweep disables the pass entirely, so only a large-pool run with reporting on could show it.

---

## CF12 — An undeclared reference upgrade bites `power_model` hard in prepost (−0.69 to −1.14 pp from a 3% Cp change) and, unexpectedly, **also in toggle** (−0.31 to −0.52 pp) — toggle's alternation cancels it for `naive_ratio` but not for `power_model`, whose fit still reaches into the pre-campaign baseline

*2026-09-04. R3, Phase A (the bite; the screen is not built yet). Driver:
`benchmarking.campaigns.reference_fixture` — T06 declared through `placebo_campaign` (no upgrade
injected, so truth is exactly 0), 12-month baseline plus 12 months prepost / 6 months toggle.
Five arms x both modes: a 3-reference pool (T15/T10/T08, the CF5 set) clean, with T15 at +3% Cp
and with T15 at −3% Cp; and a 5-reference pool (+T04/T02) clean and with both T15 and T10 at +3%.
Every change is a `ReferenceCpChange` landing at the campaign changeover and staying on — a
reference retrofitted in the same programme as the test turbine. Movement is measured against the
clean cell of the same pool size, since a 3-reference and a 5-reference estimate differ in their
own right (prepost clean: +0.343% vs +0.453%).*

**Observed — it bites in every arm, in both modes.** `power_model` headline movement, against a
0.25 pp materiality bar:

| pool | arm | prepost | toggle |
|---|---|---|---|
| 3 refs | T15 +3% | 0.343% → −0.350% (**−0.692 pp**) | 0.082% → −0.227% (**−0.309 pp**) |
| 3 refs | T15 −3% | 0.343% → +1.174% (**+0.831 pp**) | 0.082% → +0.431% (**+0.349 pp**) |
| 5 refs | T15+T10 +3% | 0.453% → −0.691% (**−1.144 pp**) | 0.358% → −0.165% (**−0.523 pp**) |

Signs are as expected and roughly symmetric: an improving reference raises the counterfactual and
drags the estimate down; a degrading one lifts it. 3% was enough on the first attempt, so no
escalation was needed. Two bad references out of five moved the estimate *more* than one out of
three (−1.14 vs −0.69 pp) despite a similar poisoned fraction (0.40 vs 0.33).

**The surprise is toggle.** R2's lesson, and the prior expectation recorded in the design, was
that toggle's rapid alternation cancels a fault present in both periods. It does — for
`naive_ratio`, which moved **0.006 pp** in toggle against 0.68 pp in prepost. It does not for
`power_model`, which retains ~45% of its prepost movement.

**Root cause of the toggle residual.** `naive_ratio` calls `restrict_to_campaign`, which drops the
pre-campaign rows, so both sides of its ratio sit entirely after the step and the reference's new
performance is common-mode. `power_model` has no such restriction: under `adaptive_time_decay` its
toggle fit still draws on the pre-campaign baseline, which straddles the step, so the model learns
a reference relationship partly from before the change and applies it after. That is the only
asymmetry between the two methods here, and it predicts the sign and rough size of what is left.

**`toggle_specialist` is structurally identical to `naive_ratio` on a headline P50.** Both compute
`rho_up / rho_base − 1` with `rho = Σ test_power / Σ ref_total`, and their `_used_mask`
implementations are line-for-line the same (`NormalOperationFilter`, `apply_stuck_filter=False`,
same completeness and availability terms); `toggle_specialist` adds only `& ~self._test_excluded(…)`,
a no-op under a schema with no `exclude_row` role. Their toggle estimates agree bit-for-bit here
and in the R2 run (CF11), which is expected rather than a defect — `toggle_specialist`'s distinct
value is its block-bootstrap sigma, its per-power-bin conditional frame and that exclusion filter,
none of which a headline comparison exercises. It also settles R3's scope question: at 0.006 pp,
`toggle_specialist` needs no defence against this failure mode.

**Implications.**
- The R-series "fault must bite" gate is cleared **per mode**, so the Phase B screen must be
  applied in both, not prepost only.
- Before building the screen, test the cheaper rival hypothesis the root cause suggests: if the
  toggle residual is purely the pre-campaign baseline in the fit, restricting `power_model`'s
  toggle fit to campaign rows removes it with no screen at all. Note this re-opens the
  `toggle_campaign_only` knob pruned as dead in Issue 16, so it needs its own evidence.
- The screen's decision rule must be **relative** (outlier against the pack), not a test against
  zero: screening each reference against a pool that still contains the bad one gives every good
  reference a common negative offset.

**Known artefact to watch.** `apply_upgrades` writes back `active_power`, `gen_rpm` and
`wind_speed` only, so a reference whose mean power moves 3% keeps its original `active_power_min`,
which `power_model` carries as a per-reference feature. Every *declared* upgrade behaves the same
way, so the fixture stays internally comparable — but a Phase B screen that turns out to key on
the mean/min inconsistency is detecting an artefact, not the performance change.

---

## CF11 — An unstable anemometer costs `power_model`'s headline almost nothing as it ships (max 0.21 pp, and exactly zero in toggle), because the standing "no reference-anemometer features" rule already closes the pathway — turning that rule on is worth **34 pp** of error

*2026-09-04. R2. Driver: `benchmarking.campaigns.sensor_fixture` — T06 plus T15/T10/T08,
declared through `placebo_campaign` (no upgrade injected, so truth is exactly 0), 12-month
baseline plus 12 months prepost / 6 months toggle. 28 arms: two fault shapes
(`SensorGainStep` at the changeover, `SensorGainDrift` ramping over the whole record) x gains
x1.5 and x0.5 x two targets (the test turbine T06, the nearest reference T15) x both modes,
plus an exposed set carrying reference anemometry as model features. Both shapes scale
`wind_speed` and `wind_speed_sd` together, so turbulence intensity is invariant by construction
and only the wind-speed axis moves.*

**Observed — the headline is effectively immune.** In the shipped configuration, seven of the
eight prepost fault arms moved `power_model` by at most 1.1e-5 pp. The single exception is a
**x0.5 gain on the reference T15**, which moved the farm estimate 0.343% -> 0.557%, **+0.214 pp**
— step and drift alike, to six decimal places. Every toggle arm moved by 1e-13 or less, i.e.
exactly zero. `naive_ratio` and `toggle_specialist` were unmoved everywhere, as expected: they
read power only.

**Root cause of the one nonzero result.** The only surviving path from reference anemometry into
`power_model` is `reference_mean_wind_speed`, which the ERA5 lag sweep locks onto. Measuring
`sync_era5(...).best_lag_rows` per arm: clean **-3**, `step_x0.5_T15` **-4**, `drift_x0.5_T15`
**-4**, `step_x1.5_T15` -3, `drift_x1.5_T15` -3. The argmax tips by one row — a **10-minute**
shift in the reanalysis alignment — which is why the step and the drift give an identical
estimate: both select the same alternative lag. Both T06 arms returned lag -3 at correlation
0.87001, bit-identical to clean, confirming the test turbine's anemometer never enters the
reference side at all.

**The conditional grid moves hard, as expected.** The `ws` conditional axis *is* the test
turbine's own anemometer, so a gain fault re-bins every row: 42 of 104 prepost `ws` cells and 28
of 104 toggle cells moved materially, up to **+600 pp** in the degenerate (2, 4] m/s bin (clean
-12.9% -> +586.7%). The `ti` axis is invariant in toggle (1e-13), as the mean-and-SD-together
design intends; its prepost movement (max 66 pp, in the degenerate (0.45, 0.5] bin) rides the
T15/ERA5 route rather than re-binning. The `power` axis moved at most 0.59 pp prepost and zero
in toggle. **Deliberately not fixed:** the conditional machinery is nascent, and R2 was scoped to
measure rather than mature it.

**The exposed arm prices the standing rule.** Repeating the clean cell and the steps with
reference anemometry carried as features (`reference_stat_cols=(wind_speed, wind_speed_sd)`):
the clean prepost estimate alone degrades **0.343% -> -1.169%**, and the faults then move it
**-23.07 pp** (x1.5 on T15) and **+34.32 pp** (x0.5 on T15). In toggle the same arms move -0.77
pp and +0.61 pp. The test-turbine target stays at ~0 throughout, since the test turbine
contributes no reference features. So the standing exclusion of
reference anemometry is no longer a stance on principle: it is worth up to 34 percentage points
of error under a reference calibration fault, and it improves clean data too.

**Toggle cancels almost all of it**, confirming the R-series prediction: the same fault that
costs 23-34 pp in prepost costs 0.6-0.8 pp in toggle, and in the shipped configuration toggle is
unmoved outright.

**Implications.**
- **R2 lands no `power_model` change.** The invariance target is already met by construction;
  there is nothing to fix in the headline path.
- The **ERA5 lag sync is the one remaining reference-anemometry pathway**. It is worth 0.21 pp,
  prepost only, and only at a 50% calibration error. If it ever matters, the fix is to lock the
  sweep onto reference *power* rather than reference wind speed.
- The **conditional `ws` axis is built on a post-treatment, drift-prone sensor**. Whoever matures
  the conditional result should read this first; the fault classes are in place to re-measure.
- **Temperature was dropped from scope** by inspection, not measurement: `ambient_temp` is a
  diagnostics-only schema role that reaches `power_model` through nothing, and ERA5 supplies
  `temperature_2m` independently. There is no pathway for it to bite.

## CF10 — A low-effort `NorthingSettings` tier was measured and dropped: the changepoint search is a small part of the runtime (a whole farm-year differs by ~2 seconds) and a smaller changepoint budget cost real detections, so there is one setting rather than a menu

*2026-09-04. Recorded when the justification was removed from the `NorthingSettings` docstring
under the `src/` "behaviour, not justification" rule; the measurement itself was made during R1.*

**Observed.** A reduced tier (smaller `changepoints_per_year`, coarser `grid`) was built and run
against the same cases as the default. It saved ~2 seconds on a whole farm-year — the search is
not where the runtime goes — and it lost genuine detections, because the changepoint budget is
what lets a long record hold every recalibration it actually contains.

**Decision.** `NorthingSettings` ships as a single default, `DEFAULT_NORTHING`. Construct one
only to tune deliberately; there is no tier to choose between. The two derived settings that do
exist — `anchoring_only` and `against_reanalysis` — are not tiers: each raises `min_step_deg`
for a reference that cannot support finer attribution.

## CF9 — R1 improved `power_model` on the real placebo in both modes: mean per-turbine error 0.515% → 0.312% prepost and 0.328% → 0.255% toggle, with `naive_ratio` and `toggle_specialist` unmoved to three decimals, so the gain is attributable to the direction feature and discovered northing alone

*2026-09-04. Reproduce: `uv run python -m benchmarking.campaigns.placebo`, Hill of Towie, both
modes, defaults (six upgraded turbines T07/T11/T12/T06/T16/T19 of 21, `upgrades=[]` so truth is 0
by construction). Compared against CF6, recorded 2026-09-02 on the same configuration before R1.
Two things changed between the runs, both of them what v1 wind-up now is: `power_model` reads each
reference's north-calibrated direction, and the placebo no longer supplies the vendored Hill of
Towie north table -- it declares `north_offsets=None` and the shared step discovers.*

**The controls are exact, so this is a clean A/B.** `naive_ratio` reads no direction signal and
`toggle_specialist` reads none either; both reproduce CF6 to every digit recorded, on every
turbine, in both modes. Nothing but `power_model` moved.

**Prepost, error % (truth 0):**

| wtg | CF6 | now | Δ |
|---|---|---|---|
| T06 | +0.616 | **+0.044** | −0.572 |
| T07 | +0.462 | +0.420 | −0.042 |
| T11 | +0.188 | +0.328 | +0.140 |
| T12 | +0.337 | +0.242 | −0.095 |
| T16 | −0.721 | −0.728 | −0.007 |
| T19 | −0.769 | **−0.110** | +0.659 |
| **mean abs** | **0.515** | **0.312** | **−0.203** |
| spread | 1.385 | 1.148 | −0.237 |
| farm | +0.0390 | +0.0759 | +0.037 |

**Toggle, error % (truth 0):**

| wtg | CF6 | now | Δ |
|---|---|---|---|
| T06 | +0.087 | +0.007 | −0.080 |
| T07 | −0.151 | −0.119 | +0.032 |
| T11 | +0.197 | +0.242 | +0.045 |
| T12 | −0.224 | −0.139 | +0.085 |
| T16 | −0.753 | −0.694 | +0.059 |
| T19 | −0.558 | −0.329 | +0.229 |
| **mean abs** | **0.328** | **0.255** | **−0.073** |
| farm | −0.2186 | −0.1516 | −0.067 |

**A 39% cut in prepost per-turbine error, 22% in toggle.** The prepost gain is concentrated:
T06 and T19 account for nearly all of it, and only T11 got worse. That T06 lands at +0.044%
matters beyond the average -- it is the R-series fixture turbine, chosen in CF5 for being the
most accurate and stable on site, and it is now essentially exact on a real placebo.

**Per-turbine and farm move in opposite directions in prepost, the mirror of CF6.** CF6 found the
farm improving 4x while per-turbine accuracy slipped, and read that as better cancellation rather
than better estimates. Here the estimates genuinely improve and the farm number drifts from
+0.039% to +0.076% -- there is less residual left to cancel. Both are well inside the ±0.2% farm
target, and the per-turbine figure is the one that says the method got better. Toggle improves on
both axes.

**Implication.** The direction feature earns its place on real data, which the frozen benchmarks
could not show: there it was neutral on the headline (−0.024 pp prepost, +0.016 pp toggle over the
`overall` cells). The benchmark measures synthetic campaigns on four turbines; the placebo is 21
real turbines with real northing faults in the record, which is where a north-calibrated direction
has something to contribute. Worth remembering when a change reads flat on the benchmark.

---

## CF8 — Veer normalisation was being defeated by its own de-stepping: measuring the sector signature around a *speculative* split removes the very veer it should describe, so the split survives. Measuring it on the normalised residual instead cut the subset study's spurious changepoints 32 → 27, left every genuine one, and ran 34% faster

*2026-09-03. Reproduce: `uv run python -m benchmarking.baselines.study_northing_subsets` (99 cases,
Hill of Towie 2016–2024), before and after the change to `estimate_north_table`. Synthetic
counterpart: `tests/wind_up/test_northing.py::TestVeerNormalisation`.*

**Observed.** `veer_normalised` works exactly as designed — on a residual with 4° of
direction-dependent veer it takes the sector spread from 10–13° to **0.00°** and the day-to-day
wander of the daily level from 4.04° to 1.19°, the no-veer baseline. And it changed **nothing**: the
same changepoints were found with it on, off, and at every `veer_sector_deg` from 45° down to 10°
(false-positive counts 33/31/31/31/31).

**Root cause.** The sector signature is measured on a residual de-stepped by the first detection
pass, so a real recalibration cannot leak into it. But under veer that pass *over-detects* — it hits
the `max_k` cap. De-stepping then removes each spurious segment's level, and those levels **are** the
veer signature. The signature is measured on a residual that no longer carries the veer, subtracts
almost nothing, and the second pass re-finds the same splits. The false positives immunise
themselves against the mechanism built to remove them.

**Fix.** Search the veer-normalised residual first, with no step structure assumed, then re-measure
the signature around only the *confident* steps that search found (`_confident_steps`,
`VEER_SIGNATURE_MIN_STEP_DEG = 10.0`). Same number of searches as before. A genuine recalibration
still dominates the first search, so it is still de-stepped; a speculative split is not.

**Evidence.** On the 99-case subset study: `extra` 32 → **27**, `matched` 211 → **211**, `missing`
8 → 8, runtime 820 s → **541 s**. The reference case (`all__full_2016_2024`, 19 changepoints) is
**byte-identical**. Every one of the 97 other cases is unchanged; the entire improvement is
`west__year_2021`, where **T11** goes from a self-cancelling cluster of six
(−12.6, +10.8, −10.4, +11.2, +8.1, −4.4) to a single 3.95° step. That is the window-dependence R1
recorded as unresolved and left as genuine ambiguity — it was this bug.

**Implication.** Veer normalisation is load-bearing and should be kept, not dropped: its apparent
weakness was this defeat, not the mechanism. Its effect is only visible once the de-stepping stops
hiding it.

---

## CF7 — A turbine is part of the farm consensus it is northed against, which pins pass 2 to pass 1 on small odd farms; leave-one-out fixes that and detects much smaller steps, but costs more spurious ones, so it is **not** adopted

*2026-09-03. Reproduce: SMARTEOLE example run with `optimize_northing_corrections=True`; synthetic
sweep in the session scratch; `study_northing_subsets` for the real-data cost. The regression test is
`test_pass_two_refines_every_device_on_an_odd_sized_farm`, **xfail(strict)** — it documents the
defect rather than a pending fix.*

**Observed.** On SMARTEOLE (7 turbines, 3 months) the discovered north table is **byte-identical to
the pass-1 table on all seven turbines**, though pass 2's reference differs from reanalysis by 8.8°
mean absolute over the same rows. Pass 2 is an exact no-op.

**Root cause.** `_farm_direction` takes a per-timestamp circular median across devices. With an odd
count the median **is** one of the devices, so wherever it is device *j*'s own reading the residual
is `raw_j − (raw_j + off_j)` — algebra, not measurement, identical on every such row. That point mass
(~10–15% of the sample) sits at the centre of the distribution and straddles the 50th percentile
(below 0.42–0.48, atom 0.10–0.15), so the median snaps to it and the other ~88% of genuine
measurements cannot move the answer.

**Leave-one-out was built, measured and reverted.** It removes the atom entirely and is much better
at what pass 2 is *for*: detection floor ~4° against ~8°, and step **sizing** 0.3–0.4° error against
1.6–3.2° — self-inclusion attenuates the step because a stepping device drags the consensus with it.
The user's alternative, averaging the three values nearest the median, removes the atom but keeps the
attenuation, so it tracks the incumbent at every farm size tried. But on the 99-case study LOO scored
`extra` 32 → **42** for one recovered changepoint, and it needs a 4-device minimum farm. Not adopted:
the cost is real and the benefit is largest exactly where farms are smallest.

**Scale.** The pin is severe only on small odd farms. Hill of Towie's 21 turbines dilute the atom to
3.3%, where leave-one-out moves offsets by 0.15° mean / 0.21° max and changes no changepoint; an even
count interpolates between two members and forms no atom at all.

**Left open.** Whether to revisit LOO now that **CF8** removes most of its false-positive cost — with
the de-stepping fixed, its synthetic penalty falls to 1 spurious against 0 while it keeps +3
detections and 4x better sizing. Not re-measured on the 99-case study.

---

## CF6 — Honouring the declared `candidate_references` moves the prepost farm number 4x closer to truth (+0.148% → +0.039%), but *per-turbine* accuracy is marginally worse: the farm gain is cancellation, not better estimates

*2026-09-02. Reproduce: `uv run python -m benchmarking.campaigns.placebo`, Hill of Towie, both
modes, defaults (six upgraded turbines T07/T11/T12/T06/T16/T19 of 21, `upgrades=[]` so truth is 0
by construction). Controlled against the same driver run from the pre-C2 commit `a1f96af`.*

**Why re-recorded.** Before C2 the campaign path ignored `candidate_references` and every method
took "every turbine in the frame except the test one" as its references — so estimating T07 used
the other five *upgraded* turbines as references, which the declaration explicitly excludes. C2
made the path honour the declaration, dropping the reference pool from 20 to 15.

**Truth is still exactly 0.0 in both modes.**

**Prepost, pre-C2 → C2 (%), truth 0:**

| wtg | `power_model` pre | `power_model` C2 | `naive_ratio` pre | `naive_ratio` C2 |
|---|---|---|---|---|
| T06 | +0.305 | +0.616 | +7.807 | +7.412 |
| T07 | +0.684 | +0.462 | −1.678 | −1.604 |
| T11 | +0.184 | +0.188 | −2.199 | −1.574 |
| T12 | +0.617 | +0.337 | +0.047 | +0.555 |
| T16 | −0.442 | −0.721 | +0.638 | −0.152 |
| T19 | −0.658 | −0.769 | −0.508 | −1.155 |
| **mean abs** | **0.481** | **0.515** | **2.146** | **2.076** |
| **farm** | **+0.1485** | **+0.0390** | **+0.2330** | **+0.2027** |

**The control reproduces CF3.** The pre-C2 run reads **+0.1485%**, matching the **+0.148%** CF3
recorded for six test turbines. The comparison is therefore a like-for-like A/B of the reference
rule, not a config difference.

**The farm improves; the turbines do not.** `power_model`'s farm error falls from +0.148% to
+0.039%, but its mean per-turbine absolute error *rises* (0.481 → 0.515 pp) and its spread widens
slightly (1.341 → 1.385 pp). So the headline gain is **better cancellation across turbines**, not
better individual estimates — unsurprising given five references were removed, which costs a
little per-turbine precision. Why the residuals cancel better under the declared pool is **not
established here**; a plausible reading is that referencing turbines that are themselves under
test induces a shared error the farm aggregate cannot cancel, but this run does not demonstrate
that mechanism.

**Toggle (%), truth 0** — no pre-C2 control was run for toggle, so these are recorded, not
compared:

| wtg | `power_model` | `naive_ratio` | `toggle_specialist` |
|---|---|---|---|
| T06 | +0.087 | −0.191 | −0.191 |
| T07 | −0.151 | +0.239 | +0.239 |
| T11 | +0.197 | +0.338 | +0.338 |
| T12 | −0.224 | −0.196 | −0.196 |
| T16 | −0.753 | −0.683 | −0.683 |
| T19 | −0.558 | −0.561 | −0.561 |
| **mean abs** | **0.328** | **0.368** | **0.368** |
| **farm** | **−0.2186** | **−0.1425** | **−0.1425** |

**`naive_ratio` and `toggle_specialist` agree exactly, by construction.** Both compute the same
headline `rho_up / rho_base − 1` over the same complete-case, availability-filtered selection;
with `naive_ratio`'s default `toggle_campaign_only=True` both also restrict to the interleaved
blocks. What the specialist adds is the non-optional bootstrap sigma and the per-bin conditional
decomposition, not a different P50. This identity predates C2 (the 2026-09-01 runs show it too)
and is expected, not a defect.

**CF1's order-of-magnitude claim does not survive this configuration.** CF1 measured prepost
against toggle on the old two-turbine T01/T04 setup. On the current six-turbine config
`power_model` reads 0.515 pp prepost against 0.328 pp toggle — toggle still wins, but by a third,
not an order of magnitude. That is a configuration difference, **not** an effect of C2: both
CF1's and this run's prepost/toggle gap are measured within one code version.

**Implication.** The declaration is worth honouring on the farm number, which is the headline the
real campaigns report. The per-turbine cost is small but real and worth watching in C3+, where
reference pools are smaller than 15 and losing five references will hurt more.

## CF5 — T06 is the failure-mode fixture turbine: `power_model` holds it to 0.34% mean error with a 0.72 pp swing across seven windows while `naive_ratio` averages 3.3% and swings **13.3 pp**. T12 is the trap — stable but biased, with no headroom at all

*2026-09-01. Reproduce: `placebo_campaign(mode="prepost", upgraded=..., turbines=HOT_TURBINES)`

> **Predates the C2 reference rule (noted 2026-09-02).** C2 made the campaign path honour the
> declared `candidate_references`, so the six upgraded turbines no longer serve as each other's
> references. The sweep behind this entry has **not** been re-run and is not reproducible as
> written: the reproduce line above now yields a different reference set. Treat the numbers as a
> record of what was observed under the old rule. Re-running the sweep is a separate exercise.
swept over seven windows (12- and 24-month baselines into post years 2017–2020), all 21 HoT
turbines, the six test candidates T07/T11/T12/T06/T16/T19. Truth is 0 by construction, so every
reading is method error.*

**The criterion.** A good R-series fixture needs `power_model` accurate *and* stable, and
`naive_ratio` inaccurate *or* unstable. The second half matters: the gap is the advanced
machinery doing visible work, and a failure mode that only bites advanced methods (a north jump,
say) has nothing to break unless that work is happening.

**Per-turbine `power_model` error (%), truth 0:**

| wtg | 12→2017 | 12→2018 | 12→2019 | 12→2020 | 24→2018 | 24→2019 | 24→2020 | mean abs |
|---|---|---|---|---|---|---|---|---|
| **T06** | +0.36 | +0.31 | +0.32 | −0.14 | +0.47 | +0.21 | +0.58 | **0.34** |
| T07 | +0.17 | +0.68 | −0.23 | +2.12 | +0.57 | +0.00 | −0.40 | 0.60 |
| T11 | +1.14 | +0.18 | +0.80 | +0.45 | +0.67 | +0.73 | +0.19 | 0.60 |
| T12 | +0.60 | +0.62 | +0.61 | +0.66 | +0.39 | +0.47 | +0.62 | 0.57 |
| T16 | −0.33 | −0.44 | −0.19 | +2.57 | −0.05 | −1.49 | +0.38 | 0.78 |
| T19 | +1.64 | −0.66 | +0.66 | +2.83 | −0.48 | +0.11 | +1.06 | 1.06 |

**The same turbines under `naive_ratio` (%):**

| wtg | 12→2017 | 12→2018 | 12→2019 | 12→2020 | 24→2018 | 24→2019 | 24→2020 | mean abs |
|---|---|---|---|---|---|---|---|---|
| **T06** | −5.53 | **+7.81** | −2.21 | +0.84 | +4.70 | +1.51 | −0.20 | **3.26** |
| T07 | +2.21 | −1.68 | +0.26 | +0.75 | −0.59 | −0.60 | +0.87 | 0.99 |
| T11 | +3.34 | −2.20 | +1.82 | +0.57 | −0.57 | +0.68 | +1.40 | 1.51 |
| T12 | +0.04 | +0.05 | +1.64 | −0.12 | +0.07 | +1.66 | +0.62 | 0.60 |
| T16 | −2.68 | +0.64 | −2.06 | −0.17 | −0.75 | −1.74 | −1.11 | 1.31 |
| T19 | +0.28 | −0.51 | +0.01 | −1.90 | −0.37 | −0.25 | −1.90 | 0.74 |

**Verdict.** **T06** is the fixture turbine — the most accurate under `power_model` *and* the
worst under `naive_ratio`, a 2.9 pp gap and 5x the next best. **T11** is the credible second
(0.60% vs 1.51%), for an independent second case.

**Two turbines to avoid, for opposite reasons.** **T19** is the only turbine where `naive_ratio`
beats `power_model` outright (0.74% vs 1.06%). **T12** looks attractive — the tightest
`power_model` spread on site, 0.27 pp — but reads **+0.6% in every single window** while
`naive_ratio` sits near zero. Reproducing to within 0.03 pp across unrelated baselines and post
years makes that a *structural* per-turbine bias, not sampling noise: something about T12's
relationship to its references that the model does not capture. With no headroom over the naive
floor, a fault injected there would have to overcome the model's own bias to be visible.

**Caveat.** Six turbines over seven **overlapping** windows: the window results are not
independent, and every ranking below T06 rests on tenths of a percent against swings of similar
size. T06's margin is large enough to act on; the T07/T16/T12 ordering is not.

---

## CF4 — Window choice matters much less than expected once the turbine set is right: six of seven windows put every turbine inside 1.7%. The one bad window (`12mo→2020`) is a **thin-baseline** artefact, not a bad year — widening 2019 to a 24-month baseline cures it

*2026-09-01. Same sweep as CF5, read per window rather than per turbine.*

> **Predates the C2 reference rule (noted 2026-09-02).** C2 made the campaign path honour the
> declared `candidate_references`, so the six upgraded turbines no longer serve as each other's
> references. The sweep behind this entry has **not** been re-run and is not reproducible as
> written: the reproduce line above now yields a different reference set. Treat the numbers as a
> record of what was observed under the old rule. Re-running the sweep is a separate exercise.

| window | max abs | mean abs | spread |
|---|---|---|---|
| **24mo→2018** | 0.674 | 0.437 | 1.15 pp |
| **12mo→2018** | 0.684 | 0.482 | 1.34 pp |
| 12mo→2019 | 0.803 | 0.467 | 1.03 pp |
| 24mo→2020 | 1.060 | 0.539 | 1.46 pp |
| 24mo→2019 | 1.492 | 0.502 | 2.22 pp |
| 12mo→2017 | 1.644 | 0.708 | 1.98 pp |
| **12mo→2020** | **2.827** | **1.458** | 2.96 pp |

**The anomaly, and what it is not.** `12mo→2020` puts three turbines above 2% (T07 +2.12, T16
+2.57, T19 +2.83) when no other window exceeds 1.7%. The obvious reading — that 2020 is a
pathological year — is **wrong**: `24mo→2020` ranks fourth of seven and is entirely usable. The
difference is the baseline, not the post period. A 2019-only baseline is too thin for 2020;
widening it to 24 months moves T07 to −0.40, T16 to +0.38 and T19 to +1.06.

**Implication.** The committed placebo window (12 months of 2017 into 12 months of 2018) is
second of seven and within 0.01 pp of the best, so it stays. The 24-month baseline buys ~0.2 pp
of spread for an extra year of data — not worth it by default, but it is the lever to reach for
when a window looks unexpectedly bad.

---

## CF3 — Reaching the ±0.2% farm target is mostly about **reference count**, not test count: going from 3 to 20 references halved `power_model`'s error before any cancellation, and six test turbines then took the farm result to **+0.148%**

*2026-09-01. Reproduce: prepost placebo, 12 months of 2017 into 12 months of 2018, test turbines

> **Predates the C2 reference rule (noted 2026-09-02).** C2 made the campaign path honour the
> declared `candidate_references`, so the six upgraded turbines no longer serve as each other's
> references. The sweep behind this entry has **not** been re-run and is not reproducible as
> written: the reproduce line above now yields a different reference set. Treat the numbers as a
> record of what was observed under the old rule. Re-running the sweep is a separate exercise.
taken as the first n of (T07, T11, T12, T06, T16, T19), every remaining turbine a reference.*

| n_test | 1 | 2 | 3 | 4 | 5 | **6** |
|---|---|---|---|---|---|---|
| `power_model` farm | +0.68% | +0.44% | +0.50% | +0.46% | +0.32% | **+0.148%** |
| `naive_ratio` farm | −1.68% | −1.94% | −1.27% | +0.34% | +0.39% | +0.23% |

**Two distinct levers, and the first is the bigger one.** At `n_test=1` no cancellation is
possible, yet `power_model` already reads +0.68% against +1.0% for the earlier six-turbine
configuration — going from 3 references to 20 roughly halved the error on its own. Test count
then removes what remains, but only because the residual is **two-sided**: at n=6 the per-turbine
errors are +0.68, +0.18, +0.62, +0.30, −0.44, −0.66, four positive and two negative.

**A superseded hypothesis, recorded because it was wrong in an instructive way.** With T01+T04 as
the test pair, `power_model` read +0.72% and +1.27% — both positive — which looked like a
common-mode bias that no amount of pooling could cancel, and pointed at the method (adaptive time
decay was the suspect). It was an artefact of those two turbines. With a set spread across the
site the errors are two-sided and cancellation works. **Turbine selection, not the method, was
the problem.**

**`naive_ratio`'s farm number is not comparable.** It reaches +0.23% at n=6, but on a **10 pp**
per-turbine spread — cancellation of large opposing errors, not accuracy. `power_model`'s 1.34 pp
spread is the real quality difference.

**Scatter.** Window-to-window variation is ~±0.5% (CF4), so a single farm number carries real
luck. +0.148% should be read as "inside target at this window", not as a settled figure.

---

## CF2 — `naive_ratio`'s prepost placebo bias is **inherent, not fixable by window choice**: every turbine reads ±0.6–3.2% with no step change anywhere, driven by inter-annual wind-direction shift redistributing wake exposure

*2026-09-01. Reproduce: 2017 vs 2018 energy ratio of each turbine against the sum of the others,

> **Predates the C2 reference rule (noted 2026-09-02).** C2 made the campaign path honour the
> declared `candidate_references`, so the six upgraded turbines no longer serve as each other's
> references. The sweep behind this entry has **not** been re-run and is not reproducible as
> written: the reproduce line above now yields a different reference set. Treat the numbers as a
> record of what was observed under the old rule. Re-running the sweep is a separate exercise.
availability-filtered, on the six-turbine (T01–T05, T07) configuration.*

**Per-turbine naive prepost estimate, truth 0:** T01 −0.68%, T02 −0.58%, T03 −3.16%, T04 +2.52%,
T05 +2.11%. An independent hand calculation gave T04 = +2.52% against the runner's +2.46%, which
also cross-checks the C1 pipeline against a from-scratch computation.

**Not a data event.** Monthly energy ratios show no step anywhere in 2017–2018 — instead a
*relative redistribution*: T04 runs hot all through 2018 (monthly mean ≈ +2.9%) while T03 runs
cold (≈ −2.9%), with March 2018 extreme for both (+11% / −11%) during the easterly "Beast from
the East" spell. Hill of Towie's real AeroUp works are all 2021–2023, so the 2017–2018 window is
genuinely upgrade-free and the placebo is a true placebo.

**Root cause.** `naive_ratio` conditions on **nothing** — not wind speed, not direction. Between
years the direction distribution shifts, changing each turbine's wake exposure relative to its
neighbours by a few percent. No campaign window removes that, because it is not seasonal
composition (see CF1) but inter-annual climate. Only conditioning fixes it, which is what
`power_model` and v0 do and what the C-series is for.

**Implication.** `naive_ratio` is the leaderboard floor by construction. Its placebo reading is
a measure of how much work conditioning has to do on this site, not a defect to be tuned away.

---

## CF1 — The placebo separates prepost from toggle by an order of magnitude, and seasonal composition explains part of the prepost gap: matching baseline and post to the same twelve months cut the per-turbine spread from 5.4 pp to 3.1 pp

*2026-09-01. Reproduce: `uv run python -m benchmarking.campaigns.placebo`, Hill of Towie, both

> **Numbers superseded (2026-09-02).** C2 made the campaign path honour the declared
> `candidate_references`, so the six upgraded turbines no longer serve as each other's
> references. The same driver re-run under that rule is recorded as **CF6**; the readings below
> are what the old rule produced.
modes, `upgrades=[]` so truth is 0 by construction.*

**Truth is exactly 0.0 in both modes** — the C1 pipeline's first end-to-end confirmation on real
SCADA.

**Toggle beats prepost by an order of magnitude.** On the original six-turbine configuration
(T01/T04 upgraded, 3-month baseline into a 3-month post): prepost per-turbine +2.49% / −5.66%
with an 8.2 pp spread, against toggle −0.79% / +1.24% at 2.0 pp. Moving toggle to six months of
50-minute blocks tightened it further to −0.10% / +0.29%, **0.39 pp** — a 5x improvement, because
finer interleaving makes the on and off blocks sample the same weather.

**Seasonal composition is part of the prepost gap.** A full year of 2017 baseline against only
Jan–Jun 2018 leaves the two periods holding *different seasons*, confounded with the (absent)
effect. Extending the post period to a full 12 months so both span the same months of the year
moved T01 from −1.71% to −0.64% and T04 from +3.71% to +2.46%, cutting the spread from 5.4 pp to
**3.1 pp**. The residual is CF2's inter-annual effect, which this does not touch.

**Implication for the campaign shape.** Prepost runs a 12-month campaign on 12 months of
baseline; toggle needs only six months, since its blocks interleave within whatever period it is
given. This is why `PLACEBO_CAMPAIGN_MONTHS` is per-mode.
