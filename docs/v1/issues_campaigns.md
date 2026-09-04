# wind-up v1 — real-world-readiness issues (drafts)

Empirical results from this tranche are logged in
[findings_campaigns.md](findings_campaigns.md).

A **fresh tranche** of v1 work whose ambition is to **round out v1 so it is usable in
the real world**. Three pillars:

1. **Realism (C-series).** Simulate a few realistic, whole-farm campaigns — modelled
   on the real Hill of Towie open-source analyses
   (`resgroup/hill-of-towie-open-source-analysis`) — and mature the v1 methods so a
   user **declares a short campaign spec and the method does the right thing** (role
   assignment, northing, filtering, data split, reference selection and validity),
   automatically. Then compare the v1 methods and v0 on these campaigns.
2. **Robustness (R-series).** Simulate, with known ground truth, the data pathologies
   real SCADA throws at wind-up — bad northing, unstable sensors, invalid references,
   missing data — and mature the methods (chiefly `power_model`) so wind-up **handles
   them on its own**. Because each failure mode is *synthesized*, we can measure
   exactly how much it degrades a method and prove a fix closes the gap.
3. **Productized release (W-series).** Compose the winning pieces into a single
   headline method named **`wind-up`**, restructure the package so the v1 tool claims
   the `wind_up` import name (legacy retained as `wind_up_v0`, `src/` layout), and
   bring the methodology doc, examples, and README up to v1 — so the tranche ends in a
   coherent **v1.0.0** release on PyPI.

**Relationship to the earlier v1 issues.** The prior effort in
[issues.md](issues.md) (Issues 1–19) and its [findings.md](findings.md) are on the
**back-burner** for this tranche — still valuable for later phases, not the current
focus. Design specs for this tranche:
`docs/superpowers/specs/2026-08-27-realistic-campaigns-design.md` (realism),
`docs/superpowers/specs/2026-08-28-robustness-failure-modes-design.md` (robustness),
and `docs/superpowers/specs/2026-08-28-v1-productization-release-design.md`
(productized release).

## Ground rules for this tranche

- **Methods carried forward:** `oracle`, `naive_ratio`, `power_model`,
  `toggle_specialist`. **`rlearner` is dropped** (see C7).
- **Estimand:** per-turbine uplift **plus a result representative of the upgrade using the whole farm data** (one
  headline campaign number, as the real HoT analyses report).
- **One simulated instance per campaign, no replicates.** This tranche is about
  realism and ease-of-use, not sampling statistics.
- **Two declarations, one source:** a private `SyntheticCampaign` (drives the
  generator; holds the injected-upgrade physics = ground truth) and the public
  `CampaignSpec` derived from it (the facts an analyst would know; **never** the
  truth). Methods only see the spec.
- **Both outputs per campaign:** a farm-wide inspection report (per-turbine + farm uplift
  vs truth) **and** the campaign through the harness scoring path at n=1.
- **Run the real HoT campaign alongside the synthetic one.** Each C-issue that has a
  real counterpart in `resgroup/hill-of-towie-open-source-analysis` runs that campaign
  too. There is no ground truth for the real one, so it is **not** scored — it tests the
  *shape* of the solution: can the campaign be declared at all, does the declaration
  survive contact with real roles, exclusions, northing and reference validity, and does
  wind-up reach a sensible answer without bespoke driver code? The synthetic twin keeps
  accuracy honest; the real one keeps the declaration honest. Gaps found this way belong
  to the issue that hits them, not to a running list here.

## Ground rules for the robustness (R) issues

- **Isolated tiny fixtures.** Each fault is developed on a **tiny purpose-built
  fixture** — one treated turbine + ~3 references, a simple known AeroUp-shaped uplift
  — so the fault signal is isolated and iteration is fast. Reuses the C1 declaration /
  runner plumbing.
- **Success = invariance, not a race against v0.** The baseline for each fault is
  `power_model`'s **own** error on the *clean* fixture; the target is that injecting
  the fault degrades it by little (`power_model`-under-fault ≈ `power_model`-clean).
- **v0 is skipped** where possible (slow, and never developed against these
  synthesized faults); an optional one-off sniff is allowed, never the yardstick.
- **The fault must bite.** Every R-issue first calibrates the fault magnitude until it
  **significantly** throws off `power_model` on the clean fixture — otherwise there is
  nothing to fix. Making it bite may take iteration and is part of "done".
- **Both modes.** Every fault is evaluated in **both prepost and toggle**; the "bites"
  check is **per-mode**. A toggle campaign's rapid on/off switching can partly or
  wholly **cancel** a fault present in both periods, so a fault biting hard in prepost
  may bite little or not at all in toggle. Apply mitigation **where it bites**; where it
  does not bite (e.g. in toggle), still apply it if it is already part of the method,
  cheap to run, and harmless — determined empirically, never assumed.
- **Fix location follows how the concern is shared:** northing (R1) is a **shared**
  feature-engineering step every method inherits; the other three are
  **`power_model`-internal** (reference selection especially, since each method uses
  references differently).
- **Then re-verify on campaigns.** The best faults are re-injected into the relevant
  whole-farm campaigns (R1/R3 ↔ C3/C5) as an in-context check.

## Updating the frozen benchmarks

Learned the hard way while landing R1, which changed a **shared** feature-engineering step and
so moved every frozen artefact at once. Read this before accepting any benchmark change.

- **There are four frozen baselines, not one.** `study_power_model_compare_baseline.json` plus
  `study_toggle_methods_compare_baseline_{linux,portable,win32}.json`. A method-internal change
  usually touches one; a change to a shared step touches all of them, because every study driver
  inherits it.
- **Commit before running a sweep.** `--accept-candidate` refuses a candidate recorded from a
  dirty tree, and rightly so: the artefact would be stamped with a commit that cannot reproduce
  it. `study_power_model_compare` captures HEAD *before* the sweep, so committing while it runs
  is safe. Untracked files do not count as dirty.
- **Isolate the change before accepting it.** Diffing a fresh run against a baseline recorded
  weeks ago measures every commit since, not your change. Re-run with the change disabled
  (`--method-overrides '{"<flag>": false}'`, which deliberately writes no candidate) and diff the
  two runs. On R1 this took under an hour and showed the intervening seven weeks of work
  contributed under 0.0002 pp — so the whole movement was attributable, and a real worry was
  retired rather than carried.
- **The MOVED / UNCHANGED verdict is a blunt instrument.** Split by condition and by bin before
  believing it. Degenerate bins — near-zero power, TI 0.4–0.5, wind speed 0–2 m/s — dominate the
  means while their medians sit at zero. On R1's toggle diff the whole "62 of 84 cells MOVED,
  max 2.96 pp" verdict came from the near-zero-power bin; the twelve headline cells moved
  +0.005 pp.
- **Mind the units.** `benchmark_comparison.csv` is in **fractions**; the logs print **percentage
  points**. The logged "max delta" is the largest of bias/spread/score, not score alone. Compare
  like for like or you will chase a factor of 100.
- **The two scripts have different accept mechanics.** `study_power_model_compare` writes a
  candidate every full sweep, so `--accept-candidate` promotes it with no re-run.
  `study_toggle_methods_compare` has no candidate: `--update-baseline` re-runs the whole sweep.
  Budget for that.
- **A method that should not move is a free control.** `toggle_specialist` reads no direction
  signal, so R1 predicted its portable baseline would not move, and it did not (max 5e-07 pp).
  The toggle script rewrites the portable file *only when it actually changes*, so "portable
  baseline unchanged" in the log is a real check, not boilerplate. Predict which cells must be
  untouched and treat a violation as a bug in the change.
- **The `power_model` baselines are machine-specific but not load-sensitive.** LightGBM's
  threaded reduction order depends on the machine, so record and diff on one box. It does *not*
  depend on machine load: R1 ran sweeps concurrently with the full test suite and still
  reproduced a baseline to 1e-6, so there is no need to keep the machine idle.

## Suggested order

`C0 ✅ → [W0 ✅ early] → C1 ✅ → C2 → [R1 R2 R3 R4] → C3 → C4 → C5 → C6 → C8 → W1 → W2.`
The R-series lands after the C1/C2 foundation: **R1 (northing) before C3** so the
prepost campaign inherits the shared northing step; R2–R4 are independent
`power_model` work, any order within the block. **W0** (package restructure) is
independent and runs **early** (after C0) so later code lands in the new layout;
**W1/W2** are **terminal** (after C6 + R4) because the composed `wind-up` method needs
the robustness and campaign pieces first. **C8** (per-turbine change histories) lands
**before W1** so the generalized declaration is what gets promoted to public API, not
the flat one. C7 (drop `rlearner`, ✅ done) was independent. **R5** (northing refinement)
is deliberately outside this order: it is future work R1 identified but does not need.

**Done so far:** C0, W0, C7, C1, C2 and R1. **Next: R2–R4**, then C3, which inherits the
shared northing step R1 landed.

---

## C0 — Housekeeping: start the new issues list, back-burner the old

**Status:** ✅ Done (2026-08-27). This file exists and is linked from
`docs/v1/README.md`; `issues.md` and `findings.md` both carry a back-burner banner
pointing here.

**Goal:** make the new tranche the visible source of truth without losing the old
one.

**Scope**
- Add this file (`docs/v1/issues_campaigns.md`) and link it from
  `docs/v1/README.md`.
- Add a short banner at the top of `issues.md` and `findings.md` noting they are
  **back-burnered** in favour of this tranche, with a pointer here.
- No code changes.

**Done when:** the new issues doc exists and is linked; the old docs point forward.

---

## C1 — Campaign declaration + runner + farm uplift + placebo campaign

**Status:** ✅ Done (2026-09-01, PR #136). `SyntheticCampaign` → `CampaignSpec`,
`CampaignRunner`, the report and both placebo campaigns landed in
`benchmarking/campaigns/`, with the pure `farm_uplift` in `src/wind_up/farm.py` and
`true_farm_uplift` alongside the other ground truth. Results are logged as CF1–CF5 in
[findings_campaigns.md](findings_campaigns.md): truth is exactly 0 in both modes; toggle
beats prepost by an order of magnitude; the farm result reaches +0.148% with six test
turbines and fifteen references. v0 was taken out of scope (see below).

**Goal:** stand up the whole pipeline on the simplest case — a **placebo** (zero
injected uplift) whole-farm campaign — proving every method reports ~0 and that a
campaign can be *declared* rather than hand-wired.

**Scope**
- **`SyntheticCampaign`** — the private, generator-facing declaration: turbines and
  their roles (upgraded / reference / excluded), upgrade timing (prepost changeover
  or `ToggleSchedule`), the injected upgrade(s) (here: none / a no-op), site context
  (coords, northing corrections, ERA5 handle), analysis period.
- **`CampaignSpec`** — derived from the campaign, public facts only: upgraded
  turbines, timing, mode, site coords + northing, candidate references, exclusions.
  No injected-upgrade physics.
- **`CampaignRunner`** — spec → for each upgraded turbine, construct the applicable
  carried-forward methods (skip `toggle_specialist` on prepost) + a per-turbine
  `MethodInput`, run, collect `MethodOutput`s. Keep the thin seam for now;
  orchestration lives in the runner (C2 revisits this).
- **Farm uplift** — energy-weighted aggregation of per-turbine uplift to one
  campaign headline; the generator supplies the matching farm-level ground truth
  (upgraded synthetic energy vs counterfactual baseline energy over the treated
  records). Pin the exact weighting definition here.
- **Both output shapes** — a per-campaign inspection report (per-turbine + farm-uplift
  tables and diagnostic plots vs truth, the whole-farm analogue of
  `inspect_wake_steering_case`), and the campaign fed through the existing harness
  scoring path at n=1.
- **v0 is out of scope for the placebo.** The placebo is a whole-farm campaign, and v0
  enumerates test/reference combinations per turbine, so a whole-farm v0 run is not
  tractable. The seam still accepts `V0BinnedMethod` unchanged; a later campaign that
  needs v0 can run it over a small turbine subset.

**Done when:** a placebo whole-farm campaign is declared once and run end-to-end;
every method's per-turbine and farm-uplift estimate is ~0 within tolerance; both the
inspection report and the n=1 harness number are produced.

---

## C2 — Seam / campaign-context decision

**Goal:** decide how the `CampaignSpec` should reach the methods so they
self-configure (northing, filtering, data split, role assignment, reference
validity) — the question deliberately deferred at design time.

**Scope**
- Weigh the options with C1 in hand and the later campaigns' needs visible:
  (a) keep the thin `MethodInput`/`MethodOutput` seam with the runner orchestrating;
  (b) enrich `MethodInput` to carry the spec so methods read it at estimate time;
  (c) a hybrid (spec both builds methods and rides on the input).
- Record the decision and its rationale; refactor C1's runner/seam to match before
  the demanding campaigns build on it.

**Done when:** a documented decision exists and the code reflects it; C3+ build on
the chosen shape; the frozen `power_model` benchmark reads `UNCHANGED` (the study path
is untouched by the re-plumbing); and, as the **closing step**, the placebo campaigns
are re-run and **CF1-CF5 in [findings_campaigns.md](findings_campaigns.md) re-recorded**
— C2 makes the campaign path honour the declared `candidate_references`, so the six
upgraded turbines stop serving as each other's references and the recorded placebo
numbers no longer describe the code.

---

## C3 — Blade enhancement (AeroUp), prepost

**Real counterpart:** `scripts/uplift_analysis_2025/aero_up.py`
(`HoT_AeroUp_T13.yaml`) — a single upgraded turbine, four references, and exclusion
periods covering the install itself, a farm-wide curtailment spell, and each
reference's own later AeroUp.

**Goal:** a realistic **prepost** single-/few-turbine blade-enhancement campaign
where reference selection, the prepost split and northing all follow from the `CampaignSpec`.

**Scope**
- `SyntheticCampaign` using a region-2 Cp gain tailing to 0 at rated (AeroUp shape)
  on the upgraded turbine(s); other farm turbines as candidate references.
- **Automatic reference selection** from the `CampaignSpec` (exclude other upgraded / excluded
  turbines; honour candidate list).
- Prepost data split and northing applied by the method/runner without hand-wiring.
- Report + n=1 score for all applicable methods (`toggle_specialist` N/A here) and
  v0.

**Done when:** the campaign is declared and run whole-farm; per-turbine and farm-uplift
estimates track truth; the report shows how each method used references and the
prepost split.

**Re-verifies:** the shared northing step (R1) and the reference-validity screen (R3),
now in-context on a realistic prepost campaign.

---

## C4 — TuneUp (controller), toggle, multi-turbine

**Real counterpart:** `scripts/uplift_analysis_2025/tune_up.py`
(`HoT_PitchTuneUp2024_{east,north,south}.yaml`) — the farm split into zones, each with
its own test set and references.

**Goal:** a realistic **toggle** campaign with ~9 upgraded turbines and a
TI/stability-shaped effect, exercising multi-turbine toggle handling and the farm uplift
at scale.

**Scope**
- `SyntheticCampaign` with a condition-dependent (stability/TI-shaped) Cp change on
  ~9 turbines, a `ToggleSchedule` (~50-min period, per the real trial), ~10
  references.
- Multi-turbine toggle: per-turbine estimates across many upgraded turbines, the
  campaign-only data split, and the energy-weighted farm uplift across them.
- `toggle_specialist` and the other methods self-configure from the `CampaignSpec`.

**Done when:** the campaign runs whole-farm; per-turbine + farm-uplift estimates track
truth for all methods and v0; the report scales to many upgraded turbines.

---

## C5 — Dynamic Yaw (wake steering + collective control), toggle

**Real counterpart:** `scripts/wfc_analysis_2026` (`uplift_ws.py`, `uplift_cc.py`,
combined by `total_uplift.py`) — the hardest of the three. Two open questions to settle
here, both surfaced by reading that driver: **non-turbine references** (it uses a LiDAR,
`non_wtg_ref_names`, which the turbine-keyed `candidate_references` cannot express), and
**one campaign fanning out into many sub-analyses** (it runs one wind-up per steering
window x reference, varying the test set per window, then combines).

**Goal:** the hard campaign — generalise today's manual
`inspect_wake_steering_case` hacks into **declared** behaviour: inter-turbine wake
dependencies, references whose validity changes with wind direction, northing-sector
logic, and excluded turbines.

**Scope**
- `SyntheticCampaign` using the existing `WakeSteering` upgrade across the farm
  (plus a collective-control component if in scope), a `ToggleSchedule`, and an
  excluded turbine (e.g. T07).
- **Wake-aware reference validity:** the method must gate references by direction so
  a reference sitting in an upgraded turbine's changed wake is dropped for those
  timestamps — declared from geometry in the `CampaignSpec`, not a script-level filter.
- Northing-sector handling folded into the method/runner (replacing the
  `wd_filter` hack), so no bespoke driver code.
- Report + n=1 score; the farm uplift nets upstream steering losses against
  downstream gains.

**Done when:** the wake-steering campaign is declared once and run whole-farm with
no manual per-pair/sector wiring; reference validity is handled automatically;
per-turbine and net-farm estimates track truth.

**Re-verifies:** the shared northing step (R1, replacing the `wd_filter` hack) and the
reference-validity screen (R3), now in-context under wake-changed references.

---

## C6 — Rated-power up/downrate

**Goal:** exercise the region-3 / rated-power path none of the other campaigns
reach.

**Scope**
- `SyntheticCampaign` using `RatedPowerChange` (an uprate and/or a downrate) on a
  set of turbines, prepost or toggle.
- Confirm the methods behave sensibly when the effect is concentrated at/around
  rated power (where baseline and upgraded both clip), including the conditional /
  per-bin views.

**Done when:** the rated-change campaign runs whole-farm; per-turbine and farm-uplift
estimates track truth; the rated-power behaviour is visible in the report.

---

## C7 — Disentangle and remove `rlearner`

**Status:** ✅ Done (2026-08-28). `make_outcome_model` relocated into
`power_model/fitting.py` (with its dedicated unit test carried over); the `rlearner`
package, its tests, and `inspect_prepost_feature_ablation.py` removed;
`inspect_era5_matching_importance.py` repointed (it had no rlearner-specific arm to
strip). `poe all-fast` green; `power_model` reads UNCHANGED against both committed
benchmarks (toggle-compare max 0.06 pp; power-model-compare 0 material moves), i.e.
identical to within LightGBM's same-machine noise floor.

**Goal:** drop `rlearner` entirely, carrying forward only the shared pieces
`power_model` needs.

**Why it's not a simple delete:** `power_model/method.py` imports
`make_outcome_model` from `benchmarking.baselines.rlearner.nuisance`;
`inspect_era5_matching_importance.py` imports it too; `era5_sync.py` was already
promoted out of `rlearner` earlier. So the factory (and any other shared bits) must
be relocated first.

**Scope**
- Relocate `make_outcome_model` (and any other shared utilities `power_model`/the
  matching-inspection scripts still use) out of `rlearner` into `power_model` or a
  shared module; repoint the importers.
- Delete the `rlearner` package, its tests, and the rlearner-only scripts
  (`inspect_prepost_feature_ablation.py`, the rlearner arm of
  `inspect_era5_matching_importance.py`).
- Keep `power_model` behaviour identical — the committed `power_model` benchmark
  should be unchanged (the acceptance test).

**Done when:** `rlearner` is gone with its tests; `power_model` and the surviving
inspection scripts run unchanged; `poe all-fast` green; the `power_model` benchmark
is bit-identical.

---

## C8 — Per-turbine change histories (generalize the campaign declaration)

**Goal:** replace the campaign-wide "one upgrade, one date" declaration with a
**per-turbine timeline of changes**, so wind-up handles the campaign shapes that are
currently awkward to express, and can name what it is assessing.

**Motivation** — three situations seen on real campaigns that C1's flat declaration
cannot express:

- **Staggered dates.** Turbines are upgraded on different dates, not one changeover.
  Very common with aerodynamic upgrades, where a farm is worked through over weeks or
  months. Today this forces either one conservative shared date (throwing away data)
  or one hand-wired run per turbine.
- **References with their own history.** A reference turbine may itself have been
  changed during or shortly before the analysis period — e.g. a TuneUp campaign on a
  farm that recently had blade upgrades. Its data is then valid for part of the period
  and invalid for the rest. Today the only lever is `excluded_turbines`, which is
  all-or-nothing per turbine and discards usable data.
- **Not always an upgrade.** Some analyses confirm *stable* performance, or quantify a
  production-loss event. The declaration should describe "what changed, on which
  turbines, when" without presuming an improvement.

**Scope**
- **Per-turbine timeline.** Each turbine carries its own ordered changes, each with a
  date (or toggle schedule) — including reference turbines. The campaign-wide
  changeover becomes the degenerate case, not the model.
- **Usability follows from the timeline.** Which of a reference's records are usable is
  *derived* from its own changes, per turbine and per time range, replacing the
  all-or-nothing `excluded_turbines`.
- **Optional naming, neutral fallback.** A change may be named (e.g. `"TuneUp"`) and
  the name flows into report and plot titles. Naming is **optional**: unnamed, wind-up
  falls back to neutral language ("the change") and never asserts an upgrade.
- **Settle the neutral vocabulary** (a naming decision in its own right). C1–C7 bake
  "upgrade" into `upgraded_turbines`, `upgrade_timing`, `SyntheticCampaign.upgrades`
  and `UpgradeEffect`. Candidate umbrella terms: **change** (leaning — plain,
  international, covers upgrade / downrate / degradation / no change), *event*,
  *intervention*. Decide once and rename throughout. The same sweep retires "treated"
  from the benchmarking layer (443 uses, 64 of them the shared `treated_mask` /
  `treated_activity_mask` helpers); `src/` is already clear of it.
- **Disambiguate "window" in the harness.** `benchmarking/harness/campaign.py` uses it
  for two different spans in one docstring: `CampaignWindow` is the whole
  baseline-plus-activity span, while its prose says "post window", "activity window" and
  "shorter windows" for the *treated* part alone. That is the ambiguity C1 renamed
  `analysis_period` to escape, so a reader who knows the harness will misread the spec
  field. Settle one term for each span and apply it.
- **Migrate C1–C6 campaigns** onto the general model; the placebo becomes a campaign
  whose turbines have an empty change history.

**Ordering:** must land **before W1/W2**. W1's composed `wind-up` method
self-configures from a `CampaignSpec` and W2 promotes that type into the public
`src/wind_up` API — generalizing after that point means breaking published API. C1
keeps the flat model but must not let consumers depend on it (see the future-proofing
note in the C1 design).

**Done when:** a campaign with staggered per-turbine dates and a reference that changes
mid-period is declared and run end-to-end, using each reference only over its valid
records; a named change appears in report and plot titles and an unnamed one falls back
to neutral language; the neutral vocabulary decision is recorded and applied, with
"window" left meaning one thing in the harness.

---

# Robustness issues (R-series)

Failure modes real SCADA throws at wind-up, synthesized with known ground truth so we
can measure the degradation and prove a fix closes it. See the ground rules above and
the design spec
`docs/superpowers/specs/2026-08-28-robustness-failure-modes-design.md`.

Every R-issue shares a two-phase acceptance, run in **both prepost and toggle**:
1. **Bites (per-mode)** — the fault, calibrated on the clean tiny fixture,
   **significantly** throws off `power_model` (otherwise there is nothing to fix). A
   fault that does not bite in a mode (toggle may cancel it) needs no mitigation there
   — recorded explicitly.
2. **Fixed** — where it bites, the fix restores `power_model`-under-fault ≈
   `power_model`-clean.

---

## R1 — Northing errors (shared fix)

**Status:** ✅ Done (2026-09-03, PR #138). Shared step in `benchmarking/harness/northing.py`,
reached by both the campaign runner and the study path (which norths per replicate, discovering
for itself). `power_model`'s direction feature is on by default; all four frozen baselines
re-recorded. Two Done-when items were closed by decision rather than built, both recorded here:

* **v0's arm is dropped.** The fixture never ran `V0BinnedMethod`, so "the step bites v0" is not
  demonstrated. Accepted because the norther has been shown to track v0 three other ways: the
  21-turbine HoT farm-scale comparison, the SMARTEOLE road-test (uplift moves 0.05 pp / 0.01 pp),
  and the natural probe, which rediscovered v0's published T05 table from the data.
* **The examples are not re-run with auto-northing.** Both ship
  `optimize_northing_corrections=False`, and flipping it would add nothing: v0's auto path is
  already covered by `tests/test_optimize_northing.py` (six tests through the adapter, including
  injected changepoints) plus the three comparisons above, and the supplied-table path the
  examples actually use is covered by the SMARTEOLE and WeDoWind end-to-end tests. W2 migrates
  the examples to the v1 API, at which point `optimize_northing_corrections` ceases to exist for
  them.


**Goal:** wind-up recovers a known uplift despite a turbine's direction reference
carrying a **step change** in its north calibration partway through the record.

**Scope**
- **Fault (generator):** inject a known **step** in reported yaw angle or wind direction for some
  turbine(s) at a date (a recalibration / sensor swap). **Steps only — no drifts.**
- **Fix:** a **shared northing-correction feature-engineering step** in the runner /
  preprocessing, upstream of every method, so every method inherits it.
- Develop on a tiny fixture; land before C3 so the prepost campaign inherits it.

Notes on potential test turbines:
T06 is thought to be the best, however its likely best reference T05 has natural step changes in its yaw direction already in 2017 and 2018. That is not necessarily a problem but it means we are not working from a clean slate. But as a starting point could try running v0 on T06 for pre-post 2017-2018 with and without northing correction to see sensitivity of this naturally occuring example of the failure mode.
T11's surrounding turbines all have stable northing in 2017 and 2018 according to HOT open optimized_northing_corrections.yaml

other notes:
- `power_model` might need a little development if it does not use reference turbine yaw/wind direction at all yet (I think it does not, precisely because it could not see north calibrated versions up till now)
- reference turbine yaw direction is generally preferred over wind direction for the same reasons power is preferred over wind speed.
- `naive_ratio` and `toggle_specialist` do not use wind direction so are out of scope in this issue

**Done when:**
the step bites `v0` and `power_model` on the clean fixture, then the shared
northing step restores invariance; C3/C5 drop their bespoke northing wiring in favour
of this step.
the developed solution can be a drop-in replacement for the existing src/wind_up_v0/optimize_northing.py. Same or better performance is proven and useful test cases are ported. It should run MUCH faster (the old solution is a hand-rolled optimizer) and not require exotic dependencies (drop `ruptures`)
`power_model`'s reference-direction feature is **on by default**, not opt-in. That means the
shared northing step has to reach the study path too (it currently runs only in
`CampaignRunner`, so the study drivers behind the frozen benchmarks have no northed column),
and `study_power_model_compare_baseline.json` is regenerated.
The study path norths **per replicate**, discovering for itself rather than being handed a prior
table — the benchmark has to measure wind-up running unaided. The step therefore lives in
`benchmarking/harness/northing.py`, which both paths can reach, rather than under `campaigns/`.
the northing tool **shows its working**: per-turbine plots of the time-averaged residual
against the reference with the fitted step function overlaid, before and after correction, so
a user can see what was changed and judge it. Time averaging is what smears out veer.

---

## R2 — Unstable sensors (`power_model`-internal fix)

**Goal:** `power_model` is unmoved when a per-turbine sensor channel it might key on
is unstable across the baseline↔treatment boundary.

**Scope**
- **Fault (generator):** inject time instability — **both step changes and slow
  drifts** — into a per-turbine anemometer channel (primary) and temperature
  (secondary), differing across the baseline↔treatment boundary.
- **Fix:** `power_model` prefers **stable cross-turbine signals (power)** and
  avoids / downweights unstable per-turbine sensor features — hardening the standing
  "no reference-anemometer features" stance into a defended rule.

**Done when:** the injected instability bites, then feature hardening restores
invariance (injecting sensor drift/steps barely moves `power_model`'s error).

---

## R3 — Invalid references (`power_model`-internal fix)

**Goal:** a reference turbine with its **own** performance shift, unrelated to the
tested upgrade, no longer biases `power_model`.

**Scope**
- **Fault (generator):** give a reference turbine an independent performance shift
  (degradation / curtailment change) appearing during the analysis period.
- **Fix:** a **method-internal reference-validity screen** — `power_model` detects and
  downweights / drops the bad reference across the pool it uses at once (its analogue
  of v0's one-at-a-time round robin, kept internal because each method uses references
  differently).

**Done when:** the bad reference bites, then the validity screen restores invariance;
re-verified in-context on C3/C5.

---

## R4 — Missing data (`power_model`-internal fix)

**Goal:** `power_model` adapts to whatever signals are present instead of assuming a
fixed feature set.

**Scope**
- **Fault (generator):** remove channels / turbines from the fixture — a reference
  offline for part of the analysis period, an absent signal — so the input no longer matches a
  hardcoded feature list.
- **Fix:** `power_model` **discovers available signals** and builds features from what
  is present; degrades gracefully rather than crashing or silently collapsing.

**Done when:** the missing-data case bites (or would crash) the current fixed-feature
`power_model`, then signal discovery restores a run that stays accurate under missing
channels / gaps.

---

## R5 — Northing refinement: small-N devices, and absolute accuracy from wake nadirs

**Status:** future work, not blocking. R1 delivered a norther good enough to ship; these are
the two places it is known to fall short, both identified while doing R1.

**Goal:** north devices a farm consensus cannot reach, and improve the *absolute* accuracy of
the answer rather than only its internal consistency.

### Part A — north one or two devices with pass 1 alone

`north_farm` refuses below `min_devices_for_farm_reference=3`, so a **two-device campaign
cannot use farm-consensus northing at all** (measured during R1's natural probe). At exactly
three the quorum is every device. Small campaigns are common, so this is a real gap, not a
corner.

Pass 1 already norths each device against reanalysis on its own, so the machinery exists; what
limits it is accuracy. `REANALYSIS_MIN_STEP_DEG = 10` exists because reanalysis carries its own
direction-dependent bias, and a spell of unusual wind moves every turbine's residual against it
together. Against a farm consensus that common-mode error cancels; against reanalysis it does
not. So a single-device answer is currently trustworthy only for gross recalibrations.

**Why this is tractable:** the development loop is unusually good. Hill of Towie has 21 turbines
with a published, independently-derived northing table, so pass 1 can be run **one turbine at a
time** and scored directly against a known answer, 21 times over, without any synthetic data.
`tests/wind_up/test_northing_real_data.py::TestSingleTurbineAgainstReanalysis` is the seed of
this; it currently asserts only that a lone turbine finds its *large* recalibration.

**Done when:** a single device is northed to a stated accuracy against the published HoT table
across all 21 turbines; `REANALYSIS_MIN_STEP_DEG` is lowered by evidence rather than assertion;
challenging synthetic cases (small steps, steps near the record edge, steps during an outage)
pass; and `north_farm`'s three-device floor is either removed or documented as the deliberate
boundary between two supported regimes.

### Part B — pass 3: nudge to absolute truth using apparent wake nadirs

Passes 1 and 2 fix the farm relative to reanalysis and then to itself. Neither has a *physical*
absolute reference. Wake nadirs do: when the wind blows along the line joining two turbines, the
downstream one sits in the upstream one's wake and its power dips, and the direction at which
that dip occurs is known from the layout geometry alone. Matching measured nadirs to geometric
bearings gives an absolute anchor that owes nothing to a reanalysis model.

**This is a third pass, not a replacement.** It runs only once the changepoints and their
relative steps are settled, because it estimates a single absolute shift per segment; asking it
to find changepoints as well would be a different and much harder problem. Order:
reanalysis anchor, farm consensus, then wake-nadir refinement.

**Existing machinery to build from:** the synthetic generator's `WakeSteering` upgrade already
derives directed pairs from geometry, and `inspect_wake_steering_case.py` computes a pair's
nadir and sector. What is missing is the inverse — *measuring* an apparent nadir from a real
power-deficit-versus-direction curve, and turning a set of those into one offset per segment.

**Done when:** measured nadirs recover a known injected absolute offset on synthetic data; on
Hill of Towie the pass-3 correction is small (it should be, since pass 1/2 already agree with
the published table to ~1°) and does not disturb the changepoints; and it degrades gracefully
where geometry gives too few usable pairs.

**Gotcha to design around:** a turbine's own wake-affected rows are exactly the rows an uplift
method wants to treat carefully, so pass 3 must not quietly change which rows downstream
analysis considers valid. It outputs an offset, nothing else.

---

# Productization issues (W-series)

Turn the winning pieces into a shippable **v1.0.0**: one headline method named
`wind-up`, a restructured package that claims the `wind_up` import name, and v1-current
methodology / examples / README. Design spec:
`docs/superpowers/specs/2026-08-28-v1-productization-release-design.md`.

---

## W0 — Repo restructure: `src/` layout + rename legacy to `wind_up_v0` (early)

**Status:** ✅ Done (2026-08-28, PR #135). `src/wind_up/` (v1) and `src/wind_up_v0/`
(legacy) with every importer repointed; examples byte- and pixel-identical.
`benchmarking*` is still packaged — dropping it from the release artifact is deferred to
W2, which already carries that item.

**Goal:** the new v1 tool claims the `wind_up` import name while the legacy tool is
retained, done **early** so all later code lands in the new layout.

**Scope**
- Move legacy `wind_up` → `src/wind_up_v0/`; stand up `src/wind_up/` as the v1
  package's home (a skeleton W1 fills). Adopt the conventional `src/` layout.
- Repoint every importer (notably the `v0_binned` baseline), tests, `pyproject`
  packaging, and examples. Distribution name **stays `res-wind-up`**; only import names
  change (`wind_up` = v1, `wind_up_v0` = legacy).
- Decide `benchmarking/`'s final home and confirm it is **excluded from the v1.0.0
  release artifact** (it is the eval harness, not the product).

**Done when:** the repo builds and tests pass under the new layout; v0 still runs as
`wind_up_v0` and its committed benchmark is **unchanged** (behaviour-preserving); no
importer still references the old `wind_up` path for the legacy tool.

---

## W1 — The composed `wind-up` method (terminal)

**Goal:** a single headline method named **`wind-up`** — the v1 deliverable — that
composes the winning pieces and self-configures from a `CampaignSpec`.

**Scope**
- Build `wind-up` in `benchmarking/baselines` (like every v1 method), composing
  **`power_model` (definite) + the shared northing step (R1) + the reference-validity
  screen (R3) + missing-data adaptation (R4)**. `toggle_specialist` inclusion is
  **TBD, settled with evidence** (it may be the toggle arm, or the composed
  `power_model` path may suffice).
- Validate `wind-up` as the headline method across the campaigns (C1–C6) and the
  failure modes (R1–R4), in **both prepost and toggle**.

**Done when:** `wind-up` runs self-configured from a `CampaignSpec`, tracks truth on the
campaigns, and stays invariant under the failure modes in both modes; the exact
composition (including the `toggle_specialist` decision) is settled and recorded.

---

## W2 — Productization & v1.0.0 release (terminal)

**Goal:** a coherent v1.0.0 release where package, method, docs, and examples all line
up.

**Scope**
- Promote the composed method into the public `src/wind_up` v1 API (decide what moves
  out of `benchmarking/` vs is re-exported).
- Replace the opaque `docs/wind-up uplift validation methodology v3.pdf` with a
  tracked **`docs/methodology.md`** describing the v1 method (the new source of truth;
  the PDF is exported from it at release).
- Migrate or remove every example (`examples/`) to the v1 API; rewrite `README.md` for
  v1. Northing changes shape in the move: the v0 examples pin a pre-computed table with
  `optimize_northing_corrections=False`, whereas v1's shared step discovers by default, so a
  migrated example should **show discovery** rather than port the pinned table across.
- **Drop `benchmarking*` from packaging** (deferred from W0, where it stayed packaged
  only for a separate project that imports `toggle_specialist`): once that external
  dependency is gone, remove `benchmarking*` from `[tool.setuptools.packages.find]`
  `include` and confirm the harness is **excluded from the v1.0.0 release artifact**.
- **Delete the `config/`, `input_data/`, `cache/` root folders** — legacy artefacts
  from before env-vars / `Path.home()` were used — and rework `wind_up_v0/constants.py`
  path handling accordingly (env vars / `Path.home()` instead of `PROJECTROOT_DIR`-
  relative, so nothing depends on those root folders).
- **Shore up `power_model` unit coverage.** R1 flipped `direction_feature` on by
  default; the existing suite was recovered by fixture edits rather than deletions, but
  it is thin in places the benchmarks cannot reach — error paths, the `CampaignContext`
  seam, and the reference-only (design note §3) guard. A released package should not
  rest on the benchmarks alone for those.
- **Clean up the development-phase documentation.** `docs/superpowers/` (design notes
  and plans) and `CLAUDE.md` are untracked and git-ignored as of 2026-09-03, so tracked
  files that cite them — `docs/v1/issues_campaigns.md`, `docs/v1/findings_campaigns.md`
  — now point at paths a fresh clone will not have. Decide per document what a released
  v1 should carry: fold what is still true into `docs/methodology.md` or `docs/v1/`, and
  drop the citations that are only development history.

- **A campaign is declared, not scripted.** `CampaignSpec` gains a simple
  user-facing declaration — a YAML file it initializes from — so an analyst describes
  turbine roles, timing, exclusions and northing without writing Python. This is the
  v0 `WindUpConfig.from_yaml` ergonomics carried into v1, with the method config that
  v0 mixes into the same file kept on the method instead.

**Done when:** a user installs `res-wind-up`, imports `wind_up`, and runs the v1
`wind-up` method end-to-end from the examples and README; `docs/methodology.md`
describes it; `benchmarking` is no longer packaged and the legacy root folders are gone;
the branch is ready to tag **v1.0.0**.

**Also done when:** each of the **three real Hill of Towie campaigns** — AeroUp and
TuneUp (`scripts/uplift_analysis_2025`) and Dynamic Yaw (`scripts/wfc_analysis_2026`) in
`resgroup/hill-of-towie-open-source-analysis` — can be **declared in a short YAML file**
and re-run through v1, with no bespoke driver code for role assignment, exclusions,
northing or reference validity. That repo is the acceptance test for "easy to use": if
re-doing those analyses still needs a hand-written script per steering window, the
declaration is not finished.
