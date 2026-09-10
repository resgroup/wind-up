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

- **What the campaigns are testing is v1 wind-up, not a field of methods.** The C-series
  exists to exercise the thing being shipped, so a campaign must run it as a user would
  get it. Concretely that means **`power_model` on, and northing discovered rather than
  supplied** (`north_offsets=None`, the shared step doing the work). A campaign that
  turns either off is testing something other than v1 wind-up, and any result from it
  should be read that way. Tests may switch `power_model` off to avoid the `ml`
  dependency; drivers should not.
  - **In v1 wind-up:** `power_model` (definite) and the shared northing step (R1).
    `toggle_specialist` is **TBD**, to be settled with evidence in W1b.
  - **Alongside, for comparison only:** `naive_ratio` (a deliberately simple yardstick,
    never a candidate for the shipped method) and `oracle` (a sanity anchor that returns
    the injected truth).
  - **`rlearner` is dropped** (see C7).
  The composition itself is W1a/W1b's business; this rule is only about how the campaigns
  must be run so their results speak about the deliverable.
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
- **Fix location follows how the concern is shared:** northing (R1, refined by R5) is a
  **shared** feature-engineering step every method inherits; R2–R4 and R6 are
  **`power_model`-internal** (reference selection especially, since each method uses
  references differently). R6 additionally carries generator work, because its fixture
  cannot be built without more realistic upgrade signals.
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

`C0 ✅ → [W0 ✅ early] → C1 ✅ → C2 ✅ → [R1 ✅ R2 ✅ R3 ✅ R4 ✅] → W1a → C3 → C4 → R5 →
C5 → R6 → C6 → C8 → C9 → W1b → W2`, with **W3 running continuously from W1a onward** rather
than at one point in the line.

The R-series lands after the C1/C2 foundation: **R1 (northing) before C3** so the
prepost campaign inherits the shared northing step; R2–R4 are independent
`power_model` work, any order within the block. **W0** (package restructure) is
independent and runs **early** (after C0) so later code lands in the new layout.

**W1 is split, and only half of it is terminal.** Its interface — composing `wind-up`
and letting it self-configure from a declared `CampaignSpec` — does not depend on the
campaign work, but its *validation* is the campaign work: W1's done-criteria are that
`wind-up` tracks truth across C1–C6 and stays invariant under R1–R4. So **W1a**
(compose + declare) is hoisted to right after R4, and **W1b** (validate) stays terminal.
**W2** stays terminal behind W1b.

**W3 (the analyst dry run) is the reason for the split.** It measures whether an analyst
with the documentation and no access to the source can drive a campaign and read what
came out — and it needs W1a's declared entry point, nothing more. Running it from W1a
onward means each C issue's outputs get interpretability-tested as that issue lands, and
W2's documentation is written against known gaps rather than guesses. Deferring it to
W2 would mean writing the release documentation blind and only then discovering what it
failed to explain.

**R5** (northing refinement) lands **before C5**. Everywhere else in this tranche a
direction error is a nuisance variable, which is why R1's norther is good enough to ship;
in C5 the treatment *is* a direction offset, so a systematic absolute error moves the
signal being measured rather than adding noise around it. R5's two known gaps bite
exactly there — Part B supplies the absolute anchor from wake nadirs that passes 1 and 2
cannot, and Part A norths the small device counts a geometry-driven steering pair can
come down to.

**R6** (abnormal behaviour) lands **after C5** because it needs the signal realism that
issue's fixtures push toward, and because its detector must not be handed a dataset in
which only the thing it is looking for ever moves pitch or rpm. It is the last robustness
issue for a reason: it is the only one that changes what *truth* means.

**C8** (per-turbine change histories) lands **before W1b** so the generalized
declaration is what gets frozen as public API at v1.0.0, not the flat one. **C9** (test
turbines as wake contributors) follows it for the same reason — it adds a field to the
`CampaignContext` seam W2 publishes — and because C8's per-turbine timeline is what makes
the wake-only role time-ranged. C7 (drop `rlearner`, ✅ done) was independent.

**Done so far:** C0, W0, C7, C1, C2, R1, R2, R3 and R4. **Next: C3**, which inherits the
shared northing step R1 landed and the reference screen R3 landed.

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
- **Decide what the shared northing step fits on when the upgrade itself steers the yaw.**
  `WakeSteering` moves the reported nacelle position on treated rows, and `yaw_usable`
  screens on power and downtime only, so those deliberately steered rows currently enter
  the northing fit and the correction can absorb part of the intervention. Two things
  limit the damage today and neither is a defence: the search needs a segment of at least
  seven days, so rapid toggling cannot forge a changepoint, and the offsets are circular
  medians, which shrug off a displaced minority. It is a level bias, unmeasured.
  **The obvious fix — exclude treated rows while fitting — is wrong as a general rule**:
  in prepost the treated rows are half the record, and a north step occurring inside the
  campaign is exactly R1's fault, so excluding them would make it undiscoverable. So the
  exclusion has to be specific to upgrades known to move the direction channel, which the
  runner cannot infer from a `CampaignSpec` that deliberately carries no truth — though a
  real analyst running a steering campaign would know. Settle it here: measure the bias
  first, then decide whether the spec should carry "this upgrade steers yaw" or the step
  should screen the rows some other way. Raised by review on PR 138.
- Report + n=1 score; the farm uplift nets upstream steering losses against
  downstream gains.

**Done when:** the wake-steering campaign is declared once and run whole-farm with
no manual per-pair/sector wiring; reference validity is handled automatically;
per-turbine and net-farm estimates track truth.

**Re-verifies:** the shared northing step (R1, replacing the `wd_filter` hack) and the
reference-validity screen (R3), now in-context under wake-changed references.

**Depends on R5**, which is scheduled immediately before this issue. This is the one
campaign whose treatment is itself a direction offset, so northing's *absolute* accuracy
— not just its internal consistency — is load-bearing here in a way it is nowhere else
in the tranche.

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

**Ordering:** must land **before W1b/W2**. W1a's composed `wind-up` method
self-configures from a `CampaignSpec` and W2 promotes that type into the public
`src/wind_up` API — generalizing after that point means breaking published API. W1a is
hoisted ahead of C8, so its declaration is explicitly *not yet frozen*: C8 generalizes
it in place, and only W2 publishes it. C1
keeps the flat model but must not let consumers depend on it (see the future-proofing
note in the C1 design).

**Done when:** a campaign with staggered per-turbine dates and a reference that changes
mid-period is declared and run end-to-end, using each reference only over its valid
records; a named change appears in report and plot titles and an unnamed one falls back
to neutral language; the neutral vocabulary decision is recorded and applied, with
"window" left meaning one thing in the harness.

---

## C9 — Test turbines as wake contributors (power-free, not absent)

**Goal:** let a turbine that is not a valid *reference* still contribute what it
physically does to its neighbours — its wake — instead of being dropped from the
estimate's frame entirely.

**Motivation.** `power_model` already has the mechanism: a `power_free` reference keeps
its direction channels and a `waking` boolean but contributes no power columns. Today
that is reachable only through R3's screen, and only for turbines already in the
candidate pool. Two gates close it off:

- `CampaignContext.select()` keeps `test_wtg`, `candidate_references` and `also` only, so
  every other turbine is dropped at the first line of `PowerModelMethod.estimate`, which
  never passes `also`.
- `_checked_power_free` raises on any name that is not already in `refs`.

So a campaign's *other* changed turbines vanish from each estimate. On the placebo that
costs little — 6 of 21 turbines. On the real AeroUp campaign **all 21 Hill of Towie
turbines were upgraded**, so each estimate's frame would keep only the turbines
hand-declared as references and none of the wake neighbours. A neighbour's wake is there
whether or not that neighbour is treated; the declaration currently has no way to say so.

**Scope**
- **A third turbine role in the context** — kept in the frame, never a candidate
  reference, never screened, never in the reference-stability table, and always
  power-free.
- **Derived, not declared.** `benchmarking/campaigns/context.py` fills it from the
  campaign's other changed turbines. Under C8's per-turbine timelines the role is
  *time-ranged* rather than whole-turbine: a turbine is a valid reference before its own
  change and a wake-only contributor after it, from the same timeline C8 already needs.
- **Union it into `power_model`'s feature-pool `power_free`**, leaving the R3 screen and
  the reference-uplift table operating on `candidate_references` alone.
- **Re-record the frozen benchmarks.** This changes the feature matrix, so every C- and
  R-series campaign number moves.

**The open question, to settle first: is `waking` treatment-invariant?** The substitute
for power is a boolean thresholded on active power. For a Cp change it is very likely
invariant — a turbine above a few percent of rated already carries most of its thrust, so
a low threshold separates waking from parked while leaking almost none of the power
level, which is the argument `_waking_features` already makes for screened references.
For **wake steering (C5)** it is not: changing the neighbour's wake *is* the
intervention, so its waking and geometry state move *with* treatment and a post-treatment
variable would enter the feature matrix (design note §3). The concern is sharper here
than for a screened reference, where the suspected change is unknown and incidental
rather than known and simultaneous. Decide — on C5's fixture, with evidence — whether the
role is uniform or whether a steering campaign must keep the neighbour out of even the
waking feature.

**Ordering:** with or just after **C8**, and **before W1b/W2**. It adds a field to
`CampaignContext`, the method seam W2 promotes to public API, and C8's per-turbine
timeline is what makes the role time-ranged rather than whole-turbine. Its §3 decision
wants C5's fixture, so C5 should land first.

**Done when:** a campaign in which every turbine is changed runs end-to-end with each
estimate still seeing its neighbours' wake; the wake-only turbines appear in no
reference-stability table and are screened by nothing; the §3 decision is recorded with
the evidence behind it; and the frozen benchmarks are re-recorded.

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
re-recorded. The northing plots are wired into the shared step and written whenever it discovers
(`north_scada(out_dir=...)`); the placebo is the demonstration, since it now supplies no prior
table. Two Done-when items were closed by decision rather than built, both recorded here:

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

**Status:** ✅ Done (2026-09-04). Measured, no `power_model` change needed — see CF11 in
[findings_campaigns.md](findings_campaigns.md). Two fault classes (`SensorGainStep`,
`SensorGainDrift`) and the `benchmarking.campaigns.sensor_fixture` driver landed; the fault
scales `wind_speed` and `wind_speed_sd` together, so turbulence intensity is invariant and only
the wind-speed axis moves. Scope was narrowed by decision to **measure the failure mode rather
than mitigate it**, on the evidence that the pathway R2 was written to close is already shut:

* **The headline is immune.** Worst case (x0.5 / x1.5 gains, both step and drift, on the test
  turbine and on the nearest reference) moved `power_model` by at most **0.214 pp** in prepost
  and **exactly zero** in toggle. The one nonzero result is the ERA5 lag sweep tipping one row
  (10 minutes); every other arm moved by <= 1.1e-5 pp.
* **The "fault must bite" gate is not met against the shipped configuration, and that is the
  finding.** An exposed arm carrying reference anemometry as features — the configuration the
  standing rule forbids — moves **34 pp**, so the exclusion is now measured rather than assumed.
* **The conditional grid does move** (the `ws` axis is the test turbine's own anemometer, so a
  gain re-bins every row: up to 600 pp in a degenerate bin). Left alone deliberately: the
  conditional machinery is nascent, and the fault classes are in place to re-measure when it
  matures.
* **Temperature was dropped** by inspection: `ambient_temp` is a diagnostics-only schema role
  that reaches `power_model` through nothing, and ERA5 supplies `temperature_2m` independently.

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

**Status:** ✅ Done (2026-09-07). A method-internal reference-validity screen landed in
`power_model`, with `ReferenceCpChange` as the generator-side fault and three drivers
(`screen_calibration`, `reference_fixture`, `screen_roadtest`) behind the numbers — see CF13 in
[findings_campaigns.md](findings_campaigns.md). Each candidate reference is estimated as if it
were a test turbine against the others, and a clear outlier from the pool median is made
**power-free**: it keeps its direction features and gains a `waking` boolean, so its wake
information survives while the channels a performance change corrupts do not.

* **It found a real one.** On clean Hill of Towie data, T17 reads **+4.67%** against a
  19-reference pool whose median is +0.26%; its median power ratio to its neighbours steps from
  0.735 across 2017 to 0.804 across 2018. Nobody was looking for it.
* **Screening earns its place.** Mean error on the injected prepost fixture arms falls from
  **1.84 pp to 0.63 pp**, and up to 3.2 pp is recovered on the two-bad-reference case.
* **Two gates, both measured.** `screen_floor = 0.025` (clean pools spread up to 1.19 pp) and
  `screen_min_campaign_days = 150`, the latter set by the benchmark sweep after a 90-day gate
  still false-positived on 3-month campaigns.
* **Prepost only.** Toggle is not vulnerable to this failure mode and the screen cannot see it
  there anyway; `power_model` stays exposed via its baseline-reaching fit window, deliberately, as
  closing that re-opens the `toggle_campaign_only` knob Issue 16 pruned.
* **Frozen benchmarks unchanged.** Zero references ruled out across the sweep and no MOVED
  verdicts; the largest movement anywhere is 0.013 pp in a degenerate conditional cell.

**Still open:** the in-context re-verification on C3/C5, which those issues carry.

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

## R4 — Missing data ✅ done 2026-09-08

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

### What actually happened — two corrections to the scope above ([CF14](findings_campaigns.md))

**This was not `power_model`-internal.** Two of the five missing-column failures die in the
**shared northing step** before `power_model` runs at all, and two of the four fixes landed
there. The heading above is wrong about where this failure mode lives.

**Signal discovery was measured and deliberately not built.** The probe
(`benchmarking.campaigns.outage_probe`) showed the absent/empty split *is* the fault line —
every NaN-shaped outage already returns a number, every absent-column one raises — so
"discover and adapt" would have converted honest raises into silent estimates, the wrong
direction. Missing data turned out to be absorbed within a 0.127 pp estimator-noise floor
in every arm but one: a reference that disappears from the delivery entirely, worth
**+0.45 pp**, and that is [CF3](findings_campaigns.md)'s reference-count effect rather than
corruption. What shipped instead is **attribution and announcement**: every failure now names
the column that caused it, the conditional step degrades instead of taking the headline down
with it, and a declared reference the data does not carry is dropped with a warning.

**Left undone deliberately:** outage *position* was held at a 30-day mid-baseline window, and
`test_empty_upgraded` (+0.20 pp) and `era5_incidental_absent` (−0.19 pp) sit just above the
floor but were not seed-swept, so they are unmeasured rather than cleared.

---

## R5 — Northing refinement: small-N devices, and absolute accuracy from wake nadirs

**Status:** scheduled **before C5**. R1 delivered a norther good enough to ship, and these are
the two places it is known to fall short — both identified while doing R1. They are refinements
everywhere a direction error is only a nuisance variable, and stop being refinements in C5,
whose treatment is itself a direction offset.

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

## R6 — Abnormal turbine behaviour: a one-off curtailment is not the upgrade's doing

**Goal:** an episode of abnormal-but-real operation during the campaign — a grid or noise
curtailment, a temporary derate, a run on a de-rated controller after a fault — is excluded
from the analysis, so the reported uplift represents **future normal behaviour** rather than
the particular incidents this campaign happened to contain.

**This is the only R issue that changes what truth means.** The others hold the truth still by
construction: R2 corrupts a *reading* and leaves power alone; R3 changes real power but only on
a *reference*, and truth is derived per test turbine. R6 changes real power **on the test
turbine**. That energy really was lost, so a naive truth absorbs it — but it is not the
upgrade's effect and will not recur predictably, so an estimate that includes it answers the
wrong question. The uplift being estimated is the upgrade's effect *under normal operation*.

**Two things must therefore agree, and neither may be told where the curtailment is:**
- ground truth is defined over normally-operating rows only, and
- the method excludes the same rows, having **discovered** them.

That is the invariance test: the estimate on a campaign with a curtailment injected matches the
estimate on the clean campaign. The analyst-declared escape hatch already exists
(`ColumnSchema.exclude_row`) and stays as the fallback; R6 is about the automatic case.

**Scope**

- **Generator: the test-turbine guard needs a principled exception, not removal.**
  `_check_faults_spare_the_test_turbines` (added in R3) refuses any `changes_power` fault aimed
  at a test turbine, precisely because it would be silently absorbed into the truth. R6 *is*
  that fault. The exception: a power-changing fault may target a test turbine **iff** it
  declares the rows it affects, and the generator excludes those rows from the truth mask —
  `true_uplift` already accepts one. A fault that changes test-turbine power without declaring
  its rows stays refused, so the guard keeps doing its job for everything else.

- **Generator: a curtailment fault** on a named turbine over a declared window, carrying the
  signature a real one has — pitched out, rpm down, power capped — so power's standard deviation
  collapses and its maximum sits near its mean.

- **Generator: more realistic upgrades, which is the bulk of the work and a precondition rather
  than a nicety.** Today `apply_upgrades` moves `active_power`, `gen_rpm` and `wind_speed` only.
  `ColumnSchema.pitch` exists but nothing writes it, and the min/max/sd companions move nowhere
  except where R3's fault scales `active_power_min`. So:
  - declared upgrades must move **pitch and rpm** consistently with the Cp or rated change they
    represent;
  - the **min, max and sd companions** of power — and, where cheap, of wind speed and rpm —
    must move with their means.

  **Why this gates the issue rather than merely improving it.** If curtailment is the only thing
  in the dataset that ever moves pitch, then a detector keying on pitch is trivially correct and
  the fixture has measured nothing. That is the same artefact class R3 hit, where the fixture's
  mean/min mismatch was an impossible channel relationship the screen could have keyed on instead
  of the Cp shift it was supposed to detect — caught late, and only by re-running with it fixed.
  The curtailment signature must differ from the upgrade signature *in kind*, not merely be the
  only signature present.

- **Method: detection and exclusion.** `NormalOperationFilter`'s docstring already claims
  curtailment is in scope — "downtime, **curtailment**, frozen/stuck sensors" — but its three
  checks are finite power, an availability counter and stuck data, none of which catch a turbine
  that is available, reporting, and simply derated. Either the filter grows a check that does, or
  the docstring stops claiming it. Filtering stays on **cause, not effect**: the rule may read
  pitch, rpm and the companion statistics, never "power lower than the model expected", which
  would drop genuine low-uplift records and bias the estimate toward the upgrade working.

**Done when:** a curtailment episode injected into the test turbine's campaign moves the
estimate by materially less than it moves an unfiltered one, in **both prepost and toggle**;
truth is defined over normal operation; the excluded rows are discovered rather than declared;
and the clean campaigns are unchanged, so nothing is paid for on a campaign with no incident.

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
  package's home (a skeleton W1a fills). Adopt the conventional `src/` layout.
- Repoint every importer (notably the `v0_binned` baseline), tests, `pyproject`
  packaging, and examples. Distribution name **stays `res-wind-up`**; only import names
  change (`wind_up` = v1, `wind_up_v0` = legacy).
- Decide `benchmarking/`'s final home and confirm it is **excluded from the v1.0.0
  release artifact** (it is the eval harness, not the product).

**Done when:** the repo builds and tests pass under the new layout; v0 still runs as
`wind_up_v0` and its committed benchmark is **unchanged** (behaviour-preserving); no
importer still references the old `wind_up` path for the legacy tool.

---

## W1a — The composed `wind-up` method: compose and declare (after R4)

**Goal:** a single headline method named **`wind-up`** — the v1 deliverable — that
composes the winning pieces and self-configures from a **declared** `CampaignSpec`.

Hoisted ahead of C3–C6 because nothing here depends on them, and because W3 cannot start
without it. What W1a delivers is the *interface*; W1b settles whether it is *right*.

**Scope**
- **Grow a truth-free path — the bulk of the work.** There is none today: `CampaignRunner`
  takes a `SyntheticDataset` (which requires `original_df`), `Replicate` requires the same,
  and `score_one` requires a `truth` argument. Every code path that runs a method is a
  benchmark path that knows the answer, so real data cannot currently be run at all. W1a
  extracts a truth-free core — `estimate_campaign()` over frame-level northing and clipping —
  and puts `CampaignRunner` back on top of it as a thin truth-adding layer, its public
  behaviour unchanged. This is what makes W3's isolation structural rather than promised: the
  code that writes the analyst report has no access to the answer key. The analyst report
  carries a reference-turbine self-uplift table (`MethodOutput.reference_uplifts`) alongside
  per-turbine and farm uplift, which every real Hill of Towie report gives equal billing.
- **`wind-up` is campaign-level, not a `Method`** — so it is *not* a new entry in
  `benchmarking/baselines` alongside the other v1 methods. At method level it would be a
  no-op relabel of `power_model`: the R3 reference-validity screen is already inside
  `PowerModelMethod` (`reference_screen=True` by default), and the R1 northing step is
  already applied farm-wide, upstream of every method. Northing runs once for the whole farm,
  so it cannot sit inside a per-turbine method without misrepresenting its scope, and the farm
  aggregation, the guards and the report are campaign-level already. `wind-up` is therefore
  the shared northing step, plus one **`power_model`** built from the accepted defaults, plus
  the truth-free report, behind one name and one YAML declaration. R4 added no adaptation
  layer to compose (see its scope correction): what it left behind is attribution and a
  pool-shrinkage warning, already inside those parts. `toggle_specialist` inclusion is **TBD**
  and is settled in W1b, not here. The multi-method `carried_forward_methods` path is
  untouched — the benchmark comparisons keep running three methods.
- **A campaign is declared, not scripted** (moved here from W2, which is too late for
  W3 to use it). `CampaignSpec` gains a simple user-facing declaration — a YAML file it
  initializes from — so an analyst describes turbine roles, timing, exclusions and
  northing without writing Python. This is the v0 `WindUpConfig.from_yaml` ergonomics
  carried into v1, with the method config that v0 mixes into the same file kept on the
  method instead.
- A runner entry point that takes a declaration and writes an output directory, so a
  campaign can be run without importing anything.

**Done when:** `wind-up` runs self-configured from a YAML-declared `CampaignSpec` on the
C1/C2 campaigns, and W3 can be attempted against it. The composition is expected to keep
moving as C3–C6 land — that is W1b's business, and W3 tests documentation and output
legibility, not API stability.

---

## W1b — The composed `wind-up` method: validate (terminal)

**Goal:** evidence that the composed `wind-up` is the right method, not just a runnable
one.

**Scope**
- Validate `wind-up` as the headline method across the campaigns (C1–C6) and the
  failure modes (R1–R4), in **both prepost and toggle**.
- Settle `toggle_specialist` inclusion **with evidence** — it may be the toggle arm, or
  the composed `power_model` path may suffice.

**Done when:** `wind-up` tracks truth on the campaigns and stays invariant under the
failure modes in both modes; the exact composition (including the `toggle_specialist`
decision) is settled and recorded.

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

- **Document the declaration W1a delivered**, and fold in every gap W3 found — that
  list, not guesswork, is what the release documentation has to answer.

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

---

## W3 — The analyst dry run: can someone use this from the documentation alone? (continuous, from W1a)

**Goal:** measure whether an analyst holding only the documentation and a campaign prompt
can run a campaign and correctly say what happened in it — and turn every place they
cannot into a documentation or output-legibility fix.

The C-series asks whether wind-up *measures* a realistic campaign correctly. This asks
the separate question of whether anyone can *use* it to do so. Both halves are known by
construction, because the synthetic generator already records ground truth, so the
answer is objectively scoreable rather than a matter of opinion.

**Shape**
- **Generator side.** Build a synthetic dataset from a known upgrade profile and a known
  fault set, plus a short campaign prompt (campaign brief) written as narrative — the prose an owner would
  send an analyst. Ground truth is recorded out of band.
- **Analyst side.** A fresh agent is given the prompt, a YAML declaration to populate
  (W1a), a runner script, the documentation under test, and — after the run — the output
  directory. It populates the declaration, runs the campaign, inspects the outputs, and
  states **(a)** the upgrade class and magnitude and **(b)** which faults were present.
- **The menu is given, not hidden.** A real analyst knows which upgrade and fault classes
  exist, so the analyst is told the candidate set. The question is which one and how big,
  not guessing an unbounded space.

**Isolation is by construction, not by instruction**
This repo contains its own answer key: `findings_campaigns.md` names the profiles and
fault classes, `benchmarking/synthetic/` *is* the generator, and the development notes
describe both. "Not allowed to read the source" is unenforceable for an agent with a
filesystem. The analyst therefore runs in a directory containing **only** the brief, the
declaration, the runner, the documentation under test and the output directory — never a
checkout.

**Scoring separates three different failures**, because they have three different fixes:

| the analyst… | what it means | who fixes it |
|---|---|---|
| could not produce a run at all | the declaration or its documentation is unusable | W1a / W2 docs |
| ran it but could not interpret the outputs | the outputs are not self-explanatory | the C issue that produced them |
| interpreted them and concluded wrong | the method or its diagnostics mislead | the method |

**The transcript is the deliverable**, as much as the score: where the analyst hesitated,
what it guessed at, and what it went looking for and did not find, is the requirements
list for W2's documentation. Runs are stochastic and model-dependent, so conclusions come
from the pattern across repeats — a single failure is a signal, not a verdict.

**Done when:** the dry run is repeatable on demand, has been run against at least one
campaign per C issue as that issue lands, and W2's documentation answers every gap it
found. It **gates W2**: a release whose documentation has never been tested by someone
who cannot read the source is not ready to tag.
