# wind-up v1 — realistic-campaigns issues (drafts)

A **fresh tranche** of v1 work: simulate a few realistic, whole-farm campaigns —
modelled on the real Hill of Towie open-source analyses
(`resgroup/hill-of-towie-open-source-analysis`) — and mature the v1 methods so a
user **declares a short campaign brief and the method does the right thing**
(role assignment, northing, filtering, data split, reference selection and
validity), automatically. Then compare the v1 methods and v0 on these campaigns.

**Relationship to the earlier v1 issues.** The prior effort in
[issues.md](issues.md) (Issues 1–19) and its [findings.md](findings.md) are on the
**back-burner** for this tranche — still valuable for later phases, not the current
focus. Design spec for this tranche:
`docs/superpowers/specs/2026-08-27-realistic-campaigns-design.md`.

## Ground rules for this tranche

- **Methods carried forward:** `oracle`, `naive_ratio`, `power_model`,
  `toggle_specialist`. **`rlearner` is dropped** (see C7).
- **Estimand:** per-turbine uplift **plus a result representative of the upgrade using the whole farm data** (one
  headline campaign number, as the real HoT analyses report).
- **One simulated instance per campaign, no replicates.** This tranche is about
  realism and ease-of-use, not sampling statistics.
- **Two declarations, one source:** a private `CampaignDefinition` (drives the
  generator; holds the injected-upgrade physics = ground truth) and the public
  `CampaignBrief` derived from it (the facts an analyst would know; **never** the
  truth). Methods only see the brief.
- **Both outputs per campaign:** a farm-wide inspection report (per-turbine + rollup
  vs truth) **and** the campaign through the harness scoring path at n=1.

## Suggested order

C0 → C1 → C2 → C3 → C4 → C5 → C6. C7 (drop `rlearner`) is independent and can be
scheduled at any time.

---

## C0 — Housekeeping: start the new issues list, back-burner the old

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

## C1 — Campaign declaration + runner + farm rollup + placebo campaign

**Goal:** stand up the whole pipeline on the simplest case — a **placebo** (zero
injected uplift) whole-farm campaign — proving every method (and v0) reports ~0 and
that a campaign can be *declared* rather than hand-wired.

**Scope**
- **`CampaignDefinition`** — the private, generator-facing declaration: turbines and
  their roles (upgraded / reference / excluded), upgrade timing (prepost changeover
  or `ToggleSchedule`), the injected upgrade(s) (here: none / a no-op), site context
  (coords, northing corrections, ERA5 handle), window.
- **`CampaignBrief`** — derived from the definition, public facts only: upgraded
  turbines, timing, mode, site coords + northing, candidate references, exclusions.
  No injected-upgrade physics.
- **`CampaignRunner`** — brief → for each upgraded turbine, construct the applicable
  carried-forward methods (skip `toggle_specialist` on prepost) + a per-turbine
  `MethodInput`, run, collect `MethodOutput`s. Keep the thin seam for now;
  orchestration lives in the runner (C2 revisits this).
- **Farm rollup** — energy-weighted aggregation of per-turbine uplift to one
  campaign headline; the generator supplies the matching farm-level ground truth
  (upgraded synthetic energy vs counterfactual baseline energy over the campaign
  window). Pin the exact weighting definition here.
- **Both output shapes** — a per-campaign inspection report (per-turbine + rollup
  tables and diagnostic plots vs truth, the whole-farm analogue of
  `inspect_wake_steering_case`), and the campaign fed through the existing harness
  scoring path at n=1.
- v0 included but optional (slow).

**Done when:** a placebo whole-farm campaign is declared once and run end-to-end;
every method's per-turbine and farm-rollup estimate is ~0 within tolerance; both the
inspection report and the n=1 harness number are produced.

---

## C2 — Seam / campaign-context decision

**Goal:** decide how the `CampaignBrief` should reach the methods so they
self-configure (northing, filtering, data split, role assignment, reference
validity) — the question deliberately deferred at design time.

**Scope**
- Weigh the options with C1 in hand and the later campaigns' needs visible:
  (a) keep the thin `MethodInput`/`MethodOutput` seam with the runner orchestrating;
  (b) enrich `MethodInput` to carry the brief so methods read it at estimate time;
  (c) a hybrid (brief both builds methods and rides on the input).
- Record the decision and its rationale; refactor C1's runner/seam to match before
  the demanding campaigns build on it.

**Done when:** a documented decision exists and the code reflects it; C3+ build on
the chosen shape.

---

## C3 — Blade enhancement (AeroUp), prepost

**Goal:** a realistic **prepost** single-/few-turbine blade-enhancement campaign
where reference selection, the prepost split and northing all follow from the brief.

**Scope**
- `CampaignDefinition` using a region-2 Cp gain tailing to 0 at rated (AeroUp shape)
  on the upgraded turbine(s); other farm turbines as candidate references.
- **Automatic reference selection** from the brief (exclude other upgraded / excluded
  turbines; honour candidate list).
- Prepost data split and northing applied by the method/runner without hand-wiring.
- Report + n=1 score for all applicable methods (`toggle_specialist` N/A here) and
  v0.

**Done when:** the campaign is declared and run whole-farm; per-turbine and rollup
estimates track truth; the report shows how each method used references and the
prepost split.

---

## C4 — TuneUp (controller), toggle, multi-turbine

**Goal:** a realistic **toggle** campaign with ~9 upgraded turbines and a
TI/stability-shaped effect, exercising multi-turbine toggle handling and the rollup
at scale.

**Scope**
- `CampaignDefinition` with a condition-dependent (stability/TI-shaped) Cp change on
  ~9 turbines, a `ToggleSchedule` (~50-min period, per the real trial), ~10
  references.
- Multi-turbine toggle: per-turbine estimates across many upgraded turbines, the
  campaign-only data split, and the energy-weighted rollup across them.
- `toggle_specialist` and the other methods self-configure from the brief.

**Done when:** the campaign runs whole-farm; per-turbine + rollup estimates track
truth for all methods and v0; the report scales to many upgraded turbines.

---

## C5 — Dynamic Yaw (wake steering + collective control), toggle

**Goal:** the hard campaign — generalise today's manual
`inspect_wake_steering_case` hacks into **declared** behaviour: inter-turbine wake
dependencies, references whose validity changes with wind direction, northing-sector
logic, and excluded turbines.

**Scope**
- `CampaignDefinition` using the existing `WakeSteering` upgrade across the farm
  (plus a collective-control component if in scope), a `ToggleSchedule`, and an
  excluded turbine (e.g. T07).
- **Wake-aware reference validity:** the method must gate references by direction so
  a reference sitting in an upgraded turbine's changed wake is dropped for those
  timestamps — declared from geometry in the brief, not a script-level filter.
- Northing-sector handling folded into the method/runner (replacing the
  `wd_filter` hack), so no bespoke driver code.
- Report + n=1 score; the farm rollup nets upstream steering losses against
  downstream gains.

**Done when:** the wake-steering campaign is declared once and run whole-farm with
no manual per-pair/sector wiring; reference validity is handled automatically;
per-turbine and net-farm estimates track truth.

---

## C6 — Rated-power up/downrate

**Goal:** exercise the region-3 / rated-power path none of the other campaigns
reach.

**Scope**
- `CampaignDefinition` using `RatedPowerChange` (an uprate and/or a downrate) on a
  set of turbines, prepost or toggle.
- Confirm the methods behave sensibly when the effect is concentrated at/around
  rated power (where baseline and upgraded both clip), including the conditional /
  per-bin views.

**Done when:** the rated-change campaign runs whole-farm; per-turbine and rollup
estimates track truth; the rated-power behaviour is visible in the report.

---

## C7 — Disentangle and remove `rlearner`

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
