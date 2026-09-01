# Design — realistic whole-farm campaigns & self-configuring v1 methods

**Date:** 2026-08-27
**Status:** approved design; feeds a fresh issues list at `docs/v1/issues_campaigns.md`
**Branch of work:** new tranche (developed off the current `v1` line)

## Problem

The v1 platform can already inject a known uplift and score methods on synthetic
data, but exercising it is still very manual. `benchmarking/baselines/inspect_wake_steering_case.py`
is the state of the art and it hard-codes turbine roles, hand-loads the northing
corrections, applies a script-level wind-direction-sector filter "hack", loops over
participants by hand, and bespoke-configures v0. Nothing about a campaign is
*declared*; it is all wired up by the driver.

The next step to mature v1 is to **simulate a few realistic, whole-farm campaigns**
— modelled on the real Hill of Towie open-source analyses — and mature the v1
methods so that a user **declares a short campaign spec and the method does the
right thing**: role assignment, northing, filtering, data split, reference
selection and reference validity, all automatically. Then we can see how the v1
methods and v0 compare on realistic campaigns.

## Scope & non-goals

**In scope**
- Five realistic whole-farm campaigns, **one simulated instance each, no
  replicates** (this tranche is about realism and ease-of-use, not sampling
  statistics).
- Maturing the carried-forward methods so campaign behaviour is *declared*, not
  hand-wired.
- A per-campaign comparison of v1 methods and v0 against the single ground truth.

**Methods carried forward:** `oracle`, `naive_ratio`, `power_model`,
`toggle_specialist`. **`rlearner` is dropped completely** (see C7 — it is entangled
with `power_model` and needs disentangling first).

**Explicit non-goals for this tranche**
- No replicate/ensemble statistics, no uncertainty-model development (the existing
  `toggle_specialist` σ machinery is used as-is where it exists, not extended here).
- No new estimator research on `power_model`'s internals beyond what a campaign
  demands.
- The pre-existing `docs/v1/issues.md` and `docs/v1/findings.md` effort goes on the
  **back-burner**; its knowledge stays valuable for later phases.

## Key decisions (from brainstorming)

1. **Estimand: per-turbine estimates + a farm-level energy-weighted uplift.** Each
   upgraded turbine gets its own uplift estimate and truth (as today), and each
   campaign additionally yields **one headline farm number**, energy-weighted —
   matching how the real HoT analyses report a single P50. For wake
   steering the farm uplift naturally nets upstream steering losses against downstream
   gains.

2. **Two declarations, one source of truth.** A full **`SyntheticCampaign`** drives
   the synthetic generator and includes the *secret* injected-upgrade physics (the
   ground truth). The **`CampaignSpec`** is *derived* from it and carries only the
   *public* facts a real analyst would have — upgraded turbines, upgrade timing,
   mode (prepost/toggle), site coords + northing, candidate references, exclusions —
   and **never the truth**. Methods only ever see the spec.

3. **A campaign runner** turns a spec into results: for each upgraded turbine it
   constructs the applicable carried-forward methods (skipping `toggle_specialist`
   on prepost, etc.) and a per-turbine `MethodInput`, runs them, collects the
   per-turbine `MethodOutput`s, computes the **farm uplift**, and emits both output
   shapes. v0 is included in the comparison but optional (it is slow).

4. **Both output shapes per campaign.** (a) A farm-wide **inspection report** —
   per-turbine + farm-uplift tables and diagnostic plots vs the single ground truth, the
   whole-farm analogue of `inspect_wake_steering_case` — as the human-facing
   artefact; **and** (b) the single campaign wired through the existing harness
   scoring path at **n=1** for a comparable, leaderboard-style number.

5. **How the spec reaches the methods is deliberately left open.** Whether the
   spec drives an orchestration layer above today's thin `MethodInput`/`MethodOutput`
   seam, enriches `MethodInput` directly, or a hybrid, is its **own investigation
   issue (C2)**, decided once the placebo pipeline exists (C1) and the demanding
   campaigns have surfaced their real needs (northing, wake-free reference gating).

## The five campaigns (in build order)

Modelled on the three real Hill of Towie trials, plus a rated-power case and a
placebo:

1. **Placebo** — whole farm, **zero injected uplift**. The starting point: it forces
   the whole declaration → runner → farm-uplift → reporting pipeline into existence
   on the simplest possible case, and is a real honesty check (every method, and
   v0, must report ~0 with no false uplift).
2. **Blade enhancement (AeroUp)** — **prepost**, some upgraded turbine(s),
   region-2 Cp gain tailing to 0 at rated. Forces automatic reference selection,
   the prepost data split, and northing.
3. **TuneUp (controller)** — **toggle**, ~9 upgraded turbines, a TI/stability-shaped
   effect. Forces multi-turbine toggle handling, the campaign-only split, and the
   farm uplift at scale.
4. **Dynamic Yaw (wake steering + collective control)** — **toggle**, whole farm.
   The hard one: inter-turbine wake dependencies, **references whose validity
   changes with wind direction** (wake-aware reference selection / wake-free
   gating), northing-sector logic, and excluded turbines (e.g. T07). Generalises
   today's manual `inspect_wake_steering_case` hacks into *declared* behaviour.
5. **Rated-power up/downrate** — exercises the region-3 / rated-power path that none
   of the others reach.

## Architecture sketch

```
SyntheticCampaign  (private: injected-upgrade physics = ground truth)
        │  generate_dataset(...)
        ▼
   SyntheticDataset  ──────────────► true per-turbine & farm uplift
        │                                     ▲
        │ derive                               │ score
        ▼                                      │
   CampaignSpec  (public facts only) ─► CampaignRunner
                                            │  per upgraded turbine:
                                            │    build method(s) + MethodInput
                                            │    run → MethodOutput
                                            ▼
                                   per-turbine results
                                            │  energy-weighted
                                            ▼
                                     farm uplift (headline)
                                            │
                         ┌──────────────────┴──────────────────┐
                         ▼                                      ▼
              inspection report                     harness scoring (n=1)
           (tables + plots vs truth)              (leaderboard-style number)
```

- The **runner** owns method selection by mode, per-turbine input construction, and
  the farm uplift; it is where "the method does the right thing" is orchestrated until
  C2 decides how much of that moves onto the seam.
- **Farm uplift** is an energy-weighted aggregation of per-turbine uplift to a
  single campaign number; the generator provides the matching farm-level ground
  truth (upgraded-turbine synthetic energy vs counterfactual baseline energy over the
  treated records).

## Issue decomposition

Tracked in `docs/v1/issues_campaigns.md`. Summary and ordering:

- **C0 — Housekeeping (first, small, standalone).** Create the new issues doc; mark
  the old `issues.md`/`findings.md` back-burnered with a pointer forward.
- **C1 — Campaign declaration + runner + farm uplift + placebo campaign.** The
  foundation: `SyntheticCampaign`/`CampaignSpec`, the runner, the farm uplift, both
  output shapes, all four carried-forward methods, v0 optional, on the placebo.
- **C2 — Seam / campaign-context decision.** The deferred architecture question,
  decided using C1's experience and the later campaigns' needs.
- **C3 — Blade enhancement (prepost):** automatic reference selection + prepost
  split + northing.
- **C4 — TuneUp (toggle, multi-turbine):** multi-turbine toggle + farm uplift at scale.
- **C5 — Dynamic Yaw (wake steering):** wake-aware reference validity + northing
  sector + exclusions.
- **C6 — Rated-power up/downrate:** the region-3 / rated path.
- **C7 — Disentangle & remove `rlearner`.** `power_model/method.py` imports
  `make_outcome_model` from `rlearner/nuisance.py` (and
  `inspect_era5_matching_importance.py` does too; `era5_sync.py` was already
  promoted out of rlearner). Relocate the outcome-model factory and any other shared
  bits into `power_model`/a shared module, repoint the importers, then delete the
  `rlearner` package, its tests, and the rlearner-only inspect scripts
  (`inspect_prepost_feature_ablation.py`, the rlearner arm of
  `inspect_era5_matching_importance.py`). Independent of the campaign work; schedule
  any time.

**Ordering:** C0 first → C1 foundation → C2 right after C1 → C3–C6 build on the
foundation → C7 slots in whenever.

## Risks & open questions

- **Seam shape (C2).** The thin seam is a deliberate design property today; enriching
  it to carry campaign context trades that for methods that self-configure. C2 is the
  place to weigh it, with real usage in hand.
- **Reference validity under wake steering (C5).** Declaring "candidate references"
  is not enough when the upgrade itself changes wakes; the method must gate
  references by direction. This is the least-understood piece and may itself spawn
  follow-up issues.
- **Farm uplift weighting.** Energy-weighting is the obvious default; the exact
  definition (per-turbine campaign MWh, counterfactual vs actual) is a C1 design
  detail to pin against the generator's farm-level truth.
- **`rlearner` disentangle (C7).** The shared `make_outcome_model` must move without
  changing `power_model` behaviour (the committed benchmark should stay identical).
```
