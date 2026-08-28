# Design — C1: campaign declaration + runner + farm rollup + placebo campaign

**Date:** 2026-08-28
**Status:** USER REVIEW NEEDED
**Issue:** C1 in `docs/v1/issues_campaigns.md`
**Parent design:** `docs/superpowers/specs/2026-08-27-realistic-campaigns-design.md`

## Problem

The v1 platform can inject a known uplift and score methods, but exercising it is
hand-wired (`benchmarking/baselines/inspect_wake_steering_case.py` hard-codes roles,
loads northing by hand, loops participants manually). C1 stands up the whole
declaration → runner → farm-rollup → reporting pipeline on the simplest case — a
**placebo** (zero injected uplift) whole-farm campaign — so a campaign is *declared*
rather than wired, and every method (and v0) is shown to report ~0.

## Scope

- A private, generator-facing `CampaignDefinition` and the public `CampaignBrief`
  derived from it.
- A `CampaignRunner` that turns a brief + generated dataset into per-turbine results,
  a farm rollup, and both output shapes.
- A pure `farm_rollup` headline function.
- Two placebo campaigns — one prepost, one toggle — declared once and run end-to-end.

Out of scope: uncertainty/P95 (Phase 1 is P50-only); the seam/context decision (C2);
any new estimator research.

## Key decisions

### 1. Module split: analyst-usable → `src/`, benchmark-only → `benchmarking/`

The test for `src/wind_up` (the product): a thing goes there only if it is purely
usable on real data with no synthetic-data or harness assumptions.

- **`src/wind_up/rollup.py`** — `farm_rollup(...)`. Pure: per-turbine
  `(estimate, treated_energy, n_records, rated_power_kw)` → one headline number. No
  toggle/synthetic/harness concepts. Depends only on numpy/pandas.
- **`benchmarking/campaigns/`** (new, benchmark-only, may import from `wind_up`):
  - `definition.py` — `CampaignDefinition` (private; holds injected upgrades = ground
    truth), `.generate()` → `SyntheticDataset`, `.brief()` → `CampaignBrief`.
  - `runner.py` — `CampaignRunner`.
  - `report.py` — the whole-farm inspection report (tables + plots vs truth).
  - a concrete placebo driver (prepost + toggle), alongside the `inspect_*` scripts.

`CampaignBrief` stays in `benchmarking/campaigns/` for C1. It is analyst-usable in
principle, but its `upgrade_timing` field references `ToggleSchedule`, which is **not**
cleanly `src`-ready: `ToggleSchedule` models a *perfectly regular* toggle (the
general real-data case is the seam's explicit `toggle_df`), and its "which timestamps
are on" semantics (`treated_mask`) live in `benchmarking/synthetic/generator.py`.
Promoting the brief and a settled toggle-timing type to `src` is deferred to W1/W2,
after C2 decides the seam/context representation — avoiding a second churn of the
toggle abstraction.

### 2. `CampaignBrief` — public facts only

Fields: `upgraded_turbines`, `upgrade_timing` (`pd.Timestamp` prepost /
`ToggleSchedule` toggle — mode inferred from the type), `candidate_references`,
`excluded_turbines`, `coords` (wtg → (lat, lon)), `north_offsets`
(`(wtg, ts, offset)`), `rated_power_kw`, `window` (`(start, end)`), `turbine_col`.

It carries **no** injected-upgrade physics. `CampaignDefinition.brief()` derives it by
dropping the `upgrades`; a test asserts the brief exposes no upgrade magnitude.

### 3. Farm rollup — pooled energy ratio with estimated, guarded counterfactuals

The truth is scoped to **treated records only** (post for prepost, on-blocks for
toggle), so pre-period relative energy is the wrong weight. The method cannot observe
the counterfactual post energy, so it estimates it from its own per-turbine uplift.

Per upgraded turbine, over its treated records:

- `Tᵢ` = actual observed treated-period energy (Σ finite active power); `Nᵢ` = count of
  those records.
- Estimated counterfactual energy `Ĉᵢ = Tᵢ / (1 + ûᵢ)`, where `ûᵢ` is the method's
  per-turbine P50.

Guards on `Ĉᵢ` (bite only in perverse cases):

1. **Capacity-factor cap** — implied mean counterfactual power `Ĉᵢ / Nᵢ` must not
   exceed `rated_maxᵢ = max(pre_rated, post_rated)`; clip `Ĉᵢ = rated_maxᵢ · Nᵢ` if it
   does. Catches `ûᵢ → −1` inflating the counterfactual. (Placebo: pre = post =
   nameplate; C6's rated-change campaign supplies both.)
2. **Non-negativity floor** — floor `Ĉᵢ` at 0, dropping any turbine whose `ûᵢ < −1`
   (negative counterfactual) or whose `Tᵢ` is negative, from the weighting.

Headline (estimate side): `(Σᵢ Tᵢ) / (Σᵢ Ĉᵢ) − 1`.
Headline (ground truth): same shape, exact, from `original_df`:
`(Σᵢ synthetic treated energy) / (Σᵢ original counterfactual energy) − 1` — the
N-turbine generalization of the existing `true_net_uplift`.

`farm_rollup` also returns the **per-turbine uplift spread** and a **guard-fired flag**
per turbine, so the "similar effect across turbines" assumption is checkable and any
clip/drop is visible rather than silent.

Rationale (see `src/wind_up_v0/combine_results.py`): v0 uses
inverse-variance weighting for its ordinary fleet total and energy weighting for the
*net* (wake-steering) case (`calc_net_uplift`, weight `mean_power_pre`). Inverse-variance
needs a per-turbine σ that Phase-1 methods (`oracle`, `naive_ratio`, `power_model`) do
not produce, and the campaign headline question is "how much energy did the fleet
gain" — an energy-weighted quantity. Once a P95/σ model lands (a later phase),
`farm_rollup` can offer an inverse-variance variant.

### 4. `CampaignRunner`

Takes the `CampaignBrief` and the generated `SyntheticDataset`. For each upgraded
turbine:

- Build the applicable carried-forward methods (skip `toggle_specialist` on prepost);
  build a per-turbine `MethodInput` (synthetic subset over the window, `test_wtg`,
  `upgrade_timing`, `turbine_col`). The thin seam is kept; the runner orchestrates
  (C2 revisits how much moves onto the seam).
- Run each method → collect `MethodOutput`s.

Then:

- Per-turbine truth via `dataset.true_uplift(mask=treated)`.
- **Estimate rollup** via `wind_up.farm_rollup`; **truth rollup** via the pooled ratio
  from `original_df`.
- **n=1 harness rows** by wrapping the campaign as one `Replicate` + one
  `CampaignWindow` spanning it and calling `score_one` per upgraded turbine — reusing
  the harness's truth alignment, no new scoring code.

v0 is included but optional (slow).

### 5. Two output shapes

1. **Inspection report** — per-turbine + rollup tables (estimate / truth /
   signed_error per method), the per-turbine uplift **spread**, any **guard-fired
   flags**, plus diagnostic plots (reusing `conditional_truth_vs_estimate` /
   `plot_conditional_uplift`). The whole-farm analogue of `inspect_wake_steering_case`.
2. **n=1 harness number** — the tidy leaderboard-style frame from `score_one`.

### 6. Placebo campaigns

Two `CampaignDefinition`s (one prepost, one toggle), ~6 HoT turbines, a **6-month
window Jan–June inclusive** (spanning a few seasons), `upgrades=[]` (zero injected
uplift → `synthetic == original` → truth = 0 by construction). A couple of turbines are
nominally "upgraded", the rest are candidate references, optionally one excluded. No
new no-op upgrade type is needed.

## Architecture

```
CampaignDefinition (benchmarking; private: injected upgrades = ground truth)
   │  .generate()                    .brief()
   ▼                                    ▼
SyntheticDataset ──► true per-turbine  CampaignBrief (benchmarking; public facts)
   │                 & farm uplift          │
   └──────────────┬───────────────────────┘
                  ▼
           CampaignRunner (benchmarking)
             per upgraded turbine: build method(s) + MethodInput → MethodOutput
             per-turbine truth; wind_up.farm_rollup (src) for the estimate headline
                  │
      ┌───────────┴────────────┐
      ▼                        ▼
inspection report        score_one at n=1
(tables + plots vs truth) (leaderboard-style rows)
```

## Testing

- Unit tests for `farm_rollup` guards: `ûᵢ → −1`, `ûᵢ < −1`, implied CF > 100%,
  negative `Tᵢ`; and the normal (no-guard) case reproducing the pooled ratio.
- `CampaignDefinition.brief()` exposes no upgrade physics.
- A fast placebo run on a **tiny synthetic fixture** (not the full HoT download)
  asserting every method's per-turbine and rollup estimate is ~0 within tolerance, in
  both modes, and that both output shapes are produced.

## Done when

Each placebo campaign is declared once and run end-to-end; every method's per-turbine
and farm-rollup estimate is ~0 within tolerance (both modes); the inspection report and
the n=1 harness number are both produced. `poe all-fast` green.
