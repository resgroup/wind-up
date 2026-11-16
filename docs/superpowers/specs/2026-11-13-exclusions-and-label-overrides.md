# Exclusions and label overrides

Status: draft for review, 2026-11-13. Written for wind-up `v1`; it names no private campaign, so it can be
moved to the wind-up repo as it is.

## Problem

A v1 campaign declares `exclusions`: `(turbine or ALL, start, end)` windows whose data "is not used". In
practice they mean three different things depending on where the code reads them, and none of them is applied
everywhere:

| Stage | What an exclusion does today |
|---|---|
| Step 2, operating-state labels and label checks | Nothing: excluded records are labelled and checked like any other |
| Northing (preparation) | Nothing: `north_tables` learns from every record, and the run applies those tables as declared offsets (`CampaignPreparation.with_north_offsets`), so excluded data reaches northing |
| Waking state | A turbine's own exclusion is "held back" (`CampaignSpec.held_back_mask`): the rows stay and keep their automatic waking state. A farm-wide exclusion or a works window drops the rows entirely |
| Span and reference planning (`wind_up.analysis_period`) | Subtracted as masked time when measuring each side of the span; data extents come from finite power, whatever the label |
| Uplift record selection (step 11) | Applied, through the `usable` frame |
| Plots: step 2, northing, waking, `11_valid_records` | Not applied: the plots show excluded records as used |

Two consequences besides misleading plots:

1. An excluded period that holds bad data still feeds northing.
2. An excluded period that holds valid data from a turbine in an unusual state keeps its automatic label. A
   turbine idling through a controller test is labelled normal operation, so the waking rules treat it as
   waking its neighbours when it wakes nothing.

The word "exclusion" also hides the analyst's intent. Two different things get declared:

- **The data is wrong** (a logger fault, frozen signals, a clock jump): it should count as missing.
- **The data is right but the turbine was not in normal operation, and the labels do not say so** (a test,
  curtailment): the turbine's state should be recorded correctly, so every stage treats it as that state.

## Design

### 1. Two declarations

- **`exclusions`**: `(turbine or ALL, start, end)`, end exclusive, with no other field. The records are
  labelled `missing`. Use when the data is wrong.
- **`label_overrides`**: `(turbine or ALL, start, end, state)`, end exclusive. The records take the named
  state. Use when the data is right but the automatic labels miss what the turbine was doing.

Works windows are unchanged: they mark the change itself, not a data state.

### 2. Where an override's state comes from

The state must be defined in the site's operating-state config: a generic state (`partial downtime`, `full
downtime`, ...) or a site label with its northing, waking and uplift validity, declared once like any other
site label. An override naming an undefined state is a configuration error raised when the campaign loads. One
state therefore has one definition, and the state-hours table, colours and plots treat it like any other site
label.

### 3. Precedence

Applied in step 2, in this order (later wins):

1. The automatic labels (generic rules and site labels).
2. `label_overrides`.
3. `exclusions`.

So an exclusion beats an override, and both beat the automatic labels. Two overrides that overlap on the same
turbine and timestamp are an error raised when the campaign loads: there is no sensible winner. An exclusion
overlapping an override is allowed (the data inside a test can still be wrong).

The label checks run after overrides and exclusions, so they only flag what no declaration explains.

### 4. Everything downstream reads the labels

Once step 2 has applied the declarations, no later stage reads `exclusions` or `label_overrides` again. Each
stage reads the state columns (`state`, `valid_northing`, `waking_state`, `valid_uplift`):

- **Northing** learns from records valid for northing, in preparation and in the run alike. The preparation's
  tables and the run's are learned from the same records.
- **Waking** reads `waking_state`. The held-back concept (`CampaignSpec.held_back_mask`, the `held` mask in
  `campaigns.run._row_masks`) is removed: every row stays in the frame with its label.
- **Uplift record selection** reads `valid_uplift`. The `usable` frame keeps only what labels cannot express:
  works windows and the declared period.
- **Span and reference planning** (`wind_up.analysis_period`, `campaigns.plans`): see section 6.

### 5. `missing` is the same as absent

Every part of v1 must treat a record labelled `missing` exactly as if that turbine-timestamp had no row. Today
the two can differ. For example:

- `campaigns.plans.data_extents` reads each turbine's extent from finite power. A record that is present but
  labelled `missing` (out-of-range power, an exclusion once this change lands) still extends the extent.
- `state_hours` counts labelled-missing records as missing hours, but an absent row counts nowhere.
- Code that reindexes onto the shared index fills validity with `False` but `waking_state` with `NaN`, not
  `missing`.

wind-up v0 avoids this by filling in every missing turbine-timestamp early in the analysis. v1 does the same:

- **Fill the grid at load.** After loading and before step 2, reindex each turbine onto the full
  turbine x timestamp grid of the campaign's timebase. The filled rows have NaN signals and are labelled
  `missing` by the existing rule (non-finite power or counter). From step 2 on, every turbine has a row at every
  timestamp.
- **Audit every consumer** for code that treats a present row as data whatever its label: extents, coverage
  fractions, row counts in titles and logs, `dropna`-style cuts, "first/last record" logic, reindexes with a
  fill value. Each one either reads the labels or relies on the full grid. The audit's findings and fixes are
  listed in the implementation plan.
- **An invariance test pins it.** Take a campaign, drop a random subset of rows labelled `missing` (including
  whole leading and trailing stretches of a turbine), rerun, and require identical results, warnings and
  plotted record counts. The test runs on the synthetic fixtures and on one real campaign.

### 6. Span and reference planning reads usable time from the labels

`wind_up.analysis_period` currently measures each side of a span as elapsed time minus masked exclusions, and
requires a power reference to have "data over the whole span" from its finite-power extent. Both become
statements about records valid for uplift:

- A side's length is the time the test turbine is valid for uplift on that side, not elapsed time minus
  exclusions.
- A turbine's extent is its first and last record valid for uplift, not its first and last finite power.
- A power reference qualifies over a span when it is valid for uplift over at least a set share of it (a new
  `PlanSettings` field; the current behaviour corresponds to 100% of non-excluded time). The default share is
  decided in the implementation plan from the Hill of Towie and synthetic campaigns.

The planner takes a per-turbine series of uplift validity (on the full grid of section 5) in place of
`exclusions` and `extents`. The `Exclusion` parameter of `plan_analysis` is removed.

### 7. Plots show what each stage used

- **Step 2 (operating states):** drawn after overrides and exclusions, so excluded records appear as
  `missing` and overridden records under their declared state.
- **Northing views (`northing_used` / `northing_not_used`):** drawn on the records northing actually learned
  from.
- **Waking views:** drawn on `waking_state` after the declarations.
- **`11_valid_records/uplift_used_<turbine>.png`:** drawn on the records step 11 actually used, for the test
  turbine (`used_ts`) and for each power reference (its own used mask), not on `valid_uplift`. A companion
  `uplift_not_used_<turbine>.png` colours the records it did not use by reason: label not valid for uplift,
  outside the span or in a works window, a power reference not usable, an upwind turbine waking or part waking
  or missing. These are the reasons `harness.record_selection` already computes.
- Titles say what was drawn ("records used for uplift by step 11", "records valid for northing").

## Migrating existing declarations

Every existing declaration is reviewed and each window becomes an exclusion or a label override. Known so far:

| Declaration | Window | What it is | Becomes |
|---|---|---|---|
| `campaigns/real.py` `HOT_AEROUP_T13` | ALL, 2022-09-03 to 2023-02-12 | Curtailment | Label override `curtailment`, unless the Hill of Towie derate label already catches it record by record, in which case the window is dropped |
| `campaigns/real.py` `HOT_TUNEUP` | ALL, 2022-09-03 to 2023-02-12 | Curtailment | As above |
| `campaigns/real.py` `HOT_TUNEUP` | ALL, 2024-03-14 07:00 to 18:00 | TuneUp deployment | Works windows: a works-table row 2024-03-14 07:00 to 18:00 for each of the nine upgraded turbines, replacing the shared `timing.changeover` (the works table then gives the timing). The implementation plan confirms from the source data whether any other turbine was worked on that day; if so it gets a row too |
| Synthetic studies (`prepost_matrix`, `study_analysis_period_doc_plot`, test fixtures) | Generated | Test inputs | Exclusions, unless a test is about a non-normal state |

A `curtailment` site label is suggested as valid for northing (the turbines yaw normally), `part waking` (a
reduced wake), not valid for uplift. Each site defines its own.

Results will move, mainly because curtailed records start to inform northing and the waking rules. The Hill of
Towie reference results are rerun and every change explained before the golden values are updated.

Campaigns outside the wind-up repo review their own declarations the same way.

## Testing

- Loader: an override naming an undefined state raises; overlapping overrides on one turbine raise; an
  exclusion overlapping an override is accepted and wins.
- Labelling: precedence (automatic, then override, then exclusion), farm-wide and per-turbine, end exclusive.
- Label checks: an override silences the check over its window; an exclusion turns the records missing.
- Northing: excluded records do not change the learned offsets; preparation and run tables are identical.
- Waking: a turbine overridden to a `not waking` state is not counted as waking its neighbours.
- Planner: side lengths and reference qualification follow uplift validity; the reference-share threshold.
- Missing equals absent: the invariance test of section 5.
- Plots: `uplift_used` record count equals the number of records step 11 used; northing views equal northing's
  learning set.
- Golden: Hill of Towie reruns, with the changes explained.

## Delivery

Three PRs, in this order, each on the previous one. Each says whether the Hill of Towie golden values moved
and why, so every change is attributable to one PR.

1. **The full grid, and `missing` equals absent (section 5).** Fill the turbine x timestamp grid at load,
   audit the consumers and fix what the audit finds, and add the invariance test. Standalone, and the rest
   relies on it. Results should not move; if a golden value does, the PR explains why.
2. **`exclusions` and `label_overrides` (sections 1-4 and 6).** The declarations and their loader checks,
   applied in step 2; northing, waking, uplift record selection and the planner reading the labels; the
   held-back concept removed; the `Exclusion` parameter of `plan_analysis` removed. Existing declarations keep
   working as exclusions. Results can move: northing stops learning from excluded records, excluded records
   count as missing for the waking rules, and the planner measures usable time from the labels.
3. **Plots and migration (section 7 and "Migrating existing declarations").** The plots drawn on what each
   stage used, the Hill of Towie declarations migrated (curtailment override or dropped, the deployment as
   works rows), the Hill of Towie golden results rerun and every change explained.

## Out of scope

- Inline validity on an override (rejected: one state, one definition).
- Changing how works windows are declared.
- New automatic label rules or label checks (separate work).
