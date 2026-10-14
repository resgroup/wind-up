# How wind-up chooses the analysis period and power references

A prepost campaign compares a test turbine's power before its upgrade with its power after, predicted
from reference turbines that were not changed. Which dates count as "before" and "after", and which
turbines count as unchanged, cannot be settled separately: a reference upgraded a month after the
test turbine is unchanged over a short after period and changed over a long one. In a staggered
rollout every test turbine faces this trade-off differently.

`wind_up.analysis_period.plan_analysis` makes the choice for one test turbine, from declared facts
only: every turbine's works windows, every turbine's data extent, the declared exclusions, the layout,
and a few settings. It never reads power. It returns an `AnalysisPlan`: the span, the power
references, the reserves, every other turbine's role with the reason, and whether the pool rule held.
The campaign runner (`benchmarking/campaigns/plans.py`) calls it once per upgraded turbine for every
campaign, with a declared span used as given, and writes each plan in the report's Part B folders
(`07_periods/periods.csv` with an `analysis_timeline_<turbine>.png` per turbine, and
`08_candidates/candidates.csv`). The span's two sides are the before and after periods of method
step 7: the before period runs from the span's start to the start of the test turbine's works, the
after period from the end of the works to the span's end.

The figure shows the plan for T13, the AeroUp trial turbine on Hill of Towie, over the 2019 to 2024
record. Each row is a turbine, the test turbine first and the rest nearest first with their distance
in rotor diameters (D): grey is data, red is works, hatching is an exclusion. The chosen span is
shaded, blue for the before period and green for the after period.

![T13's analysis timeline on Hill of Towie](images/analysis-period/analysis_timeline_T13.png)

The before period runs the full 24 months back from T13's works. The after period stops on 2022-08-30, the
day the nearest turbine T14 started its own AeroUp works, because a span that ran on would lose T14
as a power reference. T10 and T08, 16 D away, started their works earlier in August 2022 and so are
waking only: ending the span before their works would have kept them eligible as reserves, at the
cost of 26 days of after period, and the selector prefers the longer side. T07, 17 D away, is a
reserve: eligible, within reach, but beyond the nearest four. The farm-wide exclusion from September 2022 to February 2023 falls after the span. The
figures on this page are drawn by `benchmarking/baselines/study_analysis_period_doc_plot.py`.

## What makes a turbine eligible as a power reference

Over a candidate span `[start, end)`, another turbine is an eligible power reference when all three
hold:

1. **it has data over the whole span**: its data extent is its first finite active power to its
   last, plus one timebase;
2. **no works window of its own overlaps the span**: a turbine changed entirely before the span
   starts or entirely after it ends is fine; one changed during it is not;
3. **it is within 20 rotor diameters** (`max_reference_distance_d`) of the test turbine.

Distance stands in for how well two turbines' power correlates. Other upgraded turbines are eligible
on the same terms, so a rollout's earlier test turbines can serve as references for its later ones
once their own works are over.

## How a span is chosen

The search is exact over a finite set of candidate edges, so there is no optimiser to get stuck.
The candidate **starts** are:

- the test turbine's own first data;
- the longest before period sought, `works start − pre_cap` (24 months);
- the edge that yields `pre_cap` of *usable* time once exclusions are subtracted (`reach`, which
  iterates because masked time can push an edge back into more masked time);
- every other candidate's works window **end**, where that ends before the test turbine's works;
- every candidate's own **data start**, so the selector can give up some before period to reach a
  reference whose data begins late.

The candidate **ends** mirror these, with `side_cap` (12 months) in place of `pre_cap`, works
window **starts**, and candidates' data ends.

Every start × end combination is evaluated. A span is **viable** when both periods are at least
`min_side` (3 months by default) of usable time *after subtracting exclusions*, and it has at least
`min_references` (3) eligible power references.

The viable spans are then ranked, in this order:

| rank by | detail |
|---|---|
| 1. the pool rule | of the test turbine's 4 nearest candidates (`pool_size`), the **nearest must be eligible** and at least 3 of the 4 must be eligible within 20 D |
| 2. the longer short period | `min(before, after)`, capped at 12 months |
| 3. the longer after period | capped at 12 months |
| 4. the longer before period | capped at 24 months |
| 5. more power references | |
| 6. the nearer furthest reference | |
| 7. the later start, then the earlier end | a determinism tiebreak; among otherwise equal spans it prefers the shortest |

The pool rule is a strong **preference**, not a filter: a span that breaks it is still chosen if
nothing else is viable, and the plan records `pool_rule_met: false` with the reason.

The chosen span's power references are the **nearest `k`** (4 by default) of its eligible
candidates. Everything else gets a role and a reason, recorded in `candidates.csv`:

- **reserve**: eligible and within reach, but beyond the nearest `k`. Reserves are what the
  reference screen promotes from when it rules a reference out.
- **waking only**: not eligible, with the reason: `works … overlap the span`, `no data over the
  whole span`, `beyond 20 D`, or `not offered as a reference`.

## When no span is viable

If no start and end give both periods `min_side` with `min_references` eligible references, the
turbine is **not analysed**. The runner reports it in `periods.csv` with the reason, leaves it out
of the estimate and the farm headline, and continues with the other turbines. Only when no turbine
can be planned does the campaign fail.

The second figure shows why. T14 was upgraded in the 2022 rollout, when every other turbine on the
farm was upgraded within a few months of it. Any span with three months on each side of T14's works
overlaps the works of every neighbour, so the only turbine unchanged across it is T13, upgraded the
year before. One reference is not enough.

![T14's analysis timeline on Hill of Towie](images/analysis-period/analysis_timeline_T14.png)

An analyst has three ways out, in order of preference: wait for more data and a longer after
period, which does not help here because the neighbours were changed too; declare the span and the
references, accepting references whose works overlap it; or lower `min_side`, trading precision for
a reading at all. A rollout designed with wind-up's
[campaign design utility](designing-a-campaign.md) avoids the problem by keeping references
unchanged for each test turbine.

## A declared span

Declaring `analysis_period`, for the campaign or per turbine, skips the search: the span is used as
given, and the candidates are still ranked by distance with the nearest `k` as power references.
Declared `references` whose works overlap that span are **forced in** as power references, with a
warning. A declared span with no power reference at all raises.

A toggle campaign always declares its span. Its works window is the start of toggling, of no
length, so its before period is the data before toggling began and its after period the toggle
period; the off and on records inside the latter are what the method compares.

## Settings

`PlanSettings` holds the knobs. Every length is usable time: calendar time minus the test turbine's
exclusions.

| setting | default | meaning |
|---|---|---|
| `k` | 4 | power references per test turbine |
| `pre_cap` | 24 months | the longest before period sought |
| `side_cap` | 12 months | the longest short period, and the longest after period, sought |
| `min_side` | 3 months | the shortest period a chosen span may have |
| `pool_size` | 4 | how many nearest candidates the pool rule looks at |
| `pool_min_eligible` | 3 | how many of those must be eligible and within the distance limit |
| `max_reference_distance_d` | 20 | the furthest a power reference may be, in rotor diameters |
| `min_references` | 3 | the fewest power references a chosen span may have |

The benchmark's prepost campaign matrix lowers `min_side` to 14 days so that synthetic rollouts
with short gaps between works still yield a reading; a real campaign should not.

## The data screen

The plan's power references and reserves are then screened on their data (method step 9, pass 1), in
`benchmarking/campaigns/data_screen.py`. The selector chose them from the timeline; the screen checks
that each one's records can support the measurement. It runs once per test turbine, on the span, before
any method sees the plan, so the measurement screen of pass 2 starts from the references it keeps.

**Records.** Each candidate is paired with the test turbine alone, and the pair's records go through the
record selection of method step 11 (`benchmarking/harness/record_selection.py`). Each lost record is
labelled with the first reason that applies, in this order:

| reason | the record is lost because |
|---|---|
| `excluded` | the test turbine or the candidate is inside a declared exclusion or a works window |
| `subject invalid` | the test turbine's record is not valid for uplift (step 2), or missing |
| `reference invalid` | the candidate's record is not valid for uplift, or missing |
| `changed upwind of subject` | a turbine is upwind of the test turbine (step 5) while it counts as changed |
| `changed upwind of reference` | a turbine is upwind of the candidate while it counts as changed |
| `upwind unknown` | the upwind turbines of either are unknown |
| `part waking or missing upwind of subject` | the test turbine's waking scenario (step 5) is unknown or names a part waking turbine |
| `part waking or missing upwind of reference` | the candidate's waking scenario is unknown or names a part waking turbine |

A turbine is changed when it has a works window overlapping the span; the test turbine is always
changed. When it counts as changed depends on how far each works window is from the test turbine's
changeover (method step 11 iii). A window less than 6 months from it makes the turbine changed over
the whole span. A window further away makes it changed only on the far side: from the start of the
works for a change after the changeover, until the end of the works for one before it. A neighbour
upgraded a year after the test turbine therefore costs only the records from its works onward, not its
upwind directions over the whole span. The measurement itself uses the same selection over the test turbine and all its power
references at once, so a record is used only when every pair would keep it, and then applies the two
rules that need the whole reference set: the waking configuration rule of method step 11 ii (`rare
waking configuration`: the scenarios of the subject and its references together have under 24 hours in
either period) and, for a toggle test, the pairing filter of step 11 iv (`unpaired toggle record`: no
usable record of the other state within one toggle cycle). The hours lost to every reason in each period
are written as `record_losses.csv` in the turbine's measurement folder. Without the step 5 columns
(the prepost matrix and the synthetic paths skip Part A) the upwind and scenario reasons are skipped,
with a warning.

**Judging.** For each candidate the screen counts the usable hours before and after (for toggle, the
off and on records) and the Pearson correlation of its power with the test turbine's over every usable
record, both periods pooled. The medians are taken over every candidate, power references and reserves
alike. A candidate fails when its hours in either period are below `hours_fraction` of the median, or
its correlation is more than `correlation_margin` below the median correlation. A candidate with no
usable records has no correlation and fails.

**The walk.** The candidates are walked in rank order. The first `k` that pass are the power
references, the other passers stay reserves in order, and the failures become wake contributors.
Fewer than `k` passers keeps those that pass and warns that the pool rule is broken. Declared references are screened like any other. The step 8 table keeps the ranking from
before the screen.

`DataScreenSettings` holds the two thresholds, passed to `estimate_campaign` and `run_declaration` as
`data_screen` (`None` turns the screen off):

| setting | default | meaning |
|---|---|---|
| `hours_fraction` | 0.5 | the fewest usable hours a candidate may have in each period, as a fraction of the median |
| `correlation_margin` | 0.2 | how far below the median correlation a candidate's may fall |

## What the run writes

Under the report's `B_campaign/`, for every upgraded turbine (the data screen's files for every analysed one):

| file | what it says |
|---|---|
| `07_periods/periods.csv` | one row per upgraded turbine: its before and after periods and their lengths, whether the span was declared, whether the pool rule held (with the reason when it did not), any warning of a possible undeclared change inside its periods, and for a turbine no span could be found for, the reason |
| `07_periods/analysis_timeline_<turbine>.png` | the figures on this page: the span over every turbine's data, works and exclusions, each turbine labelled with its role |
| `08_candidates/candidates.csv` | every other turbine for each upgraded turbine, ranked by distance, with its role, the reason it is not a power reference, and any warning of a possible undeclared change inside the span |
| `09_screen/reference_data.csv` | one row per test turbine and candidate: its rank, distance, usable hours before and after, correlation, the three medians, and its verdict (`kept`, `spare` or `dropped`) with the reason when dropped |
| `09_screen/reference_hours.csv` | per test turbine, the joint usable hours before and after for the nearest 1, 2, ... of the screened references (kept first, then spares) |
| `09_screen/data_screen_<turbine>.png` | each candidate's hours before and after and its correlation, in rank order, against the thresholds, coloured by verdict, with the `k` cut-off |
| `09_screen/record_losses_<turbine>.png` | each candidate's records of the span split by loss reason |
| `09_screen/reference_hours_<turbine>.png` | joint usable hours against the number of references, with the number chosen |
| `09_screen/reference_map_<turbine>.png` | the layout: the test turbine, the candidates coloured by verdict and labelled with their usable share, the changed turbines ringed (solid when changed over the whole span, dashed and dated when changed from or until a date), and a wind rose of the test turbine's waking direction over the span |
| `09_screen/correlation_map_<turbine>.png` | the layout with each candidate as a bubble sized and coloured by its correlation; a dropped candidate has a dashed outline |

The warnings come from the possible yaw-alignment changes found in method step 4: one inside a
test turbine's before or after period, or inside a power reference's or reserve's span, is reported
against that turbine. It is a warning, not a bar: the analyst decides whether to declare the change
or exclude the turbine.

`wind_up.analysis_period_plots.plot_analysis_timeline` draws the figure from the same inputs the
selector used, so what it shows is what the selector saw.
