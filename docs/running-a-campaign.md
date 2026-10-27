# Running a campaign with wind-up

wind-up measures the **energy-yield uplift** of a change made to some wind turbines:

```
uplift = (energy the turbine produced) / (energy it would have produced unchanged) - 1
```

The second term is never observed, so wind-up estimates it from **reference turbines** — turbines
on the same site that did *not* change — and reports the difference. A campaign is described in
one YAML file and run from the command line.

This page assumes the upgraded turbines are already chosen. To choose them, see
[designing a campaign](designing-a-campaign.md). For what wind-up does with the data once it has
it, see [the method](v1/method.md) and, for a campaign with a single
changeover, [how wind-up handles a prepost campaign](prepost-campaigns.md).

## 1. The campaign declaration

```yaml
name: my_campaign          # identifies the run; also the default output directory name

data:
  scada: data/scada.parquet     # long-format SCADA, one row per turbine per timestamp
  schema: hill_of_towie         # the column vocabulary the SCADA is keyed by
  turbines: data/turbines.csv   # Name, Latitude, Longitude, Rotor_Diameter_m
  # timestamps is optional; the default is period start, UTC
  timestamps: {convention: start, time_zone: UTC}

# site is optional; the default is onshore. An offshore farm MUST set offshore: true;
# wind-up does not detect it from the turbine positions.
site:
  offshore: false

turbines:
  upgraded:   [T06, T11]        # the turbines whose uplift is measured
  rated_power_kw: 2300.0
  # references and excluded are optional; the defaults are every other turbine, and none

upgrade:
  expected_uplift: 0.05         # the AEP uplift expected of the change, a fraction
  # wake_reach_d: 12            # optional: how far its wake change reaches, in rotor diameters

timing:
  mode: prepost
  changeover: 2018-01-01T00:00:00Z

analysis_period:
  start: 2017-01-01T00:00:00Z
  end:   2019-01-01T00:00:00Z   # exclusive

northing:
  discover: true
```

### Staggered changes: works dates instead of a changeover

When the turbines were changed on different days, the works dates are declared rather than a
changeover and a period:

```yaml
data:
  scada: data/scada.parquet
  schema: hill_of_towie
  turbines: data/turbines.csv
  works: data/works.csv          # Turbine, First date of works, Last date of works

upgrade:
  expected_uplift: 0.05

timing:
  mode: prepost                  # no changeover: each turbine's is the end of its works

exclusions:                      # optional; end exclusive
  - {turbine: ALL, start: 2022-09-03T00:00:00Z, end: 2023-02-12T00:00:00Z}
  - {turbine: T04, start: 2021-02-01T00:00:00Z, end: 2021-02-05T00:00:00Z}

# analysis_period omitted: wind-up chooses a span for each upgraded turbine
```

- **The works table** has a `Turbine` column and the first columns whose names start `First date`
  and `Last date` (the Zenodo Hill of Towie AeroUp table works as it is). Dates are whole days: a
  window covers the first day through the end of the last. Any turbine may appear, analysed or not,
  and a turbine may appear more than once. With no shared changeover, each upgraded turbine needs
  exactly one window, and its changeover is the end of that window. Its own works rows are never used.
  Two optional columns give a row its own wake reach, matched by prefix like the dates: `Expected
  uplift` (a fraction, turned into a reach as below) and `Wake reach` (rotor diameters), which wins.
  A row with neither takes the campaign's `upgrade` reach.
- **Exclusions** are periods whose data is not used, for one turbine or `ALL`. A turbine's *own*
  exclusion leaves its rows in the analysis marked unusable, so the model can still tell whether
  it was running and making a wake over them; a farm-wide exclusion removes the records outright.
- **The span.** For each upgraded turbine wind-up picks the start and end of its analysis, and its
  **power references**: the best-ranked 4 turbines with data over the whole span and no works inside
  it, within 20 rotor diameters. Turbines are ranked by distance, except that one the test turbine is
  upwind of for more than 20 % of the site's wind, within the wake reach, ranks behind every one it
  is not. A turbine changed entirely before or after the span is fine. A
  chosen span gives both sides at least 3 months of usable time and has at least 3 power
  references; when none does, the turbine is not analysed — it is listed under `unplanned` with
  the reason and left out of the farm headline, and the other turbines still run. Among the spans
  that qualify it prefers, in order: the nearest turbine and at least 3 of the 4 nearest
  qualifying, then the longest shorter side up to 12 months, then post up to 12 months, then pre
  up to 24 months, then more power references, then a nearer furthest reference. It will give up
  some pre or post period to reach a reference whose data starts late or ends early, if that is
  what the preferences above want. Exclusions do not count towards either side's length. Every
  other turbine enters only for its wake. [How wind-up handles a prepost
  campaign](prepost-campaigns.md) describes the whole selection.
- **To fix the span by hand**, `analysis_period` is declared, once for the campaign or per turbine
  (`analysis_period: {T13: {start: ..., end: ...}}`). Declared `references` are then used as power
  references even when their works overlap that span, with a warning.
- **A shared `timing.changeover`** is every upgraded turbine's changeover. With a works table beside
  it, the table holds every turbine's other works, the upgraded turbines' included; no works window
  of an upgraded turbine may cover the changeover.

### The fields that need a decision

**`turbines.upgraded` / `references` / `excluded`.** Only `upgraded` is required: leave the other
two out and every other turbine in `turbines.csv` is offered as a reference, with none excluded,
which is what most campaigns want. Every turbine holds at most one role, and only `references` are
compared against. Every other turbine in the SCADA — upgraded, excluded, or not
listed at all — still enters each estimate for its wake, through whether it was running and
nothing else: not its power, and not where it pointed, both of which a change to that turbine
could move. Put a turbine in `excluded` when it must never be a reference — for
example, it was down for rebuild, or it had its own separate change. Leaving it unlisted has the
same effect; listing it records the decision.

**More references is better** for a campaign declared with one changeover and period, which
compares against every reference offered. Reference count is the single biggest lever on its
accuracy: a handful of references is noticeably worse than fifteen. Every turbine with no known
reason for distrust should be offered, leaving the automatic screen (below) to rule out those that
misbehave. A campaign
declared by its works uses the best-ranked 4 instead; when the screen rules one out, the next best-ranked
takes its place.

**`upgrade`.** `expected_uplift` is required: the AEP uplift expected of the change, as a fraction
(a vendor's claim will do), negative for a change expected to lose energy. It sets the change's
**wake reach**: how far, in the changed turbine's rotor diameters, its changed wake is taken to
disturb another turbine. A record is lost when a changed turbine is upwind of the subject or of a
power reference within that reach (method step 11 iii); beyond it the change is ignored.

| Expected uplift (magnitude) | 0 | 1 % | 3 % | 5 % | 10 % or more |
|---|---|---|---|---|---|
| Wake reach | 0 D | 4 D | 8 D | 12 D | 20 D |

Between the points the reach is interpolated. Offshore (`site.offshore: true`) the reach is doubled,
up to 20 D, since wakes persist further over the sea. `wake_reach_d` overrides the table; `0` declares
that the change leaves the wakes as they were (it replaces the old `wake_unchanged: true`, which is
now refused). The reach also ranks the power references (see the span, above).

**`timing.mode`.**

- `prepost` — the change was made once, at `changeover`, and stayed. Everything before is
  baseline, everything after is treated.
- `toggle` — the change was switched on and off in alternating blocks, so both states are sampled
  in the same weather. Needs `start` and `period` (a full on/off cycle, e.g. `100min`).

**`analysis_period`.** `end` is **exclusive**. For `prepost`, longer is better on both sides: the
method ([step 7](v1/method.md#7-choose-the-before-and-after-periods)) seeks up to 24 months before and
12 after, and the reference turbines' own readings correct what the seasons leave in the result
(step 13). The model trains on the longer period once the after period reaches 9 months (step 12), and
a result whose training period is under 9 months at 50 % coverage carries a high-severity warning.
`toggle` interleaves its blocks, so the seasons cancel by construction.

**`northing.discover`.** Nacelle position sensors drift and get re-zeroed, which corrupts every
direction-dependent calculation. `true` (the default) finds those steps in the data and corrects
them, writing plots of what it did ([how northing works](northing.md)). `false` with a `table:` is
used only when a trusted correction table already exists.

**Timezones.** Every time is UTC. A time written without a timezone is read as UTC; a time written
with an offset is converted to UTC. The resolved values are echoed into `campaign_resolved.yaml`
in the output, which should be checked if a result looks shifted.

**`data.timestamps`.** Whether a SCADA timestamp labels the `start` or the `end` of its period, and
the time zone the SCADA is in. The SCADA timestamps must be timezone-aware and in that time zone,
or the run stops. wind-up checks the declaration against the reanalysis: it warns when the weather
lines up best with a shift of more than 30 minutes, and stops at 1 hour or more, which usually means
the time zone or convention is wrong.

**`site.offshore`.** Set `true` for an offshore farm. wind-up does not work this out from the
turbine positions, and without it the reanalysis for an offshore farm comes from a land grid cell
(one of similar elevation to the site, possibly on the nearest coast), whose wind is unlike the
farm's. The run log and `campaign_resolved.yaml` say which kind of cell was used (`land` or `sea`),
and should be checked for an offshore or coastal site.

**Reanalysis is not declared.** wind-up fetches the weather reanalysis it needs by itself, from
the centre of every turbine in `turbines.csv` — the whole site, not just the turbines this
campaign names, so changing the roles never moves it — over the whole SCADA record. The first
run for a site downloads it; later runs on that site reuse the cache.

## 2. Running a campaign

```
python -m benchmarking.campaigns run campaign.yaml --out out
```

The command is run from the directory holding `campaign.yaml`, or given the path to it; the `data:`
paths inside the declaration are read relative to the declaration itself, so a campaign folder can
be moved or copied whole.

`--out` is the directory the report is written to, and it is used exactly as given: `name` does
**not** add a subdirectory under it. Each run needs its own `--out`, or a second run overwrites the
first. Without `--out` the report goes to `$WIND_UP_BENCHMARKING_OUTPUT_DIR/<name>` instead.

`--reference-plots` adds the full set of measurement plots for each reference, the same plots a test
turbine gets. They go in `C_turbine_measurement/<reference>_reference_for_<test turbine>/`, one
folder per reference per test turbine it serves, since a reference shared by several test turbines
is measured once for each. Each reference is then refitted rather than read from the screen, so the
run is slower; it suits a campaign with few references.

The run also writes its log to `<out>/run.log`, which should be read: some of what the run decided
is reported nowhere else, in particular the reference screen's thresholds, how big a pool it had,
and whether it stopped early.

A run takes roughly **ten minutes per upgraded turbine** at the default periods on a laptop (the
reference readings and the held-out windows of step 14 are most of it), plus a couple of minutes for
the shared northing step.

## 3. The outputs

| file | what it says |
|---|---|
| `B_campaign/per_turbine.csv` | the uplift estimate for each upgraded turbine |
| `B_campaign/analysis_warnings.csv` | every warning the run raised, ranked by severity: high questions a headline (a training period under 9 months at 50 % coverage, [method step 12](v1/method.md#12-relate-the-reference-measurements-to-the-subject-turbine-measurements)), medium changed the inputs of a measurement (a reference the screens ruled out, an indicator of an undeclared change), low is for attention only (a candidate yaw-alignment change). The counts and the list also open the run's `README.md` and close its log |
| `D_farm/farm_uplift.csv` | one headline number for the whole campaign, weighted by energy |
| `D_farm/farm_uplift_detail.csv` | how each turbine contributed, and whether any was dropped |
| `B_campaign/09_screen/reference_stability.csv` | **each reference estimated as if it were a test turbine** |
| `B_campaign/15_uplift_distributions/` | `conditional.csv` and its plots: uplift split by wind speed, turbulence and power |
| `D_farm/wind-up/` | the campaign's uplift plots, in a folder per method that ran |
| `A_data_preparation/04_northing/` | the direction corrections that were discovered, with plots, and the possible yaw-alignment changes |
| `C_turbine_measurement/<turbine>/` | one folder per measured turbine: its CSVs (`record_losses.csv` is the hours lost to each rule of step 11) and its plots, one folder per step |
| `campaign_resolved.yaml` | the campaign as wind-up understood it: the declaration with every default filled in and every timestamp resolved to UTC |
| `B_campaign/07_periods/periods.csv` | each upgraded turbine's before and after periods, whether they were declared or chosen, whether the pool rule held (with the reason when it did not), and for a turbine no span could be found for, the reason |
| `B_campaign/07_periods/analysis_timeline_<turbine>.png` | the span drawn over every turbine's data, works and exclusions, each turbine labelled with its role ([how the analysis period is chosen](analysis-period-and-reference-selection.md)) |
| `B_campaign/08_candidates/candidates.csv` | every other turbine ranked by distance for each upgraded turbine — power reference, reserve or waking only — with the reason it is not a power reference and any warning of a possible undeclared change |
| `B_campaign/09_screen/reference_data.csv` | the data screen: each candidate reference's usable hours before and after and its power correlation with the test turbine, and whether it was kept, held as a spare or dropped, with plots of the same per upgraded turbine ([the data screen](analysis-period-and-reference-selection.md#the-data-screen)) |
| `B_campaign/09_screen/reference_hours.csv` | the usable hours left as references are added, nearest first |

### First, `reference_stability.csv`

This is the campaign checking its own references. Each candidate reference is run through the same
analysis as a test turbine. **A healthy campaign reads near 0% for every reference**, because
nothing happened to them. A reference reading well away from zero either changed during the
campaign, or has a sensor fault, or is being waked differently — and if it is being used as a
reference, its problem is now inside the headline number with the sign reversed.

Rows with `screened = True` were ruled out automatically and contributed no power to the estimate.
Rows with `unjudged = True` contributed no power either, for a different reason: the campaign does
not hold enough of their upgraded data for the screen to judge them, and a turbine the screen cannot
vouch for is not one the estimate leans on. A reference down for most of the campaign lands here.
Either way the turbine keeps its place in the analysis as a wake contributor, so the model still
knows when it was running.

Rows with both flags `False` that still read far from zero are the ones to think about. The screen is
deliberately cautious: it judges each reference separately, so one reference's long outage costs only
that reference its power, and on a campaign too short to judge any of them it does not run at all and
takes nobody's power away. `run.log` names who it ruled out and who it held out.

### Judging scale from that one table

`reference_stability.csv` is also the noise yardstick, and it comes from the same single run. The
references are turbines where nothing happened, so **the spread of their readings is what this
campaign can resolve**. The upgraded turbines are compared against that spread:

- upgraded turbines sitting inside the reference spread → not distinguishable from zero
- upgraded turbines clearly outside it, and agreeing with each other → a real effect

It should not be necessary to re-run the campaign with a different reference set to answer that.
If a re-run seems necessary, that should be stated in the report: it means this run did not give
enough to judge on, which is a limitation of the tool rather than of the data.

### Then, the headline

`farm_uplift.csv` carries `estimate`, plus:

- `uplift_spread` — the gap between the best and worst turbine. A wide spread on a change that was
  applied identically to every turbine is a reason to look at `per_turbine.csv` before believing
  the headline.
- `n_guarded` — how many turbines were dropped from the aggregate. Anything above 0 means
  `farm_uplift_detail.csv` will name them and say why.

### Reading zero

An estimate is never exactly zero, and a small number is not automatically a real effect. Two
things help to judge scale: the spread across turbines in `per_turbine.csv`, and how far the
references sit from zero in `reference_stability.csv`. If the upgraded turbines and the
references sit the same distance from zero, the reading is the noise floor, not a change.

## Known limits

- A toggle campaign shares one schedule. Staggered dates are prepost only, declared through a
  works table.
- Every turbine's data is trusted for its wake. A turbine whose power reads high while it is in
  fact stopped would be counted as waking its neighbours.
- The result is a P50 estimate with a one-sigma uncertainty ([step 14](v1/method.md#14-estimate-uncertainty)); there is no P95 yet.
