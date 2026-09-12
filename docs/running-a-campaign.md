# Running a campaign with wind-up

wind-up measures the **energy-yield uplift** of a change made to some wind turbines:

```
uplift = (energy the turbine produced) / (energy it would have produced unchanged) - 1
```

The second term is never observed, so wind-up estimates it from **reference turbines** — turbines
on the same site that did *not* change — and reports the difference. A campaign is described in
one YAML file and run from the command line.

This page assumes the upgraded turbines are already chosen. To choose them, see
[designing a campaign](designing-a-campaign.md).

## 1. Describe the campaign

```yaml
name: my_campaign          # identifies the run; also the default output directory name

data:
  scada: data/scada.parquet     # long-format SCADA, one row per turbine per timestamp
  schema: hill_of_towie         # the column vocabulary the SCADA is keyed by
  turbines: data/turbines.csv   # Name, Latitude, Longitude

turbines:
  upgraded:   [T06, T11]        # the turbines whose uplift you want
  references: [T01, T02, T03]   # turbines you are willing to compare them against
  excluded:   []                # turbines never to use as a reference; their wake still counts
  rated_power_kw: 2300.0

timing:
  mode: prepost
  changeover: 2018-01-01T00:00:00Z

analysis_period:
  start: 2017-01-01T00:00:00Z
  end:   2019-01-01T00:00:00Z   # exclusive

northing:
  discover: true
```

### The fields that need a decision

**`turbines.upgraded` / `references` / `excluded`.** Every turbine holds at most one role. Only
`references` are compared against. Every other turbine in the SCADA — upgraded, excluded, or not
listed at all — still enters each estimate for its wake, through whether it was running and where
it pointed, never its power. Put a turbine in `excluded` when it must never be a reference — for
example, it was down for rebuild, or it had its own separate change. Leaving it unlisted has the
same effect; listing it records the decision.

**More references is better.** Reference count is the single biggest lever on accuracy: a handful
of references is noticeably worse than fifteen. Offer every turbine you have no reason to distrust
and let the automatic screen (below) rule out the ones that misbehave.

**`timing.mode`.**

- `prepost` — the change was made once, at `changeover`, and stayed. Everything before is
  baseline, everything after is treated.
- `toggle` — the change was switched on and off in alternating blocks, so both states are sampled
  in the same weather. Needs `start` and `period` (a full on/off cycle, e.g. `100min`).

**`analysis_period`.** `end` is **exclusive**. For `prepost`, give the treated period **the same
length as the baseline, and the same months of the year** — a full year of baseline against six
months of treated data compares two different seasons, and the seasonal difference lands in your
answer as if it were uplift. `toggle` does not have this problem, because its blocks interleave.

**`northing.discover`.** Nacelle position sensors drift and get re-zeroed, which corrupts every
direction-dependent calculation. `true` (the default) finds those steps in the data and corrects
them, writing plots of what it did. Use `false` with a `table:` only when you already have a
trusted correction table.

**Timezones.** Every time is UTC. A time written without a timezone is read as UTC; a time written
with an offset is converted to UTC. Whatever you write, the resolved values are echoed into
`resolved_campaign.json` in the output — check it if a result looks shifted.

**Reanalysis is not declared.** wind-up fetches the weather reanalysis it needs by itself, from
the centre of every turbine in `turbines.csv` — the whole site, not just the turbines this
campaign names, so changing the roles never moves it. The first run for a site downloads it;
later runs on that site reuse the cache.

## 2. Run it

```
python -m benchmarking.campaigns run campaign.yaml --out out
```

Run it from the directory holding `campaign.yaml`, or give the path to it; the `data:` paths inside
the declaration are read relative to the declaration itself, so a campaign folder can be moved or
copied whole.

`--out` is the directory the report is written to, and it is used exactly as given: `name` does
**not** add a subdirectory under it. Give each run its own `--out`, or a second run will overwrite
the first. Omit `--out` and the report goes to `$WIND_UP_BENCHMARKING_OUTPUT_DIR/<name>` instead.

The run also writes its log to `<out>/run.log`. Read it: some of what the run decided is reported
nowhere else, in particular the reference screen's thresholds, how big a pool it had, and whether
it stopped early.

Expect roughly **three minutes per upgraded turbine**, plus a couple of minutes for the shared
northing step. It is not stuck.

## 3. Read the outputs

| file | what it says |
|---|---|
| `per_turbine.csv` | the uplift estimate for each upgraded turbine |
| `farm_uplift.csv` | one headline number for the whole campaign, weighted by energy |
| `farm_uplift_detail.csv` | how each turbine contributed, and whether any was dropped |
| `reference_stability.csv` | **each reference estimated as if it were a test turbine** |
| `conditional.csv` and `conditional/` | uplift split by wind speed, turbulence and power |
| `northing/` | the direction corrections that were discovered, with plots |
| `resolved_campaign.json` | the campaign as wind-up understood it |

### Start with `reference_stability.csv`

This is the campaign checking its own references. Each candidate reference is run through the same
analysis as a test turbine. **A healthy campaign reads near 0% for every reference**, because
nothing happened to them. A reference reading well away from zero either changed during the
campaign, or has a sensor fault, or is being waked differently — and if it is being used as a
reference, its problem is now inside your headline number with the sign reversed.

Rows with `screened = True` were ruled out automatically and contributed no power to the estimate.
Rows with `screened = False` that still read far from zero are the ones to think about; the screen
is deliberately cautious and only runs on campaigns long enough to judge.

### Judging scale from that one table

`reference_stability.csv` is also your noise yardstick, and it comes from the same single run. The
references are turbines where nothing happened, so **the spread of their readings is what this
campaign can resolve**. Compare your upgraded turbines against that spread:

- upgraded turbines sitting inside the reference spread → not distinguishable from zero
- upgraded turbines clearly outside it, and agreeing with each other → a real effect

You should not need to re-run the campaign with a different reference set to answer that. If you
find yourself wanting to, say so in your report — it means this run did not give you enough to
judge on, which is a limitation of the tool rather than of the data.

### Then the headline

`farm_uplift.csv` carries `estimate`, plus:

- `uplift_spread` — the gap between the best and worst turbine. A wide spread on a change that was
  applied identically to every turbine is a reason to look at `per_turbine.csv` before believing
  the headline.
- `n_guarded` — how many turbines were dropped from the aggregate. Anything above 0 means
  `farm_uplift_detail.csv` will name them and say why.

### Reading zero

An estimate is never exactly zero, and a small number is not automatically a real effect. Two
things help you judge scale: the spread across turbines in `per_turbine.csv`, and how far the
references sit from zero in `reference_stability.csv`. If your upgraded turbines and your
references are the same distance from zero, you are looking at the noise floor, not a change.

## Known limits

- Every turbine's change must share one date (or one toggle schedule). Staggered per-turbine dates
  are not yet expressible.
- Every turbine's data is trusted for its wake. A turbine whose power reads high while it is in
  fact stopped would be counted as waking its neighbours.
- The result is a P50 estimate. There is no uncertainty interval yet.
