# How wind-up handles a prepost campaign

A **prepost** campaign changed the turbine once and left it changed. Everything before the
changeover is the baseline; everything after is the upgraded period. This page covers only what
prepost adds to, or changes about, [the method](v1/method.md).

The contrast is the simplest one wind-up has, and the hardest to get right. In a toggle campaign
the two states interleave in the same weather, so the comparison is between rows that sit minutes or hours
apart. In prepost the two states are **different calendar time**, so anything that changed between
them for a reason unrelated to the upgrade — the season, the long-term climate, a reference's own
drift, the wind farm's exposure, the controller behaviour of some or all turbines, the calibration of measurement instruments, the data's own gaps — lands in the answer looking exactly like uplift. Almost every
prepost-specific mechanism below exists to narrow that gap.

## The contrast

`resolve_toggle` (`benchmarking/harness/toggle.py`) turns the turbine's timing into three row sets.
For prepost, where the timing is a single timestamp:

- **upgraded** — `index >= changeover`
- **campaign baseline** — everything before it
- **training baseline** — the same rows

The last two differ only for toggle, where the lenient training baseline adds the pre-campaign rows
to the interleaved off-blocks. For prepost they are identical, which means the conditional step's
deliberate use of the strict `campaign_baseline` has no effect here.

Each upgraded turbine has **its own changeover**. With a works table, it is the end of that
turbine's works window; with a declared `timing.changeover`, every turbine shares one.

Which period the model trains on follows [method step 12](v1/method.md#12-relate-the-reference-measurements-to-the-subject-turbine-measurements):
the before period, unless the after period spans at least 9 months, is longer and holds more usable
hours, in which case the model trains on the after period, predicts the before period, and every
uplift is inverted to read as upgraded over baseline. A training period under 9 months at 50 %
coverage raises a high-severity analysis warning.

## Choosing the span and the power references

This is the largest piece of prepost-only machinery: `wind_up.analysis_period`
(`src/wind_up/analysis_period.py`), driven by `benchmarking/campaigns/plans.py`. It runs for every
campaign: a declared `analysis_period`, for the campaign or per turbine, is used as given, and
otherwise the span is chosen. Either way the power references are the nearest eligible candidates.

Each upgraded turbine gets its own span and its own reference set, because a staggered rollout
means every turbine's usable window is different. The span and the power references are chosen
together from the works windows, the data extents and the exclusions, never from anything measured:
a candidate is eligible when it has data over the whole span and no works inside it, and the span
is the one that keeps the nearest candidates eligible with the longest sides. A turbine no viable
span can be found for is reported under `unplanned` and left out of the farm headline. The rules,
the ranking and the settings are in [how the analysis period is chosen](analysis-period-and-reference-selection.md).

## Exclusions and held-back rows

Three different things remove a turbine's data, and wind-up distinguishes them:

| kind | effect | stays in the frame |
|---|---|---|
| the turbine's own **works window** | its data is never usable | no |
| a **farm-wide** exclusion | nobody's data is usable | no |
| the turbine's **own** exclusion | its data is not usable for uplift | **yes**, marked invalid |

Exclusions apply to a toggle campaign in the same way; the method describes them under
[step 1](v1/method.md#1-identify-changes-to-the-wind-farm) and [step 7](v1/method.md#7-choose-the-before-and-after-periods).

The third is what `held_back_mask` picks out. Those rows stay in the frame so that the waking
scenarios can still read *whether the turbine was running* over them, even though its power may not
be trusted.
They carry no north-calibrated direction, because the shared northing step runs on the usable rows
only.

Exclusions also shorten the sides: `pre` and `post` in a plan are calendar time **minus** the
masked time inside them, which is why a plan's `pre_days` can be well short of the span it covers.

### What a reference carries over its own exclusion

Nothing: its power and direction are missing across the exclusion, and the record is not valid for
uplift. Its waking state still reaches the other turbines' waking scenarios (method step 5).

## The parts of the core method that are prepost-only

| mechanism | behaviour |
|---|---|
| **the reference screen** | runs on prepost only. A reference is screened across the campaign's own changeover (`screening_timing`). Toggle is skipped entirely: a reference change before the test is common-mode across every on and off block, so the on/off contrast is blind to it, and splitting the test period in half leaves too little a side to see one |
| **reserves and refilling** | only a planned campaign supplies reserves, so only prepost refills a screened-out reference. A flat campaign's screened reference stays in the pool and goes power-free (contributes no features) instead |
| **the reference correction** | the default prepost headline ([method step 13](v1/method.md#13-calculate-the-uplift), option C): the forward ratio divided by the kept references' mean reading, so the estimator's own level on turbines that did nothing is removed; its standard error joins the sigma. Toggle keeps the forward ratio |
| **the training-period swap** | prepost only (method step 12): the after period trains the model when it is at least 9 months long, longer than the before period and has more usable hours |
| **the held-out training period** | prepost only ([method step 14](v1/method.md#14-estimate-uncertainty)): the training period is read window by window, each refitted on the rest, and the spread beyond the sampling sigma joins the headline's sigma |
| **`headline_estimator="reversal"`** | an alternative prepost headline, off by default: it also fits the reverse direction and reports `sqrt((1+r_fwd)/(1+r_rev)) − 1`. Not adopted on the campaign matrix (method.md, "Decided so far") |
| **campaign-proximity time decay** | a prepost-only lever, because in toggle the on and off rows interleave inside the campaign interval and every row weighs 1. **Off by default** since [CF22](v1/findings_campaigns.md): on a record whose pre and post periods were the same days interleaved — truth 0 — it moved the reading +0.154 pp, and with it off the disguised-prepost and toggle legs agreed to the last digit. The tilt works against the thing a full-year baseline is chosen for, which is covering every season evenly |

## What the campaign matrix measures

The evidence for every prepost choice is the prepost campaign matrix
(`benchmarking/baselines/study_prepost_campaign_matrix.py`): synthetic rollout campaigns on three open
sites at injected magnitudes including **zero**, where the correct answer is exactly 0, plus two real
campaigns whose references provide a truth of zero. [Method.md's validation section](v1/method.md#validation)
states the targets, the decisions they led to (no weather trimming, ERA5 out of the model, the
reference correction, the training-period swap, the sigma's terms) and where the method is valid; the
runs are recorded as CF25 onward in [findings_campaigns.md](v1/findings_campaigns.md). Two things the
matrix found that a prepost analyst should keep in mind:

- **A placebo reading's error is block-scale variation of the subject-to-reference relation** (CF33),
  present inside the before period as strongly as between the periods, which is why the sigma reads the
  held-out training period rather than a fixed model's resampling alone.
- **Reference downtime has leverage.** A row on which a power reference has no valid data is dropped
  from both periods (method step 11 i), because a few reference outages moved a placebo reading by
  more than 1 pp before that rule (CF27).
