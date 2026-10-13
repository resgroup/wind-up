# How wind-up handles a prepost campaign

A **prepost** campaign changed the turbine once and left it changed. Everything before the
changeover is the baseline; everything after is the upgraded period. This page covers only what
prepost adds to, or changes about, [the core estimator](estimating-uplift.md).

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
TODO exclusions are a general concept that toggle tests would use too but they are not explained in docs/estimating-uplift.md

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
| **`headline_estimator="reversal"`** | applies to prepost only. It also fits the reverse direction — train the upgraded rows, predict the baseline rows — and reports `sqrt((1+r_fwd)/(1+r_rev)) − 1`, so a shrinkage common to both directions cancels. **Off by default**; toggle always keeps the forward ratio |
| **campaign-proximity time decay** | a prepost-only lever, because in toggle the on and off rows interleave inside the campaign interval and every row weighs 1. **Off by default** since [CF22](v1/findings_campaigns.md): on a record whose pre and post periods were the same days interleaved — truth 0 — it moved the reading +0.154 pp, and with it off the disguised-prepost and toggle legs agreed to the last digit. The tilt works against the thing a full-year baseline is chosen for, which is covering every season evenly |

## Known weak spots

These are measured, not suspected. The evidence is the prepost campaign matrix
(`benchmarking/baselines/study_prepost_campaign_matrix.py`), whose small size runs Penmanshiel and
Kelmarsh rollout campaigns at three injected magnitudes — including **zero**, where the correct
answer is exactly 0 — and the probes recorded in [findings_campaigns.md](v1/findings_campaigns.md).

**Accuracy gets worse as the post period grows.** From the `82c6398` small run, over the 18
placebo (truth 0) test-turbine readings at each post length:

| post length asked | post days delivered | sd of the reading | worst reading |
|---|---|---|---|
| 3 months | 90–110 | **0.34 pp** | 0.68 pp |
| 12 months | 266–326 | **0.86 pp** | 1.68 pp |

More data, a worse answer. That is the seasonal mismatch doing its work: a longer post period sits
further from the baseline it is compared against. [CF24](v1/findings_campaigns.md) measures the same
thing from the other side — balancing the two periods so they hold the same months record for
record removes about two thirds of the placebo bias, and nearly all of the between-turbine spread.

**A long post period is also not the post period you asked for.** The window runs to the last trial
turbine's works end plus the requested length, and the selector then truncates it at the first
rollout turbine whose works fall inside. So "12 months" delivered 266 to 326 days above, and at
Kelmarsh it also cost a power reference (4 asked, 3 chosen).

**Reference downtime has large leverage, and it is not symmetric about the changeover.** Adding a
few random outages to the reference turbines moved one Penmanshiel test turbine's placebo reading
from −0.37 pp to −1.6 pp. Three outages touched its four references; two of them fell in the post
period, costing it about 1.3 % of its post-period reference records, and the third fell in the pre
period. Its own reference-stability readings moved by at most 0.19 pp over the same change, so the
sanity check did not see what moved the headline.

**The two exclusion channels disagree.** Across 141 reference readings in that run, `"booleans"` and
`"nan"` dropped bit-for-bit identical record counts, on identical plans and identical reference
sets, and still produced estimates differing by up to 0.38 pp (median 0.09 pp). Same rows in,
different answer out.

**A reference reading is not independent of the upgrade.** A reference's reading should not change
when the injected upgrade's magnitude changes, since none of its own data moves. It does, by a
median of 0.12 pp and up to 0.35 pp, monotonically in the magnitude — the signature of the upgraded
turbine's scaled power reaching the waking booleans the references are read against
([CF25](v1/findings_campaigns.md) §1).
