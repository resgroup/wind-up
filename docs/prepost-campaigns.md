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
(`src/wind_up/analysis_period.py`), driven by `benchmarking/campaigns/plans.py`. It runs when the
campaign declares no shared `timing.changeover`, or declares its `analysis_period` per turbine
rather than as one span for the whole campaign (`CampaignSpec.uses_plans`). In practice that means
whenever the campaign is declared by a works table.

Each upgraded turbine gets its own span and its own reference set, because a staggered rollout
means every turbine's usable window is different.

### What makes a turbine eligible as a power reference

Over a candidate span `[start, end)`, another turbine is an eligible power reference when all three
hold:

1. **it has data over the whole span** — its data extent is its first finite active power to its
   last, plus one timebase;
2. **no works window of its own overlaps the span** — a turbine changed entirely before the span
   starts or entirely after it ends is fine; one changed during it is not;
3. **it is within 20 rotor diameters** (`max_reference_distance_d`) of the test turbine.

Distance stands in for how well two turbines' power correlates.

### How a span is chosen

The search is exact over a finite set of candidate edges, so there is no optimiser to get stuck.
The candidate **starts** are:

- the test turbine's own first data;
- the longest pre side sought, `changeover − pre_cap` (24 months);
- the edge that yields `pre_cap` of *usable* time once exclusions are subtracted (`reach`, which
  iterates because masked time can push an edge back into more masked time);
- every other candidate's works window **end**, where that ends before the test turbine's works;
- every candidate's own **data start**, so the selector can give up some pre period to reach a
  reference whose data begins late.

The candidate **ends** mirror these, with `side_cap` (12 months) in place of `pre_cap`, works
window **starts**, and candidates' data ends.

Every start × end combination is evaluated. A span is **viable** when both sides are at least
`min_side` (3 months by default) of usable time *after subtracting exclusions*, and it has at least
`min_references` (3) eligible power references. If none is viable, the turbine is **not analysed**
— it is reported under `unplanned` with the reason, left out of the estimate and out of the farm
headline, and the campaign continues with the others. Only when no turbine can be planned does the
campaign fail.

The viable spans are then ranked, in this order:

| rank by | detail |
|---|---|
| 1. the pool rule | of the test turbine's 4 nearest candidates (`pool_size`), the **nearest must be eligible** and at least 3 of the 4 must be eligible within 20 D |
| 2. the longer short side | `min(pre, post)`, capped at 12 months |
| 3. the longer post | capped at 12 months |
| 4. the longer pre | capped at 24 months |
| 5. more power references | |
| 6. the nearer furthest reference | |
| 7. the later start, then the earlier end | a determinism tiebreak; among otherwise equal spans it prefers the shortest |

The pool rule is a strong **preference**, not a filter: a span that breaks it is still chosen if
nothing else is viable, and the plan records `pool_rule_met: false` with the reason.

The chosen span's power references are the **nearest `k`** (4 by default) of its eligible
candidates. Everything else gets a role and a reason, recorded in `analysis_plans.csv`:

- **reserve** — eligible and within reach, but beyond the nearest `k`. Reserves are what the
  reference screen promotes from when it rules a reference out.
- **waking only** — not eligible, with the reason: `works … overlap the span`, `no data over the
  whole span`, `beyond 20 D`, or `not offered as a reference`.

The plan also records a **reading pool** per reference and reserve — the eligible turbines within
20 D *of that reference*, nearest first, the test turbine left out — which is what each reference is
read against in `reference_stability.csv`. A reference is therefore judged against its own
neighbours rather than against the test turbine's.

### A declared span

Declaring `analysis_period`, for the campaign or per turbine, skips the search: the span is used as
given. Declared `references` are then **forced in** as power references even when their works
overlap that span, with a warning. A declared span with no power reference at all raises.

## Exclusions and held-back rows

Three different things remove a turbine's data, and wind-up distinguishes them:

| kind | effect | stays in the frame |
|---|---|---|
| the turbine's own **works window** | its data is never usable | no |
| a **farm-wide** exclusion | nobody's data is usable | no |
| the turbine's **own** exclusion | its data is not usable for uplift | **yes**, marked invalid |
TODO exclusions are a general concept that toggle tests would use too but they are not explained in docs/estimating-uplift.md

The third is what `held_back_mask` picks out. Those rows stay in the frame so that a method can
still read *whether the turbine was running* over them, even though its power may not be trusted.
They carry no north-calibrated direction, because the shared northing step runs on the usable rows
only.

Exclusions also shorten the sides: `pre` and `post` in a plan are calendar time **minus** the
masked time inside them, which is why a plan's `pre_days` can be well short of the span it covers.

### `exclusion_channels`

What a still-powered reference carries over its own exclusions is a method setting with two values:

- **`"booleans"` (the default)** — the reference's `waking` and `normal_operation` booleans are
  recomputed from the **unfiltered** frame over its whole record and joined on, so they are present
  across the exclusion. Its power, power minimum and direction stay missing there.
- **`"nan"`** — no such join. Every column of that reference is missing across its exclusion.

This applies only to references that still carry power; a power-free reference already contributes
nothing but those booleans. The two settings therefore drop exactly the same records and differ
only in how much the model is told about the dropped stretch.

## The parts of the core method that are prepost-only

| mechanism | behaviour |
|---|---|
| **the reference screen** | runs on prepost only. A reference is screened across the campaign's own changeover (`screening_timing`). Toggle is skipped entirely: a reference change before the test is common-mode across every on and off block, so the on/off contrast is blind to it, and splitting the test period in half leaves too little a side to see one |
| **reserves and refilling** | only a planned campaign supplies reserves, so only prepost refills a screened-out reference. A flat campaign's screened reference stays in the pool and goes power-free instead |
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
