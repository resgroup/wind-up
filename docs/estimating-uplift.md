# How wind-up estimates uplift

This page describes the estimator itself — the part that is the same whether a campaign is prepost
or toggle. What prepost adds on top of it is [a separate page](prepost-campaigns.md). For how a
campaign is declared and run, see [running a campaign](running-a-campaign.md).

wind-up answers one question per upgraded turbine:

```
uplift = (energy the turbine produced while upgraded) / (energy it would have produced unchanged) - 1
```

The denominator is never observed. wind-up builds it by **learning the test turbine's power from
signals the upgrade cannot have touched** — the other turbines on the site, and reanalysis weather
— over a period when the turbine was unchanged, then asking that model what the turbine would have
produced over the period when it was changed. The estimate is the ratio of the two energies summed
over the upgraded records, not a mean of per-record ratios.

TODO possibly a source of bias we need to address is using the observed numerator and the unobserved denominator in the same equation. The spirit of the equation is right but the method might be wrong. See "C:\Users\aclerc\OneDrive - RES Group\Controls role\Adaptive Tuneup\DoubleDebiased_ML.pdf" for some relevant discussion about bias with ML estimators.
TODO if the test turbine is casting a wake on references, and the wake has changed because of the upgrade (definitely the case with wake steering and possibly true with major energy changes) then it does have the potential to affect them. Probably should not use data from references when the test turbine wakes them by default; a user can override this if they are confident the filter is not necessary.

Expressing expected power *through the references* is what forms the test-versus-reference
contrast. A seasonal swing, a long-term climate trend or a site-wide curtailment moves the test
turbine and its references together, and so can be assumed to cancel with large enough datasets.

TODO not in place today, but we want wind-up to also be able to use reference wind speed measurements (masts, LiDARs, ...) where possible. Note this as future functionality for v1; v0 supports it so we need to get v1 to parity before release. The advantage of these devices is they are more likely to be stable over time, the disadvantages is they might measure erratically in wakes and what they measure (generally horizontal wind speed, wind direction, air density ingredients (pressure, temperature, humidity), turbulence, possibly vertical wind speed, all at possibly multiple heights) has a complex relationship with turbine power.

The estimator is `PowerModelMethod`
(`benchmarking/baselines/power_model/method.py`); `benchmarking/campaigns/composed.py` is the one
place that pins the accepted defaults and gives it the name `wind-up`.

## The shape of a run

| step | scope | what it does |
|---|---|---|
| plan | per upgraded turbine | choose each turbine's analysis span and power references (prepost only — see [prepost campaigns](prepost-campaigns.md)) |
| northing | whole farm, once | discover and apply each turbine's north calibration ([how northing works](northing.md)) |
| estimate | per upgraded turbine | screen the references, fit the counterfactual, take the energy ratio, report each reference |
| aggregate | whole campaign | combine the per-turbine uplifts into one energy-weighted farm number |
TODO in aggregate we need an equivalent aggregation for reference turbine measurements to verify measured uplift for references is near 0% 

Northing runs once for the farm, before any estimate, so every turbine inherits the same
correction rather than each estimate hand-rolling one. It writes a `northed_<column>` companion
beside each direction column and leaves the original untouched. It runs on the usable rows only;
rows held back by a turbine's own exclusions are added back afterwards with no north-calibrated
direction.

Each upgraded turbine is then estimated independently, against its own references, and the results
are combined at the end.

## What a method is allowed to know

A method never reads the campaign declaration. The runner turns the declaration into a
`CampaignContext` (`benchmarking/harness/context.py`) per test turbine, and that object carries
answers rather than fields:

| the context says | the method uses it for |
|---|---|
| `candidate_references` | the reference pool. A turbine present in the data but not offered here is **not** a reference, whatever its data looks like |
| `wake_contributors` | every other turbine present — the campaign's other changed turbines, its excluded ones, and any it never declared |
| `valid_for_uplift` | per turbine, per timestamp: may this turbine's data contribute to the uplift estimate |
| `timing` | this turbine's changeover, or its toggle schedule |
| `reserve_references`, `reading_pools` | who replaces a reference the screen rules out, and who each reference is read against |
TODO I don't understand what `reading_pools` is - please add more detail

`context.select(scada_df)` applies the first three: it keeps the test turbine, its candidate
references and its wake contributors, and drops each one's rows that are not valid for uplift. An
invalid *reference* row drops that reference for that timestamp only, leaving the row usable; an
invalid *test-turbine* row drops the record outright.
TODO I think we need to brainstorm how waking information can stay in despite the need for data filtering

## Which rows the estimate is built from

After the campaign's own validity, the **test turbine alone** passes through
`NormalOperationFilter` (`benchmarking/baselines/filtering.py`). Three checks:

- **finite power** — a NaN active power is downtime or missing energy, always dropped;
- **availability** — the availability counter must show the turbine ready to operate for the whole
  period. This is required: a missing availability column is a configuration error, not a silent
  no-op;
- **stuck data** — a record where every numeric signal is unchanged from the one before it is a
  frozen data stream, unless the wind speed is below 1.5 m/s, which is a genuine calm.

The rule behind all three is **filter on cause, not effect**. Nothing selects on "power lower than
expected", which would drop genuine low-uplift records and bias the answer.

**References are deliberately not filtered.** Their operating state is information the model is
meant to learn, not noise to be removed: a parked reference is a real fact about the wake the test
turbine was in. A reference's rows are dropped only when the campaign says they are invalid.
TODO the above paragraph seems to be inconsistent with the earlier text "invalid *reference* row drops that reference for that timestamp only, leaving the row usable"

Below 10 normally-operating baseline rows the estimate raises rather than returning a number.
TODO this should be an hours-of-data limit, not row count, because wind-up should be OK with any aggregate timebase (10min is most common in wind industry but others are possible)
TODO 10 * 10min seems like a very small lower limit, I'd require at least 24h of data

## The features

Every feature must be **upgrade-invariant** — derived from reference turbines or from reanalysis,
never from the test turbine, whose signals the upgrade distorts. `check_reference_only` enforces
this: any feature column qualified with the test turbine's name raises.

Feature columns are named `<source tag> @ <turbine>`, and the references are laid out in sorted
order whatever order the declaration listed them in, so the line order of a YAML file cannot reach
the model.

**Each reference that still carries power contributes:**

| feature | from | on by default |
|---|---|---|
| active power | the `active_power` role | yes |
| active power minimum | the `active_power_min` role | yes |
| north-calibrated nacelle position, as `sin` and `cos` | `northed_<nacelle_position>` | yes (`direction_feature`) |
| availability | the `availability` role | **no** (`availability_feature=False`) |

The direction is always the north-calibrated column, never the raw one, and it is always split
into `sin`/`cos` because a tree cannot see that 359° is next to 1°.

**Each power-free reference and every wake contributor contributes two booleans and nothing else:**

| feature | meaning |
|---|---|
| `waking_<power>` | active power at or above 5 % of rated (`WAKING_RATED_FRACTION`) — is this turbine physically making a wake |
| `normal_operation_<availability>` | availability counter at or above half the period (`NORMAL_OPERATION_AVAILABLE_FRACTION`) — was it able to run |
TODO we have already found that the 5 % of rated rule is perhaps a source of leak / bias where test turbine power has changed a lot. Need to brainstorm alternatives. Have a look back at how v0 handled waking information.

A turbine that may have changed keeps its wake information and loses every channel a performance
change can move — its power, and also its direction, since a realignment or wake steering moves
where it points. The booleans are floats, not booleans, so that a timestamp the turbine has no row
for stays NaN (unknown) rather than collapsing to `False`.

**Reanalysis** contributes every raw Open-Meteo ERA5 column under its original name, plus `sin`/`cos`
companions for the direction fields. Five columns are dropped by default
(`CURATED_ERA5_EXCLUDE`): `apparent_temperature`, `dew_point_2m`, `precipitation`, `rain`,
`snowfall` — redundant thermodynamic derivatives and the precipitation trio, which a removal
ablation found the feature set does better without.
TODO revisit as this ablation study is old and possibly overfit. Switch to just including a small list of ERA5 features which have clear physical justification to be included.

ERA5 arrives hourly and is aligned to the analysis timebase by `sync_era5`
(`benchmarking/baselines/era5_sync.py`): resample and forward-fill within each hour, then apply the
single integer row shift, swept over ±24 h, that maximises the correlation between ERA5
`wind_speed_100m` and the **mean wind speed across the reference pool**. The reference mean is used
rather than the test turbine's own, so the alignment stays upgrade-invariant; it is not itself a
feature.

Deliberately excluded everywhere: reactive power, blade pitch, and anything else whose relationship
to power is a coincidence rather than a cause.

## The model

A single LightGBM L2 regressor for `E[power | features]`
(`benchmarking/baselines/power_model/fitting.py`): 600 trees, learning rate 0.03, 63 leaves,
`min_child_samples` 50, 0.8 subsample and column sample. NaNs are handled natively — no
complete-case dropping, so one reference's gap does not cost the whole record.

`deterministic=True` and `force_row_wise=True` are set so a fit reproduces. Without them LightGBM
chooses its histogram strategy by timing the machine, and the same estimate moved with the load on
the box and with the thread count. Feature names are renamed positionally to `f0`, `f1`, … because
LightGBM rejects the JSON special characters that real source tags carry.

Predictions are clipped to the physically plausible range of the outcome they were fitted on:
`min(0, min(y_train))` below and `max(rated, max(y_train))` above. Tree boosting sums trees, so a
prediction can drift slightly past the training range; the clip binds only there.

There is **no cross-fitting and no propensity model**. The counterfactual is fitted on the baseline
rows and predicts the disjoint upgraded rows, so there is no in-sample leakage to correct for.

Alongside the headline fit, a second model is fitted on a train split of the baseline and predicts
a held-out baseline slice, purely as an honest fit-quality diagnostic. The split is **time-blocked**
— the baseline is cut into 25 contiguous blocks dealt round-robin into 5 folds, and fold 0 is held
out — because a shuffled holdout sits minutes from its training rows and autocorrelation would make
its residuals look better than they are. Under 20 baseline rows, the in-sample fit is reported
instead.
TODO is this value adding anymore? I never look at it

## The reference-validity screen

A reference whose own performance shifts across the campaign boundary biases the counterfactual
with the sign reversed. The screen looks for that.

Each candidate reference is estimated **as if it were a test turbine**, against the other
candidates, using the same method with the conditional step, the plots and the screen itself turned
off. The statistic is each candidate's deviation from the **pool's median** estimate, and the rule
is an absolute floor of **2.5 pp** (`_DEFAULT_SCREEN_FLOOR`) on that deviation.

The median, not zero, is the reference point, and that distinction is the whole design. Turbines
are not expected to stay the same — references drift and step — and the assumption is only that
most of them do so the way the test turbine would have. A pack all losing 1 % a year reads ~0 on
each other, correctly. What biases an estimate is a reference that moves *unlike the rest*.

Only the worst reference is dropped per pass, because a bad reference infects every other estimate;
the pool is then re-judged. The loop stops when nobody stands out. If a reference still stands out
after a majority's worth have been ruled out, that is a farm-wide problem rather than a
reference-validity one, and **no estimate is offered**.

The floor is set high on purpose. Clean placebo pools spread up to 1.19 pp with nothing injected,
so 2.5 pp leaves roughly a factor of two before a healthy reference is at risk, and on the clean
fixture a false positive moved the estimate 0.88 pp — ruling out a good reference costs more than
leaving a mild bad one in.

Three gates stop the screen running where it cannot do better than chance:

| gate | value | why |
|---|---|---|
| pool size | at least 3 candidates | below that there is no majority to rule with; two references are always equidistant from their own midpoint |
| campaign data per candidate | 150 days of normally-operating upgraded data (`screen_min_campaign_days`) | at 90 days the benchmark sweep still produced false positives, and acting on them made the result worse. Counted from records through the same normal-operation mask the fit uses, not from the calendar span |
| mode | prepost only | see [prepost campaigns](prepost-campaigns.md) |

A candidate holding too little campaign data is **unjudged** rather than trusted: the rest of the
pool is still screened, and the unjudged candidate loses its power channels too. If no candidate
holds enough, the screen does not run at all and nobody loses anything.

**What being ruled out costs a reference.** Without reserves, it stays in the pool and becomes
power-free — it keeps its waking and normal-operation booleans and loses its power, its power
minimum and its direction. With reserves (which a planned prepost campaign supplies), it leaves the
pool for the wake contributors, the nearest reserve is promoted in its place, and the refilled pool
is screened again; rounds continue until a pass rules out nobody new or the reserves run out.

Because the verdict is one answer for the whole campaign — the same pool judged across the same
contrast — a campaign runs the screen once and shares the result across its test turbines through a
cache.

TODO the reference screen needs to be re-visited after the main method is better. Ideally the reference screen just runs the main method with no modifications and as little judgement logic as needed.

## What the run reports

**The headline** is `Σ actual / Σ counterfactual − 1` over the test turbine's upgraded rows.

**`reference_stability.csv`** re-estimates every candidate reference as if it were a test turbine,
against the surviving pool, and reports it with `screened` and `unjudged` flags. It is computed
after the screen, so it answers what the *final* analysis says its references did. When nothing
lost its power and the screen already ran the same contrast on the same pool, its final pass is
reused rather than refitting. A healthy campaign reads near 0 % for every reference, and **the
spread of those readings is the campaign's own noise yardstick** — see [running a
campaign](running-a-campaign.md#start-with-reference_stabilitycsv).

**The farm number** (`wind_up.farm.farm_uplift`) is energy-weighted, not a mean of uplifts. Each
turbine's counterfactual energy is reconstructed as `actual / (1 + uplift)` and capped at
`rated_power_kw × n_records`; the farm uplift is `Σ actual / Σ counterfactual − 1` over the turbines
that survive the guards. A turbine is guarded out when its uplift is non-finite or ≤ −1, when its
energy is non-finite or negative, when its rating is not a positive finite number, or when it has
no records. `uplift_spread` is the gap between the best and worst surviving turbine.

**The conditional breakdown** (`conditional.csv`) splits the uplift by wind speed, turbulence
intensity and power. It is the optional, expensive last step, it requires ERA5, and nothing else
depends on it. It works differently from the headline:

1. **Match** the baseline and upgraded rows on coarsened ERA5 weather cells — `wind_speed_100m` in
   2 m/s bins, `wind_gusts_10m` in 3 m/s bins, `wind_direction_100m` in 20° sectors. Cells present
   on only one side are dropped (common support) and the larger side is seeded-subsampled down to
   the smaller within each cell.
2. **Fit both directions** on the matched rows: forward (train baseline, predict upgraded) and
   reverse (train upgraded, predict baseline).
3. **Combine** per bin as `sqrt((1 + r_fwd) / (1 + r_rev)) − 1`, so a shrinkage common to both
   directions cancels.
4. **Impute** bins with fewer than 50 matched rows per side, where the combine overshoots, and
   **re-level** the whole shape by full-upgraded energy so the bins aggregate back to the headline.

Every power bin is labelled by its *counterfactual prediction*, never by the actual power that also
sits in that side's ratio numerator — labelling on the outcome would select each bin on its own
noise and manufacture a regression-to-the-mean slope out of a flat truth.

TODO revisit conditional after the headline method is sorted out

## The knobs, and what they are set to

These are the accepted defaults `wind_up_method` pins. Everything here is a `PowerModelMethod`
field.

| knob | default | note |
|---|---|---|
| `availability_feature` | `False` | per-reference availability is not a feature; the role is still required for the downtime filter |
| `normal_operation_feature` | `True` | power-free references get a second boolean beside their waking one |
| `direction_feature` | `True` | north-calibrated direction as sin/cos; requires the shared northing step to have run |
| `era5_exclude` | `CURATED_ERA5_EXCLUDE` | five raw columns dropped |
| `model_params` | `min_child_samples=50` | loosened from the common 200; 20 overshoots into overfit |
| `reference_screen` | `True` | |
| `screen_floor` | `0.025` | 2.5 pp from the pool median |
| `screen_min_campaign_days` | `150` | |
| `report_reference_uplifts` | `True` | a report, not an input — turning it off never moves the headline |
| `headline_estimator` | `"forward"` | `"reversal"` is prepost-only and off |
| `adaptive_time_decay` / `time_decay_half_life_days` | `False` / `None` | campaign-proximity weighting is off; every training row weighs the same |
| `exclusion_channels` | `"booleans"` | prepost-relevant; see [prepost campaigns](prepost-campaigns.md#exclusions-and-held-back-rows) |
| `conditions` | all three | `()` skips the expensive conditional step |

## Known limits

- **Every turbine's data is trusted for its wake.** A turbine whose power reads high while it is in
  fact stopped is counted as waking its neighbours.
- **The result is a P50.** There is no uncertainty interval. The spread of the reference readings is
  the intended stand-in.
- **The screen judges references, not the test turbine.** A test turbine that changed for a reason
  unrelated to the upgrade is not detected.
- **Curtailment held at zero while available** reads as normal operation: the availability counter
  does not see it.
- **A campaign with fewer than three candidate references is never screened**, so its references are
  trusted unconditionally.
