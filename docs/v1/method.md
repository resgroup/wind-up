# wind-up uplift measurement method

The wind-up tool measures energy yield uplift and uncertainty of wind turbines after they are changed in
some way such as a turbine upgrade or implementation of wind farm control. The wind-up methodology uses
best practice from the turbine power performance standard IEC 61400-12-1 [1] and adds innovative steps to
achieve additional goals such as measurement of uplift under all conditions (including waked operation).

The wind-up tool can use met mast data and LiDAR data to measure uplift of upgraded test turbines as would
be the case in an IEC 61400-12-1 power performance test. However, this document focuses on the case where
mast and LiDAR data is not available. In that case neighbouring turbines that have not been changed are
used as reference turbines to measure the uplift of upgraded test turbines.

This document describes the target state of the v1 method. Where more than one approach is still under
consideration the options are listed, with the current default marked. An option is retired once the
validation described at the end of this document has decided between them.

## Principles

The following principles apply to every step:

- **Upgrade invariance.** Every input used to predict the test turbine's power must be unaffected by the
  upgrade. Signals from the test turbine itself are never used as model inputs, and signals from any
  turbine that may have changed during the analysis period are reduced to those that cannot carry a
  performance change. Upgrades can change wakes so data affected by test turbine wakes is also removed.
- **Wind turbine anemometers are never used as model inputs.** Their calibration drifts and can change
  with a turbine upgrade. They may still be inspected as a possible indicator of performance change.
- **Direction data is only used after it has been vetted and corrected** (step 4).
- **Filter on cause, not effect.** Data is excluded because of what a turbine was doing (offline,
  curtailed, faulted) and never because of what it produced. Selecting on lower-than-expected power
  would bias the result.
- **Matching conditions.** Bias risk is reduced when the before and after datasets are exposed to the
  same conditions: weather, season, and operating state of the surrounding turbines (wakes).
- **Data quantities are expressed in hours, not records**, so that the method applies to any SCADA
  timebase.

## Method steps

The wind-up methodology comprises the following steps.

### 1. Identify changes to the wind farm

Identify any changes to the wind farm or in the vicinity of the wind farm that would change the
relative performance of turbines in comparative analysis. Examples could include neighbouring wind
farm construction, tree felling or other upgrades outside the scope of the upgrade under test. Any
such changes need to be evaluated and potentially would mean certain turbines or time periods need
to be excluded from data analysis.

In addition to any changes that are known in advance, the SCADA data is inspected for the following
indicators of an undeclared change:

- i. changes in reactive power behaviour;
- ii. changes in the relationship between power, wind speed, pitch angle and rotor speed;
- iii. changes in data coverage, for example new signals appearing, which could indicate new turbine
  software.

Each identified change can be considered for exclusion. A turbine with an
exclusion window is not used as a power reference over any period that overlaps it (step 6), but it
may still contribute waking information (step 8).

### 2. Derive operational state and data validity

Historic SCADA data (typically 10-minute but other timebases e.g. 1-minute can be used) for all
turbines on the wind farm are used to derive an operational state for each turbine and each record.
This is done using power, pitch angle, rotor speed, availability and state counters. The states
distinguished are at least: normal operation, curtailed operation (with the cause where it is known,
for example a noise mode or a grid limit), partial downtime, full downtime and missing data.

Data records are flagged as valid or not for each step of the analysis separately, because an
operating state can be valid for some purposes but not for others. For example, curtailed data can be
valid for northing (step 4) and for waking state (step 8) but not for the uplift calculation.

Some wind farms will have predictable curtailment which is expected to continue unchanged in future
(for example noise modes). In such cases it may be justified to retain curtailed data in the uplift
analysis, provided the curtailment applies equally in the before and after periods.

It is important to carefully inspect power against wind speed, and pitch angle and rotor speed against
power and wind speed, for both the excluded and the retained data of every turbine, to confirm that the
operational state classification matches expectations.

TODO for Hill of Towie specifically we need to improve the data laeblling:
1 normal operation (most of the data)
2 noise mode (eg T16)
3 grid curtailment
4 partial downtime
5 full downtime
The private adaptive-tuneup repo has logic to identify noise and grid curtailment
Then each state needs to be related to how it can be used in the analysis. For HoT 1-3 are ok for northing, 1-2 are ok for uplift, 1-2 are waking and 5 is not waking
wind-up should have a fallback method which classifies data on its own (lean on the v0 method) and should let callers pass in labels and/or instructions for labelling and analysis validitiy mappings for each label

### 3. Add reanalysis data

Reanalysis data (ERA5) is merged with the SCADA data. ERA5 is used as a source of upgrade-invariant
weather information: it is unaffected by any change to the turbines.

ERA5 is obtained from the Open-Meteo archive API with the ERA5 model selected. Its values are
interpreted as follows:

- i. **Time.** Timestamps are in UTC. Wind components, temperature and pressure are instantaneous
  values valid at the hourly timestamp, not averages over the hour. Accumulated fields such as
  precipitation are totals over the hour ending at the timestamp, and the 10 m wind gust is the maximum
  over the hour ending at the timestamp.
- ii. **Space.** ERA5 has a native resolution of about 31 km and is provided on a 0.25° grid. A grid
  point value represents an average over the model grid box, not the conditions at the requested
  coordinates, and it cannot represent variability on spatial scales smaller than the grid box. By
  default Open-Meteo returns a nearby land grid cell of similar elevation to the requested coordinates
  rather than strictly the nearest one.
TODO override this default, we want strictly the lat-lon requested
- iii. **Smoothness.** Being a grid box value, an instantaneous ERA5 value is smooth in time and does
  not represent the short-term variability of a turbine's 10-minute mean. ERA5 is therefore a
  weather-state feature rather than a substitute for a measured wind speed.

The hourly values are aligned to the SCADA timebase as follows.

- **Option A (current default).** Forward-fill each hourly value through the hour, then apply the
  single whole-record time shift (searched over ±24 hours) that maximises the correlation between ERA5
  100 m wind speed and the mean wind speed of the reference turbines.
- **Option B.** Establish the SCADA timestamp convention (period start or period end, and time zone)
  explicitly, then interpolate the instantaneous ERA5 fields linearly to the centre of each SCADA
  period. Hourly accumulations and maxima are assigned to the SCADA periods within the hour ending at
  their timestamp, and are not interpolated. The time-shift search is retained only as a check, which is expected to find a shift near zero.
  Where it does not, the timestamp convention is investigated rather than the shift being applied.

### 4. Northing

The nacelle direction of every turbine is analysed to detect its north calibration and any shifts in it
over the full data period, and is corrected so that a north-calibrated direction is available for every
turbine and every record. The method is described in [northing.md](../northing.md).

Small north changes on an individual turbine may optionally be reported as possible yaw alignment
changes, since these are a candidate cause of performance change (step 1).

### 5. Define the before and after periods

The dataset is divided into two periods: before and after the upgrade. For a toggle test these two
periods are the toggle off and toggle on data, respectively. For a non-toggle (prepost) test the after
period starts when the upgrade installation of the test turbine is complete and will normally run until
the day when the analysis is conducted, to use as much data as possible. Where turbines are upgraded
one after another, each test turbine has its own before and after periods.

It is important that neither of the periods include any other known changes or upgrades to either the
test or reference turbines. This is ensured by a combination of removing reference turbines that
exhibit changes (steps 1 and 6) and trimming the periods as required. The periods and the reference
turbines are therefore chosen together: a longer period may cost reference turbines whose own changes
fall within it.

The length of the before period is chosen as follows.

- **Option A (current default).** Use all available data before the upgrade, up to a cap of 24 months,
  with every record weighted equally.
- **Option B.** Use the same calendar months as the after period, in the previous year or years (the
  seasonal matching of wind-up v0).
- **Option C.** trim / filter the pre and post datasets so that their conditions match as closely as possible while still retaining sufficient data. Which conditions are significant (eg calendar time, turbine power, ERA5 fields, etc) needs to be determined analytically from the available data.

A long before period gives the counterfactual model (step 10) more data and generalises better to the
after period. A seasonally matched before period exposes both periods to the same seasons. Evidence to
date favours each in different circumstances, and the matching step (step 7) may make the choice less
important, so it is decided by validation.

For a toggle test, a toggle pairing filter is applied which only keeps data where the time difference
between a data record and the nearest valid record with the opposite toggle state is within expected
limits.

### 6. Select reference turbines

A set of reference turbines is chosen for each test turbine. These are the power references: turbines
whose power is used to predict the test turbine's power. At least three reference turbines are required,
and more are used where available. The following factors are considered when selecting reference turbines:

- i. Turbines for which the active power correlates most strongly with the test turbine are favoured.
  Normally these are also the closest turbines to the test turbine, and in practice the nearest
  eligible turbines within 20 rotor diameters are taken.
- ii. A reference turbine must have data over the whole analysis period, should have no known
  performance issues, and must not have undergone any significant change itself during the analysis
  period (step 1).
- iii. It helps where possible for a reference turbine to be un-waked in the predominant wind direction,
  especially by the test turbine itself.

Every other turbine on the wind farm, including other test turbines and turbines excluded as power
references, still contribute to the waking scenario (section 8).

Reference turbines are analysed together as one model for each test turbine, not one test-reference pair
at a time. A single-reference estimate carries roughly twice the shared bias of a multi-reference one,
and each reference's own drift relative to the test turbine is averaged down as more references are
added.

### 7. Match conditions between the periods

The before and after periods are trimmed so that each is exposed to the same conditions. For a toggle
test the pairing filter of step 5 already ensures this for weather. For both test types the following
are applied:

- i. **Operating configuration.** For each record, the configuration is the set of power references
  that are in normal operation. Only configurations with at least 24 hours of data in both periods are
  retained, in the same way that wind-up v0 retained only the combinations of operational turbines with
  at least one day of data. In particular, after-period records in which a power reference is offline
  are dropped unless that configuration is well represented in the before period. A record in which a
  reference is not operating is otherwise one the model has rarely or never seen, and its prediction is
  an extrapolation. This is the largest single source of bias identified to date.
- ii. **Weather.** Both periods should see the same weather.
  - **Option A (current default).** No weather matching beyond the choice of period in step 5.
  - **Option B.** Common-support trimming: coarsen ERA5 wind speed, direction and a stability proxy
    into cells, and drop the records of either period that fall in cells with less than a minimum
    number of hours in the other period.
- iii. **Waking scenarios.** Both periods should see the same waking scenarios. Option B of ii,
  extended with the waking configuration of the upwind turbines, covers this.

Trimming is kept as natural as possible: where practical, whole days are removed rather than individual
records, so that the retained data stays contiguous and autocorrelated in the same way in both periods.

Features that reveal the period rather than the conditions, such as a signal that only exists in one
period, must not be used for matching. Weighting the records by a model of the probability of belonging
to each period is not used, because on real data the period is almost perfectly predictable from such
features.

### 8. Determine waking state

The waking state of every turbine on the wind farm is determined for each record, from the operational
state of step 2. Each operational state is classed as waking (for example normal operation, noise
mode), not waking (for example full downtime) or partially waking (for example partial downtime, grid
curtailment).

The waking state of a turbine that may have changed (a test turbine, or a turbine excluded as a power
reference) must itself be upgrade invariant.

- **Option A (current default).** A turbine is waking when its active power is at least 5 % of rated
  power. This has been shown to leak the upgrade into the estimate: records close to the threshold
  change waking state when an upgraded turbine's power changes.
- **Option B.** Waking state is derived from the cause signals only (operating state label).

Records where the test turbine, or any other upgraded turbine, is upwind of a power reference (within
the IEC 61400-12-1 disturbed sector, using the north-calibrated direction of the reference) are
labelled. An upgrade can change the wake a turbine casts, so these records are not used for the uplift
calculation by default. The label depends only on direction, so it removes the same sectors from both
periods. The analyst can override this where the wake is known not to change.

- **Option A.** Drop the labelled records.
- **Option B.** Keep the records but withhold the waked reference's power for them, so the model sees
  only its waking state. This keeps more data, but creates a reference configuration that must pass the
  matching of step 7 i.

TODO re-write the above so it behaves more like wind-up v0. For the test turbine and each reference turbine the waking scenario of each record is calculated by recording the state of all turbines upwind (according to IEC definition). This avoids the need to give the ML model loads of booleans from every turbine in the farm which it then has to reason through from scratch. This also allows wind-up to match like-for-like waking scenarios in pre and post for record matching. The 'normal' state would be no turbines upwind or all turbines upwind are waking. Abnormal states can then carry descriptive labels about what is abnormal. Records with partial waking upwind turbines need to be easy to identify because in general we want to not use those (waking state is too ambiguous) 

### 9. Engineer features

The features used to predict the test turbine's power are:

- i. **Power references:** active power, and the north-calibrated nacelle direction as sine and cosine.
  - **Option A (current default).** Active power only. ERA5 resolves the very low and very high wind
    speeds where reference power carries little information.
  - **Option B.** Active power, pitch angle and rotor speed, which allows high wind speed operation to
    be resolved.
- ii. **All other turbines:** waking state only (step 8).
- iii. **ERA5:** a short list of variables with a clear physical reason to affect the test turbine's
  power relative to its references: 100 m wind speed, 100 m wind direction (as sine and cosine), air
  density derived from temperature, pressure and humidity, a shear exponent derived from the 10 m and
  100 m wind speeds, and a turbulence proxy derived from the gust and mean 10 m wind speeds.

Adding more information about the references has not reduced bias to date. Features are therefore kept
to those with a clear physical justification.

### 10. Relate the reference measurements to the test turbine measurements

A machine learning model is trained which predicts test turbine power from the features of step 9.
This takes the place of the directional detrending of wind-up v0: rather than a ratio of wind speeds
by direction bin, the model learns how the test turbine's power relates to its references under every
combination of direction, weather and waking state present in the data. Only data valid for the uplift
calculation (steps 2, 7 and 8) is used.

The model is a gradient-boosted tree regressor minimising squared error. It is configured to give
identical results on any machine.

### 11. Calculate the uplift

The model of step 10 trained on the before period, $\hat f_B$, predicts the test turbine's power in the
after period. Writing $y$ for measured test turbine power and $A$ and $B$ for the after and before
records, the forward uplift is

$$
U_{fwd} = \frac{\sum_A y}{\sum_A \hat f_B} - 1
$$

This compares a measured sum with a predicted one, and is biased by any error the model makes on
average when predicting data unlike its training data. That error grows with the time separating the
two periods and is the main source of bias left once steps 6 to 8 are applied. One of the following is
used to remove it:

- **Option A (current default).** The forward uplift alone.
- **Option B, reversibility.** As in wind-up v0, the calculation is also performed in reverse: a
  second model $\hat f_A$ is trained on the after period and predicts the before period,

  $$
  U_{rev} = \frac{\sum_B y}{\sum_B \hat f_A} - 1
  $$

  With no bias, $(1 + U_{rev}) = 1/(1 + U_{fwd})$. A bias common to both directions cancels in

  $$
  U = \sqrt{\frac{1 + U_{fwd}}{1 + U_{rev}}} - 1
  $$

  This removes the bias on long campaigns but is harmful on short ones (below about six months),
  where the reverse model is trained on too little data to generalise. The correction is therefore
  weighted: in log space, with $f = \ln(1 + U_{fwd})$, $g = \ln(1 + U_{rev})$ and the estimated bias
  $\hat\beta = (f + g)/2$, the uplift is $\ln(1 + U) = f - \lambda\hat\beta$, where $\lambda$ between 0
  (forward only) and 1 (full reversal) is calculated from the measured precision of the two directions
  to minimise the expected error.
- **Option C, before-period backtest.** The model's own bias is measured inside the before period, by
  placing a pseudo-upgrade date so that the held-out final part of the before period is separated from
  its training data in the same way as the after period, and the forward uplift is divided by the ratio
  measured there. This mirrors the geometry of the real comparison, at the cost of training the
  backtest model on less data than the real one.

The reverse calculation of option B does not use the test turbine anemometer and so can be performed in
full, unlike wind-up v0, which had to fall back on power-only versions of both calculations.

### 12. Analyse each reference turbine as a test turbine

The process is repeated using each reference turbine as a test turbine, predicted from the other
reference turbines, with the same periods, matching and features. The expectation is that 0 % uplift is
measured for each reference turbine. The purpose of this step is to detect any bias in the reference
turbines, by seeing whether any of them have a measurable change in performance relative to the others.

A reference turbine whose uplift differs from the median of the reference turbines by more than a floor
(2.5 percentage points by default) is excluded as a power reference and the remaining turbines are
judged again, excluding at most one turbine per pass. The median rather than zero is used because the
reference turbines may all drift together, which is not a source of bias. Where a replacement reference
turbine is available it takes the excluded turbine's place. This step requires at least three reference
turbines and enough after-period data to separate a bad reference from noise.

The combined reference uplift (step 14) is reported alongside every result as the campaign's own check
of bias, and the spread of the individual reference uplifts as its own measure of noise.

The reference uplifts could also be used to correct the test turbine uplift. This is not done, because
the references would then read 0 % by construction and could no longer be used as a check of bias.

### 13. Calculate uplift distributions

Uplift is calculated as a function of condition variables specified by the analyst. Uplift against
wind speed is generally the most important for pre-construction inputs, but it has to be based on a
stable, free-stream wind speed estimate derived from the reference turbines (power and pitch angle), not
from the test turbine. For certain upgrades other conditions are important, for example wake steering
also requires uplift against direction and atmospheric stability.

Records are assigned to a bin by an upgrade-invariant variable, never by the test turbine's measured
power, which would select each bin on its own noise. The binned uplifts are scaled so that they combine
to the headline uplift of step 11. The tool should also determine which variables the uplift is
sensitive to.

### 14. Combine turbines

Turbine results are combined where the turbines have the same role or upgrade. Test turbines that have
received the same upgrade are combined into one uplift, and all reference turbines are combined into one
combined reference uplift, which is expected to be 0.0 %. In both cases the combination is
energy-weighted rather than a mean of the uplifts:

$$
U_{combined} = \frac{\sum_t \sum_A y_t}{\sum_t \sum_A y_t / (1 + U_t)} - 1
$$

### 15. Estimate the annual energy production uplift

For most upgrades the uplift against wind speed (step 13) combined with a long-term wind speed
distribution is sufficient to estimate the uplift in annual energy production (AEP). The measured
uplift distribution only covers the conditions seen during the campaign: it needs to be extended to the
long-term distribution, with the uncertainty increased for conditions that were not covered. ERA5 is
used to estimate the long-term wind speed distribution.

### 16. Estimate uncertainty

This revision of the method addresses the central (P50) estimate only. The uncertainty method of
wind-up v0 (statistical, block bootstrap and reversibility components, combined across reference and
test turbines) is the starting point for the uncertainty of v1.

## Validation

The method is judged on synthetic campaigns with known ground truth, built from real SCADA data, and on
real campaigns where the reference turbines provide truth of zero. The target is:

- i. the combined reference uplift and the uplift of placebo test turbines (upgraded with zero effect)
  are 0.0 % on average, within the noise of the campaigns;
- ii. on synthetic upgrades of known size, a straight line fitted to measured against true uplift has a
  slope of 1 and an intercept of 0, with as little scatter as possible;
- iii. neither depends on the length of the after period, the number of reference turbines or the
  season of the upgrade.

The options above are decided by these criteria.

## Open questions

- Which before period (step 5 options) is preferred once matching (step 7) is in place.
- How the ERA5 stability proxy and weather cells of step 7 are defined, and the minimum hours per cell.
- How the free-stream wind speed estimate of step 13 is constructed.

## References

[1] Wind energy generation systems - Part 12-1: Power performance measurements of electricity producing
wind turbines (IEC 61400-12-1:2022). International Electrotechnical Commission, September 2022.
