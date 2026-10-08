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
  performance change. Upgrades can change wakes so data affected by test (or changed) turbine wakes is also removed.
- **Wind turbine anemometers are never used as model inputs.** Their calibration drifts and can change
  with a turbine upgrade. They may still be inspected as a possible indicator of performance change and used to help verify (eg ERA5 sync is correct).
- **Direction data is only used after it has been vetted and corrected** (step 4).
- **Filter on cause, not effect.** Data is excluded because of what a turbine was doing (offline,
  curtailed, faulted) and never because of what it produced. Selecting on lower-than-expected power
  would bias the result.
- **Matching conditions.** Bias risk is reduced when the before and after datasets are exposed to the
  same conditions: weather and operating state of the surrounding turbines (ie wake exposure).
- **Data quantities are expressed in hours, not records**, so that the method applies to any SCADA
  timebase.

## Method steps

The wind-up methodology comprises the following steps, in three parts.

- **Part A, data preparation (steps 1 to 6)** is done once per dataset. Nothing in it depends on which
  turbine is the test turbine or on which turbines are its references, so it is never repeated.
- **Part B, the uplift estimator (steps 7 to 10)** is a procedure that takes one test turbine, one set
  of power references and the before and after periods, and returns an uplift. Nothing in it changes
  the reference set or the periods.
- **Part C, the campaign (steps 11 to 16)** calls the estimator: first with each candidate reference as
  the test turbine, to screen the references (step 11), then for the test turbine with the screened
  references (step 12), and assembles the results.

This separation is what resolves a circularity that is otherwise easy to fall into: the usable data
depends on the references, the references depend on a screen, and the screen needs the usable data.
Here the periods are defined from the test turbine alone, the references are screened with those
periods fixed, and the test turbine's uplift is then calculated with the screened references.

## Part A: data preparation

### 1. Identify changes to the wind farm

Identify any changes to the wind farm or in the vicinity of the wind farm that would change the
relative performance of turbines in comparative analysis. Examples could include neighbouring wind
farm construction, tree felling or other upgrades outside the scope of the upgrade under test. Any
such changes need to be evaluated and potentially would mean certain turbines or time periods need
to be excluded from data analysis.

In addition to any changes that are known in advance, the SCADA data is inspected for the following
indicators of an undeclared change:

- i. changes in the relationship between power, wind speed, pitch angle and rotor speed;
- ii. changes in reactive power behaviour;
- iii. changes in data coverage, for example new signals appearing, which could indicate new turbine
  software.

Each identified change can be used to consider how to configure operation state labels, analysis date range and used turbine, and data exclusions.
Records inside an exclusion window are not used in the uplift calculation, but the turbine still
contributes waking information in those records (step 5). An exclusion does not make a turbine changed:
for example, a temporary de-rate for a component health issue leaves the turbine notionally the same
before and after it.

The outcome of this step is plots the analyst can review to confirm the data is fit for purpose and to help define manual exclusions and operational state classification.

### 2. Derive operational state and data validity

Historic SCADA data (typically 10-minute but other timebases e.g. 1-minute can be used) for all
turbines on the wind farm are used to derive an operational state for each turbine and each record.

The state is derived from cause signals only: controller state codes, time-in-state counters (for
example time ready to operate, time in operation, time power reduced, time ice detected), the active
power setpoint and, where needed, pitch angle to detect a parked rotor. Active power and anemometer wind speed are ideally never used to
define a state.

This whole step is site dependent. The signals available, the states worth distinguishing, the rules
that identify them and the use each state can be put to all differ from site to site and with the
goal of the analysis. For each site the analyst therefore gives wind-up two things: the labelling
rules (or already labelled data), and a table of how each label may be used for northing, waking
state and uplift. The states and mapping below are an example of a typical configuration, not a
fixed vocabulary.

The states distinguished in this example are:

- i. **normal operation**. This includes the setpoint limit applied for one interval after cut-in,
  which is part of normal operation;
- ii. **noise mode**: the setpoint is at one of the known noise-mode steps of that turbine, or a
  noise-mode flag is set. The steps are site configuration;
- iii. **grid curtailment**: the setpoint is below rated and is neither a noise-mode step nor the
  start-up limit, within the dates curtailment is known to have been in force;
- iv. **icing**: an ice flag or ice time counter is set. Where the site has no working ice signal,
  icing is identified the IEA Wind Task 19 way: cold records whose power falls well below the
  turbine's own power curve from warm weather. This is a second exception to the rule that active
  power and the anemometer do not define a state;
- v. **partial downtime**: the availability (ready-to-operate) counter is above zero and below the
  full period, or the rotor is parked (pitch beyond a threshold declared for the turbine type);
- vi. **full downtime**: the availability counter is zero;
- vii. **missing**: the signals are absent or out of range, or stuck (unchanged from the turbine's
  own previous record while the wind is not calm), which indicates frozen telemetry. The calm test
  reads the nacelle anemometer: this is the one exception to the rule that the anemometer does not
  define a state, because no upgrade-invariant wind signal exists before step 3.

Data records are flagged as valid or not for each step of the analysis separately, because an
operating state can be valid for some purposes but not for others. The mapping in this example is:

| State             | Northing (step 4) | Waking state (step 5) | Uplift (step 8)         |
| ----------------- | ----------------- | --------------------- | ----------------------- |
| normal operation  | valid             | waking                | valid                   |
| noise mode        | valid             | waking                | see below               |
| grid curtailment  | valid             | part waking           | not valid               |
| icing             | not valid         | part waking           | not valid               |
| partial downtime  | not valid         | part waking           | not valid               |
| full downtime     | not valid         | not waking            | not valid               |
| missing           | not valid         | missing               | not valid               |

Some wind farms will have predictable curtailment which is expected to continue unchanged in future
(for example noise modes). In such cases it may be justified to retain curtailed data in the uplift
analysis, provided the curtailment applies equally in the before and after periods. Noise mode is
therefore valid for uplift only where the analyst declares it so.

Ownership of the states is split:

- wind-up owns the generic states (missing, full downtime, partial downtime and normal operation)
  and their validity, which the analyst cannot redefine. It identifies them itself, from the
  availability counter, stuck telemetry and the declared parked-pitch threshold, so every dataset
  gets at least those states even when the analyst supplies nothing else;
- the analyst supplies a label column and a validity table for the site states only (noise mode,
  grid curtailment, icing and so on in the example above);
- the analyst may also assign a generic state from site rules, for example a stop setpoint becomes
  partial downtime. The generic rules still come first: a record the analyst marks partial downtime
  but whose power is absent is missing.

It is important to carefully inspect power against wind speed, and pitch angle and rotor speed against
power and wind speed, for both the excluded and the retained data of every turbine, to confirm that the
operational state classification matches expectations. Apart from the icing fallback, power is used
here to check the labels, never to make them.

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
  100 m wind speed and the mean wind speed of the unchanged turbines (step 1).
- **Option B.** Establish the SCADA timestamp convention (period start or period end, and time zone)
  explicitly, then interpolate the instantaneous ERA5 fields linearly to the centre of each SCADA
  period. Hourly accumulations and maxima are assigned to the SCADA periods within the hour ending at
  their timestamp, and are not interpolated. The time-shift search is retained only as a check, which is expected to find a shift near zero.
  Where it does not, the timestamp convention is investigated rather than the shift being applied.
TODO just switch to Option B which is more technically justified.

### 4. Northing

The nacelle direction of every turbine is analysed to detect its north calibration and any shifts in it
over the full data period, and is corrected so that a north-calibrated direction is available for every
turbine and every record. The method is described in [northing.md](../northing.md).

Small north changes on an individual turbine may optionally be reported as possible yaw alignment
changes, since these are a candidate cause of performance change (step 1).

### 5. Determine waking state and waking scenarios

The waking state of every turbine on the wind farm, and the waking scenario of every turbine that may
serve as a test or reference turbine, are determined for each record. This is done in data
preparation, before any test turbine or reference set is chosen, because the matching of step 8 and
the features of step 6 depend on it and because it must not change when a reference is dropped.

- i. **Waking state.** The waking state of every turbine is derived from the operational state of step
  2. Each operational state is classed as waking (for example normal operation, noise mode), not waking
  (for example full downtime), part waking (for example partial downtime, grid curtailment) or
  missing. Because the states of step 2 are derived from cause signals only, the waking state of a
  changed turbine is upgrade invariant, as it must be.
- ii. **Direction.** A direction is needed for each turbine to decide which turbines are upwind of it.
  An unchanged turbine (step 1) uses its own north-calibrated direction (step 4). A changed turbine,
  or an unchanged turbine whose own direction is missing, uses the circular mean of the
  north-calibrated directions of up to four of the nearest unchanged turbines that have a valid
  direction in that record. Where no such direction is available the record's waking scenario is
  "unknown". A test turbine's own direction is never used, in keeping with the upgrade invariance
  principle. Because the rule depends only on the changed and unchanged sets of step 1, the columns
  of this step are unaffected by which turbines are later chosen or dropped as references.
- iii. **Waking scenario.** For each turbine, the turbines upwind of it are found for each record:
  those within the IEC 61400-12-1 disturbed sector [1] for the direction of ii, given the turbine
  positions and rotor diameters. A waking scenario column is recorded per turbine (for example
  `T01_waking_scenario`). Its value is:
  - "normal" when no turbine is upwind or every upwind turbine is waking;
  - otherwise a descriptive label of what is abnormal, listing the abnormal upwind turbines and their
    waking state, for example "T03 offline; T04 part waking; T05 offline". The turbines are listed in
    a fixed order so that the same situation always gives the same label;
  - "unknown" when the direction of ii is unavailable or the waking state of any upwind turbine is
    missing.

  This mirrors wind-up v0, which labelled each record of a test-reference pair with the turbines
  upwind of the pair that were offline. Summarising the upwind turbines this way means the
  counterfactual model (step 9) does not have to be given the state of every turbine on the wind farm
  and work out for itself which ones matter, and it lets the before and after periods be matched on
  like-for-like waking scenarios (step 8). Records with part waking upwind turbines are labelled
  explicitly because their wake effect is too ambiguous, and they are not used by default.
- iv. **Changed turbines upwind.** A second column per turbine (for example `T05_changed_upwind`)
  lists the changed turbines (step 1) that are within its disturbed sector for the direction of ii.
  An upgrade can change the wake a turbine casts, so these records are not used for the uplift
  calculation by default (step 8). The column depends only on direction, so it is consistent between
  the before and after periods by construction: it would not make sense to keep records with the test
  turbine upwind in one period and drop them in the other. The analyst can override the default where
  the wake is known not to change.

### 6. Engineer features

The features from which the test turbine's power is predicted are computed for every record. They are
computed for every unchanged turbine, so that any of them can serve as a power reference without
repeating this step. The model of step 9 uses those of the reference set it is given.

- i. **Power references:** for every unchanged turbine, active power and the north-calibrated nacelle
  direction (step 4) as sine and cosine. The direction is a feature in both options below: the test
  turbine's wake exposure within the "normal" scenario varies with direction, and ERA5 direction is
  too coarse to resolve it.
  - **Option A (current default).** No further signals. ERA5 resolves the very low and very high wind
    speeds where reference power carries little information.
  - **Option B.** Pitch angle and rotor speed are added, which allows high wind speed operation to be
    resolved.
- ii. **Waking scenarios:** the waking scenario column (step 5 iii) of each turbine, as a categorical
  feature. After the matching of step 8 most values are "normal", so these features only act where a
  non-normal scenario was retained in both periods.
- iii. **ERA5:** a short list of variables with a clear physical reason to affect the test turbine's
  power relative to its references: 100 m wind speed, 100 m wind direction (as sine and cosine), air
  density derived from temperature, pressure and humidity, a shear exponent derived from the 10 m and
  100 m wind speeds, and a turbulence proxy derived from the gust and mean 10 m wind speeds.

Adding more information about the references has not reduced bias to date. Features are therefore kept
to those with a clear physical justification.

## Part B: the uplift estimator

Steps 7 to 10 take one test turbine, one set of power references and the data of Part A, and return
an uplift. They are written for the test turbine, but step 11 runs them with a reference turbine in
the role of test turbine and the remaining references as its references.

### 7. Define the before and after periods

The dataset is divided into two periods: before and after the upgrade. For a toggle test these two
periods are the toggle off and toggle on data, respectively. For a non-toggle (prepost) test the after
period starts when the upgrade installation of the test turbine is complete and will normally run until
the day when the analysis is conducted, to use as much data as possible. Where turbines are upgraded
one after another, each test turbine has its own before and after periods.

The periods are defined from the test turbine alone: its upgrade date, and any other known change to
the test turbine (step 1), which neither period may include. They do not depend on the references. A
candidate reference that has a change of its own inside the periods is ineligible (step 11) rather
than the periods being trimmed to accommodate it. Where a valuable reference would be kept by a
shorter before period, that is a deliberate choice the analyst makes by setting the period, and the
tool reports the trade-off.

The length of the before period is chosen as follows.

- **Option A (current default).** Use all available data before the upgrade, up to a cap of 24 months,
  with every record weighted equally.
- **Option B.** Use the same calendar months as the after period, in the previous year or years (the
  seasonal matching of wind-up v0). The same effect can be had analytically by including calendar time
  among the conditions matched in step 8, which is the form in which this option is tested.

A long before period gives the counterfactual model (step 9) more data and generalises better to the
after period. A matched before period exposes both periods to the same conditions. Evidence to
date favours each in different circumstances, and the matching step (step 8) may make the choice less
important, so it is decided by validation.

For a toggle test, a toggle pairing filter is applied which only keeps data where the time difference
between a data record and the nearest valid record with the opposite toggle state is within expected
limits.

TODO at this point warnings should be emitted if there are any apparent changes in key turbines eg reactive power changes

### 8. Select valid records and match conditions between the periods

- i. **Validity.** A record is used only if the test turbine and every power reference are valid for
  the uplift calculation in that record (step 2) and none of them is inside an exclusion window
  (step 1). Because the references are modelled together (step 9), the loss of data is the union of
  the references' invalid data: a record in which any one reference is in downtime is lost for all of
  them. This puts pressure against using many references, and is one side of the trade-off of
  step 11.
- ii. **Waking configuration.** For each record, the configuration is the combination of the waking
  scenarios (step 5 iii) of the test turbine and all of its power references, taken together. Only
  configurations with at least 24 hours of data in both periods are retained, in the same way that
  wind-up v0 retained only the waking scenarios with at least one day of data. By default any
  configuration in which a scenario contains a part waking turbine or is "unknown" is dropped, however
  much data it has. A configuration seen in one period only is one the model has rarely or never
  seen, and its prediction for it is an extrapolation. The largest single source of bias identified to
  date was a related extrapolation: after-period records in which a power reference was offline, so
  that its power was missing from the model's inputs. Part i already excludes such records, so this
  part is about the wakes the test turbine and references are exposed to rather than about missing
  reference power. Whether a mismatch in waking configurations between the periods is itself a
  measurable source of bias has not yet been tested and is decided by validation.
- iii. **Changed turbines upwind.** Records in which a changed turbine is upwind of the test turbine
  or of a reference (step 5 iv) are dropped, unless the analyst has overridden this for a wake known
  not to change.
- iv. **Weather.** Both periods should see the same weather. For a toggle test the pairing filter of
  step 7 already ensures this.
  - **Option A (current default).** No weather matching beyond the choice of period in step 7.
  - **Option B.** Common-support trimming: coarsen ERA5 wind speed, direction and a stability proxy
    into cells, and drop the records of either period that fall in cells with less than a minimum
    number of hours in the other period. Calendar time may be added as a condition (step 7 Option B).

Trimming is kept as natural as possible: where practical, whole days are removed rather than individual
records, so that the retained data stays contiguous and autocorrelated in the same way in both periods.

Features that reveal the period rather than the conditions, such as a signal that only exists in one
period, must not be used for matching. Weighting the records by a model of the probability of belonging
to each period is not used, because on real data the period is almost perfectly predictable from such
features.

### 9. Relate the reference measurements to the test turbine measurements

A machine learning model is trained which predicts test turbine power from the features of step 6 for
the reference set. This takes the place of the directional detrending of wind-up v0: rather than a
ratio of wind speeds by direction bin, the model learns how the test turbine's power relates to its
references under every combination of weather and waking state present in the data. Only the records
retained by step 8 are used.

The model is a gradient-boosted tree regressor minimising squared error. It is configured to give
identical results on any machine.

### 10. Calculate the uplift

The model of step 9 trained on the before period, $\hat f_B$, predicts the test turbine's power in the
after period. Writing $y$ for measured test turbine power and $A$ and $B$ for the after and before
records, the forward uplift is

$$
U_{fwd} = \frac{\sum_A y}{\sum_A \hat f_B} - 1
$$

This compares a measured sum with a predicted one, and is biased by any error the model makes on
average when predicting data unlike its training data. That error grows with the time separating the
two periods and is the main source of bias left once steps 5 to 8 are applied. One of the following is
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

The reverse calculation of option B does not use the test turbine anemometer and so can be performed in
full, unlike wind-up v0, which had to fall back on power-only versions of both calculations.

## Part C: the campaign

### 11. Select and screen reference turbines

A set of reference turbines is chosen for each test turbine. These are the power references: turbines
whose power is used to predict the test turbine's power. The selection has two stages: a ranked pool
of candidates, and a screen that runs the estimator of Part B with each candidate in the role of test
turbine.

**Candidates.** Every unchanged turbine (step 1) is a candidate. The following factors are considered
when ranking them:

- i. Turbines for which the active power correlates most strongly with the test turbine are favoured.
  Normally these are also the closest turbines to the test turbine, and in practice the nearest
  eligible turbines within 20 rotor diameters are taken.
- ii. A candidate must have data over the whole of the periods of step 7, should have no known
  performance issues, and must not have a change of its own (step 1) inside those periods. A
  candidate with such a change is ineligible; the periods are not trimmed to keep it (step 7).
- iii. It helps where possible for a reference turbine to be un-waked in the predominant wind direction,
  especially by the test turbine itself.

At least three reference turbines are required, and more can be used where available. The number
used is a trade-off. Each reference's own drift relative to the test turbine is averaged down as more
references are added, but the usable data is the intersection of the references' valid data
(step 8 i), so each added reference costs hours. The tool reports usable hours against number of
references so the analyst can see the cost.

Every other turbine on the wind farm, including other test turbines and turbines excluded as power
references, still contribute to the waking scenario (step 5).

**Screen.** The estimator (steps 7 to 10) is run with each candidate in turn as the test turbine,
predicted from the other candidates, with the same periods, matching and features. The expectation is
that 0 % uplift is measured for each candidate. The purpose is to detect any bias in the references,
by seeing whether any of them have a measurable change in performance relative to the others.

A candidate whose uplift differs from the median of the candidates by more than a floor (2.5
percentage points by default) is excluded as a power reference and the remaining candidates are
judged again, excluding at most one turbine per pass. The median rather than zero is used because the
reference turbines may all drift together, which is not a source of bias. When a candidate is
excluded the next in the ranking takes its place. The screen requires at least three candidates and
enough after-period data to separate a bad reference from noise. Because the columns of step 5 do
not depend on the reference set, excluding a candidate changes nothing in Part A. The screened set is
the set of power references used for the test turbine in step 12.

The combined reference uplift (step 14) is reported alongside every result as the campaign's own check
of bias, and the spread of the individual reference uplifts as its own measure of noise. It is
reported both before and after the screen: the value after the screen is near zero partly by
construction, so the value before it is the more honest check.

The reference uplifts could also be used to correct the test turbine uplift. This is not done, because
the references would then read 0 % by construction and could no longer be used as a check of bias.

**Joint or paired references.** By default the references are analysed together as one model for each
test turbine (step 9), not one test-reference pair at a time. A single-reference estimate carries
roughly twice the shared bias of a multi-reference one, and each reference's own drift relative to
the test turbine is averaged down as more references are added.

- **Option A (current default).** One joint model per test turbine.
- **Option B.** The estimator is run once per test-reference pair and the pair results are combined.
  This keeps more data, because the validity of step 8 i is not unioned across references, and lets
  a reference be dropped after the fact without re-running. Whether its larger shared bias remains
  once the rest of the method is in place is decided by validation.

### 12. Calculate the test turbine uplift

The estimator of Part B (steps 7 to 10) is run for the test turbine with the screened reference set
of step 11. Its output is the headline uplift of the test turbine.

### 13. Calculate uplift distributions

Uplift is calculated as a function of condition variables specified by the analyst. Uplift against
wind speed is generally the most important for pre-construction inputs, but it has to be based on a
stable, free-stream wind speed estimate derived from the reference turbines (power and pitch angle), not
from the test turbine. For certain upgrades other conditions are important, for example wake steering
also requires uplift against direction and atmospheric stability.

Records are assigned to a bin by an upgrade-invariant variable, never by the test turbine's measured
power, which would select each bin on its own noise. The binned uplifts are scaled so that they combine
to the headline uplift of step 12. The tool should also determine which variables the uplift is
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

- Which before period (step 7 options) is preferred once matching (step 8) is in place.
- How the ERA5 stability proxy and weather cells of step 8 are defined, and the minimum hours per cell.
- Whether a mismatch in waking configurations between the periods (step 8 ii) is a measurable source of
  bias now that offline references are excluded by step 8 i, and so how much the 24 hour rule matters.
- How the free-stream wind speed estimate of step 13 is constructed.
- How site-specific operating states (step 2) reach wind-up: pre-labelled data or a labelling schema.
- How many references to use by default (step 11), given the union of invalid data in step 8 i.

## References

[1] Wind energy generation systems - Part 12-1: Power performance measurements of electricity producing
wind turbines (IEC 61400-12-1:2022). International Electrotechnical Commission, September 2022.
