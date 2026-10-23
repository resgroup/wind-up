# wind-up uplift measurement method

The wind-up tool measures energy yield uplift and uncertainty of wind turbines after they are changed in
some way such as a turbine upgrade or implementation of wind farm control. The wind-up methodology uses
best practice from the turbine power performance standard IEC 61400-12-1 [1] and adds innovative steps to
achieve additional goals such as measurement of uplift under all conditions (including waked operation).

Met mast or LiDAR data could serve as the reference, as in an IEC 61400-12-1 power performance test,
but this document covers only the case where they are not available. Neighbouring turbines that have
not been changed are then used as reference turbines to measure the uplift of upgraded test turbines.

This document describes the target state of the v1 method. Where more than one approach is still under
consideration the options are listed, with the current default marked. An option is retired once the
validation described at the end of this document has decided between them.

## Principles

The following principles apply to every step:

- **Upgrade invariance.** Every input used to predict the test turbine's power must be unaffected by the
  upgrade. Signals from the test turbine itself are never used as model inputs, and signals from any
  turbine that may have changed during the analysis period are reduced to those that cannot carry a
  performance change. Upgrades can change wakes, so data affected by test (or changed) turbine wakes
  is also removed.
- **Wind turbine anemometers are never used as model inputs.** Their calibration drifts and can change
  with a turbine upgrade. They may still be inspected as a possible indicator of performance change
  and used to help verify, for example that the ERA5 time alignment is correct.
- **Direction data is only used after it has been vetted and corrected** (step 4).
- **Filter on cause, not effect.** Data is excluded because of what a turbine was doing (offline,
  curtailed, faulted) and never because of what it produced. Selecting on lower-than-expected power
  would bias the result.
- **Matching conditions.** Bias risk is reduced when the before and after datasets are exposed to the
  same conditions: weather and operating state of the surrounding turbines (ie wake exposure).
- **Data quantities are expressed in hours, not records**, so that the method applies to any SCADA
  timebase.

## Method steps

The wind-up methodology comprises the following steps, in four parts.

- **Part A, data preparation (steps 1 to 6)** is done once per dataset. It depends on the test turbines only
  as the set declared for the campaign, and not on which turbines are their references, so it is never
  repeated.
- **Part B, the campaign for one test turbine (steps 7 to 10)** chooses the before and after periods and
  the candidate references together from the campaign timeline, screens the candidates by measuring each
  as if it were a test turbine, and then measures the test turbine with the screened references.
- **Part C, the measurement of one turbine's uplift (steps 11 to 15)** is a procedure that takes one
  subject turbine, one set of power references and the before and after periods, and returns an uplift.
  It is the same procedure whether the subject is the test turbine (step 10) or a candidate reference
  (step 9). Nothing in it changes the reference set or the periods.
- **Part D, across turbines (steps 16 and 17)** combines the measurements of the test and reference
  turbines and extrapolates the result to annual energy production.

This separation is what resolves a circularity that is otherwise easy to fall into: the usable data
depends on the references, the references depend on a screen, and the screen needs the usable data.
Here the periods and the candidate references are chosen from the campaign timeline alone, which is
declared rather than measured; the references are screened with those periods fixed; and the test
turbine's uplift is then calculated with the screened references.

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

Each identified change informs how the operating-state labels (step 2), the analysis dates and
turbines (steps 7 and 8) and the data exclusions are configured.
Records inside an exclusion window are not used in the uplift calculation, but the turbine still
contributes waking information in those records (step 5). An exclusion does not make a turbine changed:
for example, a temporary de-rate for a component health issue leaves the turbine notionally the same
before and after it.

The outcome of this step is plots the analyst can review to confirm the data is fit for purpose and
to help define manual exclusions and the operating-state classification.

### 2. Derive operational state and data validity

Historic SCADA data (typically 10-minute but other timebases e.g. 1-minute can be used) for all
turbines on the wind farm are used to derive an operational state for each turbine and each record.

The state is derived from cause signals only: controller state codes, time-in-state counters (for
example time ready to operate, time in operation, time power reduced, time ice detected), the active
power setpoint and, where needed, pitch angle to detect a parked rotor. Active power and anemometer
wind speed are ideally never used to define a state.

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
- vii. **missing**: the active power or the availability counter is absent or out of range, or stuck
  (unchanged from the turbine's own previous record while the wind is not calm), which indicates
  frozen telemetry. The calm test
  reads the nacelle anemometer: this is the one exception to the rule that the anemometer does not
  define a state, because no upgrade-invariant wind signal exists before step 3.

Data records are flagged as valid or not for each step of the analysis separately, because an
operating state can be valid for some purposes but not for others. The mapping in this example is:

| State             | Northing (step 4) | Waking state (step 5) | Uplift (step 11)        |
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
weather information: it is unaffected by any change to the turbines. This is done once, in data
preparation, and every later step uses the same aligned ERA5. It serves the time-alignment check
below, northing (step 4) and the wind rose of the reference ranking (step 8); it is not a feature of
the power model (step 6), nor an axis of the uplift distributions (step 15).

ERA5 is obtained from the Open-Meteo archive API with the ERA5 model selected, at the centroid of the
wind farm and over the whole SCADA record provided. Its values are interpreted as follows:

- i. **Time.** Timestamps are in UTC. Wind components, temperature and pressure are instantaneous
  values valid at the hourly timestamp, not averages over the hour. Accumulated fields such as
  precipitation are totals over the hour ending at the timestamp, and the 10 m wind gust is the maximum
  over the hour ending at the timestamp.
- ii. **Space.** ERA5 has a native resolution of about 31 km and is provided on a 0.25° grid. A grid
  point value represents an average over the model grid box, not the conditions at the requested
  coordinates, and it cannot represent variability on spatial scales smaller than the grid box. By
  default Open-Meteo returns a nearby land grid cell of similar elevation to the requested coordinates
  rather than strictly the nearest one. For an offshore wind farm a sea grid cell is requested instead.
- iii. **Smoothness.** Being a grid box value, an instantaneous ERA5 value is smooth in time and does
  not represent the short-term variability of a turbine's 10-minute mean. ERA5 is therefore a
  weather-state feature rather than a substitute for a measured wind speed.

The hourly values are aligned to the SCADA timebase as follows.

- **Timestamps.** The SCADA timestamp convention (period start or period end) and time zone are
  declared, by default period start and UTC. The SCADA timestamps must be timezone-aware and in the
  declared time zone; they are then converted to period-start UTC.
- **Instantaneous fields** are interpolated linearly to the centre of each SCADA period. Wind
  directions are interpolated through the wind vector components at the same height, so that the
  interpolation is correct across north; wind speed is interpolated as a scalar.
- **Hourly accumulations and maxima** are assigned to the SCADA periods within the hour ending at
  their timestamp, and are not interpolated.
- **Gaps and coverage.** ERA5 is not extrapolated beyond its record. A gap in the ERA5 record (a
  missing hour or a missing value) is an error. Where ERA5 does not cover the whole SCADA record a
  warning is given.

  > **TODO (ERA5 missing at the end of the record).** ERA5 lags real time, so the latest month or two
  > of a live campaign's SCADA may have no ERA5. Test that an ERA5 record ending one to two months
  > before the SCADA leaves only that tail uncovered, with the warning, and that a fetch on a cache miss
  > does not store hours the archive cannot yet provide, so that they are fetched again later.
- **Time-shift check.** The time shift (searched over ±24 hours) that maximises the correlation between
  ERA5 100 m wind speed and the mean nacelle wind speed of the unchanged turbines (step 1, records valid
  for northing in step 2) is found as a check only, and is never applied. A best shift of more than
  30 minutes gives a warning, and 1 hour or more is an error: either points at a mis-declared time
  zone or timestamp convention, which is investigated rather than corrected by a shift.

The outcome of this step is the aligned ERA5 and plots of the time-shift check and of ERA5 against the
site wind speed.

### 4. Northing

The nacelle direction of every turbine is analysed to detect its north calibration and any shifts in it
over the full data period, and is corrected so that a north-calibrated direction is available for every
turbine and every record. The method is described in [northing.md](../northing.md).

The north tables are found once for the whole wind farm over the whole SCADA record, and are applied to
every record of every turbine. They are learnt from the records valid for northing (step 2) in which
the turbine generates more than 5% of its rated power, because wind veer is larger at low wind speed.

Small north changes on an individual turbine are reported as possible yaw alignment changes, since these
are a candidate cause of performance change (step 1). A north change of 30° or more that lasts less
than 90 days is a north calibration excursion; one that lasts longer is a re-calibration and becomes the
new reference frame. Every other change, between two segments in the reference frame, is a candidate yaw
alignment change. For each candidate, the change in the turbine's apparent power coefficient (its power
divided by the cube of its own nacelle wind speed, binned by wind speed) is measured between the 60 days
before and the 60 days after, using the records valid for northing (above 5% of rated power, as for the
north tables) and for uplift. It is compared with the largest such change at any other date of the
turbine's record that is more than 60 days from every candidate. The report is for the analyst to inspect
alongside the declared changes; it does not change the uplift calculation.

> **TODO (ERA5 missing at the end of the record).** The reanalysis anchor cannot find a north change in
> a tail with no ERA5. On a farm with enough turbines the changes are then found against the farm
> consensus, which does not need ERA5; on a smaller farm they are found against reanalysis, and a change
> in the tail would go undetected. Test with a north step placed in an ERA5-less tail, on a large and a
> small farm, and decide what the small farm falls back to.

### 5. Determine waking state and waking scenarios

The waking state of every turbine on the wind farm, and the waking scenario of every turbine that may
serve as a test or reference turbine, are determined for each record. This is done in data
preparation, before any test turbine or reference set is chosen, because the matching of step 11 and
the features of step 6 depend on it and because it must not change when a reference is dropped.

- i. **Waking state.** The waking state of every turbine is derived from the operational state of step
  2. Each operational state is classed as waking (for example normal operation or noise mode; waking
  means capable of making a wake if the wind is high enough, even if the turbine is not spinning at
  the time), not waking (for example full downtime), part waking (for example partial downtime, grid
  curtailment) or missing. Because the states of step 2 are derived from cause signals only, the
  waking state of a changed turbine is upgrade invariant, as it must be.
- ii. **Direction.** A direction is needed for each turbine to decide which turbines are upwind of it.
  A turbine's direction is valid in a record that is valid for northing (step 2), without the power
  restriction of step 4: a turbine that is offline may not be facing into the wind. Every turbine other
  than a test turbine uses its own north-calibrated direction (step 4) where it is valid. A test
  turbine, or another turbine without a valid direction, uses the circular mean of the north-calibrated
  directions of up to four of the nearest turbines that are not test turbines and have a valid
  direction in that record. Where no such direction is available the record's waking
  scenario is "unknown". A test turbine's own direction is never used, in keeping with the upgrade
  invariance principle. Other changed turbines do contribute direction: a dataset can include a full
  roll-out in which every turbine is changed at some point, and a change to a turbine shifts its
  north-calibrated direction far less than the width of a disturbed sector. The test turbines are
  those declared for the campaign, so the columns of this step are unaffected by which turbines are
  later chosen or dropped as references.
- iii. **Waking scenario.** For each turbine, the turbines upwind of it are found for each record:
  those within the IEC 61400-12-1 disturbed sector [1] for the direction of ii, given the turbine
  positions and rotor diameters. A waking scenario is recorded for every turbine and record. Its
  value is:
  - "normal" when no turbine is upwind or every upwind turbine is waking;
  - otherwise a descriptive label of what is abnormal, listing the abnormal upwind turbines and their
    waking state, for example "T03 not waking; T04 part waking; T05 not waking". The turbines are listed in
    a fixed order so that the same situation always gives the same label;
  - "unknown" when the direction of ii is unavailable or the waking state of any upwind turbine is
    missing.

  This mirrors wind-up v0, which labelled each record of a test-reference pair with the turbines
  upwind of the pair that were offline. Summarising the upwind turbines this way means the
  counterfactual model (step 12) does not have to be given the state of every turbine on the wind farm
  and work out for itself which ones matter, and it lets the before and after periods be matched on
  like-for-like waking scenarios (step 11). Records with part waking upwind turbines are labelled
  explicitly because their wake effect is too ambiguous, and they are not used by default.
- iv. **Upwind turbines.** A second value per turbine and record lists every turbine
  within its disturbed sector for the direction of ii, whatever its waking state, in a fixed order. Its
  value is "none" when no turbine is upwind and "unknown" when the direction of ii is unavailable. Which
  of these turbines count as changed depends on the before and after periods of the test turbine, so
  the records with a changed turbine upwind are found in step 11 rather than here.

### 6. Engineer features

The features from which the test turbine's power is predicted are defined here for every record and
every unchanged turbine, from the outputs of steps 2 to 5, so that any turbine can serve as a power
reference without repeating Part A. The measurement of Part C assembles those of the reference set it is
given.

- i. **Power references:** for every unchanged turbine, active power and the north-calibrated nacelle
  direction (step 4) as sine and cosine. The direction is a feature in both options below: the test
  turbine's wake exposure within the "normal" scenario varies with direction.
  - **Option A (current default).** No further signals.
  - **Option B.** Pitch angle and rotor speed are added, which allows high wind speed operation to be
    resolved.
- ii. **Waking scenarios:** the waking scenario column (step 5 iii) of the test turbine and of each
  power reference, as a categorical feature. They carry the waking state of every other turbine on
  the wind farm, which contributes no feature of its own. After the matching of step 11 most values
  are "normal", so these features only act where a non-normal scenario was retained in both periods.
- iii. **Solar elevation:** the sun's apparent elevation at the centroid of the turbines, at the middle
  of each record's period. It correlates with atmospheric stability, and so with shear, and needs
  nothing beyond the timestamp, so it covers every record.

ERA5 is not a feature: a model fitted on one period leans on it in a way that does not carry over to a
period it has not seen, and it lags real time, so the last month or two of a live campaign may have
none (CF34). Features are kept to those with a clear physical justification; adding more information
about the references has not reduced bias to date.

## Part B: the campaign for one test turbine

Steps 7 to 10 are run once per test turbine. They decide the inputs of every measurement made for that
test turbine (the periods and the power references) and call the measurement of Part C: first with each
candidate reference as the subject, to screen the references (step 9), then with the test turbine as
the subject and the screened references as its power references (step 10).

### 7. Choose the before and after periods

The dataset is divided into two periods: before and after the upgrade. For a toggle test these two
periods are the toggle off and toggle on data, respectively; data from before toggling began may
also be used to train the model of step 12, but the matching of step 11 and the uplift of step 13
use the toggle data only. For a non-toggle (prepost) test the after period starts when the upgrade
installation of the test turbine is complete, and the before period ends when that installation
starts. Where turbines are upgraded one after another, each test turbine has its own before and
after periods.

The periods are chosen so that enough nearby turbines stay eligible as references (step 8), from the
campaign timeline alone: the works windows and data extents of every turbine and the declared
exclusions, never anything measured. Longer periods are preferred, each up to a cap (24 months before
and 12 months after by default), but neither period may include a known change to the test turbine
(step 1), and a period is shortened where that keeps enough good references unchanged: a span that
ends at a neighbour's upgrade and keeps four unchanged references nearby beats a longer one that leaves
only distant references. When no span gives both periods a minimum length (3 months by default) with
enough references, the test turbine is reported as not analysed, with the reason, and the campaign
continues without it. The analyst can instead declare the periods, which are then used as given. How
the trade-off is made is described in
[analysis-period-and-reference-selection.md](../analysis-period-and-reference-selection.md).

Every record of the before period is weighted equally. A long before period gives the counterfactual
model (step 12) more data and generalises better to the after period; the matching of step 11 is what
exposes both periods to the same conditions.

### 8. Rank the candidate references

A set of reference turbines is chosen for each test turbine. These are the power references: turbines
whose power is used to predict the test turbine's power. This step decides, from the timeline alone,
which turbines may serve and in what order; step 9 then tests them on their data and may replace some.

A candidate is eligible when it has data over the whole of the periods of step 7, no change of its own
(step 1) inside them, and lies within a distance limit of the test turbine (20 rotor diameters by
default). A turbine changed entirely before or entirely after the periods is eligible. The eligible
candidates are ranked by distance, which stands in for how strongly their power correlates with the
test turbine's, with one exception: a candidate the test turbine is upwind of, within the wake reach of
the change (step 11 iii), for more than a share of the site's wind (20 % by default) is **demoted**
behind every candidate that is not. Its records would be lost in those directions (step 11 iii), so a
slightly further candidate that the test turbine does not wake is the better reference. The share is
the part of the wind rose, built from the step 3 reanalysis direction over the campaign's data, inside
the IEC disturbed sector of the test turbine as seen from the candidate. The best-ranked few (four by
default) are the power references the screen starts from; the rest, in order, are reserves, from which
the screen draws a replacement for each reference it rules out.

An indicator of an undeclared change (step 1) inside the periods, for example a change in reactive
power behaviour, is reported as a warning against the candidate, so that the analyst can declare the
change or exclude the turbine; it does not by itself make the candidate ineligible.

At least three reference turbines are required, and more can be used where available. The number
used is a trade-off. Each reference's own drift relative to the test turbine is averaged down as more
references are added, but the usable data is the intersection of the references' valid data
(step 11 i), so each added reference costs hours. The data screen of step 9 reports that cost.

Every other turbine on the wind farm, including other test turbines and turbines excluded as power
references, still contribute to the waking scenario (step 5).

### 9. Screen the references

The power references of step 8 are screened in two passes, the first on what each candidate's data can
support and the second on what it shows. A candidate dropped by either pass is replaced by the nearest
reserve (step 8), which is screened in its turn. The order is always step 8's ranking: the screen
only removes and replaces, it never re-ranks.

**Pass 1: the data screen.** For each candidate, the record selection of step 11 is applied to the pair
of the test turbine and that candidate alone. The filters that apply to each turbine on its own are
used: validity (step 11 i), exclusions, waking scenario and changed turbines upwind (step 11 iii).
The filters that depend on the whole reference set, the waking configuration rules of step 11 ii,
are not. From the records that survive, the usable hours in each period and the correlation of the
candidate's power with the test turbine's over all usable records are computed. Correlation is
insensitive to a scale change, so the after period contributes to it without the upgrade leaking
in. A candidate is dropped when its usable hours in either period are below half the median of the
candidates, or its correlation is more than 0.2 below their median; both defaults are decided by
validation. This is what catches a reference that sits downwind of the test turbine in the prevailing
wind: it is close and unchanged, but most of its records are lost to the changed-upwind rule, and
keeping it would cost every other reference those hours too (step 11 i). Every candidate's hours and
correlation are reported, and with them usable hours against number of references, so the analyst
can see the cost of each reference. The mechanics are in
[analysis-period-and-reference-selection.md](../analysis-period-and-reference-selection.md).

**Pass 2: the measurement screen.** The measurement of Part C (steps 11 to 14) is run with each
remaining candidate in turn as the subject turbine, predicted from the other candidates, with the
periods of step 7 and the same matching and features. The uplift distributions of step 15 are not
needed for the screen and are off by default. The expectation is that 0 % uplift is measured for each
candidate. The purpose is to detect any bias in the references, by seeing whether any of them have a
measurable change in performance relative to the others.

A candidate whose uplift differs from the median of the candidates by at least a floor (2.5
percentage points by default) is excluded as a power reference and the remaining candidates are
judged again, excluding at most one turbine per pass and fewer than half of the candidates in all.
The median rather than zero is used because the reference turbines may all drift together, which is
not a source of bias. The screen requires at
least three candidates, and a candidate is only judged when it has enough after-period data to
separate a bad reference from noise (150 days of the candidate's own after-period records valid for
uplift (step 2) by default, counted before the record selection of step 11; the value is decided by
validation). A candidate with less is not judged: it is not used as a power reference,
and it is reported as unjudged rather than ruled out; like a dropped candidate, it is replaced by
the nearest reserve. When no candidate has enough, the screen is not run and every candidate stays a
power reference. Because the columns of step 5 do not depend
on the reference set, excluding a candidate changes nothing in Part A. The screened set is the set
of power references used for the test turbine in step 10.

This pass is run for prepost tests only. In a toggle test a change to a reference before the test
affects the toggle on and toggle off data alike, so the contrast cannot show it; the references
are still measured and reported, with the same toggle periods as the test turbine, as the
campaign's check of noise.

Each candidate's measurement from this pass is what is kept and reported; the reference is not
measured again against a different set of references. The combined reference uplift of step 16,
before and after the screen, is the campaign's own check of bias.

### 10. Measure the test turbine

The measurement of Part C (steps 11 to 15) is run for the test turbine with the screened reference set
of step 9 and the periods of step 7. Its output is the headline uplift of the test turbine and its
uplift distributions, each with the uncertainty of step 14.

## Part C: measure one turbine's uplift

Steps 11 to 15 take one subject turbine, one set of power references, the before and after periods and
the data of Part A, and return an uplift. The subject is the test turbine in step 10 and a candidate
reference in step 9; the procedure is the same for both, and the steps below say "subject turbine" for
either. Nothing in Part C changes the reference set or the periods.

### 11. Select valid records and match conditions between the periods

- i. **Validity.** A record is used only if the subject turbine and every power reference are valid for
  the uplift calculation in that record (step 2) and none of them is inside an exclusion window
  (step 1). Because the references are modelled together (step 12), the loss of data is the union of
  the references' invalid data: a record in which any one reference is in downtime is lost for all of
  them. This puts pressure against using many references, and is one side of the trade-off of
  step 8. A measurement needs a minimum of usable data in each period (96 hours by default); below
  it no uplift is returned, and a test turbine is reported as not analysed with the reason, as in
  step 7. Quantities here are hours, not records.
- ii. **Waking configuration.** A record's configuration is the waking scenarios (step 5 iii) of the
  subject turbine and all of its power references, taken together. Two rules apply:
  - a record is dropped when any of those scenarios contains a part waking turbine or is "unknown"
    (this includes the records of part iii without a direction). This rule looks at each turbine on its
    own, so the data screen (step 9) applies it too;
  - a record is dropped when its configuration has less than 24 hours of data in either period, as
    wind-up v0 kept only the waking scenarios with at least a day of data. A configuration seen in one
    period only is one the model must extrapolate to. For a toggle test the periods are the toggle on
    and off data, and a record from before toggling began (step 7) is dropped when its configuration is
    rare in them.

  The hours each rule removes are reported per period. The second rule fragments as references are
  added, since every combination of their scenarios is its own configuration, and the campaign matrix
  shows a small bias where it removes the most hours (open question).
- iii. **Changed turbines upwind.** For this measurement, a turbine is changed if it has a change
  (step 1) between the start of the before period and the end of the after period.
  A turbine whose changes all fall outside that span casts the same wake in both periods. Records in
  which a turbine is upwind of the subject turbine or of a reference (step 5 iv) at a time it counts as
  changed, or in which either upwind column is "unknown", are dropped. A change counts only within its
  **wake reach**: upwind of a turbine at most that many of the changed turbine's rotor diameters away.
  The reach follows from the AEP uplift expected of the change, declared with the campaign: 0 D for
  no uplift, 4 D at 1 %, 8 D at 3 %, 12 D at 5 % and 20 D at 10 % or more, interpolated between and
  doubled offshore up to 20 D. It is the largest spacing at which a wake model (FLORIS, with the gain
  taken from induction, the upper bound on wake change) puts the reference's sector-mean energy change,
  times the sector's share of a uniform rose, above 0.05 pp. A declared reach overrides it, and each
  works window may carry its own. When it counts as changed
  depends on how far its change is from the test turbine's changeover:
  - **Near** (less than 6 months between the two works windows, the default; the test turbine itself
    is always near): changed over the whole span. An upgrade can change the wake a turbine casts, so
    the rule depends only on direction and drops the same conditions from both periods: it would not
    make sense to keep records with the test turbine upwind in one period and drop them in the other.
  - **Far**: changed only on the side of its works window away from the changeover, the window
    included. A turbine upgraded late in the after period costs only the records from the start of its
    works; one upgraded early in the before period costs only the records until the end of its works.
    The records kept see it in the state it had at the changeover. Its wake is the same in all of them,
    and dropping its upwind directions for the whole span would waste most of the data when the
    neighbours are upgraded one after another.

  Each works window of a turbine is treated this way and the dropped times add up. Where the wake is
  known not to change the analyst declares a reach of 0, so the change costs no records, while the
  turbine's other works and every other turbine's still count.
- iv. **Weather.** Both periods should see the same weather. For a toggle test a toggle pairing filter
  keeps a record only when the nearest valid record with the opposite toggle state is within one toggle
  cycle (the on and off durations together, by default). A prepost test has no weather matching beyond
  the choice of period in step 7. Trimming both periods to common ERA5 weather support was tried and not
  adopted: no shift between the periods predicted a reading's error, and every trim cost precision
  (CF33). Weighting records by their probability of belonging to each period is not used either, since
  on real data the period is almost perfectly predictable.

### 12. Relate the reference measurements to the subject turbine measurements

A machine learning model is trained which predicts subject turbine power from the features of step 6
for the reference set. This takes the place of the directional detrending of wind-up v0: rather than a
ratio of wind speeds by direction bin, the model learns how the subject turbine's power relates to its
references under every combination of weather and waking state present in the data. Only the records
retained by step 11 are used.

The model is a gradient-boosted tree regressor minimising squared error. It is configured so that
repeated runs on one machine give identical results; results on different machines can differ by a
few tenths of a percentage point.

The references are modelled together, one model per subject turbine, not one subject-reference pair at
a time: a single-reference estimate carries roughly twice the shared bias of a multi-reference one, and
each reference's own drift relative to the subject turbine is averaged down as more are added. Pairs
would keep more data, since the validity of step 11 i would not be unioned across references, but would
need their own screen and their own steps 13 and 16; they are revisited only if validation shows the
data lost in step 11 i limits accuracy.

### 13. Calculate the uplift

The model of step 12 trained on the before period, $\hat f_B$, predicts the subject turbine's power in
the after period. Writing $y$ for measured subject turbine power and $A$ and $B$ for the after and
before records, the forward uplift is

$$
U_{fwd} = \frac{\sum_A y}{\sum_A \hat f_B} - 1
$$

Each prediction is clipped to the range from the smaller of zero and the lowest power the model was
trained on to the larger of the rated power and the largest power the model was trained on.

The forward uplift is biased by any error the model makes on average when predicting data unlike its
training data. That error grows with the time separating the two periods and is the main source of bias
left once steps 5, 6, 7 and 11 are applied. It is removed with the references (option C, the default).

- **Option C, reference correction (current default).** The screen of step 9 has already measured each
  power reference as if it were a test turbine, with the same periods, the same kind of model and the
  other references as its power references. A reference has no upgrade, so its forward uplift $U_i$ is
  the model's error over those periods, and the test turbine's forward uplift carries much of the same
  error. The uplift is the forward uplift less the mean of the $K$ references kept by the screen, in log
  space:

  $$
  \ln(1 + U) = \ln(1 + U_{fwd}) - \frac{1}{K}\sum_{i=1}^{K} \ln(1 + U_i)
  $$

  The references ruled out or not judged by the screen, and any that could not be measured, are left
  out of the mean; with no reference left the uplift is not reported. It needs no extra model fits, as
  the reference measurements of step 9 are reused. The bins of step 15 move with it, as they are shifted
  to the headline. The references themselves are not corrected: their own uplifts are reported for every
  test turbine together with the correction made from them, and a correction far from zero raises the
  uncertainty (step 14).
- **Option A, the forward uplift alone,** is what option C corrects.
- **Option B, reversibility (tried, not adopted).** As in wind-up v0, a second model $\hat f_A$ is
  trained on the after period and predicts the before period,

  $$
  U_{rev} = \frac{\sum_B y}{\sum_B \hat f_A} - 1, \qquad U = \sqrt{\frac{1 + U_{fwd}}{1 + U_{rev}}} - 1,
  $$

  which cancels a bias common to both directions; a weighted form shrinks the correction by a
  calculated $\lambda$ between 0 (forward) and 1 (full reversal). It over-corrects when the after period
  is shorter than the before period, because the reverse model is trained on less data, and failed the
  slope target of the validation. Step 15 still uses the full reversal per bin. Unlike wind-up v0 it
  needs no anemometer, so both directions are computed in full.

### 14. Estimate uncertainty

An uncertainty is reported for the headline uplift of step 13 and for each measured bin of the
uplift distributions of step 15. It has three components, reported apart: a sampling component, the
uncertainty of the reference correction of step 13 option C, and a systematic component for the
error that remains when the relation between the subject turbine and its references changes within the
campaign. The sampling component is independent of the other two, but the other two are not
independent of each other: the correction removes the error the subject turbine shares with its
references, and the systematic component measures that same kind of error. They combine as

$$\sigma^2 = \sigma_{\text{sampling}}^2 + \sigma_{\text{reference}}^2 + \sigma_{\text{systematic}}^2
- 2\rho\,\sigma_{\text{reference}}\,\sigma_{\text{systematic}},$$

with $\rho = 0.5$ by default (set by validation) and the reference and systematic terms together
floored at zero. A sigma a little too large is preferred to one too small.

The sampling component is a circular block bootstrap with the model of step 12 fixed. The after
period's records of the subject turbine, each with its measured and its counterfactual power, are
resampled in blocks of wall-clock time (one day by default) tiled over the after period, the ratio
of sums of step 13 is re-formed for each resample, and the sigma is the standard deviation of the
resampled uplifts. The bins of step 15 are resampled with the same blocks, each resample re-measured,
re-shifted and re-cast, with the model and the bin labels fixed, so the uncertainty never moves an
uplift; the reference gate of step 15 ii is not resampled, and a bin with no measured power bin behind
it has no sigma. Each bin's sigma also takes, in quadrature, the references' spread of deviations in
that power bin (step 15 ii), cast to the other axes by counterfactual energy like the uplift. The sigma
is also recorded at
other block lengths (6 hours, 1 day, 7 days and 28 days by default) so that validation can choose
the block length by the coverage of the sigma against known truth. Every reference turbine
measured in step 9 carries the same sampling sigma, so its reading against the expected 0 % can be
judged in sigmas.

The uncertainty of the reference correction is the standard deviation of the kept references'
$\ln(1 + U_i)$ over $\sqrt{K}$, in quadrature with half the size of their mean, scaled to the uplift:
a large correction means the model extrapolated poorly between the periods, and the test turbine's
error then follows its references' less closely. With a single kept reference it is not known and
neither is the sigma. It is added to the headline's sampling sigma and to each measured bin's, as
above, since the correction moves every bin by the same factor. The sampling sigma alone is recorded
too.

The systematic component is measured inside the before period. Windows as long as the after period,
or half the before period when that is shorter, are held out of the before period in turn, tiled
back from its end (at most 12 by default). For each window the model of step 12 is trained on the
rest of the before period and predicts the window, which is read as if it were an after period; a
window, or the rest of the before period, with fewer usable hours than step 11 i requires is skipped.
With no upgrade in the before period each reading is an error. Its spread is the readings' root mean
square less their own sampling sigmas (the same bootstrap, per window) in quadrature, floored at
zero, and multiplied by the one-sigma quantile of Student's $t$ with as many degrees of freedom as
windows, since a few windows estimate it poorly; with no window left it is not known and neither is
the sigma. It is added to the headline and to each measured bin, as above. It is a sigma only: the
readings' mean is not used to correct the uplift. Each window costs one model fit. A toggle test has no systematic component, since its periods interleave.

wind-up v0 instead took the largest of its statistical, bootstrap and reversibility sigmas and
floored it at the weighted mean absolute reference uplift. How sigmas combine across references and
across test turbines is left until the components are trusted.

### 15. Calculate uplift distributions

The headline of step 13 is distributed over conditions: against free-stream wind speed, the input a
pre-construction or wake model needs, and against direction, a stability proxy and wake exposure. The
distributions say what shape the uplift has; they never move the headline, and their total equals it.

- i. **The measured shape.** The after period's records are binned by the subject turbine's
  counterfactual power from the model of step 12, in eleven even bins from −5 % to 105 % of rated. The
  prediction comes from the references, so it is upgrade-invariant, and a regression model's error has
  zero mean among the records that share a prediction, so the forward ratio of step 13 needs no
  correction on this axis. A bin with at least 24 hours of records is measured: its uplift is
  $\sum y / \sum \hat f_B - 1$ over its records. The measured bins are then shifted by one constant so
  that, weighted by counterfactual energy, they combine to the headline; a bin not measured carries the
  headline.
- ii. **The reference gate.** A model's bias against predicted power, strongest near cut-in, shows in the
  references too, which have no uplift. Each kept reference of step 13 is measured the same way, and
  each power bin's deviation from its own overall uplift is recorded. A measured bin of the subject
  turbine keeps its own uplift only where its deviation differs from the references' mean deviation by
  more than two sigma, the sigma joining the bin's sampling sigma and the references' spread of
  deviations in that bin (step 14); the headline's reference and systematic components move every bin
  alike, so they do not enter the test. The mean and spread need two references. Every other measured bin carries the headline, and all the measured bins
  then move by one more constant so that they still combine to the headline. The references' mean is a
  test, never subtracted.
- iii. **The cast to the other axes.** Each record carries its power bin's uplift, and a bin of any other
  axis reports the counterfactual-energy-weighted mean over its records. A cast bin has no minimum, but
  is flagged thin below 24 hours. The axes, all upgrade-invariant:
  - **free-stream wind speed**, in 1 m/s bins: the mean wind speed of the power references in each
    record, times one factor fitted on the before period so that the subject turbine's electrical Cp,
    over the before-period records between 40 % and 60 % of rated power, meets a target, 0.46 by
    default with air density 1.225 kg/m³ (the analyst may declare either). The same factor applies to
    both periods, so the after period's labels cannot absorb the uplift. Each reference's wind speed is
    compared month by month with the others' mean, and a reference whose ratio moves by more than 3 %
    is reported for the analyst; it is not corrected;
  - **direction**, in 30° sectors: the direction of step 5 ii;
  - **solar elevation**, as a stability proxy: night (below −10°), dawn and dusk (−10° to 10°) and day;
  - **wake exposure**: waked when a turbine is upwind of the subject turbine (step 5 iv), front row
    otherwise.
- iv. **The subsets check.** The cast assumes the uplift depends only on the baseline power. For
  direction, solar elevation and wake exposure, the power shape is measured again within each sector,
  band or side, in six power bins (five or four where six cannot each reach 24 hours; a subset that
  cannot fill four is reported too thin). Neither the subset's shape nor the whole period's on the same
  bins is shifted, and where any bin differs by more than two sigma of the difference the subset is
  flagged: the variable then has an effect beyond restating the power dependence.

Each bin is reported with its uncertainty (step 14). The subject turbine's own signals are never an
axis, and neither is ERA5, which may be missing for the last months of a live campaign. An upgrade
whose effect varies with wind speed above rated cannot be told apart by the predicted-power axis, and
measuring one needs a measured wind speed, not neighbouring turbines.

This step is run for the test turbine (step 10). It is off for the candidates of the screen (step 9),
where only the headline uplift is needed.

## Part D: across turbines

### 16. Combine turbines

Turbine results are combined where the turbines have the same role or upgrade. Test turbines that have
received the same upgrade are combined into one uplift, and all reference turbines are combined into one
combined reference uplift, which is expected to be 0.0 %. In both cases the combination is
energy-weighted rather than a mean of the uplifts:

$$
U_{combined} = \frac{\sum_t \sum_A y_t}{\sum_t \sum_A y_t / (1 + U_t)} - 1
$$

The combined reference uplift is reported alongside every result as the campaign's own check of bias,
and the spread of the individual reference uplifts (step 9), the largest minus the smallest, as its
own measure of noise. The combined uplift is reported both before and after the screen: the value
after the screen is near zero partly by construction, so the value before it is the more honest
check.

The reference uplifts are used to correct the test turbine uplift (step 13 option C). The combined
reference uplift reported here is of the references' own, uncorrected uplifts: it measures the bias the
correction removes, and is no longer a check that the corrected test turbine uplift is unbiased.

> **TODO (combined uplift distributions).** Combine the test turbines' uplift distributions of step 15
> into whole-site distributions, bin by bin on each axis, weighted by each turbine's counterfactual energy
> in the bin, so the whole-site results carry the uplift against free-stream wind speed a
> pre-construction model needs. How the bins' sigmas combine is open: the turbines share references,
> so their errors are partly correlated; taking them as fully correlated is the cautious choice.

### 17. Estimate the annual energy production uplift

For most upgrades the uplift against wind speed (step 15) combined with a long-term wind speed
distribution is sufficient to estimate the uplift in annual energy production (AEP). The measured
uplift distribution only covers the conditions seen during the campaign: it needs to be extended to the
long-term distribution, with the uncertainty increased for conditions that were not covered. ERA5 is
used to estimate the long-term wind speed distribution.

## Validation

The method is judged on synthetic campaigns with known ground truth, built from real SCADA data, and on
real campaigns where the reference turbines provide truth of zero. The target is:

- i. the uplift of placebo test turbines (upgraded with zero effect) is 0.0 % on average, within the
  noise of the campaigns, and so is the combined reference uplift for any option that does not correct
  by the references (step 13 option C makes them read 0 % by construction);
- ii. on synthetic upgrades of known size, a straight line fitted to measured against true uplift has a
  slope of 1 and an intercept of 0, with as little scatter as possible;
- iii. neither depends on the length of the before or after period, the number of reference turbines
  or the season of the upgrade;
- iv. the reported sigma covers the error at the rate it claims: about 68 % of the errors against
  truth, headline and per bin, fall within one sigma, and the reference uplifts read 0 % within
  their sigmas at the same rate. A sigma a little too large passes (up to 90 % within one sigma);
  one too small fails.
- v. the uplift distributions of step 15 are unbiased in every bin of every axis, beat reporting the
  headline in every bin (a lower RMSE against truth), and each bin's sigma covers its error as in iv,
  judged bin by bin rather than pooled.

The targets are expected to pass only where the method is meant to be used: at least 12 months of
before data, and at least 6 months of after data, ideally fewer. The campaign matrix also runs shorter
periods (6 months before, 1 to 3 months after) and fewer references. They are not judged against the
targets. Instead they show whether the method degrades gracefully: its error grows, but stays
unbiased, and the sigma grows with it rather than claiming a precision the data cannot give.

The options above are decided by these criteria, one at a time rather than in every combination. An
option is kept when it improves them.

Decided so far, on the prepost campaign matrix:

- **Step 6: ERA5 out of the model, solar elevation in (CF34).** With ERA5 as features the references
  read about +0.1 pp; without it, and with solar elevation, the estimator's bias fell from +0.13 to
  +0.07 pp and the placebo RMSE from 0.66 to 0.52 pp. No feature set without ERA5 reached 0.0 pp.
- **Step 11 iv: no weather trimming (CF33).** No shift between the periods predicted a reading's error,
  and every trim cost precision.
- **Step 13: option C.** Option B, with the calculated $\lambda$ or in full, failed the slope of ii and
  over-corrects short after periods. Re-scored on run 6's records, option C took the synthetic test
  turbines' mean error from +0.07 to −0.01 pp, within 0.1 pp at every combination of period lengths,
  and lowered the scatter at every one. Run 7 judged it as built, in the domain: i, ii and iii pass
  except the season of the upgrade (spring +0.35 pp, z 2.8), which that matrix could not tell apart
  from the campaign, since each campaign's works fall in one season. A season study runs every
  campaign in all four seasons to judge it within a campaign.
- **Step 15: one shape against predicted power, cast to the other axes.** On 104 local cells of the
  matrix (Kelmarsh and Penmanshiel, in the domain) the forward shape against predicted power matched
  the previous estimator (two models per direction on records matched by ERA5 weather, 0.83 against
  0.81 pp energy-weighted RMSE), and cast to ERA5 wind speed it beat it (0.65 against 0.82 pp) without
  the low-wind failures (0–2 m/s RMSE 2.5 against 17.8 pp). Correcting each bin by the references'
  readings in that bin was worse than either (1.30 and 1.12 pp). Shifting the shape to the headline
  rather than scaling it keeps placebos sound, where both are near zero.
- **Step 15 ii: the reference gate.** On the same 104 cells the shape read low near cut-in (lowest power
  bin −2.8 pp, RMSE 3.7). Subtracting the references' mean deviation in each bin fixed that bin (−0.8 pp)
  but the references' noise spread into the others (RMSE 1.46 to 1.33 pp over the power axis). The
  gate, then one constant to restore the headline, did best: RMSE 1.16 pp (flat headline 1.81), lowest
  bin −1.0 pp, free-stream 3–4 m/s −0.9 pp (from −2.4), and 80 % of bin errors within one sigma once
  the references' spread joined it (63 % without). Re-shifting only the kept bins was worse (1.31 pp),
  and setting the others to the headline without restoring it left the total up to 1 pp off. Leaving
  the headline's level sigma out of the gate's sigma kept more bins (31 % against 25 %) and did better
  again: RMSE 1.10 pp, free-stream above 12 m/s 0.4–0.9 pp against 1.1–1.2, 82 % covered; on the real
  T13 and TuneUp campaigns the gate with it had kept 0–3 of 11 bins and flattened clear declining shapes.
- **Step 14: $\rho = 0.5$.** Re-scored on run 7, $\rho = 0$ (plain quadrature) covered 82 % of the errors
  within one sigma, $\rho = 0.5$ 76 % (95 % within two sigma, as a normal error would) and $\rho = 1$
  64 %.

Still to try:

- step 6 i option B (pitch angle and rotor speed as features);
- step 11 ii with a minimum below 24 hours, so that more configurations with a turbine offline upwind
  are kept for the model to learn from rather than dropped as rare.

### Where the method is valid

Once the targets pass, this section states the situations the method is validated for, from the
campaign matrix: the shortest before and after periods, the fewest reference turbines, the sites
and kinds of upgrade tested, and how the accuracy and the sigma behave outside them. Until then the
intended domain is at least 12 months before and at least 6 months after.

| Condition | Validated for | Outside it |
| --- | --- | --- |
| Before period | to be decided, intended 12 months or more | to be decided |
| After period | to be decided, intended 6 months or more | to be decided |
| Reference turbines kept | to be decided | to be decided |
| Upgrade size | to be decided | to be decided |

## Open questions

- How the ERA5 stability proxy and weather cells of step 11 are defined, and the minimum hours per cell.
- Whether the waking configuration rule of step 11 ii causes bias. On run 7 of the campaign matrix the
  third of test turbines (all cells) where it removes the largest share of after-period hours read
  −0.16 pp (z −3.1), the rest about zero; whether a lower minimum or a per-turbine rule fixes it is
  untested.
- Whether the free-stream scale factor of step 15 should depend on direction (a model of the subject
  turbine's wind relative to its references', like the model of step 12).
- How site-specific operating states (step 2) reach wind-up: pre-labelled data or a labelling schema.
- How many references to use by default (step 8), given the union of invalid data in step 11 i.
- The data screen's thresholds (step 9): the usable-hours fraction and the correlation margin below
  which a candidate is dropped.

## References

[1] Wind energy generation systems - Part 12-1: Power performance measurements of electricity producing
wind turbines (IEC 61400-12-1:2022). International Electrotechnical Commission, September 2022.
