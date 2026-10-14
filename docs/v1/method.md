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
- vii. **missing**: the signals are absent or out of range, or stuck (unchanged from the turbine's
  own previous record while the wind is not calm), which indicates frozen telemetry. The calm test
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
preparation, and every later step uses the same aligned ERA5.

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
are a candidate cause of performance change (step 1). A north change of 30° or more that returns within
90 days is a north calibration excursion; one that does not return is a re-calibration and becomes the
new reference frame. Every other change, between two segments in the reference frame, is a candidate yaw
alignment change. For each candidate, the change in the turbine's apparent power coefficient (its power
divided by the cube of its own nacelle wind speed, binned by wind speed) is measured between the 60 days
before and the 60 days after, using the records valid for northing (above 5% of rated power, as for the
north tables) and for uplift. It is compared with the largest such change at any other date of the
turbine's record that is more than 60 days from every candidate. The report is for the analyst to inspect
alongside the declared changes; it does not change the uplift calculation.

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
  positions and rotor diameters. A waking scenario column is recorded per turbine (for example
  `T01_waking_scenario`). Its value is:
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
- iv. **Upwind turbines.** A second column per turbine (for example `T05_upwind`) lists every turbine
  within its disturbed sector for the direction of ii, whatever its waking state, in a fixed order. Its
  value is "none" when no turbine is upwind and "unknown" when the direction of ii is unavailable. Which
  of these turbines count as changed depends on the before and after periods of the test turbine, so
  the records with a changed turbine upwind are found in step 11 rather than here.

### 6. Engineer features

The features from which the test turbine's power is predicted are computed for every record. They are
computed for every unchanged turbine, so that any of them can serve as a power reference without
repeating this step. The model of step 12 uses those of the reference set it is given.

- i. **Power references:** for every unchanged turbine, active power and the north-calibrated nacelle
  direction (step 4) as sine and cosine. The direction is a feature in both options below: the test
  turbine's wake exposure within the "normal" scenario varies with direction, and ERA5 direction is
  too coarse to resolve it.
  - **Option A (current default).** No further signals. ERA5 resolves the very low and very high wind
    speeds where reference power carries little information.
  - **Option B.** Pitch angle and rotor speed are added, which allows high wind speed operation to be
    resolved.
- ii. **Waking scenarios:** the waking scenario column (step 5 iii) of the test turbine and of each
  power reference, as a categorical feature. They carry the waking state of every other turbine on
  the wind farm, which contributes no feature of its own. After the matching of step 11 most values
  are "normal", so these features only act where a non-normal scenario was retained in both periods.
- iii. **ERA5:** a short list of variables with a clear physical reason to affect the test turbine's
  power relative to its references: 100 m wind speed, 100 m wind direction (as sine and cosine), air
  density derived from temperature, pressure and humidity, a shear exponent derived from the 10 m and
  100 m wind speeds, and a turbulence proxy derived from the gust and mean 10 m wind speeds.
- iv. **Optional: atmospheric stability.** Possible candidates are the solar position or the
  irradiation from ERA5.

Adding more information about the references has not reduced bias to date. Features are therefore kept
to those with a clear physical justification.

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
the trade-off is made is described in [analysis-period-and-reference-selection.md](../analysis-period-and-reference-selection.md).

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
test turbine's. The nearest few (four by default) are the power references the screen starts from; the
rest, in order, are reserves, from which the screen draws a replacement for each reference it rules
out.

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
reserve (step 8), which is screened in its turn. The order is always by distance: the screen only
removes and replaces, it never re-ranks.

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
  step 8. A measurement needs a minimum of usable data in each period (24 hours by default); below
  it no uplift is returned, and a test turbine is reported as not analysed with the reason, as in
  step 7. Quantities here are hours, not records.
- ii. **Waking configuration.** For each record, the configuration is the combination of the waking
  scenarios (step 5 iii) of the subject turbine and all of its power references, taken together. Only
  configurations with at least 24 hours of data in both periods are retained, in the same way that
  wind-up v0 retained only the waking scenarios with at least one day of data. By default a record is
  also dropped when the scenario of the subject turbine or of any power reference contains a part
  waking turbine or is "unknown", however much data its configuration has. This is the per-turbine
  part of the rule, which the data screen (step 9) applies too; the records of part iii without a
  direction are among those it drops. A configuration seen in one period only is one the model has
  rarely or never seen, and its prediction for it is an extrapolation. For a toggle test the two
  periods are the toggle on and toggle off data, and a record from before toggling began (step 7) is
  dropped when its configuration is rare in them. The hours each part removes are reported by the
  measurement, in hours per period.
- iii. **Changed turbines upwind.** For this measurement, a turbine is changed if it has a change
  (step 1) between the start of the before period and the end of the after period.
  A turbine whose changes all fall outside that span casts the same wake in both periods. Records in
  which a turbine is upwind of the subject turbine or of a reference (step 5 iv) at a time it counts as
  changed, or in which either upwind column is "unknown", are dropped. When it counts as changed
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

  Each works window of a turbine is treated this way and the dropped times add up. The analyst can
  override the rule where the wake is known not to change.
- iv. **Weather.** Both periods should see the same weather. For a toggle test a toggle pairing filter
  is applied, which only keeps data where the time difference between a data record and the nearest
  valid record with the opposite toggle state is within a limit, one toggle cycle (the on and off
  durations together) by default. For a prepost test one of the following is used.
  - **Option A (current default).** No weather matching beyond the choice of period in step 7.
  - **Option B.** Common-support trimming: coarsen ERA5 wind speed, direction and a stability proxy
    into cells, and drop the records of either period that fall in cells with less than a minimum
    number of hours in the other period. Calendar time is not a condition: matching the months of
    the two periods would shorten the before period, which step 7 keeps long on purpose.

Under option B, trimming is kept as natural as possible: where practical, whole days are removed rather
than individual records, so that the retained data stays contiguous and autocorrelated in the same way
in both periods.

Features that reveal the period rather than the conditions, such as a signal that only exists in one
period, must not be used for matching. Weighting the records by a model of the probability of belonging
to each period is not used, because on real data the period is almost perfectly predictable from such
features.

### 12. Relate the reference measurements to the subject turbine measurements

A machine learning model is trained which predicts subject turbine power from the features of step 6
for the reference set. This takes the place of the directional detrending of wind-up v0: rather than a
ratio of wind speeds by direction bin, the model learns how the subject turbine's power relates to its
references under every combination of weather and waking state present in the data. Only the records
retained by step 11 are used.

The model is a gradient-boosted tree regressor minimising squared error. It is configured so that
repeated runs on one machine give identical results; results on different machines can differ by a
few tenths of a percentage point.

**Joint or paired references.** By default the references are analysed together as one model for each
subject turbine, not one subject-reference pair at a time. A single-reference estimate carries
roughly twice the shared bias of a multi-reference one, and each reference's own drift relative to
the subject turbine is averaged down as more references are added.

- **Option A (current default).** One joint model per subject turbine.
- **Option B (not pursued in this revision).** Steps 11 to 13 are run once per subject-reference
  pair and the pair results are combined. This keeps more data, because the validity of step 11 i
  is not unioned across references, and lets a reference be dropped after the fact without
  re-running. It is not tested now: the data cost of the joint model is already measured and
  reported by the data screen (step 9), the number of references is the lever that trades it off,
  and a paired method would need its own screen, its own step 13 and its own step 16. It is
  revisited only if validation shows the data lost in step 11 i is what limits accuracy.

### 13. Calculate the uplift

The model of step 12 trained on the before period, $\hat f_B$, predicts the subject turbine's power in
the after period. Writing $y$ for measured subject turbine power and $A$ and $B$ for the after and
before records, the forward uplift is

$$
U_{fwd} = \frac{\sum_A y}{\sum_A \hat f_B} - 1
$$

Each prediction is clipped to the range from the smaller of zero and the lowest power the model was
trained on to the larger of the rated power and the largest power the model was trained on. This compares
a measured sum with a predicted one, and is biased by any error the model makes on average when
predicting data unlike its training data. That error grows with the time separating the two periods and
is the main source of bias left once steps 5, 6, 7 and 11 are applied. One of the following is used to
remove it:

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
  to minimise the expected error. The precision of each direction is estimated within this step, from
  the spread of its uplift over time-blocked folds of its training data or over a block bootstrap of
  it; it does not depend on step 14. Option A is the $\lambda = 0$ limit of option B, so the two are
  not compared as alternatives: validation checks that $\lambda$ falls towards 0 on short after
  periods and that the reference turbines read 0 % at every campaign length.

The reverse calculation of option B does not use the subject turbine anemometer and so can be performed
in full, unlike wind-up v0, which had to fall back on power-only versions of both calculations.

### 14. Estimate uncertainty

An uncertainty is reported for the headline uplift of step 13 and for each measured bin of the
uplift distributions of step 15. It has two components, reported apart: a sampling component, and a
systematic component for the error that remains when the reference turbines or the model are
biased. Only the sampling component is calculated so far.

The sampling component is a circular block bootstrap with the model of step 12 fixed. The after
period's records of the subject turbine, each with its measured and its counterfactual power, are
resampled in blocks of wall-clock time (one day by default) tiled over the after period, the ratio
of sums of step 13 is re-formed for each resample, and the sigma is the standard deviation of the
resampled uplifts. Each bin of step 15 is treated the same way over the records matched for it,
with the two direction models fixed and the bin labels frozen, so the uncertainty never moves an
uplift; a bin that was filled rather than measured has no sigma. The sigma is also recorded at
other block lengths (6 hours, 1 day, 7 days and 28 days by default) so that validation can choose
the block length by the coverage of the sigma against known truth. Every reference turbine
measured in step 9 carries the same sigma, so its reading against the expected 0 % can be judged
in sigmas.

The systematic component is not yet specified. The candidates are the reversal disagreement of
step 13 option B and the spread of the reference uplifts of step 16, as in wind-up v0, which took
the largest of its statistical, bootstrap and reversibility sigmas and floored it at the weighted
mean absolute reference uplift. How sigmas combine across references and across test turbines is
left until the components are trusted.

### 15. Calculate uplift distributions

Uplift is calculated as a function of condition variables specified by the analyst. Uplift against
wind speed is generally the most important for pre-construction inputs, but it has to be based on a
stable, free-stream wind speed estimate derived from the reference turbines (power and pitch angle), not
from the subject turbine. For certain upgrades other conditions are important, for example wake steering
also requires uplift against direction and atmospheric stability.

The binned uplift is not the ratio of the headline's per-record sums within each bin. The model of
step 12 has a conditional bias that integrates to about zero over the whole window but not within
a bin: it over-predicts low power and under-predicts high power, so a plain per-bin ratio shows a
spurious slope even where the true uplift is flat. The bin shape is therefore estimated with the
full reversal of step 13 option B applied per bin, on records matched between the periods as in
step 11 iv option B: the before and after records are matched on coarsened ERA5 cells (100 m wind
speed in 2 m/s steps, 10 m gust in 3 m/s steps and 100 m direction in 20° sectors by default), cells
present in one period only are dropped, and in every other cell the larger side is subsampled to the
size of the smaller so the two sides carry equal weight per cell; a model is fitted in each direction
on the matched records, and each bin's uplift is the geometric combination of the forward and reverse
ratios, which cancels the conditional bias common to both. A bin with too few matched hours on either
side (8 hours by default) is not measured; its value is filled from the nearest measured bin above it,
and the bins above the last measured bin take 0 %, since at and above rated the baseline and the
upgraded turbine produce the same power. The measured bins are then scaled by one factor, with the
filled bins held as filled, so that, weighted by the after-period energy in each bin, they combine to
the headline uplift of step 13; the headline itself is never moved by this step. Each bin is reported
with its uncertainty (step 14).

Records are assigned to a bin by an upgrade-invariant variable, never by any signal of the subject
turbine: not its measured power, which would select each bin on its own noise, and not its nacelle
wind speed or turbulence. By default the uplift is reported against the counterfactual power
prediction (as a fraction of rated, in six bins) and against the ERA5 100 m wind speed (in 2 m/s
bins). The other variables available are the ERA5 variables of step 6 iii and a free-stream wind
speed derived from the references once one is defined (open question).

This step is run for the test turbine (step 10). It is off by default for the candidates of the screen
(step 9), where only the headline uplift is needed.

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

The reference uplifts could also be used to correct the test turbine uplift. This is not done, because
the references would then read 0 % by construction and could no longer be used as a check of bias.

### 17. Estimate the annual energy production uplift

For most upgrades the uplift against wind speed (step 15) combined with a long-term wind speed
distribution is sufficient to estimate the uplift in annual energy production (AEP). The measured
uplift distribution only covers the conditions seen during the campaign: it needs to be extended to the
long-term distribution, with the uncertainty increased for conditions that were not covered. ERA5 is
used to estimate the long-term wind speed distribution.

## Validation

The method is judged on synthetic campaigns with known ground truth, built from real SCADA data, and on
real campaigns where the reference turbines provide truth of zero. The target is:

- i. the combined reference uplift and the uplift of placebo test turbines (upgraded with zero effect)
  are 0.0 % on average, within the noise of the campaigns;
- ii. on synthetic upgrades of known size, a straight line fitted to measured against true uplift has a
  slope of 1 and an intercept of 0, with as little scatter as possible;
- iii. neither depends on the length of the after period, the number of reference turbines or the
  season of the upgrade;
- iv. the reported sigma covers the error at the rate it claims: about 68 % of the errors against
  truth, headline and per bin, fall within one sigma, and the reference uplifts read 0 % within
  their sigmas at the same rate.

The options above are decided by these criteria, one at a time rather than in every combination:

1. the current defaults, with step 11 ii and step 15 as written above, give the baseline;
2. step 13 option B with the calculated $\lambda$ is added (option A is its $\lambda = 0$ limit);
3. step 11 iv option B is added on top of that result, since trimming and the reversal correction
   both act on the model's extrapolation between the periods;
4. step 6 i option B is added last.

An option is kept when it improves the criteria above. Step 12 option B is not run.

## Open questions

- How the ERA5 stability proxy and weather cells of step 11 are defined, and the minimum hours per cell.
- Whether a mismatch in waking configurations between the periods (step 11 ii) is a measurable source of
  bias now that offline references are excluded by step 11 i, and so how much the 24 hour rule matters.
- How the free-stream wind speed estimate of step 15 is constructed.
- Whether step 15 should report which condition variables the uplift is sensitive to, and how.
- How site-specific operating states (step 2) reach wind-up: pre-labelled data or a labelling schema.
- How many references to use by default (step 8), given the union of invalid data in step 11 i.
- The data screen's thresholds (step 9): the usable-hours fraction and the correlation margin below
  which a candidate is dropped.

## References

[1] Wind energy generation systems - Part 12-1: Power performance measurements of electricity producing
wind turbines (IEC 61400-12-1:2022). International Electrotechnical Commission, September 2022.
