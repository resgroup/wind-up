# The facts API

This page is the reference for the objects of wind-up's Python interface: `WindFarmFacts`,
`Upgrade`, `Label`, `AnalysisSpec` and what a method returns. For a worked example on open data see
[`examples/smarteole_example.ipynb`](../examples/smarteole_example.ipynb). For the other way to run
wind-up, one YAML file from the command line, see [running a campaign](running-a-campaign.md).

An analysis is described in three layers:

1. **Facts** (`WindFarmFacts`) say what is true of the wind farm and what was done to it: the
   records, what their columns mean, the turbines, the upgrades and the analyst's declarations.
   Facts do not depend on the question asked or on the method.
2. **An analysis** (`AnalysisSpec`) is one question asked of the facts: the uplift of one upgrade
   on a group of turbines, from a choice of records.
3. **A method** answers analyses: `method.estimate(facts, analyses)` returns one `AnalysisOutput`
   per analysis, in order. Every method takes the same facts and analyses, and has its own options.

```python
from wind_up import AnalysisSpec, Label, Upgrade, WindFarmFacts, combine, combine_subjects
from wind_up import NonTurbineMeasurement, Reanalysis
from wind_up.schema import ColumnSchema
from wind_up.toggle import ToggleSchedule
from wind_up.operating_state import OperatingStateConfig, StateValidity
from wind_up.methods import ToggleEnergyRatioMethod
```

The declarations (`WindFarmFacts`, `Upgrade`, `Label`, `AnalysisSpec`) are frozen dataclasses that
check themselves when they are built, so a declaration that contradicts itself fails at once with a
message naming the upgrade, analysis or label. To change one field of an existing declaration use
`dataclasses.replace`. `ColumnSchema` is frozen too but is checked later: the facts require its
turbine and active power roles, and each method the roles it reads. `AnalysisOutput` is an ordinary
dataclass.

Contents: [WindFarmFacts](#windfarmfacts) · [ColumnSchema](#columnschema) · [Upgrade](#upgrade) ·
[Label](#label) · [Record labels](#record-labels) ·
[Leaving records out](#leaving-records-out) · [AnalysisSpec](#analysisspec) ·
[Methods and their output](#methods-and-their-output) · [Combining](#combining) ·
[Warnings](#warnings) · [A minimal run](#a-minimal-run)

## WindFarmFacts

| Field | Type | Meaning |
|---|---|---|
| `records` | `DataFrame` on a time index | Long-format SCADA as measured: one row per turbine and timestamp, every year supplied, nothing dropped |
| `columns` | `ColumnSchema` | Which column of `records` holds each signal ([below](#columnschema)) |
| `timebase` | `Timedelta` | The records' period, for example 10 minutes |
| `turbines` | `DataFrame` | One row per turbine: `name`, `latitude`, `longitude`, `rated_power_kw`; `rotor_diameter_m` and `hub_height_m` when a method needs them |
| `upgrades` | sequence of `Upgrade` | At least one, each with its own name |
| `operating_states` | `OperatingStateConfig` | The site's own operating states; wind-up's generic rules alone when omitted |
| `exclusions` | `(turbine or None, start, end)` | The data is wrong over `[start, end)`: its records read as missing |
| `operating_state_overrides` | `(turbine or None, start, end, state)` | The data is right and the operating state over `[start, end)` was `state` |
| `labels` | sequence of `Label` | The analyst's own labels ([below](#label)) |
| `north_offsets` | `(turbine, from, offset in degrees)` | The declared northing: the offset to add to the turbine's nacelle position from that time on |
| `non_turbine_measurements` | sequence of `NonTurbineMeasurement` | Masts and LiDARs: `name`, `latitude`, `longitude`, `records` on a time index, and `columns` mapping a role to a column |
| `reanalysis` | sequence of `Reanalysis` | Reanalysis nodes: `name`, `latitude`, `longitude`, `records` on a time index |

In a tuple, a turbine of `None` means every turbine.

**The records.** Only the turbine and active power columns are required. Building the facts sets a
value outside its signal's plausible range to NaN and logs a warning per column; the ranges are
`wind_up.schema.ROLE_RANGES` and `ROLE_RANGES_RATED`. The records must not hold a turbine the
turbine table lacks.

**Checks when the facts are built.** The records have a time index and every named column; the
turbine table has its required columns, each name once and a positive rated power; every upgrade,
exclusion, override and north offset names known turbines; a mast or LiDAR does not share a
turbine's name; each label column is true or false on every record.

**Reanalysis.** `wind_up.era5.get_era5_hourly_df(lat=..., lon=..., start_date=..., end_date=...)`
fetches hourly ERA5 from the Open-Meteo archive and caches it; pass the result as a `Reanalysis`
node's `records`.

**Northing.** `wind_up.northing_step.discover_north_offsets(facts)` returns north offsets found
from the records valid for northing, against the reanalysis direction and the farm's own consensus.
Declare them with `dataclasses.replace(facts, north_offsets=...)` so that every method reads the same
northing. `wind_up.northing_step.northed_records(facts)` returns the records with the declared
offsets applied, in a column named by `columns.northed("nacelle_position")`. See
[how northing works](northing.md).

Useful members:

| Member | Returns |
|---|---|
| `record_labels()` | The labels of every record ([below](#record-labels)). Computed on each call; keep the frame if you need it twice |
| `warnings()` | What the facts cannot support, as analysis warnings ([below](#warnings)) |
| `turbine_names` | Every turbine's name, in the turbine table's order |
| `upgrade(name)` | The upgrade of that name |
| `subjects_of(analysis)`, `references_of(analysis)` | The turbines an analysis measures and its candidate references, with the defaults resolved |
| `check(analysis)` | Raises when the analysis names an upgrade, turbine, measurement or label the facts do not hold |
| `comparison_sides(analysis, record_labels=...)` | The side of the analysis's comparison each record is on: `on`, `off` or `neither` |
| `rated_power_kw(turbine)`, `layout` | A turbine's rated power; the farm layout (needs `rotor_diameter_m`) |

## ColumnSchema

wind-up assumes no column names. `ColumnSchema` names the column of the records for each role; a
role left `None` is a signal the records lack. Each method checks the roles it reads.

| Role | Unit | Notes |
|---|---|---|
| `turbine` | | Required. The turbine identifier of the long-format records |
| `active_power` | kW | Required. Mean active power |
| `wind_speed`, `wind_speed_sd` | m/s | Nacelle wind speed, mean and standard deviation |
| `gen_rpm` | rpm | Mean generator speed |
| `availability` | s | Seconds of the record the turbine was ready to operate. Without it downtime cannot be labelled, and the facts say so in `warnings()` |
| `active_power_min` | kW | Minimum active power in the record |
| `pitch` | deg | Blade pitch angle; one blade is enough |
| `reactive_power` | kvar | Mean reactive power |
| `nacelle_position`, `nacelle_position_min`, `nacelle_position_max` | deg | Nacelle position, cumulative or not; northing calibrates it to a wind direction |
| `ambient_temp` | degC | Ambient temperature |
| `power_setpoint` | kW | The power setpoint at the end of each record |

## Upgrade

What was done to a group of turbines, and when.

| Field | Meaning |
|---|---|
| `name` | The upgrade's name; analyses refer to it |
| `turbines` | The turbines that received it, each named once |
| `timing` | When it was on; the forms are below |
| `settling_time` | Toggle only. A record of the upgrade's turbines that starts within this long after a switch between on and off keeps its state, and is not valid for uplift or northing |
| `installation_windows` | Per turbine, the windows `[start, end)` when the upgrade was being fitted. Their records are in neither state |
| `upgrade_state_overrides` | `(turbine or None, start, end, state)`: the upgrade state over `[start, end)` where the timing does not describe it; `state` is `on`, `off` or `neither` |
| `toggle_cycle`, `toggle_datum` | Toggle only: the controller's cycle and a boundary of its grid ([below](#the-toggle-cycle-and-its-datum)) |
| `expected_uplift` or `wake_reach_d` | The uplift expected, as a fraction, from which the reach of the upgrade's wake change follows, or that reach in rotor diameters. At most one of the two |

### The forms of `timing`

| `timing` | Mode | Meaning |
|---|---|---|
| a `Timestamp` | prepost | Every turbine is off before the date and on from it |
| a mapping of turbine to `Timestamp` | prepost | A date per turbine; every turbine of the upgrade needs one |
| `None` | prepost | Every turbine has installation windows: it is off before its first, in neither state until the end of its last, and on from then |
| a `ToggleSchedule` | toggle | A regular toggle shared by the turbines: `period` is one full cycle, half off then half on (`start_on=True` for on first), from `start` |
| a `DataFrame` | toggle | A toggle frame shared by the turbines: boolean `toggle_on` and `toggle_off` columns on a time index, one row per timestamp |
| a mapping of turbine to `DataFrame` | toggle | A toggle frame per turbine, for a toggle that runs on its own schedule on each turbine. Every turbine of the upgrade needs a frame |

`upgrade.mode` is `"prepost"` or `"toggle"`, and `upgrade.per_turbine` says whether a toggle has a
frame per turbine. A mapping is all dates or all frames.

A prepost date inside one of the turbine's installation windows widens the changeover to that
window.

### Upgrade states

`upgrade.states(index, turbine=...)` gives a turbine's state at each timestamp, by these rules in
order:

1. A turbine that did not receive the upgrade is off.
2. A permanent upgrade is off before its date and on from it.
3. A toggle is off before its signal starts and then follows the signal. A record of a toggle frame
   that is neither `toggle_on` nor `toggle_off`, or that the frame lacks, is in neither state.
4. Installation windows are in neither state.
5. The overrides are applied last.

### The toggle cycle and its datum

Methods that work in blocks of whole cycles read `upgrade.cycle` and `upgrade.datum(...)`.

- **`toggle_cycle`** is one on and off cycle of the controller, an equal time in each state. A
  declared cycle is used as given. For a schedule it must equal the schedule's period. When
  omitted, a schedule's period is used, or for frames the median gap between the starts of their
  on runs, pooled over the frames.
- **`toggle_datum`** is any boundary of the controller's cycle grid. When omitted it is a
  schedule's start, or for frames the Unix epoch in the frame's time zone.

A frame is checked against the cycle and datum, not rejected by them: when more than 1 % of a
frame's switches between on and off fall away from the datum plus a whole number of half cycles,
`upgrade.warnings()` gives a medium warning against the frame's turbine (`"all"` for a shared
frame). The answer is to declare `toggle_cycle` and `toggle_datum`.

## Label

An analyst's label is a named true or false on every record.

| Field | Meaning |
|---|---|
| `name` | The label's name, as analyses and reports refer to it |
| `column` | The boolean column of the records holding it: False where unknown, never NaN |
| `withholds` | The uses that the label's records are withheld from: `"uplift"`, `"northing"`, an iterable of both, or `"all"`. By default no use is withheld. Stored as a frozen set |

A label on its own changes nothing; it is a column the analyses can select by. It acts in two
ways:

- **`withholds`** makes the label's records not valid for a use, on every turbine that holds the
  label, references included, in every analysis. The record keeps its operating state and its
  waking state, so the turbine still counts in its neighbours' wake exposure.
- **An analysis's `include_labels` and `exclude_labels`** select the records of that analysis's
  subjects only ([below](#analysisspec)).

A label that is true for the whole farm at a timestamp, such as a wind direction sector, repeats
its value on every turbine's record of that timestamp.
`wind_up.labels.direction_sector(direction, sector=(from, to))` makes such a label from a wind
direction series. The direction must not come from a turbine whose records the label will select
as a subject.

## Record labels

`facts.record_labels()` returns one row per turbine and timestamp of the full grid, ordered by
timestamp then turbine. A record never supplied reads as missing.

| Column | Meaning |
|---|---|
| the turbine column | The turbine's name. The column has the same name as in the records (`columns.turbine`) |
| `operating_state` | `missing`, `full downtime`, `partial downtime`, `normal operation`, or one of the site's own states |
| `valid_northing`, `valid_uplift` | Whether the record is valid for each use |
| `waking_state` | `waking`, `part waking`, `not waking` or `missing` |
| `upgrade_state: <upgrade name>` | `on`, `off` or `neither`, one column per upgrade (`wind_up.labels.upgrade_state_column`) |
| each label's name | The label, true or false |
| `not_valid_uplift_because`, `not_valid_northing_because` | Why a record is not valid for the use: its operating state and the labels that withhold it, joined by `; `. Empty where it is valid |

A record is valid for a use when its operating state allows the use, no label on it withholds the
use, and it is not settling after a toggle switch.

**Operating states.** The generic rules label a record missing, full downtime (no availability),
partial downtime (less than full availability) or normal operation. `OperatingStateConfig` adds a
site's own states from a column of the records: `site_state_column` names it and `site_states`
maps each state to a `StateValidity(northing=..., waking=..., uplift=...)`.
`parked_pitch_above_deg` or `parked_pitch_below_deg` declares the pitch of a parked rotor.

## Leaving records out

There are five ways, and they differ in what they say about the data.

| Declaration | Says | Effect |
|---|---|---|
| `WindFarmFacts.exclusions` | The data is wrong | The records read as missing: not valid for any use, and their waking state is missing too |
| `WindFarmFacts.operating_state_overrides` | The data is right and the turbine was in this state | The records take the state and its validity. A missing record stays missing |
| `Label.withholds` | These records are real but not fit for this use | Not valid for the use on every turbine holding the label, in every analysis; the waking state is kept |
| `AnalysisSpec.include_labels`, `exclude_labels` | This question is about these records | Selects the subjects' records of that one analysis; references keep theirs |
| `AnalysisSpec.period` | This question is about this time | No record outside `[start, end)` is compared; earlier records stay available as history |

`Upgrade.installation_windows` and `Upgrade.upgrade_state_overrides` do not leave records out by
themselves. They put records in neither upgrade state, so that they are on neither side of a
comparison.

## AnalysisSpec

One question asked of the facts.

| Field | Default | Meaning |
|---|---|---|
| `name` | | The analysis's name, for output folders and warnings |
| `upgrade` | | The name of the upgrade measured |
| `subjects` | every turbine that received the upgrade | The turbines measured, together |
| `references` | every turbine unchanged over the period | The candidate references: turbines, and masts or LiDARs by name |
| `period` | `(None, None)` | `(start, end)`, either side `None` for open |
| `include_labels` | none | Only a subject's records holding all of these labels are compared |
| `exclude_labels` | none | No subject's record holding any of these labels is compared |
| `report` | both | The uplifts wanted: `"aep"`, `"campaign"` or both |
| `combination` | `"inverse_variance"` | How a group of subjects' uplifts combine into the analysis's |
| `placebo_timing` | none | Toggle only: a toggle that a reference turbine never applied, to measure it as a placebo |

**Subjects and references.** A subject must have received the analysis's upgrade, and no turbine
is both subject and reference. A default reference is a turbine that holds one constant on or off
state under every upgrade over the period, so a turbine upgraded before the period starts can
serve. A method may screen the candidates further.

**The two uplifts.** The AEP uplift is the uplift in annual energy. The campaign uplift is the
uplift over the records compared. A campaign much shorter than a year, or records selected by a
label, usually support only the campaign uplift; a method that cannot give a figure returns NaN
and a warning that says why.

**Combination.** `"inverse_variance"` weights each subject's uplift by the inverse of its
variance. It suits one upgrade on turbines that do not interact. `"net"` is the uplift of the
subjects' energy together. It suits turbines that interact, such as a wake-steering pair, where one
turbine gives up energy for another to gain more.

**Placebo toggles.** `placebo_timing` maps a turbine to a `ToggleSchedule` or toggle frame it
never applied. The turbine must be one of the analysis's turbine references and must not have
received the upgrade. It is measured as a subject against that timing, with the analysis's other
references as its references, and it stays a reference of the real subjects. Its row of
`per_turbine` is marked `placebo` and is left out of the analysis's uplift, of `actual_energy`
and of `combine_subjects`. A placebo that reads far from zero, against its sigma, questions the
analysis.

## Methods and their output

A method is anything with a `name` and `estimate(facts, analyses) -> list[AnalysisOutput]`
(`wind_up.method.Method`). A configuration or facts a method refuses for every subject raises
`wind_up.MethodConfigurationError`. One subject that cannot be measured is left unmeasured, with a
warning, and the others are answered.

| Method | Import | Upgrades | Status |
|---|---|---|---|
| Toggle energy ratio | `wind_up.methods.ToggleEnergyRatioMethod` | toggle | In the installed package |
| Binned power curve (the wind-up 0.x method) | `benchmarking.baselines.v0_binned.V0BinnedMethod` | prepost and toggle | In the repository only; moves to `wind_up.methods` before 1.0 |
| Power model (the v1 method, [described here](v1/method.md)) | `benchmarking.baselines.power_model.PowerModelMethod` | prepost and toggle | In the repository only; moves to `wind_up.methods` before 1.0. The [YAML campaign run](running-a-campaign.md) uses it |

Not every method reads every part of an analysis. The toggle energy ratio and the binned power
curve read `include_labels`, `exclude_labels` and `placebo_timing`; the power model raises when an
analysis gives any of them. Nothing imported from `benchmarking` is part of the product: those
imports change when the methods move.

`AnalysisOutput`:

| Field | Meaning |
|---|---|
| `analysis` | The analysis answered |
| `aep_uplift`, `aep_sigma` | The subjects' uplift in annual energy, together, as a fraction, and its 1-sigma |
| `campaign_uplift`, `campaign_sigma` | The subjects' uplift over the records compared, and its 1-sigma |
| `per_turbine` | One row per subject, then per placebo turbine: `turbine`, the four figures, `actual_energy`, `hours_on`, `hours_off`, `placebo`. A turbine that could not be measured is a row of NaN |
| `per_reference` | One row per subject and reference, from a method that works pairwise |
| `by_condition` | Per-condition estimates: `condition`, `condition_bin`, `p50_uplift` and optionally `sigma_uplift` |
| `labeled_rows` | Every subject's records with the method's selection |
| `reference_uplifts`, `screen_passes`, `selection_accounting`, `uncertainty_diagnostics` | A method's own diagnostics; `None` from a method that has none |
| `warnings` | The analysis warnings the estimate raised |
| `actual_energy` | The energy the subjects produced with the upgrade on |

A figure is NaN when it was not asked for or the method cannot give it; `warnings` says which.

## Combining

`wind_up.combine(outputs)` combines several analyses' outputs, and
`wind_up.combine_subjects(output)` the subjects of one output. Both take
`combination="net"` (the default) or `"inverse_variance"` and return a `CombinedUplift`.

- `net` weights by energy: the total actual energy over the total energy without the upgrade,
  less one. Each output or subject needs an `actual_energy`.
- `inverse_variance` weights each uplift by the inverse of its variance. Each needs a positive
  sigma.

Each sigma is given twice, as `aep_sigma_independent` and `aep_sigma_correlated` (and the same
for `campaign`): with the errors of the parts independent, and with them fully correlated, as when
they share a reference. For an `inverse_variance` combination `aep_sigma` and `campaign_sigma` give
the mean of the two; for `net` they are NaN and the two ends are the range. `outputs` is the table
of what was combined, and `warnings` holds the parts' warnings and the combination's own.

## Warnings

Facts, methods and combinations report through `AnalysisWarning(severity, step, turbine,
message)`. High questions the headline, medium changed the inputs of a measurement without
questioning its result, and low is for the analyst's attention only. `step` is the step of
[the method](v1/method.md) that raised it and `turbine` is `"all"` when it is against the farm.

`facts.warnings()` holds the facts' own, such as records without an availability counter or a
toggle frame that does not fit its cycle. `output.warnings` holds those of one estimate.
`wind_up.analysis_warnings.ranked` sorts warnings by severity and `warnings_frame` tabulates them.

## A minimal run

A toggle on one turbine of three, on made-up records, to show the shape of a run. The records are
long-format SCADA on a time index with `turbine`, `power_kw` and `ready_s` columns.

```python
import numpy as np
import pandas as pd

from wind_up import AnalysisSpec, Upgrade, WindFarmFacts
from wind_up.methods import ToggleEnergyRatioMethod
from wind_up.schema import ColumnSchema
from wind_up.toggle import ToggleSchedule

timebase = pd.Timedelta(minutes=10)
index = pd.date_range("2024-01-01", "2024-03-01", freq=timebase, tz="UTC", inclusive="left", name="timestamp")
rng = np.random.default_rng(0)
farm_power = 600 + 300 * np.sin(np.arange(len(index)) / 500) + rng.normal(0, 30, len(index))
records = pd.concat(
    [
        pd.DataFrame(
            {"turbine": name, "power_kw": farm_power + rng.normal(0, 20, len(index)), "ready_s": 600.0}, index=index
        )
        for name in ["T1", "T2", "T3"]
    ]
)
turbines = pd.DataFrame(
    {
        "name": ["T1", "T2", "T3"],
        "latitude": [55.0, 55.0, 55.0],
        "longitude": [-3.0, -2.995, -2.99],
        "rated_power_kw": 2000.0,
    }
)
upgrade = Upgrade(
    name="controller",
    turbines=["T1"],
    timing=ToggleSchedule(period=pd.Timedelta(hours=2), start=pd.Timestamp("2024-01-08", tz="UTC")),
    settling_time=timebase,
)
facts = WindFarmFacts(
    records=records,
    columns=ColumnSchema(turbine="turbine", active_power="power_kw", availability="ready_s"),
    timebase=timebase,
    turbines=turbines,
    upgrades=[upgrade],
)
analysis = AnalysisSpec(name="controller on T1", upgrade="controller", report="campaign")

(output,) = ToggleEnergyRatioMethod().estimate(facts, [analysis])
print(f"{output.campaign_uplift:+.2%} ± {output.campaign_sigma:.2%}")
for warning in [*facts.warnings(), *output.warnings]:
    print(warning.severity, warning.message)
```

`subjects` and `references` are left to their defaults here: T1 is measured against T2 and T3.
