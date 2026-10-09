# Step 2: operating state and data validity — design

Date: 2026-10-07. Branch `v1-C3`. Part of the compliance audit of the composed v1 path against
`docs/v1/method.md` ("method.md" below). Step 1 (input plots, flat run folders) was done in commit
8e259a7.

This spec is self-contained: it names every file and symbol it touches, and needs no other notes.

## Goal

Bring method.md step 2 into the composed v1 path: every SCADA record of every turbine gets one
operating state, derived from cause signals only, and each state carries a validity for northing
(step 4), waking state (step 5) and uplift (step 8). Labelled plots let the analyst confirm the
labels.

## Audit finding (why this is non-compliant today)

- There are no state labels and no validity table. Validity is one boolean (`exclude_row`, which the
  composed path does not use) plus `NormalOperationFilter`, which drops the test turbine's non-finite
  power, availability below the full period and stuck rows.
- Nothing identifies curtailment, noise mode or a parked rotor. The HoT campaign parquet already holds
  `wtc_PowerRef_endvalue` and `wtc_PitcPosA_mean`, but no rule reads them.
- Two later steps define state from active power, against the principle:
  - waking: `waking_<power>` is power ≥ `WAKING_RATED_FRACTION` × rated (power_model features);
  - northing: `yaw_usable` requires power > 5% of rated.

  These are fixed at the step 4 and step 5 audits by reading the validity this step produces. They
  are out of scope here.

## Decisions (agreed with Alex)

Hybrid ownership:

- **wind-up owns the generic states.** Their meaning and validity are fixed and reserved: a caller
  cannot redefine them.
- **The caller owns the site states.** It supplies a per-record label column and a table giving the
  validity of each of its own labels.

### Generic states

The first matching rule wins:

| Precedence | State | Rule | Northing | Waking | Uplift |
| --- | --- | --- | --- | --- | --- |
| 1 | missing | active power or availability counter not finite (eg NaN) or out of range (see below), or the record is stuck, or the caller labelled it `missing` | not valid | missing | not valid |
| 2 | full downtime | availability counter <= 0, or the caller labelled it `full downtime` | not valid | not waking | not valid |
| 3 | partial downtime | 0 < availability counter < full period, or the rotor is parked (pitch beyond the declared threshold), or the caller labelled it `partial downtime` | not valid | part waking | not valid |
| 4 | *caller site label* | the caller's column holds a label | from caller table | from caller table | from caller table |
| 5 | normal operation | anything else | valid | waking | valid |

"Out of range" means active power outside the open interval (-rated, 2 × rated), or the
availability counter outside (-0.05 × full period, 1.05 × full period). Rated is the declaration's
`rated_power_kw` (2300 kW at HoT).

Details:

- **Full period:** the timebase in seconds (600 at HoT).
- **Availability counter:** the `ColumnSchema.availability` column (`benchmarking/synthetic/schema.py`).
- **Stuck:** today's `NormalOperationFilter._stuck` rule (`benchmarking/baselines/filtering.py`), but
  restricted to the schema's measured signals: every `ColumnSchema` role except `turbine` that is set
  and present in the frame (`active_power`, `wind_speed`, `wind_speed_sd`, `gen_rpm`,
  `availability`, and any of `active_power_min`, `pitch`, `reactive_power`, `nacelle_position`,
  `ambient_temp`). A record is stuck when every one of them is unchanged from the turbine's own
  previous record, and calms are exempt (wind speed < 1.5 m/s, today's `_VERY_LOW_WIND`). A
  turbine's first record is never stuck.
  - The calm exemption reads the nacelle anemometer. That is a documented deviation from "ideally
    never used to define a state": no upgrade-invariant wind signal exists before step 3.
  - The rule only reads the turbine's own previous record. It does not compare against power.
- **Parked rotor:** declared per site as a threshold and a direction (HoT: pitch > 45°), read from
  the `ColumnSchema.pitch` column. It is omitted when the site has no pitch signal. A NaN pitch is
  not parked.
- **Generic names in the caller's column:** the caller may write a generic state's name (e.g.
  `partial downtime`) when its site rules identify that state. This is how HoT's setpoint-0 stop
  becomes partial downtime.
  - The generic precedence still applies. A record with a caller `partial downtime` and a NaN power
    is `missing`.
  - The caller's table may not list a generic name.

### Validity values

- Northing and uplift: `valid` / `not valid`.
- Waking: `waking` / `part waking` / `not waking` / `missing`.
- Noise mode's uplift validity is whatever the caller table says. There is no "see below" value.

### Hill of Towie site labels

The site labeller reads `wtc_PowerRef_endvalue`, the setpoint at interval end. The setpoint at
interval start is the turbine's previous record's end value. It is NaN after a gap, and a NaN
setpoint end is not reduced. Each end is classified separately; a record takes the label of either
end in this order:

1. **partial downtime:** either setpoint is 0 (a stop command).
2. **start-up limit → no label:** either setpoint is 100 kW, the one-interval limit after cut-in.
   It is part of normal operation.
3. **BM curtailment:** the record is at or after 2018-11-07T00:00Z, when HoT joined the Balancing
   Mechanism, and either setpoint is below 2300 kW and not a noise step of that turbine.
4. **noise mode:** either setpoint is one of that turbine's noise steps:
   - T16, T17: {1993, 2116, 2130, 2207, 2208, 2261} kW;
   - T19: {2207, 2208} kW.

"Below 2300 kW" is rated power: there is no extra tolerance fraction.

Before 2018-11-07, reduced setpoints get no label. Evidence (a one-off analysis of all 21 turbines,
2016–2022; the script was not kept, and the numbers are recorded here as the rationale):

- Before 2018-11-07, non-zero reduced setpoints limit power (power within 50 kW of the setpoint,
  fully available) for 222 turbine-hours out of 429,393 producing hours (0.05%). They fall on 146
  days, almost all in 2016–early 2017, with a mean power of 1,929 kW against a mean setpoint of
  1,830 kW.
- After 2018-11-07, 87% of reduced-setpoint hours bite.

No active-power test is used. The cost is about 1,840 h after 2018-11-07 that are labelled
BM curtailment but did not bite, mostly in low wind (0.24% of hours).

HoT caller table, in `campaign.yaml`:

```yaml
operating_state:
  label_column: state
  parked_pitch_above_deg: 45
  labels:
    noise mode:      {northing: valid, waking: waking,      uplift: not valid}
    BM curtailment:  {northing: valid, waking: part waking, uplift: not valid}
```

Noise mode is `not valid` for uplift until the analyst declares otherwise. T13 has no noise steps;
T16, T17 and T19 do.

Icing is not identified for now. HoT has `wtc_IceRelay_timeon` should it be wanted later.

## Design

### `benchmarking/harness/operating_state.py` (new)

It sits next to `northing.py`, the other farm-wide preparation step.

- `GENERIC_STATES`: the reserved generic table (names, precedence, validity).
- `StateValidity`: a frozen dataclass of `northing: bool`, `waking: str` and `uplift: bool`.
- `OperatingStateConfig`: a frozen dataclass with these fields:
  - `label_column: str | None`;
  - `labels: dict[str, StateValidity]`;
  - `parked_pitch_above_deg: float | None`, plus `parked_pitch_below_deg: float | None` for types
    that feather negative. At most one may be set.

  Its constructor rejects a generic name in `labels`.
- `label_operating_states(scada_df, *, columns, config, timebase, rated_power_kw) -> pd.DataFrame`:
  - returns a copy of the long frame with four added columns: `operating_state` (str),
    `valid_northing` (bool), `waking_state` (str) and `valid_uplift` (bool);
  - raises if the label column holds a label that is in neither the caller table nor the generic
    names, or if the label column is named but absent;
  - a NaN label means "no site label".
- `state_hours(labelled, *, columns, timebase) -> pd.DataFrame`: hours per turbine and state, for a
  CSV next to the plots (the principle that quantities are expressed in hours).

Output column names are module constants, so later steps import them rather than repeating
strings.

### Declaration and loader

- `campaign.yaml` gains an optional `operating_state:` block (shape above). The loader parses it into
  `Declaration.operating_state: OperatingStateConfig`.
- Absent block: generic states only, with no parked-pitch rule. Every dataset still gets missing,
  downtime and normal, as method.md requires.
- `Declaration.resolved()` echoes the block into the run output.

### Composed run (`run_declaration`)

`run_declaration` labels states straight after `read_parquet` and before the step 1 input plots.
The labelled frame is what flows on to ERA5, planning and methods.

### Uplift use (step 8 consumer, wired now)

- In the power model (`benchmarking/baselines/power_model/method.py`), this step supersedes
  `NormalOperationFilter`. Its test-turbine row selection (`_select_rows`, and the coverage count in
  screening, `_upgraded_days`, the other `NormalOperationFilter` call in that file) keeps `valid_uplift` rows with a
  finite outcome.
- When the frame carries no `valid_uplift` (the synthetic harness and other unlabelled callers), the
  power model labels it itself with `label_operating_states` and a generic-only config (no site
  labels, no parked-pitch rule). Its behaviour may therefore shift slightly: the stuck rule now reads
  only schema roles, and out-of-range records are dropped.
- `NormalOperationFilter` stays for the other methods (`naive_ratio.py`, `toggle_specialist.py`).
  Its stuck rule moves into `operating_state` and the filter calls it, so there is one definition.
  The filter therefore gains the schema roles it needs, and its stuck verdicts may shift slightly
  for those methods too. This is accepted.
- Reference handling (power-free references, held-back references, `normal_operation_*` features) is
  unchanged here. It is audited at steps 5 and 9.

### HoT labeller (`benchmarking/campaigns/real.py`)

- `label_hot_site_states(scada_long) -> pd.Series` implements the rules above. It reads
  `wtc_PowerRef_endvalue` by name (it is HoT-specific, not a `ColumnSchema` role). The constants
  (noise steps, BM date, start-up limit, the 2300 kW reduced threshold) live with the HoT source
  module, `benchmarking/synthetic/sources/hill_of_towie.py`.
- `write_hot_aeroup_t13` writes the `state` column into `scada.parquet` and the `operating_state`
  block into `campaign.yaml`. A campaign folder therefore stays self-contained.
- The campaign has to be rewritten once:

  ```
  python -m benchmarking.campaigns.real <campaign dir>
  ```

### Labelled inspection plots (step 1/2 folder `input_data_plots/`)

Farm-wide, every turbine, all data, written with the step 1 plots:

- `operating_states_<turbine>.png`: a grid of scatters coloured by state.
  - Panels: power vs wind speed, rotor speed vs power, pitch vs power, rotor speed vs wind speed and
    pitch vs wind speed (the step 1 pairs).
  - States are drawn in a fixed categorical order, normal first and underneath, with the legend
    giving hours per state.
  - Power is drawn here to check the labels, never to make them.
- `operating_state_hours.csv`: from `state_hours`.
- `operating_state_hours.png`: stacked monthly hours per state for all turbines, small multiples per
  turbine, so curtailment and noise periods are visible over time.

The run-level `1_inputs` folder gets the same per-turbine scatter for the test turbine and its power
references over the span.

## Tests

`tests/benchmarking/harness/test_operating_state.py`:

- each generic rule;
- precedence (missing beats a caller label; downtime beats a caller label; a caller
  `partial downtime` is kept);
- parked pitch in both directions, and off;
- a stuck record with a calm exemption;
- an unknown label raises; a generic name in the caller table raises;
- validity columns follow the tables;
- hours sum to the record count × timebase.

`tests/benchmarking/campaigns/test_real.py` (the HoT labeller):

- a setpoint of 0 at either end → partial downtime;
- 100 kW → no label;
- reduced before and after 2018-11-07;
- a noise step on T16 versus the same value on T13 (curtailment after the BM date);
- start setpoint taken from the previous record, and NaN after a gap.

Loader tests: the `operating_state` block is parsed, an absent block gives the generic states, and
bad values raise.

Power model tests: with `valid_uplift` present, a curtailed record is dropped and a normal one is
kept; on an unlabelled frame, an out-of-range record is dropped.

`NormalOperationFilter` tests: the existing ones still pass, or are updated where the stuck rule
now ignores non-schema columns.

Plot tests: the files are written; a turbine with no pitch signal skips those panels.

## method.md edits

`docs/v1/method.md`:

- Step 2: replace "Whether the analyst labels the data … is decided at implementation" with the
  hybrid rule:
  - wind-up owns the generic states (missing, full and partial downtime, normal) with fixed validity;
  - the analyst supplies a label column and a table for site states only;
  - the analyst may assign a generic state from site rules (e.g. a stop setpoint → partial downtime).
- Step 2: partial downtime includes a parked rotor (pitch beyond a declared threshold).
- Step 2: the stuck rule's calm exemption uses the anemometer, noted as the one exception.

## Out of scope (recorded for later steps)

- Step 4: northing's `yaw_usable` power > 5% rule should read `valid_northing`.
- Step 5: waking from the power threshold should read `waking_state`. How `part waking` is
  represented is decided there.
- Steps 8 and 9: reference rows and the `normal_operation_*` features against `valid_uplift`.
- Icing.
