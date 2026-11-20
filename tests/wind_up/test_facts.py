"""Tests for the wind farm facts, the upgrade states and the record labels."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd
import pytest

from wind_up.analysis import AnalysisSpec
from wind_up.facts import SIDE_COL, NonTurbineMeasurement, Upgrade, WindFarmFacts
from wind_up.labels import (
    NEITHER,
    NORTHING,
    NORTHING_REASON_COL,
    OFF,
    ON,
    UPLIFT,
    UPLIFT_REASON_COL,
    Label,
    upgrade_state_column,
)
from wind_up.operating_state import (
    FULL_DOWNTIME,
    MISSING,
    NORMAL_OPERATION,
    PARTIAL_DOWNTIME,
    STATE_COL,
    STATE_COLUMNS,
    VALID_NORTHING_COL,
    VALID_UPLIFT_COL,
    WAKING_STATE_COL,
    OperatingStateConfig,
    StateValidity,
    apply_declarations,
    fill_grid,
    label_operating_states,
)
from wind_up.record_selection import ChangedInterval
from wind_up.toggle import ToggleSchedule

from .facts_fixtures import (
    COLS,
    FULL,
    NAMES,
    RATED,
    START,
    TIMEBASE,
    at,
    facts,
    permanent,
    records,
    toggle_frame,
    turbine_table,
)

INDEX = pd.date_range(START, periods=36, freq=TIMEBASE, tz="UTC")
MINUTES = pd.Timedelta(minutes=1)


def states(upgrade: Upgrade, turbine: str = "T1", index: pd.DatetimeIndex = INDEX) -> list[str]:
    return list(upgrade.states(index, turbine=turbine))


def of(labels: pd.DataFrame, turbine: str) -> pd.DataFrame:
    return labels[labels[COLS.turbine] == turbine]


# ---- Upgrade: declaration -----------------------------------------------------------------------


def test_an_upgrade_needs_a_name() -> None:
    with pytest.raises(ValueError, match="needs a name"):
        permanent(name=" ")


@pytest.mark.parametrize("turbines", [(), ("T1", "T1")])
def test_an_upgrade_needs_turbines_each_named_once(turbines: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="at least one turbine"):
        permanent(turbines=turbines)


def test_an_upgrade_declares_at_most_one_of_expected_uplift_and_reach() -> None:
    with pytest.raises(ValueError, match="at most one"):
        permanent(expected_uplift=0.03, wake_reach_d=8.0)


def test_the_reach_is_declared_or_follows_from_the_expected_uplift() -> None:
    assert permanent().reach_d is None
    assert permanent(wake_reach_d=6.0).reach_d == 6.0
    assert permanent(expected_uplift=0.05).reach_d == pytest.approx(12.0)


def test_an_installation_window_belongs_to_a_turbine_of_the_upgrade() -> None:
    with pytest.raises(ValueError, match="did not receive it"):
        permanent(installation_windows={"T2": [(at(1), at(2))]})


def test_a_window_cannot_end_before_it_starts() -> None:
    with pytest.raises(ValueError, match="ends before it starts"):
        permanent(installation_windows={"T1": [(at(2), at(1))]})
    with pytest.raises(ValueError, match="ends before it starts"):
        permanent(upgrade_state_overrides=[("T1", at(2), at(1), NEITHER)])


def test_an_override_names_an_upgrade_state() -> None:
    with pytest.raises(ValueError, match="upgrade state is one of"):
        permanent(upgrade_state_overrides=[("T1", at(1), at(2), "paused")])


def test_settling_is_for_a_toggle_and_never_negative() -> None:
    with pytest.raises(ValueError, match="settling_time"):
        permanent(settling_time=5 * MINUTES)
    with pytest.raises(ValueError, match="settling_time"):
        Upgrade(name="control", turbines=["T1"], timing=toggle_frame(), settling_time=-MINUTES)


def test_every_turbine_of_a_permanent_upgrade_needs_a_date_or_an_installation_window() -> None:
    with pytest.raises(ValueError, match=r"no upgrade date or installation window for \['T2'\]"):
        Upgrade(name="blades", turbines=["T1", "T2"], timing={"T1": at(3)})
    with pytest.raises(ValueError, match=r"no upgrade date or installation window for \['T2'\]"):
        Upgrade(name="blades", turbines=["T1", "T2"], installation_windows={"T1": [(at(1), at(2))]})


def test_a_toggle_frame_needs_its_two_columns() -> None:
    with pytest.raises(ValueError, match="toggle_df must have columns"):
        Upgrade(name="control", turbines=["T1"], timing=toggle_frame().drop(columns="toggle_off"))


def test_the_mode_follows_the_timing() -> None:
    assert permanent().mode == "prepost"
    assert Upgrade(name="control", turbines=["T1"], timing=toggle_frame()).mode == "toggle"
    assert (
        Upgrade(name="control", turbines=["T1"], timing=ToggleSchedule(period=pd.Timedelta(hours=1))).mode == "toggle"
    )


# ---- Upgrade: changeover and span ---------------------------------------------------------------


def test_the_changeover_of_a_dated_upgrade_is_its_date() -> None:
    assert permanent().changeover("T1") == (at(3), at(3))
    per_turbine = Upgrade(name="blades", turbines=["T1", "T2"], timing={"T1": at(2), "T2": at(4)})
    assert per_turbine.changeover("T2") == (at(4), at(4))


def test_the_changeover_widens_to_the_installation_window_holding_the_date() -> None:
    upgrade = permanent(installation_windows={"T1": [(at(0.5), at(1)), (at(2.5), at(3.5))]})
    assert upgrade.changeover("T1") == (at(2.5), at(3.5))


def test_the_changeover_of_an_undated_upgrade_is_the_envelope_of_its_windows() -> None:
    upgrade = Upgrade(name="blades", turbines=["T1"], installation_windows={"T1": [(at(1), at(2)), (at(3), at(4))]})
    assert upgrade.changeover("T1") == (at(1), at(4))


def test_a_toggle_or_another_turbine_has_no_changeover() -> None:
    with pytest.raises(ValueError, match="no changeover"):
        permanent().changeover("T2")
    with pytest.raises(ValueError, match="no changeover"):
        Upgrade(name="control", turbines=["T1"], timing=toggle_frame()).changeover("T1")


def test_the_signal_span_of_a_frame_runs_from_its_first_to_its_last_stated_record() -> None:
    frame = toggle_frame()
    frame.iloc[-2:] = False
    assert Upgrade(name="control", turbines=["T1"], timing=frame).signal_span() == (INDEX[6], INDEX[-3])


def test_the_signal_span_of_a_schedule_starts_at_its_start_and_stays_open() -> None:
    schedule = ToggleSchedule(period=pd.Timedelta(hours=1), start=at(1))
    assert Upgrade(name="control", turbines=["T1"], timing=schedule).signal_span() == (at(1), None)


def test_a_frame_that_never_states_on_or_off_has_an_open_span() -> None:
    frame = toggle_frame()
    frame[:] = False
    assert Upgrade(name="control", turbines=["T1"], timing=frame).signal_span() == (None, None)


def test_a_permanent_upgrade_has_no_signal_span() -> None:
    with pytest.raises(ValueError, match="not a toggle"):
        permanent().signal_span()


# ---- Upgrade: states ----------------------------------------------------------------------------


def test_a_permanent_upgrade_is_off_before_its_date_and_on_from_it() -> None:
    assert states(permanent()) == [OFF] * 18 + [ON] * 18


def test_a_turbine_that_did_not_receive_the_upgrade_is_off() -> None:
    assert states(permanent(), "T2") == [OFF] * 36


def test_an_installation_window_is_in_neither_state() -> None:
    upgrade = permanent(installation_windows={"T1": [(at(2.5), at(3.5))]})
    assert states(upgrade) == [OFF] * 15 + [NEITHER] * 6 + [ON] * 15


def test_an_undated_upgrade_is_in_neither_state_from_its_first_window_to_its_last() -> None:
    upgrade = Upgrade(name="blades", turbines=["T1"], installation_windows={"T1": [(at(1), at(2)), (at(3), at(4))]})
    assert states(upgrade) == [OFF] * 6 + [NEITHER] * 18 + [ON] * 12


def test_an_override_sets_the_state_of_its_turbine_over_its_window() -> None:
    upgrade = permanent(upgrade_state_overrides=[("T1", at(4), at(5), NEITHER), ("T2", at(0), at(1), NEITHER)])
    assert states(upgrade) == [OFF] * 18 + [ON] * 6 + [NEITHER] * 6 + [ON] * 6
    assert states(upgrade, "T2") == [NEITHER] * 6 + [OFF] * 30


def test_an_override_without_a_turbine_applies_to_every_turbine() -> None:
    upgrade = permanent(upgrade_state_overrides=[(None, at(0), at(1), NEITHER)])
    assert states(upgrade, "T1")[:7] == [NEITHER] * 6 + [OFF]
    assert states(upgrade, "T3")[:7] == [NEITHER] * 6 + [OFF]


def test_a_toggle_frame_is_off_before_it_starts_then_follows_the_signal() -> None:
    upgrade = Upgrade(name="control", turbines=["T1"], timing=toggle_frame(block=3, first=6))
    assert states(upgrade)[:15] == [OFF] * 6 + [OFF] * 3 + [ON] * 3 + [OFF] * 3
    assert states(upgrade, "T2") == [OFF] * 36


def test_a_toggle_frame_is_in_neither_state_where_it_states_none_or_has_no_record() -> None:
    frame = toggle_frame(n=24)
    frame.iloc[12:15] = False
    upgrade = Upgrade(name="control", turbines=["T1"], timing=frame)
    got = states(upgrade)
    assert got[12:15] == [NEITHER] * 3
    assert got[24:] == [NEITHER] * 12
    earlier = pd.date_range(START - pd.Timedelta(hours=1), periods=3, freq=TIMEBASE, tz="UTC")
    assert states(upgrade, index=earlier) == [OFF] * 3


def test_a_toggle_frame_that_never_states_on_or_off_is_in_neither_state() -> None:
    frame = toggle_frame()
    frame[:] = False
    assert states(Upgrade(name="control", turbines=["T1"], timing=frame)) == [NEITHER] * 36


def test_settling_puts_the_records_after_each_change_in_neither_state() -> None:
    frame = toggle_frame(block=3, first=6)
    short = Upgrade(name="control", turbines=["T1"], timing=frame, settling_time=5 * MINUTES)
    longer = Upgrade(name="control", turbines=["T1"], timing=frame, settling_time=15 * MINUTES)
    # the first change is off to on at record 9; the start of the signal at record 6 is not a change
    assert states(short)[6:15] == [OFF] * 3 + [NEITHER, ON, ON] + [NEITHER, OFF, OFF]
    assert states(longer)[6:15] == [OFF] * 3 + [NEITHER, NEITHER, ON] + [NEITHER, NEITHER, OFF]


def test_settling_follows_a_change_between_adjacent_records_only() -> None:
    frame = toggle_frame(block=3, first=6)
    frame.iloc[9] = False  # the first on record states neither, so on is reached from neither
    upgrade = Upgrade(name="control", turbines=["T1"], timing=frame, settling_time=5 * MINUTES)
    assert states(upgrade)[6:13] == [OFF] * 3 + [NEITHER, ON, ON] + [NEITHER]


def test_a_schedule_is_off_before_its_start_then_alternates() -> None:
    schedule = ToggleSchedule(period=pd.Timedelta(hours=1), start=at(1))
    upgrade = Upgrade(name="control", turbines=["T1"], timing=schedule)
    assert states(upgrade)[:18] == [OFF] * 6 + [OFF] * 3 + [ON] * 3 + [OFF] * 3 + [ON] * 3


def test_settling_on_a_schedule_follows_each_change_after_the_first_block() -> None:
    schedule = ToggleSchedule(period=pd.Timedelta(hours=1), start=at(1))
    upgrade = Upgrade(name="control", turbines=["T1"], timing=schedule, settling_time=10 * MINUTES)
    assert states(upgrade)[:18] == [OFF] * 9 + [NEITHER, ON, ON] + [NEITHER, OFF, OFF] + [NEITHER, ON, ON]


def test_a_schedule_without_a_start_alternates_from_the_first_timestamp() -> None:
    schedule = ToggleSchedule(period=pd.Timedelta(hours=1))
    upgrade = Upgrade(name="control", turbines=["T1"], timing=schedule, settling_time=10 * MINUTES)
    assert states(upgrade)[:9] == [OFF] * 3 + [NEITHER, ON, ON] + [NEITHER, OFF, OFF]
    assert states(upgrade, index=INDEX[:0]) == []


# ---- WindFarmFacts: declaration -----------------------------------------------------------------


def test_the_facts_freeze_their_collections() -> None:
    made = facts()
    assert isinstance(made.upgrades, tuple)
    assert isinstance(made.labels, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        made.timebase = pd.Timedelta(hours=1)  # type: ignore[misc]


def test_the_records_need_a_time_index() -> None:
    with pytest.raises(TypeError, match="time index"):
        facts(records().reset_index(drop=True))


def test_the_timebase_is_positive() -> None:
    with pytest.raises(ValueError, match="timebase"):
        facts(timebase=pd.Timedelta(0))


def test_the_records_need_the_turbine_and_power_columns() -> None:
    with pytest.raises(ValueError, match="active_power"):
        facts(records().drop(columns=COLS.active_power))


def test_the_records_need_the_availability_column_the_schema_names() -> None:
    with pytest.raises(ValueError, match="availability"):
        facts(records().drop(columns=COLS.availability))


def test_pitch_outside_a_full_turn_is_blanked_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    frame = records()
    frame.iloc[[0, 5, 9], frame.columns.get_loc(COLS.pitch)] = [3.8e12, -360.5, 360.0]
    given = frame.copy()
    with caplog.at_level(logging.WARNING, logger="wind_up.facts"):
        made = facts(frame)
    pitch = made.records[COLS.pitch]
    assert pitch.isna().sum() == 2
    assert np.isnan(pitch.iloc[0])
    assert np.isnan(pitch.iloc[5])
    assert pitch.iloc[9] == 360.0
    pd.testing.assert_frame_equal(frame, given)
    assert "2 record(s)" in caplog.text
    assert "pitch" in caplog.text
    assert "values -360.5 to 3.8e+12 deg" in caplog.text


def test_a_single_blanked_value_is_shown_in_the_warning(caplog: pytest.LogCaptureFixture) -> None:
    frame = records()
    frame.iloc[4, frame.columns.get_loc(COLS.pitch)] = 999.0
    with caplog.at_level(logging.WARNING, logger="wind_up.facts"):
        facts(frame)
    assert "1 record(s)" in caplog.text
    assert "value 999 deg" in caplog.text


def test_pitch_within_a_full_turn_is_kept_without_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    frame = records()
    with caplog.at_level(logging.WARNING, logger="wind_up.facts"):
        made = facts(frame)
    assert made.records is frame
    assert not caplog.records


def test_wind_speed_outside_its_range_is_blanked_and_its_bounds_are_kept() -> None:
    frame = records()
    frame.iloc[[0, 1, 2, 3], frame.columns.get_loc(COLS.wind_speed)] = [-0.1, 0.0, 200.0, 200.1]
    wind_speed = facts(frame).records[COLS.wind_speed]
    assert wind_speed.iloc[:4].isna().tolist() == [True, False, False, True]


EXTRA_COLS = dataclasses.replace(
    COLS, ambient_temp="temp", reactive_power="q", active_power_min="p_min", power_setpoint="setpoint"
)


def test_every_checked_role_is_blanked_outside_its_range(caplog: pytest.LogCaptureFixture) -> None:
    frame = records().assign(temp=10.0, q=0.0, p_min=500.0, setpoint=1.0e9)
    frame.iloc[0, frame.columns.get_loc("temp")] = -100.5
    frame.iloc[1, frame.columns.get_loc("temp")] = 200.0
    frame.iloc[2, frame.columns.get_loc("q")] = -2 * RATED - 1
    frame.iloc[3, frame.columns.get_loc("p_min")] = 2 * RATED + 1
    with caplog.at_level(logging.WARNING, logger="wind_up.facts"):
        made = facts(frame, columns=EXTRA_COLS).records
    assert made["temp"].isna().sum() == 1
    assert np.isnan(made["temp"].iloc[0])
    assert made["q"].isna().sum() == 1
    assert made["p_min"].isna().sum() == 1
    assert not made["setpoint"].isna().any()
    for column in ("temp", "q", "p_min"):
        assert repr(column) in caplog.text
    assert "setpoint" not in caplog.text


def test_a_range_in_rated_power_reads_each_turbines_own_rating() -> None:
    frame = records().assign(q=1500.0)
    made = facts(frame, columns=EXTRA_COLS, turbines=turbine_table(rated_power_kw=[500.0, RATED, RATED, RATED])).records
    blanked = made.loc[made["q"].isna(), COLS.turbine]
    assert set(blanked) == {"T1"}
    assert len(blanked) == len(frame) // len(NAMES)


def test_a_label_needs_its_column_and_its_own_name() -> None:
    with pytest.raises(ValueError, match="in_sector"):
        facts(labels=[Label(name="in sector", column="in_sector")])
    frame = records().assign(a=True)
    with pytest.raises(ValueError, match="its own name"):
        facts(frame, labels=[Label(name="x", column="a"), Label(name="x", column="a")])


@pytest.mark.parametrize("value", [np.nan, "yes", 2])
def test_a_label_column_is_true_or_false_on_every_record(value: object) -> None:
    frame = records().assign(flag=True).astype({"flag": object})
    frame.iloc[0, frame.columns.get_loc("flag")] = value
    with pytest.raises(ValueError, match="true or false on every record"):
        facts(frame, labels=[Label(name="flag", column="flag")])


def test_the_turbine_table_needs_its_columns() -> None:
    with pytest.raises(ValueError, match="rated_power_kw"):
        facts(turbines=turbine_table().drop(columns="rated_power_kw"))


def test_the_turbine_table_names_each_turbine_once() -> None:
    with pytest.raises(ValueError, match="more than once"):
        facts(turbines=turbine_table(("T1", "T2", "T3", "T4", "T4")))


@pytest.mark.parametrize("rated", [0.0, -1.0, np.nan])
def test_every_turbine_needs_a_rated_power(rated: float) -> None:
    with pytest.raises(ValueError, match="rated power"):
        facts(turbines=turbine_table(rated_power_kw=rated))


def test_the_records_hold_only_turbines_of_the_table() -> None:
    with pytest.raises(ValueError, match=r"\['T4'\]"):
        facts(turbines=turbine_table(("T1", "T2", "T3")))


def test_the_facts_need_an_upgrade_and_each_its_own_name() -> None:
    with pytest.raises(ValueError, match="at least one upgrade"):
        facts(upgrades=[])
    with pytest.raises(ValueError, match="its own name"):
        facts(upgrades=[permanent(), permanent()])


def test_an_upgrade_names_only_turbines_of_the_table() -> None:
    with pytest.raises(ValueError, match=r"\['T9'\]"):
        facts(upgrades=[permanent(turbines=("T9",))])
    with pytest.raises(ValueError, match=r"\['T9'\]"):
        facts(upgrades=[permanent(upgrade_state_overrides=[("T9", at(0), at(1), NEITHER)])])


@pytest.mark.parametrize(
    "declared",
    [
        {"exclusions": [("T9", at(0), at(1))]},
        {"operating_state_overrides": [("T9", at(0), at(1), FULL_DOWNTIME)]},
        {"north_offsets": [("T9", at(0), 3.0)]},
    ],
)
def test_a_declaration_names_only_turbines_of_the_table(declared: dict) -> None:
    with pytest.raises(ValueError, match=r"\['T9'\]"):
        facts(**declared)


def test_a_non_turbine_measurement_cannot_carry_a_turbines_name() -> None:
    mast = NonTurbineMeasurement(name="T1", latitude=57.5, longitude=-3.0, records=pd.DataFrame())
    with pytest.raises(ValueError, match="carry a turbine's name"):
        facts(non_turbine_measurements=[mast])


def test_a_reference_may_be_a_measurement_and_a_subject_may_not() -> None:
    lidar = NonTurbineMeasurement(name="LiDAR", latitude=57.5, longitude=-3.0, records=pd.DataFrame())
    given = facts(non_turbine_measurements=[lidar])
    analysis = AnalysisSpec(name="a", upgrade="blades", references=["T2", "LiDAR"])
    given.check(analysis)
    assert given.measurement_names == ["LiDAR"]
    assert given.measurement_references(analysis) == ["LiDAR"]
    assert given.turbine_references(analysis) == ["T2"]
    with pytest.raises(ValueError, match="LiDAR"):
        given.check(AnalysisSpec(name="a", upgrade="blades", subjects=["LiDAR"]))


# ---- WindFarmFacts: lookups ---------------------------------------------------------------------


def test_the_turbine_names_follow_the_table() -> None:
    assert facts().turbine_names == list(NAMES)


def test_the_layout_comes_from_the_turbine_table() -> None:
    layout = facts().layout
    assert list(layout.frame["name"]) == list(NAMES)
    assert layout.distance_m[0, 1] == pytest.approx(500.0, rel=0.01)


def test_a_layout_needs_a_rotor_diameter() -> None:
    with pytest.raises(ValueError, match="rotor_diameter_m"):
        _ = facts(turbines=turbine_table().drop(columns="rotor_diameter_m")).layout


def test_a_turbines_rated_power_is_read_from_the_table() -> None:
    made = facts(turbines=turbine_table(rated_power_kw=[2000.0, 2300.0, 2000.0, 2000.0]))
    assert made.rated_power_kw("T2") == 2300.0
    with pytest.raises(KeyError, match="T9"):
        made.rated_power_kw("T9")


def test_an_upgrade_is_found_by_its_name() -> None:
    made = facts(upgrades=[permanent(), permanent("tune", turbines=("T2",))])
    assert made.upgrade("tune").turbines == ("T2",)
    with pytest.raises(KeyError, match="no upgrade is called 'paint'"):
        made.upgrade("paint")


def test_facts_with_an_availability_counter_raise_no_warning() -> None:
    assert facts().warnings() == ()


def test_facts_without_an_availability_counter_warn_that_downtime_is_not_labelled() -> None:
    (warning,) = facts(columns=dataclasses.replace(COLS, availability=None)).warnings()
    assert warning.severity == "medium"
    assert "downtime could not be labelled" in warning.message


# ---- WindFarmFacts: record labels ---------------------------------------------------------------


def test_the_record_labels_hold_a_row_per_turbine_and_timestamp() -> None:
    labels = facts().record_labels()
    assert len(labels) == 36 * 4
    assert list(labels.columns) == [
        COLS.turbine,
        *STATE_COLUMNS,
        upgrade_state_column("blades"),
        UPLIFT_REASON_COL,
        NORTHING_REASON_COL,
    ]
    assert labels[COLS.turbine].iloc[:4].tolist() == list(NAMES)
    assert (labels[STATE_COL] == NORMAL_OPERATION).all()
    assert labels[VALID_UPLIFT_COL].all()
    assert (labels[UPLIFT_REASON_COL] == "").all()


def test_a_record_never_supplied_reads_as_missing() -> None:
    frame = records()
    kept = ~((frame[COLS.turbine] == "T2") & (frame.index == INDEX[5]))
    labels = of(facts(frame[kept]).record_labels(), "T2")
    assert len(labels) == 36
    assert labels[STATE_COL].iloc[5] == MISSING
    assert not labels[VALID_UPLIFT_COL].iloc[5]
    assert labels[UPLIFT_REASON_COL].iloc[5] == MISSING


def test_the_operating_states_are_those_of_the_step_two_functions() -> None:
    frame = records()
    frame.loc[(frame[COLS.turbine] == "T3") & (frame.index < at(1)), [COLS.availability, COLS.active_power]] = 0.0
    frame.loc[(frame[COLS.turbine] == "T4") & (frame.index < at(1)), COLS.availability] = FULL / 2
    config = OperatingStateConfig(parked_pitch_above_deg=4.0)
    exclusions = [("T1", at(1), at(2))]
    overrides = [(None, at(2), at(2.5), PARTIAL_DOWNTIME)]
    made = facts(frame, operating_states=config, exclusions=exclusions, operating_state_overrides=overrides)
    expected = apply_declarations(
        label_operating_states(
            fill_grid(frame, turbine_col=COLS.turbine, timebase=TIMEBASE),
            columns=COLS,
            config=config,
            timebase=TIMEBASE,
            rated_power_kw=RATED,
        ),
        turbine_col=COLS.turbine,
        config=config,
        exclusions=exclusions,
        overrides=overrides,
    )
    labels = made.record_labels()
    pd.testing.assert_frame_equal(labels[[COLS.turbine, *STATE_COLUMNS]], expected[[COLS.turbine, *STATE_COLUMNS]])
    assert set(labels[STATE_COL]) == {MISSING, FULL_DOWNTIME, PARTIAL_DOWNTIME, NORMAL_OPERATION}


def test_a_site_state_is_read_from_its_column() -> None:
    frame = records().assign(site=None)
    frame.loc[frame.index < at(1), "site"] = "noise mode"
    config = OperatingStateConfig(
        site_state_column="site",
        site_states={"noise mode": StateValidity(northing=True, waking="waking", uplift=False)},
    )
    labels = facts(frame, operating_states=config).record_labels()
    early = labels[labels.index < at(1)]
    assert (early[STATE_COL] == "noise mode").all()
    assert not early[VALID_UPLIFT_COL].any()
    assert early[VALID_NORTHING_COL].all()
    assert (early[UPLIFT_REASON_COL] == "noise mode").all()
    assert (early[NORTHING_REASON_COL] == "").all()


def test_each_turbine_is_labelled_at_its_own_rated_power() -> None:
    frame = records()
    frame.loc[frame.index == INDEX[4], COLS.active_power] = 4500.0  # out of range at 2000 kW, in range at 3000 kW
    table = turbine_table(rated_power_kw=[2000.0, 3000.0, 2000.0, 3000.0])
    labels = facts(frame, turbines=table).record_labels()
    at_spike = labels[labels.index == INDEX[4]].set_index(COLS.turbine)[STATE_COL]
    assert at_spike.to_dict() == {"T1": MISSING, "T2": NORMAL_OPERATION, "T3": MISSING, "T4": NORMAL_OPERATION}
    assert labels[VALID_UPLIFT_COL].dtype == bool
    assert labels[COLS.turbine].iloc[:4].tolist() == list(NAMES)


def test_without_an_availability_counter_the_states_follow_power_alone() -> None:
    frame = records().drop(columns=COLS.availability)
    frame.loc[(frame[COLS.turbine] == "T1") & (frame.index == INDEX[2]), COLS.active_power] = np.nan
    labels = facts(frame, columns=dataclasses.replace(COLS, availability=None)).record_labels()
    assert of(labels, "T1")[STATE_COL].iloc[2] == MISSING
    assert (labels[STATE_COL] != MISSING).sum() == 36 * 4 - 1
    assert set(labels[STATE_COL]) == {MISSING, NORMAL_OPERATION}


def test_each_upgrade_has_its_state_on_every_record() -> None:
    control = Upgrade(name="control", turbines=["T2", "T3"], timing=toggle_frame(block=3, first=6))
    labels = facts(upgrades=[permanent(), control]).record_labels()
    assert of(labels, "T1")[upgrade_state_column("blades")].tolist() == [OFF] * 18 + [ON] * 18
    assert of(labels, "T2")[upgrade_state_column("blades")].tolist() == [OFF] * 36
    assert of(labels, "T3")[upgrade_state_column("control")].tolist()[:12] == [OFF] * 9 + [ON] * 3
    assert of(labels, "T1")[upgrade_state_column("control")].tolist() == [OFF] * 36


def test_an_analyst_label_is_carried_under_its_name_and_changes_nothing_by_default() -> None:
    frame = records().assign(in_sector=False)
    frame.loc[frame.index < at(1), "in_sector"] = True
    labels = facts(frame, labels=[Label(name="in sector", column="in_sector")]).record_labels()
    assert labels["in sector"].tolist() == [True] * 24 + [False] * 120
    assert labels["in sector"].dtype == bool
    assert labels[VALID_UPLIFT_COL].all()
    assert labels[VALID_NORTHING_COL].all()


def test_an_analyst_label_is_false_on_a_record_never_supplied() -> None:
    frame = records().assign(in_sector=True)
    kept = ~((frame[COLS.turbine] == "T2") & (frame.index == INDEX[5]))
    labels = of(facts(frame[kept], labels=[Label(name="in sector", column="in_sector")]).record_labels(), "T2")
    assert not labels["in sector"].iloc[5]
    assert labels["in sector"].sum() == 35


def test_a_label_withholding_uplift_leaves_the_operating_and_waking_states_alone() -> None:
    frame = records().assign(curtailed=False)
    frame.loc[(frame[COLS.turbine] == "T1") & (frame.index < at(1)), "curtailed"] = True
    label = Label(name="temporary curtailment", column="curtailed", withholds=frozenset({UPLIFT}))
    labels = of(facts(frame, labels=[label]).record_labels(), "T1")
    assert labels[VALID_UPLIFT_COL].tolist() == [False] * 6 + [True] * 30
    assert labels[UPLIFT_REASON_COL].tolist() == ["temporary curtailment"] * 6 + [""] * 30
    assert (labels[STATE_COL] == NORMAL_OPERATION).all()
    assert (labels[WAKING_STATE_COL] == "waking").all()
    assert labels[VALID_NORTHING_COL].all()


def test_the_reason_names_the_state_and_every_withholding_label() -> None:
    frame = records().assign(a=True, b=True)
    frame.loc[frame[COLS.turbine] == "T1", [COLS.availability, COLS.active_power]] = 0.0
    kinds = [
        Label(name="a", column="a", withholds=frozenset({UPLIFT, NORTHING})),
        Label(name="b", column="b", withholds=frozenset({UPLIFT})),
    ]
    labels = facts(frame, labels=kinds).record_labels()
    assert (of(labels, "T1")[UPLIFT_REASON_COL] == f"{FULL_DOWNTIME}; a; b").all()
    assert (of(labels, "T2")[UPLIFT_REASON_COL] == "a; b").all()
    assert (of(labels, "T2")[NORTHING_REASON_COL] == "a").all()
    assert not labels[VALID_NORTHING_COL].any()


# ---- WindFarmFacts: analyses --------------------------------------------------------------------


def sides(made: WindFarmFacts, analysis: AnalysisSpec, turbine: str = "T1") -> list[str]:
    labels = made.record_labels()
    side = made.comparison_sides(analysis, record_labels=labels)
    assert side.name == SIDE_COL
    return side[(labels[COLS.turbine] == turbine).to_numpy()].tolist()


def test_with_an_open_period_every_record_of_a_subject_is_compared() -> None:
    made = facts()
    analysis = AnalysisSpec(name="blades on T1", upgrade="blades")
    assert sides(made, analysis) == [OFF] * 18 + [ON] * 18
    assert sides(made, analysis, "T2") == [NEITHER] * 36


def test_a_period_only_restricts_the_records_compared() -> None:
    analysis = AnalysisSpec(name="short", upgrade="blades", period=(at(2), at(4)))
    assert sides(facts(), analysis) == [NEITHER] * 12 + [OFF] * 6 + [ON] * 6 + [NEITHER] * 12
    assert (
        sides(facts(), AnalysisSpec(name="late", upgrade="blades", period=(at(4), None))) == [NEITHER] * 24 + [ON] * 12
    )
    assert (
        sides(facts(), AnalysisSpec(name="early", upgrade="blades", period=(None, at(1)))) == [OFF] * 6 + [NEITHER] * 30
    )


def test_a_toggle_compares_only_the_records_within_its_signal() -> None:
    frame = toggle_frame(n=30, block=3, first=6)
    control = Upgrade(name="control", turbines=["T1", "T2"], timing=frame, settling_time=5 * MINUTES)
    got = sides(facts(upgrades=[control]), AnalysisSpec(name="control", upgrade="control"))
    assert got[:15] == [NEITHER] * 6 + [OFF] * 3 + [NEITHER, ON, ON] + [NEITHER, OFF, OFF]
    assert got[29] in {ON, OFF}
    assert got[30:] == [NEITHER] * 6


def test_a_turbine_overridden_to_neither_is_on_no_side() -> None:
    control = Upgrade(
        name="control",
        turbines=["T1", "T2"],
        timing=toggle_frame(block=3, first=0),
        upgrade_state_overrides=[("T2", at(0), at(3), NEITHER)],
    )
    made = facts(upgrades=[control])
    analysis = AnalysisSpec(name="control", upgrade="control")
    assert sides(made, analysis, "T2")[:19] == [NEITHER] * 18 + [OFF]
    assert sides(made, analysis, "T1")[:6] == [OFF] * 3 + [ON] * 3


def test_an_include_label_keeps_only_the_records_holding_it() -> None:
    frame = records().assign(in_sector=False)
    frame.loc[(frame.index >= at(2)) & (frame.index < at(4)), "in_sector"] = True
    made = facts(frame, labels=[Label(name="in sector", column="in_sector")])
    analysis = AnalysisSpec(name="steer", upgrade="blades", include_labels=["in sector"])
    assert sides(made, analysis) == [NEITHER] * 12 + [OFF] * 6 + [ON] * 6 + [NEITHER] * 12


def test_an_exclude_label_drops_the_records_holding_it() -> None:
    frame = records().assign(in_sector=False)
    frame.loc[(frame.index >= at(2)) & (frame.index < at(4)), "in_sector"] = True
    made = facts(frame, labels=[Label(name="in sector", column="in_sector")])
    analysis = AnalysisSpec(name="not steering", upgrade="blades", exclude_labels=["in sector"])
    assert sides(made, analysis) == [OFF] * 12 + [NEITHER] * 12 + [ON] * 12


def test_only_the_named_subjects_are_compared() -> None:
    made = facts(upgrades=[permanent(turbines=("T1", "T2"))])
    analysis = AnalysisSpec(name="one", upgrade="blades", subjects=["T2"])
    assert made.subjects_of(analysis) == ["T2"]
    assert sides(made, analysis, "T1") == [NEITHER] * 36
    assert sides(made, analysis, "T2") == [OFF] * 18 + [ON] * 18
    assert made.subjects_of(AnalysisSpec(name="all", upgrade="blades")) == ["T1", "T2"]


def test_an_analysis_names_what_the_facts_hold() -> None:
    frame = records().assign(a=True)
    made = facts(frame, upgrades=[permanent(turbines=("T1", "T2"))], labels=[Label(name="a", column="a")])
    made.check(AnalysisSpec(name="fine", upgrade="blades", subjects=["T1"], references=["T3"], include_labels=["a"]))
    with pytest.raises(KeyError, match="no upgrade is called 'paint'"):
        made.check(AnalysisSpec(name="x", upgrade="paint"))
    with pytest.raises(ValueError, match=r"\['T9'\]"):
        made.check(AnalysisSpec(name="x", upgrade="blades", references=["T9"]))
    with pytest.raises(ValueError, match="did not receive 'blades'"):
        made.check(AnalysisSpec(name="x", upgrade="blades", subjects=["T3"]))
    with pytest.raises(ValueError, match="both subject and reference"):
        made.check(AnalysisSpec(name="x", upgrade="blades", references=["T2", "T3"]))
    with pytest.raises(ValueError, match=r"names the labels \['b'\]"):
        made.check(AnalysisSpec(name="x", upgrade="blades", exclude_labels=["b"]))


def test_the_named_references_are_the_candidates() -> None:
    analysis = AnalysisSpec(name="x", upgrade="blades", references=["T4", "T3"])
    assert facts().references_of(analysis) == ["T4", "T3"]


def test_by_default_every_unchanged_turbine_is_a_candidate_reference() -> None:
    assert facts().references_of(AnalysisSpec(name="x", upgrade="blades")) == ["T2", "T3", "T4"]


def test_a_turbine_upgraded_long_before_can_be_a_reference() -> None:
    earlier = Upgrade(name="aero", turbines=["T2"], timing=START - pd.Timedelta(days=400))
    during = Upgrade(name="paint", turbines=["T3"], timing=at(2))
    made = facts(upgrades=[permanent(), earlier, during])
    assert made.references_of(AnalysisSpec(name="x", upgrade="blades")) == ["T2", "T4"]


def test_a_turbine_in_neither_state_is_not_a_default_reference() -> None:
    upgrade = permanent(upgrade_state_overrides=[("T4", at(1), at(2), NEITHER)])
    made = facts(upgrades=[upgrade])
    assert made.references_of(AnalysisSpec(name="x", upgrade="blades")) == ["T2", "T3"]


def test_a_change_outside_the_period_does_not_rule_a_reference_out() -> None:
    during = Upgrade(name="paint", turbines=["T3"], timing=at(2))
    made = facts(upgrades=[permanent(), during])
    analysis = AnalysisSpec(name="x", upgrade="blades", period=(at(2.5), at(5)))
    assert made.references_of(analysis) == ["T2", "T3", "T4"]


def test_settling_changes_nothing_on_a_signal_that_never_changes() -> None:
    frame = toggle_frame(block=40, first=6)
    upgrade = Upgrade(name="control", turbines=["T1"], timing=frame, settling_time=5 * MINUTES)
    assert states(upgrade) == [OFF] * 36


# ---- WindFarmFacts: changed turbines ------------------------------------------------------------

SPAN = (at(0), at(6))
EVERYWHERE = (ChangedInterval(None, None, None),)


def changed(made: WindFarmFacts, *, subject: str = "T1", near: pd.Timedelta | None = None) -> dict:
    analysis = AnalysisSpec(name="x", upgrade="blades")
    if near is None:
        return made.changed_turbines(analysis, subject=subject, span=SPAN)
    return made.changed_turbines(analysis, subject=subject, span=SPAN, near=near)


def test_the_subject_counts_as_changed_throughout() -> None:
    assert changed(facts()) == {"T1": EVERYWHERE}


def test_a_change_near_the_subjects_counts_throughout() -> None:
    made = facts(upgrades=[permanent(), Upgrade(name="paint", turbines=["T3"], timing=at(2))])
    assert changed(made) == {"T1": EVERYWHERE, "T3": EVERYWHERE}


def test_a_change_far_from_the_subjects_counts_from_its_start_or_until_its_end() -> None:
    later = Upgrade(name="paint", turbines=["T3"], timing=at(5))
    earlier = Upgrade(name="seal", turbines=["T4"], installation_windows={"T4": [(at(0.5), at(1))]})
    made = facts(upgrades=[permanent(), later, earlier])
    got = changed(made, near=pd.Timedelta(hours=1))
    assert got["T3"] == (ChangedInterval(at(5), None, None),)
    assert got["T4"] == (ChangedInterval(None, at(1), None),)


def test_a_change_outside_the_span_does_not_count() -> None:
    made = facts(upgrades=[permanent(), Upgrade(name="aero", turbines=["T2"], timing=at(-48))])
    assert changed(made) == {"T1": EVERYWHERE}


def test_a_turbine_in_neither_state_counts_as_changed_only_while_it_is() -> None:
    upgrade = permanent(upgrade_state_overrides=[("T4", at(1), at(2), NEITHER), ("T2", at(7), at(8), NEITHER)])
    got = changed(facts(upgrades=[upgrade]))
    assert got == {"T1": EVERYWHERE, "T4": (ChangedInterval(at(1), at(2), None),)}


def test_an_override_to_on_or_off_is_a_change_window() -> None:
    upgrade = permanent(upgrade_state_overrides=[("T3", at(4), at(5), ON)])
    got = changed(facts(upgrades=[upgrade]), near=pd.Timedelta(minutes=30))
    assert got["T3"] == (ChangedInterval(at(4), None, None),)


def test_a_change_counts_only_upwind_of_the_turbines_within_its_reach() -> None:
    paint = Upgrade(name="paint", turbines=["T3"], timing=at(2), wake_reach_d=6.0)
    held = permanent(upgrade_state_overrides=[("T4", at(1), at(2), NEITHER)], wake_reach_d=6.0)
    got = changed(facts(upgrades=[held, paint]))
    assert got["T1"] == (ChangedInterval(None, None, frozenset({"T2"})),)
    assert got["T3"] == (ChangedInterval(None, None, frozenset({"T2", "T4"})),)
    assert got["T4"] == (ChangedInterval(at(1), at(2), frozenset({"T3"})),)


def test_another_subject_of_a_toggle_counts_throughout() -> None:
    control = Upgrade(name="blades", turbines=["T1", "T2"], timing=toggle_frame(block=3, first=6))
    assert changed(facts(upgrades=[control])) == {"T1": EVERYWHERE, "T2": EVERYWHERE}


def test_a_schedule_without_a_start_changes_at_the_first_record() -> None:
    schedule = ToggleSchedule(period=pd.Timedelta(hours=1))
    control = Upgrade(name="blades", turbines=["T1", "T2"], timing=schedule)
    assert control.change_windows("T2", first_record=at(0)) == [(at(0), at(0))]
    assert changed(facts(upgrades=[control])) == {"T1": EVERYWHERE, "T2": EVERYWHERE}


def test_the_change_windows_hold_the_changeover_and_the_other_installation_windows() -> None:
    upgrade = permanent(installation_windows={"T1": [(at(0.5), at(1)), (at(2.5), at(3.5))]})
    assert upgrade.change_windows("T1", first_record=at(0)) == [(at(0.5), at(1)), (at(2.5), at(3.5))]
    assert upgrade.change_windows("T2", first_record=at(0)) == []


def test_only_a_turbine_that_received_the_upgrade_can_be_measured() -> None:
    with pytest.raises(ValueError, match="'T2' did not receive 'blades'"):
        changed(facts(), subject="T2")


# ---- Upgrade: toggle cycle and datum ------------------------------------------------------------


def test_a_schedules_cycle_and_datum_are_its_period_and_start() -> None:
    schedule = ToggleSchedule(period=pd.Timedelta(hours=1), start=at(1))
    upgrade = Upgrade(name="control", turbines=["T1"], timing=schedule)
    assert upgrade.cycle == pd.Timedelta(hours=1)
    assert upgrade.datum(first_record=at(0)) == at(1)
    open_ended = Upgrade(name="control", turbines=["T1"], timing=ToggleSchedule(period=pd.Timedelta(hours=1)))
    assert open_ended.datum(first_record=at(0)) == at(0)


def test_a_frames_cycle_is_inferred_from_its_runs_and_its_datum_is_its_first_stated_record() -> None:
    upgrade = Upgrade(name="control", turbines=["T1"], timing=toggle_frame(block=3, first=6))
    assert upgrade.cycle == pd.Timedelta(hours=1)
    assert upgrade.datum(first_record=at(0)) == at(1)


def test_a_declared_cycle_and_datum_are_used_as_given() -> None:
    upgrade = Upgrade(
        name="control",
        turbines=["T1"],
        timing=toggle_frame(block=3, first=6),
        toggle_cycle=pd.Timedelta(hours=1),
        toggle_datum=at(-0.5),
    )
    assert upgrade.cycle == pd.Timedelta(hours=1)
    assert upgrade.datum(first_record=at(0)) == at(-0.5)


@pytest.mark.parametrize("timing", [toggle_frame(block=3, first=6), ToggleSchedule(period=pd.Timedelta(hours=1))])
def test_a_declared_cycle_that_disagrees_with_the_timing_raises(timing: object) -> None:
    with pytest.raises(ValueError, match="disagrees with the toggle timing"):
        Upgrade(name="control", turbines=["T1"], timing=timing, toggle_cycle=pd.Timedelta(hours=2))  # type: ignore[arg-type]


def test_a_cycle_is_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Upgrade(name="control", turbines=["T1"], timing=toggle_frame(), toggle_cycle=-pd.Timedelta(hours=1))


def test_a_permanent_upgrade_has_no_cycle() -> None:
    with pytest.raises(ValueError, match="are for a toggle"):
        permanent(toggle_cycle=pd.Timedelta(hours=1))
    with pytest.raises(ValueError, match="not a toggle"):
        _ = permanent().cycle


def test_settling_gaps_in_a_frame_do_not_shorten_its_cycle() -> None:
    frame = toggle_frame(block=3, first=6)
    starts = frame["toggle_on"] & ~frame["toggle_on"].shift(fill_value=False)
    frame.loc[starts, ["toggle_on", "toggle_off"]] = False
    upgrade = Upgrade(name="control", turbines=["T1"], timing=frame, toggle_cycle=pd.Timedelta(hours=1))
    assert upgrade.cycle == pd.Timedelta(hours=1)


def test_a_frame_with_fewer_than_two_on_runs_states_no_cycle() -> None:
    frame = toggle_frame(block=40, first=6)
    with pytest.raises(ValueError, match="fewer than two on runs"):
        _ = Upgrade(name="control", turbines=["T1"], timing=frame).cycle
