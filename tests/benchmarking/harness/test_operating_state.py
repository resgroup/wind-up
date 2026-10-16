"""Tests for the operating-state labels and their validity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.harness.operating_state import (
    FULL_DOWNTIME,
    GENERIC_STATES,
    MISSING,
    NORMAL_OPERATION,
    PARTIAL_DOWNTIME,
    STATE_COL,
    VALID_NORTHING_COL,
    VALID_UPLIFT_COL,
    WAKING_STATE_COL,
    OperatingStateConfig,
    StateValidity,
    label_operating_states,
    state_hours,
    stuck_records,
)
from benchmarking.synthetic import HOT_COLUMNS

COLS = HOT_COLUMNS
RATED = 2300.0
TIMEBASE = pd.Timedelta(minutes=10)
FULL = TIMEBASE.total_seconds()
CURTAILED = "curtailed"
SITE = OperatingStateConfig(
    label_column="state",
    labels={CURTAILED: StateValidity(northing=True, waking="part waking", uplift=False)},
)


def frame(n: int = 4, *, turbine: str = "T01", **overrides: list) -> pd.DataFrame:
    """``n`` normally-operating records of one turbine, every signal varying, with ``overrides`` applied."""
    index = pd.date_range("2020-01-01", periods=n, freq=TIMEBASE, tz="UTC")
    values = np.arange(n, dtype=float)
    data: dict[str, object] = {
        COLS.turbine: turbine,
        COLS.active_power: 1000.0 + values,
        COLS.active_power_min: 900.0 + values,
        COLS.wind_speed: 8.0 + values / 10,
        COLS.wind_speed_sd: 1.0 + values / 10,
        COLS.gen_rpm: 1500.0 + values,
        COLS.availability: FULL,
        COLS.pitch: 1.0 + values / 10,
    }
    data.update(overrides)
    return pd.DataFrame(data, index=index)


def label(scada: pd.DataFrame, config: OperatingStateConfig | None = None) -> pd.DataFrame:
    return label_operating_states(
        scada, columns=COLS, config=config or OperatingStateConfig(), timebase=TIMEBASE, rated_power_kw=RATED
    )


def states(scada: pd.DataFrame, config: OperatingStateConfig | None = None) -> list[str]:
    return label(scada, config)[STATE_COL].tolist()


def test_ordinary_records_are_normal_operation() -> None:
    assert states(frame()) == [NORMAL_OPERATION] * 4


@pytest.mark.parametrize(
    ("power", "availability"),
    [
        (np.nan, FULL),
        (1000.0, np.nan),
        (-RATED, FULL),
        (2 * RATED, FULL),
        (1000.0, -0.05 * FULL),
        (1000.0, 1.05 * FULL),
    ],
)
def test_non_finite_or_out_of_range_records_are_missing(power: float, availability: float) -> None:
    scada = frame(2, **{COLS.active_power: [1000.0, power], COLS.availability: [FULL, availability]})
    assert states(scada) == [NORMAL_OPERATION, MISSING]


def test_just_inside_the_ranges_is_not_missing() -> None:
    scada = frame(
        3,
        **{
            COLS.active_power: [-0.99 * RATED, 1.99 * RATED, 1000.0],
            COLS.availability: [FULL, FULL, 1.04 * FULL],
        },
    )
    assert states(scada) == [NORMAL_OPERATION] * 3


def test_availability_counter_sets_downtime() -> None:
    scada = frame(3, **{COLS.availability: [0.0, 0.5 * FULL, FULL]})
    assert states(scada) == [FULL_DOWNTIME, PARTIAL_DOWNTIME, NORMAL_OPERATION]


def test_stuck_record_is_missing_unless_calm() -> None:
    calm = frame(3, **{COLS.wind_speed: [1.0, 1.0, 1.0]})
    calm.iloc[2] = calm.iloc[1]
    assert states(calm) == [NORMAL_OPERATION] * 3
    windy = frame(3)
    windy.iloc[2] = windy.iloc[1]
    assert states(windy) == [NORMAL_OPERATION, NORMAL_OPERATION, MISSING]


def test_stuck_ignores_signals_outside_the_schema() -> None:
    scada = frame(3)
    scada.iloc[2] = scada.iloc[1]
    scada["not_a_role"] = [1.0, 2.0, 3.0]
    assert states(scada)[2] == MISSING


def test_stuck_reads_each_turbines_own_previous_record() -> None:
    a = frame(2, turbine="T01")
    b = frame(2, turbine="T02")
    scada = pd.concat([a, b]).sort_index(kind="stable")
    # T02's first record repeats T01's first record but is T02's first: never stuck.
    assert states(scada) == [NORMAL_OPERATION] * 4


def test_a_turbines_first_record_is_never_stuck() -> None:
    scada = frame(2)
    assert not stuck_records(scada, columns=COLS).iloc[0]


@pytest.mark.parametrize(
    ("config", "pitch", "expected"),
    [
        (OperatingStateConfig(parked_pitch_above_deg=45.0), [50.0, 40.0, np.nan], [PARTIAL_DOWNTIME, None, None]),
        (OperatingStateConfig(parked_pitch_below_deg=-45.0), [-50.0, -40.0, np.nan], [PARTIAL_DOWNTIME, None, None]),
        (OperatingStateConfig(), [90.0, 90.5, 91.0], [None, None, None]),
    ],
)
def test_parked_pitch(config: OperatingStateConfig, pitch: list[float], expected: list[str | None]) -> None:
    got = states(frame(3, **{COLS.pitch: pitch}), config)
    assert got == [e or NORMAL_OPERATION for e in expected]


def test_parked_pitch_is_skipped_without_a_pitch_signal() -> None:
    scada = frame(2).drop(columns=COLS.pitch)
    assert states(scada, OperatingStateConfig(parked_pitch_above_deg=45.0)) == [NORMAL_OPERATION] * 2


def test_site_labels_sit_between_downtime_and_normal() -> None:
    scada = frame(
        5,
        state=[CURTAILED, CURTAILED, CURTAILED, None, PARTIAL_DOWNTIME],
        **{
            COLS.active_power: [np.nan, 1000.0, 1000.0, 1000.0, 1000.0],
            COLS.availability: [FULL, 0.0, FULL, FULL, FULL],
        },
    )
    assert states(scada, SITE) == [MISSING, FULL_DOWNTIME, CURTAILED, NORMAL_OPERATION, PARTIAL_DOWNTIME]


def test_a_caller_generic_label_is_kept_but_missing_wins() -> None:
    scada = frame(
        3,
        state=[PARTIAL_DOWNTIME, FULL_DOWNTIME, PARTIAL_DOWNTIME],
        **{COLS.active_power: [1000.0, 1000.0, np.nan]},
    )
    assert states(scada, SITE) == [PARTIAL_DOWNTIME, FULL_DOWNTIME, MISSING]


def test_a_caller_missing_label_is_missing() -> None:
    assert states(frame(1, state=[MISSING]), SITE) == [MISSING]


def test_validity_columns_follow_the_tables() -> None:
    scada = frame(
        4,
        state=[None, CURTAILED, None, None],
        **{COLS.availability: [FULL, FULL, 0.0, 0.5 * FULL], COLS.active_power: [1000.0, 1000.0, 1000.0, 1000.0]},
    )
    labelled = label(scada, SITE)
    assert labelled[VALID_NORTHING_COL].tolist() == [True, True, False, False]
    assert labelled[WAKING_STATE_COL].tolist() == ["waking", "part waking", "not waking", "part waking"]
    assert labelled[VALID_UPLIFT_COL].tolist() == [True, False, False, False]
    assert labelled[VALID_UPLIFT_COL].dtype == bool
    missing = label(frame(2, **{COLS.active_power: [1000.0, np.nan]}))
    assert missing[WAKING_STATE_COL].tolist() == ["waking", "missing"]


def test_labelling_returns_a_copy_in_the_original_order() -> None:
    scada = pd.concat([frame(2, turbine="T02"), frame(2, turbine="T01")])
    labelled = label(scada)
    assert labelled.index.equals(scada.index)
    assert labelled[COLS.turbine].tolist() == scada[COLS.turbine].tolist()
    assert STATE_COL not in scada.columns


def test_an_unknown_label_raises() -> None:
    with pytest.raises(ValueError, match="icing"):
        label(frame(1, state=["icing"]), SITE)


def test_a_named_label_column_that_is_absent_raises() -> None:
    with pytest.raises(ValueError, match="state"):
        label(frame(1), SITE)


def test_the_config_rejects_generic_names_and_two_pitch_rules() -> None:
    validity = StateValidity(northing=True, waking="waking", uplift=True)
    with pytest.raises(ValueError, match="normal operation"):
        OperatingStateConfig(label_column="state", labels={NORMAL_OPERATION: validity})
    with pytest.raises(ValueError, match="pitch"):
        OperatingStateConfig(parked_pitch_above_deg=45.0, parked_pitch_below_deg=-45.0)
    with pytest.raises(ValueError, match="label_column"):
        OperatingStateConfig(labels={"x": validity})
    with pytest.raises(ValueError, match="waking"):
        StateValidity(northing=True, waking="sometimes", uplift=True)


def test_generic_states_are_in_precedence_order() -> None:
    assert list(GENERIC_STATES) == [MISSING, FULL_DOWNTIME, PARTIAL_DOWNTIME, NORMAL_OPERATION]


def test_hours_sum_to_the_record_count() -> None:
    scada = pd.concat([frame(3, **{COLS.availability: [0.0, FULL, FULL]}), frame(5, turbine="T02")])
    hours = state_hours(label(scada), columns=COLS, timebase=TIMEBASE)
    assert hours["hours"].sum() == pytest.approx(len(scada) * TIMEBASE / pd.Timedelta(hours=1))
    t01 = hours[hours["turbine"] == "T01"].set_index("operating_state")["hours"]
    assert t01[FULL_DOWNTIME] == pytest.approx(1 / 6)
    assert t01[NORMAL_OPERATION] == pytest.approx(2 / 6)
