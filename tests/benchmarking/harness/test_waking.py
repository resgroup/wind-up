"""Tests for the waking direction, upwind turbines and waking scenarios of step 5."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns.declaration import layout_from_coords
from benchmarking.diagnostics.waking_scenarios import DIRECTION_CSV, SCENARIO_HOURS_CSV, write_waking_outputs
from benchmarking.harness.operating_state import (
    NOT_WAKING,
    PART_WAKING,
    VALID_NORTHING_COL,
    WAKING,
    WAKING_MISSING,
    WAKING_STATE_COL,
)
from benchmarking.harness.waking import (
    DIRECTION_COL,
    DIRECTION_SOURCE_COL,
    NEIGHBOURS,
    NONE,
    NORMAL,
    OWN,
    UNKNOWN,
    UPWIND_COL,
    WAKING_SCENARIO_COL,
    _nearest_valid_mean,
    add_waking_scenarios,
    waking_directions,
)
from benchmarking.synthetic import HOT_COLUMNS

if TYPE_CHECKING:
    from pathlib import Path

COLS = HOT_COLUMNS
NORTHED = COLS.northed("nacelle_position")
# Five rotor diameters apart, west to east along the equator.
SPACING_DEG = 0.0045
DIAMETER = 100.0
LINE = {f"T0{i + 1}": (0.0, i * SPACING_DEG) for i in range(3)}


def frame(records: dict[str, list[tuple[float, bool, str]]]) -> pd.DataFrame:
    """Long SCADA of ``(northed direction, valid for northing, waking state)`` per turbine and record."""
    n = len(next(iter(records.values())))
    index = pd.date_range("2020-01-01", periods=n, freq="10min", tz="UTC")
    parts = [
        pd.DataFrame(
            {
                COLS.turbine: turbine,
                NORTHED: [d for d, _, _ in rows],
                VALID_NORTHING_COL: [v for _, v, _ in rows],
                WAKING_STATE_COL: [s for _, _, s in rows],
            },
            index=index,
        )
        for turbine, rows in records.items()
    ]
    return pd.concat(parts)


def of(result: pd.DataFrame, turbine: str, col: str) -> list:
    """The ``col`` values of ``turbine``'s records, in time order."""
    return result.loc[result[COLS.turbine] == turbine, col].tolist()


def test_nearest_valid_mean_takes_the_first_four_valid_and_wraps_north() -> None:
    donors = np.array(
        [
            [350.0, 10.0, np.nan, np.nan, np.nan],
            [np.nan, 0.0, 0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0, 10.0, 200.0],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )
    result = _nearest_valid_mean(donors)
    assert result[0] == pytest.approx(0.0, abs=1e-9) or result[0] == pytest.approx(360.0)
    assert result[1] == pytest.approx(0.0, abs=1e-9)
    assert result[2] == pytest.approx(10.0)
    assert np.isnan(result[3])


def test_direction_is_own_unless_a_test_turbine_or_invalid() -> None:
    layout = layout_from_coords(LINE, rotor_diameter_m=DIAMETER)
    scada = frame(
        {
            "T01": [(260.0, True, WAKING), (100.0, False, WAKING), (np.nan, True, WAKING)],
            "T02": [(270.0, True, WAKING), (280.0, True, WAKING), (np.nan, False, WAKING)],
            "T03": [(275.0, True, WAKING), (290.0, True, WAKING), (np.nan, False, WAKING)],
        }
    )
    result = add_waking_scenarios(scada, columns=COLS, layout=layout, test_turbines=["T02"])
    assert of(result, "T01", DIRECTION_SOURCE_COL) == [OWN, NEIGHBOURS, UNKNOWN]
    assert of(result, "T01", DIRECTION_COL)[:2] == pytest.approx([260.0, 290.0])
    assert np.isnan(of(result, "T01", DIRECTION_COL)[2])
    # T02 is tested, so it never uses its own direction and never lends it.
    assert of(result, "T02", DIRECTION_SOURCE_COL) == [NEIGHBOURS, NEIGHBOURS, UNKNOWN]
    assert of(result, "T02", DIRECTION_COL)[1] == pytest.approx(290.0)


def test_directions_hold_neighbour_mean_alongside_own() -> None:
    layout = layout_from_coords(LINE, rotor_diameter_m=DIAMETER)
    scada = frame(
        {
            "T01": [(260.0, True, WAKING)],
            "T02": [(270.0, True, WAKING)],
            "T03": [(280.0, True, WAKING)],
        }
    )
    found = waking_directions(scada, columns=COLS, layout=layout, test_turbines=[])
    # T02's neighbours are T01 and T03, equidistant.
    assert found.neighbours[0, 1] == pytest.approx(270.0)
    assert found.direction[0].tolist() == pytest.approx([260.0, 270.0, 280.0])


def test_scenarios_list_upwind_turbines_that_are_not_waking() -> None:
    layout = layout_from_coords(LINE, rotor_diameter_m=DIAMETER)
    west, east = 270.0, 90.0
    scada = frame(
        {
            "T01": [(west, True, NOT_WAKING), (west, True, WAKING), (east, True, WAKING), (west, True, WAKING)],
            "T02": [
                (west, True, PART_WAKING),
                (west, True, WAKING),
                (east, True, WAKING),
                (west, True, WAKING_MISSING),
            ],
            "T03": [(west, True, WAKING), (west, True, WAKING), (east, True, WAKING), (west, True, WAKING)],
        }
    )
    result = add_waking_scenarios(scada, columns=COLS, layout=layout, test_turbines=[])
    assert of(result, "T01", UPWIND_COL) == [NONE, NONE, "T02; T03", NONE]
    assert of(result, "T01", WAKING_SCENARIO_COL) == [NORMAL, NORMAL, NORMAL, NORMAL]
    assert of(result, "T03", UPWIND_COL) == ["T01; T02", "T01; T02", NONE, "T01; T02"]
    assert of(result, "T03", WAKING_SCENARIO_COL) == [
        "T01 not waking; T02 part waking",
        NORMAL,
        NORMAL,
        UNKNOWN,
    ]


def test_no_direction_is_unknown() -> None:
    layout = layout_from_coords(LINE, rotor_diameter_m=DIAMETER)
    scada = frame({t: [(np.nan, False, WAKING)] for t in LINE})
    result = add_waking_scenarios(scada, columns=COLS, layout=layout, test_turbines=[])
    assert set(result[UPWIND_COL]) == {UNKNOWN}
    assert set(result[WAKING_SCENARIO_COL]) == {UNKNOWN}


def test_a_layout_turbine_without_scada_is_missing() -> None:
    layout = layout_from_coords(LINE, rotor_diameter_m=DIAMETER)
    scada = frame({"T01": [(270.0, True, WAKING)], "T03": [(270.0, True, WAKING)]})
    result = add_waking_scenarios(scada, columns=COLS, layout=layout, test_turbines=[])
    assert of(result, "T03", UPWIND_COL) == ["T01; T02"]
    assert of(result, "T03", WAKING_SCENARIO_COL) == [UNKNOWN]


def test_the_input_frame_is_unchanged() -> None:
    layout = layout_from_coords(LINE, rotor_diameter_m=DIAMETER)
    scada = frame({t: [(270.0, True, WAKING)] for t in LINE})
    before = scada.copy()
    add_waking_scenarios(scada, columns=COLS, layout=layout, test_turbines=[])
    pd.testing.assert_frame_equal(scada, before)


def test_outputs_are_written(tmp_path: Path) -> None:
    layout = layout_from_coords(LINE, rotor_diameter_m=DIAMETER)
    scada = frame(
        {
            "T01": [(270.0, True, NOT_WAKING), (90.0, True, WAKING), (np.nan, False, WAKING)],
            "T02": [(270.0, True, PART_WAKING), (90.0, True, WAKING), (np.nan, False, WAKING)],
            "T03": [(275.0, True, WAKING), (95.0, True, WAKING), (np.nan, False, WAKING)],
        }
    )
    directions = waking_directions(scada, columns=COLS, layout=layout, test_turbines=["T02"])
    labelled = add_waking_scenarios(scada, columns=COLS, layout=layout, test_turbines=["T02"], directions=directions)
    written = write_waking_outputs(
        labelled,
        directions=directions,
        layout=layout,
        columns=COLS,
        timebase=pd.Timedelta(minutes=10),
        out_dir=tmp_path,
    )
    assert all(p.exists() for p in written)
    hours = pd.read_csv(tmp_path / SCENARIO_HOURS_CSV)
    assert hours["hours"].sum() == pytest.approx(9 / 6)
    direction = pd.read_csv(tmp_path / DIRECTION_CSV, index_col="turbine")
    assert direction.loc["T02", "hours_own"] == 0.0
    assert direction.loc["T01", "median_abs_own_minus_neighbours_deg"] == pytest.approx(5.0)


def test_directions_built_from_another_frame_are_refused() -> None:
    layout = layout_from_coords(LINE, rotor_diameter_m=DIAMETER)
    shorter = frame(
        {
            "T01": [(270.0, True, NOT_WAKING), (90.0, True, WAKING)],
            "T02": [(270.0, True, PART_WAKING), (90.0, True, WAKING)],
            "T03": [(275.0, True, WAKING), (95.0, True, WAKING)],
        }
    )
    scada = frame(
        {
            "T01": [(270.0, True, NOT_WAKING), (90.0, True, WAKING), (100.0, True, WAKING)],
            "T02": [(270.0, True, PART_WAKING), (90.0, True, WAKING), (100.0, True, WAKING)],
            "T03": [(275.0, True, WAKING), (95.0, True, WAKING), (105.0, True, WAKING)],
        }
    )
    directions = waking_directions(shorter, columns=COLS, layout=layout, test_turbines=["T02"])
    with pytest.raises(ValueError, match="directions"):
        add_waking_scenarios(scada, columns=COLS, layout=layout, test_turbines=["T02"], directions=directions)
