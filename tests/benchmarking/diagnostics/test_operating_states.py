"""Operating-state plots: the labelled records drawn for the analyst to confirm the labels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.diagnostics.context import DiagnosticContext
from benchmarking.diagnostics.operating_states import (
    STATE_HOURS_CSV,
    plot_operating_states,
    plot_run_operating_states,
    state_colours,
    state_order,
    write_operating_state_plots,
)
from benchmarking.harness.operating_state import (
    FULL_DOWNTIME,
    MISSING,
    NORMAL_OPERATION,
    PARTIAL_DOWNTIME,
    OperatingStateConfig,
    label_operating_states,
)
from benchmarking.synthetic import ColumnSchema

if TYPE_CHECKING:
    from pathlib import Path

TIMEBASE = pd.Timedelta(minutes=10)
COLUMNS = ColumnSchema(
    turbine="turbine",
    active_power="power",
    wind_speed="ws",
    wind_speed_sd="ws_sd",
    gen_rpm="rpm",
    pitch="pitch",
    availability="avail",
)
NO_PITCH = ColumnSchema(
    turbine="turbine", active_power="power", wind_speed="ws", wind_speed_sd="ws_sd", gen_rpm="rpm", availability="avail"
)


def _farm(turbines: list[str], *, days: int = 70) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    index = pd.date_range("2020-01-01", periods=days * 144, freq="10min", tz="UTC")
    parts = []
    for turbine in turbines:
        ws = rng.uniform(3, 18, len(index))
        parts.append(
            pd.DataFrame(
                {
                    "turbine": turbine,
                    "power": np.clip(0.5 * ws**3, 0, 2300),
                    "ws": ws,
                    "ws_sd": 1.0,
                    "rpm": np.clip(ws * 90, 0, 1600),
                    "pitch": np.clip(ws - 12, 0, None),
                    "avail": rng.choice([0.0, 300.0, 600.0], size=len(index), p=[0.05, 0.05, 0.9]),
                    "state": rng.choice([None, "curtailed"], size=len(index), p=[0.9, 0.1]),
                },
                index=index,
            )
        )
    return pd.concat(parts)


def _labelled(scada: pd.DataFrame, *, columns: ColumnSchema = COLUMNS) -> pd.DataFrame:
    config = OperatingStateConfig(
        label_column="state", labels={"curtailed": OperatingStateConfig().validity()[PARTIAL_DOWNTIME]}
    )
    return label_operating_states(scada, columns=columns, config=config, timebase=TIMEBASE, rated_power_kw=2300.0)


def test_normal_operation_is_drawn_first_and_colours_are_fixed() -> None:
    order = state_order([MISSING, "noise", NORMAL_OPERATION, FULL_DOWNTIME, "curtailed"])
    assert order == [NORMAL_OPERATION, "curtailed", "noise", FULL_DOWNTIME, MISSING]
    assert state_colours(["noise"])["noise"] == state_colours(["noise", MISSING])["noise"]


def test_every_turbine_is_drawn_with_the_hours(tmp_path: Path) -> None:
    folder = tmp_path / "states"
    write_operating_state_plots(_labelled(_farm(["T1", "T2"])), columns=COLUMNS, timebase=TIMEBASE, out_dir=folder)
    assert (folder / "operating_states_T1.png").exists()
    assert (folder / "operating_states_T2.png").exists()
    assert (folder / "operating_state_hours.png").exists()
    hours = pd.read_csv(folder / STATE_HOURS_CSV)
    assert hours["hours"].sum() == 2 * 70 * 24
    assert "curtailed" in set(hours["operating_state"])


def test_an_unlabelled_frame_gets_no_operating_state_plots(tmp_path: Path) -> None:
    assert write_operating_state_plots(_farm(["T1"]), columns=COLUMNS, timebase=TIMEBASE, out_dir=tmp_path) == []


def test_a_turbine_without_pitch_skips_the_pitch_panels(tmp_path: Path) -> None:
    rows = _labelled(_farm(["T1"]).drop(columns="pitch"), columns=NO_PITCH)
    path = plot_operating_states(rows, turbine="T1", columns=NO_PITCH, timebase=TIMEBASE, out_dir=tmp_path)
    assert path is not None
    assert path.exists()


def test_the_run_level_plots_cover_the_test_turbine_and_power_references(tmp_path: Path) -> None:
    scada = _labelled(_farm(["T1", "T2", "T3"], days=10))
    index = pd.DatetimeIndex(pd.unique(scada.index)).sort_values()
    ctx = DiagnosticContext(
        run_dir=tmp_path / "run",
        test_wtg="T1",
        turbine_col="turbine",
        columns=COLUMNS,
        scada_df=scada,
        treated_ts=np.asarray(index >= index[len(index) // 2]),
        used_ts=np.ones(len(index), dtype=bool),
        timebase=TIMEBASE,
        mode="prepost",
        power_references=["T2"],
    )
    written = plot_run_operating_states(ctx)
    assert sorted(p.name for p in written) == ["operating_states_T1.png", "operating_states_T2.png"]
    assert all(p.parent == tmp_path / "run" / "plots" / "02_operating_states" for p in written)
