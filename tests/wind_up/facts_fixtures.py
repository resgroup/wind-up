"""A small wind farm's facts for the facts, analysis and method tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wind_up.facts import Upgrade, WindFarmFacts
from wind_up.schema import ColumnSchema

COLS = ColumnSchema(
    turbine="turbine",
    active_power="power",
    wind_speed="wind_speed",
    availability="available_s",
    pitch="pitch",
)
TIMEBASE = pd.Timedelta(minutes=10)
FULL = TIMEBASE.total_seconds()
RATED = 2000.0
START = pd.Timestamp("2020-01-01", tz="UTC")
NAMES = ("T1", "T2", "T3", "T4")


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def at(hours: float) -> pd.Timestamp:
    """The timestamp ``hours`` after the records start."""
    return START + pd.Timedelta(hours=hours)


def records(n: int = 36, *, turbines: tuple[str, ...] = NAMES) -> pd.DataFrame:
    """``n`` normally operating ten-minute records of each turbine, every signal varying."""
    index = pd.date_range(START, periods=n, freq=TIMEBASE, tz="UTC")
    values = np.arange(n, dtype=float)
    frames = [
        pd.DataFrame(
            {
                COLS.turbine: name,
                COLS.active_power: 1000.0 + 10.0 * i + values,
                COLS.wind_speed: 8.0 + values / 10,
                COLS.availability: FULL,
                COLS.pitch: 1.0 + values / 10,
            },
            index=index,
        )
        for i, name in enumerate(turbines)
    ]
    return pd.concat(frames).sort_index(kind="stable")


def turbine_table(turbines: tuple[str, ...] = NAMES, **columns: object) -> pd.DataFrame:
    """A turbine table on a north-south line, 500 m apart."""
    table = pd.DataFrame(
        {
            "name": list(turbines),
            "latitude": [57.5 + 0.0045 * i for i in range(len(turbines))],
            "longitude": -3.0,
            "rated_power_kw": RATED,
            "rotor_diameter_m": 100.0,
        }
    )
    return table.assign(**columns)


def permanent(
    name: str = "blades", *, turbines: tuple[str, ...] = ("T1",), hours: float = 3.0, **fields: object
) -> Upgrade:
    """A permanent upgrade ``hours`` after the records start."""
    return Upgrade(name=name, turbines=turbines, timing=at(hours), **fields)  # type: ignore[arg-type]


def toggle_frame(n: int = 36, *, block: int = 3, first: int = 6) -> pd.DataFrame:
    """A toggle signal over ``n`` records: neither state for ``first`` records, then off and on in turn."""
    index = pd.date_range(START, periods=n, freq=TIMEBASE, tz="UTC")
    position = np.arange(n) - first
    started = position >= 0
    on = started & ((position // block) % 2 == 1)
    return pd.DataFrame({"toggle_on": on, "toggle_off": started & ~on}, index=index)


def facts(frame: pd.DataFrame | None = None, **fields: object) -> WindFarmFacts:
    """The fixture farm's facts, with ``fields`` replacing the defaults."""
    given: dict[str, object] = {
        "records": records() if frame is None else frame,
        "columns": COLS,
        "timebase": TIMEBASE,
        "turbines": turbine_table(),
        "upgrades": [permanent()],
    }
    given.update(fields)
    return WindFarmFacts(**given)  # type: ignore[arg-type]
