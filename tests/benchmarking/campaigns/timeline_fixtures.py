"""A small staggered campaign on a line of turbines, for the planning and context tests."""

from __future__ import annotations

import dataclasses

import pandas as pd

from benchmarking.campaigns import CampaignSpec, layout_from_coords
from benchmarking.synthetic import HOT_COLUMNS
from tests.wind_up.layouts import line_layout

N_TURBINES = 7
SPACING_M = 250.0  # about 3 rotor diameters of 82 m, so T6 is within 20 D of T0
ROTOR_DIAMETER_M = 82.0
DATA_START = pd.Timestamp("2018-01-01", tz="UTC")
DATA_END = pd.Timestamp("2021-01-01", tz="UTC")


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def names() -> list[str]:
    return [f"T{i}" for i in range(N_TURBINES)]


def staggered_spec(**overrides: object) -> CampaignSpec:
    """T0 and T6 analysed at the two ends of the line; T0 worked 2019-06, T6 worked 2019-09."""
    frame = line_layout([i * SPACING_M for i in range(N_TURBINES)], rotor_diameter_m=ROTOR_DIAMETER_M)
    coords = {
        str(n): (float(lat), float(lon))
        for n, lat, lon in zip(frame["name"], frame["latitude"], frame["longitude"], strict=True)
    }
    spec = CampaignSpec(
        upgraded_turbines=["T0", "T6"],
        upgrade_timing=None,
        candidate_references=["T1", "T2", "T3", "T4", "T5"],
        excluded_turbines=[],
        layout=layout_from_coords(coords, rotor_diameter_m=ROTOR_DIAMETER_M),
        north_offsets=None,
        rated_power_kw=2300.0,
        analysis_period=None,
        works={
            "T0": [(utc("2019-06-01"), utc("2019-06-06"))],
            "T6": [(utc("2019-09-01"), utc("2019-09-04"))],
        },
        exclusions=[("T4", utc("2019-02-01"), utc("2019-02-05"))],
    )
    return dataclasses.replace(spec, **overrides)


def hourly_scada(turbines: list[str] | None = None) -> pd.DataFrame:
    """Flat, fully available hourly SCADA for every turbine over the data extent."""
    index = pd.date_range(DATA_START, DATA_END, freq="1h", inclusive="left")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    HOT_COLUMNS.turbine: wtg,
                    HOT_COLUMNS.active_power: 900.0,
                    HOT_COLUMNS.active_power_min: 850.0,
                    HOT_COLUMNS.wind_speed: 8.0,
                    HOT_COLUMNS.wind_speed_sd: 0.8,
                    HOT_COLUMNS.gen_rpm: 1400.0,
                    HOT_COLUMNS.availability: 3600.0,
                    HOT_COLUMNS.nacelle_position: 180.0,
                },
                index=index,
            )
            for wtg in (turbines or names())
        ]
    )
