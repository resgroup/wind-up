"""Build a wind-up campaign input (Greenbyte-schema long parquet + turbines.csv) from PAWP."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from benchmarking.pawp_trial.read import WORK_DIR, read_wide
from benchmarking.synthetic.sources import greenbyte as gb

# Table A1 of the data paper (WGS84, foundation centre).
COORDS = {
    "G001": (52.60457, 4.24072), "G002": (52.60516, 4.22669), "G003": (52.60240, 4.23342),
    "G004": (52.59964, 4.24016), "G005": (52.59689, 4.24689), "G006": (52.60537, 4.21240),
    "G007": (52.60269, 4.21922), "G008": (52.60000, 4.22604), "G009": (52.59732, 4.23285),
    "G010": (52.59463, 4.23967), "G011": (52.59194, 4.24649), "G012": (52.58926, 4.25330),
    "G013": (52.58657, 4.26012), "G014": (52.60249, 4.20467), "G015": (52.59989, 4.21157),
    "G016": (52.59728, 4.21847), "G017": (52.59467, 4.22538), "G018": (52.59207, 4.23227),
    "G019": (52.58945, 4.23917), "G020": (52.58684, 4.24606), "G021": (52.58424, 4.25295),
    "G022": (52.58163, 4.25985), "G023": (52.59943, 4.19696), "G024": (52.59690, 4.20393),
    "G025": (52.59438, 4.21093), "G026": (52.59186, 4.21789), "G027": (52.58933, 4.22488),
    "G028": (52.58680, 4.23185), "G029": (52.58428, 4.23882), "G030": (52.58174, 4.24580),
    "G031": (52.57921, 4.25277), "G032": (52.59385, 4.19641), "G033": (52.59140, 4.20345),
    "G034": (52.58896, 4.21050), "G035": (52.58651, 4.21756), "G036": (52.58406, 4.22461),
    "G037": (52.58162, 4.23166), "G038": (52.57917, 4.23872), "G039": (52.57672, 4.24576),
    "G040": (52.57427, 4.25282), "G041": (52.59070, 4.18879), "G042": (52.58834, 4.19593),
    "G043": (52.58598, 4.20305), "G044": (52.58361, 4.21018), "G045": (52.58124, 4.21731),
    "G046": (52.57888, 4.22444), "G047": (52.57651, 4.23156), "G048": (52.57414, 4.23868),
    "G049": (52.57178, 4.24581), "G050": (52.58529, 4.18859), "G051": (52.58300, 4.19579),
    "G052": (52.58072, 4.20298), "G053": (52.57843, 4.21017), "G054": (52.57615, 4.21736),
    "G055": (52.57385, 4.22456), "G056": (52.57156, 4.23176), "G057": (52.58012, 4.18874),
    "G058": (52.57791, 4.19600), "G059": (52.57570, 4.20326), "G060": (52.57349, 4.21052),
}  # fmt: skip
ROTOR_DIAMETER_M = 80.0
INT16_SENTINEL = 3276.0  # |value| >= this is a scaled int16 overflow (+-32768 / 10)
DIRECTION_SENTINEL = 1000.0


def _clean(frame: pd.DataFrame, *, direction: bool = False) -> pd.DataFrame:
    out = frame.mask(frame.abs() >= INT16_SENTINEL)
    if direction:
        out = out.mask((out >= DIRECTION_SENTINEL) | (out < 0)) % 360
    return out


def build(start: str, end: str, name: str, *, strict_availability: bool = True) -> None:
    window = slice(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") - pd.Timedelta(minutes=1))
    wide = {
        gb.POWER: _clean(read_wide("Power_mean")),
        gb.POWER_MIN: _clean(read_wide("Power_min")),
        gb.WIND_SPEED: read_wide("WindSpeed_mean"),
        gb.WIND_SPEED_SD: read_wide("WindSpeed_std"),
        gb.GEN_RPM: read_wide("GeneratorSpeed_mean"),
        gb.PITCH: _clean(read_wide("PitchAngle_mean")),
        gb.NACELLE_POSITION: _clean(read_wide("NacelleDirection_mean"), direction=True),
        gb.AMBIENT_TEMP: _clean(read_wide("AmbientTemperature_mean")),
        "OperatingState": read_wide("OperatingState"),
        "FirstAlarm10min": read_wide("FirstAlarm10min"),
        "WindDirection": _clean(read_wide("WindDirection_mean"), direction=True),
    }
    wide = {k: v.loc[window] for k, v in wide.items()}
    state = wide["OperatingState"]
    alarm = wide["FirstAlarm10min"]
    # No ready-to-operate counter is published: a record counts as fully available when the turbine
    # reports running (state 1-3), or idling (state 0) with no alarm; from 2015 alarm stops in
    # strong wind are reported as state 0.
    ready = (state.ge(0.5) & state.le(3.5)) | (state.eq(0) & alarm.eq(0)) if strict_availability else state.lt(4)
    wide[gb.AVAILABILITY] = pd.DataFrame(
        np.where(state.isna(), np.nan, np.where(ready, 600.0, 0.0)), index=state.index, columns=state.columns
    )
    wide[gb.REACTIVE_POWER] = state * np.nan
    long = pd.concat({k: v.stack(future_stack=True) for k, v in wide.items()}, axis=1)
    long.index.names = ["TimeStamp_EndFormat", gb.TURBINE]
    long = long.reset_index(level=gb.TURBINE).astype({k: "float32" for k in wide})
    out = WORK_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out / "scada.parquet")
    pd.DataFrame(
        [{"Name": t, "Latitude": la, "Longitude": lo, "Rotor_Diameter_m": ROTOR_DIAMETER_M} for t, (la, lo) in COORDS.items()]
    ).to_csv(out / "turbines.csv", index=False)
    print(f"wrote {len(long)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--name", required=True)
    a = parser.parse_args()
    build(a.start, a.end, a.name)
