"""Real campaigns declared from open data, written as campaign folders ``wind-up`` can run.

Write the Hill of Towie AeroUp T13 campaign::

    python -m benchmarking.campaigns.real OUT_DIR
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml

from benchmarking.harness.operating_state import PARTIAL_DOWNTIME
from benchmarking.synthetic.sources.hill_of_towie import (
    HOT_AMBIENT_TEMP_COL,
    HOT_BM_START,
    HOT_COLUMNS,
    HOT_COORDINATES,
    HOT_HIGH_WIND_DERATE_MS,
    HOT_ICING_MAX_TEMP_C,
    HOT_ICING_MIN_BIN_RECORDS,
    HOT_ICING_MIN_EXPECTED_KW,
    HOT_ICING_MIN_RUN_RECORDS,
    HOT_ICING_POWER_FRACTION,
    HOT_ICING_REFERENCE_MIN_TEMP_C,
    HOT_NOISE_SETPOINTS_KW,
    HOT_POWER_SETPOINT_COL,
    HOT_RATED_POWER_KW,
    HOT_ROTOR_DIAMETER_M,
    HOT_STARTUP_SETPOINT_KW,
    HOT_STOP_SETPOINT_KW,
    HOT_TURBINE_COL,
    TIMEBASE_S,
    ensure_hot_data_files,
    get_data_dir,
    load_hot_scada,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

HOT_AEROUP_WORKS = "Hill_of_Towie_AeroUp_install_dates.csv"
HOT_AEROUP_T13_START = pd.Timestamp("2019-01-01", tz="UTC")
HOT_AEROUP_T13_END = pd.Timestamp("2023-01-01", tz="UTC")

# Hill of Towie's site operating states, written into the campaign's SCADA as HOT_STATE_COL.
HOT_STATE_COL = "state"
NOISE_MODE = "noise mode"
BM_CURTAILMENT = "BM curtailment"
HIGH_WIND_DERATE = "high wind derate"
ICING = "icing"
HOT_OPERATING_STATE: dict[str, Any] = {
    "label_column": HOT_STATE_COL,
    "parked_pitch_above_deg": 45,
    "labels": {
        NOISE_MODE: {"northing": "valid", "waking": "waking", "uplift": "not valid"},
        BM_CURTAILMENT: {"northing": "valid", "waking": "part waking", "uplift": "not valid"},
        HIGH_WIND_DERATE: {"northing": "valid", "waking": "part waking", "uplift": "not valid"},
        ICING: {"northing": "not valid", "waking": "part waking", "uplift": "not valid"},
    },
}

# T13's AeroUp retrofit, with every other turbine's AeroUp works in the works table. The one
# exclusion is the farm-wide curtailment period of tests/test_data/hot/HoT_AeroUp_T13.yaml.
HOT_AEROUP_T13: dict[str, Any] = {
    "name": "hot_aeroup_t13",
    "data": {
        "scada": "data/scada.parquet",
        "schema": "hill_of_towie",
        "turbines": "data/turbines.csv",
        "works": "data/works.csv",
    },
    "turbines": {"upgraded": ["T13"], "rated_power_kw": 2300.0},
    "timing": {"mode": "prepost"},
    "exclusions": [{"turbine": "ALL", "start": "2022-09-03T00:00:00Z", "end": "2023-02-12T00:00:00Z"}],
    "northing": {"discover": True},
    "operating_state": HOT_OPERATING_STATE,
}


def label_hot_site_states(scada_long: pd.DataFrame) -> pd.Series:
    """Return each Hill of Towie record's site operating-state label, None where it has none.

    Reads the power setpoint at the end of each record; the setpoint at its start is the turbine's
    previous record's end, NaN after a gap. A record takes, from either end and in this order:

    1. partial downtime: a stop setpoint;
    2. no label: the start-up setpoint;
    3. high wind derate: wind speed at or above ``HOT_HIGH_WIND_DERATE_MS`` and a setpoint below rated
       that is not one of the turbine's noise setpoints;
    4. BM curtailment: from the Balancing Mechanism start, below that wind speed, a setpoint below
       rated that is not one of the turbine's noise setpoints;
    5. noise mode: one of the turbine's noise setpoints.

    A record without a setpoint label is icing when it is cold and produces under a fraction of the
    turbine's warm-weather median power at its wind speed, for a run of consecutive records (the
    ``HOT_ICING_*`` constants).

    :param scada_long: long-format Hill of Towie SCADA, indexed by timestamp
    :return: the labels on ``scada_long``'s index, in its order
    """
    turbine = scada_long[HOT_TURBINE_COL].astype(str).to_numpy()
    stamps = pd.DatetimeIndex(scada_long.index)
    order = np.lexsort((stamps.asi8, pd.factorize(turbine)[0]))
    wtg = turbine[order]
    when = stamps[order]
    end = scada_long[HOT_POWER_SETPOINT_COL].to_numpy(dtype=float)[order]
    start = np.concatenate([[np.nan], end[:-1]])
    follows = np.concatenate([[False], (wtg[1:] == wtg[:-1]) & (np.diff(when.asi8) == TIMEBASE_S * 10**9)])
    start[~follows] = np.nan
    wind_speed = scada_long[HOT_COLUMNS.wind_speed].to_numpy(dtype=float)[order]
    high_wind = wind_speed >= HOT_HIGH_WIND_DERATE_MS

    def noise(setpoint: np.ndarray) -> np.ndarray:
        hit = np.zeros(len(setpoint), dtype=bool)
        for name, steps in HOT_NOISE_SETPOINTS_KW.items():
            hit |= (wtg == name) & np.isin(setpoint, list(steps))
        return hit

    def reduced(setpoint: np.ndarray) -> np.ndarray:
        return (setpoint < HOT_RATED_POWER_KW) & ~noise(setpoint)

    def derated(setpoint: np.ndarray) -> np.ndarray:
        return high_wind & reduced(setpoint)

    def curtailed(setpoint: np.ndarray) -> np.ndarray:
        return (when >= HOT_BM_START) & ~high_wind & reduced(setpoint)

    def either(rule: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        return rule(end) | rule(start)

    # Lowest precedence first, so a higher rule overwrites.
    labels = np.full(len(end), None, dtype=object)
    labels[either(noise)] = NOISE_MODE
    labels[either(curtailed)] = BM_CURTAILMENT
    labels[either(derated)] = HIGH_WIND_DERATE
    labels[either(lambda s: s == HOT_STARTUP_SETPOINT_KW)] = None
    labels[either(lambda s: s == HOT_STOP_SETPOINT_KW)] = PARTIAL_DOWNTIME
    labels[_iced(scada_long, order=order, follows=follows, unlabelled=pd.isna(labels))] = ICING
    result = np.empty(len(end), dtype=object)
    result[order] = labels
    return pd.Series(result, index=scada_long.index, name=HOT_STATE_COL)


def _iced(scada_long: pd.DataFrame, *, order: np.ndarray, follows: np.ndarray, unlabelled: np.ndarray) -> np.ndarray:
    """Return which of the ``order``-sorted records are icing, among the ``unlabelled`` ones.

    ``follows`` marks the records that directly follow the previous one of the same turbine.
    """
    rows = pd.DataFrame(
        {
            "turbine": scada_long[HOT_TURBINE_COL].astype(str).to_numpy()[order],
            "bin": np.round(scada_long[HOT_COLUMNS.wind_speed].to_numpy(dtype=float)[order] * 2) / 2,
            "power": scada_long[HOT_COLUMNS.active_power].to_numpy(dtype=float)[order],
        }
    )
    ambient = scada_long[HOT_AMBIENT_TEMP_COL].to_numpy(dtype=float)[order]
    warm = unlabelled & (ambient > HOT_ICING_REFERENCE_MIN_TEMP_C)
    curve = rows[warm].groupby(["turbine", "bin"])["power"].agg(["median", "count"])
    curve = curve.loc[curve["count"] >= HOT_ICING_MIN_BIN_RECORDS, "median"].rename("expected")
    expected = rows.join(curve, on=["turbine", "bin"])["expected"].to_numpy()
    with np.errstate(invalid="ignore"):
        low = (
            unlabelled
            & (ambient <= HOT_ICING_MAX_TEMP_C)
            & (expected >= HOT_ICING_MIN_EXPECTED_KW)
            & (rows["power"].to_numpy() < HOT_ICING_POWER_FRACTION * expected)
        )
    continues = follows & np.concatenate([[False], low[:-1]])
    run = np.cumsum(~continues)
    run_length = pd.Series(low).groupby(run).transform("sum").to_numpy()
    return low & (run_length >= HOT_ICING_MIN_RUN_RECORDS)


def write_hot_aeroup_t13(
    out_dir: Path,
    *,
    start: pd.Timestamp = HOT_AEROUP_T13_START,
    end: pd.Timestamp = HOT_AEROUP_T13_END,
    data_dir: Path | None = None,
) -> Path:
    """Write the Hill of Towie AeroUp T13 campaign folder under ``out_dir`` and return its declaration.

    Writes ``campaign.yaml`` and, under ``data/``, every turbine's SCADA over ``[start, end)`` with its
    site operating-state labels, the turbines file and the published AeroUp works table.

    :param out_dir: the campaign folder
    :param start: the first SCADA record written
    :param end: the end of the SCADA written, exclusive
    :param data_dir: the Hill of Towie download cache; defaults to its usual place
    """
    data_dir = data_dir if data_dir is not None else get_data_dir()
    data = out_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    scada, _ = load_hot_scada(start_dt=start, end_dt_excl=end, data_dir=data_dir)
    scada[HOT_STATE_COL] = label_hot_site_states(scada)
    scada.to_parquet(data / "scada.parquet")
    pd.DataFrame(
        {
            "name": list(HOT_COORDINATES),
            "latitude": [lat for lat, _ in HOT_COORDINATES.values()],
            "longitude": [lon for _, lon in HOT_COORDINATES.values()],
            "rotor_diameter_m": HOT_ROTOR_DIAMETER_M,
        }
    ).to_csv(data / "turbines.csv", index=False)
    ensure_hot_data_files([HOT_AEROUP_WORKS], data_dir=data_dir)
    shutil.copyfile(data_dir / HOT_AEROUP_WORKS, data / "works.csv")
    path = out_dir / "campaign.yaml"
    path.write_text(yaml.safe_dump(HOT_AEROUP_T13, sort_keys=False))
    logger.info("Wrote the Hill of Towie AeroUp T13 campaign to %s", out_dir)
    return path


def main(argv: list[str] | None = None) -> None:
    """Write the Hill of Towie AeroUp T13 campaign folder."""
    parser = argparse.ArgumentParser(prog="python -m benchmarking.campaigns.real", description=__doc__)
    parser.add_argument("out_dir", type=Path, help="the campaign folder to write")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    write_hot_aeroup_t13(args.out_dir)


if __name__ == "__main__":
    main()
