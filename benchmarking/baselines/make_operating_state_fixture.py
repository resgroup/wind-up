"""Write the Hill of Towie operating-state fixture and its golden tables, for the real-data labelling tests.

The fixture is every turbine's SCADA columns that :func:`benchmarking.campaigns.real.label_hot_operating_states`
reads, for three months either side of the Balancing Mechanism start (winter included), so it holds the Hill of
Towie labels: stops before and after the BM start, BM curtailment, noise modes and winter icing. Values are stored as
float32 (~10 MB under git-lfs); equal float64 values stay equal, so the stuck-record rule reads the same.

The golden tables are what an analyst reads off ``02_operating_states``: hours per turbine, month and state with each
state's validity (:func:`state_hours_table`), and the label checks of :func:`label_checks_table`. Record them with
the code whose labels are the reference -- run from a checkout of that code -- and the tests then hold every later
change to them:

    uv run python -m benchmarking.baselines.make_operating_state_fixture            # fixture and golden tables
    uv run python benchmarking/baselines/make_operating_state_fixture.py --golden-only
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarking.campaigns.real import label_hot_operating_states
from benchmarking.harness.operating_state import (
    FULL_DOWNTIME,
    NORMAL_OPERATION,
    STATE_COL,
    VALID_NORTHING_COL,
    VALID_UPLIFT_COL,
    WAKING_STATE_COL,
)
from benchmarking.synthetic.sources.hill_of_towie import (
    HOT_ACTIVE_POWER_MAX_COL,
    HOT_COLUMNS,
    HOT_POWER_SETPOINT_COL,
    HOT_TURBINE_COL,
    TIMEBASE_S,
    load_hot_scada,
)

REPO = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO / "tests" / "test_data" / "hot" / "operating_states"
FIXTURE = FIXTURE_DIR / "operating_state_inputs.parquet"
GOLDEN_HOURS = FIXTURE_DIR / "golden_operating_state_hours.csv"
GOLDEN_CHECKS = FIXTURE_DIR / "golden_label_checks.csv"
WINDOW = ("2018-08-01", "2019-02-01")

# A parked rotor counted as running, and a running rotor counted as down (the two mislabels that move results).
PARKED_RPM = 50.0
PARKED_MIN_WIND_MS = 5.0
RUNNING_KW = 100.0

logger = logging.getLogger(__name__)


def fixture_columns() -> list[str]:
    """Return the columns the Hill of Towie labels read: the schema's roles, the power max and the setpoint."""
    roles = [getattr(HOT_COLUMNS, f) for f in HOT_COLUMNS.__dataclass_fields__]
    columns = [c for c in roles if isinstance(c, str)]
    return list(dict.fromkeys([HOT_TURBINE_COL, *columns, HOT_ACTIVE_POWER_MAX_COL, HOT_POWER_SETPOINT_COL]))


def fixture_frame(scada: pd.DataFrame) -> pd.DataFrame:
    """Keep the labels' columns, the numeric ones as float32."""
    out = scada[[c for c in fixture_columns() if c in scada.columns]].copy()
    for col in out.columns:
        if col != HOT_TURBINE_COL:
            out[col] = out[col].astype("float32")
    return out


def load_fixture(path: Path = FIXTURE) -> pd.DataFrame:
    """Return the fixture as SCADA the labels read: float64, UTC period-start index."""
    frame = pd.read_parquet(path)
    numeric = [c for c in frame.columns if c != HOT_TURBINE_COL]
    frame[numeric] = frame[numeric].astype(float)
    frame[HOT_TURBINE_COL] = frame[HOT_TURBINE_COL].astype(str)
    return frame


def state_hours_table(labelled: pd.DataFrame) -> pd.DataFrame:
    """Hours per turbine, calendar month and state, with the state's validity columns."""
    keys = [VALID_NORTHING_COL, WAKING_STATE_COL, VALID_UPLIFT_COL]
    frame = pd.DataFrame(
        {
            "turbine": labelled[HOT_TURBINE_COL].astype(str).to_numpy(),
            "month": pd.DatetimeIndex(labelled.index).strftime("%Y-%m"),
            STATE_COL: labelled[STATE_COL].astype(str).to_numpy(),
            **{k: labelled[k].astype(str).to_numpy() for k in keys},
        }
    )
    counts = frame.groupby(["turbine", "month", STATE_COL, *keys]).size().rename("records").reset_index()
    counts["hours"] = counts.pop("records") * TIMEBASE_S / 3600.0
    return counts.sort_values(["turbine", "month", STATE_COL]).reset_index(drop=True)


def label_checks_table(labelled: pd.DataFrame) -> pd.DataFrame:
    """Hours per turbine of the mislabels that move a result, for an analyst to confirm stay small.

    ``parked_as_normal``: normal operation with the generator at rest in wind that would turn it.
    ``running_as_full_downtime``: full downtime while generating, so counted as waking nothing downstream.
    """
    power = labelled[HOT_COLUMNS.active_power].to_numpy(dtype=float)
    rpm = labelled[HOT_COLUMNS.gen_rpm].to_numpy(dtype=float)
    wind = labelled[HOT_COLUMNS.wind_speed].to_numpy(dtype=float)
    state = labelled[STATE_COL].to_numpy()
    with np.errstate(invalid="ignore"):
        checks = {
            "parked_as_normal": (state == NORMAL_OPERATION) & (rpm < PARKED_RPM) & (wind >= PARKED_MIN_WIND_MS),
            "running_as_full_downtime": (state == FULL_DOWNTIME) & (power > RUNNING_KW),
        }
    frame = pd.DataFrame({"turbine": labelled[HOT_TURBINE_COL].astype(str).to_numpy(), **checks})
    hours = frame.groupby("turbine")[list(checks)].sum() * TIMEBASE_S / 3600.0
    return hours.reset_index()


def write_golden(fixture: pd.DataFrame) -> None:
    """Label the fixture with the installed code and write both golden tables."""
    labelled = label_hot_operating_states(fixture)
    state_hours_table(labelled).to_csv(GOLDEN_HOURS, index=False)
    label_checks_table(labelled).to_csv(GOLDEN_CHECKS, index=False)
    logger.info("Wrote %s and %s", GOLDEN_HOURS, GOLDEN_CHECKS)


def main() -> None:
    """Write the fixture from the open data (unless ``--golden-only``), then the golden tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-only", action="store_true", help="relabel the existing fixture only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.golden_only:
        start, end = (pd.Timestamp(t, tz="UTC") for t in WINDOW)
        scada, _ = load_hot_scada(start_dt=start, end_dt_excl=end)
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        fixture_frame(scada).to_parquet(FIXTURE)
        logger.info("Wrote %s (%d rows, %.1f MB)", FIXTURE, len(scada), FIXTURE.stat().st_size / 1e6)
    write_golden(load_fixture())


if __name__ == "__main__":
    main()
