"""Write the Hill of Towie ``north_farm`` input fixture the real-data northing tests run on.

The fixture is exactly what :func:`benchmarking.baselines.study_wake_nadir_golden.hot_inputs` hands
``north_farm`` in ``study_northing_degradation`` -- every turbine's yaw, power and nacelle wind
speed, the ``yaw_usable`` mask and the ERA5 direction -- for 2017-2020 (two two-year test windows), so a test can
replay a study case end to end, wake-nadir-shift included, without the raw open-data download.

One wide frame on the 10-minute index, integer-scaled to keep the git-lfs object small (~15 MB):
``reference_decideg`` plus ``<turbine>_yaw_decideg``, ``<turbine>_power_kw`` and
``<turbine>_ws_cm_s`` per turbine, as nullable int16. A turbine's signals are kept only on its
``yaw_usable`` rows and blanked elsewhere -- ``north_farm`` reads nothing off those rows, wake-nadir-shift
included -- so a turbine's usable mask is simply where its yaw is present, and rows where no turbine
is usable are dropped.

    uv run python -m benchmarking.baselines.make_northing_farm_fixture
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarking.baselines.study_wake_nadir_golden import hot_inputs

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "test_data" / "hot" / "northing" / "northing_farm_inputs.parquet"
WINDOW = ("2017-01-01", "2021-01-01")

logger = logging.getLogger(__name__)


# column suffix -> (north_farm input, scale to the stored integer)
SIGNALS = {"yaw_decideg": ("direction", 10.0), "power_kw": ("power", 1.0), "ws_cm_s": ("wind_speed", 100.0)}


def _scaled(values: np.ndarray, scale: float, keep: np.ndarray | None = None) -> pd.arrays.IntegerArray:
    scaled = np.round(np.asarray(values, dtype=float) * scale)
    if keep is not None:
        scaled = np.where(keep, scaled, np.nan)
    return pd.array(scaled, dtype="Float64").astype("Int16")


def fixture_frame(inputs: dict) -> pd.DataFrame:
    """Flatten ``north_farm`` inputs into the fixture's wide, integer-scaled frame."""
    columns = {"reference_decideg": _scaled(inputs["reference"], 10.0)}
    for turbine in sorted(inputs["direction"]):
        usable = np.asarray(inputs["usable"][turbine], dtype=bool)
        for suffix, (field, scale) in SIGNALS.items():
            columns[f"{turbine}_{suffix}"] = _scaled(inputs[field][turbine], scale, usable)
    frame = pd.DataFrame(columns, index=pd.DatetimeIndex(inputs["index"], name="timestamp"))
    return frame[frame.filter(like="_yaw_").notna().any(axis=1)]


def main() -> None:
    """Load the study's Hill of Towie inputs for ``WINDOW`` and write the fixture."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _, inputs = hot_inputs(start=WINDOW[0], end=WINDOW[1])
    frame = fixture_frame(inputs)
    frame.to_parquet(FIXTURE, compression="zstd", compression_level=19)
    logger.info("wrote %s: %d rows x %d columns, %.1f MB", FIXTURE, *frame.shape, FIXTURE.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
