"""Real campaigns declared from open data, written as campaign folders ``wind-up`` can run.

Write the Hill of Towie AeroUp T13 campaign::

    python -m benchmarking.campaigns.real OUT_DIR
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from benchmarking.synthetic.sources.hill_of_towie import (
    HOT_COORDINATES,
    HOT_ROTOR_DIAMETER_M,
    ensure_hot_data_files,
    get_data_dir,
    load_hot_scada,
)

logger = logging.getLogger(__name__)

HOT_AEROUP_WORKS = "Hill_of_Towie_AeroUp_install_dates.csv"
HOT_AEROUP_T13_START = pd.Timestamp("2019-01-01", tz="UTC")
HOT_AEROUP_T13_END = pd.Timestamp("2023-01-01", tz="UTC")

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
}


def write_hot_aeroup_t13(
    out_dir: Path,
    *,
    start: pd.Timestamp = HOT_AEROUP_T13_START,
    end: pd.Timestamp = HOT_AEROUP_T13_END,
    data_dir: Path | None = None,
) -> Path:
    """Write the Hill of Towie AeroUp T13 campaign folder under ``out_dir`` and return its declaration.

    Writes ``campaign.yaml`` and, under ``data/``, every turbine's SCADA over ``[start, end)``, the
    turbines file and the published AeroUp works table.

    :param out_dir: the campaign folder
    :param start: the first SCADA record written
    :param end: the end of the SCADA written, exclusive
    :param data_dir: the Hill of Towie download cache; defaults to its usual place
    """
    data_dir = data_dir if data_dir is not None else get_data_dir()
    data = out_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    scada, _ = load_hot_scada(start_dt=start, end_dt_excl=end, data_dir=data_dir)
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
