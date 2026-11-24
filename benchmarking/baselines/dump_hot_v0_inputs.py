"""Dump the inputs the published Hill of Towie v0 analyses read, for ``reproduce_hot_v0`` to run again.

Run it by its path, from the root of the hill-of-towie-open-source-analysis repo, in that repo's
environment (with ``res-wind-up`` and its ``hot_open`` package)::

    python WIND_UP/benchmarking/baselines/dump_hot_v0_inputs.py dump-t13 INPUTS
    python WIND_UP/benchmarking/baselines/dump_hot_v0_inputs.py dump-dy INPUTS STEERING_OUTPUT

``WIND_UP`` is this repo's checkout. The module imports nothing from wind-up, whose ``wind_up``
package shares its name with ``res-wind-up``'s.

``STEERING_OUTPUT`` is the reference steering run's output folder (``HOT_dynamic_yaw``): the
collective control run leaves out the records the steering runs used, read from its ``pp_df``
files, and its steer table is copied from there.
"""

from __future__ import annotations

import sys
from functools import reduce
from pathlib import Path

import pandas as pd
import yaml

STEERING_USED_LABEL = "used by the steering runs"
STEER_TABLE = "wake_steer_meta.csv"


def dump_t13(out: Path) -> None:
    """Dump the AeroUp T13 run's inputs: its SCADA, metadata and reanalysis."""
    from hot_open.era5_helpers import get_hot_reanalysis_datasets  # type: ignore[import-not-found]  # noqa: PLC0415
    from hot_open.unpack import (  # type: ignore[import-not-found]  # noqa: PLC0415
        unpack_local_meta_data,
        unpack_local_scada_data_v1,
    )

    out.mkdir(parents=True, exist_ok=True)
    unpack_local_scada_data_v1().to_parquet(out / "scada_df.parquet")
    unpack_local_meta_data().to_parquet(out / "metadata_df.parquet")
    for ds in get_hot_reanalysis_datasets():
        ds.data.to_parquet(out / f"reanalysis_{ds.id}.parquet")


def dump_dy(out: Path, *, steering_output: Path) -> None:
    """Dump the Dynamic Yaw runs' inputs, and the records the collective control run leaves out."""
    sys.path.insert(0, "scripts/wfc_analysis_2026")
    from hot_open.era5_helpers import get_hot_reanalysis_datasets  # noqa: PLC0415
    from hot_open.lidar_helpers import load_hot_lidar_10min_data  # type: ignore[import-not-found]  # noqa: PLC0415
    from hot_open.settings import get_data_dir  # type: ignore[import-not-found]  # noqa: PLC0415
    from hot_open.unpack import unpack_local_meta_data  # noqa: PLC0415
    from uplift_ws import hot_dy_scada_df, hot_dy_toggle_df  # type: ignore[import-not-found]  # noqa: PLC0415

    out.mkdir(parents=True, exist_ok=True)
    scada_df = hot_dy_scada_df()
    scada_df.to_parquet(out / "scada_df.parquet")
    config = yaml.safe_load(
        Path("scripts/wfc_analysis_2026/wind_up_config/HOT_dynamic_yaw.yaml").read_text().replace("!include", "#")
    )
    tests = [t for t in config["test_wtgs"] if t != "T07"]
    refs = [*config["ref_wtgs"], *config["non_wtg_ref_names"]]
    used = pd.Series(data=False, index=range(len(scada_df)))
    for run_dir in steering_output.glob("T[0-9]*_T[0-9]*_*"):
        up, down = run_dir.stem.split("_")[:2]
        ref = "_".join(run_dir.stem.split("_")[2:])
        if up not in tests or down not in tests or ref not in refs:
            continue
        pp = run_dir / "pp_df"
        paths = [pp / f"{wtg}_{ref}_{side}_df.parquet" for side in ("pre", "post") for wtg in (up, down)]
        frames = [pd.read_parquet(p) for p in paths if p.exists()]
        if frames:
            index = reduce(lambda a, b: a.union(b), (f.index for f in frames))
            used |= scada_df.index.isin(index) & scada_df["TurbineName"].isin([up, down]).to_numpy()
    pd.DataFrame({STEERING_USED_LABEL: used.to_numpy()}).to_parquet(out / "steering_used.parquet")
    hot_dy_toggle_df(scada_df).to_parquet(out / "toggle_df.parquet")
    metadata = unpack_local_meta_data(data_dir=get_data_dir(), scada_index_name=scada_df.index.name)
    metadata.to_parquet(out / "metadata_df.parquet")
    for ds in load_hot_lidar_10min_data(
        data_dir=get_data_dir() / "lidar_data", start_dt=scada_df.index.min(), end_dt_excl=scada_df.index.max()
    ):
        ds.data.to_parquet(out / f"lidar_{ds.id}.parquet")
    for ds in get_hot_reanalysis_datasets():
        ds.data.to_parquet(out / f"reanalysis_{ds.id}.parquet")
    (out / STEER_TABLE).write_text((steering_output / STEER_TABLE).read_text())


def main(argv: list[str]) -> None:
    """Run the command ``argv`` names; see the module's documentation."""
    command, rest = argv[0], argv[1:]
    if command == "dump-t13":
        dump_t13(Path(rest[0]))
    elif command == "dump-dy":
        dump_dy(Path(rest[0]), steering_output=Path(rest[1]))
    else:
        msg = f"unknown command {command!r}; see the module's documentation"
        raise SystemExit(msg)


if __name__ == "__main__":
    main(sys.argv[1:])
