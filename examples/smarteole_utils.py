"""Read the SMARTEOLE open dataset into wind-up's long-format tables, keeping the dataset's own column names."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SCADA_FILE_PATH = "SMARTEOLE-WFC-open-dataset/SMARTEOLE_WakeSteering_SCADA_1minData.csv"
METADATA_FILE_PATH = "SMARTEOLE-WFC-open-dataset/SMARTEOLE_WakeSteering_Coordinates_staticData.csv"
TOGGLE_FILE_PATH = "SMARTEOLE-WFC-open-dataset/SMARTEOLE_WakeSteering_ControlLog_1minData.csv"

TIMESTAMP = "timestamp"
TURBINE = "turbine"
NOT_DERATED = "not_derated_s"
MINIMUM_DATA_COUNT_COVERAGE = 0.5  # a field is kept where at least half its samples are present

# the fields kept from the 1-minute SCADA, with how each is aggregated to the analysis timebase
# and the count column that decides whether the aggregate is kept
SCADA_FIELDS: dict[str, tuple[str, str]] = {
    "active_power_avg": ("mean", "active_power_count"),
    "active_power_std": ("mean", "active_power_count"),
    "wind_speed_avg": ("mean", "wind_speed_count"),
    "wind_speed_std": ("mean", "wind_speed_count"),
    "nacelle_position_avg": ("circular_mean", "nacelle_position_count"),
    "nacelle_position_min": ("min", "nacelle_position_count"),
    "nacelle_position_max": ("max", "nacelle_position_count"),
    "blade_1_pitch_angle_avg": ("mean", "blade_1_pitch_angle_count"),
    "generator_speed_avg": ("mean", "generator_speed_count"),
    "temperature_avg": ("mean", "temperature_count"),
}


def _split_field(column: str) -> tuple[str, str]:
    """Split a wide SCADA column such as ``active_power_6_avg`` into ``("SMV6", "active_power_avg")``."""
    parts = column.split("_")
    if len(parts[-1]) == 1:
        return f"SMV{parts[-1]}", "_".join(parts[:-1])
    return f"SMV{parts[-2]}", "_".join([*parts[:-2], parts[-1]])


class SmartEoleExtractor:
    """Read the SMARTEOLE SCADA, turbine coordinates and wake steering control log at an analysis timebase.

    Every table has a UTC ``timestamp`` index marking the start of each record.
    """

    def __init__(self, data_dir: Path, *, analysis_timebase_s: int = 600) -> None:
        self.data_dir = data_dir
        self.analysis_timebase_s = analysis_timebase_s

    @classmethod
    def from_zip_file(cls, zipped_fpath: Path, *, analysis_timebase_s: int = 600) -> SmartEoleExtractor:
        """Extract the three data files next to the zip file and read them from there."""
        output_dir = zipped_fpath.parent
        with zipfile.ZipFile(zipped_fpath) as zf:
            for fname in (SCADA_FILE_PATH, METADATA_FILE_PATH, TOGGLE_FILE_PATH):
                (output_dir / Path(fname).name).write_bytes(zf.read(fname))
        return cls(data_dir=output_dir, analysis_timebase_s=analysis_timebase_s)

    def unpack_smarteole_scada(self) -> pd.DataFrame:
        """Return one row per turbine and record: ``turbine``, the fields of ``SCADA_FIELDS`` and ``not_derated_s``.

        A field is NaN where fewer than half its 1-minute samples are present. ``not_derated_s`` is the
        record's length less 60 s for each minute the dataset flags as derated.
        """
        wide = pd.read_csv(self.data_dir / Path(SCADA_FILE_PATH).name, parse_dates=[0], index_col=0)
        wide.columns = pd.MultiIndex.from_tuples((_split_field(c) for c in wide.columns), names=[TURBINE, "field"])
        long = wide.stack(level=0, future_stack=True).reset_index(TURBINE)  # noqa: PD013
        circular = [field for field, (how, _) in SCADA_FIELDS.items() if how == "circular_mean"]
        for field in circular:
            radians = np.deg2rad(long[field])
            long[f"{field}_sin"], long[f"{field}_cos"] = np.sin(radians), np.cos(radians)
            long[f"{field}_missing"] = long[field].isna()
        counts = sorted({count for _, count in SCADA_FIELDS.values()})
        aggregations = (
            {field: how for field, (how, _) in SCADA_FIELDS.items() if how != "circular_mean"}
            | {f"{field}_{part}": "mean" for field in circular for part in ("sin", "cos")}
            | {f"{field}_missing": "sum" for field in circular}
            | dict.fromkeys(counts, "sum")
            | {"derate": "sum"}
        )
        records = (
            long.groupby(TURBINE).resample(f"{self.analysis_timebase_s}s").aggregate(aggregations).reset_index(TURBINE)
        )
        for field in circular:  # NaN if any sample in the record is missing
            angle = np.rad2deg(np.arctan2(records[f"{field}_sin"], records[f"{field}_cos"])) % 360
            records[field] = angle.mask(records[f"{field}_missing"] > 0)
        minimum_count = self.analysis_timebase_s * MINIMUM_DATA_COUNT_COVERAGE
        for field, (_, count) in SCADA_FIELDS.items():
            records[field] = records[field].mask(records[count] < minimum_count)
        records[NOT_DERATED] = self.analysis_timebase_s - 60.0 * records["derate"]
        records.index = records.index.tz_localize("UTC")
        return records.loc[:, [TURBINE, *SCADA_FIELDS, NOT_DERATED]].rename_axis(TIMESTAMP).rename_axis(None, axis=1)

    def unpack_smarteole_metadata(self) -> pd.DataFrame:
        """Return the turbines' ``Turbine``, ``Latitude`` and ``Longitude``."""
        coordinates = pd.read_csv(self.data_dir / Path(METADATA_FILE_PATH).name)
        turbines = coordinates[coordinates["Turbine"].str.startswith("SMV")]
        return turbines.loc[:, ["Turbine", "Latitude", "Longitude"]].reset_index(drop=True)

    def unpack_smarteole_toggle_data(self) -> pd.DataFrame:
        """Return the wake steering toggle frame: ``toggle_on``, ``toggle_off`` and ``yaw_offset_command``.

        A record is on (off) when steering was active for at least 95 % (at most 5 %) of the
        control log samples, and at least half the samples are present.
        """
        minimum_count = self.analysis_timebase_s * MINIMUM_DATA_COUNT_COVERAGE
        toggle_value_threshold = 0.95
        log = pd.read_csv(self.data_dir / Path(TOGGLE_FILE_PATH).name, parse_dates=[0], index_col=0)
        toggle = log.resample(f"{self.analysis_timebase_s}s").agg(
            {
                "control_log_offset_active_avg": "mean",
                "control_log_offset_active_count": "sum",
                "control_log_offset_avg": "mean",
            }
        )
        enough = toggle["control_log_offset_active_count"] >= minimum_count
        active = toggle["control_log_offset_active_avg"]
        toggle["toggle_on"] = (active >= toggle_value_threshold) & enough
        toggle["toggle_off"] = (active <= (1 - toggle_value_threshold)) & enough
        toggle["yaw_offset_command"] = toggle["control_log_offset_avg"]
        toggle.index = toggle.index.tz_localize("UTC")
        return toggle.loc[:, ["toggle_on", "toggle_off", "yaw_offset_command"]].rename_axis(TIMESTAMP)
