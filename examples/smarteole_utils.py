"""
Helper code bespoke to the SMARTEOLE dataset
"""

from __future__ import annotations

import logging
import zipfile
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Union

import pandas as pd
from scipy.stats import circmean

from wind_up.caching import with_parquet_cache
from wind_up.constants import PROJECTROOT_DIR, TIMESTAMP_COL, DataColumns

if TYPE_CHECKING:
    FilePathOrBuffer = Union[str, bytearray]

SCADA_FILE_PATH = "SMARTEOLE-WFC-open-dataset/SMARTEOLE_WakeSteering_SCADA_1minData.csv"
METADATA_FILE_PATH = "SMARTEOLE-WFC-open-dataset/SMARTEOLE_WakeSteering_Coordinates_staticData.csv"
TOGGLE_FILE_PATH = "SMARTEOLE-WFC-open-dataset/SMARTEOLE_WakeSteering_ControlLog_1minData.csv"
CACHE_DIR = PROJECTROOT_DIR / "cache" / "smarteole_example_data"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

MINIMUM_DATA_COUNT_COVERAGE = 0.5  # 50% of the data must be present

logger = logging.getLogger(__name__)


class SmartEoleExtractor:
    def __init__(self, data_dir: Path, analysis_timebase_s: int = 600):
        self.data_dir = data_dir
        self.analysis_timebase_s = analysis_timebase_s

    @classmethod
    def from_zip_file(cls, zipped_fpath: Path, analysis_timebase_s: int = 600) -> SmartEoleExtractor:
        # unzip specific data files in the same folder as the zip file
        output_dir = zipped_fpath.parent
        with zipfile.ZipFile(zipped_fpath) as zf:
            for fname in (SCADA_FILE_PATH, METADATA_FILE_PATH, TOGGLE_FILE_PATH):
                with Path.open(output_dir / Path(fname).name, "wb") as f:
                    f.write(zf.read(fname))
        return cls(data_dir=output_dir, analysis_timebase_s=analysis_timebase_s)

    @with_parquet_cache(CACHE_DIR / "smarteole_scada.parquet")
    def unpack_smarteole_scada(self) -> pd.DataFrame:
        """Function that translates 1-minute SCADA data to x minute data in the wind-up expected format"""

        def _separate_turbine_id_from_field(x: str) -> tuple[str, str]:
            parts = x.split("_")
            if len(parts[-1]) == 1:
                wtg_id = parts[-1]
                col_name = "_".join(parts[:-1])
            else:
                wtg_id = parts[-2]
                col_name = "_".join(parts[:-2] + [parts[-1]])
            return f"SMV{wtg_id}", col_name

        def _make_turbine_id_a_column(df: pd.DataFrame) -> pd.DataFrame:
            df.columns = pd.MultiIndex.from_tuples(
                (_separate_turbine_id_from_field(i) for i in df.columns),
                names=[DataColumns.turbine_name, "field"],
            )
            return df.stack(level=0, future_stack=True).reset_index(DataColumns.turbine_name)  # noqa: PD013

        def _map_and_mask_cols(df: pd.DataFrame) -> pd.DataFrame:
            x_minutes_count_lower_limit = self.analysis_timebase_s * MINIMUM_DATA_COUNT_COVERAGE
            mask_active_power = df["active_power_count"] < x_minutes_count_lower_limit
            mask_wind_speed = df["wind_speed_count"] < x_minutes_count_lower_limit
            mask_pitch_angle = df["blade_1_pitch_angle_count"] < x_minutes_count_lower_limit
            mask_gen_rpm = df["generator_speed_count"] < x_minutes_count_lower_limit
            mask_temperature = df["temperature_count"] < x_minutes_count_lower_limit
            mask_nacelle_position = df["nacelle_position_count"] < x_minutes_count_lower_limit
            return df.assign(
                **{
                    DataColumns.active_power_mean: lambda d: d["active_power_avg"].mask(mask_active_power),
                    DataColumns.active_power_sd: lambda d: d["active_power_std"].mask(mask_active_power),
                    DataColumns.wind_speed_mean: lambda d: d["wind_speed_avg"].mask(mask_wind_speed),
                    DataColumns.wind_speed_sd: lambda d: d["wind_speed_std"].mask(mask_wind_speed),
                    DataColumns.yaw_angle_mean: lambda d: d["nacelle_position_avg"].mask(mask_nacelle_position),
                    DataColumns.yaw_angle_min: lambda d: d["nacelle_position_min"].mask(mask_nacelle_position),
                    DataColumns.yaw_angle_max: lambda d: d["nacelle_position_max"].mask(mask_nacelle_position),
                    DataColumns.pitch_angle_mean: lambda d: d["blade_1_pitch_angle_avg"].mask(mask_pitch_angle),
                    DataColumns.gen_rpm_mean: lambda d: d["generator_speed_avg"].mask(mask_gen_rpm),
                    DataColumns.ambient_temp: lambda d: d["temperature_avg"].mask(mask_temperature),
                    DataColumns.shutdown_duration: 0,
                }
            )

        # unzipping the data in memory and only reading the relevant files
        circular_mean = partial(circmean, low=0, high=360)
        return (
            pd.read_csv(self.data_dir / Path(SCADA_FILE_PATH).name, parse_dates=[0], index_col=0)
            .pipe(_make_turbine_id_a_column)
            .groupby(DataColumns.turbine_name)
            .resample(f"{self.analysis_timebase_s}s")
            .aggregate(
                {
                    "active_power_avg": "mean",
                    "active_power_std": "mean",
                    "active_power_count": "sum",
                    "wind_speed_avg": "mean",
                    "wind_speed_std": "mean",
                    "wind_speed_count": "sum",
                    "blade_1_pitch_angle_avg": "mean",  # no need for circular_mean because no wrap
                    "blade_1_pitch_angle_count": "sum",
                    "generator_speed_avg": "mean",
                    "generator_speed_count": "sum",
                    "temperature_avg": "mean",
                    "temperature_count": "sum",
                    "nacelle_position_avg": circular_mean,
                    "nacelle_position_max": "max",
                    "nacelle_position_min": "min",
                    "nacelle_position_count": "sum",
                }
            )
            .reset_index(DataColumns.turbine_name)
            .pipe(_map_and_mask_cols)
            .loc[:, DataColumns.all()]
            .rename_axis(TIMESTAMP_COL, axis=0)
            .rename_axis(None, axis=1)
        )

    @with_parquet_cache(CACHE_DIR / "smarteole_metadata.parquet")
    def unpack_smarteole_metadata(self) -> pd.DataFrame:
        return (
            pd.read_csv(self.data_dir / Path(METADATA_FILE_PATH).name, index_col=0)
            .reset_index()
            .rename(columns={"Turbine": "Name"})
            .query("Name.str.startswith('SMV')")  # is a turbine
            .loc[:, ["Name", "Latitude", "Longitude"]]
            .assign(TimeZone="UTC", TimeSpanMinutes=self.analysis_timebase_s / 60, TimeFormat="Start")
        )

    @with_parquet_cache(CACHE_DIR / "smarteole_toggle.parquet")
    def unpack_smarteole_toggle_data(self) -> pd.DataFrame:
        ten_minutes_count_lower_limit = self.analysis_timebase_s * MINIMUM_DATA_COUNT_COVERAGE
        toggle_value_threshold: float = 0.95

        raw_df = pd.read_csv(self.data_dir / Path(TOGGLE_FILE_PATH).name, parse_dates=[0], index_col=0)

        required_in_cols = [
            "control_log_offset_active_avg",
            "control_log_offset_active_count",
            "control_log_offset_avg",
        ]
        toggle_df = (
            raw_df[required_in_cols]
            .resample(f"{self.analysis_timebase_s}s")
            .agg(
                {
                    "control_log_offset_active_avg": "mean",
                    "control_log_offset_active_count": "sum",
                    "control_log_offset_avg": "mean",
                }
            )
        )
        toggle_df["toggle_on"] = (toggle_df["control_log_offset_active_avg"] >= toggle_value_threshold) & (
            toggle_df["control_log_offset_active_count"] >= ten_minutes_count_lower_limit
        )
        toggle_df["toggle_off"] = (toggle_df["control_log_offset_active_avg"] <= (1 - toggle_value_threshold)) & (
            toggle_df["control_log_offset_active_count"] >= ten_minutes_count_lower_limit
        )
        toggle_df["yaw_offset_command"] = toggle_df["control_log_offset_avg"]

        toggle_df.index = toggle_df.index.tz_localize("UTC")
        toggle_df.index.name = TIMESTAMP_COL
        return toggle_df[["toggle_on", "toggle_off", "yaw_offset_command"]]
