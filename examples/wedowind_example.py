# example based on https://relight.cloud/doc/turbine-upgrade-dataset-9zw1vl/turbineperformance

import datetime as dt
import logging
import math
import sys
import zipfile
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field

from wind_up.constants import OUTPUT_DIR, PROJECTROOT_DIR, TIMESTAMP_COL, DataColumns
from wind_up.interface import AssessmentInputs
from wind_up.main_analysis import run_wind_up_analysis
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.reanalysis_data import ReanalysisDataset
from wind_up.wind_funcs import calc_cp

sys.path.append(str(PROJECTROOT_DIR))
from examples.helpers import download_zenodo_data, setup_logger

CACHE_DIR = PROJECTROOT_DIR / "cache"
ASSESSMENT_NAME = "wedowind_example"
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / ASSESSMENT_NAME
ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

PARENT_DIR = Path(__file__).parent
ZIP_FILENAME = "Turbine_Upgrade_Dataset.zip"

setup_logger(ANALYSIS_OUTPUT_DIR / "analysis.log")
logger = logging.getLogger(__name__)


class WeDoWindScadaColumns(Enum):
    Y_CTRL_NORM = "y_ctrl(normalized)"
    Y_TEST_NORM = "y_test(normalized)"
    UPGRADE_STATUS = "upgradestatus"
    WIND_SPEED = "V"
    WIND_DIRECTION = "D"


class WeDoWindTurbineNames(Enum):
    REF = "Ref"
    TEST = "Test"


class KeyDates(NamedTuple):
    analysis_first_dt_utc_start: dt.datetime
    upgrade_first_dt_utc_start: dt.datetime
    analysis_last_dt_utc_start: dt.datetime
    lt_first_dt_utc_start: dt.datetime
    lt_last_dt_utc_start: dt.datetime
    detrend_first_dt_utc_start: dt.datetime
    detrend_last_dt_utc_start: dt.datetime
    pre_first_dt_utc_start: dt.datetime
    pre_last_dt_utc_start: dt.datetime
    post_first_dt_utc_start: dt.datetime
    post_last_dt_utc_start: dt.datetime


class WeDoWindScadaUnpacker:
    def __init__(self, scada_file_name: str, wedowind_zip_file_path: Path = CACHE_DIR / ZIP_FILENAME) -> None:
        self.scada_file_name = scada_file_name
        self.wedowind_zip_file_path = wedowind_zip_file_path
        self.scada_df = None

    def unpack(self, rated_power_kw: float) -> pd.DataFrame:
        if self.scada_df is None:
            raw_df = self._read_raw_df()
            scada_df_test = self._construct_scada_df_test(scada_df_raw=raw_df)
            scada_df_ref = self._construct_scada_df_ref(scada_df_raw=raw_df)
            self.scada_df = self._format_scada_df(
                scada_df=pd.concat([scada_df_test, scada_df_ref]), rated_power_kw=rated_power_kw
            )
        return self.scada_df

    def _read_raw_df(self) -> pd.DataFrame:
        with zipfile.ZipFile(self.wedowind_zip_file_path) as zf:
            raw_df = pd.read_csv(zf.open(self.scada_file_name), parse_dates=[1], index_col=0).drop(
                columns=["VcosD", "VsinD"]
            )
        raw_df.columns = raw_df.columns.str.replace(" ", "")
        return raw_df

    @staticmethod
    def _format_scada_df(scada_df: pd.DataFrame, rated_power_kw: float) -> pd.DataFrame:
        scada_df[DataColumns.active_power_mean] = scada_df["normalized_power"] * rated_power_kw
        # map some mast data to the turbine for convenience
        scada_df[DataColumns.wind_speed_mean] = scada_df[WeDoWindScadaColumns.WIND_SPEED.value]
        scada_df[DataColumns.yaw_angle_mean] = scada_df[WeDoWindScadaColumns.WIND_DIRECTION.value]
        # placeholder values for other required columns
        scada_df[DataColumns.pitch_angle_mean] = 0
        scada_df[DataColumns.gen_rpm_mean] = 1000
        scada_df[DataColumns.shutdown_duration] = 0

        scada_df = scada_df.set_index("time")
        scada_df.index.name = TIMESTAMP_COL
        # make index UTC
        scada_df.index = scada_df.index.tz_localize("UTC")
        return scada_df

    @staticmethod
    def _construct_scada_df_test(scada_df_raw: pd.DataFrame) -> pd.DataFrame:
        return (
            scada_df_raw.drop(columns=[WeDoWindScadaColumns.Y_CTRL_NORM.value])
            .copy()
            .assign(TurbineName=WeDoWindTurbineNames.TEST.value)
            .rename(columns={WeDoWindScadaColumns.Y_TEST_NORM.value: "normalized_power"})
        )

    @staticmethod
    def _construct_scada_df_ref(scada_df_raw: pd.DataFrame) -> pd.DataFrame:
        return (
            scada_df_raw.drop(columns=[WeDoWindScadaColumns.Y_TEST_NORM.value])
            .copy()
            .assign(TurbineName=WeDoWindTurbineNames.REF.value)
            .rename(columns={WeDoWindScadaColumns.Y_CTRL_NORM.value: "normalized_power"})
        )


class WeDoWindAnalysisConf(BaseModel):
    scada_file_name: str = Field(description="e.g. 'Turbine Upgrade Dataset(Pitch Angle Pair).csv'")
    wd_ranges_polluted_with_mast_shadow: list[tuple[int, int]] = Field(description="e.g. [(250, 315)]")
    clip_rated_power_pp: bool


def download_wedowind_data_from_zenodo() -> None:
    logger.info("Downloading example data from Zenodo")
    # https://zenodo.org/records/5516556
    download_zenodo_data(record_id="5516556", output_dir=CACHE_DIR, filenames={ZIP_FILENAME})


def create_fake_wedowind_metadata_df() -> pd.DataFrame:
    coords_df = pd.DataFrame(
        {
            "Name": ["WT1", "WT2", "WT3", "WT4", "MAST1", "MAST2"],
            "X": [500, 2200, 9836, 7571, 0, 9571],
            "Y": [9136, 9436, 0, 2050, 9836, 50],
        }
    )
    assumed_wf_lat = 40
    assumed_wf_lon = -89
    m_per_deglat = 40_075_000 / 360
    coords_df["Latitude"] = assumed_wf_lat + (coords_df["Y"] - coords_df["Y"].mean()) / m_per_deglat
    coords_df["Longitude"] = assumed_wf_lon + (coords_df["X"] - coords_df["X"].mean()) / (
        m_per_deglat * math.cos(assumed_wf_lat * math.pi / 180)
    )
    return coords_df.loc[:, ["Name", "Latitude", "Longitude"]].assign(
        TimeZone="UTC", TimeSpanMinutes=10, TimeFormat="Start"
    )


def create_fake_wedowind_reanalysis_dataset(scada_df_index_minimum: dt.datetime) -> ReanalysisDataset:
    rng = np.random.default_rng(0)
    rows = 100
    return ReanalysisDataset(
        id="dummy_reanalysis_data",
        data=pd.DataFrame(
            data={
                "100_m_hws_mean_mps": rng.uniform(5, 10, rows),
                "100_m_hwd_mean_deg-n_true": rng.uniform(0, 360, rows),
            },
            index=pd.DatetimeIndex(pd.date_range(start=scada_df_index_minimum, periods=rows, freq="h", tz="UTC")),
        ),
    )


def establish_wedowind_key_dates(scada_df: pd.DataFrame) -> KeyDates:
    """
    Extracts important dates from the SCADA data. These dates may then be used in the WindUpConfig.

    Args:
        scada_df:

    Returns: tuple of dates that may be passed to the WindUpConfig

    """
    analysis_first_dt_utc_start = scada_df.index.min()
    upgrade_first_dt_utc_start = scada_df[scada_df[WeDoWindScadaColumns.UPGRADE_STATUS.value] > 0].index.min()
    analysis_last_dt_utc_start = scada_df[scada_df[WeDoWindScadaColumns.UPGRADE_STATUS.value] > 0].index.max()
    lt_first_dt_utc_start = scada_df.index.min()
    lt_last_dt_utc_start = (
        scada_df.index.min()
        + (
            scada_df[scada_df[WeDoWindScadaColumns.UPGRADE_STATUS.value] > 0].index.max()
            - scada_df[scada_df[WeDoWindScadaColumns.UPGRADE_STATUS.value] > 0].index.min()
        )
        - pd.Timedelta(minutes=10)
    )
    detrend_first_dt_utc_start = scada_df.index.min()
    detrend_last_dt_utc_start = (
        scada_df[scada_df[WeDoWindScadaColumns.UPGRADE_STATUS.value] > 0].index.min()
        - pd.DateOffset(weeks=1)
        - pd.Timedelta(minutes=10)
    )
    pre_first_dt_utc_start = scada_df.index.min()
    pre_last_dt_utc_start = (
        scada_df.index.min()
        + (
            scada_df[scada_df[WeDoWindScadaColumns.UPGRADE_STATUS.value] > 0].index.max()
            - scada_df[scada_df[WeDoWindScadaColumns.UPGRADE_STATUS.value] > 0].index.min()
        )
        - pd.Timedelta(minutes=10)
    )
    post_first_dt_utc_start = scada_df[scada_df[WeDoWindScadaColumns.UPGRADE_STATUS.value] > 0].index.min()
    post_last_dt_utc_start = scada_df[scada_df[WeDoWindScadaColumns.UPGRADE_STATUS.value] > 0].index.max()

    return KeyDates(
        analysis_first_dt_utc_start=analysis_first_dt_utc_start,
        upgrade_first_dt_utc_start=upgrade_first_dt_utc_start,
        analysis_last_dt_utc_start=analysis_last_dt_utc_start,
        lt_first_dt_utc_start=lt_first_dt_utc_start,
        lt_last_dt_utc_start=lt_last_dt_utc_start,
        detrend_first_dt_utc_start=detrend_first_dt_utc_start,
        detrend_last_dt_utc_start=detrend_last_dt_utc_start,
        pre_first_dt_utc_start=pre_first_dt_utc_start,
        pre_last_dt_utc_start=pre_last_dt_utc_start,
        post_first_dt_utc_start=post_first_dt_utc_start,
        post_last_dt_utc_start=post_last_dt_utc_start,
    )


def generate_custom_exploratory_plots(
    scada_df: pd.DataFrame, assumed_rated_power_kw: float, rotor_diameter_m: int
) -> Path:
    """
    These custom plots are to help with SCADA data exploration.
    It was created because it was unclear how the SCADA data is related to the metadata so helped in looking for wakes
    in the data.

    Returns: path to directory containing the plots
    """
    custom_plots_dir_root = ANALYSIS_OUTPUT_DIR / "custom_plots"
    custom_plots_dir_timeseries = custom_plots_dir_root / "timeseries"

    custom_plots_dir_root.mkdir(exist_ok=True, parents=True)
    custom_plots_dir_timeseries.mkdir(exist_ok=True)

    for name, df in scada_df.groupby(DataColumns.turbine_name):
        for col in df.columns:
            plt.figure()
            plt.scatter(df.index, df[col], s=1)
            title = f"{name} {col}"
            plt.xlabel(TIMESTAMP_COL)
            plt.ylabel(col)
            plt.xticks(rotation=90)
            plt.grid()
            plt.tight_layout()
            plt.savefig(custom_plots_dir_timeseries / f"{title}.png")
            plt.close()

    region2_df = scada_df[(scada_df["normalized_power"] > 0.2) & (scada_df["normalized_power"] < 0.8)]  # noqa PLR2004

    binned_by_turbine = {}
    for name, df in region2_df.groupby(DataColumns.turbine_name):
        if name == "Mast":
            continue
        # find mean normalized_power and V binned by D
        _df = df.copy()
        _df[f"{WeDoWindScadaColumns.WIND_DIRECTION.value}_bin"] = pd.cut(
            _df[WeDoWindScadaColumns.WIND_DIRECTION.value], bins=range(0, 361, 5)
        )
        binned = _df.groupby(f"{WeDoWindScadaColumns.WIND_DIRECTION.value}_bin", observed=False)[
            [WeDoWindScadaColumns.WIND_DIRECTION.value, "normalized_power", "V"]
        ].mean()
        binned_by_turbine[name] = binned
        plt.figure()
        plt.plot(
            binned[WeDoWindScadaColumns.WIND_DIRECTION.value],
            calc_cp(
                power_kw=binned["normalized_power"] * assumed_rated_power_kw,
                ws_ms=binned[WeDoWindScadaColumns.WIND_SPEED.value],
                air_density_kgpm3=1.2,
                rotor_diameter_m=rotor_diameter_m,
            ),
            marker=".",
        )
        title = f"{name} Cp vs {WeDoWindScadaColumns.WIND_DIRECTION.value}"
        plt.title(title)
        plt.xlabel(WeDoWindScadaColumns.WIND_DIRECTION.value)
        plt.ylabel("Cp")
        plt.xticks(rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.savefig(custom_plots_dir_root / f"{title}.png")
        plt.close()

    plt.figure()
    for name, binned in binned_by_turbine.items():
        plt.plot(
            binned[WeDoWindScadaColumns.WIND_DIRECTION.value],
            calc_cp(
                power_kw=binned["normalized_power"] * assumed_rated_power_kw,
                ws_ms=binned[WeDoWindScadaColumns.WIND_SPEED.value],
                air_density_kgpm3=1.2,
                rotor_diameter_m=rotor_diameter_m,
            ),
            label=name,
            marker=".",
        )
    plt.ylim(0.2, 0.7)
    title = f"Cp vs {WeDoWindScadaColumns.WIND_DIRECTION.value}"
    plt.title(title)
    plt.xlabel(WeDoWindScadaColumns.WIND_DIRECTION.value)
    plt.ylabel("Cp")
    plt.xticks(rotation=90)
    plt.grid()
    plt.tight_layout()
    plt.savefig(custom_plots_dir_root / f"{title}.png")
    plt.close()

    logger.info("Custom plots saved to directory: %s", custom_plots_dir_root)
    return custom_plots_dir_root


def main() -> None:
    generate_custom_plots = False  # whether to create exploratory scada data plots
    analysis_name = "Pitch Angle"  # "Pitch Angle" or "Vortex Generator"

    download_wedowind_data_from_zenodo()

    assumed_rated_power_kw = 1500
    rotor_diameter_m = 80
    cutout_ws_mps = 20

    analysis_specific_config = {
        "Pitch Angle": WeDoWindAnalysisConf(
            scada_file_name="Turbine Upgrade Dataset(Pitch Angle Pair).csv",
            wd_ranges_polluted_with_mast_shadow=[(70, 150)],  # determined by inspecting the custom plots
            clip_rated_power_pp=False,
        ),
        "Vortex Generator": WeDoWindAnalysisConf(
            scada_file_name="Turbine Upgrade Dataset(VG Pair).csv",
            wd_ranges_polluted_with_mast_shadow=[(30, 115), (250, 315)],  # determined by inspecting the custom plots
            clip_rated_power_pp=True,  # Vortex Generators are not expected to increase rated power
        ),
    }
    if analysis_name not in analysis_specific_config:
        msg = f"analysis_name must be one of {list(analysis_specific_config.keys())}"
        raise ValueError(msg)

    analysis_conf = analysis_specific_config[analysis_name]

    logger.info("Preprocessing turbine SCADA data")
    scada_df = WeDoWindScadaUnpacker(scada_file_name=analysis_conf.scada_file_name).unpack(
        rated_power_kw=assumed_rated_power_kw
    )
    metadata_df = create_fake_wedowind_metadata_df()

    if generate_custom_plots:
        generate_custom_exploratory_plots(
            scada_df=scada_df, assumed_rated_power_kw=assumed_rated_power_kw, rotor_diameter_m=rotor_diameter_m
        )

    # based on the above I think the objects are MAST1, test=WT1 and ref=WT2
    scada_df = scada_df.replace(
        {
            DataColumns.turbine_name: {
                WeDoWindTurbineNames.TEST.value: "WT1",
                WeDoWindTurbineNames.REF.value: "WT2",
                "Mast": "MAST1",
            }
        }
    )
    # drop everything except the turbines from the metadata
    metadata_df = metadata_df[metadata_df["Name"].isin(["WT1", "WT2"])]
    # Reanalysis data is required by WindUp, therefore create a fake reanalysis object
    reanalysis_dataset = create_fake_wedowind_reanalysis_dataset(scada_df_index_minimum=scada_df.index.min())

    # Construct wind-up Configurations

    wtg_map = {
        x: {
            "name": x,
            "turbine_type": {
                "turbine_type": "unknown turbine type",
                "rotor_diameter_m": rotor_diameter_m,
                "rated_power_kw": assumed_rated_power_kw,
                "cutout_ws_mps": cutout_ws_mps,
                "normal_operation_pitch_range": (-10.0, 35.0),
                "normal_operation_genrpm_range": (0, 2000.0),
            },
        }
        for x in ["WT1", "WT2"]
    }

    key_dates = establish_wedowind_key_dates(scada_df=scada_df)

    cfg = WindUpConfig(
        assessment_name=analysis_name,
        ref_wd_filter=[150, 240],  # apparent wake free sector (determined by inspecting the custom plots)
        use_lt_distribution=False,
        out_dir=OUTPUT_DIR / analysis_name,
        test_wtgs=[wtg_map[x] for x in ["WT1"]],
        ref_wtgs=[wtg_map[x] for x in ["WT2"]],
        years_offset_for_pre_period=1,
        years_for_lt_distribution=1,
        years_for_detrend=1,
        ws_bin_width=1.0,
        analysis_first_dt_utc_start=key_dates.analysis_first_dt_utc_start,
        upgrade_first_dt_utc_start=key_dates.upgrade_first_dt_utc_start,
        analysis_last_dt_utc_start=key_dates.analysis_last_dt_utc_start,
        lt_first_dt_utc_start=key_dates.lt_first_dt_utc_start,
        lt_last_dt_utc_start=key_dates.lt_last_dt_utc_start,
        detrend_first_dt_utc_start=key_dates.detrend_first_dt_utc_start,
        detrend_last_dt_utc_start=key_dates.detrend_last_dt_utc_start,
        asset={"name": "Mystery Wind Farm", "wtgs": list(wtg_map.values())},
        missing_scada_data_fields=[DataColumns.yaw_angle_min, DataColumns.yaw_angle_max],
        prepost={
            "pre_first_dt_utc_start": key_dates.pre_first_dt_utc_start,
            "pre_last_dt_utc_start": key_dates.pre_last_dt_utc_start,
            "post_first_dt_utc_start": key_dates.post_first_dt_utc_start,
            "post_last_dt_utc_start": key_dates.post_last_dt_utc_start,
        },
        optimize_northing_corrections=False,
        clip_rated_power_pp=analysis_conf.clip_rated_power_pp,
    )

    plot_cfg = PlotConfig(show_plots=False, save_plots=True, plots_dir=cfg.out_dir / "plots")

    wd_ranges_to_exclude = analysis_conf.wd_ranges_polluted_with_mast_shadow
    scada_df_for_assessment = scada_df.copy()
    for wdr in wd_ranges_to_exclude:
        logger.info("Filtering out wind directions between %s", wdr)
        mask = (scada_df_for_assessment[DataColumns.yaw_angle_mean] >= wdr[0]) & (
            scada_df_for_assessment[DataColumns.yaw_angle_mean] <= wdr[1]
        )
        scada_df_for_assessment = scada_df_for_assessment.loc[~mask, :]

    cache_assessment = CACHE_DIR / analysis_name
    cache_assessment.mkdir(parents=True, exist_ok=True)

    assessment_inputs = AssessmentInputs.from_cfg(
        cfg=cfg,
        plot_cfg=plot_cfg,
        scada_df=scada_df_for_assessment,
        metadata_df=metadata_df,
        reanalysis_datasets=[reanalysis_dataset],
        cache_dir=cache_assessment,
    )

    # Run Analysis
    results_per_test_ref_df = run_wind_up_analysis(assessment_inputs)  # noqa: F841


if __name__ == "__main__":
    main()
