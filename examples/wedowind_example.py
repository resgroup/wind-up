# example based on https://relight.cloud/doc/turbine-upgrade-dataset-9zw1vl/turbineperformance

import logging
import math
import sys
import zipfile
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from wind_up.constants import OUTPUT_DIR, PROJECTROOT_DIR, TIMESTAMP_COL, DataColumns
from wind_up.interface import AssessmentInputs
from wind_up.main_analysis import run_wind_up_analysis
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.reanalysis_data import ReanalysisDataset
from wind_up.wind_funcs import calc_cp

sys.path.append(str(PROJECTROOT_DIR))
from examples.helpers import download_zenodo_data, setup_logger

CACHE_DIR = PROJECTROOT_DIR / "cache" / "wedowind_example_data"
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


class TurbineNames(Enum):
    REF = "Ref"
    TEST = "Test"


class MetadataColumns(Enum):
    NAME = "Name"
    LATITUDE = "Latitude"
    LONGITUDE = "Longitude"


class WDWScadaUnpacker:
    def __init__(self, scada_file_name: str, wdw_zip_file_path: Path = CACHE_DIR / ZIP_FILENAME) -> None:
        self.scada_file_name = scada_file_name
        self.wdw_zip_file_path = wdw_zip_file_path
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
        with zipfile.ZipFile(self.wdw_zip_file_path) as zf:
            return pd.read_csv(zf.open(self.scada_file_name), parse_dates=[1], index_col=0).drop(
                columns=["VcosD", "VsinD"]
            )

    @staticmethod
    def _format_scada_df(scada_df: pd.DataFrame, rated_power_kw: float) -> pd.DataFrame:
        scada_df[DataColumns.active_power_mean] = scada_df["normalized_power"] * rated_power_kw
        # map some mast data to the turbine for convenience
        scada_df[DataColumns.wind_speed_mean] = scada_df["V"]
        scada_df[DataColumns.yaw_angle_mean] = scada_df["D"]
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
            .assign(TurbineName=TurbineNames.TEST.value)
            .rename(columns={WeDoWindScadaColumns.Y_TEST_NORM.value: "normalized_power"})
        )

    @staticmethod
    def _construct_scada_df_ref(scada_df_raw: pd.DataFrame) -> pd.DataFrame:
        return (
            scada_df_raw.drop(columns=[WeDoWindScadaColumns.Y_TEST_NORM.value])
            .copy()
            .assign(TurbineName=TurbineNames.REF.value)
            .rename(columns={WeDoWindScadaColumns.Y_CTRL_NORM.value: "normalized_power"})
        )


def make_wdw_metadata_df() -> pd.DataFrame:
    coords_df = pd.DataFrame(
        {
            MetadataColumns.NAME.value: ["WT1", "WT2", "WT3", "WT4", "MAST1", "MAST2"],
            "X": [500, 2200, 9836, 7571, 0, 9571],
            "Y": [9136, 9436, 0, 2050, 9836, 50],
        }
    )
    assumed_wf_lat = 40
    assumed_wf_lon = -89
    m_per_deglat = 40_075_000 / 360
    coords_df[MetadataColumns.LATITUDE.value] = assumed_wf_lat + (coords_df["Y"] - coords_df["Y"].mean()) / m_per_deglat
    coords_df[MetadataColumns.LONGITUDE.value] = assumed_wf_lon + (coords_df["X"] - coords_df["X"].mean()) / (
        m_per_deglat * math.cos(assumed_wf_lat * math.pi / 180)
    )
    return coords_df.loc[
        :, [MetadataColumns.NAME.value, MetadataColumns.LATITUDE.value, MetadataColumns.LONGITUDE.value]
    ].assign(TimeZone="UTC", TimeSpanMinutes=10, TimeFormat="Start")


def run_custom_plots(scada_df: pd.DataFrame, assumed_rated_power_kw: float, rotor_diameter_m: int) -> Path:
    """
    It is unclear how the scada data is related to the metadata so look for wakes in the data

    Returns: None (but displays plots)

    """
    custom_plots_dir_root = ANALYSIS_OUTPUT_DIR / "custom_plots"
    custom_plots_dir_timeseries = custom_plots_dir_root / "timeseries"

    custom_plots_dir_root.mkdir(exist_ok=True, parents=True)
    custom_plots_dir_timeseries.mkdir(exist_ok=True)

    for name, df in scada_df.groupby("TurbineName"):
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
    for name, df in region2_df.groupby("TurbineName"):
        if name == "Mast":
            continue
        # find mean normalized_power and V binned by D
        _df = df.copy()
        _df["D_bin"] = pd.cut(_df["D"], bins=range(0, 361, 5))
        binned = _df.groupby("D_bin", observed=False)[["D", "normalized_power", "V"]].mean()
        binned_by_turbine[name] = binned
        plt.figure()
        plt.plot(
            binned["D"],
            calc_cp(
                power_kw=binned["normalized_power"] * assumed_rated_power_kw,
                ws_ms=binned["V"],
                air_density_kgpm3=1.2,
                rotor_diameter_m=rotor_diameter_m,
            ),
            marker=".",
        )
        title = f"{name} Cp vs D"
        plt.title(title)
        plt.xlabel("D")
        plt.ylabel("Cp")
        plt.xticks(rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.savefig(custom_plots_dir_root / f"{title}.png")
        plt.close()

    plt.figure()
    for name, binned in binned_by_turbine.items():
        plt.plot(
            binned["D"],
            calc_cp(
                power_kw=binned["normalized_power"] * assumed_rated_power_kw,
                ws_ms=binned["V"],
                air_density_kgpm3=1.2,
                rotor_diameter_m=rotor_diameter_m,
            ),
            label=name,
            marker=".",
        )
    plt.ylim(0.2, 0.7)
    title = "Cp vs D"
    plt.title(title)
    plt.xlabel("D")
    plt.ylabel("Cp")
    plt.xticks(rotation=90)
    plt.grid()
    plt.tight_layout()
    plt.savefig(custom_plots_dir_root / f"{title}.png")
    plt.close()

    logger.info("Custom plots saved to directory: %s", custom_plots_dir_root)
    return custom_plots_dir_root


def download_wdw_data_from_zenodo() -> None:
    logger.info("Downloading example data from Zenodo")
    download_zenodo_data(record_id="5516556", output_dir=CACHE_DIR, filenames={ZIP_FILENAME})
    download_zenodo_data(
        record_id="5516552", output_dir=CACHE_DIR, filenames={"Inland_Offshore_Wind_Farm_Dataset1.zip"}
    )


def main() -> None:
    download_wdw_data_from_zenodo()

    assumed_rated_power_kw = 1500
    rotor_diameter_m = 80
    cutout_ws_mps = 20
    scada_file_name = "Turbine Upgrade Dataset(Pitch Angle Pair).csv"  # or Turbine Upgrade Dataset(VG Pair).csv

    logger.info("Preprocessing turbine SCADA data")
    scada_df = WDWScadaUnpacker(scada_file_name=scada_file_name).unpack(rated_power_kw=assumed_rated_power_kw)
    metadata_df = make_wdw_metadata_df()

    run_custom_plots(
        scada_df=scada_df, assumed_rated_power_kw=assumed_rated_power_kw, rotor_diameter_m=rotor_diameter_m
    )

    # based on the above I think the objects are MAST1, test=WT1 and ref=WT2
    scada_df = scada_df.replace(
        {"TurbineName": {TurbineNames.TEST.value: "WT1", TurbineNames.REF.value: "WT2", "Mast": "MAST1"}}
    )
    # drop everything except the turbines from the metadata
    metadata_df = metadata_df[metadata_df["Name"].isin(["WT1", "WT2"])]

    rng = np.random.default_rng(0)
    rows = 100
    reanalysis_dataset = ReanalysisDataset(
        id="dummy_reanalysis_data",
        data=pd.DataFrame(
            data={
                "100_m_hws_mean_mps": rng.uniform(5, 10, rows),
                "100_m_hwd_mean_deg-n_true": rng.uniform(0, 360, rows),
            },
            index=pd.DatetimeIndex(pd.date_range(start=scada_df.index.min(), periods=rows, freq="h", tz="UTC")),
        ),
    )

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

    cfg = WindUpConfig(
        assessment_name=ASSESSMENT_NAME,
        ref_wd_filter=[150, 240],  # apparent wake free sector
        use_lt_distribution=False,
        out_dir=OUTPUT_DIR / ASSESSMENT_NAME,
        test_wtgs=[wtg_map[x] for x in ["WT1"]],
        ref_wtgs=[wtg_map[x] for x in ["WT2"]],
        analysis_first_dt_utc_start=scada_df.index.min(),
        upgrade_first_dt_utc_start=scada_df[scada_df["upgrade status"] > 0].index.min(),
        analysis_last_dt_utc_start=scada_df[scada_df["upgrade status"] > 0].index.max(),
        years_offset_for_pre_period=1,
        lt_first_dt_utc_start=scada_df.index.min(),
        lt_last_dt_utc_start=scada_df.index.min()
        + (scada_df[scada_df["upgrade status"] > 0].index.max() - scada_df[scada_df["upgrade status"] > 0].index.min())
        - pd.Timedelta(minutes=10),
        detrend_first_dt_utc_start=scada_df.index.min(),
        detrend_last_dt_utc_start=scada_df[scada_df["upgrade status"] > 0].index.min()
        - pd.DateOffset(weeks=1)
        - pd.Timedelta(minutes=10),
        years_for_lt_distribution=1,
        years_for_detrend=1,
        ws_bin_width=1.0,
        asset={
            "name": "Mystery Wind Farm",
            "wtgs": list(wtg_map.values()),
        },
        missing_scada_data_fields=["YawAngleMin", "YawAngleMax"],
        prepost={
            "pre_first_dt_utc_start": scada_df.index.min(),
            "pre_last_dt_utc_start": scada_df.index.min()
            + (
                scada_df[scada_df["upgrade status"] > 0].index.max()
                - scada_df[scada_df["upgrade status"] > 0].index.min()
            )
            - pd.Timedelta(minutes=10),
            "post_first_dt_utc_start": scada_df[scada_df["upgrade status"] > 0].index.min(),
            "post_last_dt_utc_start": scada_df[scada_df["upgrade status"] > 0].index.max(),
        },
        optimize_northing_corrections=False,
    )

    plot_cfg = PlotConfig(show_plots=False, save_plots=True, plots_dir=cfg.out_dir / "plots")

    assessment_inputs = AssessmentInputs.from_cfg(
        cfg=cfg,
        plot_cfg=plot_cfg,
        scada_df=scada_df[(scada_df["D"] < 70) | (scada_df["D"] > 150)],  # noqa PLR2004 filter out apparent mast waked sector
        metadata_df=metadata_df,
        reanalysis_datasets=[reanalysis_dataset],
        cache_dir=CACHE_DIR,
    )

    # Run Analysis
    results_per_test_ref_df = run_wind_up_analysis(assessment_inputs)  # noqa: F841


if __name__ == "__main__":
    main()
