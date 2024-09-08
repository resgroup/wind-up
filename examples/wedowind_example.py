# example based on https://relight.cloud/doc/turbine-upgrade-dataset-9zw1vl/turbineperformance

import logging
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from wind_up.constants import OUTPUT_DIR, PROJECTROOT_DIR, TIMESTAMP_COL, DataColumns
from wind_up.interface import AssessmentInputs
from wind_up.main_analysis import run_wind_up_analysis
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.wind_funcs import calc_cp

sys.path.append(str(PROJECTROOT_DIR))
from examples.helpers import download_zenodo_data, setup_logger

CACHE_DIR = PROJECTROOT_DIR / "cache" / "wedowind_example_data"
ASSESSMENT_NAME = "wedowind_example"
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / ASSESSMENT_NAME
ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

PARENT_DIR = Path(__file__).parent
ZIP_FILENAME = "Turbine_Upgrade_Dataset.zip"


def unpack_wedowind_scada(rated_power_kw: float, filename: str) -> pd.DataFrame:
    with zipfile.ZipFile(CACHE_DIR / ZIP_FILENAME) as zf:
        scada_df_raw = pd.read_csv(zf.open(filename), parse_dates=[1], index_col=0).drop(columns=["VcosD", "VsinD"])
    scada_df_test = (
        scada_df_raw.drop(columns=["y_ctrl(normalized)"])
        .copy()
        .assign(TurbineName="Test")
        .rename(columns={"y_test(normalized)": "normalized_power"})
    )
    scada_df_ref = (
        scada_df_raw.drop(columns=["y_test(normalized)"])
        .copy()
        .assign(TurbineName="Ref")
        .rename(columns={"y_ctrl(normalized)": "normalized_power"})
    )
    scada_df = pd.concat([scada_df_test, scada_df_ref])
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


def make_wedowind_metadata_df() -> pd.DataFrame:
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


if __name__ == "__main__":
    setup_logger(ANALYSIS_OUTPUT_DIR / "analysis.log")
    logger = logging.getLogger(__name__)

    logger.info("Downloading example data from Zenodo")
    download_zenodo_data(record_id="5516556", output_dir=CACHE_DIR, filenames={ZIP_FILENAME})
    download_zenodo_data(
        record_id="5516552", output_dir=CACHE_DIR, filenames={"Inland_Offshore_Wind_Farm_Dataset1.zip"}
    )
    assumed_rated_power_kw = 1500
    rotor_diameter_m = 80
    cutout_ws_mps = 20

    filename = "Turbine Upgrade Dataset(Pitch Angle Pair).csv"  # or Turbine Upgrade Dataset(VG Pair).csv
    logger.info("Preprocessing turbine SCADA data")
    scada_df = unpack_wedowind_scada(rated_power_kw=assumed_rated_power_kw, filename=filename)
    metadata_df = make_wedowind_metadata_df()

    # it is unclear how the scada data is related to the metadata so look for wakes in the data
    make_custom_plots = True
    if make_custom_plots:
        (ANALYSIS_OUTPUT_DIR / "custom_plots").mkdir(exist_ok=True, parents=True)
        (ANALYSIS_OUTPUT_DIR / "custom_plots" / "timeseries").mkdir(exist_ok=True)
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
                plt.savefig(ANALYSIS_OUTPUT_DIR / "custom_plots" / "timeseries" / f"{title}.png")
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
            plt.savefig(ANALYSIS_OUTPUT_DIR / "custom_plots" / f"{title}.png")
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
        plt.savefig(ANALYSIS_OUTPUT_DIR / "custom_plots" / f"{title}.png")
        plt.close()

    # based on the above I think the objects are MAST1, test=WT1 and ref=WT2
    scada_df = scada_df.replace({"TurbineName": {"Test": "WT1", "Ref": "WT2", "Mast": "MAST1"}})
    # drop everything except the turbines from the metadata
    metadata_df = metadata_df[metadata_df["Name"].isin(["WT1", "WT2"])]
    # make up reanalysis for now
    from wind_up.reanalysis_data import ReanalysisDataset

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
    msg = f"{cfg.out_dir=}"
    logger.info(msg)
    plot_cfg = PlotConfig(show_plots=False, save_plots=True, plots_dir=cfg.out_dir / "plots")
    assessment_inputs = AssessmentInputs.from_cfg(
        cfg=cfg,
        plot_cfg=plot_cfg,
        scada_df=scada_df[(scada_df["D"] < 70) | (scada_df["D"] > 150)],  # noqa PLR2004 filter out apparent mast waked sector
        metadata_df=metadata_df,
        reanalysis_datasets=[reanalysis_dataset],
        cache_dir=CACHE_DIR,
    )
    results_per_test_ref_df = run_wind_up_analysis(assessment_inputs)
