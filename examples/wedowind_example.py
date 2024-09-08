import logging
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from wind_up.constants import OUTPUT_DIR, PROJECTROOT_DIR, TIMESTAMP_COL, DataColumns

sys.path.append(str(PROJECTROOT_DIR))
from examples.helpers import download_zenodo_data, setup_logger

CACHE_DIR = PROJECTROOT_DIR / "cache" / "wedowind_example_data"
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / "wedowind_example"
ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

PARENT_DIR = Path(__file__).parent
ZIP_FILENAME = "Turbine_Upgrade_Dataset.zip"


def unpack_wedowind_scada(rated_power_kw: float) -> pd.DataFrame:
    scada_fpath = "Turbine Upgrade Dataset(Pitch Angle Pair).csv"
    with zipfile.ZipFile(CACHE_DIR / ZIP_FILENAME) as zf:
        scada_df_raw = pd.read_csv(zf.open(scada_fpath), parse_dates=[1], index_col=0).drop(columns=["VcosD", "VsinD"])
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
    scada_df[DataColumns.wind_speed_mean] = scada_df["V"]
    scada_df[DataColumns.yaw_angle_mean] = scada_df["D"]
    scada_df[DataColumns.pitch_angle_mean] = 0
    scada_df[DataColumns.gen_rpm_mean] = 1000
    scada_df[DataColumns.shutdown_duration] = 0

    scada_df_mast = (
        scada_df_raw.drop(columns=["y_ctrl(normalized)", "y_test(normalized)"]).copy().assign(TurbineName="Mast")
    )
    scada_df = pd.concat([scada_df, scada_df_mast])

    scada_df = scada_df.set_index("time")
    scada_df.index.name = TIMESTAMP_COL
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

    logger.info("Preprocessing turbine SCADA data")
    assumed_rated_power_kw = 2000
    scada_df = unpack_wedowind_scada(rated_power_kw=assumed_rated_power_kw)
    metadata_df = make_wedowind_metadata_df()

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
