import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd

from wind_up.constants import OUTPUT_DIR, PROJECTROOT_DIR

sys.path.append(str(PROJECTROOT_DIR))
from examples.helpers import download_zenodo_data, setup_logger

CACHE_DIR = PROJECTROOT_DIR / "cache" / "wedowind_example_data"
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / "wedowind_example"
ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

PARENT_DIR = Path(__file__).parent
ZIP_FILENAME = "Turbine_Upgrade_Dataset.zip"


def unpack_wedowind_scada() -> pd.DataFrame:
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
    scada_df_mast = (
        scada_df_raw.drop(columns=["y_ctrl(normalized)", "y_test(normalized)"]).copy().assign(TurbineName="Mast")
    )
    return pd.concat([scada_df_test, scada_df_ref, scada_df_mast])


if __name__ == "__main__":
    setup_logger(ANALYSIS_OUTPUT_DIR / "analysis.log")
    logger = logging.getLogger(__name__)

    logger.info("Downloading example data from Zenodo")
    download_zenodo_data(record_id="5516556", output_dir=CACHE_DIR, filenames={ZIP_FILENAME})
    download_zenodo_data(
        record_id="5516552", output_dir=CACHE_DIR, filenames={"Inland_Offshore_Wind_Farm_Dataset1.zip"}
    )

    logger.info("Preprocessing turbine SCADA data")
    scada_df = unpack_wedowind_scada()
