"""Example submission for https://www.kaggle.com/competitions/predict-the-wind-speed-at-a-wind-turbine/"""

import logging
from pathlib import Path

from examples.helpers import setup_logger
from wind_up.constants import OUTPUT_DIR, PROJECTROOT_DIR

CACHE_DIR = PROJECTROOT_DIR / "cache" / "kelmarsh_kaggle"
ASSESSMENT_NAME = "kelmarsh_kaggle"
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / ASSESSMENT_NAME
ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
setup_logger(ANALYSIS_OUTPUT_DIR / "analysis.log")
logger = logging.getLogger(__name__)


def main(analysis_name: str, *, generate_custom_plots: bool = True) -> None:
    # verify the data is in the correct location
    data_path = Path("kelmarsh_kaggle_data")
    expected_files = [
        "train.csv",
        "test.csv",
        "sample_submission.csv",
        "metaData.csv",
        "blah.csv",
    ]
    data_ok = all((data_path / file).exists() for file in expected_files)
    if not data_ok:
        data_url = r"https://www.kaggle.com/competitions/predict-the-wind-speed-at-a-wind-turbine/data"
        raise FileNotFoundError(
            f"Expected files not found in {data_path}.\nPlease download the data from the Kaggle "
            f"at {data_url} and save them in {data_path.resolve()}."
        )

    # assumptions below are based on Table 5.1 of Data Science for Wind Energy (Yu Ding 2020)
    assumed_rated_power_kw = 2050  # from metaData.csv
    assumed_rotor_diameter_m = 92  # from metaData.csv
    cutout_ws_mps = 25  # assumption

    logger.info("Unpacking turbine SCADA data")


if __name__ == "__main__":
    main("messin around")
