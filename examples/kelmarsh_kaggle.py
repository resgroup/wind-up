"""Example submission for https://www.kaggle.com/competitions/predict-the-wind-speed-at-a-wind-turbine/"""

import logging
from pathlib import Path

import pandas as pd

from examples.helpers import format_and_print_results_table, setup_logger
from examples.wedowind_example import create_fake_wedowind_reanalysis_dataset
from wind_up.constants import OUTPUT_DIR, PROJECTROOT_DIR, TIMESTAMP_COL, DataColumns
from wind_up.interface import AssessmentInputs
from wind_up.main_analysis import run_wind_up_analysis
from wind_up.models import PlotConfig, WindUpConfig

CACHE_DIR = PROJECTROOT_DIR / "cache" / "kelmarsh_kaggle"
ASSESSMENT_NAME = "kelmarsh_kaggle"
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / ASSESSMENT_NAME
ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
logger = logging.getLogger(__name__)


class KelmarshKaggleScadaUnpacker:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.scada_df = None

    def unpack(self) -> pd.DataFrame:
        if self.scada_df is None:
            raw_df = pd.read_csv(self.data_dir / "train.csv", header=[0, 1], index_col=[0], parse_dates=[1])
            workings_df = raw_df.set_index(("Timestamp", "Unnamed: 1_level_1"))
            workings_df = workings_df.drop(columns=[("training", "Unnamed: 52_level_1")])
            new_cols = pd.MultiIndex.from_tuples(
                [
                    ("Wind speed (m/s)", "Kelmarsh 1") if x == ("target_feature", "Unnamed: 53_level_1") else x
                    for x in workings_df.columns
                ]
            )
            workings_df.columns = new_cols
            workings_df = workings_df.stack().swaplevel(0, 1, axis=0)
            workings_df = workings_df.reset_index(level=0, names="TurbineName")
            self.scada_df = self._format_scada_df(scada_df=workings_df)
        return self.scada_df

    @staticmethod
    def _format_scada_df(scada_df: pd.DataFrame) -> pd.DataFrame:
        scada_df = scada_df.rename(
            columns={
                "Wind speed (m/s)": DataColumns.wind_speed_mean,
                "Wind speed, Standard deviation (m/s)": DataColumns.wind_speed_sd,
                "Nacelle position (°)": DataColumns.yaw_angle_mean,
                "Power (kW)": DataColumns.active_power_mean,
                "Nacelle ambient temperature (°C)": DataColumns.ambient_temp,
                "Generator RPM (RPM)": DataColumns.gen_rpm_mean,
                "Blade angle (pitch position) A (°)": DataColumns.pitch_angle_mean,
            }
        )
        # placeholder values for other required columns
        scada_df[DataColumns.shutdown_duration] = 0
        scada_df.index.name = TIMESTAMP_COL
        # make index UTC
        scada_df.index = scada_df.index.tz_localize("UTC")
        return scada_df


def kelmarsh_kaggle_metadata_df(data_dir: Path) -> pd.DataFrame:
    metadata_df = pd.read_csv(data_dir / "metaData.csv")[["Title", "Latitude", "Longitude"]].rename(
        columns={"Title": "Name"}
    )
    return metadata_df.assign(TimeZone="UTC", TimeSpanMinutes=10, TimeFormat="Start")


def main(analysis_name: str) -> None:
    # verify the data is in the correct location
    data_dir = Path("kelmarsh_kaggle_data")
    expected_files = [
        "train.csv",
        "test.csv",
        "sample_submission.csv",
        "metaData.csv",
    ]
    data_ok = all((data_dir / file).exists() for file in expected_files)
    if not data_ok:
        data_url = r"https://www.kaggle.com/competitions/predict-the-wind-speed-at-a-wind-turbine/data"
        msg = (
            f"Expected files not found in {data_dir}.\nPlease download the data from the Kaggle "
            f"at {data_url} and save them in {data_dir.resolve()}."
        )
        raise FileNotFoundError(msg)

    logger.info("Unpacking turbine SCADA data")
    scada_df = KelmarshKaggleScadaUnpacker(data_dir=data_dir).unpack()
    metadata_df = kelmarsh_kaggle_metadata_df(data_dir=data_dir)

    # Construct wind-up Configurations
    wtg_map = {
        x: {
            "name": x,
            "turbine_type": {
                "turbine_type": "Senvion MM92",
                "rotor_diameter_m": 92,
                "rated_power_kw": 2050,
                "normal_operation_pitch_range": (-10.0, 35.0),
                "normal_operation_genrpm_range": (0, 2000.0),
            },
        }
        for x in metadata_df["Name"]
    }

    # is it OK to use ERA5???
    reanalysis_dataset = create_fake_wedowind_reanalysis_dataset(start_datetime=scada_df.index.min())

    cfg = WindUpConfig(
        assessment_name=analysis_name,
        use_lt_distribution=False,
        out_dir=ANALYSIS_OUTPUT_DIR / analysis_name,
        test_wtgs=[wtg_map[x] for x in ["Kelmarsh 1", "Kelmarsh 2"]],
        ref_wtgs=[wtg_map[x] for x in ["Kelmarsh 3"]],
        years_offset_for_pre_period=1,
        years_for_lt_distribution=1,
        years_for_detrend=1,
        ws_bin_width=1.0,
        analysis_first_dt_utc_start=scada_df.index.min(),
        upgrade_first_dt_utc_start=scada_df.index.min() + (scada_df.index.max() - scada_df.index.min()) / 2,
        analysis_last_dt_utc_start=scada_df.index.max(),
        lt_first_dt_utc_start=scada_df.index.min(),
        lt_last_dt_utc_start=scada_df.index.max(),
        detrend_first_dt_utc_start=scada_df.index.min(),
        detrend_last_dt_utc_start=scada_df.index.max(),
        asset={"name": "Kelmarsh", "wtgs": list(wtg_map.values())},
        missing_scada_data_fields=[DataColumns.yaw_angle_min, DataColumns.yaw_angle_max],
        prepost={
            "pre_first_dt_utc_start": scada_df.index.min(),
            "pre_last_dt_utc_start": scada_df.index.min()
            + (scada_df.index.max() - scada_df.index.min()) / 2
            - pd.Timedelta(minutes=10),
            "post_first_dt_utc_start": scada_df.index.min() + (scada_df.index.max() - scada_df.index.min()) / 2,
            "post_last_dt_utc_start": scada_df.index.max(),
        },
        optimize_northing_corrections=False,
    )

    plot_cfg = PlotConfig(show_plots=False, save_plots=True, plots_dir=cfg.out_dir / "plots")

    cache_assessment = CACHE_DIR / analysis_name
    cache_assessment.mkdir(parents=True, exist_ok=True)

    assessment_inputs = AssessmentInputs.from_cfg(
        cfg=cfg,
        plot_cfg=plot_cfg,
        scada_df=scada_df,
        metadata_df=metadata_df,
        reanalysis_datasets=[reanalysis_dataset],
        cache_dir=cache_assessment,
    )

    # Run Analysis
    results_per_test_ref_df = run_wind_up_analysis(assessment_inputs)
    results_per_test_ref_df.to_csv(cfg.out_dir / "results_per_test_ref.csv", index=False)
    _ = format_and_print_results_table(results_per_test_ref_df)


if __name__ == "__main__":
    setup_logger(ANALYSIS_OUTPUT_DIR / "analysis.log")
    main("messin around")
