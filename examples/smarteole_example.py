import logging
import sys
import zipfile
from functools import partial
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.stats import circmean

from wind_up.caching import with_parquet_cache
from wind_up.combine_results import calc_net_uplift
from wind_up.constants import OUTPUT_DIR, PROJECTROOT_DIR, TIMESTAMP_COL, DataColumns
from wind_up.interface import AssessmentInputs
from wind_up.main_analysis import run_wind_up_analysis
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.reanalysis_data import ReanalysisDataset

sys.path.append(str(PROJECTROOT_DIR))
from examples.helpers import download_zenodo_data, format_and_print_results_table, setup_logger

CACHE_DIR = PROJECTROOT_DIR / "cache" / "smarteole_example_data"
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / "smarteole_example"
ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

ANALYSIS_TIMEBASE_S = 600
CACHE_SUBDIR = CACHE_DIR / f"timebase_{ANALYSIS_TIMEBASE_S}"
CACHE_SUBDIR.mkdir(exist_ok=True, parents=True)

CHECK_RESULTS = True
PARENT_DIR = Path(__file__).parent
ZIP_FILENAME = "SMARTEOLE-WFC-open-dataset.zip"
MINIMUM_DATA_COUNT_COVERAGE = 0.5  # 50% of the data must be present


@with_parquet_cache(CACHE_SUBDIR / "smarteole_scada.parquet")
def unpack_smarteole_scada(timebase_s: int) -> pd.DataFrame:
    """
    Function that translates 1-minute SCADA data to x minute data in the wind-up expected format
    """

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
        x_minutes_count_lower_limit = timebase_s * MINIMUM_DATA_COUNT_COVERAGE
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
    scada_fpath = "SMARTEOLE-WFC-open-dataset/SMARTEOLE_WakeSteering_SCADA_1minData.csv"
    circular_mean = partial(circmean, low=0, high=360)
    with zipfile.ZipFile(CACHE_DIR / ZIP_FILENAME) as zf:
        return (
            pd.read_csv(zf.open(scada_fpath), parse_dates=[0], index_col=0)
            .pipe(_make_turbine_id_a_column)
            .groupby(DataColumns.turbine_name)
            .resample(f"{timebase_s}s")
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
def unpack_smarteole_metadata(timebase_s: int) -> pd.DataFrame:
    md_fpath = "SMARTEOLE-WFC-open-dataset/SMARTEOLE_WakeSteering_Coordinates_staticData.csv"
    with zipfile.ZipFile(CACHE_DIR / ZIP_FILENAME) as zf:
        return (
            pd.read_csv(zf.open(md_fpath), index_col=0)
            .reset_index()
            .rename(columns={"Turbine": "Name"})
            .query("Name.str.startswith('SMV')")  # is a turbine
            .loc[:, ["Name", "Latitude", "Longitude"]]
            .assign(TimeZone="UTC", TimeSpanMinutes=timebase_s / 60, TimeFormat="Start")
        )


@with_parquet_cache(CACHE_SUBDIR / "smarteole_toggle.parquet")
def unpack_smarteole_toggle_data(timebase_s: int) -> pd.DataFrame:
    ten_minutes_count_lower_limit = timebase_s * MINIMUM_DATA_COUNT_COVERAGE
    toggle_value_threshold: float = 0.95

    _fpath = "SMARTEOLE-WFC-open-dataset/SMARTEOLE_WakeSteering_ControlLog_1minData.csv"
    with zipfile.ZipFile(CACHE_DIR / ZIP_FILENAME) as zf:
        raw_df = pd.read_csv(zf.open(_fpath), parse_dates=[0], index_col=0)

    required_in_cols = [
        "control_log_offset_active_avg",
        "control_log_offset_active_count",
        "control_log_offset_avg",
    ]
    toggle_df = (
        raw_df[required_in_cols]
        .resample(f"{timebase_s}s")
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


def define_smarteole_example_config() -> WindUpConfig:
    wtg_map = {
        f"SMV{i}": {
            "name": f"SMV{i}",
            "turbine_type": {
                "turbine_type": "Senvion-MM82-2050",
                "rotor_diameter_m": 82.0,
                "rated_power_kw": 2050.0,
                "cutout_ws_mps": 25,
                "normal_operation_pitch_range": (-10.0, 35.0),
                "normal_operation_genrpm_range": (250.0, 2000.0),
                "rpm_v_pw_margin_factor": 0.05,
                "pitch_to_stall": False,
            },
        }
        for i in range(1, 7 + 1)
    }
    northing_corrections_utc = [
        ("SMV1", pd.Timestamp("2020-02-17 16:30:00+0000"), 5.750994540354649),
        ("SMV2", pd.Timestamp("2020-02-17 16:30:00+0000"), 5.690999999999994),
        ("SMV3", pd.Timestamp("2020-02-17 16:30:00+0000"), 5.558000000000042),
        ("SMV4", pd.Timestamp("2020-02-17 16:30:00+0000"), 5.936999999999996),
        ("SMV5", pd.Timestamp("2020-02-17 16:30:00+0000"), 6.797253350869262),
        ("SMV6", pd.Timestamp("2020-02-17 16:30:00+0000"), 5.030130916842758),
        ("SMV7", pd.Timestamp("2020-02-17 16:30:00+0000"), 4.605999999999972),
    ]

    wd_filter_margin = 3 + 7 * ANALYSIS_TIMEBASE_S / 600
    return WindUpConfig(
        assessment_name="smarteole_example",
        timebase_s=ANALYSIS_TIMEBASE_S,
        require_ref_wake_free=True,
        detrend_min_hours=12,
        ref_wd_filter=[207 - wd_filter_margin, 236 + wd_filter_margin],  # steer is from 207-236
        filter_all_test_wtgs_together=True,
        use_lt_distribution=False,
        out_dir=ANALYSIS_OUTPUT_DIR,
        test_wtgs=[wtg_map["SMV6"], wtg_map["SMV5"]],
        ref_wtgs=[wtg_map["SMV7"]],
        ref_super_wtgs=[],
        non_wtg_ref_names=[],
        analysis_first_dt_utc_start=pd.Timestamp("2020-02-17 16:30:00+0000"),
        upgrade_first_dt_utc_start=pd.Timestamp("2020-02-17 16:30:00+0000"),
        analysis_last_dt_utc_start=pd.Timestamp("2020-05-25 00:00:00+0000") - pd.Timedelta(seconds=ANALYSIS_TIMEBASE_S),
        lt_first_dt_utc_start=pd.Timestamp("2020-02-17 16:30:00+0000"),
        lt_last_dt_utc_start=pd.Timestamp("2020-05-25 00:00:00+0000") - pd.Timedelta(seconds=ANALYSIS_TIMEBASE_S),
        detrend_first_dt_utc_start=pd.Timestamp("2020-02-17 16:30:00+0000"),
        detrend_last_dt_utc_start=pd.Timestamp("2020-05-25 00:00:00+0000") - pd.Timedelta(seconds=ANALYSIS_TIMEBASE_S),
        years_for_lt_distribution=0,
        years_for_detrend=0,
        ws_bin_width=1.0,
        asset={
            "name": "Sole du Moulin Vieux",
            "wtgs": list(wtg_map.values()),
            "masts_and_lidars": [],
        },
        northing_corrections_utc=northing_corrections_utc,
        toggle={
            "name": "wake steering",
            "toggle_file_per_turbine": False,
            "toggle_filename": "SMV_offset_active_toggle_df.parquet",
            "detrend_data_selection": "use_toggle_off_data",
            "pairing_filter_method": "any_within_timedelta",
            "pairing_filter_timedelta_seconds": 3600,
            "toggle_change_settling_filter_seconds": 120,
        },
    )


def print_smarteole_results(
    results_per_test_ref_df: pd.DataFrame, *, print_small_table: bool = False, check_results: bool = False
) -> None:
    print_df = format_and_print_results_table(results_per_test_ref_df, print_small_table=print_small_table)

    if check_results:
        # raise an error if results don't match expected
        expected_print_df = pd.DataFrame(
            {
                "turbine": ["SMV6", "SMV5"],
                "reference": ["SMV7", "SMV7"],
                "energy uplift": ["-1.0%", "3.1%"],
                "uplift uncertainty": ["0.6%", "1.2%"],
                "uplift P95": ["-2.0%", "1.2%"],
                "uplift P5": ["-0.1%", "5.0%"],
                "valid hours toggle off": [132 + 3 / 6, 133 + 0 / 6],
                "valid hours toggle on": [136 + 0 / 6, 137 + 1 / 6],
                "mean power toggle on": [1148, 994],
            },
            index=[0, 1],
        )

        assert_frame_equal(print_df, expected_print_df)


if __name__ == "__main__":
    setup_logger(ANALYSIS_OUTPUT_DIR / "analysis.log")
    logger = logging.getLogger(__name__)

    logger.info("Downloading example data from Zenodo")
    download_zenodo_data(record_id="7342466", output_dir=CACHE_DIR, filenames={ZIP_FILENAME})

    logger.info("Preprocessing (and caching) turbine SCADA data")
    scada_df = unpack_smarteole_scada(ANALYSIS_TIMEBASE_S)
    logger.info("Preprocessing (and caching) turbine metadata")
    metadata_df = unpack_smarteole_metadata(ANALYSIS_TIMEBASE_S)
    logger.info("Preprocessing (and caching) toggle data")
    toggle_df = unpack_smarteole_toggle_data(ANALYSIS_TIMEBASE_S)

    logger.info("Merging SMV6 yaw offset command signal into SCADA data")
    toggle_df_no_tz = toggle_df.copy()
    toggle_df_no_tz.index = toggle_df_no_tz.index.tz_localize(None)
    scada_df = scada_df.merge(toggle_df_no_tz["yaw_offset_command"], left_index=True, right_index=True, how="left")
    scada_df["yaw_offset_command"] = scada_df["yaw_offset_command"].where(scada_df["TurbineName"] == "SMV6", 0)
    del toggle_df_no_tz

    logger.info("Loading reference reanalysis data")
    reanalysis_dataset = ReanalysisDataset(
        id="ERA5T_50.00N_2.75E_100m_1hr",
        data=pd.read_parquet(PARENT_DIR / "smarteole_data" / "ERA5T_50.00N_2.75E_100m_1hr_20200201_20200531.parquet"),
    )

    logger.info("Defining Assessment Configuration")
    cfg = define_smarteole_example_config()
    plot_cfg = PlotConfig(show_plots=False, save_plots=True, plots_dir=cfg.out_dir / "plots")

    assessment_inputs = AssessmentInputs.from_cfg(
        cfg=cfg,
        plot_cfg=plot_cfg,
        toggle_df=toggle_df,
        scada_df=scada_df,
        metadata_df=metadata_df,
        reanalysis_datasets=[reanalysis_dataset],
        cache_dir=CACHE_SUBDIR,
    )
    results_per_test_ref_df = run_wind_up_analysis(assessment_inputs)

    net_p50, net_p95, net_p5 = calc_net_uplift(results_per_test_ref_df, confidence=0.9)
    print(f"net P50: {net_p50:.1%}, net P95: {net_p95:.1%}, net P5: {net_p5:.1%}")

    print_smarteole_results(results_per_test_ref_df, check_results=CHECK_RESULTS)
