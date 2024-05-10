import sys
import zipfile
from pathlib import Path

import pandas as pd

from wind_up.caching import with_parquet_cache
from wind_up.constants import OUTPUT_DIR, PROJECTROOT_DIR, TIMESTAMP_COL, DataColumns
from wind_up.interface import AssessmentInputs
from wind_up.main_analysis import run_wind_up_analysis
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.reanalysis_data import ReanalysisDataset

sys.path.append(str(PROJECTROOT_DIR))
from examples.helpers import download_zenodo_data, setup_logger

CACHE_FLD = PROJECTROOT_DIR / "cache" / "kelmarsh_example_data"
TURBINE_METADATA_FILENAME = "Kelmarsh_WT_static.csv"
SCADA_DATA_FILENAME = "Kelmarsh_SCADA_2022_4457.zip"
PARENT_DIR = Path(__file__).parent


@with_parquet_cache(CACHE_FLD / "_kelmarsh_scada.parquet")
def _unpack_scada() -> pd.DataFrame:
    turbine_dfs = []

    # unzipping the data in memory and only reading the relevant files
    with zipfile.ZipFile(CACHE_FLD / SCADA_DATA_FILENAME) as zf:
        for inner_file in zf.filelist:
            if not inner_file.filename.startswith("Turbine"):
                continue
            turbine_name = f'KWF{inner_file.filename.split("_")[3]}'
            turbine_dfs.append(pd.read_csv(zf.open(inner_file.filename), skiprows=9).assign(turbine_name=turbine_name))

    # reshaping the turbine data to a single standard dataframe format
    return (
        pd.concat(turbine_dfs)
        .rename(
            columns={
                "Wind speed (m/s)": DataColumns.wind_speed_mean,
                "Wind speed, Standard deviation (m/s)": DataColumns.wind_speed_sd,
                "Power (kW)": DataColumns.active_power_mean,
                "Power, Standard deviation (kW)": DataColumns.active_power_sd,
                "Nacelle ambient temperature (°C)": DataColumns.ambient_temp,
                "Generator RPM (RPM)": DataColumns.gen_rpm_mean,
                "Blade angle (pitch position) A (°)": DataColumns.pitch_angle_mean,
                "Yaw bearing angle (°)": DataColumns.yaw_angle_mean,
                "Yaw bearing angle, Max (°)": DataColumns.yaw_angle_max,
                "Yaw bearing angle, Min (°)": DataColumns.yaw_angle_min,
                "turbine_name": DataColumns.turbine_name,
            }
        )
        .assign(
            **{
                DataColumns.shutdown_duration: lambda d: (1 - d["Time-based System Avail."]) * 600,
                TIMESTAMP_COL: lambda d: pd.to_datetime(d["# Date and time"], utc=True),
            }
        )
        .set_index(TIMESTAMP_COL)
        .loc[:, DataColumns.all()]
    )


@with_parquet_cache(CACHE_FLD / "_kelmarsh_metadata.parquet")
def _unpack_metadata() -> pd.DataFrame:
    md_fpath = CACHE_FLD / "Kelmarsh_WT_static.csv"
    return (
        pd.read_csv(md_fpath, index_col=0)
        .reset_index()
        .rename(columns={"Alternative Title": "Name"})
        .loc[:, ["Name", "Latitude", "Longitude"]]
        .assign(TimeZone="UTC", TimeSpanMinutes=10, TimeFormat="Start")
    )


if __name__ == "__main__":
    analysis_output_dir = OUTPUT_DIR / "kelmarsh-example"
    analysis_output_dir.mkdir(exist_ok=True, parents=True)
    setup_logger(analysis_output_dir / "analysis.log")

    download_zenodo_data(
        record_id="8252025",
        output_dir=CACHE_FLD,
        filenames={TURBINE_METADATA_FILENAME, SCADA_DATA_FILENAME},
    )

    turbine_map = {
        row["Alternative Title"]: {
            "name": row["Alternative Title"],
            "turbine_type": {
                "turbine_type": "Senvion-MM92-2050",
                "rotor_diameter_m": 92.0,
                "rated_power_kw": 2050.0,
                "cutout_ws_mps": 25,
                "normal_operation_pitch_range": (-10.0, 35.0),
                "normal_operation_genrpm_range": (250.0, 2000.0),
                "rpm_v_pw_margin_factor": 0.05,
                "pitch_to_stall": False,
            },
        }
        for _, row in pd.read_csv(CACHE_FLD / "Kelmarsh_WT_static.csv").iterrows()
    }

    metadata_df = _unpack_metadata()
    turbine_comb_df = _unpack_scada()

    pre_first_dt_utc_start = turbine_comb_df.index.min()
    post_last_dt_utc_start = turbine_comb_df.index.max()
    post_first_dt_utc_start = pre_first_dt_utc_start + (post_last_dt_utc_start - pre_first_dt_utc_start) / 2
    pre_last_dt_utc_start = post_first_dt_utc_start - pd.Timedelta(minutes=10)

    cfg = WindUpConfig(
        assessment_name="kelmarsh-example",
        asset={
            "name": "Kelmarsh",
            "wtgs": list(turbine_map.values()),
        },
        test_wtgs=[turbine_map["KWF1"]],
        ref_wtgs=[turbine_map["KWF2"]],
        out_dir=analysis_output_dir,
        analysis_first_dt_utc_start=pre_first_dt_utc_start,
        upgrade_first_dt_utc_start=post_first_dt_utc_start,
        analysis_last_dt_utc_start=post_last_dt_utc_start,
        lt_first_dt_utc_start=pre_first_dt_utc_start,
        lt_last_dt_utc_start=post_last_dt_utc_start,
        detrend_first_dt_utc_start=pre_first_dt_utc_start,
        detrend_last_dt_utc_start=pre_last_dt_utc_start,
        years_offset_for_pre_period=0,
        years_for_lt_distribution=0,
        years_for_detrend=0,
        ws_bin_width=1.0,
        prepost={
            "pre_first_dt_utc_start": pre_first_dt_utc_start,
            "pre_last_dt_utc_start": pre_last_dt_utc_start,
            "post_first_dt_utc_start": post_first_dt_utc_start,
            "post_last_dt_utc_start": post_last_dt_utc_start,
        },
        optimize_northing_corrections=False,
        northing_corrections_utc=[
            ("KWF1", pd.Timestamp("2022-01-01 00:00:00+0000"), 9.090411376952998),
            ("KWF1", pd.Timestamp("2022-04-23 10:30:00+0000"), 5.755374908447257),
            ("KWF2", pd.Timestamp("2022-01-01 00:00:00+0000"), 4.068655395508082),
            ("KWF3", pd.Timestamp("2022-01-01 00:00:00+0000"), 1.8756744384765625),
            ("KWF4", pd.Timestamp("2022-01-01 00:00:00+0000"), 7.840148925780969),
            ("KWF5", pd.Timestamp("2022-01-01 00:00:00+0000"), 11.577139806747734),
            ("KWF6", pd.Timestamp("2022-01-01 00:00:00+0000"), 4.946038818359088),
        ],
        exclusion_periods_utc=[
            ("ALL", pd.Timestamp("2022-09-30 14:20:00+0000"), pd.Timestamp("2022-09-30 17:50:00+0000")),
            ("ALL", pd.Timestamp("2022-10-05 01:10:00+0000"), pd.Timestamp("2022-10-07 14:30:00+0000")),
        ],
    )
    plot_cfg = PlotConfig(show_plots=False, save_plots=True, plots_dir=cfg.out_dir / "plots")

    reanalysis_dataset = ReanalysisDataset(
        id="ERA5T_52.50N_-1.00E_100m_1hr",
        data=pd.read_parquet(PARENT_DIR / "kelmarsh_data" / "ERA5T_52.50N_-1.00E_100m_1hr_2022.parquet"),
    )
    assessment_inputs = AssessmentInputs.from_cfg(
        cfg=cfg,
        plot_cfg=plot_cfg,
        scada_df=turbine_comb_df,
        metadata_df=metadata_df,
        toggle_df=None,
        reanalysis_datasets=[reanalysis_dataset],
        cache_dir=CACHE_FLD,
    )
    results_per_test_ref_df = run_wind_up_analysis(assessment_inputs)
