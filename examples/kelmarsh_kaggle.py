"""Example submission for https://www.kaggle.com/competitions/predict-the-wind-speed-at-a-wind-turbine/"""

import logging
from pathlib import Path

import pandas as pd

from examples.helpers import setup_logger
from wind_up.constants import (
    OUTPUT_DIR,
    PROJECTROOT_DIR,
    REANALYSIS_WD_COL,
    REANALYSIS_WS_COL,
    TIMESTAMP_COL,
    WINDFARM_YAWDIR_COL,
    DataColumns,
)
from wind_up.detrend import calc_wsratio_v_wd_scen
from wind_up.interface import AssessmentInputs
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.northing import check_wtg_northing
from wind_up.plots.data_coverage_plots import plot_detrend_data_cov
from wind_up.reanalysis_data import ReanalysisDataset
from wind_up.waking_state import add_waking_scen, get_distance_and_bearing
from wind_up.windspeed_drift import check_windspeed_drift

DATA_DIR = Path("kelmarsh_kaggle_data")
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
    expected_files = [
        "train.csv",
        "test.csv",
        "sample_submission.csv",
        "metaData.csv",
    ]
    data_ok = all((DATA_DIR / file).exists() for file in expected_files)
    if not data_ok:
        data_url = r"https://www.kaggle.com/competitions/predict-the-wind-speed-at-a-wind-turbine/data"
        msg = (
            f"Expected files not found in {DATA_DIR}.\nPlease download the data from the Kaggle "
            f"at {data_url} and save them in {DATA_DIR.resolve()}."
        )
        raise FileNotFoundError(msg)

    logger.info("Unpacking turbine SCADA data")
    scada_df = KelmarshKaggleScadaUnpacker(data_dir=DATA_DIR).unpack()
    metadata_df = kelmarsh_kaggle_metadata_df(data_dir=DATA_DIR)

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
    reanalysis_dataset = ReanalysisDataset(
        id="ERA5T_52.50N_-1.00E_100m_1hr",
        data=pd.read_parquet(DATA_DIR / "ERA5T_52.50N_-1.00E_100m_1hr.parquet"),
    )

    # calculated previously by setting optimize_northing_corrections to True
    northing_corrections_utc = [
        ("Kelmarsh 2", pd.Timestamp("2017-10-01 00:00:00+0000"), 3.4831420898439944),
        ("Kelmarsh 3", pd.Timestamp("2017-10-01 00:00:00+0000"), 1.6804382324219773),
        ("Kelmarsh 4", pd.Timestamp("2017-10-01 00:00:00+0000"), 3.7531753316334004),
        ("Kelmarsh 5", pd.Timestamp("2017-10-01 00:00:00+0000"), 7.918688964843739),
        ("Kelmarsh 5", pd.Timestamp("2020-04-16 10:00:00+0000"), 12.944992828369152),
        ("Kelmarsh 5", pd.Timestamp("2020-04-23 18:10:00+0000"), 8.455931250697915),
        ("Kelmarsh 6", pd.Timestamp("2017-10-01 00:00:00+0000"), 5.209234619141114),
    ]

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
        optimize_northing_corrections=False,  # switch to True to recalculate northing_corrections_utc
        northing_corrections_utc=northing_corrections_utc,
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

    predict_kelmarsh_t1_windspeed(assessment_inputs)


def predict_kelmarsh_t1_windspeed(assessment_inputs: AssessmentInputs) -> None:
    wf_df = assessment_inputs.wf_df
    pc_per_ttype = assessment_inputs.pc_per_ttype
    cfg = assessment_inputs.cfg
    plot_cfg = assessment_inputs.plot_cfg
    pre_post_splitter = assessment_inputs.pre_post_splitter

    test_wtg = cfg.test_wtgs[0]
    test_pw_col = "pw_clipped"
    test_ws_col = "raw_WindSpeedMean"
    test_df = wf_df.loc[test_wtg.name].copy()
    test_name = test_wtg.name

    test_df.columns = ["test_" + x for x in test_df.columns]
    test_pw_col = "test_" + test_pw_col
    test_ws_col = "test_" + test_ws_col

    test_max_ws_drift, test_max_ws_drift_pp_period = check_windspeed_drift(
        wtg_df=test_df,
        wtg_name=test_name,
        ws_col=test_ws_col,
        reanalysis_ws_col="test_" + REANALYSIS_WS_COL,
        cfg=cfg,
        plot_cfg=plot_cfg,
    )

    scada_pc = pc_per_ttype[test_wtg.turbine_type.turbine_type]

    for ref_wtg in cfg.ref_wtgs:
        ref_name = ref_wtg.name
        ref_pw_col = "pw_clipped"
        ref_wd_col = "YawAngleMean"
        ref_ws_col = "ws_est_blend"
        ref_df = wf_df.loc[ref_name].copy()
        ref_max_northing_error_v_reanalysis = check_wtg_northing(
            ref_df,
            wtg_name=ref_name,
            north_ref_wd_col=REANALYSIS_WD_COL,
            timebase_s=cfg.timebase_s,
            plot_cfg=plot_cfg,
            sub_dir=f"{test_name}/{ref_name}",
        )
        ref_max_northing_error_v_wf = check_wtg_northing(
            ref_df,
            wtg_name=ref_name,
            north_ref_wd_col=WINDFARM_YAWDIR_COL,
            timebase_s=cfg.timebase_s,
            plot_cfg=plot_cfg,
            sub_dir=f"{test_name}/{ref_name}",
        )

        ref_pw_col = "ref_" + ref_pw_col
        ref_ws_col = "ref_" + ref_ws_col
        ref_wd_col = "ref_" + ref_wd_col
        ref_df.columns = ["ref_" + x for x in ref_df.columns]

        test_lat = test_wtg.latitude
        test_long = test_wtg.longitude
        ref_lat = ref_wtg.latitude
        ref_long = ref_wtg.longitude

        distance_m, bearing_deg = get_distance_and_bearing(
            lat1=test_lat,
            long1=test_long,
            lat2=ref_lat,
            long2=ref_long,
        )

        ref_max_ws_drift, ref_max_ws_drift_pp_period = check_windspeed_drift(
            wtg_df=ref_df,
            wtg_name=ref_name,
            ws_col=ref_ws_col,
            reanalysis_ws_col="ref_" + REANALYSIS_WS_COL,
            cfg=cfg,
            plot_cfg=plot_cfg,
            sub_dir=f"{test_name}/{ref_name}",
        )

        detrend_df = test_df.merge(ref_df, how="left", left_index=True, right_index=True)
        detrend_df = detrend_df[cfg.detrend_first_dt_utc_start : cfg.detrend_last_dt_utc_start]  # type: ignore[misc]

        detrend_df = add_waking_scen(
            test_name=test_name,
            ref_name=ref_name,
            test_ref_df=detrend_df,
            cfg=cfg,
            wf_df=wf_df,
            ref_wd_col=ref_wd_col,
            ref_lat=ref_lat,
            ref_long=ref_long,
        )

        plot_detrend_data_cov(
            cfg=cfg,
            test_name=test_name,
            ref_name=ref_name,
            test_df=test_df,
            test_ws_col=test_ws_col,
            ref_df=ref_df,
            ref_ws_col=ref_ws_col,
            ref_wd_col=ref_wd_col,
            detrend_df=detrend_df,
            plot_cfg=plot_cfg,
        )

        wsratio_v_dir_scen = calc_wsratio_v_wd_scen(
            test_name=test_name,
            ref_name=ref_name,
            ref_lat=ref_lat,
            ref_long=ref_long,
            detrend_df=detrend_df,
            test_ws_col=test_ws_col,
            ref_ws_col=ref_ws_col,
            ref_wd_col=ref_wd_col,
            cfg=cfg,
            plot_cfg=plot_cfg,
        )

        print(wsratio_v_dir_scen)


if __name__ == "__main__":
    setup_logger(ANALYSIS_OUTPUT_DIR / "analysis.log")
    main("messin around")