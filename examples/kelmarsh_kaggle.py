"""Example submission for https://www.kaggle.com/competitions/predict-the-wind-speed-at-a-wind-turbine/"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from examples.helpers import setup_logger
from examples.kaggle_pipeline import prepare_submission
from wind_up.constants import (
    OUTPUT_DIR,
    PROJECTROOT_DIR,
    REANALYSIS_WD_COL,
    REANALYSIS_WS_COL,
    TIMESTAMP_COL,
    WINDFARM_YAWDIR_COL,
    DataColumns,
)
from wind_up.detrend import apply_wsratio_v_wd_scen, calc_wsratio_v_wd_scen, check_applied_detrend
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
            # unpack train.csv
            raw_df = pd.read_csv(self.data_dir / "train.csv", header=[0, 1], index_col=[0], parse_dates=[1])
            id_df = self._format_index(
                raw_df[[("Timestamp", "Unnamed: 1_level_1")]]
                .reset_index()
                .set_index(("Timestamp", "Unnamed: 1_level_1"))
                .droplevel(1, axis=1)
            )
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
            train_scada_df = self._format_scada_df(scada_df=workings_df)
            train_scada_df = train_scada_df.merge(id_df, left_index=True, right_index=True)

            # unpack test.csv
            raw_df = pd.read_csv(self.data_dir / "test.csv", header=[0, 1], index_col=[0], parse_dates=[1])
            id_df = self._format_index(
                raw_df[[("Timestamp", "Unnamed: 1_level_1")]]
                .reset_index()
                .set_index(("Timestamp", "Unnamed: 1_level_1"))
                .droplevel(1, axis=1)
            )
            workings_df = raw_df.set_index(("Timestamp", "Unnamed: 1_level_1"))
            new_cols = pd.MultiIndex.from_tuples(
                [
                    ("Wind speed (m/s)", "Kelmarsh 1") if x == ("target_feature", "Unnamed: 53_level_1") else x
                    for x in workings_df.columns
                ]
            )
            workings_df.columns = new_cols
            workings_df = workings_df.stack().swaplevel(0, 1, axis=0)
            workings_df = workings_df.reset_index(level=0, names="TurbineName")
            test_scada_df = self._format_scada_df(scada_df=workings_df)
            test_scada_df = test_scada_df.merge(id_df, left_index=True, right_index=True)
            # verify train_scada_df has 6 turbines
            assert train_scada_df["TurbineName"].nunique() == 6
            # verify test_scada_df has 5 turbines
            assert test_scada_df["TurbineName"].nunique() == 5
            # verify train_scada_df and test_scada_df have the same columns
            assert train_scada_df.columns.equals(test_scada_df.columns)
            # verify train_scada_df and test_scada_df have no matching DatetimeIndex index entries
            assert len(train_scada_df.index.intersection(test_scada_df.index)) == 0
            # verify train_scada_df and test_scada_df have no matching values in id column
            assert len(set(train_scada_df["id"]) & set(test_scada_df["id"])) == 0
            # combine train_scada_df and test_scada_df
            self.scada_df = pd.concat([train_scada_df, test_scada_df], axis=0).sort_index()
        return self.scada_df

    def _format_index(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()
        new_df.index.name = TIMESTAMP_COL
        # make index UTC
        new_df.index = new_df.index.tz_localize("UTC")
        return new_df

    def _format_scada_df(self, scada_df: pd.DataFrame) -> pd.DataFrame:
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
        return self._format_index(scada_df)


def kelmarsh_kaggle_metadata_df(data_dir: Path) -> pd.DataFrame:
    metadata_df = pd.read_csv(data_dir / "metaData.csv")[["Title", "Latitude", "Longitude"]].rename(
        columns={"Title": "Name"}
    )
    return metadata_df.assign(TimeZone="UTC", TimeSpanMinutes=10, TimeFormat="Start")


def make_windup_features(analysis_name: str) -> None:
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
        test_wtgs=[wtg_map[x] for x in ["Kelmarsh 1"]],
        ref_wtgs=[wtg_map[x] for x in ["Kelmarsh 2", "Kelmarsh 3", "Kelmarsh 4", "Kelmarsh 5", "Kelmarsh 6"]],
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

    save_t1_detrend_dfs(assessment_inputs)


def save_t1_detrend_dfs(assessment_inputs: AssessmentInputs) -> None:
    wf_df = assessment_inputs.wf_df
    pc_per_ttype = assessment_inputs.pc_per_ttype
    cfg = assessment_inputs.cfg
    plot_cfg = assessment_inputs.plot_cfg

    wf_df.to_parquet(CACHE_DIR / cfg.assessment_name / "wf_df.parquet")

    test_wtg = cfg.test_wtgs[0]
    test_ws_col = "raw_WindSpeedMean"
    test_df = wf_df.loc[test_wtg.name].copy()
    test_name = test_wtg.name

    test_df.columns = ["test_" + x for x in test_df.columns]
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

        detrend_ws_col = "ref_ws_detrended"
        detrend_df = apply_wsratio_v_wd_scen(
            detrend_df, wsratio_v_dir_scen, ref_ws_col=ref_ws_col, ref_wd_col=ref_wd_col
        )
        detrend_r2_improvement, _ = check_applied_detrend(
            test_name=test_name,
            ref_name=ref_name,
            ref_lat=ref_lat,
            ref_long=ref_long,
            pre_df=detrend_df,
            post_df=detrend_df,
            test_ws_col=test_ws_col,
            ref_ws_col=ref_ws_col,
            detrend_ws_col=detrend_ws_col,
            ref_wd_col=ref_wd_col,
            cfg=cfg,
            plot_cfg=plot_cfg,
        )

        detrend_df.to_parquet(CACHE_DIR / cfg.assessment_name / f"{test_name}_{ref_name}_detrend.parquet")
        # compare detrend_ws_col to test_ws_col
        dropna_df = detrend_df.dropna(subset=[detrend_ws_col, test_ws_col])
        plt.figure()
        plt.scatter(dropna_df[test_ws_col], dropna_df[detrend_ws_col], s=1)
        plt.ylabel(detrend_ws_col)
        plt.xlabel(test_ws_col)
        title = f"test={test_name} ref={ref_name} {detrend_ws_col} vs {test_ws_col}"
        plt.title(title)
        plt.grid()
        plt.tight_layout()
        plt.savefig(plot_cfg.plots_dir / f"{title}.png")
        print(f"{len(dropna_df)=}")
        mae = (dropna_df[test_ws_col] - dropna_df[detrend_ws_col]).abs().mean()
        print(f"{mae=}")


if __name__ == "__main__":
    setup_logger(ANALYSIS_OUTPUT_DIR / "analysis.log")
    recacl_windup_files = False
    if recacl_windup_files:
        make_windup_features("messin around 2")

    train_df = pd.read_csv(DATA_DIR / "train.csv", header=[0, 1], index_col=[0], parse_dates=[1])
    test_df = pd.read_csv(DATA_DIR / "test.csv", header=[0, 1], index_col=[0], parse_dates=[1])

    # Separate features and target
    target_column = ("target_feature", "Unnamed: 53_level_1")
    y_train = train_df[target_column]
    X_train = train_df.drop(columns=[target_column])
    X_test = test_df

    # First make a copy of the timestamp column to work with
    timestamp_train = pd.to_datetime(X_train[("Timestamp", "Unnamed: 1_level_1")])
    timestamp_test = pd.to_datetime(X_test[("Timestamp", "Unnamed: 1_level_1")])

    # Create multiple time-based features
    time_features_train = pd.DataFrame(
        {
            ("Time", "hour"): timestamp_train.dt.hour,
            ("Time", "month"): timestamp_train.dt.month,
            ("Time", "hour_sin"): np.sin(2 * np.pi * timestamp_train.dt.hour / 24),  # Cyclical encoding
            ("Time", "hour_cos"): np.cos(2 * np.pi * timestamp_train.dt.hour / 24),
        }
    )

    time_features_test = pd.DataFrame(
        {
            ("Time", "hour"): timestamp_test.dt.hour,
            ("Time", "month"): timestamp_test.dt.month,
            ("Time", "hour_sin"): np.sin(2 * np.pi * timestamp_test.dt.hour / 24),
            ("Time", "hour_cos"): np.cos(2 * np.pi * timestamp_test.dt.hour / 24),
        }
    )

    # Add these new features to X_train and X_test
    X_train = pd.concat([X_train, time_features_train], axis=1)
    X_test = pd.concat([X_test, time_features_test], axis=1)

    # Find the common columns
    common_columns = X_train.columns.intersection(X_test.columns)

    # Print info about the difference
    print(f"Columns only in X_train: {set(X_train.columns) - set(X_test.columns)}")
    print(f"Columns only in X_test: {set(X_test.columns) - set(X_train.columns)}")
    print(f"Number of common columns: {len(common_columns)}")

    # Keep only the common columns in both datasets
    X_train = X_train[common_columns]
    X_test = X_test[common_columns]

    # Get numeric columns from both dataframes
    numeric_columns = X_train.select_dtypes(include=["int32", "int64", "float64"]).columns
    print(f"Number of numeric columns: {len(numeric_columns)}")
    print(f"Removed columns: {set(X_train.columns) - set(numeric_columns)}")

    # Keep only numeric columns
    X_train = X_train[numeric_columns]
    X_test = X_test[numeric_columns]

    # Verify the shapes and dtypes
    print("\nNew shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Verify all columns are numeric
    print("\nColumn dtypes in X_train:")
    print(X_train.dtypes.value_counts())

    # Verify the shapes
    print("\nNew shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    feature_names = [" ".join(col).strip() for col in X_train.columns]

    print(f"Number of NaN values in target: {y_train.isna().sum()}")
    print(f"Total samples: {len(y_train)}")

    mask = ~y_train.isna()
    X_train = X_train[mask]
    y_train = y_train[mask]

    # First, evaluate all feature selection methods
    pipeline = prepare_submission(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        sample_submission_path=ANALYSIS_OUTPUT_DIR / "sample_submission.csv",
        evaluate_features=True,
    )

    # Then, use the best method for your final model
    pipeline = prepare_submission(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        sample_submission_path=ANALYSIS_OUTPUT_DIR / "sample_submission.csv",
        evaluate_features=False,
        feature_method="model_based",  # or 'mutual_info' or 'boruta'
    )
