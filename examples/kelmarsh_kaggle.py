"""Example submission for https://www.kaggle.com/competitions/predict-the-wind-speed-at-a-wind-turbine/."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import ephem
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from examples.helpers import setup_logger
from wind_up.constants import (
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
from wind_up.waking_state import add_waking_scen
from wind_up.windspeed_drift import check_windspeed_drift

DATA_DIR = Path("kelmarsh_kaggle_data")
OUTPUT_DIR = Path("kelmarsh_kaggle_output")
CACHE_DIR = Path("kelmarsh_kaggle_cache")
ASSESSMENT_NAME = "kelmarsh_kaggle"
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / ASSESSMENT_NAME
ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
logger = logging.getLogger(__name__)


class KelmarshKaggleScadaUnpacker:
    """Class to unpack the Kaggle Kelmarsh SCADA data."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.scada_df = None

    def unpack(self) -> pd.DataFrame:
        """Unpack the Kaggle Kelmarsh SCADA data."""
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
            _expected_n_turbines_train = 6
            assert train_scada_df["TurbineName"].nunique() == _expected_n_turbines_train
            _expected_n_turbines_test = 5
            assert test_scada_df["TurbineName"].nunique() == _expected_n_turbines_test
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
    """Return the metadata DataFrame for the Kelmarsh Kaggle competition."""
    metadata_df = pd.read_csv(data_dir / "metaData.csv")[["Title", "Latitude", "Longitude"]].rename(
        columns={"Title": "Name"}
    )
    return metadata_df.assign(TimeZone="UTC", TimeSpanMinutes=10, TimeFormat="Start")


def make_windup_features(analysis_name: str) -> None:
    """Run standard wind-up analysis up to directional detrending saving results to parquet files."""
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

    # confirmed by emailing Charlie Plumley that using ERA5 is allowed since it's public data
    # which would generally be available for any wind farm
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
    """Save the detrended dataframes for Kelmarsh 1 and the reference turbines.

    note most of this logic is copied from wind_up/main_analysis.py"""
    wf_df = assessment_inputs.wf_df
    cfg = assessment_inputs.cfg
    plot_cfg = assessment_inputs.plot_cfg

    wf_df.to_parquet(CACHE_DIR / cfg.assessment_name / "wf_df.parquet")

    test_wtg = cfg.test_wtgs[0]
    test_ws_col = "raw_WindSpeedMean"
    test_df = wf_df.loc[test_wtg.name].copy()
    test_name = test_wtg.name

    test_df.columns = ["test_" + x for x in test_df.columns]
    test_ws_col = "test_" + test_ws_col

    check_windspeed_drift(
        wtg_df=test_df,
        wtg_name=test_name,
        ws_col=test_ws_col,
        reanalysis_ws_col="test_" + REANALYSIS_WS_COL,
        cfg=cfg,
        plot_cfg=plot_cfg,
    )

    for ref_wtg in cfg.ref_wtgs:
        ref_name = ref_wtg.name
        ref_wd_col = "YawAngleMean"
        ref_ws_col = "ws_est_blend"
        ref_df = wf_df.loc[ref_name].copy()
        check_wtg_northing(
            ref_df,
            wtg_name=ref_name,
            north_ref_wd_col=REANALYSIS_WD_COL,
            timebase_s=cfg.timebase_s,
            plot_cfg=plot_cfg,
            sub_dir=f"{test_name}/{ref_name}",
        )
        check_wtg_northing(
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

        ref_lat = ref_wtg.latitude
        ref_long = ref_wtg.longitude

        check_windspeed_drift(
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

        detrend_ws_col = "ref_ws_detrended"
        detrend_df = apply_wsratio_v_wd_scen(
            detrend_df, wsratio_v_dir_scen, ref_ws_col=ref_ws_col, ref_wd_col=ref_wd_col
        )
        check_applied_detrend(
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
        logger.info(f"{len(dropna_df)=}")
        mae = (dropna_df[test_ws_col] - dropna_df[detrend_ws_col]).abs().mean()
        logger.info(f"{mae=}")


def create_time_based_features(
    df: pd.DataFrame, *, timestamp_col: str | tuple[str, str], dataset_min_timestamp: pd.Timestamp
) -> pd.DataFrame:
    """Create time-based features from a timestamp column."""
    # make a copy of the timestamp column to work with
    timestamps = pd.to_datetime(df[timestamp_col])
    # Calculate continuous month value (1-12.999...)
    days_in_month = timestamps.dt.days_in_month
    continuous_month = (
        timestamps.dt.month
        + (timestamps.dt.day - 1) / days_in_month
        + timestamps.dt.hour / (24 * days_in_month)
        + timestamps.dt.minute / (24 * 60 * days_in_month)
    )
    continuous_hour = timestamps.dt.hour + timestamps.dt.minute / 60 + timestamps.dt.second / 3600
    return pd.DataFrame(
        {
            ("age in days", "Time"): (timestamps - dataset_min_timestamp).dt.total_seconds() / (24 * 3600),
            ("hour", "Time"): continuous_hour,
            ("month", "Time"): continuous_month,
            # Cyclical encoding of relevant features
            ("month_sin", "Time"): np.sin(2 * np.pi * (continuous_month - 1) / 12),
            ("month_cos", "Time"): np.cos(2 * np.pi * (continuous_month - 1) / 12),
            ("hour_sin", "Time"): np.sin(2 * np.pi * continuous_hour / 24),
            ("hour_cos", "Time"): np.cos(2 * np.pi * continuous_hour / 24),
        }
    )


def validate_and_rename_reanalysis_cols(
    x_df: pd.DataFrame, expected_reanalysis_cols: list[tuple[str, str]]
) -> pd.DataFrame:
    """Validate and rename reanalysis columns."""
    x_df = x_df.copy()
    rename_mapping = {}
    for col in expected_reanalysis_cols:
        if col not in x_df.columns:
            msg = f"Expected {col} in columns"
            raise ValueError(msg)
        rename_mapping[col] = (col[0].replace("ref_", ""), "Reanalysis")
    # Convert columns to list for manipulation
    new_columns = x_df.columns.to_list()
    # Replace each column name based on mapping
    for i, col in enumerate(new_columns):
        if col in rename_mapping:
            new_columns[i] = rename_mapping[col]

    # Assign new column names
    x_df.columns = pd.MultiIndex.from_tuples(new_columns, names=x_df.columns.names)

    return x_df


def add_cos_and_sin_of_direction_features(
    x_df: pd.DataFrame, *, direction_deg_cols: list[tuple[str, str]]
) -> pd.DataFrame:
    """Add sine and cosine of features which are directions in degrees (0-360)."""
    x_df = x_df.copy()
    for col in direction_deg_cols:
        angles_deg = x_df[col]
        if any(angles_deg < 0) or any(angles_deg > 360):  # noqa:PLR2004
            msg = f"Expected angles to be in range [0,360] for {col}"
            raise ValueError(msg)
        x_df[(f"{col[0]}_cos", col[1])] = np.cos(np.radians(angles_deg))
        x_df[(f"{col[0]}_sin", col[1])] = np.sin(np.radians(angles_deg))
    return x_df


def sun_alt(
    row: pd.Series,
    *,
    observer: ephem.Observer,
    latitude: float,
    longitude: float,
    utc_timestamp_col: str | tuple[str, str],
    time_shift: pd.Timedelta,
) -> float:
    """Calculate sun altitude for a given row in a DataFrame.

    This code was adapted from https://github.com/NREL/flasc"""
    observer.lat = str(latitude)
    observer.long = str(longitude)
    observer.date = row[utc_timestamp_col] + time_shift
    sun = ephem.Sun()
    sun.compute(observer)
    return float(sun.alt) * 180 / np.pi


def add_sun_alt_to_df(
    input_df: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
    utc_timestamp_col: str | tuple[str, str],
    time_shift: pd.Timedelta,
) -> pd.DataFrame:
    """Calculate sun altitude for a given row in a DataFrame.

    This code was adapted from https://github.com/NREL/flasc"""
    out_df = input_df.copy()
    observer = ephem.Observer()
    return out_df.assign(
        sun_altitude=out_df.apply(
            lambda x: sun_alt(
                x,
                observer=observer,
                latitude=latitude,
                longitude=longitude,
                utc_timestamp_col=utc_timestamp_col,
                time_shift=time_shift,
            ),
            axis=1,
        )
    )


def make_kelmarsh_kaggle_y_train_x_train_x_test(
    *, recacl_windup_files: bool = False, look_at_feature_importance: bool = False
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    msg = "Making wind-up features and saving to parquet"
    logger.info(msg)
    assessment_name = "kelmarsh_kaggle"
    if recacl_windup_files:
        make_windup_features(assessment_name)

    msg = "Loading kaggle competition input data"
    logger.info(msg)

    train_df = pd.read_csv(DATA_DIR / "train.csv", header=[0, 1], index_col=[0], parse_dates=[1])
    test_df = pd.read_csv(DATA_DIR / "test.csv", header=[0, 1], index_col=[0], parse_dates=[1])

    msg = "Making classic y_train, X_train and X_test pandas Series/DataFrames"
    logger.info(msg)
    target_column = ("target_feature", "Unnamed: 53_level_1")
    y_train = train_df[target_column]
    X_train = train_df.drop(columns=[target_column])
    X_test = test_df

    msg = "Adding time based features"
    logger.info(msg)

    # create time-based features
    timestamp_col = ("Timestamp", "Unnamed: 1_level_1")
    dataset_min_timestamp = min(X_train[timestamp_col].min(), X_test[timestamp_col].min())
    time_features_train = create_time_based_features(
        X_train, timestamp_col=timestamp_col, dataset_min_timestamp=dataset_min_timestamp
    )
    time_features_test = create_time_based_features(
        X_test, timestamp_col=timestamp_col, dataset_min_timestamp=dataset_min_timestamp
    )

    # Add these new features to X_train and X_test
    X_train = pd.concat([X_train, time_features_train], axis=1)
    X_test = pd.concat([X_test, time_features_test], axis=1)

    msg = "Adding wind-up features"
    logger.info(msg)

    # load in parquet files and add features
    ref_turbines_to_use = ["Kelmarsh 2", "Kelmarsh 3", "Kelmarsh 4", "Kelmarsh 5", "Kelmarsh 6"]
    for ref_wtg_name in ref_turbines_to_use:
        detrend_df = pd.read_parquet(CACHE_DIR / assessment_name / f"Kelmarsh 1_{ref_wtg_name}_detrend.parquet")
        detrend_df_no_tz = detrend_df.copy()
        detrend_df_no_tz.index = detrend_df_no_tz.index.tz_localize(None)
        assert detrend_df_no_tz.index.min() == min(
            X_train[("Timestamp", "Unnamed: 1_level_1")].min(), X_test[("Timestamp", "Unnamed: 1_level_1")].min()
        )
        assert detrend_df_no_tz.index.max() == max(
            X_train[("Timestamp", "Unnamed: 1_level_1")].max(), X_test[("Timestamp", "Unnamed: 1_level_1")].max()
        )
        assert len(detrend_df) == len(X_train) + len(X_test)
        detrend_df_to_merge = detrend_df_no_tz.copy()
        # add a column level with the ref_wtg_name
        detrend_df_to_merge.columns = pd.MultiIndex.from_tuples(
            [(x, ref_wtg_name) for x in detrend_df_to_merge.columns]
        )
        reanalysis_cols = ["ref_reanalysis_ws", "ref_reanalysis_wd"]
        features_to_add = [
            ("ref_ws_detrended", ref_wtg_name),
        ] + [(col, ref_wtg_name) for col in reanalysis_cols]

        for col in features_to_add:
            if col not in detrend_df_to_merge.columns:
                continue
            if col in X_train.columns:
                continue
            if col[0] in reanalysis_cols and col[1] != ref_turbines_to_use[0]:
                continue  # only add these columns for Kelmarsh 2
            # Create temporary timestamp columns for merging
            timestamp_train = X_train[("Timestamp", "Unnamed: 1_level_1")].copy()
            timestamp_test = X_test[("Timestamp", "Unnamed: 1_level_1")].copy()
            X_train = X_train.merge(
                detrend_df_to_merge[[col]], left_on=timestamp_train, right_index=True, how="left"
            ).drop(columns=("key_0", ""))
            X_test = X_test.merge(
                detrend_df_to_merge[[col]], left_on=timestamp_test, right_index=True, how="left"
            ).drop(columns=("key_0", ""))
            logger.info(f"Added {col} to X_train and X_test")

    expected_reanalysis_cols = [(x, ref_turbines_to_use[0]) for x in reanalysis_cols]
    X_train = validate_and_rename_reanalysis_cols(X_train, expected_reanalysis_cols)
    X_test = validate_and_rename_reanalysis_cols(X_test, expected_reanalysis_cols)

    msg = "adding cos and sin of direction features"
    logger.info(msg)

    direction_cols = [
        x for x in X_test.columns if x[0] in {"Wind direction (°)", "Nacelle position (°)", "reanalysis_wd"}
    ]
    X_train = add_cos_and_sin_of_direction_features(X_train, direction_deg_cols=direction_cols)
    X_test = add_cos_and_sin_of_direction_features(X_test, direction_deg_cols=direction_cols)

    msg = "adding solar position features"
    logger.info(msg)
    kelmarsh_metadata = pd.read_csv(DATA_DIR / "metaData.csv")
    kelmarsh_latitude = kelmarsh_metadata["Latitude"].mean()
    kelmarsh_longitude = kelmarsh_metadata["Longitude"].mean()
    X_train = add_sun_alt_to_df(
        X_train,
        latitude=kelmarsh_latitude,
        longitude=kelmarsh_longitude,
        utc_timestamp_col=timestamp_col,
        time_shift=pd.Timedelta(hours=0),
    )
    X_test = add_sun_alt_to_df(
        X_test,
        latitude=kelmarsh_latitude,
        longitude=kelmarsh_longitude,
        utc_timestamp_col=timestamp_col,
        time_shift=pd.Timedelta(hours=0),
    )

    msg = "Ensure X_train and X_test are consistent and numeric only"
    logger.info(msg)

    # Find the common columns
    common_columns = X_train.columns.intersection(X_test.columns)

    # Print info about the difference
    logger.info(f"Columns only in X_train: {set(X_train.columns) - set(X_test.columns)}")
    logger.info(f"Columns only in X_test: {set(X_test.columns) - set(X_train.columns)}")
    logger.info(f"Number of common columns: {len(common_columns)}")

    # Keep only the common columns in both datasets
    X_train = X_train[common_columns]
    X_test = X_test[common_columns]

    # Get numeric columns from both dataframes
    numeric_columns = X_train.select_dtypes(include=["int32", "int64", "float64"]).columns
    logger.info(f"Number of numeric columns: {len(numeric_columns)}")
    logger.info(f"Removed columns: {set(X_train.columns) - set(numeric_columns)}")

    # Keep only numeric columns
    X_train = X_train[numeric_columns]
    X_test = X_test[numeric_columns]

    # Verify the shapes and dtypes
    logger.info("\nNew shapes after dropping non numeric columns:")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")

    # Verify all columns are numeric
    logger.info("\nColumn dtypes in X_train:")
    logger.info(X_train.dtypes.value_counts())

    if look_at_feature_importance:
        logger.info("Calculating feature importances using Random Forest")
        mask = ~y_train.isna()
        X_train_dropna = X_train[mask]
        y_train_dropna = y_train[mask]
        X_imputed = pd.DataFrame(
            SimpleImputer(strategy="mean").fit_transform(X_train_dropna),
            columns=X_train_dropna.columns,
            index=X_train_dropna.index,
        )
        model = RandomForestRegressor(random_state=0)
        logger.info("Fitting")
        model.fit(X_imputed, y_train_dropna)
        # Log feature importance details
        importance_df = pd.DataFrame(
            {"feature": X_train_dropna.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        logger.info("\nRandom Forest Feature Importances (top 20):")
        logger.info(importance_df.head(20))

        logger.info("\nRandom Forest Feature Importances (all):")
        logger.info(importance_df)
        importance_df.to_csv(ANALYSIS_OUTPUT_DIR / f"feature_importances_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv")

    logger.info(f"Number of rows in y_train: {len(y_train)}")
    logger.info(f"Number of NaN values in y_train: {y_train.isna().sum()}")
    msg = "Removing nan rows from y_train (and X_train)"
    logger.info(msg)
    mask = ~y_train.isna()
    X_train = X_train[mask]
    y_train = y_train[mask]

    logger.info("\nNew shapes after removing nan rows from y_train (and X_train):")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    return y_train, X_train, X_test


def flatten_and_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # First join the levels with an underscore
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = ["_".join(map(str, col)).strip() for col in df.columns]
    else:
        flat_cols = df.columns.tolist()

    # Then clean any remaining special characters
    clean_cols = [re.sub(r"[^A-Za-z0-9_]+", "_", col).strip("_") for col in flat_cols]

    # Ensure names are unique
    seen = set()
    unique_cols = []
    for col in clean_cols:
        if col in seen:
            counter = 1
            while f"{col}_{counter}" in seen:
                counter += 1
            col = f"{col}_{counter}"  # noqa:PLW2901
        seen.add(col)
        unique_cols.append(col)

    df.columns = unique_cols
    return df


if __name__ == "__main__":
    setup_logger(ANALYSIS_OUTPUT_DIR / f"automl_{pd.Timestamp.now():%Y%m%d_%H%M%S}.log")

    recacl_windup_files = True  # set to False after running one time to generate wind-up results
    y_train, X_train, X_test = make_kelmarsh_kaggle_y_train_x_train_x_test(recacl_windup_files=recacl_windup_files)

    # avoids lightgbm.basic.LightGBMError: Do not support special JSON characters in feature name.
    X_train = flatten_and_clean_columns(X_train)
    X_test = flatten_and_clean_columns(X_test)

    automl_settings = {
        "time_budget": 3600 * 1,  # 12 hours was used for best solution
        "ensemble": True,
        "task": "regression",
        "metric": "mae",  # kaggle competition uses MAE
        "estimator_list": [
            "lgbm",
            "xgboost",
            "xgb_limitdepth",
            "rf",
            "extra_tree",
        ],  # "catboost" was also in the list for the best solution but seems to need numpy <2.0 at present
        "log_file_name": "automl.log",
        "seed": 0,
    }
    automl = AutoML()

    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
    msg = f"{automl.best_config_per_estimator=}"
    logger.info(msg)

    new_starting_points = automl.best_config_per_estimator
    logger.info(f"{new_starting_points=}")

    y_test_pred = automl.predict(X_test)

    submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    submission.iloc[:, 1] = y_test_pred
    output_path = ANALYSIS_OUTPUT_DIR / f"automl_submission_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
    submission.to_csv(output_path, index=False)
    msg = f"Submission saved to {output_path}"
    logger.info(msg)
