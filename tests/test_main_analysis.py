import logging
import math

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal
from pytest_benchmark.fixture import BenchmarkFixture

from tests.test_data.hot.data_loader import WindUpComponents
from wind_up.constants import TIMESTAMP_COL, DataColumns
from wind_up.main_analysis import _filter_turbine_df_by_other_turbine_dfs, _toggle_pairing_filter
from wind_up.models import WindUpConfig
from wind_up.plots.scada_funcs_plots import print_filter_stats


def test_toggle_pairing_filter_method_none() -> None:
    pre_tstamps = pd.date_range(start="2021-01-01 00:00:00", tz="UTC", periods=9, freq="10min")
    post_tstamps = pd.date_range(start=pre_tstamps.max() + pd.Timedelta("10min"), tz="UTC", periods=9, freq="10min")
    detrend_ws_col = "ref_ws_detrended"
    test_pw_col = "test_pw_clipped"
    ref_wd_col = "ref_YawAngleMean"

    pre_df = pd.DataFrame(
        data={
            detrend_ws_col: [5.1, 5.1, 5.1, 0.0, 0.0, 0.0, np.nan, 5.1, 5.1],
            test_pw_col: [5.1, 5.1, np.nan, 0.0, 0.0, 0.0, 5.1, 5.1, 5.1],
            ref_wd_col: [5.1, 5.1, 5.1, 0.0, 0.0, 0.0, 5.1, 5.1, np.nan],
            "some_col_with_nans": [np.nan] * 9,
        },
        index=pre_tstamps,
    )

    post_df = pd.DataFrame(
        data={
            detrend_ws_col: [5.1, 5.1, 5.1, 0.0, 0.0, 0.0, np.nan, 5.1, 5.1],
            test_pw_col: [5.1, 5.1, np.nan, 0.0, 0.0, 0.0, 5.1, 5.1, 5.1],
            ref_wd_col: [5.1, 5.1, 5.1, 0.0, 0.0, 0.0, 5.1, 5.1, np.nan],
            "some_col_with_nans": [np.nan] * 9,
        },
        index=post_tstamps,
    )

    filt_pre_df, filt_post_df = _toggle_pairing_filter(
        pre_df=pre_df,
        post_df=post_df,
        pairing_filter_method="none",
        pairing_filter_timedelta_seconds=0,
        detrend_ws_col=detrend_ws_col,
        test_pw_col=test_pw_col,
        ref_wd_col=ref_wd_col,
        timebase_s=600,
    )
    assert_frame_equal(filt_pre_df, pre_df)
    assert_frame_equal(filt_post_df, post_df)


def test_toggle_pairing_filter_method_any_within_timedelta() -> None:
    pre_tstamps = pd.date_range(start="2021-01-01 00:00:00", tz="UTC", periods=9, freq="10min")
    post_tstamps = pd.date_range(start=pre_tstamps.max() + pd.Timedelta("10min"), tz="UTC", periods=9, freq="10min")
    detrend_ws_col = "ref_ws_detrended"
    test_pw_col = "test_pw_clipped"
    ref_wd_col = "ref_YawAngleMean"

    pre_df = pd.DataFrame(
        data={
            detrend_ws_col: [5.1, 5.1, 5.1, 0.0, 0.0, 0.0, np.nan, 5.1, 5.1],
            test_pw_col: [5.1, 5.1, np.nan, 0.0, 0.0, 0.0, 5.1, 5.1, 5.1],
            ref_wd_col: [5.1, 5.1, 5.1, 0.0, 0.0, 0.0, 5.1, 5.1, np.nan],
            "some_col_with_nans": [np.nan] * 9,
        },
        index=pre_tstamps,
    )
    pre_df.index.name = TIMESTAMP_COL
    post_df = pd.DataFrame(
        data={
            detrend_ws_col: [5.1, 5.1, 5.1, 0.0, 0.0, 0.0, np.nan, 5.1, 5.1],
            test_pw_col: [5.1, 5.1, np.nan, 0.0, 0.0, 0.0, 5.1, 5.1, 5.1],
            ref_wd_col: [5.1, 5.1, 5.1, 0.0, 0.0, 0.0, 5.1, 5.1, np.nan],
            "some_col_with_nans": [np.nan] * 9,
        },
        index=post_tstamps,
    )
    post_df.index.name = TIMESTAMP_COL

    pairing_filter_timedelta_seconds = 50 * 60
    exp_filt_pre_df = pre_df.copy()
    exp_filt_pre_df = exp_filt_pre_df.dropna(subset=[detrend_ws_col, test_pw_col, ref_wd_col])
    exp_filt_post_df = post_df.copy()
    exp_filt_post_df = exp_filt_post_df.dropna(subset=[detrend_ws_col, test_pw_col, ref_wd_col])
    a = exp_filt_pre_df.copy()
    b = exp_filt_post_df.copy()
    # Set the tolerance in minutes (change this value according to your requirements)
    tolerance_minutes = 50

    def copy_of_make_extended_time_index(
        original_index: pd.DatetimeIndex,
        timebase: pd.Timedelta,
        max_timedelta_seconds: int,
    ) -> pd.DatetimeIndex:
        extended_index = original_index
        timedelta_multiple = -math.floor(max_timedelta_seconds / timebase.total_seconds())
        max_timedelta_multiple = math.floor(max_timedelta_seconds / timebase.total_seconds())
        while timedelta_multiple <= max_timedelta_multiple:
            shifted_index = original_index + (timebase * timedelta_multiple)
            extended_index = extended_index.union(shifted_index)
            timedelta_multiple += 1
        return extended_index.sort_values().drop_duplicates()

    exp_filt_pre_df = a[
        [x in copy_of_make_extended_time_index(b.index, pd.Timedelta("10min"), tolerance_minutes * 60) for x in a.index]
    ]
    exp_filt_post_df = b[
        [x in copy_of_make_extended_time_index(a.index, pd.Timedelta("10min"), tolerance_minutes * 60) for x in b.index]
    ]
    filt_pre_df, filt_post_df = _toggle_pairing_filter(
        pre_df=pre_df,
        post_df=post_df,
        pairing_filter_method="any_within_timedelta",
        pairing_filter_timedelta_seconds=pairing_filter_timedelta_seconds,
        detrend_ws_col=detrend_ws_col,
        test_pw_col=test_pw_col,
        ref_wd_col=ref_wd_col,
        timebase_s=600,
    )
    assert_frame_equal(filt_pre_df, exp_filt_pre_df)
    assert_frame_equal(filt_post_df, exp_filt_post_df)


class TestFilterTurbineDfByOtherTurbineDfs:
    """Tests for filtering turbine dataframes by other turbine data."""

    @pytest.fixture
    def setup_data(self, hot_windup_components: WindUpComponents) -> dict:
        """Prepare data for all tests."""
        wf_df = hot_windup_components.scada_df.copy().reset_index().set_index([DataColumns.turbine_name, TIMESTAMP_COL])
        cfg = hot_windup_components.wind_up_config

        test_turbine_name = "T01"
        test_pw_col = DataColumns.active_power_mean
        test_ws_col = DataColumns.wind_speed_mean

        return {
            "cfg": cfg,
            "test_turbine_name": test_turbine_name,
            "test_pw_col": test_pw_col,
            "test_ws_col": test_ws_col,
            "wf_df": wf_df,
            "test_df_pandas": wf_df.loc[test_turbine_name].copy(),
            "test_df_polars": pl.from_pandas(wf_df.loc[test_turbine_name].reset_index()),
            "wf_df_polars": pl.from_pandas(wf_df.reset_index()),
        }

    # Correctness tests
    def test_pandas_implementation_correctness(self, hot_windup_components: WindUpComponents) -> None:
        """Test that pandas implementation produces correct results."""
        wf_df = hot_windup_components.scada_df.copy().reset_index().set_index([DataColumns.turbine_name, TIMESTAMP_COL])
        cfg = hot_windup_components.wind_up_config

        test_turbine_name = "T01"
        test_pw_col = DataColumns.active_power_mean
        test_ws_col = DataColumns.wind_speed_mean

        test_df = wf_df.loc[test_turbine_name]

        actual = filter_turbine_df_by_other_turbine_dfs_pandas(
            wind_up_cfg=cfg,
            turbine_name=test_turbine_name,
            df=test_df,
            pw_col=test_pw_col,
            ws_col=test_ws_col,
            windfarm_df=wf_df,
        )

        # Verify against polars version
        polars_test_df = pl.from_pandas(wf_df.loc[test_turbine_name].reset_index())
        expected = _filter_turbine_df_by_other_turbine_dfs(
            wind_up_cfg=cfg,
            test_turbine_name=test_turbine_name,
            test_df=polars_test_df,
            pw_col=test_pw_col,
            ws_col=test_ws_col,
            windfarm_df=pl.from_pandas(wf_df.reset_index()),
        )
        expected_pd = expected.to_pandas().set_index(TIMESTAMP_COL)

        assert_frame_equal(actual, expected_pd)

    def test_polars_implementation_correctness(
        self, hot_windup_components: WindUpComponents, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that polars implementation produces correct results and logging."""
        wf_df = hot_windup_components.scada_df.copy().reset_index().set_index([DataColumns.turbine_name, TIMESTAMP_COL])
        cfg = hot_windup_components.wind_up_config

        test_turbine_name = "T01"
        test_pw_col = DataColumns.active_power_mean
        test_ws_col = DataColumns.wind_speed_mean

        test_df = pl.from_pandas(wf_df.loc[test_turbine_name].reset_index())

        with caplog.at_level(logging.INFO):
            actual = _filter_turbine_df_by_other_turbine_dfs(
                wind_up_cfg=cfg,
                test_turbine_name=test_turbine_name,
                test_df=test_df,
                pw_col=test_pw_col,
                ws_col=test_ws_col,
                windfarm_df=pl.from_pandas(wf_df.reset_index()),
            )

        assert "filter_all_test_wtgs_together T13 set 2948 rows [1.5%] to NA" in caplog.text
        assert isinstance(actual, pl.DataFrame)

    # Benchmark tests
    def test_benchmark_pandas_version(self, benchmark: BenchmarkFixture, setup_data: dict) -> None:
        """Benchmark the pandas implementation."""

        def run_pandas() -> pd.DataFrame:
            # Create a fresh copy for each run to avoid mutation issues
            test_df = setup_data["test_df_pandas"].copy()
            return filter_turbine_df_by_other_turbine_dfs_pandas(
                wind_up_cfg=setup_data["cfg"],
                turbine_name=setup_data["test_turbine_name"],
                df=test_df,
                pw_col=setup_data["test_pw_col"],
                ws_col=setup_data["test_ws_col"],
                windfarm_df=setup_data["wf_df"],
            )

        result = benchmark(run_pandas)
        assert result is not None

    def test_benchmark_polars_version(self, benchmark: BenchmarkFixture, setup_data: dict) -> None:
        """Benchmark the polars implementation."""

        def run_polars() -> pl.DataFrame:
            # Polars DataFrames are immutable, so cloning is cheap
            test_df = setup_data["test_df_polars"].clone()
            return _filter_turbine_df_by_other_turbine_dfs(
                wind_up_cfg=setup_data["cfg"],
                test_turbine_name=setup_data["test_turbine_name"],
                test_df=test_df,
                pw_col=setup_data["test_pw_col"],
                ws_col=setup_data["test_ws_col"],
                windfarm_df=setup_data["wf_df_polars"],
            )

        result = benchmark(run_polars)
        assert result is not None


# Helper function
def filter_turbine_df_by_other_turbine_dfs_pandas(
    wind_up_cfg: WindUpConfig,
    turbine_name: str,
    df: pd.DataFrame,
    pw_col: str,
    ws_col: str,
    windfarm_df: pd.DataFrame,
) -> pd.DataFrame:
    """Pandas implementation for benchmarking comparison."""
    for other_test_wtg in wind_up_cfg.test_wtgs:
        if other_test_wtg.name == turbine_name:
            continue
        pw_na_before = df[DataColumns.active_power_mean].isna().sum()
        other_test_df = windfarm_df.loc[other_test_wtg.name]
        timestamps_to_filter = other_test_df[other_test_df[pw_col].isna() | other_test_df[ws_col].isna()].index
        cols_to_filter = list({pw_col, ws_col, DataColumns.active_power_mean, DataColumns.wind_speed_mean})
        df.loc[timestamps_to_filter, cols_to_filter] = pd.NA
        pw_na_after = df[DataColumns.active_power_mean].isna().sum()
        print_filter_stats(
            filter_name=f"filter_all_test_wtgs_together {other_test_wtg.name}",
            na_rows=pw_na_after - pw_na_before,
            total_rows=len(df),
        )
    return df
