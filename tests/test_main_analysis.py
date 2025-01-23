import math

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from wind_up.constants import TIMESTAMP_COL
from wind_up.main_analysis import _toggle_pairing_filter


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
