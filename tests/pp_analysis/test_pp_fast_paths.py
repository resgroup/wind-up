"""Check the numpy paths of _pp_raw_df and _cook_pp against their pandas paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tests.conftest import TEST_DATA_FLD
from tests.pp_analysis.test_cook_pp import _make_pp_raw_df, _make_site_mean_pc_df
from wind_up.pp_analysis import _cook_pp, _cook_pp_with_pandas, _cut, _pp_raw_df, _pp_raw_df_with_pandas
from wind_up.result_manager import result_manager

WS_COL = "ref_ws_detrended"
PW_COL = "test_pw_clipped"
RAW_KWARGS = {"ws_col": WS_COL, "ws_bin_edges": np.arange(0, 26, 1.0), "pw_col": PW_COL, "timebase_s": 600}


def _pre_df() -> pd.DataFrame:
    return pd.read_parquet(TEST_DATA_FLD / "LSA_T13_LSA_T12_pre_df.parquet", columns=[WS_COL, PW_COL])


def _cook_both_ways(pp_df: pd.DataFrame, **kwargs: object) -> tuple[pd.DataFrame, list, pd.DataFrame, list]:
    result_manager.stored_warnings = []
    actual = _cook_pp(pp_df, **kwargs)
    actual_warnings = result_manager.stored_warnings
    result_manager.stored_warnings = []
    expected = _cook_pp_with_pandas(pp_df, **kwargs)
    expected_warnings = result_manager.stored_warnings
    result_manager.stored_warnings = []
    return actual, actual_warnings, expected, expected_warnings


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d,
        lambda d: d[(d[WS_COL] < 6) | (d[WS_COL] > 9)].iloc[::5],  # invalid bins and a gap to interpolate across
    ],
)
@pytest.mark.parametrize(("clip_to_rated", "with_site_mean_pc"), [(True, False), (False, True)])
def test_numpy_paths_match_pandas_paths(change: callable, clip_to_rated: bool, with_site_mean_pc: bool) -> None:  # noqa: FBT001
    df = change(_pre_df())

    raw = _pp_raw_df(df, "pre", **RAW_KWARGS)
    assert_frame_equal(raw, _pp_raw_df_with_pandas(df, "pre", **RAW_KWARGS), check_exact=True)

    actual, actual_warnings, expected, expected_warnings = _cook_both_ways(
        raw,
        pre_or_post="pre",
        ws_bin_width=1.0,
        rated_power=2000.0,
        clip_to_rated=clip_to_rated,
        site_mean_pc_df=_make_site_mean_pc_df(rated_power=2000.0) if with_site_mean_pc else None,
    )
    assert_frame_equal(actual, expected, check_exact=True)
    assert actual_warnings == expected_warnings


def test_cook_pp_numpy_path_warns_as_pandas_path() -> None:
    bin_mids = [float(m) for m in range(1, 26)]
    pp_df = _make_pp_raw_df(bin_mids, [min(1300.0, max(0.0, 130.0 * (m - 3))) for m in bin_mids], [50.0] * 25)
    pp_df["pw_sem_pre"] = np.nan

    actual, actual_warnings, expected, expected_warnings = _cook_both_ways(
        pp_df, pre_or_post="pre", ws_bin_width=1.0, rated_power=1300.0, clip_to_rated=False
    )

    assert_frame_equal(actual, expected, check_exact=True)
    assert actual_warnings == expected_warnings == ["pp_df has missing values"]


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.astype(np.float32),
        lambda d: d.astype("Float64"),
        lambda d: d.dropna().astype({PW_COL: np.int64}),
        lambda d: pd.concat([d, d.iloc[:100]]).dropna(),
    ],
    ids=["float32", "Float64", "int64", "repeated timestamps"],
)
def test_other_inputs_take_the_pandas_paths(change: callable) -> None:
    df = change(_pre_df().iloc[:5000])

    raw = _pp_raw_df(df, "pre", **RAW_KWARGS)
    assert_frame_equal(raw, _pp_raw_df_with_pandas(df, "pre", **RAW_KWARGS), check_exact=True)

    raw = raw.astype({"pw_mean_pre": np.float32}) if raw["pw_mean_pre"].dtype == np.float64 else raw
    kwargs = {"pre_or_post": "pre", "ws_bin_width": 1.0, "rated_power": 2000.0, "clip_to_rated": True}
    assert_frame_equal(_cook_pp(raw, **kwargs), _cook_pp_with_pandas(raw, **kwargs), check_exact=True)
    result_manager.stored_warnings = []


def test_cook_pp_repeated_bin_takes_the_pandas_path() -> None:
    bin_mids = [float(m) for m in range(1, 26)]
    pp_df = _make_pp_raw_df(bin_mids, [min(1300.0, max(0.0, 130.0 * (m - 3))) for m in bin_mids], [50.0] * 25)
    pp_df.index = pd.Index([*bin_mids[:-1], bin_mids[0]], name=pp_df.index.name)
    kwargs = {"pre_or_post": "pre", "ws_bin_width": 1.0, "rated_power": 1300.0, "clip_to_rated": True}

    assert_frame_equal(_cook_pp(pp_df, **kwargs), _cook_pp_with_pandas(pp_df, **kwargs), check_exact=True)


def test_cut_matches_pandas() -> None:
    bin_edges = np.arange(0, 25.5, 0.5)
    values = np.concatenate([np.linspace(-1, 30, 1001), bin_edges, bin_edges + 1e-12, bin_edges - 1e-12, [np.inf]])

    actual = _cut(values, bin_edges=bin_edges)

    expected = pd.cut(values, bins=bin_edges)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.codes, expected.codes)
