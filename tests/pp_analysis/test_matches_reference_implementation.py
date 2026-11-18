"""Check _pp_raw_df and _cook_pp against their wind-up 0.7.1 implementations."""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tests.conftest import TEST_DATA_FLD
from tests.pp_analysis.test_cook_pp import _make_pp_raw_df, _make_site_mean_pc_df
from wind_up.pp_analysis import _calc_rated_ws, _cook_pp, _pp_raw_df
from wind_up.result_manager import result_manager

WS_COL = "ref_ws_detrended"
PW_COL = "test_pw_clipped"


def _reference_pp_raw_df(
    pre_or_post_df: pd.DataFrame,
    pre_or_post: str,
    *,
    ws_col: str,
    ws_bin_edges: np.ndarray,
    pw_col: str,
    timebase_s: int,
) -> pd.DataFrame:
    return (
        pre_or_post_df.loc[:, [pw_col, ws_col]]
        .dropna()
        .groupby(
            by=pd.cut(pre_or_post_df[ws_col], bins=ws_bin_edges, retbins=False),
            observed=False,
        )
        .agg(
            count=pd.NamedAgg(column=pw_col, aggfunc=len),
            ws_mean=pd.NamedAgg(column=ws_col, aggfunc="mean"),
            ws_std=pd.NamedAgg(column=ws_col, aggfunc="std"),
            pw_mean=pd.NamedAgg(column=pw_col, aggfunc="mean"),
            pw_std=pd.NamedAgg(column=pw_col, aggfunc="std"),
        )
        .assign(
            ws_std=lambda x: x["ws_std"].fillna(0),
            pw_std=lambda x: x["pw_std"].fillna(0),
            hours=lambda x: x["count"] * timebase_s / 3600,
            ws_sem=lambda x: x["ws_std"] / np.sqrt(x["count"].clip(lower=1)),
            pw_sem=lambda x: x["pw_std"] / np.sqrt(x["count"].clip(lower=1)),
        )
        .pipe(lambda d: d.set_axis(d.columns.map(lambda x: f"{x}_{pre_or_post}"), axis="columns"))
        .assign(
            bin_left=lambda x: [i.left for i in x.index],
            bin_mid=lambda x: [i.mid for i in x.index],
            bin_right=lambda x: [i.right for i in x.index],
            bin_closed_right=lambda x: [i.closed_right for i in x.index],
        )
        .set_index("bin_mid", drop=False)
        .rename_axis(f"{ws_col}_bin_mid", axis=0)
    )


def _reference_cook_pp(
    pp_df: pd.DataFrame,
    *,
    pre_or_post: str,
    ws_bin_width: float,
    rated_power: float,
    clip_to_rated: bool,
    site_mean_pc_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    pp_df = pp_df.copy()

    valid_col = f"{pre_or_post}_valid"
    raw_pw_col = f"pw_mean_{pre_or_post}_raw"
    raw_hours_col = f"hours_{pre_or_post}_raw"
    pw_col = f"pw_mean_{pre_or_post}"
    pw_sem_col = f"pw_sem_{pre_or_post}"
    hours_col = f"hours_{pre_or_post}"
    ws_col = f"ws_mean_{pre_or_post}"
    pw_at_mid_col = f"pw_at_mid_{pre_or_post}"
    pw_sem_at_mid_col = f"pw_sem_at_mid_{pre_or_post}"

    pp_df[raw_pw_col] = pp_df[pw_col]
    pp_df[raw_hours_col] = pp_df[hours_col]

    hrs_per_mps = 3
    pp_df[valid_col] = pp_df[f"hours_{pre_or_post}"] > (ws_bin_width * hrs_per_mps)
    pp_df.loc[~pp_df[valid_col], [pw_col, hours_col, ws_col, pw_sem_col]] = np.nan
    pp_df[hours_col] = pp_df[hours_col].fillna(0)
    pp_df[ws_col] = pp_df[ws_col].fillna(pp_df["bin_mid"])

    rated_ws = _calc_rated_ws(pp_df=pp_df, pw_col=raw_pw_col, rated_power=rated_power)
    empty_rated_bins_fill_value = rated_power
    if not clip_to_rated:
        with contextlib.suppress(IndexError):
            empty_rated_bins_fill_value = pp_df.loc[
                (pp_df["bin_mid"] >= rated_ws) & ~pp_df[pw_col].isna(), pw_col
            ].iloc[-1]
    pp_df.loc[(pp_df["bin_mid"] >= rated_ws) & pp_df[pw_col].isna(), pw_col] = empty_rated_bins_fill_value

    if site_mean_pc_df is not None:
        pp_df.loc[~pp_df[valid_col], pw_col] = np.interp(
            pp_df.loc[~pp_df[valid_col], "bin_mid"], site_mean_pc_df["bin_mid"], site_mean_pc_df["pw_clipped"]
        )

    pp_df[pw_col] = pp_df[pw_col].clip(lower=0)
    if clip_to_rated:
        pp_df[pw_col] = pp_df[pw_col].clip(upper=rated_power)

    pp_df[pw_sem_col] = pp_df[pw_sem_col].ffill().bfill()

    pp_df.loc[pp_df.index[pp_df[pw_col].isna().cummin()], [pw_col, pw_sem_col]] = 0

    pp_df[pw_col] = pp_df[pw_col].interpolate()

    pp_df[pw_at_mid_col] = np.interp(pp_df["bin_mid"], pp_df[ws_col], pp_df[pw_col])
    pp_df[pw_at_mid_col] = pp_df[pw_at_mid_col].clip(lower=0)
    if clip_to_rated:
        pp_df[pw_at_mid_col] = pp_df[pw_at_mid_col].clip(upper=rated_power)
    pp_df[pw_sem_at_mid_col] = pp_df[pw_sem_col] / pp_df[pw_col] * pp_df[pw_at_mid_col]
    pp_df[pw_sem_at_mid_col] = pp_df[pw_sem_at_mid_col].fillna(0)
    pp_df[pw_sem_at_mid_col] = pp_df[pw_sem_at_mid_col].clip(lower=0, upper=pp_df[pw_sem_col])

    if pp_df[[col for col in pp_df.columns if col != raw_pw_col]].isna().any().any():
        msg = "pp_df has missing values"
        result_manager.warning(msg)

    return pp_df


def _assert_identical(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    assert_frame_equal(actual, expected, check_exact=True)
    assert type(actual.index) is type(expected.index)


def _time_series_frames() -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(TEST_DATA_FLD / "LSA_T13_LSA_T12_pre_df.parquet", columns=[WS_COL, PW_COL])
    rng = np.random.default_rng(0)
    frames = {"all": df}
    for size in (5000, 500, 40, 3, 0):
        frames[f"sample of {size}"] = df.iloc[np.sort(rng.choice(len(df), size=size, replace=True))].reset_index()
    frames["high wind"] = df[df[WS_COL] > 9]
    frames["low wind"] = df[df[WS_COL] < 6]
    frames["constant power"] = df.assign(**{PW_COL: 1234.5})
    frames["one wind speed"] = df.assign(**{WS_COL: 7.0})
    frames["one row"] = df.dropna().iloc[:1]
    frames["no wind speed"] = df.assign(**{WS_COL: np.nan})
    frames["beyond last bin"] = df.assign(**{WS_COL: df[WS_COL] * 3})
    return frames


TIME_SERIES_FRAMES = _time_series_frames()
BIN_EDGES_AND_TIMEBASE = [
    (np.arange(0, 25 + 1.0, 1.0), 600),
    (np.arange(0, 25 + 0.5, 0.5), 600),
    (np.arange(0, 20 + 1, 1), 60),
]
SITE_MEAN_PC_DF = _make_site_mean_pc_df(rated_power=2000.0)


@pytest.mark.parametrize("frame_name", TIME_SERIES_FRAMES)
@pytest.mark.parametrize(("ws_bin_edges", "timebase_s"), BIN_EDGES_AND_TIMEBASE)
@pytest.mark.parametrize("pre_or_post", ["pre", "post"])
def test_pp_raw_df_and_cook_pp_match_reference(
    frame_name: str, ws_bin_edges: np.ndarray, timebase_s: int, pre_or_post: str
) -> None:
    df = TIME_SERIES_FRAMES[frame_name]
    raw_kwargs = {"ws_col": WS_COL, "ws_bin_edges": ws_bin_edges, "pw_col": PW_COL, "timebase_s": timebase_s}

    expected_raw = _reference_pp_raw_df(df, pre_or_post, **raw_kwargs)
    actual_raw = _pp_raw_df(df, pre_or_post, **raw_kwargs)

    _assert_identical(actual_raw, expected_raw)
    for rated_power, clip_to_rated, site_mean_pc_df in [
        (2000.0, True, None),
        (1500.0, False, None),
        (2000.0, False, SITE_MEAN_PC_DF),
        (5000.0, True, SITE_MEAN_PC_DF),
    ]:
        cook_kwargs = {
            "pre_or_post": pre_or_post,
            "ws_bin_width": float(ws_bin_edges[1] - ws_bin_edges[0]),
            "rated_power": rated_power,
            "clip_to_rated": clip_to_rated,
            "site_mean_pc_df": site_mean_pc_df,
        }
        result_manager.stored_warnings = []
        expected = _reference_cook_pp(expected_raw, **cook_kwargs)
        expected_warnings = result_manager.stored_warnings
        result_manager.stored_warnings = []
        actual = _cook_pp(actual_raw, **cook_kwargs)

        _assert_identical(actual, expected)
        assert result_manager.stored_warnings == expected_warnings
        _assert_identical(actual_raw, expected_raw)  # the input is left as it was
    result_manager.stored_warnings = []


BIN_MIDS = list(range(1, 26))
HAND_BUILT_HOURS = {
    "all valid": [50.0] * 25,
    "high wind invalid": [50.0 if m <= 8 else 1.0 for m in BIN_MIDS],
    "low wind invalid": [1.0 if m <= 4 else 50.0 for m in BIN_MIDS],
    "every third invalid": [50.0 if m % 3 else 1.0 for m in BIN_MIDS],
    "all invalid": [1.0] * 25,
}
HAND_BUILT_POWER = {
    "reaches rated": [min(1300.0, max(0.0, 1300.0 * (m - 3) / 9)) for m in BIN_MIDS],
    "gap below rated": [None if 10 <= m <= 14 else min(700.0, max(0.0, 700.0 * (m - 3) / 5)) for m in BIN_MIDS],
    "negative and above rated": [None if m < 5 else -5.0 if m < 7 else 1400.0 for m in BIN_MIDS],
    "all missing": [np.nan] * 25,
}


@pytest.mark.parametrize("hours_name", HAND_BUILT_HOURS)
@pytest.mark.parametrize("power_name", HAND_BUILT_POWER)
@pytest.mark.parametrize("pre_or_post", ["pre", "post"])
@pytest.mark.parametrize("integer_bin_mids", [True, False])
def test_cook_pp_matches_reference_for_hand_built_bins(
    hours_name: str,
    power_name: str,
    pre_or_post: str,
    integer_bin_mids: bool,  # noqa: FBT001
) -> None:
    bin_mids = BIN_MIDS if integer_bin_mids else [m - 0.5 for m in BIN_MIDS]
    pp_raw_df = _make_pp_raw_df(bin_mids, HAND_BUILT_POWER[power_name], HAND_BUILT_HOURS[hours_name], pre_or_post)

    for rated_power, clip_to_rated, site_mean_pc_df in [
        (1300.0, True, None),
        (1000.0, False, None),
        (1300.0, False, _make_site_mean_pc_df()),
        (3000.0, True, _make_site_mean_pc_df()),
    ]:
        cook_kwargs = {
            "pre_or_post": pre_or_post,
            "ws_bin_width": 1.0,
            "rated_power": rated_power,
            "clip_to_rated": clip_to_rated,
            "site_mean_pc_df": site_mean_pc_df,
        }
        result_manager.stored_warnings = []
        expected = _reference_cook_pp(pp_raw_df, **cook_kwargs)
        expected_warnings = result_manager.stored_warnings
        result_manager.stored_warnings = []
        actual = _cook_pp(pp_raw_df, **cook_kwargs)

        _assert_identical(actual, expected)
        assert result_manager.stored_warnings == expected_warnings
    result_manager.stored_warnings = []
