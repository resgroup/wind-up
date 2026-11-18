"""Check the wind farm yaw direction against its wind-up 0.7.1 implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from wind_up.circular_math import circ_median
from wind_up.constants import RAW_DOWNTIME_S_COL, RAW_POWER_COL, REANALYSIS_WD_COL, TIMESTAMP_COL, WINDFARM_YAWDIR_COL
from wind_up.northing import _calc_wf_yawdir_df, _circ_median_and_count, add_wf_yawdir, calc_northed_col_name
from wind_up.northing_utils import add_ok_yaw_col

if TYPE_CHECKING:
    from wind_up.models import WindUpConfig

YAW_COL = calc_northed_col_name(REANALYSIS_WD_COL)


def _reference_calc_wf_yawdir_df(
    wf_df: pd.DataFrame, *, best_yaw_dir_col: str, min_num_turbines: int, cfg: WindUpConfig
) -> pd.DataFrame:
    wf_df = wf_df.copy()
    wf_df = add_ok_yaw_col(
        wf_df,
        new_col_name="ok_for_yawdir",
        wd_col=best_yaw_dir_col,
        rated_power=cfg.get_max_rated_power(),
        timebase_s=cfg.timebase_s,
    )
    wf_df = wf_df.loc[wf_df["ok_for_yawdir"]].dropna(subset=[best_yaw_dir_col])
    wf_yawdir_df = wf_df.groupby(TIMESTAMP_COL).agg(
        wf_yawdir=pd.NamedAgg(column=best_yaw_dir_col, aggfunc=circ_median),
        num_turbines=pd.NamedAgg(column=best_yaw_dir_col, aggfunc=lambda x: x.count()),
    )
    wf_yawdir_df = wf_yawdir_df[wf_yawdir_df["num_turbines"] >= min_num_turbines]
    return wf_yawdir_df.dropna()


def _make_wf_df(cfg: WindUpConfig, *, timestamps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    turbines = [x.name for x in cfg.asset.wtgs]
    idx = pd.MultiIndex.from_product(
        [turbines, pd.date_range("2021-01-01", periods=timestamps, freq="10min", tz="UTC")],
        names=["TurbineName", TIMESTAMP_COL],
    )
    wind_dir = rng.uniform(0, 360, size=timestamps)
    spread = rng.choice([1, 20, 90, 180], size=timestamps)
    yaw = (wind_dir[None, :] + rng.normal(size=(len(turbines), timestamps)) * spread[None, :]).ravel()
    yaw = yaw - rng.choice([0, 360, -360], size=len(idx))
    yaw[rng.uniform(size=len(idx)) < 0.2] = np.nan
    rated_power = cfg.get_max_rated_power()
    return pd.DataFrame(
        {
            YAW_COL: yaw,
            REANALYSIS_WD_COL: np.tile(wind_dir, len(turbines)),
            RAW_POWER_COL: rng.uniform(-0.1, 1, size=len(idx)) * rated_power,
            RAW_DOWNTIME_S_COL: rng.choice([0, 0, 0, 600, np.nan], size=len(idx)),
        },
        index=idx,
    )


@pytest.mark.parametrize("min_num_turbines", [0, 3])
@pytest.mark.parametrize("case", ["full", "rows missing", "unsorted", "no turbine ok"])
def test_calc_wf_yawdir_df_matches_reference(
    test_lsa_t13_config: WindUpConfig, case: str, min_num_turbines: int
) -> None:
    cfg = test_lsa_t13_config
    wf_df = _make_wf_df(cfg, timestamps=2000, seed=0)
    if case == "rows missing":
        wf_df = wf_df[np.random.default_rng(1).uniform(size=len(wf_df)) > 0.3]
    elif case == "unsorted":
        wf_df = wf_df.iloc[np.random.default_rng(2).permutation(len(wf_df))]
    elif case == "no turbine ok":
        wf_df[RAW_POWER_COL] = 0.0
    kwargs = {"best_yaw_dir_col": YAW_COL, "min_num_turbines": min_num_turbines, "cfg": cfg}

    actual = _calc_wf_yawdir_df(wf_df, **kwargs)

    expected = _reference_calc_wf_yawdir_df(wf_df, **kwargs)
    assert_frame_equal(actual, expected, check_exact=True)
    if case != "no turbine ok":
        assert actual["num_turbines"].nunique() > 5


def test_circ_median_and_count_with_timestamp_index() -> None:
    rng = np.random.default_rng(3)
    timestamps = pd.date_range("2021-01-01", periods=50, freq="10min", tz="UTC", name=TIMESTAMP_COL)
    angles = pd.Series(rng.uniform(-360, 720, size=500), index=timestamps[rng.integers(0, 50, size=500)])

    actual = _circ_median_and_count(angles.groupby(level=TIMESTAMP_COL))

    expected = angles.groupby(level=TIMESTAMP_COL).agg([circ_median, "count"])
    np.testing.assert_array_equal(actual["wf_yawdir"].to_numpy(), expected["circ_median"].to_numpy())
    np.testing.assert_array_equal(actual["num_turbines"].to_numpy(), expected["count"].to_numpy())
    pd.testing.assert_index_equal(actual.index, expected.index)


def test_add_wf_yawdir_fills_with_reanalysis(test_lsa_t13_config: WindUpConfig) -> None:
    cfg = test_lsa_t13_config
    wf_df = _make_wf_df(cfg, timestamps=500, seed=4)

    actual = add_wf_yawdir(wf_df, cfg=cfg)

    expected_wf_yawdir = _reference_calc_wf_yawdir_df(wf_df, best_yaw_dir_col=YAW_COL, min_num_turbines=3, cfg=cfg)
    expected = wf_df.merge(
        expected_wf_yawdir[["wf_yawdir"]].rename(columns={"wf_yawdir": WINDFARM_YAWDIR_COL}),
        how="left",
        left_index=True,
        right_index=True,
    )
    expected[WINDFARM_YAWDIR_COL] = expected[WINDFARM_YAWDIR_COL].fillna(expected[REANALYSIS_WD_COL])
    assert_frame_equal(actual, expected, check_exact=True)
