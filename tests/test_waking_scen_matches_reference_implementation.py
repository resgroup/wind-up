"""Check add_waking_scen against its wind-up 0.7.1 implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from wind_up.constants import TIMESTAMP_COL
from wind_up.waking_state import (
    add_waking_scen,
    add_waking_scen_from_wf_waking_state,
    calc_scen_name_from_wtgs_not_waking,
    calc_wf_waking_state,
    get_iec_upwind_turbines,
    lat_long_is_valid,
)

if TYPE_CHECKING:
    from wind_up.models import WindUpConfig

REF_WD_COL = "ref_YawAngleMean"
NUM_TURBINES = 9


def _reference_add_waking_scen(
    *,
    test_name: str,
    ref_name: str,
    test_ref_df: pd.DataFrame,
    cfg: WindUpConfig,
    wf_df: pd.DataFrame,
    ref_wd_col: str,
    ref_lat: float,
    ref_long: float,
) -> pd.DataFrame:
    test_ref_df = test_ref_df.copy()
    test_ref_df["waking_scenario"] = "not calculated"
    try:
        test_wtg = next(x for x in cfg.asset.wtgs if x.name == test_name)
    except StopIteration as exc:
        msg = f"{test_name} not found in cfg.asset.wtgs"
        raise ValueError(msg) from exc

    all_turbines_waking = wf_df.groupby(TIMESTAMP_COL)["waking"].all().to_frame(name="all_turbines_waking")
    test_ref_df = test_ref_df.merge(all_turbines_waking, how="left", left_index=True, right_index=True)
    test_ref_df.loc[test_ref_df["all_turbines_waking"], "waking_scenario"] = "none offline"

    test_ref_df["rounded_wd"] = test_ref_df[ref_wd_col].round(0).mod(360)
    waking_df = wf_df[["waking", "not_waking", "unknown_waking"]].pivot_table(
        index=TIMESTAMP_COL,
        columns="TurbineName",
        aggfunc="max",
        observed=False,
    )

    for wd in sorted(test_ref_df["rounded_wd"].unique()):
        if wd >= 0:
            test_upwind_wtgs = get_iec_upwind_turbines(
                latlongs=test_wtg.get_latlongs(),
                wind_direction=wd,
                cfg=cfg,
                object_name=test_name,
            )
            if lat_long_is_valid(ref_lat, ref_long):
                ref_upwind_wtgs = get_iec_upwind_turbines(
                    latlongs=[(ref_lat, ref_long)],
                    wind_direction=wd,
                    cfg=cfg,
                    object_name=ref_name,
                )
            else:
                ref_upwind_wtgs = []
            upwind_wtgs = sorted(set(test_upwind_wtgs + ref_upwind_wtgs))
            if len(upwind_wtgs) == 0:
                test_ref_df.loc[test_ref_df["rounded_wd"] == wd, "waking_scenario"] = "none offline"
            else:
                for idx in test_ref_df[
                    (test_ref_df["rounded_wd"] == wd) & (test_ref_df["waking_scenario"] == "not calculated")
                ].index:
                    if waking_df.loc[idx, "unknown_waking"].loc[upwind_wtgs].any():
                        test_ref_df.loc[idx, "waking_scenario"] = "unknown"
                    elif waking_df.loc[idx, "waking"].loc[upwind_wtgs].all():
                        test_ref_df.loc[idx, "waking_scenario"] = "none offline"
                    else:
                        not_waking_series = waking_df.loc[idx, "not_waking"].loc[upwind_wtgs]
                        test_ref_df.loc[idx, "waking_scenario"] = calc_scen_name_from_wtgs_not_waking(
                            list(not_waking_series[not_waking_series].index),
                        )

    if (test_ref_df.dropna(subset=ref_wd_col)["waking_scenario"] == "not calculated").any():
        msg = "some rows with defined ref_wd_col have waking scenario = 'not calculated'"
        raise RuntimeError(msg)

    return test_ref_df.drop(columns=["rounded_wd", "all_turbines_waking"])


@pytest.fixture
def grid_layout_config(test_lsa_t13_config: WindUpConfig) -> WindUpConfig:
    """Config with nine turbines on a three by three grid with about four rotor diameters between neighbours."""
    cfg = test_lsa_t13_config
    cfg.asset.wtgs = cfg.asset.wtgs[:NUM_TURBINES]
    for i, wtg in enumerate(cfg.asset.wtgs):
        wtg.latitude = 55.0 + 0.003 * (i // 3)
        wtg.longitude = -3.0 + 0.005 * (i % 3)
    return cfg


def _make_wf_df(cfg: WindUpConfig, *, timestamps: int, seed: int, waking_frc: float = 0.9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [
            [x.name for x in cfg.asset.wtgs],
            pd.date_range("2021-01-01", periods=timestamps, freq="10min", tz="UTC"),
        ],
        names=["TurbineName", TIMESTAMP_COL],
    )
    waking = rng.uniform(size=len(idx)) < waking_frc
    not_waking = ~waking & (rng.uniform(size=len(idx)) < 0.8)
    return pd.DataFrame(
        {"waking": waking, "not_waking": not_waking, "unknown_waking": ~waking & ~not_waking, "other": 1.0},
        index=idx,
    )


def _make_test_ref_df(wf_df: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = wf_df.index.get_level_values(TIMESTAMP_COL).unique().sort_values()
    timestamps = timestamps[rng.uniform(size=len(timestamps)) < 0.8]
    wd = rng.uniform(-5, 365, size=len(timestamps))
    wd[rng.uniform(size=len(timestamps)) < 0.05] = np.nan
    return pd.DataFrame({REF_WD_COL: wd, "test_other": rng.uniform(size=len(timestamps))}, index=timestamps)


def _run(func: callable, **kwargs: object) -> pd.DataFrame | tuple[type, str]:
    try:
        return func(**kwargs)
    except (KeyError, ValueError, RuntimeError) as exc:
        return type(exc), str(exc)


def _change_wf_df(wf_df: pd.DataFrame, *, case: str) -> pd.DataFrame:
    if case == "categorical turbine names":
        wf_df.index = wf_df.index.set_levels(pd.CategoricalIndex(wf_df.index.levels[0]), level="TurbineName")
    elif case == "rows missing from wf_df":
        wf_df = wf_df[np.random.default_rng(3).uniform(size=len(wf_df)) > 0.02]
    elif case == "turbine missing from wf_df":
        wf_df = wf_df.loc[wf_df.index.levels[0][[0, *range(2, NUM_TURBINES)]]]
    return wf_df


def _change_test_ref_df(test_ref_df: pd.DataFrame, *, case: str) -> pd.DataFrame:
    if case in {"repeated timestamps in test_ref_df", "repeated timestamps and a corner test turbine"}:
        repeated = test_ref_df.iloc[:100].assign(**{REF_WD_COL: lambda d: (d[REF_WD_COL] + 77) % 360})
        test_ref_df = pd.concat([test_ref_df, repeated]).sort_index()
    elif case == "unsorted test_ref_df":
        test_ref_df = test_ref_df.iloc[np.random.default_rng(2).permutation(len(test_ref_df))]
    elif case == "no rows in test_ref_df":
        test_ref_df = test_ref_df.iloc[:0]
    elif case == "timestamps missing from wf_df":
        test_ref_df.index = test_ref_df.index + pd.Timedelta("5min")
    return test_ref_df


@pytest.mark.parametrize(
    "case",
    [
        "full grid",
        "categorical turbine names",
        "self reference",
        "reference without a location",
        "few turbines waking",
        "repeated timestamps in test_ref_df",
        "repeated timestamps and a corner test turbine",
        "unsorted test_ref_df",
        "no rows in test_ref_df",
        "rows missing from wf_df",
        "timestamps missing from wf_df",
        "turbine missing from wf_df",
        "test turbine not in cfg",
    ],
)
def test_add_waking_scen_matches_reference(grid_layout_config: WindUpConfig, case: str) -> None:
    cfg = grid_layout_config
    test_wtg, ref_wtg = cfg.asset.wtgs[4], cfg.asset.wtgs[0]
    wf_df = _make_wf_df(cfg, timestamps=700, seed=0, waking_frc=0.4 if case == "few turbines waking" else 0.9)
    test_ref_df = _change_test_ref_df(_make_test_ref_df(wf_df, seed=1), case=case)
    wf_df = _change_wf_df(wf_df, case=case)
    kwargs = {
        "test_name": test_wtg.name,
        "ref_name": ref_wtg.name,
        "cfg": cfg,
        "ref_wd_col": REF_WD_COL,
        "ref_lat": ref_wtg.latitude,
        "ref_long": ref_wtg.longitude,
    }
    if case == "self reference":
        kwargs |= {"ref_name": test_wtg.name, "ref_lat": test_wtg.latitude, "ref_long": test_wtg.longitude}
    elif case == "reference without a location":
        kwargs |= {"ref_name": "reanalysis", "ref_lat": np.nan, "ref_long": np.nan}
    elif case == "test turbine not in cfg":
        kwargs |= {"test_name": "not a turbine"}
    elif case == "repeated timestamps and a corner test turbine":
        kwargs |= {"test_name": ref_wtg.name, "ref_name": "reanalysis", "ref_lat": np.nan, "ref_long": np.nan}

    expected = _run(_reference_add_waking_scen, test_ref_df=test_ref_df, wf_df=wf_df, **kwargs)
    actual = _run(add_waking_scen, test_ref_df=test_ref_df, wf_df=wf_df, **kwargs)
    actual_from_wf_waking_state = _run(
        add_waking_scen_from_wf_waking_state,
        test_ref_df=test_ref_df,
        wf_waking_state=calc_wf_waking_state(wf_df),
        **kwargs,
    )

    if isinstance(expected, tuple):
        assert actual == expected
        assert actual_from_wf_waking_state == expected
    else:
        assert_frame_equal(actual, expected, check_exact=True)
        assert_frame_equal(actual_from_wf_waking_state, expected, check_exact=True)
        if len(expected) > 0:
            assert expected["waking_scenario"].nunique() > 10


def test_wf_waking_state_can_be_reused(grid_layout_config: WindUpConfig) -> None:
    cfg = grid_layout_config
    wf_df = _make_wf_df(cfg, timestamps=300, seed=4)
    wf_waking_state = calc_wf_waking_state(wf_df)

    assert all(x.shape == (300, NUM_TURBINES) for x in wf_waking_state.state_arrays.values())
    assert set(wf_waking_state.state_arrays) == {"waking", "not_waking", "unknown_waking"}
    for seed, (test_wtg, ref_wtg) in enumerate(
        [(cfg.asset.wtgs[4], cfg.asset.wtgs[0]), (cfg.asset.wtgs[8], cfg.asset.wtgs[5])]
    ):
        test_ref_df = _make_test_ref_df(wf_df, seed=seed)
        kwargs = {
            "test_name": test_wtg.name,
            "ref_name": ref_wtg.name,
            "test_ref_df": test_ref_df,
            "cfg": cfg,
            "ref_wd_col": REF_WD_COL,
            "ref_lat": ref_wtg.latitude,
            "ref_long": ref_wtg.longitude,
        }

        actual = add_waking_scen_from_wf_waking_state(wf_waking_state=wf_waking_state, **kwargs)

        assert_frame_equal(actual, _reference_add_waking_scen(wf_df=wf_df, **kwargs), check_exact=True)
