"""Tests of add_waking_scen on inputs that take the row by row path."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from wind_up.constants import TIMESTAMP_COL
from wind_up.waking_state import (
    add_waking_scen,
    add_waking_scen_from_wf_waking_state,
    calc_wf_waking_state,
    distance_and_bearing_cache,
    upwind_wtgs_cache,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from wind_up.models import WindUpConfig

WAKING_COLS = ["waking", "not_waking", "unknown_waking"]


@pytest.fixture(autouse=True)
def _empty_waking_caches() -> Iterator[None]:
    """Leave the distance, bearing and upwind turbine caches empty for later tests which time them."""
    yield
    distance_and_bearing_cache.clear()
    upwind_wtgs_cache.clear()


def _inputs(cfg: WindUpConfig, *, wd: float = 249.9 - 180) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """HMR_T00 is upwind of HMR_T01 and HMR_T02 at the default wind direction and offline, then unknown."""
    wtgs = ["HMR_T00", "HMR_T01", "HMR_T02"]
    tstamps = pd.date_range(start="2021-01-01", tz="UTC", periods=3, freq="10min")
    idx = pd.MultiIndex.from_product([wtgs, tstamps], names=["TurbineName", TIMESTAMP_COL])
    wf_df = pd.DataFrame(
        {
            "waking": [True, False, False] + [True] * 6,
            "not_waking": [False, True, False] + [False] * 6,
            "unknown_waking": [False, False, True] + [False] * 6,
        },
        index=idx,
    )
    test_ref_df = pd.DataFrame({"ref_YawAngleMean": wd}, index=tstamps)
    ref_wtg = next(x for x in cfg.asset.wtgs if x.name == "HMR_T02")
    kwargs = {
        "test_name": "HMR_T01",
        "ref_name": "HMR_T02",
        "cfg": cfg,
        "ref_wd_col": "ref_YawAngleMean",
        "ref_lat": ref_wtg.latitude,
        "ref_long": ref_wtg.longitude,
    }
    return wf_df, test_ref_df, kwargs


def test_nullable_boolean_waking_columns_with_missing_values(test_homer_with_t00_config: WindUpConfig) -> None:
    wf_df, test_ref_df, kwargs = _inputs(test_homer_with_t00_config)
    wf_df = wf_df.astype(dict.fromkeys(WAKING_COLS, "boolean"))
    wf_df.loc[("HMR_T02", test_ref_df.index[0]), "unknown_waking"] = pd.NA

    assert calc_wf_waking_state(wf_df).state_arrays == {}
    actual = add_waking_scen(test_ref_df=test_ref_df, wf_df=wf_df, **kwargs)
    assert actual["waking_scenario"].tolist() == ["none offline", "HMR_T00 offline", "unknown"]


def test_repeated_timestamps(test_homer_with_t00_config: WindUpConfig) -> None:
    wf_df, test_ref_df, kwargs = _inputs(test_homer_with_t00_config)
    wf_df.loc[("HMR_T01", test_ref_df.index[0]), ["waking", "not_waking"]] = [False, True]  # not upwind
    test_ref_df = pd.concat([test_ref_df, test_ref_df.iloc[[1]]]).sort_index()

    actual = add_waking_scen(test_ref_df=test_ref_df, wf_df=wf_df, **kwargs)

    assert actual["waking_scenario"].tolist() == ["none offline", "HMR_T00 offline", "HMR_T00 offline", "unknown"]


def test_no_upwind_turbines(test_homer_with_t00_config: WindUpConfig) -> None:
    wf_df, test_ref_df, kwargs = _inputs(test_homer_with_t00_config, wd=180.0)

    actual = add_waking_scen(test_ref_df=test_ref_df, wf_df=wf_df, **kwargs)

    assert actual["waking_scenario"].tolist() == ["none offline"] * 3


def test_upwind_turbine_missing_from_wf_df(test_homer_with_t00_config: WindUpConfig) -> None:
    wf_df, test_ref_df, kwargs = _inputs(test_homer_with_t00_config)
    wf_df.loc[("HMR_T02", test_ref_df.index[1]), ["waking", "not_waking"]] = [False, True]
    wf_df = wf_df.drop(index="HMR_T00")

    with pytest.raises(KeyError):
        add_waking_scen_from_wf_waking_state(
            test_ref_df=test_ref_df, wf_waking_state=calc_wf_waking_state(wf_df), **kwargs
        )


def test_wf_waking_state_tabulates_numpy_booleans(test_homer_with_t00_config: WindUpConfig) -> None:
    wf_df, _, _ = _inputs(test_homer_with_t00_config)

    state = calc_wf_waking_state(wf_df)

    assert state.turbine_names.tolist() == ["HMR_T00", "HMR_T01", "HMR_T02"]
    np.testing.assert_array_equal(state.state_arrays["not_waking"][:, 0], [False, True, False])
