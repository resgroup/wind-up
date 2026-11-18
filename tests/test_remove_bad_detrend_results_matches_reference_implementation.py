"""Check _remove_bad_detrend_results against its wind-up 0.7.1 implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from wind_up.detrend import _remove_bad_detrend_results
from wind_up.result_manager import result_manager
from wind_up.smart_data import add_smart_lat_long_to_cfg
from wind_up.waking_state import get_iec_upwind_turbines, lat_long_is_valid, list_wtgs_offline_in_scen

if TYPE_CHECKING:
    from wind_up.models import WindUpConfig

TEST_DATA_FLD = Path(__file__).parents[0] / "test_data"


def _reference_remove_bad_detrend_results(
    wsratio_v_dir_scen: pd.DataFrame,
    test_name: str,
    ref_name: str,
    ref_lat: float,
    ref_long: float,
    cfg: WindUpConfig,
) -> pd.DataFrame:
    try:
        none_offline_df = wsratio_v_dir_scen.dropna(subset="ws_rom").loc["none offline"]
    except KeyError:
        result_manager.warning("cannot remove_bad_detrend_results, no 'none offline' rows with ws_rom defined")
        return wsratio_v_dir_scen

    try:
        test_wtg = next(x for x in cfg.asset.wtgs if x.name == test_name)
    except StopIteration as exc:
        msg = f"{test_name} not found in cfg.asset.wtgs"
        raise ValueError(msg) from exc

    for wd in none_offline_df.index.unique():
        test_upwind_wtgs = get_iec_upwind_turbines(
            latlongs=test_wtg.get_latlongs(), wind_direction=wd, cfg=cfg, object_name=test_name
        )
        if lat_long_is_valid(ref_lat, ref_long):
            ref_upwind_wtgs = get_iec_upwind_turbines(
                latlongs=[(ref_lat, ref_long)], wind_direction=wd, cfg=cfg, object_name=ref_name
            )
        else:
            ref_upwind_wtgs = []

        scens_to_check = [
            x
            for x in wsratio_v_dir_scen.dropna(subset="ws_rom")
            .loc[pd.IndexSlice[:, wd], :]
            .index.unique(level="waking_scenario")
            if x != "none offline"
        ]
        for scen in scens_to_check:
            upwind_wtgs_offline = list_wtgs_offline_in_scen(scen)
            test_upwind_wtgs_offline = sorted(set(test_upwind_wtgs) & set(upwind_wtgs_offline))
            ref_upwind_wtgs_offline = sorted(set(ref_upwind_wtgs) & set(upwind_wtgs_offline))
            if (len(test_upwind_wtgs_offline) + len(ref_upwind_wtgs_offline) > 0) and (
                len(test_upwind_wtgs_offline) == 0 or len(ref_upwind_wtgs_offline) == 0
            ):
                sign = 1 if len(ref_upwind_wtgs_offline) == 0 else -1
                scen_ws_rom = wsratio_v_dir_scen.loc[pd.IndexSlice[scen, wd], "ws_rom"]
                none_offline_ws_rom = none_offline_df.loc[wd, "ws_rom"]
                abs_tol = 1e-6
                if sign * scen_ws_rom < (sign * none_offline_ws_rom - abs_tol):
                    wsratio_v_dir_scen.loc[pd.IndexSlice[scen, wd], "ws_rom"] = np.nan

    return wsratio_v_dir_scen


@pytest.fixture
def lsa_config_with_latlongs(test_lsa_t13_config: WindUpConfig) -> WindUpConfig:
    md_df = pd.read_csv(TEST_DATA_FLD / "smart_data/Lisa Wind Farm/Lisa Wind Farm_md.csv")
    return add_smart_lat_long_to_cfg(md=md_df, cfg=test_lsa_t13_config)


def _random_wsratio_v_dir_scen(cfg: WindUpConfig, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wtgs = [x.name for x in cfg.asset.wtgs]
    scens = ["none offline", *(f"{x} offline" for x in wtgs), f"{wtgs[0]} {wtgs[3]} offline"]
    frames = []
    for scen in rng.permutation(scens):
        directions = np.sort(rng.choice(360, size=rng.integers(50, 360), replace=False))
        ws_rom = rng.normal(1, 0.03, size=len(directions))
        ws_rom[rng.uniform(size=len(directions)) < 0.2] = np.nan
        frames.append(pd.DataFrame({"hours": 30.0, "ws_rom": ws_rom, "waking_scenario": scen, "direction": directions}))
    return pd.concat(frames).set_index(["waking_scenario", "direction"])


@pytest.mark.parametrize(
    "case", ["LSA_T13 check", "random", "random reference without location", "no none offline", "unknown test"]
)
def test_remove_bad_detrend_results_matches_reference(lsa_config_with_latlongs: WindUpConfig, case: str) -> None:
    cfg = lsa_config_with_latlongs
    ref_wtg = next(x for x in cfg.asset.wtgs if x.name == "LSA_T12")
    kwargs = {"test_name": "LSA_T13", "ref_name": "LSA_T12", "ref_lat": ref_wtg.latitude, "ref_long": ref_wtg.longitude}
    if case == "LSA_T13 check":
        wsratio_v_dir_scen = pd.read_parquet(TEST_DATA_FLD / "LSA_T13_LSA_T12_check_pre_wsratio_v_dir_scen.parquet")
    else:
        wsratio_v_dir_scen = _random_wsratio_v_dir_scen(cfg, seed=0)
    if case == "random reference without location":
        kwargs |= {"ref_name": "reanalysis", "ref_lat": np.nan, "ref_long": np.nan}
    elif case == "no none offline":
        wsratio_v_dir_scen = wsratio_v_dir_scen.drop(index="none offline", level="waking_scenario")
    elif case == "unknown test":
        kwargs |= {"test_name": "not a turbine"}

    if case == "unknown test":
        with pytest.raises(ValueError, match=r"not a turbine not found in cfg\.asset\.wtgs"):
            _reference_remove_bad_detrend_results(wsratio_v_dir_scen.copy(), cfg=cfg, **kwargs)
        with pytest.raises(ValueError, match=r"not a turbine not found in cfg\.asset\.wtgs"):
            _remove_bad_detrend_results(wsratio_v_dir_scen.copy(), cfg=cfg, **kwargs)
        return

    result_manager.stored_warnings = []
    expected = _reference_remove_bad_detrend_results(wsratio_v_dir_scen.copy(), cfg=cfg, **kwargs)
    expected_warnings = result_manager.stored_warnings
    result_manager.stored_warnings = []
    actual = _remove_bad_detrend_results(wsratio_v_dir_scen.copy(), cfg=cfg, **kwargs)

    assert_frame_equal(actual, expected, check_exact=True)
    assert result_manager.stored_warnings == expected_warnings
    result_manager.stored_warnings = []
    if case == "random":
        assert expected["ws_rom"].isna().sum() > wsratio_v_dir_scen["ws_rom"].isna().sum()
