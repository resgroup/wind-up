"""Check the wind speed ratio by direction against its wind-up 0.7.1 implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from wind_up.circular_math import circ_diff
from wind_up.detrend import _calc_wsratio_v_wd

TEST_WS_COL = "test_ws_est_blend"
REF_WS_COL = "ref_ws_est_blend"
REF_WD_COL = "ref_YawAngleMean"


def _reference_calc_wsratio_v_wd(
    *,
    detrend_df: pd.DataFrame,
    test_ws_col: str,
    ref_ws_col: str,
    ref_wd_col: str,
    min_hours: int,
    timebase_s: int,
    dir_bin_width: float = 10.0,
) -> pd.DataFrame:
    detrend_df = detrend_df.dropna(subset=[test_ws_col, ref_ws_col, ref_wd_col]).copy()

    test_ws_ll = 4
    test_ws_ul = 16
    ref_ws_ll = test_ws_ll * detrend_df[ref_ws_col].mean() / detrend_df[test_ws_col].mean()
    ref_ws_ul = test_ws_ul * detrend_df[ref_ws_col].mean() / detrend_df[test_ws_col].mean()
    detrend_df = detrend_df[(detrend_df[test_ws_col] >= test_ws_ll) & (detrend_df[test_ws_col] < test_ws_ul)]
    detrend_df = detrend_df[(detrend_df[ref_ws_col] >= ref_ws_ll) & (detrend_df[ref_ws_col] < ref_ws_ul)]

    directions = []
    hours = []
    test_rf_ws_roms = []
    for d in list(range(0, 360, 1)):
        detrend_df["circ_diff_to_d"] = circ_diff(detrend_df[ref_wd_col], d)
        detrend_df["within_dir_bin"] = detrend_df["circ_diff_to_d"].abs() < dir_bin_width / 2
        subsector_df = detrend_df[detrend_df["within_dir_bin"]].copy()
        if len(subsector_df) > 0:
            directions.append(d)
            rows_per_hour = 3600 / timebase_s
            hours.append(len(subsector_df) / rows_per_hour)
            min_count = min_hours * rows_per_hour
            accept_sector = len(subsector_df) >= min_count
            iec_ws_threshold = 8
            accept_sector = accept_sector and ((subsector_df[test_ws_col] < iec_ws_threshold).sum() >= (min_count / 4))
            accept_sector = accept_sector and ((subsector_df[test_ws_col] >= iec_ws_threshold).sum() >= (min_count / 4))
            if accept_sector:
                rom = subsector_df[test_ws_col].mean() / subsector_df[ref_ws_col].mean()
                test_rf_ws_roms.append(rom)
            else:
                test_rf_ws_roms.append(np.nan)

    return pd.DataFrame({"direction": directions, "hours": hours, "ws_rom": test_rf_ws_roms})


def _random_detrend_df(*, rows: int, seed: int, dtype: type) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    test_ws = rng.gamma(shape=3, scale=3, size=rows)
    ref_ws = test_ws * rng.normal(1, 0.1, size=rows)
    ref_wd = rng.vonmises(mu=np.pi, kappa=0.5, size=rows) * 180 / np.pi + rng.choice([0, 360], size=rows)
    df = pd.DataFrame({TEST_WS_COL: test_ws, REF_WS_COL: ref_ws, REF_WD_COL: ref_wd}).astype(dtype)
    df.iloc[rng.choice(rows, size=rows // 20), rng.integers(0, 3, size=rows // 20)] = np.nan
    return df


@pytest.mark.parametrize("min_hours", [0, 2, 24])
@pytest.mark.parametrize("data", ["LSA_T13 pre", "random", "random float32", "few rows", "no rows"])
def test_calc_wsratio_v_wd_matches_reference(data: str, min_hours: int) -> None:
    if data == "LSA_T13 pre":
        detrend_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_LSA_T12_pre_df.parquet")
    elif data == "random":
        detrend_df = _random_detrend_df(rows=20000, seed=0, dtype=np.float64)
    elif data == "random float32":
        detrend_df = _random_detrend_df(rows=20000, seed=1, dtype=np.float32)
    elif data == "few rows":
        detrend_df = _random_detrend_df(rows=30, seed=2, dtype=np.float64)
    else:
        detrend_df = _random_detrend_df(rows=30, seed=3, dtype=np.float64).iloc[:0]
    kwargs = {
        "detrend_df": detrend_df,
        "test_ws_col": TEST_WS_COL,
        "ref_ws_col": REF_WS_COL,
        "ref_wd_col": REF_WD_COL,
        "min_hours": min_hours,
        "timebase_s": 600,
    }

    actual = _calc_wsratio_v_wd(**kwargs)

    expected = _reference_calc_wsratio_v_wd(**kwargs)
    assert_frame_equal(actual, expected, check_exact=True)
    if data in {"LSA_T13 pre", "random"} and min_hours == 2:
        assert actual["ws_rom"].notna().sum() > 100
