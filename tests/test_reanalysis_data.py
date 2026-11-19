from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tests.conftest import TEST_DATA_FLD
from wind_up.constants import TIMESTAMP_COL
from wind_up.models import WindUpConfig
from wind_up.reanalysis_data import (
    ReanalysisDataset,
    _calc_wf_mean_wind_speed_and_data_coverage,
    _corr_with_shifted_reanalysis,
    _wind_speed_by_timestamp_and_turbine,
    add_reanalysis_data,
    get_dsid_and_dates_from_filename,
)


def test_get_dsid_and_dates_from_filename() -> None:
    assert get_dsid_and_dates_from_filename("ERA5T_47.50N_-3.25E_100m_1hr_19900101_20231031.parquet") == (
        "ERA5T_47.50N_-3.25E_100m_1hr",
        pd.Timestamp("1990-01-01", tz="UTC"),
        pd.Timestamp("2023-10-31", tz="UTC"),
    )


def test_add_reanalysis_data(test_homer_config: WindUpConfig) -> None:
    cfg = test_homer_config
    cfg.lt_first_dt_utc_start = pd.Timestamp("2023-07-01 00:00:00", tz="UTC")
    cfg.analysis_last_dt_utc_start = pd.Timestamp("2023-07-31 23:50:00", tz="UTC")
    wf_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/Homer Wind Farm_July2023_scada_improved.parquet")
    reanalysis_datasets = [
        ReanalysisDataset(id=fp.stem, data=pd.read_parquet(fp))
        for fp in (TEST_DATA_FLD / "reanalysis" / "Homer Wind Farm").glob("*.parquet")
    ]
    wf_df_after = add_reanalysis_data(wf_df, cfg=cfg, plot_cfg=None, reanalysis_datasets=reanalysis_datasets)
    assert len(wf_df_after.dropna(subset=["reanalysis_ws", "reanalysis_wd"])) == 8928
    assert wf_df_after["reanalysis_ws"].mean() == pytest.approx(7.095553315412186)
    assert wf_df_after["reanalysis_ws"].min() == pytest.approx(0.19)
    assert wf_df_after["reanalysis_ws"].max() == pytest.approx(16.35)
    assert wf_df_after["reanalysis_wd"].mean() == pytest.approx(217.18956093189962)
    assert wf_df_after["reanalysis_wd"].min() == pytest.approx(4.4)
    assert wf_df_after["reanalysis_wd"].max() == pytest.approx(359.6)


def _two_turbine_wf_df() -> pd.DataFrame:
    timestamps = pd.date_range("2021-01-01", periods=3, freq="10min", tz="UTC", name=TIMESTAMP_COL)
    index = pd.MultiIndex.from_product([["T1", "T2"], timestamps], names=["TurbineName", TIMESTAMP_COL])
    return pd.DataFrame({"WindSpeedMean": [1.0, 2.0, np.nan, 3.0, np.nan, np.nan]}, index=index)


@pytest.mark.parametrize(
    ("layout", "is_tabulated"),
    [
        ("sorted by turbine", True),
        ("sorted by timestamp", False),
        ("row missing", False),
        ("timestamp column", False),
    ],
)
def test_calc_wf_mean_wind_speed_and_data_coverage(layout: str, *, is_tabulated: bool) -> None:
    wf_df = _two_turbine_wf_df()
    if layout == "sorted by timestamp":
        wf_df = wf_df.sort_index(level=TIMESTAMP_COL)
    elif layout == "row missing":
        wf_df = wf_df.iloc[:-1]
    elif layout == "timestamp column":
        wf_df = wf_df.reset_index()

    actual = _calc_wf_mean_wind_speed_and_data_coverage(wf_df, num_turbines=2)

    expected = pd.DataFrame(
        {"WindSpeedMean": [2.0, 2.0, np.nan], "data_coverage": [1.0, 0.5, 0.0]},
        index=pd.DatetimeIndex(
            ["2021-01-01 00:00", "2021-01-01 00:10", "2021-01-01 00:20"], tz="UTC", name=TIMESTAMP_COL
        ),
    )
    assert_frame_equal(actual, expected, check_exact=True)
    assert (_wind_speed_by_timestamp_and_turbine(wf_df) is not None) == is_tabulated


def test_calc_wf_mean_wind_speed_and_data_coverage_drops_missing_timestamps() -> None:
    timestamps = pd.DatetimeIndex(["2021-01-01 00:00", "2021-01-01 00:10"], tz="UTC")
    index = pd.MultiIndex(
        levels=[["T1", "T2"], timestamps], codes=[[0, 0, 1, 1], [0, -1, 0, -1]], names=["TurbineName", TIMESTAMP_COL]
    )
    wf_df = pd.DataFrame({"WindSpeedMean": [5.0, 6.0, 7.0, 8.0]}, index=index)

    actual = _calc_wf_mean_wind_speed_and_data_coverage(wf_df, num_turbines=2)

    expected = pd.DataFrame(
        {"WindSpeedMean": [6.0], "data_coverage": [1.0]},
        index=pd.DatetimeIndex(["2021-01-01 00:00"], tz="UTC", name=TIMESTAMP_COL),
    )
    assert_frame_equal(actual, expected, check_exact=True)


def test_corr_with_shifted_reanalysis_with_unsorted_wind_farm_data() -> None:
    rng = np.random.default_rng(0)
    reanalysis_ws = pd.Series(
        rng.gamma(shape=2.5, scale=3.2, size=500),
        index=pd.date_range("2021-01-01", periods=500, freq="10min", tz="UTC"),
    )
    ws_df = reanalysis_ws.shift(5).iloc[100:400].to_frame("WindSpeedMean")
    shifts = list(range(-10, 10))

    corrs = _corr_with_shifted_reanalysis(ws_df, reanalysis_ws, shifts=shifts)
    unsorted_corrs = _corr_with_shifted_reanalysis(ws_df.iloc[::-1], reanalysis_ws, shifts=shifts)

    assert shifts[int(np.argmax(corrs))] == 5
    assert max(corrs) == pytest.approx(1)
    np.testing.assert_allclose(unsorted_corrs, corrs, rtol=1e-12)
