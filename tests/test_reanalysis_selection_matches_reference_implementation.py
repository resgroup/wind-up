"""Check the reanalysis shift search and wind farm mean wind speed against their wind-up 0.7.1 implementations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tests.conftest import TEST_DATA_FLD
from wind_up import reanalysis_data
from wind_up.constants import REANALYSIS_WS_COL, TIMESTAMP_COL
from wind_up.reanalysis_data import (
    ReanalysisDataset,
    _calc_wf_mean_wind_speed_and_data_coverage,
    _corr_with_shifted_reanalysis,
    _find_best_shift_and_corr,
    _reanalysis_upsample,
    _wind_speed_by_timestamp_and_turbine,
    add_reanalysis_data,
)

if TYPE_CHECKING:
    from wind_up.models import WindUpConfig

NUM_TURBINES = 7
TIMEBASE_S = 600


def _reference_corr_with_shifted_reanalysis(
    ws_df: pd.DataFrame, reanalysis_ws: pd.Series, *, shifts: list[int]
) -> list[float]:
    return [float(ws_df.corrwith(reanalysis_ws.shift(s)).squeeze()) for s in shifts]


def _reference_calc_wf_mean_wind_speed_and_data_coverage(wf_df: pd.DataFrame, *, num_turbines: int) -> pd.DataFrame:
    return wf_df.groupby(TIMESTAMP_COL).agg(
        WindSpeedMean=pd.NamedAgg(column="WindSpeedMean", aggfunc=lambda x: x.mean()),
        data_coverage=pd.NamedAgg(column="WindSpeedMean", aggfunc=lambda x: x.count() / num_turbines),
    )


def _make_wf_df(*, timestamps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [
            [f"T{i:02d}" for i in range(NUM_TURBINES)],
            pd.date_range("2021-01-01", periods=timestamps, freq=f"{TIMEBASE_S}s", tz="UTC", name=TIMESTAMP_COL),
        ],
        names=["TurbineName", TIMESTAMP_COL],
    )
    ws = rng.gamma(shape=2.5, scale=3.2, size=len(idx))
    ws[rng.uniform(size=len(idx)) < 0.1] = np.nan
    ws[: timestamps // 50] = np.nan  # a timestamp where no turbine has a wind speed
    for i in range(1, NUM_TURBINES):
        ws[i * timestamps : i * timestamps + timestamps // 50] = np.nan
    return pd.DataFrame({"WindSpeedMean": ws, "other": 1.0}, index=idx)


def _change_wf_df(wf_df: pd.DataFrame, *, case: str) -> pd.DataFrame:
    if case == "rows missing from wf_df":
        wf_df = wf_df[np.random.default_rng(3).uniform(size=len(wf_df)) > 0.02]
    elif case == "turbine missing from wf_df":
        wf_df = wf_df.drop(index="T02", level="TurbineName")
    elif case == "rows sorted by timestamp":
        wf_df = wf_df.sort_index(level=TIMESTAMP_COL, sort_remaining=True)
    elif case == "turbines out of order":
        wf_df = pd.concat([wf_df.loc[["T03"]], wf_df.drop(index="T03", level="TurbineName")])
    elif case == "turbine repeated":
        wf_df = pd.concat([wf_df, wf_df.loc[["T03"]]])
    elif case == "timestamps out of order":
        timestamps = wf_df.index.get_level_values(TIMESTAMP_COL).unique()
        wf_df = wf_df.reindex(
            pd.MultiIndex.from_product([wf_df.index.levels[0], timestamps[::-1]], names=["TurbineName", TIMESTAMP_COL])
        )
    elif case == "float32 wind speed":
        wf_df = wf_df.astype({"WindSpeedMean": np.float32})
    elif case == "timestamp index":
        wf_df = wf_df.loc["T01"]
    elif case == "timestamp level first":
        wf_df = wf_df.swaplevel().sort_index()
    elif case == "no rows":
        wf_df = wf_df.iloc[:0]
    return wf_df


@pytest.mark.parametrize(
    "case",
    [
        "full grid",
        "rows missing from wf_df",
        "turbine missing from wf_df",
        "rows sorted by timestamp",
        "turbines out of order",
        "turbine repeated",
        "timestamps out of order",
        "float32 wind speed",
        "timestamp index",
        "timestamp level first",
        "no rows",
    ],
)
def test_calc_wf_mean_wind_speed_and_data_coverage_matches_reference(case: str) -> None:
    wf_df = _change_wf_df(_make_wf_df(timestamps=1000, seed=0), case=case)

    actual = _calc_wf_mean_wind_speed_and_data_coverage(wf_df, num_turbines=NUM_TURBINES)

    expected = _reference_calc_wf_mean_wind_speed_and_data_coverage(wf_df, num_turbines=NUM_TURBINES)
    assert_frame_equal(actual, expected, check_exact=True)
    is_tabulated = _wind_speed_by_timestamp_and_turbine(wf_df) is not None
    assert is_tabulated == (case in {"full grid", "turbine missing from wf_df", "turbines out of order"})


def test_calc_wf_mean_wind_speed_and_data_coverage_with_bottleneck(monkeypatch: pytest.MonkeyPatch) -> None:
    wf_df = _make_wf_df(timestamps=300, seed=1)
    monkeypatch.setattr(reanalysis_data.importlib.util, "find_spec", lambda _: object())
    with pd.option_context("compute.use_bottleneck", True):  # noqa: FBT003
        actual = _calc_wf_mean_wind_speed_and_data_coverage(wf_df, num_turbines=NUM_TURBINES)

    expected = _reference_calc_wf_mean_wind_speed_and_data_coverage(wf_df, num_turbines=NUM_TURBINES)
    assert_frame_equal(actual, expected, check_exact=True)


def _make_ws_df_and_reanalysis_ws(*, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    hourly_index = pd.date_range("2020-12-30", "2021-01-12", freq="1h", tz="UTC")
    reanalysis_raw_df = pd.DataFrame(
        {
            "100_m_hws_mean_mps": rng.gamma(shape=2.5, scale=3.2, size=len(hourly_index)),
            "100_m_hwd_mean_deg-n_true": rng.uniform(0, 360, size=len(hourly_index)),
        },
        index=hourly_index,
    )
    reanalysis_ws = _reanalysis_upsample(reanalysis_raw_df, timebase_s=TIMEBASE_S)[REANALYSIS_WS_COL]
    wf_index = pd.date_range("2021-01-01", "2021-01-08", freq=f"{TIMEBASE_S}s", tz="UTC", name=TIMESTAMP_COL)
    wf_ws = reanalysis_ws.shift(5).reindex(wf_index).to_numpy() + rng.normal(scale=1.5, size=len(wf_index))
    ws_df = pd.DataFrame({"WindSpeedMean": wf_ws}, index=wf_index)
    ws_df = ws_df[rng.uniform(size=len(ws_df)) > 0.3]
    return ws_df[ws_df["WindSpeedMean"] >= 0], reanalysis_ws


def _change_ws_df_and_reanalysis_ws(
    ws_df: pd.DataFrame, reanalysis_ws: pd.Series, *, case: str
) -> tuple[pd.DataFrame, pd.Series]:
    if case == "gaps in reanalysis":
        reanalysis_ws = reanalysis_ws.where(np.random.default_rng(4).uniform(size=len(reanalysis_ws)) > 0.1)
    elif case == "reanalysis shorter than wind farm data":
        reanalysis_ws = reanalysis_ws.loc["2021-01-03":"2021-01-05"]
    elif case == "wind farm data off the reanalysis timestamps":
        ws_df = ws_df.set_axis(ws_df.index + pd.Timedelta("5min")).iloc[::7]
        ws_df = pd.concat([ws_df, ws_df.set_axis(ws_df.index - pd.Timedelta("5min")).iloc[::2]]).sort_index()
    elif case == "no overlap":
        ws_df = ws_df.set_axis(ws_df.index + pd.Timedelta("400D"))
    elif case == "reanalysis out of order":
        reanalysis_ws = reanalysis_ws.iloc[::-1]
    elif case == "wind farm data repeated timestamp":
        ws_df = pd.concat([ws_df, ws_df.iloc[[10]]]).sort_index()
    elif case == "two columns":
        ws_df = ws_df.assign(other=ws_df["WindSpeedMean"] * 2)
    return ws_df, reanalysis_ws


@pytest.mark.parametrize(
    "case",
    [
        "aligned",
        "gaps in reanalysis",
        "reanalysis shorter than wind farm data",
        "wind farm data off the reanalysis timestamps",
        "no overlap",
        "reanalysis out of order",
        "wind farm data repeated timestamp",
        "two columns",
    ],
)
def test_corr_with_shifted_reanalysis_matches_reference(case: str) -> None:
    ws_df, reanalysis_ws = _change_ws_df_and_reanalysis_ws(*_make_ws_df_and_reanalysis_ws(seed=0), case=case)
    shifts = list(range(-144, 144, 1))

    if case == "two columns":
        with pytest.raises(TypeError):
            _reference_corr_with_shifted_reanalysis(ws_df, reanalysis_ws, shifts=shifts)
        with pytest.raises(TypeError):
            _corr_with_shifted_reanalysis(ws_df, reanalysis_ws, shifts=shifts)
        return

    actual = _corr_with_shifted_reanalysis(ws_df, reanalysis_ws, shifts=shifts)

    expected = _reference_corr_with_shifted_reanalysis(ws_df, reanalysis_ws, shifts=shifts)
    np.testing.assert_array_equal(actual, expected)
    if case == "aligned":
        assert shifts[int(np.argmax(actual))] == 5


def test_find_best_shift_and_corr_finds_the_shift() -> None:
    ws_df, reanalysis_ws = _make_ws_df_and_reanalysis_ws(seed=1)

    best_corr, best_s = _find_best_shift_and_corr(
        wf_ws_df=ws_df,
        reanalysis_df=reanalysis_ws.to_frame(),
        wf_name="wf",
        datastream_id="reanalysis",
        timebase_s=TIMEBASE_S,
        plot_cfg=None,
    )

    expected_corrs = _reference_corr_with_shifted_reanalysis(ws_df, reanalysis_ws, shifts=list(range(-144, 144)))
    assert best_s == 5
    assert best_corr == max(expected_corrs)


def test_add_reanalysis_data_matches_reference(
    test_homer_config: WindUpConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = test_homer_config
    cfg.lt_first_dt_utc_start = pd.Timestamp("2023-07-01 00:00:00", tz="UTC")
    cfg.analysis_last_dt_utc_start = pd.Timestamp("2023-07-31 23:50:00", tz="UTC")
    wf_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/Homer Wind Farm_July2023_scada_improved.parquet")
    reanalysis_datasets = [
        ReanalysisDataset(id=fp.stem, data=pd.read_parquet(fp))
        for fp in (TEST_DATA_FLD / "reanalysis" / "Homer Wind Farm").glob("*.parquet")
    ]

    assert _wind_speed_by_timestamp_and_turbine(wf_df) is not None
    actual = add_reanalysis_data(wf_df, cfg=cfg, plot_cfg=None, reanalysis_datasets=reanalysis_datasets)

    monkeypatch.setattr(reanalysis_data, "_corr_with_shifted_reanalysis", _reference_corr_with_shifted_reanalysis)
    monkeypatch.setattr(
        reanalysis_data,
        "_calc_wf_mean_wind_speed_and_data_coverage",
        _reference_calc_wf_mean_wind_speed_and_data_coverage,
    )
    expected = add_reanalysis_data(wf_df, cfg=cfg, plot_cfg=None, reanalysis_datasets=reanalysis_datasets)
    assert_frame_equal(actual, expected, check_exact=True)
