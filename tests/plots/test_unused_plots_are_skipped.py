import logging
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from tests.conftest import TEST_DATA_FLD
from wind_up.constants import DataColumns
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.plots.input_data import _validate_input_data_within_exclusions
from wind_up.plots.pp_analysis_plots import plot_pp_data_coverage, plot_pre_post_pp_analysis
from wind_up.plots.windspeed_drift_plots import plot_rolling_windspeed_diff_one_wtg
from wind_up.plots.yaw_direction_plots import plot_yaw_direction_pre_post
from wind_up.pp_analysis import _pre_post_pp_analysis
from wind_up.result_manager import result_manager


def no_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        msg = "a figure was drawn"
        raise AssertionError(msg)

    monkeypatch.setattr(plt, "figure", fail)
    monkeypatch.setattr(plt, "subplots", fail)


def _plots_off(tmp_path: Path) -> PlotConfig:
    return PlotConfig(show_plots=False, save_plots=False, plots_dir=tmp_path)


def _pre_post_inputs(cfg: WindUpConfig) -> dict:
    return {
        "cfg": cfg,
        "test_wtg": next(x for x in cfg.asset.wtgs if x.name == "LSA_T13"),
        "ref_name": "LSA_T12",
        "lt_df": pd.read_parquet(TEST_DATA_FLD / "LSA_T13_lt_wtg_df_filt.parquet"),
        "pre_df": pd.read_parquet(TEST_DATA_FLD / "LSA_T13_LSA_T12_pre_df.parquet"),
        "post_df": pd.read_parquet(TEST_DATA_FLD / "LSA_T13_LSA_T12_post_df.parquet"),
        "ws_col": "ref_ws_detrended",
        "pw_col": "test_pw_clipped",
        "wd_col": "ref_YawAngleMean",
    }


def _plot_pre_post_pp(cfg: WindUpConfig, plot_cfg: PlotConfig) -> list[str]:
    inputs = _pre_post_inputs(cfg)
    _, pp_df = _pre_post_pp_analysis(**inputs, plot_cfg=None)
    pre_df = inputs["pre_df"].drop(columns="ref_" + DataColumns.ambient_temp)
    result_manager.stored_warnings = []

    plot_pre_post_pp_analysis(
        test_name="LSA_T13",
        ref_name="LSA_T12",
        pp_df=pp_df,
        pre_df=pre_df,
        post_df=inputs["post_df"],
        ws_col=inputs["ws_col"],
        pw_col=inputs["pw_col"],
        wd_col=inputs["wd_col"],
        timebase_s=600,
        plot_cfg=plot_cfg,
        confidence_level=0.9,
    )
    plot_pp_data_coverage(
        test_name="LSA_T13",
        ref_name="LSA_T12",
        pp_df=pp_df,
        test_df_pp_period=pd.read_parquet(TEST_DATA_FLD / "LSA_T13_test_df.parquet"),
        ws_bin_width=1.0,
        timebase_s=600,
        plot_cfg=plot_cfg,
    )
    warnings = result_manager.stored_warnings
    result_manager.stored_warnings = []
    return warnings


def test_pre_post_pp_plots_are_skipped_but_warn_as_when_drawn(
    test_lsa_t13_config: WindUpConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plots_dir = tmp_path / "drawn"
    (plots_dir / "LSA_T13" / "LSA_T12").mkdir(parents=True)
    warnings_when_drawn = _plot_pre_post_pp(
        test_lsa_t13_config, PlotConfig(show_plots=False, save_plots=True, plots_dir=plots_dir)
    )
    assert len(list((plots_dir / "LSA_T13" / "LSA_T12").iterdir())) > 5

    no_figures(monkeypatch)
    warnings_when_skipped = _plot_pre_post_pp(test_lsa_t13_config, _plots_off(tmp_path / "skipped"))

    assert warnings_when_skipped == warnings_when_drawn
    assert warnings_when_drawn == [
        "plot_pre_post_condition_histogram ref_name=LSA_T12 pre_df missing required column ref_AmbientTemp"
    ]


def test_yaw_direction_and_windspeed_drift_plots_are_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    no_figures(monkeypatch)
    index = pd.date_range("2020-01-01", periods=100, freq="10min", tz="UTC")
    df = pd.DataFrame({"test_YawAngleMean": 10.0, "ref_ws": 8.0, "ref_wd": 12.0}, index=index)

    plot_yaw_direction_pre_post(
        df,
        df,
        test_name="T1",
        ref_name="T2",
        ref_ws_col="ref_ws",
        ref_wd_col="ref_wd",
        plot_cfg=_plots_off(tmp_path),
    )
    plot_rolling_windspeed_diff_one_wtg(
        ser=pd.Series(np.zeros(100), index=index),
        wtg_name="T1",
        ws_col="ws",
        plot_cfg=_plots_off(tmp_path),
        sub_dir=None,
    )

    assert list(tmp_path.iterdir()) == []


def test_validate_input_data_within_exclusions(caplog: pytest.LogCaptureFixture) -> None:
    index = pd.MultiIndex.from_product(
        [["T1", "T2"], pd.date_range("2020-01-01", periods=10, freq="1D", tz="UTC")],
        names=[DataColumns.turbine_name, "TimeStamp_StartFormat"],
    )
    wf_df = pd.DataFrame({DataColumns.active_power_mean: 1.0, DataColumns.yaw_angle_mean: np.nan}, index=index)
    exclusion = ("T1", pd.Timestamp("2020-01-02", tz="UTC"), pd.Timestamp("2020-01-04", tz="UTC"))
    inputs = SimpleNamespace(
        cfg=SimpleNamespace(exclusion_periods_utc=[exclusion], yaw_data_exclusions_utc=[exclusion]), wf_df=wf_df
    )

    with caplog.at_level(logging.WARNING):
        _validate_input_data_within_exclusions(inputs)  # type: ignore[arg-type]

    assert caplog.text.count("Data is not all NaN within exclusion period") == 1
