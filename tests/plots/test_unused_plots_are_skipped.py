import logging
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from tests.conftest import TEST_DATA_FLD
from wind_up_v0.constants import DataColumns
from wind_up_v0.models import PlotConfig, WindUpConfig
from wind_up_v0.plots.input_data import _validate_input_data_within_exclusions
from wind_up_v0.plots.pp_analysis_plots import plot_pp_data_coverage, plot_pre_post_pp_analysis
from wind_up_v0.plots.windspeed_drift_plots import plot_rolling_windspeed_diff_one_wtg
from wind_up_v0.plots.yaw_direction_plots import plot_yaw_direction_pre_post
from wind_up_v0.pp_analysis import _pre_post_pp_analysis
from wind_up_v0.result_manager import result_manager


@pytest.fixture
def no_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        msg = "a figure was drawn"
        raise AssertionError(msg)

    monkeypatch.setattr(plt, "figure", fail)
    monkeypatch.setattr(plt, "subplots", fail)


@pytest.mark.usefixtures("no_figures")
def test_unused_plots_are_skipped_but_still_warn(test_lsa_t13_config: WindUpConfig, tmp_path: Path) -> None:
    plot_cfg = PlotConfig(show_plots=False, save_plots=False, plots_dir=tmp_path)
    assert plot_cfg.plots_unused
    pre_df = pd.read_parquet(TEST_DATA_FLD / "LSA_T13_LSA_T12_pre_df.parquet")
    post_df = pd.read_parquet(TEST_DATA_FLD / "LSA_T13_LSA_T12_post_df.parquet")
    _, pp_df = _pre_post_pp_analysis(
        cfg=test_lsa_t13_config,
        test_wtg=next(x for x in test_lsa_t13_config.asset.wtgs if x.name == "LSA_T13"),
        ref_name="LSA_T12",
        lt_df=pd.read_parquet(TEST_DATA_FLD / "LSA_T13_lt_wtg_df_filt.parquet"),
        pre_df=pre_df,
        post_df=post_df,
        ws_col="ref_ws_detrended",
        pw_col="test_pw_clipped",
        wd_col="ref_YawAngleMean",
        plot_cfg=None,
    )
    result_manager.stored_warnings = []

    plot_pre_post_pp_analysis(
        test_name="LSA_T13",
        ref_name="LSA_T12",
        pp_df=pp_df,
        pre_df=pre_df.drop(columns="ref_" + DataColumns.ambient_temp),
        post_df=post_df,
        ws_col="ref_ws_detrended",
        pw_col="test_pw_clipped",
        wd_col="ref_YawAngleMean",
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
    plot_yaw_direction_pre_post(
        pre_df,
        post_df,
        test_name="LSA_T13",
        ref_name="LSA_T12",
        ref_ws_col="ref_ws_detrended",
        ref_wd_col="ref_YawAngleMean",
        plot_cfg=plot_cfg,
    )
    plot_rolling_windspeed_diff_one_wtg(
        ser=pd.Series(np.zeros(3)), wtg_name="LSA_T13", ws_col="ws", plot_cfg=plot_cfg, sub_dir=None
    )

    assert result_manager.stored_warnings == [
        "plot_pre_post_condition_histogram ref_name=LSA_T12 pre_df missing required column ref_AmbientTemp"
    ]
    result_manager.stored_warnings = []
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
